"""Disk-backed, target-conditioned imitation episodes.

The teach pendant does not expose its requested joint targets to ROS. For
manual demonstrations the closest honest action label is therefore the next
measured arm configuration interpolated onto a fixed-rate grid. The manifest says that
explicitly; consumers must not reinterpret it as a controller command log.
"""

import bisect
import json
import shutil
import tempfile
import threading
from pathlib import Path


DEFAULT_GRIPPER_JOINT_NAMES = [
    "F1M1", "F1M2", "F1M3", "F1M4",
    "F2M1", "F2M2", "F2M3", "F2M4",
    "F3M1", "F3M2", "F3M3", "F3M4",
]


def _stamp_ns(msg):
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def pair_frames(frames, states, tolerance_ns=50_000_000):
    """Attach the nearest named arm state without fabricating stale values."""
    states = sorted(
        (state for state in states if state["stamp_ns"] > 0),
        key=lambda state: state["stamp_ns"],
    )
    stamps = [state["stamp_ns"] for state in states]
    result = []
    for frame in frames:
        timestamp = frame["stamp_ns"]
        index = bisect.bisect_left(stamps, timestamp)
        candidates = [i for i in (index - 1, index) if 0 <= i < len(states)]
        row = dict(frame)
        if (timestamp > 0 and 0 < index < len(states)
                and stamps[index]-stamps[index-1] <= 100_000_000):
            left, right = states[index-1], states[index]
            weight = (timestamp-stamps[index-1])/(stamps[index]-stamps[index-1])
            row.update(joints=[a+(b-a)*weight for a,b in zip(left['joints'],right['joints'])],
                       state_stamp_ns=timestamp, skew_ns=0,
                       state_bracket_ns=[stamps[index-1],stamps[index]])
            result.append(row)
            continue
        if timestamp > 0 and candidates:
            nearest = min(candidates, key=lambda i: abs(stamps[i] - timestamp))
            skew = stamps[nearest] - timestamp
            if abs(skew) <= tolerance_ns:
                row.update(
                    joints=states[nearest]["joints"],
                    state_stamp_ns=stamps[nearest],
                    skew_ns=skew,
                )
        result.append(row)
    return result


def _pair_gripper(rows, states, tolerance_ns=150_000_000):
    states = sorted(
        (state for state in states if state["stamp_ns"] > 0),
        key=lambda state: state["stamp_ns"],
    )
    stamps = [state["stamp_ns"] for state in states]
    result = []
    for source in rows:
        row = dict(source)
        timestamp = row["stamp_ns"]
        index = bisect.bisect_left(stamps, timestamp)
        candidates = [i for i in (index - 1, index) if 0 <= i < len(states)]
        if timestamp > 0 and candidates:
            nearest = min(candidates, key=lambda i: abs(stamps[i] - timestamp))
            skew = stamps[nearest] - timestamp
            if abs(skew) <= tolerance_ns:
                state = states[nearest]
                row.update(
                    gripper_joints=state["joints"],
                    gripper_effort=state.get("effort", []),
                    gripper_state_stamp_ns=stamps[nearest],
                    gripper_skew_ns=skew,
                )
        result.append(row)
    return result


def _stream_metrics(frames):
    stamps = sorted(set(row["stamp_ns"] for row in frames if row["stamp_ns"] > 0))
    duration = (stamps[-1] - stamps[0]) / 1e9 if len(stamps) > 1 else 0.0
    rate = (len(stamps) - 1) / duration if duration > 0 else 0.0
    gap = max((b - a for a, b in zip(stamps, stamps[1:])), default=0) / 1e9
    return rate, gap


def fixed_rate_rows(rows, fps, close_time_ns=None):
    """Uniform segments; never discard good spans or bridge a missing span."""
    import numpy as np
    rows = sorted(rows, key=lambda row: row['stamp_ns'])
    if close_time_ns is not None:
        rows = [row for row in rows if row['stamp_ns'] <= close_time_ns]
    if len(rows) < 2:
        return []
    stamps = [row['stamp_ns'] for row in rows]
    step = round(1e9 / fps)
    end_ns = close_time_ns if close_time_ns is not None else stamps[-1]
    segments, current = [], []
    for t in range(stamps[0], end_ns+1, step):
        i = bisect.bisect_left(stamps,t)
        lo, hi = max(0,i-1), min(i,len(rows)-1)
        if stamps[hi] == t:
            lo = hi
        nearest = min((lo,hi),key=lambda j:abs(stamps[j]-t))
        if stamps[hi]-stamps[lo] > 250_000_000 or abs(stamps[nearest]-t)>step//2:
            if len(current)>=2:
                segments.append(current)
            current=[]
            continue
        weight=(t-stamps[lo])/(stamps[hi]-stamps[lo]) if hi!=lo else 0
        row=dict(rows[nearest]);row['sample_stamp_ns']=t
        for key in ('joints','gripper_joints'):
            row[key]=((1-weight)*np.asarray(rows[lo][key])+weight*np.asarray(rows[hi][key])).tolist()
        current.append(row)
    if len(current)>=2:
        segments.append(current)
    result=[]
    for segment_id,segment in enumerate(segments):
        for i,row in enumerate(segment):
            row['segment_id']=segment_id
            row['segment_end']=i==len(segment)-1
            result.append(row)
    return result


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, sort_keys=False))
    temporary.replace(path)


def update_training_readiness(directory, operator_label):
    """Persist one operator grade and readiness to both policy manifests."""
    sequence_path = Path(directory) / "sequence" / "sequence.json"
    if not sequence_path.is_file():
        return False
    try:
        sequence = json.loads(sequence_path.read_text())
        quality = sequence["quality"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    label = str(operator_label).lower()
    ready = bool(quality.get("data_ready") and label == "good")
    sequence["label_source"] = "operator"
    sequence["operator_label"] = label
    quality["training_ready"] = ready
    quality["operator_label"] = label
    if ready:
        quality["readiness_reason"] = "quality checks passed; operator GOOD (authoritative)"
    elif not quality.get("data_ready"):
        quality["readiness_reason"] = "recording quality checks failed"
    else:
        quality["readiness_reason"] = "operator label is not GOOD"
    _atomic_json(sequence_path, sequence)
    sample_path = Path(directory) / "sample.json"
    if sample_path.is_file():
        sample = json.loads(sample_path.read_text())
        sample["label_source"] = "operator"
        sample["operator_label"] = label
        sample["sequence_quality"] = quality
        _atomic_json(sample_path, sample)
    return ready


class DemoSequence:
    """Record raw streams and finalize them into one policy episode."""

    def __init__(self, joint_names, gripper_joint_names=None,
                 task="Grasp the highlighted date", target_fps=10.0,
                 action_source="pendant_demonstration"):
        self.names = list(joint_names)
        self.gripper_names = list(
            gripper_joint_names or DEFAULT_GRIPPER_JOINT_NAMES)
        self.action_source = str(action_source)
        self.task = str(task)
        self.target_fps = float(target_fps)
        self.path = Path(tempfile.mkdtemp(prefix="datepalm-demo-"))
        self.lock = threading.RLock()
        self.frames = []
        self.states = []
        self.gripper_states = []
        self.commands = []
        self.target = None
        self.closed = False
        self.finalized = False
        self.staged = False
        self.errors = 0

    def state(self, msg, received_ns):
        values = dict(zip(msg.name, msg.position))
        if not all(name in values for name in self.names):
            return
        with self.lock:
            if not self.closed:
                self.states.append({
                    "stamp_ns": _stamp_ns(msg),
                    "received_ns": int(received_ns),
                    "joints": [float(values[name]) for name in self.names],
                })

    def gripper_state(self, msg, received_ns):
        names = list(msg.name)
        if names:
            indices = {name: index for index, name in enumerate(names)}
            if not all(name in indices for name in self.gripper_names):
                return
            order = [indices[name] for name in self.gripper_names]
        elif len(msg.position) >= len(self.gripper_names):
            order = list(range(len(self.gripper_names)))
        else:
            return
        effort = list(getattr(msg, "effort", []) or [])
        with self.lock:
            if not self.closed:
                self.gripper_states.append({
                    "stamp_ns": _stamp_ns(msg),
                    "received_ns": int(received_ns),
                    "joints": [float(msg.position[index]) for index in order],
                    "effort": [float(effort[index]) for index in order]
                    if effort and len(effort) > max(order) else [],
                })

    def frame(self, msg, received_ns):
        with self.lock:
            if self.closed:
                return
            name = f"{len(self.frames):06d}.jpg"
            try:
                (self.path / name).write_bytes(bytes(msg.data))
                self.frames.append({
                    "image": name,
                    "stamp_ns": _stamp_ns(msg),
                    "received_ns": int(received_ns),
                })
            except OSError:
                self.errors += 1

    def command(self, name, value, received_ns, source_stamp_ns=None):
        with self.lock:
            if not self.closed:
                self.commands.append({
                    "name": str(name),
                    "source_stamp_ns": source_stamp_ns,
                    "value": float(value),
                    "received_ns": int(received_ns),
                })

    def set_target(self, image, pixel_xy, image_shape, target_label):
        height, width = int(image_shape[0]), int(image_shape[1])
        px, py = (float(pixel_xy[0]), float(pixel_xy[1]))
        selected_id = target_label.get("selected_fruit_id")
        selected = next(
            (candidate for candidate in target_label.get("candidates", [])
             if candidate.get("fruit_id") == selected_id),
            None,
        )
        base_xyz = selected.get("base_xyz") if selected else None
        with self.lock:
            self.target = {
                "image": str(image),
                "pixel_xy": [px, py],
                "pixel_xy_normalized": [px / width, py / height],
                "image_size": [width, height],
                "selected_fruit_id": selected_id,
                "identity_verified": bool(target_label.get("identity_verified")),
                "operator_point_confirmed": bool(target_label.get("operator_point_confirmed")),
                "base_xyz": [float(value) for value in base_xyz]
                if base_xyz and len(base_xyz) == 3 else None,
            }

    def _motion_rows(self, rows):
        starts = [c['source_stamp_ns'] for c in self.commands
                  if c['name'] == 'recording.motion_start'
                  and c.get('source_stamp_ns') is not None]
        closes = [c['source_stamp_ns'] for c in self.commands
                  if c['name'] == 'gripper.close_request' and c['value'] >= .5
                  and c.get('source_stamp_ns') is not None]
        return [r for r in rows
                if (not starts or r['stamp_ns'] >= min(starts))
                and (not closes or r['stamp_ns'] <= min(closes))]

    def _rows_and_quality(self, close_requested):
        raw_rows = _pair_gripper(pair_frames(self.frames, self.states), self.gripper_states)
        rows = self._motion_rows(raw_rows)
        rate, gap = _stream_metrics(rows)
        paired = sum("joints" in row for row in rows)
        gripper_paired = sum("gripper_joints" in row for row in rows)
        target_verified = bool(
            self.target
            and (self.target.get("operator_point_confirmed")
                 or (self.target.get("identity_verified")
                     and self.target.get("base_xyz") is not None))
        )
        close_recorded = any(
            item["name"] == "gripper.close_request" and item["value"] >= 0.5
            for item in self.commands
        )
        arm_coverage = paired / len(rows) if rows else 0.0
        gripper_coverage = gripper_paired / len(rows) if rows else 0.0
        timing_ok = bool(
            len(rows) >= 10
            and arm_coverage >= 0.95
            and rate >= max(5.0, self.target_fps * 0.8)
            and gap <= 0.25
            and not self.errors
        )
        data_ready = bool(
            timing_ok
            and gripper_coverage >= 0.90
            and target_verified
            and (close_recorded if close_requested else True)
        )
        quality = {
            "raw_frames": len(raw_rows),
            "quality_scope": "motion_start_to_close_when_recorded",
            "frames": len(rows),
            "paired_frames": paired,
            "paired_fraction": arm_coverage,
            "gripper_paired_frames": gripper_paired,
            "gripper_paired_fraction": gripper_coverage,
            "effective_hz": rate,
            "target_hz": self.target_fps,
            "max_frame_gap_s": gap,
            "write_errors": self.errors,
            "timing_ok": timing_ok,
            "target_identity_verified": target_verified,
            "close_command_recorded": close_recorded,
            "data_ready": data_ready,
            "training_ready": False,
            "max_arm_pair_skew_ms": 50,
            "max_gripper_pair_skew_ms": 150,
            "clock_requirement": "camera, arm and gripper headers must share a synchronized clock",
            "note": "training_ready requires operator GOOD; automatic grasp verdicts are not recorded or used",
        }
        return raw_rows, quality

    def snapshot_quality(self):
        with self.lock:
            _, quality = self._rows_and_quality(close_requested=False)
            return quality

    def stage(self, destination):
        """Move the live spool into the episode directory and keep recording."""
        with self.lock:
            if self.staged:
                return self.snapshot_quality()
            sequence_path = Path(destination) / "sequence"
            if sequence_path.exists():
                raise FileExistsError(f"sequence already exists: {sequence_path}")
            shutil.move(str(self.path), str(sequence_path))
            self.path = sequence_path
            self.staged = True
            return self.snapshot_quality()

    def finalize(self, destination=None, complete=False,
                 close_requested=True):
        if destination is not None:
            self.stage(destination)
        with self.lock:
            if self.finalized:
                return json.loads((self.path / "sequence.json").read_text())["quality"]
            self.closed = True
            rows, quality = self._rows_and_quality(close_requested=close_requested)
            usable = [row for row in self._motion_rows(rows)
                      if "joints" in row and "gripper_joints" in row]
            close_times = [c['source_stamp_ns'] for c in self.commands
                           if c['name']=='gripper.close_request' and c['value']>=.5
                           and c.get('source_stamp_ns') is not None]
            export_hz = min(self.target_fps, 5.0)
            usable = fixed_rate_rows(usable, export_hz, min(close_times) if close_times else None)
            quality['uniform_training_frames'] = len(usable)
            quality['export_hz'] = export_hz
            quality['segments'] = len(set(r['segment_id'] for r in usable))
            quality['data_ready'] = bool(quality['data_ready'] and len(usable)>=10)
            target_features = (
                list(self.target.get("pixel_xy_normalized", []))
                if self.target else []
            )
            training_frames = []
            for index, row in enumerate(usable):
                next_joints = (usable[index + 1]["joints"]
                               if not row["segment_end"] else row["joints"])
                terminal = row["segment_end"]
                close_action = bool(index == len(usable)-1 and close_times
                    and 0 <= min(close_times)-row["sample_stamp_ns"] <= 200_000_000)
                training_frames.append({
                    "image": row["image"],
                    "source_stamp_ns": row["stamp_ns"],
                    "sample_stamp_ns": row["sample_stamp_ns"],
                    "action_dt_s": 1.0/export_hz,
                    "segment_id": row["segment_id"],
                    "observation_state": (
                        list(row["joints"])
                        + list(row["gripper_joints"])
                        + target_features
                    ),
                    "action": list(next_joints) + [
                        1.0 if close_action and close_requested else 0.0
                    ],
                    "terminal": terminal,
                })
            quality['close_action_exported'] = any(row['action'][-1] > .5 for row in training_frames)
            if close_requested and not quality['close_action_exported']:
                quality['data_ready'] = False
            observation_names = (
                [f"arm.{name}.position" for name in self.names]
                + [f"gripper.{name}.position" for name in self.gripper_names]
                + ["target.u_normalized", "target.v_normalized"]
            )
            action_names = (
                [f"arm.{name}.next_observed_position" for name in self.names]
                + ["gripper.close_request"]
            )
            manifest = {
                "schema_version": 5,
                "complete": bool(complete),
                "task": self.task,
                "policy_scope": "APPROACH_TO_GRASP_POSE_AND_CLOSE_REQUEST",
                "label_source": "operator",
                "operator_label": None,
                "joint_names": self.names,
                "gripper_joint_names": self.gripper_names,
                "observation_state_names": observation_names,
                "action_names": action_names,
                "trajectory_source": self.action_source,
                "action_source": (
                    "arm targets are the next interpolated measured state on a fixed-rate grid; "
                    "the terminal scalar requests the existing force-aware close controller"
                ),
                "synchronization": "nearest source timestamp; unmatched samples retained in raw frames only",
                "target": self.target,
                "quality": quality,
                "training_frames": training_frames,
                "frames": rows,
                "joint_states": self.states,
                "gripper_states": self.gripper_states,
                "commands": self.commands,
            }
            _atomic_json(self.path / "sequence.json", manifest)
            self.finalized = True
            return quality

    def save(self, destination):
        """Compatibility helper for timing tests and non-grasp recordings."""
        return self.finalize(destination, close_requested=False)

    def close(self):
        with self.lock:
            self.closed = True
            if not self.staged and self.path.exists():
                shutil.rmtree(self.path)
