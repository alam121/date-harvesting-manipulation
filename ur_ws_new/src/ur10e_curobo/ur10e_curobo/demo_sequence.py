"""Disk-backed source-timestamped demo streams; never pair stale states silently."""
import bisect
import json
import shutil
import tempfile
import threading
from pathlib import Path


def pair_frames(frames, states, tolerance_ns=50_000_000):
    states = sorted((s for s in states if s['stamp_ns'] > 0), key=lambda s: s['stamp_ns'])
    stamps = [s['stamp_ns'] for s in states]
    result = []
    for frame in frames:
        t = frame['stamp_ns']
        i = bisect.bisect_left(stamps, t)
        candidates = [j for j in (i-1, i) if 0 <= j < len(states)]
        row = dict(frame)
        if t > 0 and candidates:
            j = min(candidates, key=lambda j: abs(stamps[j]-t))
            skew = stamps[j]-t
            if abs(skew) <= tolerance_ns:
                row.update(joints=states[j]['joints'], state_stamp_ns=stamps[j], skew_ns=skew)
        result.append(row)
    return result


class DemoSequence:
    def __init__(self, joint_names):
        self.names = list(joint_names)
        self.path = Path(tempfile.mkdtemp(prefix='datepalm-demo-'))
        self.lock = threading.Lock()
        self.frames, self.states = [], []
        self.closed = False
        self.errors = 0

    def state(self, msg, received_ns):
        values = dict(zip(msg.name, msg.position))
        if not all(n in values for n in self.names):
            return
        with self.lock:
            if not self.closed:
                self.states.append(dict(stamp_ns=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec,
                                        received_ns=received_ns, joints=[values[n] for n in self.names]))

    def frame(self, msg, received_ns):
        with self.lock:
            if self.closed:
                return
            name = f'{len(self.frames):06d}.jpg'
            try:
                (self.path/name).write_bytes(bytes(msg.data))
                self.frames.append(dict(image=name, stamp_ns=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec,
                                        received_ns=received_ns))
            except OSError:
                self.errors += 1

    def save(self, destination):
        with self.lock:
            self.closed = True
            rows = pair_frames(self.frames, self.states)
            paired = sum('joints' in r for r in rows)
            stamps = sorted(set(r['stamp_ns'] for r in rows))
            duration = (stamps[-1]-stamps[0])/1e9 if len(stamps)>1 else 0
            rate = (len(stamps)-1)/duration if duration>0 else 0
            gap = max((b-a for a,b in zip(stamps,stamps[1:])), default=0)/1e9
            quality = dict(frames=len(rows), paired_frames=paired, effective_hz=rate,
                           max_frame_gap_s=gap, write_errors=self.errors,
                           timing_ok=bool(len(rows)>=10 and paired==len(rows) and rate>=5 and gap<=0.5 and not self.errors),
                           max_pair_skew_ms=50, clock_requirement='image and joint headers must share a synchronized clock',
                           training_ready=False, note='Timing check only. Review visibility, labels and clock synchronization before training.')
            (self.path/'sequence.json').write_text(json.dumps(dict(schema_version=1, joint_names=self.names,
                synchronization='nearest source timestamp; unmatched frames retained without state',
                quality=quality, frames=rows, joint_states=self.states),indent=1))
            shutil.move(str(self.path), str(Path(destination)/'sequence'))
            return quality

    def close(self):
        with self.lock:
            self.closed = True
            if self.path.exists():
                shutil.rmtree(self.path)
