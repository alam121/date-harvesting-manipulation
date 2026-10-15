# ur10e_curobo: speed and localization review

**Implementation update — September 16, 2026:** The review below describes the original findings. Code fixes have now been applied; see [fix validation and remaining work](ur_ws_new/src/ur10e_curobo/FIX_VALIDATION.md). The original “no application code changed” statement below applies to the September 14 review only.

Reviewed September 14, 2026. Package: `ur_ws_new/src/ur10e_curobo`.

This is a static review with isolated logic checks, not a robot acceptance test. Coverage includes the package inventory, all 71 Python files at syntax/structure level, and detailed tracing of perception, target selection, reacquisition, planning, final approach, gripper handling, calibration, and shared state. Supporting launch, configuration, documentation, URDF, and saved benchmark artifacts were also inspected. This does not mean every line of every support file or binary asset received equal manual scrutiny. Existing source edits were preserved; no application code was changed.

## Fix first: correctness affecting localization and execution

### 1. Gripper messages erase the cached arm state — reproduced

`ur10e_curobo/managers/state_manager.py:257`, `_joint_state_cb`, replaces arm positions with whatever arm joints occur in each message. A gripper-only message produces an empty arm position list and zero arm velocities. `gripper.py` publishes fake gripper joint states on the shared `/joint_states` topic; the callback also documents the shared real-driver topic.

Isolated result: arm positions `[1, 2]` become `[]` after a finger-only message; velocities become `[0, 0]`. This can cause intermittent missing FK/planning state and incorrectly indicate that the arm is stationary.

**Change:** ignore messages without arm joints; validate joint-name and array completeness, and update a complete arm snapshot atomically. If merging partial arm messages, retain per-joint timestamps and enforce freshness. Missing velocity must remain unknown rather than become a measured zero.

### 2. Motion completion can succeed while moving or after a stop — reproduced

`ur10e_curobo/utils.py:167`, `wait_until_xyz`, returns true on the first in-tolerance sample even when velocity exceeds the threshold: the first-sample branch initializes its timer and falls through to success. Stop requests and caught exceptions can also return true because the return value depends only on `stalled`. Timeout can accept a previous closest approach rather than the current pose.

Isolated results: target already reached with all joint velocities at 0.5 rad/s returns true; a stop request also returns true. Some callers add a position check or inspect the stop flag, but that does not make this helper's completion contract reliable.

**Change:** return explicit reached/stopped/timed-out/failed status. Require fresh position, orientation, and velocity feedback to remain within limits for a short continuous dwell before closing the gripper. Validate actual orientation error in degrees, not only an opaque quaternion-dot threshold.

### 3. RGB detections, depth, TF, and quality need one observation identity

`vision/node.py:1063–1175` consumes current camera packets while YOLO completes asynchronously. Tracking the RGB inference timestamp alone does not establish that the depth used for the mask came from that image. `vision/node.py:1408` publishes target positions and quality separately; `node.py:255` stores them independently. `goals.py:3044` uses a position index to fetch quality from the independently updated list.

**Change:** carry a frame ID and acquisition timestamp with RGB, aligned depth, detections, and the TF used for projection. Publish each target's ID, position, quality, and geometric estimates in one stamped observation. Reject mismatched/over-age data. A bounded frame cache can preserve alignment without creating an unbounded backlog.

Use the camera's image timestamp with an explicit conversion to ROS time. The current grab-return wall timestamp improves queue delay accounting but still omits sensor-to-grab latency. At 0.3 m/s, 50 ms of timestamp mismatch corresponds to 15 mm of translation before rotational effects.

### 4. Reacquisition can accept stale data or a neighboring fruit

`goals.py:3095`, `_reacquire_goal_pose_impl`, eventually proceeds after its freshness wait expires. The cached all-fruit list remains a candidate source. The default confirmation count is one. The permissive one-axis matching branch can accept a candidate close in XY but far in Z, or close in Z but far in XY; it can then combine candidate and seed coordinates into a synthetic position. With multiple confirmation frames, equal coordinates are skipped even if they came from genuinely new frames.

**Change:** require a new acquisition sequence after settling, bound both lateral and depth association error, and associate by stable target ID. Deduplicate by sequence rather than coordinate equality. Distinguish a deliberate seed fallback from an actual measured refinement. Reject nonfinite quality values and make missing-quality policy explicit.

### 5. Tip anchoring can discard the final reacquisition correction

`goals.py:5002` and `5071` construct `_tip_anchored_final_grasp` before the final reacquisition. Reacquisition updates `x, y, z` at line 5969, but final target construction at line 6117 selects the earlier tip anchor whenever present.

**Change:** explicitly reconcile the tip estimate and the new observation. Re-estimate the tip from the matched fresh target, or propagate a validated same-target displacement while preserving insertion geometry. If retaining the original anchor is intentional, report that reacquisition did not update the commanded closure position. Do not translate an anchor using an unverified neighboring-fruit match.

### 6. Envelop closure calibration is configured but unused by motion

`config.py:729` defines `envelop_closure_center_offset_tcp_m`; `managers/config_manager.py` declares, loads, and updates it. Repository references otherwise only report the value. Final TCP correction at `goals.py:6134` always selects `closure_center_offset_tcp_m`.

**Change:** select the closure-center offset from the active grasp mode and use the same selection during preflight, approach, final execution, and retargeting. The current profile values happen to match, which hides this until envelop calibration is tuned independently.

### 7. Tracking confirmation and smoothing mix old or unrelated targets — reproduced

`vision/tracking.py:86`: persistence does not require prior confirmation; confirmed detections remain in output on the same iteration they exceed the miss limit. `frames_seen` is cumulative rather than consecutive. With confirmation=2 and persistence=2, one detection followed by a miss produces output counts `[0, 1]`. Two detections followed by three misses produce `[0, 1, 1, 1, 1]`.

Centimeter-rounded position hashes are also discontinuous: a 2 mm change across a rounding boundary changes identity. Camera-frame identities cannot remain stable during wrist movement, and attempt bookkeeping must use the same coordinate convention as registration.

`vision/node.py:3540` smooths the selected point through a shared history, clearing on a large jump rather than every identity change. Nearby fruit can therefore contribute to the same smoothed point.

**Change:** explicit confirmed/lost/deleted track states, persistent IDs associated in the base frame, and separate filtering/cache history per target. Remove stale tracks before output. Invalidate tip caches on age, target change, rotation, cumulative motion, and depth/mask changes; cached geometry must retain its original acquisition age.

### 8. Velocity clipping does not enforce trajectory speed limits — reproduced

`utils.py:106`, `build_trajectory`, clips derivative fields while leaving positions and timestamps unchanged. Isolated input `[0, 1, 2]` rad with 10 ms spacing requires 100 rad/s between samples, but `max_vel=0.5` only changes the published middle velocity to 0.5 rad/s; the trajectory still ends at 30 ms. This synthetic example demonstrates the helper's limitation, not a measured robot command.

**Change:** retime positions to satisfy velocity and acceleration bounds; preserve and correctly scale valid planner timing. Validate the measured-start bridge too. `motions.py:1980` onward accepts cached preplans with up to 0.10 rad start mismatch; acceptance alone does not establish a feasible transition from the actual state at publication.

### 9. Cartesian path checks should reject excessive deviation

`goals.py:1299` computes lateral TCP deviation but logs it rather than enforcing a bound. Successful IK at sparse Cartesian waypoints followed by joint interpolation does not guarantee a straight TCP path between them. The inflight retarget path also needs the same complete validation and publication ownership as ordinary motion.

**Change:** adaptively subdivide where FK shows excessive deviation, then validate joint bounds, self/world collision, and timing for the final path. Serialize motion publication with an active-motion generation token so a late retarget cannot supersede a stopped or completed phase. Reconcile the state used for planning with the state at publication.

### 10. Calibration verification can PASS without TF — reproduced

`vision/verify_hand_eye.py:121`: when all camera poses are missing, `valid` is empty and both diversity rejection flags become false. Three identical reported target positions with three missing camera poses return verdict 0, PASS.

**Change:** missing TF or insufficient valid paired stations must return INCONCLUSIVE. Keep existing rotation/translation diversity checks, add held-out board poses and physical TCP reach measurements, and report per-axis bias separately from jitter. Low stationary jitter does not establish absolute localization accuracy.

## Highest-value speed work

### A. Repair and measure the GPU handoff

The saved benchmark CSVs contain:

| Recording | Samples | Median inference | p95 inference | Maximum |
|---|---:|---:|---:|---:|
| `benchmarks/results/yolo_timing_idle.csv` | 600 | 39.56 ms | 47.57 ms | 79.74 ms |
| `benchmarks/results/yolo_timing_contended.csv` | 400 | 529.94 ms | 710.77 ms | 829.65 ms |

All 400 contended samples exceed 120 ms; none of the 600 idle samples do. These are existing recordings, not a new controlled experiment or a promised 13× end-to-end improvement. Hardware, workload, and version provenance should be recorded before using them as current performance claims.

`vision/node.py:1145` ignores the boolean result of `wait_until_idle(timeout=0.5)` and announces paused even if the wait times out. `goals.py:4075` waits 0.30 s and falls back to another 0.12 s. This does not guarantee the work has drained. The capture/depth worker also needs to participate in the handoff.

**Change:** acknowledge a numbered pause request only after all relevant work drains; keep timeout distinct from success. Make repeated requests idempotent. Measure time from request to actual idle and time from resume to first fresh usable target. Existing process separation and GIL-switch tuning are already implemented and should not be counted as new optimizations.

### B. Eliminate repeated IK work conditionally

`goals.py:1428` preflights an approach/final chain; final execution solves much of that geometry again. Conditional branch-search failures can trigger many additional solver calls, each with its own internal seeds.

**Change:** preserve successful joint-chain results with goal, orientation, start-state, and collision-world versions. Reuse only when those inputs remain valid; otherwise use them as warm seeds. Bound retries and measure normal versus fallback paths separately. Batch independent candidates only with a solver configuration that supports the warmed tensor shapes; dependent Cartesian waypoints still require continuity checks.

### C. Keep repeated FK and file I/O off critical callbacks

`fk.py` already has an isolated locked model and batched FK, but repeated monitoring calls still construct device tensors and copy results back. Cache FK by complete joint-sample sequence and share it among consumers; benchmark CPU FK for monitoring separately from planner FK.

`node.py:1150` onward writes raw video within the camera callback/lock. Episode recording performs image/JSON output synchronously from the outcome path. Move recording to a bounded worker queue, define a drop policy, and measure joint-feedback age while recording. Serialize collision-world updates with the planner: `dynamic_obstacle.py:19` and `managers/motion_executor.py:537` update shared planning state without a consistent common owner.

### D. Optimize vision copies and sampling after frame correctness

Profile allocation, preprocessing, inference, mask transfer, geometry, and publication separately. Reuse inference buffers and batch necessary device-to-host transfers rather than transferring each crop independently. Use device events for GPU stages and a host monotonic clock for end-to-end latency; asynchronous execution makes host call duration alone misleading.

The dual-camera path uses random point subsampling (`vision/node.py:652`). Compare deterministic spatial coverage restricted to target regions, so depth stability does not vary purely from a new random sample. Evaluate the nearest-depth-fraction estimator (`vision/node.py:2523`) against eroded-mask coherent depth clusters and robust estimates; measure bias near branches, occlusion, and mask edges before changing the default. Refit surface-to-center correction for the chosen estimator and camera profile.

### E. Replace blind waits with bounded completion conditions

Final visual verification (`goals.py:6536`) sleeps for a configured duration. Once stamped observations exist, return when enough fresh valid verification frames arrive, with a maximum deadline. Apply the same principle to settling and reacquisition. Do not remove waits that currently compensate for unknown motion or observation state before fixing that state contract.

## Validation and implementation order

1. Fix joint-state ownership, motion completion status, tracker lifecycle, unused mode calibration, and missing-TF verdict. Add focused regression cases for these behaviors.
2. Introduce stamped atomic observations; fix association/freshness and reconcile reacquisition with tip anchoring.
3. Repair trajectory retiming, Cartesian validation, and retarget publication ownership.
4. Repair GPU handoff, then profile IK reuse, FK caching, recording queues, and image transfers on the deployed Jetson.
5. Compare the same recorded scenes and then held-out physical targets. Report observation age and per-stage p50/p95/p99 latency, per-axis localization bias and p95 error in millimeters, commanded-versus-measured TCP error, first-attempt grasp success, and total successful harvest cycle time including retries.

Checks performed during review: all 71 Python files parsed successfully; package XML and robot URDF parsed successfully. Isolated functions were extracted without importing ROS, starting cameras, loading CUDA models, or commanding hardware. These checks reproduced the behaviors described above; they are not integration tests. No speedup or improved physical accuracy has yet been measured from a code change.
