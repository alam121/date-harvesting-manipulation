# Review fixes — September 16, 2026

Application changes are uncommitted and preserve the pre-existing workspace edits.

Implemented:

- Complete, finite arm joint messages update state atomically. Finger-only, partial, duplicate-name, and malformed messages do not erase it. Missing velocity remains unknown.
- Cartesian and joint completion require stationary feedback, with no extra fixed dwell. Stale feedback, stop, shutdown, exceptions, and a past closest approach cannot establish success. The direct-motion caller now propagates failure. Short motions can complete without a 1 cm departure.
- Trajectories use exact cubic Hermite extrema (and conservative bounds for optional quintics) to avoid unnecessary duration increases while respecting derivative limits. Velocity and acceleration fields are scaled consistently with time. Cancellation discards the whole unfinished message. Cached preplans receive a measured-start bridge, renewed constraint validation, and retiming.
- Mode-specific closure offsets are used in approach, preflight, and final construction. Accepted bounded reacquisition displacement is propagated into frozen tip geometry.
- Reacquisition consumes one immutable position/quality observation, requires an acquisition after settling, bounds both lateral and depth distance, counts distinct acquisition stamps, and checks stability across all axes. Missing data returns a seed fallback rather than fabricated refinement.
- `/vision/fruit_observation` carries a JSON object in `std_msgs/String`: base frame, acquisition nanoseconds, sequence, positions, and matching quality rows. Existing display/legacy topics remain published. Motion does not use their independently received quality for refinement.
- Mini-only capture retains the exact RGB-D packet submitted to inference and uses the SDK image timestamp. Metric TF lookup no longer silently substitutes latest TF. Legacy multi-camera/LiDAR temporal alignment still needs a separate integration pass.
- Tracks use bounded one-to-one base-frame association, confirmed persistence, and consecutive confirmation. Attempt notifications use fruit position rather than offset TCP position. Smoothing resets on identity changes. Old persisted targets are excluded from fresh observation packets; heatmap geometry is recomputed per accepted acquisition. Tip confirmation rejects old or duplicate acquisition stamps.
- Pause requests and acknowledgements carry request IDs. Pending YOLO work marks the worker busy at submission; timed-out drains cannot announce success. The Mini capture lock participates in the drain. Harvest pause timeouts stop the batch rather than assume a fixed sleep completed the handoff.
- Final retarget publication and completion share a lock and generation counter. Late or substantially stale proposals are rejected. Direct paths and retargets receive cuRobo constraint validation; required Cartesian direct paths exceeding the lateral tolerance are rejected.
- Hand-eye verification reports INCONCLUSIVE when paired TF stations are missing or insufficient.

Validation performed:

```bash
python -m unittest discover -s ur_ws_new/src/ur10e_curobo/tests -v
git diff --check
```

44 regression tests passed (including the best-target publication path with real ROS messages and blank-frame display suppression). All 75 Python files in the package parsed successfully. A real ROS `JointTrajectory` serialization/deserialization round-trip passed, and the installed ZED image timestamp API was verified without opening a camera. Driver-bound functions are extracted for most unit tests so imports cannot start hardware or initialize planning models. These are logic tests, not end-to-end ROS/GPU tests.

Deployment and behavior changes:

- Restart both motion and vision processes together: the new pause protocol and atomic refinement message need both updated implementations.
- Reacquisition can now time out honestly when perception is too old or absent. The configured seed-fallback policy remains in force.
- Missing joint velocity prevents arrival confirmation; verify that the active arm driver supplies it.
- Retiming can lengthen moves; strict path checks can reject paths previously executed. More frequent heatmap computation may increase processing cost. No speedup or physical accuracy gain is claimed yet.
- Robot/camera execution was not started. Validate recorded frames first, then measured TCP/grasp behavior on the deployed setup before relying on these changes for harvesting.

Remaining review work: multi-camera/LiDAR acquisition alignment; full collision-world update ownership; held-out calibration and depth-estimator evaluation; profiling-led IK reuse, asynchronous recording, and transfer optimization. The new checks validate sampled paths and interpolation derivative bounds; they are not a proof of continuous collision freedom or physical grasp accuracy. Very close fruit can still be ambiguous under position-only association.

## Speed correction

The initial patch added conservatism without a measured throughput benefit. The follow-up removes its unconditional 100 ms arrival dwell while retaining measured stationary velocity and freshness checks. It also:

- Runs TF lookup only for an accepted inference result, once for base and once for gripper. RGB-only iterations no longer perform those lookups, and Mini results no longer do a second pair.
- Caches FK for exactly unchanged joints and the same kinematics model, shared across the short-lived motion-executor facades. Changed joints or model identity invalidate the cache. Planning-time fallback refuses a cached pose from different joints. A cache miss transfers position and quaternion together.
- Uses exact cubic extrema instead of loose control-polygon bounds. A synthetic rest-to-rest one-radian test with identical limits serialized to 6 s versus the previous patch's 12 s. An already-within-limits example remained 0.2 s. These are trajectory durations, not robot timing measurements.

The regression test verifies 100 unchanged-state requests invoke FK once and that returned copies cannot corrupt the cache. Existing interpolated velocity/acceleration tests still pass. No hardware run or total harvest-cycle speedup is claimed.

Crash fix: initialize the selected base-frame point before publishing its tip timestamp. The new regression reproduced the reported UnboundLocalError before the fix and passes for valid, absent, and nonfinite tip inputs afterward. The package was rebuilt with colcon; source/build/install vision files match. The build completed with an existing setuptools/setuptools-scm compatibility warning. Running vision processes must be restarted to load the fix.

Display flicker fix: retain the annotated image between inference results instead of queuing blank RGB frames into the limited display slots. Raw preview remains available at startup, while paused, or after a 0.5 s inference stall. Camera motion no longer clears base-frame track confirmation. Unconfirmed detections appear as `date (confirming)` without becoming confirmed motion targets. Missing acquisition TF allows labeled 2D-only detection display but still prevents 3D target publication. Two display regression tests cover inter-result suppression and preview recovery. The package was rebuilt and source/build/install vision files match; live-camera behavior remains untested.
