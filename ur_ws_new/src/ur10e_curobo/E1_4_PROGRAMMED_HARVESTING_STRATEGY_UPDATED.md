# E1.4 - Programmed Harvesting Strategy

**Status:** Updated against the current ROS 2 implementation  
**Primary strategy:** Grip, controlled reverse, transport, and drop-off

## 1. Basic Information

This sub-deliverable documents the programmed harvesting strategy developed for
the robotic date-harvesting prototype. The software converts AI-based fruit
detections into executable actions using the UR10e manipulator and the
Tesollo/Delto three-finger adaptive gripper.

The current implementation focuses on a grip-and-pull workflow:

1. detect and localize a target fruit;
2. stabilize and lock the selected target;
3. classify the target as center, left, right, low, or very low;
4. move through a suitable staging and approach corridor;
5. execute a short final insertion;
6. close the gripper while recording finger-force profiles;
7. reverse along stored approach waypoints;
8. verify that the fruit remains held;
9. move through a safe drop-off route;
10. release the fruit and return home.

Precision cutting and vibration-based harvesting remain future candidate
strategies and require separate tools, control logic, and validation.

## 2. Perception and Target Selection

In the current dual-camera configuration, the ZED X One supplies the primary
detection image and the ZED X Mini supplies depth information. YOLO detections
are combined with aligned depth data to estimate fruit positions. Calibrated
camera-to-robot transformations convert each target into the robot base frame.

The target-selection logic uses:

- detection confidence;
- depth consistency and visible-mask quality;
- distance and reachability information;
- ellipse/short-axis availability;
- target-history and sticky-selection bonuses;
- neighboring-fruit clearance when the target is locked.

The selected target is stabilized over multiple observations. Once execution
begins, a target-lock message prevents vision from switching to a different
fruit.

An optional reacquisition mode can refine a target from a closer viewpoint.
This capability is implemented, but `reacquire_after_approach` is currently
disabled by default. It should therefore be described as an available
configuration rather than a mandatory stage in every harvesting cycle.

## 3. Approach Geometry

The perception module estimates useful approach geometry from the fruit mask
and depth data.

Ellipse fitting provides a short-axis estimate when a suitable contour is
available. The direction is transformed from the camera frame into the robot
base frame before being published to the motion system.

A depth-derived heatmap identifies a visible front-surface region. Its peak is
temporally smoothed as:

$$
\mathbf{p}_{k}^{s}
=
0.7\mathbf{p}_{k-1}^{s}
+
0.3\mathbf{p}_{k}.
$$

The approach direction can blend the estimated surface normal and heatmap
direction:

$$
\mathbf{d}
=
\operatorname{normalize}
\left(
0.7\mathbf{n}
+
0.3\mathbf{d}_{h}
\right).
$$

This reduces frame-to-frame jitter and provides a more stable local reference
than the segmentation-box center alone.

## 4. Target Classification and Staging

The motion layer classifies targets using bunch-relative image coordinates
when available. Full-image coordinates and trunk-relative position provide
fallback information.

Different target classes use different motion corridors:

- center targets use the center home and center approach;
- side targets use left or right staging postures;
- low side targets use calibrated lateral standoff offsets;
- very-low targets use a dedicated center strategy and preflight safety search.

A front-zone override avoids unnecessary side motion when the fruit is still
visually in front of the robot, even if its bunch-relative coordinate lies near
an edge.

## 5. Very-Low-Target Safety Preflight

Very-low targets can place the UR10e forearm and flange in tightly folded
configurations. The current implementation performs a preflight search before
executing these approaches.

The search evaluates multiple:

- inverse-kinematics branches;
- tool pitch offsets;
- wrist rotations;
- joint-displacement costs;
- Cartesian path lengths;
- forearm-to-tool clearances.

Only candidates above the configured minimum clearance are accepted. The
current configuration requires at least 50 mm of modeled sphere-surface
clearance for very-low preflight candidates.

If no safe candidate exists, the robot returns home and skips the target.
During normal approaches, safety rejection can also trigger wrist-orientation
variants at approximately -20, +20, -40, and +40 degrees before the target is
abandoned.

## 6. Motion Planning and Trajectory Safety

Larger robot movements use cuRobo trajectory generation. Short approach
corrections and final insertion motions use seeded inverse kinematics and
interpolated joint trajectories to preserve the current kinematic branch.

The implementation validates candidate motions using:

- joint wraparound rejection;
- adjacent-waypoint joint-step limits;
- Cartesian path-length ratio checks;
- endpoint overshoot trimming;
- reach and stall monitoring;
- forearm-to-tool collision-sphere clearance;
- stop-request handling.

The collision model now includes a 37.5 mm UR flange sphere on `tool0`.
Forearm collision-sphere radii were also refined. Based on an observed
protective clamping event at a modeled clearance of approximately 41.8 mm, the
general self-clearance threshold is currently set to 45 mm.

Unsafe final trajectories are rejected. Unsafe approach trajectories are also
rejected rather than intentionally stopping at an arbitrary truncated pose.

Depth-derived voxel and ESDF processing is available as an optional
pre-execution verification layer. It is disabled by default and is not yet
integrated as full-body dynamic obstacle geometry inside cuRobo optimization.

## 7. Adaptive Grasp Execution

The gripper closes incrementally over ten commanded steps. Force is evaluated
as the absolute change from the open-gripper baseline:

$$
\Delta F_i
=
\left|F_i-F_{i,0}\right|.
$$

The current controller uses a 4.0 N baseline-delta threshold for contact on
each of the three active channels. Closure stops early when all three channels
reach that threshold. Separately, the recorded force profile identifies the
first step where any finger exceeds a 0.5 N change.

Grasp quality is evaluated using:

- first-contact timing within the closure sequence;
- whether closure stopped early;
- the final force changes on all three fingers;
- a force-confirmation condition when at least two fingers exceed 1.5 N.

A contact ratio near 40% of the closure sequence is treated as the nominal
target. When enabled, regrip correction can use force imbalance and contact
timing to calculate small lateral, depth, and vertical adjustments. This
automatic regrip feature is currently disabled by default.

The earlier draft's statement that the runtime controller enforces a 2 N
maximum force per finger is not consistent with the current software. A 2 N
value may only be reported as an experimental handling limit if it is supported
by separate calibration records and distinguished from the implemented 4 N
closure threshold.

The software also does not currently implement the rule
$F_{\mathrm{meas}}>F_{\mathrm{limit}}\Rightarrow$ immediate arm retraction.
Force limits stop gripper closure; robot retraction begins as the next
supervisory stage.

## 8. Controlled Reverse and Hold Verification

Approach and final motion states are stored during execution. After closure,
the robot reverses along these stored waypoints instead of generating an
unrelated withdrawal path.

The nominal reverse clearances are:

- 0.35 m for center approaches;
- 0.55 m for side approaches.

After reversal, force is sampled over a 0.30 s window and median baseline
changes are computed. The fruit is considered held when any of the following
conditions is satisfied:

- at least two fingers exceed 1.5 N;
- the strongest finger exceeds 2.3 N;
- the sum of the three force changes exceeds 3.0 N.

This post-retraction check distinguishes persistent fruit contact from a
temporary touch during closure.

## 9. Drop-Off Safety and Cycle Optimization

Drop-off planning now begins in a background thread while the reverse and hold
check are being completed.

For side approaches, the robot first returns to center home before moving to
drop-off. This reduces the risk of crossing the trunk from a lateral posture.

The system:

- waits for the background plan instead of starting a competing planner;
- executes the preplanned trajectory when available;
- retries drop-off planning if necessary;
- attempts a center-home recovery route after a failed direct plan;
- keeps the gripper closed and ends automatic execution if safe drop-off
  recovery fails.

Home, drop-off, final, gripper-open, gripper-close, and reverse timing
parameters have also been reduced or adjusted to improve cycle time while
retaining configured velocity, acceleration, and clearance limits.

## 10. Supervisory State Sequence

The harvesting behavior is implemented as a procedural supervisory state
machine with published motion phases:

| State | Function |
| --- | --- |
| Idle | Wait for an operator or scheduler request. |
| Perception | Detect, localize, stabilize, and select a fruit. |
| Planning/Staging | Classify the target and select a safe motion corridor. |
| Approach | Move to the standoff pose using cuRobo or direct IK. |
| Final | Execute the short precision insertion. |
| Grasp | Close the gripper and evaluate the force profile. |
| Reverse/Hold | Retract along stored waypoints and verify persistent contact. |
| Home/Drop-off | Clear the trunk when required and transport the fruit. |
| Release | Open the gripper and return to home. |
| Error/Recovery | Stop automatic progression while retaining a safe manual recovery path. |

The system exposes explicit states and transition conditions, although the
implementation is distributed through the harvesting procedure rather than a
standalone generic FSM framework.

## 11. Results Requiring Evidence

The following statements from the earlier draft should only remain in the
final deliverable when the corresponding test records, plots, photographs, or
logs are attached:

| Claimed result | Required evidence |
| --- | --- |
| Maximum measured force of 0.27 N | Raw force logs, baseline convention, calibration, and plotted trials |
| No visible damage in 14 of 15 trials | Trial sheet, sample definition, inspection method, and photographs |
| Sub-centimeter trajectory accuracy | Measured endpoint-error distribution |
| Real-time replanning under canopy motion | Planner timestamps and a demonstrated replan event |
| Collision-free trajectory execution | Test count, obstacle setup, clearance logs, and protective-stop count |

Experimental force values must also distinguish raw signed sensor readings
from baseline-relative force changes used by the controller.

## 12. Current Outcomes

The programmed harvesting system currently provides:

1. perception-driven target selection and lock;
2. bunch-relative target classification;
3. center and side staging strategies;
4. dedicated very-low-target IK and clearance preflight;
5. cuRobo and direct-IK motion generation;
6. calibrated forearm-to-flange self-clearance checks;
7. heatmap and orientation-assisted approach geometry;
8. incremental three-finger closure with force-profile recording;
9. contact-timing and force-based grasp assessment;
10. stored-path reverse and post-retraction hold verification;
11. side-safe center-home routing before drop-off;
12. background drop-off preplanning and guarded failure recovery;
13. per-phase and complete-cycle timing logs.

## 13. Evidence and Traceability

| Function | Source |
| --- | --- |
| Harvesting sequence and recovery logic | `ur10e_curobo/goals.py` |
| Motion, clearance, speed, and grasp configuration | `ur10e_curobo/config.py` |
| Incremental closure and force profile | `ur10e_curobo/delto_gripper_controller.py` |
| Target localization and heatmap smoothing | `ur10e_curobo/vision/node.py` |
| Candidate scoring and neighboring-fruit clearance | `ur10e_curobo/vision/scoring.py` |
| Robot collision spheres | `curobo/content/configs/robot/spheres/ur10e.yml` |
| Joint-space and drop-off preplanning | `ur10e_curobo/motions.py` |

