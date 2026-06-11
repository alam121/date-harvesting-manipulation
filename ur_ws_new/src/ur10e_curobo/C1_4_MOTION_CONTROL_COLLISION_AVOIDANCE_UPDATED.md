# C1.4 - Motion Control and Collision-Avoidance Logic

**Sub-deliverable:** Software architecture governing manipulator motion through date-palm canopies while reducing the risk of contact with fruits, fronds, the trunk, sensors, and the mounting platform.

**Completion:** 50%  
**Target:** December 2026  
**Invoicing eligible:** [TO BE CONFIRMED]

## 1. Basic Information

Motion control and collision avoidance are central components of the robotic date-palm automation system. During harvesting, pollination, and future pruning operations, the manipulator must move through a constrained and partially observed environment containing fronds, fruit bunches, stems, the palm trunk, sensors, and tool attachments. The system must maintain accurate end-effector positioning while avoiding sudden or unnecessarily long motions.

This sub-deliverable defines the software architecture used to convert perception-derived targets into feasible robot motions. The current prototype integrates:

- a UR10e manipulator;
- a Tesollo/Delto gripper with finger-force feedback;
- wrist-mounted ZED sensing;
- ROS 2 communication and supervision;
- cuRobo GPU-accelerated motion generation;
- robot-link collision geometry and static obstacle primitives;
- depth-derived voxel and Euclidean signed distance field (ESDF) processing;
- RViz visualization, operator commands, and execution monitoring.

The current harvesting routine uses staged motion rather than a single unconstrained move. A target is first stabilized and classified, the robot moves to a suitable staging or approach pose, vision can reacquire the target from the closer viewpoint, the end effector performs a short final insertion, the gripper closes, and the arm retracts along stored approach waypoints before moving to the drop-off configuration.

This staged structure supports repeatable operation and limits the effect of perception noise, inverse-kinematics branch changes, and long motions near the canopy.

## 2. System Architecture

The software is divided into perception, goal management, planning, execution, gripper feedback, and visualization layers.

1. **Perception and target selection:** ZED images and depth information are used to detect fruit, estimate 3D positions, score candidates, and publish the selected target in the robot base frame.
2. **Goal management:** Stable target observations are accepted and classified by height and lateral location. Bunch-relative image coordinates are preferred when available.
3. **Staging and approach generation:** The software selects center, left, or right approach behavior and computes a standoff pose with an appropriate tool orientation.
4. **Trajectory generation:** Larger motions use cuRobo planning. Short, precise motions use seeded inverse kinematics and interpolated joint trajectories to preserve the current kinematic branch.
5. **Trajectory safety checks:** Candidate paths are checked for joint wraparound, abrupt joint changes, excessive Cartesian detours, overshoot, and critical robot self-clearance.
6. **Execution and monitoring:** Joint trajectories are sent through the ROS 2 scaled joint trajectory controller. Robot state, stop requests, and target reach conditions are monitored.
7. **Grasp and retraction:** Gripper force profiles are used to estimate contact and grasp quality. The arm retracts along stored trajectory states before proceeding to drop-off.

The vision and planning workloads share GPU resources. Vision can therefore be placed in a paused or reacquisition mode during planning and precision motion, reducing GPU contention and preventing target switching after the robot commits to a fruit.

## 3. Methodology

### 3.1 Robot and Environment Representation

The UR10e, gripper, camera, tool center point, and mounting components are represented through the robot kinematic configuration and collision-sphere model. Static environmental geometry is maintained in a single configuration and used for both planning and RViz visualization.

The current static world includes floor/table and trunk primitives. Where cuRobo's configured collision checker requires cuboids, cylindrical objects are conservatively converted to bounding cuboids.

Depth data can additionally be converted into a local voxel representation. The configured workspace is approximately:

- dimensions: 1.5 m x 1.5 m x 1.5 m;
- voxel resolution: 0.02 m;
- maximum ESDF distance: 0.30 m;
- nominal verification safety margin: 0.03 m.

The voxel manager caches incoming point-cloud data and creates an on-demand snapshot. This avoids changing the obstacle map while a trajectory is being generated.

### 3.2 Perception-Driven Goal Generation

Fruit detections are transformed into the robot base frame using calibrated sensor-to-robot transformations. Target acceptance includes observation stability and target tracking so that isolated noisy detections do not immediately command motion.

The selected fruit is classified using:

- vertical image position or base-frame height;
- bunch-relative horizontal and vertical coordinates when a bunch detection is available;
- full-image position or trunk-relative location as fallback information.

This classification selects a center, left, or right motion corridor. Very-low targets can be forced to the center strategy to avoid unsafe side postures.

Once execution begins, a target-lock message prevents the vision system from switching to another fruit. A local reacquisition stage can refine the target position after staging or approach, when the camera has a closer and less occluded view.

### 3.3 Staged Harvesting Motion

For each accepted target, the software can define:

- a known home or side-home reference posture;
- an optional side staging pose;
- an approach or standoff pose;
- a final grasp pose;
- a partial reverse path;
- a drop-off and return-home path.

Low left and low right targets use side-specific standoff offsets and known side-home joint configurations. The staging pose, approach corridor, and wrist orientation are constructed consistently to reduce elbow and wrist branch changes.

Center and mid/high targets use their own configurable offsets and orientation rules. The final motion is deliberately short and slower than the global repositioning motion.

### 3.4 Motion Planning

A robot trajectory is represented as a sequence of six-joint configurations:

$$
\mathbf{q} = \{\mathbf{q}_1,\ldots,\mathbf{q}_N\}, \qquad
\mathbf{q}_k \in \mathbb{R}^6.
$$

Finite differences provide joint velocity, acceleration, and jerk estimates:

$$
\dot{\mathbf{q}}_k \approx
\frac{\mathbf{q}_{k+1}-\mathbf{q}_k}{\Delta t},
$$

$$
\ddot{\mathbf{q}}_k \approx
\frac{\mathbf{q}_{k+1}-2\mathbf{q}_k+\mathbf{q}_{k-1}}{\Delta t^2},
$$

$$
\dddot{\mathbf{q}}_k \approx
\frac{\ddot{\mathbf{q}}_{k+1}-\ddot{\mathbf{q}}_k}{\Delta t}.
$$

The planner uses these quantities, joint limits, end-effector pose error, and configured collision geometry to produce a time-parameterized path. Conceptually, the optimization objective can be written as:

$$
J =
w_s J_{\mathrm{smooth}} +
w_c J_{\mathrm{collision}} +
w_o J_{\mathrm{orientation}}.
$$

The smoothness component penalizes large velocity, acceleration, and jerk:

$$
J_{\mathrm{smooth}} =
\sum_{k=1}^{N}
\left(
\alpha_1 \|\dot{\mathbf{q}}_k\|^2 +
\alpha_2 \|\ddot{\mathbf{q}}_k\|^2 +
\alpha_3 \|\dddot{\mathbf{q}}_k\|^2
\right).
$$

An orientation error can be represented using desired and actual unit quaternions:

$$
e_{\mathrm{ori}} =
1-\left(\mathbf{q}_d \cdot \mathbf{q}\right)^2.
$$

The squared quaternion dot product makes the expression invariant to the equivalent quaternion signs $\mathbf{q}$ and $-\mathbf{q}$.

Large global movements are generated with cuRobo. Short approach corrections and final insertion motions use seeded inverse kinematics. Each Cartesian waypoint is seeded from the previous solution to preserve continuity. If waypoint IK attempts to switch branches, the motion is retried or rejected depending on the endpoint joint displacement.

### 3.5 Trajectory Validation and Safety Guards

The current implementation adds several checks beyond basic planner success:

- **Joint wraparound rejection:** trajectories with excessive accumulated rotation on any joint are replanned or rejected.
- **Approach joint-step guard:** an abrupt change between adjacent approach waypoints triggers replanning; a persistently discontinuous path is rejected.
- **Path-length sanity check:** the Cartesian path length is compared with the straight-line displacement. For an excessive detour, an additional plan is generated and the shorter valid path is selected.
- **Overshoot trimming:** trailing waypoints after the closest valid approach to the goal can be removed.
- **Forearm-to-tool self-clearance check:** sampled collision spheres on `forearm_link` and `tool0` are checked before execution. A path below the configured safety threshold can be truncated at the last safe waypoint or rejected when no useful safe prefix exists.
- **Stop handling:** stop requests prevent trajectory publication or halt the active sequence.
- **Reach monitoring:** the measured end-effector position is compared with the commanded endpoint, with optional path-deviation logging.

These checks address practical failure modes observed during laboratory development, including wrist wraparound, inverse-kinematics branch changes, unnecessarily long paths, endpoint overshoot, and configurations associated with UR protective clamping.

### 3.6 Voxel and ESDF Collision Verification

The voxel obstacle module converts a filtered point cloud into an occupancy grid and computes an ESDF. The ESDF provides the distance from each grid location to the nearest observed obstacle and can be used to test trajectory samples against a configurable safety margin.

The implemented verification path:

1. takes a fresh depth snapshot;
2. filters and voxelizes the point cloud;
3. computes the ESDF;
4. samples trajectory waypoints;
5. obtains the end-effector position through forward kinematics;
6. checks the ESDF distance, excluding a small region around the intended target;
7. requests replanning when a sampled point violates the safety margin.

At the current development stage, this module is an optional pre-execution verification layer. The configuration flag `verify_before_execute` is currently `False` by default, and the generated voxel grid is not currently injected into cuRobo's trajectory-optimization collision world. Therefore, the present implementation should be described as **on-demand ESDF generation and optional endpoint-path verification**, not continuous full-body ESDF optimization.

Full robot-link checking against the depth ESDF and direct integration of the voxel world into cuRobo remain planned development tasks.

### 3.7 Retraction and Grasp Feedback

Approach and final trajectory states are stored during execution. After gripper closure, the system reverses a sufficient portion of the stored path to create Cartesian clearance from the bunch. Side approaches use a larger configurable reverse distance than center approaches.

The gripper records force values during closure. Grasp assessment uses:

- the point in the closure sequence at which force first rises;
- early-stop behavior;
- per-finger force changes;
- a post-retraction median-force hold check.

This is gripper-force-based grasp verification. The current code does not implement the flange force-torque Cartesian admittance law described in earlier drafts. UR10e flange force-torque compliance should only be reported after the corresponding controller and validation data are added.

## 4. Results and Current Outcomes

The current prototype provides the following implemented capabilities:

1. Stable perception-driven target acquisition and target locking.
2. Bunch-relative left, center, right, low, and very-low motion classification.
3. Staged side approaches based on known safe joint postures.
4. GPU-accelerated cuRobo planning for larger movements.
5. Seeded inverse-kinematics trajectories for short, precise final motions.
6. Joint wraparound, joint-step, path-length, overshoot, and self-clearance guards.
7. On-demand voxel/ESDF generation and optional pre-execution trajectory verification.
8. RViz publication of goals, planned paths, collision geometry, and vision overlays.
9. Stored-path partial reversal after grasp.
10. Gripper force-profile analysis and post-retraction hold verification.
11. Per-phase and full-cycle timing logs for laboratory evaluation.

The system has progressed from isolated pose execution to an integrated harvesting state machine covering target acceptance, approach, reacquisition, final insertion, grasp, retraction, hold verification, drop-off, and return home.

### 4.1 Quantitative Validation to Add

The following values should be extracted from laboratory logs before final submission:

| Metric | Result |
| --- | --- |
| Number of attempted harvesting cycles | [TO ADD] |
| Planning success rate | [TO ADD] |
| Median and 95th-percentile planning time | [TO ADD] |
| Median approach execution time | [TO ADD] |
| Median complete cycle time | [TO ADD] |
| End-effector endpoint error | [TO ADD] |
| Minimum measured/modelled clearance | [TO ADD] |
| Number of rejected or replanned unsafe trajectories | [TO ADD] |
| Grasp success and post-retraction hold rate | [TO ADD] |
| Number of unintended contacts or protective stops | [TO ADD] |

The software already records phase timing, cycle timing, path deviation, replan events, self-clearance, and grasp outcomes. These logs can be aggregated without changing the motion architecture.

## 5. Remaining Work

Work planned toward the December 2026 target includes:

- enable and validate voxel trajectory verification under controlled test conditions;
- integrate depth-derived obstacles directly into cuRobo's collision world;
- extend ESDF verification from end-effector samples to all relevant robot links and tool geometry;
- validate behavior with moving fronds and repeated environmental snapshots;
- add calibrated UR10e flange force-torque interaction control if required by the operational procedure;
- collect statistically meaningful planning, tracking, clearance, grasp, and cycle-time results;
- validate pollination and pruning tool configurations separately from harvesting;
- complete field testing under representative canopy, lighting, occlusion, and wind conditions.

## 6. Evidence and Traceability

The implementation is maintained in the ROS 2 workspace. Principal evidence includes:

| Evidence | Location |
| --- | --- |
| Harvest state machine, target lock, IK motion, trajectory guards, retraction, timing | `ur10e_curobo/goals.py` |
| Planner, obstacle, speed, clearance, and logging configuration | `ur10e_curobo/config.py` |
| cuRobo initialization and trajectory publication | `ur10e_curobo/managers/motion_executor.py` |
| Voxel occupancy, ESDF generation, snapshots, and verification | `ur10e_curobo/voxel_obstacle.py` |
| Vision, target tracking, depth fusion, and target publication | `ur10e_curobo/vision/node.py` |
| Approach-direction and local fruit-clearance scoring | `ur10e_curobo/vision/scoring.py` |
| Vision and classification-zone visualization | `ur10e_curobo/vision/visualization.py` |
| Gripper force acquisition and closure profiles | `ur10e_curobo/delto_gripper_controller.py` |
| Planned-path and collision marker visualization | `ur10e_curobo/markers.py` |
| RViz operator interface | `rviz_ur10e_panel/` |

Supporting evidence for the final deliverable should include:

- RViz screenshots showing robot geometry, target poses, and planned paths;
- PolyScope records confirming joint and TCP execution;
- depth-cloud and voxel/ESDF visualizations;
- representative planner and safety-check logs;
- videos of laboratory approach, grasp, reverse, and drop-off cycles;
- a summarized quantitative validation table.

## 7. Figure Captions

**Figure C1.4.1.** UR10e PolyScope Teach Pendant interface used during validation of ROS 2 trajectory execution, TCP alignment, joint configuration, and protective-stop status.

**Figure C1.4.2.** On-demand voxel and ESDF processing pipeline. A local ZED point cloud is filtered and voxelized, an ESDF is computed, and sampled trajectory positions can be checked against a configured obstacle-clearance margin.

**Figure C1.4.3.** Staged obstacle-aware harvesting motion used during laboratory tests: side or center staging, approach, target reacquisition, final insertion, grasp, partial path reversal, drop-off, and return home.

