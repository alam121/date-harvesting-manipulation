# UR10e Date Harvesting Codebase Architecture

This is a visual map of how the main pieces fit together: vision detects and localizes dates, the motion node turns selected goals into cuRobo/IK trajectories, RViz/GUI sends commands, and the gripper/force stack verifies the grasp.

## 1. Package Map

```mermaid
flowchart TB
    launch["launch/combined.launch.py<br/>Starts UR driver, main node, vision, teleop"]

    subgraph ur10e["ur10e_curobo Python package"]
        main["main.py<br/>entry point"]
        node["node.py<br/>UR10eCuroboMoveIt ROS node"]
        cfg["config.py<br/>planner, joints, offsets, speed knobs"]
        goals["goals.py<br/>harvest state machine"]
        motions["motions.py<br/>home/dropoff/manual motion helpers"]
        gripper["gripper.py<br/>adaptive open/close + outcome callbacks"]
        fk["fk.py<br/>FK + fast IK helpers"]
        voxel["voxel_obstacle.py<br/>depth cloud -> cuRobo voxel world"]

        subgraph managers["managers/"]
            cm["config_manager.py<br/>declares/loads runtime config"]
            sm["state_manager.py<br/>joint state, robot state, trunk, markers"]
            me["motion_executor.py<br/>cuRobo setup, trajectory pub, teleop, obstacles"]
        end

        subgraph vision["vision/"]
            vnode["node.py<br/>ZED + YOLO perception loop"]
            vcfg["config.py<br/>camera/model/zone thresholds"]
            yolo["yolo_thread.py<br/>TensorRT YOLO inference thread"]
            zed["zed_utils.py<br/>camera settings"]
            scoring["scoring.py<br/>target score + approach direction"]
            viz["visualization.py<br/>image overlay + classification zones"]
            track["tracking.py<br/>fruit ID stability / blacklist"]
        end
    end

    subgraph rviz["rviz_ur10e_panel"]
        panel["ur10e_panel.cpp<br/>RViz command panel"]
        logpanel["log_panel.cpp<br/>ROS log viewer"]
        gripanel["gripper_panel.cpp<br/>force/gripper monitor"]
    end

    launch --> main --> node
    node --> cm
    node --> sm
    node --> me
    node --> goals
    goals --> motions
    goals --> gripper
    goals --> fk
    me --> voxel
    vnode --> yolo
    vnode --> zed
    vnode --> scoring
    vnode --> viz
    vnode --> track
    panel --> node
```

## 2. ROS Runtime Data Flow

```mermaid
flowchart LR
    camera["ZED / ZED Mini / LiDAR"]
    vision["Vision node<br/>zed_date_detector_ros"]
    main["Main motion node<br/>ur10e_curobo_moveit_node"]
    curobo["cuRobo planner<br/>MotionGen + IK"]
    robot["UR controller<br/>joint_trajectory_controller"]
    gripper["Delto gripper<br/>force + commands"]
    rviz["RViz panel / GUI"]

    camera -->|"image + depth"| vision
    vision -->|"/external_goal_pose"| main
    vision -->|"/datefruit_direction"| main
    vision -->|"/fruit_radius"| main
    vision -->|"/vision/all_fruit_poses"| main
    vision -->|"/trunk_position"| main
    vision -->|"/zed_depth_pointcloud"| main
    vision -->|"/vision/display"| rviz

    rviz -->|"/ui_command<br/>subscribe, execute, confirm, update_voxel"| main
    rviz -->|"/vision/overlay_command"| vision

    main -->|"/vision/mode<br/>full / reacquire / paused"| vision
    main -->|"/target_lock"| vision
    main -->|"/exclude_fruit_positions"| vision

    main --> curobo
    curobo -->|planned joint states| main
    main -->|"/scaled_joint_trajectory_controller/joint_trajectory"| robot
    robot -->|"/joint_states"| main
    robot -->|"/robot_program_running"| main

    main -->|"open / close"| gripper
    gripper -->|"/gripper/force"| main

    main -->|"/goal_info, markers, paths"| rviz
```

## 3. Harvest Cycle Motion Pipeline

```mermaid
flowchart TB
    A["User presses Subscribe<br/>/ui_command = subscribe"] --> B["goals.subscribe_to_goal_pose()<br/>collect stable /external_goal_pose"]
    B --> C["Goal queued in ThreadSafeGoalList<br/>classification: LOW / VERY LOW / MID-HIGH + LEFT / CENTER / RIGHT"]
    C --> D["User presses Execute"]
    D --> E["plan_and_execute()<br/>pop next goal, lock target, pause vision"]

    E --> F{"Side target?"}
    F -->|"LOW LEFT/RIGHT"| G["Staging-from-approach<br/>HOME_LEFT_LOW_STAGING / HOME_RIGHT_LOW_STAGING<br/>same corridor + same side-low wrist orientation"]
    F -->|"CENTER or skipped side"| H["Use current/home posture"]
    F -->|"non-low side fallback"| I["Fixed side preset or normal center approach"]

    G --> J["Optional side-home reacquire<br/>vision mode = reacquire"]
    H --> K["Compute APPROACH pose"]
    I --> K
    J --> K

    K --> L{"Approach distance small?"}
    L -->|"yes"| M["_direct_ik_move()<br/>seeded IK + interpolation"]
    L -->|"no"| N["plan_and_send()<br/>cuRobo trajectory optimization"]
    M --> O["Wait for TCP at approach<br/>open gripper adaptively"]
    N --> O

    O --> P{"Reacquire after approach?"}
    P -->|"yes"| Q["reacquire_goal_pose()<br/>tight XYZ search around seed"]
    P -->|"no"| R["Use seed"]
    Q --> S["Compute FINAL target offsets"]
    R --> S

    S --> T["_direct_ik_move(FINAL)<br/>Cartesian IK waypoints + branch checks"]
    T --> U["Close gripper<br/>force profile + grasp classifier"]
    U --> V["Optional depth correction<br/>late-contact nudge"]
    V --> W["Partial reverse<br/>clearance out of bunch"]
    W --> X["Hold check<br/>median force window"]
    X --> Y["Move DROP-OFF"]
    Y --> Z["Open gripper, return HOME, release target lock"]
```

## 4. Vision Pipeline

```mermaid
flowchart TB
    A["ZED image frame"] --> B["YoloThread<br/>TensorRT predict()"]
    C["ZED Mini / stereo / LiDAR depth"] --> D["Depth alignment<br/>point cloud / dense map"]

    B --> E["detections.py<br/>boxes + masks"]
    D --> F["_extract_target_3d()<br/>mask/bbox depth -> XYZ"]
    E --> F

    F --> G["FruitTracker<br/>stable IDs + blacklist attempts"]
    G --> H["scoring.py<br/>visibility, depth, trunk relation, reachability"]
    H --> I["Best target selection<br/>hysteresis + target lock"]

    I --> J["Publish best goal<br/>/external_goal_pose"]
    I --> K["Publish geometry<br/>/datefruit_direction, /fruit_radius, /vision/all_fruit_poses"]
    I --> L["Publish trunk<br/>/trunk_position, /trunk_position_cam"]
    D --> M["Publish depth cloud<br/>/zed_depth_pointcloud"]

    E --> N["visualization.py<br/>target labels, rejection overlays, class zones"]
    I --> N
    N --> O["/vision/display"]
```

## 5. Key Files By Responsibility

| Area | File | What to look for |
| --- | --- | --- |
| Main ROS node | `ur10e_curobo/node.py` | ROS pubs/subs, UI commands, goal info, voxel update, set-home-current |
| Harvest sequence | `ur10e_curobo/goals.py` | goal subscribe, classification, staging, approach, reacquire, final, reverse, dropoff |
| Planner parameters | `ur10e_curobo/config.py` | joints, offsets, staging extra, speeds, final offsets, logging knobs |
| cuRobo setup | `ur10e_curobo/managers/motion_executor.py` | MotionGen init, world config, trajectory publisher, voxel manager |
| Robot state | `ur10e_curobo/managers/state_manager.py` | joint states, robot running, trunk, fruit image metadata |
| Vision node | `ur10e_curobo/vision/node.py` | camera loop, YOLO/depth fusion, publications, reacquire/paused mode |
| YOLO thread | `ur10e_curobo/vision/yolo_thread.py` | inference model, input size, predict settings |
| Vision overlays | `ur10e_curobo/vision/visualization.py` | target drawing, classification zone overlay |
| RViz panel | `rviz_ur10e_panel/src/ur10e_panel.cpp` | buttons and `/ui_command` / overlay publishers |

## 6. Mental Model

The system has two big loops:

1. **Vision loop:** continuously detects fruit, estimates 3D position, scores candidates, and publishes the best target.
2. **Motion loop:** waits for a selected stable target, plans a staged approach, performs final TCP insertion, grasps, reverses, checks hold, and drops off.

The most important boundary is this:

```text
vision/node.py publishes facts about the scene
goals.py decides how the robot should act on those facts
motion_executor.py/cuRobo turns that decision into robot joint trajectories
```

