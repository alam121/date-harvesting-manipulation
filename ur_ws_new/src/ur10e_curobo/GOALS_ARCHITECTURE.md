# `goals.py` Detailed Architecture

`goals.py` is the harvest state machine. It turns vision goals into a full robot behavior: accept a stable date, classify it, stage the arm, approach, reacquire, insert the TCP, grasp, reverse, verify hold, drop off, and return.

## 1. Responsibility Map

```mermaid
flowchart TB
    goals["goals.py"]

    subgraph input["Goal Input + Classification"]
        ts["ThreadSafeGoalList"]
        sub1["subscribe_to_goal_pose()"]
        subm["subscribe_multi_goals()"]
        side["image_lateral_side()<br/>LEFT / CENTER / RIGHT"]
        low["height logic<br/>LOW / VERY LOW / MID-HIGH"]
    end

    subgraph geom["Geometry + Orientation"]
        lowoff["low_side_standoff_offsets()"]
        wrist["side_low_wrist3_orientation()"]
        minrot["minimize_rotation_orientation()"]
        quat["quat_* helpers<br/>slerp, yaw, pitch, local-axis align"]
        dyn["compute_dynamic_side_home()<br/>older date-clamped helper"]
    end

    subgraph plan["Planning + Execution"]
        main["plan_and_execute()"]
        direct["_direct_ik_move()<br/>seeded IK, branch retry, Cartesian waypoints"]
        traj["plan_and_send()<br/>cuRobo plan, voxel check, path guard, trim"]
        rev["execute_partial_reverse()<br/>reverse stored approach/final states"]
        reacq["reacquire_goal_pose()<br/>tight vision search around seed"]
    end

    subgraph grasp["Grasp + Feedback"]
        lock["lock_target() / unlock_target()"]
        notify["notify_grasp_attempt()"]
        hold["after-reverse HOLD_CHECK"]
        learn["GraspRecord / learner hooks"]
    end

    goals --> input
    goals --> geom
    goals --> plan
    goals --> grasp

    sub1 --> ts
    subm --> ts
    sub1 --> side
    sub1 --> low
    main --> side
    main --> low
    main --> lowoff
    main --> wrist
    main --> minrot
    main --> direct
    main --> traj
    main --> reacq
    main --> rev
    main --> hold
    main --> lock
```

## 2. High-Level Harvest State Machine

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Subscribe: UI subscribe
    Subscribe --> GoalQueued: stable /external_goal_pose accepted
    GoalQueued --> Execute: UI execute

    Execute --> TargetLock: pop goal + /target_lock
    TargetLock --> SideStaging: side low target
    TargetLock --> ApproachBuild: center or no side staging

    SideStaging --> SideReacquire: optional reacquire from staging
    SideReacquire --> ApproachBuild
    SideStaging --> ApproachBuild: fallback/no reacquire

    ApproachBuild --> ApproachMove
    ApproachMove --> ReacquireAfterApproach: if enabled and not skipped
    ApproachMove --> FinalBuild: low-center or already reacquired
    ReacquireAfterApproach --> FinalBuild

    FinalBuild --> FinalMove
    FinalMove --> GripperClose
    GripperClose --> DepthCorrect: late contact
    GripperClose --> Reverse
    DepthCorrect --> Reverse

    Reverse --> HoldCheck
    HoldCheck --> SlipRetry: slip reacquire detects fruit still there
    SlipRetry --> GoalQueued: retry same fruit
    HoldCheck --> Dropoff
    Dropoff --> ReturnHome
    ReturnHome --> TargetUnlock
    TargetUnlock --> GoalQueued: more queued goals
    TargetUnlock --> Idle: queue empty
```

## 3. Single-Goal Acceptance Path

```mermaid
sequenceDiagram
    participant UI as RViz/GUI
    participant Main as node.py
    participant Goals as goals.py
    participant Vision as vision/node.py

    UI->>Main: /ui_command = subscribe
    Main->>Goals: subscribe_to_goal_pose(node)
    Goals->>Goals: reset_goal_tracking()
    Goals->>Vision: subscribe /external_goal_pose<br/>VOLATILE + BEST_EFFORT
    Vision-->>Goals: PoseStamped best date
    Goals->>Goals: collect recent XYZ readings
    Goals->>Goals: accept if stable pair after min_settle<br/>or median after max_wait
    Goals->>Goals: classify height from cy / z
    Goals->>Goals: classify side from bunch rx if available<br/>fallback to image cx / trunk_x
    Goals->>Main: append [x,y,z,qw,qx,qy,qz]<br/>to ThreadSafeGoalList
    Goals->>Vision: publish goal marker / update fruit obstacle
    Goals-->>UI: logs "Primary goal accepted..."
```

Important details:

- `subscribe_to_goal_pose()` accepts one goal only.
- `subscribe_multi_goals()` can queue several score-ordered fruits from `/vision/all_fruit_poses`.
- Lateral classification prefers `fruit_bunch_rel_x` over full-image `cx`, so side zones move with the bunch.
- Very low or lower-boundary fruits can force `CENTER`.

## 4. Classification Decision Logic

```mermaid
flowchart TB
    A["Accepted goal"] --> B{"fruit_image_norm exists?"}
    B -->|"yes"| C["Height from cy_norm<br/>cy > 0.60 => LOW"]
    B -->|"no"| D["Height from z<br/>z < LOW_Z_THRESH => LOW"]

    C --> E{"Very low?"}
    E -->|"cy >= very_low_center_cy_thresh<br/>or bunch rel_y >= lower band"| F["VERY LOW CENTER"]
    E -->|"no"| G["image_lateral_side()"]

    G --> H{"bunch rel_x exists?"}
    H -->|"yes"| I["Use bunch-relative thresholds<br/>low: low_left/right_thresh<br/>mid: mid_high_left/right_thresh"]
    H -->|"no"| J["Use image cx thresholds"]

    I --> K["LEFT / CENTER / RIGHT"]
    J --> K
    D --> L["Fallback side from fruit_x vs trunk_x"]
```

## 5. Side-Low Staging From Approach

This is the newer fix for the unnatural dynamic side-home motion.

```mermaid
flowchart TB
    A["LOW + LEFT/RIGHT target"] --> B["Pick fixed preset joints<br/>HOME_LEFT_LOW / HOME_RIGHT_LOW"]
    B --> C["low_side_standoff_offsets()<br/>x/y/z offset for this side"]
    C --> D["Build approach corridor vector<br/>dir = [-x_offset, -y_offset, 0]"]
    D --> E["side_low_wrist3_orientation()<br/>same wrist orientation used for approach"]
    E --> F["Compute staging position<br/>fruit + offset * (1 + side_home_staging_extra_m / norm(offset))"]
    F --> G["IK seeded from fixed side-home preset<br/>keeps branch close to known safe posture"]
    G --> H{"IK success?"}
    H -->|"yes"| I["plan_execute_js()<br/>HOME_LEFT_LOW_STAGING / HOME_RIGHT_LOW_STAGING"]
    H -->|"no"| J["plan_and_send() Cartesian fallback"]
    I --> K{"staging move ok?"}
    J --> K
    K -->|"no"| L["fallback fixed side-home preset"]
    K -->|"yes"| M["Proceed to side-home reacquire / approach"]
```

Key property:

```text
staging -> approach -> final uses the same side corridor and the same side-low wrist orientation.
```

That reduces the old problem where a dynamic side-home target was safe in XYZ but different in wrist/elbow posture.

## 6. Approach Pose Builder

```mermaid
flowchart TB
    A["plan_and_execute() has x,y,z"] --> B{"is_low?"}

    B -->|"LOW"| C{"side approach?"}
    C -->|"yes"| D["low_side_standoff_offsets()<br/>side-specific x/y/z"]
    D --> E["side_low_wrist3_orientation()<br/>yaw wrist/front toward fruit corridor"]
    E --> F["approach = fruit + side offsets + orientation"]

    C -->|"no CENTER"| G["center low offsets<br/>low_center or very_low_center y/z"]
    G --> H["orientation = minimize_rotation_orientation()"]
    H --> I["approach = fruit + center offsets"]

    B -->|"MID/HIGH"| J["standoff in +Y<br/>mid_center_approach_y_offset<br/>right side can use larger standoff"]
    J --> K["orientation = minimize_rotation_orientation()<br/>optional center pitch"]
    K --> L["approach = [x, y + standoff, z + z_offset]"]
```

## 7. Motion Planner Wrappers

```mermaid
flowchart LR
    subgraph direct["_direct_ik_move()"]
        A1["current joints"] --> A2["cuRobo IK seeded from current"]
        A2 --> A3["Jacobian IK fallback"]
        A3 --> A4["nearest_joint_config()<br/>remove 2π wrap"]
        A4 --> A5["branch retry with perturbed seeds"]
        A5 --> A6{"label == FINAL?"}
        A6 -->|"yes"| A7["Cartesian IK waypoints<br/>seed each waypoint from previous"]
        A6 -->|"no"| A8["joint-space interpolation"]
        A7 --> A9["build trajectory + publish"]
        A8 --> A9
    end

    subgraph curobo["plan_and_send()"]
        B1["MotionGen.plan_single()"] --> B2["reject joint wraparound"]
        B2 --> B3["APPROACH max joint-step guard"]
        B3 --> B4["voxel collision verification"]
        B4 --> B5["path length ratio sanity check"]
        B5 --> B6["_STAGING detour reject"]
        B6 --> B7["FK path + trim overshoot"]
        B7 --> B8["build timed trajectory + publish"]
    end
```

Use rule:

- `_direct_ik_move()` is used for close, precise moves where branch consistency matters.
- `plan_and_send()` is used for larger global moves where cuRobo trajectory optimization is needed.

## 8. Reacquire Logic

```mermaid
flowchart TB
    A["reacquire_goal_pose(seed)"] --> B["vision mode = reacquire"]
    B --> C["settle depth for depth_settle_s"]
    C --> D["read latest_goal_pose first"]
    D --> E["scan all_fruit_poses if best target is not seed"]
    E --> F{"candidate near seed?"}
    F -->|"full 3D<br/>dxy <= radius and dz <= z_tol"| G["count stable reading"]
    F -->|"depth unstable<br/>XY-only gate"| H["use detected XY, seed Z"]
    F -->|"one dimension close"| I["merge close dimension with seed"]
    F -->|"no"| J["keep searching until timeout"]
    G --> K{"stable_count reached<br/>and Z_std OK?"}
    H --> K
    I --> K
    K -->|"yes"| L["return refined XYZ"]
    K -->|"no"| J
    J --> M{"timeout with enough stable data?"}
    M -->|"yes"| N["soft-accept"]
    M -->|"no"| O["return None, use seed"]
```

Normal places reacquire is used:

- From side staging before approach, if side target and `reacquire_after_approach` is enabled.
- After approach, unless already reacquired or low-center skip is active.
- Optional slip check after reverse if enabled.

## 9. Final TCP Insertion + Grasp

```mermaid
flowchart TB
    A["After approach/reacquire"] --> B["Compute final y/z offsets<br/>low side, low center, mid center"]
    B --> C["Recompute approach_dir<br/>current EE -> fruit"]
    C --> D["Reuse approach orientation"]
    D --> E{"low side?"}
    E -->|"yes"| F["small capped front tilt<br/>align local +Z/front toward fruit"]
    E -->|"no"| G["keep approach orientation<br/>or unpitched mid-center final"]
    F --> H["final_target = [x, y - offset_y, z + offset_z, quat]"]
    G --> H
    H --> I["_direct_ik_move(FINAL)"]
    I --> J{"IK ok?"}
    J -->|"no, tilted side-low"| K["retry original approach orientation"]
    J -->|"no"| L["plan_and_send FINAL_PLAN fallback"]
    J -->|"yes"| M["check overshoot along approach_dir"]
    K --> M
    L --> M
    M --> N["close gripper"]
    N --> O["force profile: first_contact, deltas, stopped_early"]
    O --> P{"grip accepted?"}
    P -->|"yes but late contact"| Q["DEPTH_CORRECT<br/>closed-gripper forward nudge"]
    P -->|"regrip enabled + weak"| R["open, correction, close again"]
    P -->|"accepted"| S["store GraspRecord"]
    Q --> S
    R --> S
```

## 10. Reverse, Hold Check, Dropoff

```mermaid
flowchart TB
    A["Grasp complete"] --> B["execute_partial_reverse()<br/>reverse stored approach/final states"]
    B --> C["sample gripper forces over short window"]
    C --> D["median force deltas"]
    D --> E{"held?"}
    E -->|"2+ fingers OR strong max OR enough sum"| F["HELD"]
    E -->|"below thresholds"| G["NOT_HELD warning"]
    F --> H{"slip_check_reacquire enabled?"}
    G --> H
    H -->|"yes and fruit still visible at grasp"| I["append retry goal<br/>reuse slip retry approach"]
    H -->|"no / no slip"| J["park fruit obstacle below floor"]
    J --> K["move_to_dropoff_position()"]
    K --> L{"dropoff failed?"}
    L -->|"yes"| M["HOME first, then dropoff"]
    L -->|"no"| N["open gripper"]
    M --> N
    N --> O["return HOME or prepare next queued goal"]
    O --> P["unlock target + reset tracking"]
```

## 11. Helper Index

| Helper | Purpose |
| --- | --- |
| `ThreadSafeGoalList` | Safe queue for goal callbacks and execute thread |
| `lock_target()` / `unlock_target()` | Tell vision to stay on the chosen fruit during motion |
| `image_lateral_side()` | Convert image/bunch position into `LEFT`, `CENTER`, `RIGHT` |
| `is_bunch_lower_boundary()` | Force lower-bunch fruit to center/very-low behavior |
| `low_side_standoff_offsets()` | Side-specific low approach offsets |
| `side_low_wrist3_orientation()` | Adjust wrist 3 through FK to keep side-low branch consistent |
| `minimize_rotation_orientation()` | Use closest equivalent target quaternion and blend from current |
| `_direct_ik_move()` | Close-range IK motion with branch protection |
| `plan_and_send()` | Full cuRobo planning with wraparound, voxel, path-ratio, and trim checks |
| `reacquire_goal_pose()` | Lightweight vision loop for final XYZ correction |
| `execute_partial_reverse()` | Back out along stored trajectory to clear the bunch |
| `notify_grasp_attempt()` | Let vision tracking count attempts for the fruit |

## 12. Short Mental Model

```text
subscribe_to_goal_pose()
    selects one stable fruit and stores it

plan_and_execute()
    decides the route and runs the whole harvest cycle

_direct_ik_move()
    handles short precise TCP moves without wrist branch jumps

plan_and_send()
    handles larger cuRobo-optimized moves and rejects bad detours

reacquire_goal_pose()
    refreshes fruit XYZ only when the arm is already staged/settled
```

