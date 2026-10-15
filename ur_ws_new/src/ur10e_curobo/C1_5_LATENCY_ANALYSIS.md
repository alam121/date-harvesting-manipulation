# C1.5 — Latency Analysis

**Sub-deliverable:** Report measuring the time between sensor detection and autonomous decision execution in the UR10e date-palm harvesting prototype.

**Completion:** 25%  
**Target:** December 2026  
**Invoicing Eligible:** [YES/NO]

---

## 1. Basic Information

This sub-deliverable reports the latency of the perception–decision–execution pipeline in the robotic prototype. The objective is to measure the time required for the system to acquire sensor data, run AI model inference, process detection outputs, estimate target poses, and pass the resulting decision to the robot-control or motion-planning module.

At the current stage, latency is being evaluated primarily through the system's **loop FPS**, which measures the throughput of the full perception and processing pipeline. This differs from **AI inference FPS**, which measures only the neural-network inference rate. Loop FPS includes image acquisition, preprocessing, AI detection, depth fusion, 3D localization, coordinate transformation, visualization, and communication with downstream ROS 2 modules.

The latency analysis is currently under experimental evaluation. More detailed timing breakdowns will be provided in the next reporting period as additional logging and benchmarking tools are integrated into the robotic prototype.

---

## 2. Methodology

The latency measurement begins with monitoring the full sensing and AI perception pipeline during real-time operation. Sensor data are acquired from the ZED camera and processed by the AI detection model. The model output is then combined with depth information to estimate 3D target positions, which are transformed into the robot base frame and published to the ROS 2 control pipeline.

Two primary timing metrics are reported, displayed on the live HUD overlay during operation:

| Metric | Description |
|---|---|
| **AI inference FPS** (`YOLO FPS`) | Frame rate of the AI model inference step only. Measured as `1 / dt` where `dt` is the wall-clock time of one `model.predict()` forward pass on GPU. |
| **Loop FPS** | Full processing-loop throughput, measured as `1 / (t_now − t_prev)` at each ZED camera grab. Includes sensor acquisition, preprocessing, inference, depth fusion, localization, coordinate transformation, visualization, and ROS 2 communication. |

The current evaluation uses loop FPS as the primary indicator of end-to-end perception latency. The system additionally logs per-stage timing breakdowns to the terminal every 30 frames (`[PERF]`) and every 20 visualization frames (`[VIZ_PERF]`), allowing finer-grained analysis.

Future measurements will include detailed breakdowns of latency across sensor acquisition, AI inference, target localization, planning trigger, motion-planning computation, and command publication.

---

## 3. Results / Outcomes

### 3.1 Perception Pipeline Latency Breakdown

The perception pipeline executes as a continuous loop, with the loop rate bounded by the camera hardware frame rate. The following table describes each instrumented stage and its known timing constraints, derived from the software implementation.

**ZED Camera Configuration (hardware frame-rate limits):**

| Camera Mode | Resolution | Frame Rate | Per-Frame Budget |
|---|---|---|---|
| ZED X One Mono (HDR enabled) | QHDPLUS | ~6 FPS (hardware cap) | ~167 ms |
| ZED X One Mono (HDR disabled) | QHDPLUS | 15 FPS (configured) | ~67 ms |
| ZED X Mini (depth camera) | HD1080 | 15 FPS (configured) | ~67 ms |
| ZED stereo | HD1080 | 30 FPS (SDK default) | ~33 ms |

The main perception loop (`loop_fps`) cannot exceed the camera hardware rate. In the current dual-camera configuration (ZED X One + ZED X Mini), the loop is bounded at **15 FPS (~67 ms/frame)** in standard operation, or **~6 FPS (~167 ms/frame)** when HDR is active at QHDPLUS resolution.

**Per-Stage Timing (instrumented in software, logged every 30 frames):**

| Stage | Variable | Description | Known Constraint |
|---|---|---|---|
| **Camera grab** | implicit in `loop_fps` | `zed.grab()` blocks until a new frame is ready — sets the fundamental loop cadence | Bounded by camera FPS |
| **Image retrieval** | `_t1 − _t0` | `zed.retrieve_image()` — transfers frame buffer to CPU at display resolution | Sub-frame (~5–20 ms) |
| **TF cache refresh** | included in `_t1 → _t2` | Two `tf_buffer.lookup_transform()` calls per frame (camera → base_link and camera → gripper_tip). Previously 8+ separate calls at ~20 ms each on Jetson; per-frame caching reduced this cost by ~120–160 ms/frame. | <1 ms (cached) |
| **3D localization** | `_t3 − _t2` | `_process_objects()`: depth fusion, 3D centroid estimation, heatmap (every 10th frame), coordinate transformation, approach-direction geometry | Dominant per-frame cost |
| **Temporal stabilization** | `_tp1 − _t3` | `stabilize_detections()`: deque-based fruit position smoothing and tracking | Lightweight |
| **Trunk + depth publish** | `_tp2 − _tp1` | Trunk position and voxel depth cloud published to ROS 2 topics (every 5th frame for trunk; burst-mode for depth cloud) | Lightweight |
| **Best-fruit selection** | `_tp3 − _tp2` | `_select_best_fruit()`: scoring-based selection across tracked targets | Lightweight |
| **Approach direction** | `_tp4 − _tp3` | `_process_best_target()`: gripper approach direction computation and goal-pose update | Lightweight |

**YOLO Inference (background thread, decoupled from main loop):**

YOLO inference runs in a dedicated background thread and does not block the main perception loop. The main loop submits a frame and continues; the inference result is polled on the next iteration. If no new result is ready, the loop reuses the last valid detection. This decoupling means:

- Loop FPS reflects camera throughput, not YOLO throughput.
- YOLO FPS reflects GPU inference rate independently.
- If YOLO is slower than the camera, detections are reused across multiple frames.
- If the camera is slower than YOLO, every frame receives a fresh detection.

During CuRobo motion planning, an `inference_lock` is acquired before planning begins, pausing YOLO inference on the GPU until planning is complete. The vision node also enters `paused` mode during robot execution, eliminating GPU contention.

### 3.2 ROS 2 Communication Latency

| Topic | Publisher Rate | Max Communication Latency |
|---|---|---|
| `/external_goal_pose` | 50 Hz timer (20 ms period) | ≤20 ms from goal-pose update to next publish |
| `/datefruit_direction` | 50 Hz timer (20 ms period) | ≤20 ms from direction update to next publish |
| `/vision/display` | ~15 FPS (throttled; 66 ms min interval) | Visualization only; non-blocking background thread |
| `/vision/all_fruit_poses` | Per-frame (when targets visible) | Synchronized with perception loop |

The 50 Hz goal publisher ensures that the motion-planning node receives a fresh goal pose within at most 20 ms of any target-position update from the perception loop.

### 3.3 Motion Planning Latency

CuRobo motion planning is performed on the same CUDA device as YOLO inference. Planning is initiated after a stable goal pose is accepted. Key timing parameters:

| Parameter | Value | Description |
|---|---|---|
| `interpolation_dt` | 4 ms | CuRobo internal trajectory time step |
| `min_dt` | 10 ms | Minimum trajectory step (below this causes UR controller joint velocity faults) |
| `max_dt` | 50 ms | Maximum trajectory step |
| `max_attempts` | 20 | Maximum cuRobo solver iterations per plan request |
| `global_speed_multiplier` | 5.0× | Overall trajectory speed scaling factor |

CuRobo planning time on the Jetson platform is not yet systematically logged. This is a planned addition for the next reporting period. Planning time is expected to range from approximately 100 ms to 500 ms depending on trajectory complexity, obstacle density, and GPU load.

### 3.4 End-to-End Latency Estimate

Based on the architecture and configured parameters, the estimated latency from sensor acquisition to motion command publication is:

| Sub-pipeline | Estimated Latency |
|---|---|
| Camera frame capture (15 FPS mode) | ~67 ms/frame |
| Per-frame processing (retrieve + localize + select) | 20–80 ms (hardware dependent) |
| ROS 2 goal publish delay | ≤20 ms |
| CuRobo motion planning | ~100–500 ms (to be measured) |
| Trajectory controller execution onset | <10 ms (joint controller internal) |
| **Total: detection → motion command** | **~200–600 ms** |

These figures represent estimates based on hardware configuration and architectural constraints. Precise per-stage measurements from live system logs will be reported in the next period.

---

## 4. Evidence / Backup Data

### 4.1 Loop FPS and YOLO FPS — HUD Display

Loop FPS and YOLO FPS are computed in software and rendered in real time on the HUD overlay of the vision feed. The relevant implementation is in:

- **Loop FPS computation:** [`vision/node.py:595–597`](ur10e_curobo/vision/node.py#L595-L597)
  ```python
  t_now = time()
  loop_fps = 1.0 / (t_now - t_prev) if (t_now - t_prev) > 0 else 0.0
  t_prev = t_now
  ```

- **YOLO inference FPS computation:** [`vision/yolo_thread.py:92–93`](ur10e_curobo/vision/yolo_thread.py#L92-L93)
  ```python
  dt = time() - t0
  self.net_fps = (1.0 / dt) if dt > 0 else 0.0
  ```

- **HUD rendering:** [`vision/visualization.py:470–491`](ur10e_curobo/vision/visualization.py#L470-L491) — `draw_hud()` overlays both values on every display frame.

### 4.2 Per-Stage Timing Log (`[PERF]`)

Every 30 frames, the perception loop prints a structured timing breakdown to the terminal:

```
[PERF] retrieve=Xms  process=Xms  post=Xms
       (stab=Xms trunk=Xms sel=Xms best=Xms)  total=Xms
```

Source: [`vision/node.py:853–859`](ur10e_curobo/vision/node.py#L853-L859)

| Field | Stage | Timestamp variables |
|---|---|---|
| `retrieve` | ZED image transfer | `_t1 − _t0` |
| `process` | `_process_objects()` (3D localization) | `_t3 − _t2` |
| `stab` | Temporal stabilization | `_tp1 − _t3` |
| `trunk` | Trunk + depth cloud publish | `_tp2 − _tp1` |
| `sel` | Best-fruit selection | `_tp3 − _tp2` |
| `best` | Approach direction + goal update | `_tp4 − _tp3` |
| `total` | Full per-frame processing overhead | `_t4 − _t0` |

### 4.3 Visualization Pipeline Timing Log (`[VIZ_PERF]`)

Every 20 rendered frames, the background visualization thread prints:

```
[VIZ_PERF] render=Xms  cvt=Xms  encode=Xms  publish=Xms  total=Xms  img=WxH
```

Source: [`vision/node.py:539–544`](ur10e_curobo/vision/node.py#L539-L544)

### 4.4 Camera Configuration References

- ZED X One Mono frame rate: [`vision/node.py:911`](ur10e_curobo/vision/node.py#L911) (`camera_fps = 15`)
- HDR QHDPLUS hardware cap: [`vision/node.py:914`](ur10e_curobo/vision/node.py#L914) (comment: "HDR at QHDPLUS caps hardware to ~6fps")
- ZED X Mini depth frame rate: [`vision/config.py:135`](ur10e_curobo/vision/config.py#L135) (`ZEDMINI_DEPTH_FPS = 15`)

### 4.5 ROS 2 Goal Publisher Rate

- 50 Hz goal pose timer: [`vision/node.py:196`](ur10e_curobo/vision/node.py#L196) (`self.node.create_timer(0.02, publish_timer_cb)`)
- Visualization throttle: [`vision/node.py:560`](ur10e_curobo/vision/node.py#L560) (`_VIZ_MIN_INTERVAL = 0.066` → ~15 FPS)

### 4.6 Motion Planning Configuration

- CuRobo trajectory parameters: [`config.py:141–165`](ur10e_curobo/config.py#L141-L165) (`interpolation_dt`, `min_dt`, `max_dt`, `global_speed_multiplier`, speed factors per motion type)
- Planning lock (prevents concurrent YOLO + cuRobo GPU ops): [`motions.py:56`](ur10e_curobo/motions.py#L56)

### 4.7 Next Steps for Quantitative Data

The following measurements are planned for the next reporting period:

1. **Runtime log collection:** Capture `[PERF]` and `[VIZ_PERF]` terminal output during field operation; tabulate per-stage means and distributions.
2. **YOLO inference time:** Record `net_fps` time-series; derive mean inference latency per frame by class-set size.
3. **CuRobo planning duration:** Add wall-clock timing around `plan_single()` / `plan_single_js()` calls; log per-plan duration and success rate.
4. **End-to-end trace:** Instrument with ROS 2 message header timestamps to measure perception-to-execution delay across the full topic chain: `/vision/all_fruit_poses` → `/external_goal_pose` → `/joint_trajectory_controller/joint_trajectory`.
