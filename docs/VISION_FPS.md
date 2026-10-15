# Vision Pipeline FPS

Live captures from the `vision` node (`--use_zedx_mini_only`, ZED X Mini RGBD /
harvest mode) on the Jetson AGX Orin. The HUD is drawn by
`draw_hud()` in [`vision/visualization.py`](../ur_ws_new/src/ur10e_curobo/ur10e_curobo/vision/visualization.py);
frames were pulled straight off the `/vision/display` topic.

- **YOLO FPS** — inference rate of the detector (`net_fps`).
- **Loop FPS** — the perception loop rate: grab + depth + publish.

## Perception loop, GPU free

![Vision node running with the GPU to itself — YOLO 22 fps, Loop 36 fps](images/vision_fps_uncontended.png)

`YOLO FPS: 22.1  ·  Loop FPS: 35.9`

The perception loop runs well above the old **15 fps** cap — this is the
`ZEDMINI_RGBD_FPS` 15 → 30 change in effect. With the arm/cuRobo stack down,
YOLO also has the whole GPU, so inference sits in the low 20s.

## Same loop, under GPU contention

![Vision node with the GPU under load — YOLO drops to 10 fps, Loop holds at 28 fps](images/vision_fps_under_gpu_load.png)

`YOLO FPS: 10.2  ·  Loop FPS: 28.5`

With another process contending for the GPU (a stand-in for cuRobo's
planning-time GPU use — **not** the physical arm), **YOLO inference roughly
halves (22 → 10)** while the **perception loop holds near 30 (36 → 28)**.

This is the key result: the loop-rate improvement survives contention. YOLO's
frame rate is the part that dips when the GPU is shared, which matches the
~12–13 fps seen during real harvesting when cuRobo is planning. The bottleneck
is GPU sharing, not the camera loop.

---

*Captured this session. YOLO figures are `net_fps` (inference only). The
contended shot uses a synthetic GPU load as a stand-in for cuRobo; no robot
motion was involved.*
