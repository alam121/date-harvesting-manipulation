"""Standalone file playback for Raw YOLO inspection."""

from pathlib import Path
from threading import Thread
from time import monotonic, sleep, time

import cv2

from .visualization import VisionVisualizer
from .yolo_thread import YoloThread


def run_raw_video(args) -> None:
    """Run the normal YOLO worker on an OpenCV video without ROS or ZED."""
    path = Path(args.video).expanduser().resolve()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0.0 or source_fps > 240.0:
        source_fps = 30.0
    frame_period = 1.0 / source_fps
    worker = YoloThread(
        weights=args.weights,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        max_det=args.max_det,
        raw_view=True,
        use_numpy_masks=False,
    )
    thread = Thread(target=worker.run, name="raw_video_yolo", daemon=True)
    thread.start()
    visualizer = VisionVisualizer({}, [1.0, 1.0], display_scale=0.6)
    window = f"Raw YOLO — {path.name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    last_display = None
    loop_times = []

    print(
        f"[RawVideo] Playing {path} at {source_fps:.1f} FPS; "
        "press Q or ESC to stop, SPACE to pause.")
    paused = False
    try:
        while True:
            loop_start = monotonic()
            if not paused:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                worker.set_image(frame_bgra, capture_time=time())

            if worker.dets_ready.is_set():
                worker.dets_ready.clear()
                result = worker.get_raw_result()
                if result is not None:
                    now = monotonic()
                    loop_times.append(now)
                    loop_times = loop_times[-30:]
                    loop_fps = 0.0
                    if len(loop_times) > 1 and loop_times[-1] > loop_times[0]:
                        loop_fps = (len(loop_times) - 1) / (loop_times[-1] - loop_times[0])
                    last_display = visualizer.render_frame(
                        result["image"], [], [], None,
                        worker.net_fps, loop_fps,
                        viz_only=result["detections"],
                    )
            if last_display is not None:
                cv2.imshow(window, last_display)

            elapsed = monotonic() - loop_start
            key = cv2.waitKey(max(1, int(max(0.0, frame_period - elapsed) * 1000.0))) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" "):
                paused = not paused
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if paused:
                sleep(0.01)
    finally:
        capture.release()
        worker.stop()
        worker.stopped.wait(timeout=5.0)
        cv2.destroyAllWindows()
