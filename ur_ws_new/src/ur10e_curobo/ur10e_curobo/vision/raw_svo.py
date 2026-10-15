"""Safe SVO-to-video preparation for standalone Raw YOLO playback."""

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path


def _cache_path(source: Path) -> Path:
    signature = f"{source.resolve()}:{source.stat().st_size}:{source.stat().st_mtime_ns}"
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"datepalm_raw_svo_{digest}.mp4"


def prepare_svo_video(source_value: str) -> Path:
    """Export an SVO in a child process, then return its cached RGB video."""
    source = Path(source_value).expanduser().resolve()
    if not source.is_file():
        raise RuntimeError(f"SVO file does not exist: {source}")
    output = _cache_path(source)
    if output.is_file() and output.stat().st_size > 0:
        print(f"[RawSVO] Using cached RGB playback: {output}")
        return output

    print(
        "[RawSVO] Preparing RGB playback in an isolated ZED process. "
        "This is done once per recording; later launches use the cache.")
    partial = output.with_name(f"{output.stem}.partial{output.suffix}")
    subprocess.run(
        [sys.executable, "-m", __name__, str(source), str(partial)],
        check=True,
    )
    if not partial.is_file() or partial.stat().st_size <= 0:
        raise RuntimeError(f"SVO export did not produce a video: {partial}")
    partial.replace(output)
    return output


def _export(source: Path, output: Path) -> None:
    import cv2
    import pyzed.sl as sl

    input_type = sl.InputType()
    input_type.set_from_svo_file(str(source))
    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.sdk_verbose = 0
    camera = sl.Camera()
    status = camera.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open SVO {source}: {status}")

    writer = None
    image = sl.Mat()
    frames = 0
    try:
        info = camera.get_camera_information().camera_configuration
        width = int(info.resolution.width)
        height = int(info.resolution.height)
        fps = float(info.fps) if float(info.fps) > 0 else 30.0
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create cached playback video: {output}")

        while True:
            status = camera.grab()
            if status != sl.ERROR_CODE.SUCCESS:
                break
            camera.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU)
            bgra = image.get_data()
            writer.write(cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR))
            frames += 1
            if frames % 300 == 0:
                print(f"[RawSVO] Exported {frames} frames…", flush=True)
    finally:
        if writer is not None:
            writer.release()
        camera.close()
    if frames == 0:
        raise RuntimeError(f"SVO contains no readable frames: {source}")
    print(f"[RawSVO] Export complete: {frames} frames -> {output}")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m ...raw_svo INPUT.svo2 OUTPUT.mp4")
    _export(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
