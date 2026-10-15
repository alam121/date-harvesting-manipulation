import pyzed.sl as sl
import cv2
from pathlib import Path
import sys


# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------

# Change this to your SVO/SVO2 file
SVO_PATH = "/home/datepalm2/manipulatorsdatepalm/ur_ws_new/src/HD1080_SN57931814_03-41-49.svo2"

# Folder where extracted images will be stored
OUTPUT_DIR = "svo_images"

# Number of images to save per second
TARGET_IMAGES_PER_SECOND = 3


def main():
    svo_path = Path(SVO_PATH).expanduser().resolve()
    output_dir = Path(OUTPUT_DIR).expanduser().resolve()

    # Check that the SVO exists
    if not svo_path.is_file():
        print(f"Error: SVO file not found:\n{svo_path}")
        sys.exit(1)

    if svo_path.suffix.lower() not in {".svo", ".svo2"}:
        print("Error: Input file must have .svo or .svo2 extension.")
        sys.exit(1)

    if TARGET_IMAGES_PER_SECOND <= 0:
        print("Error: TARGET_IMAGES_PER_SECOND must be greater than zero.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ZED X One is a monocular camera
    zed = sl.CameraOne()

    # Configure SVO playback
    init_params = sl.InitParametersOne()
    init_params.set_from_svo_file(str(svo_path))

    # Process the SVO as quickly as possible
    init_params.svo_real_time_mode = False

    print(f"Opening SVO:\n{svo_path}")

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open SVO: {status}")
        zed.close()
        sys.exit(1)

    # Read recording information
    camera_info = zed.get_camera_information()
    camera_config = camera_info.camera_configuration

    fps = float(camera_config.fps)
    resolution = camera_config.resolution
    total_frames = zed.get_svo_number_of_frames()

    if fps <= 0:
        print(f"Error: Invalid SVO frame rate: {fps}")
        zed.close()
        sys.exit(1)

    if TARGET_IMAGES_PER_SECOND > fps:
        print(
            f"Warning: Requested {TARGET_IMAGES_PER_SECOND} images/second, "
            f"but the SVO contains only {fps:g} frames/second."
        )
        save_every_n_frames = 1
    else:
        save_every_n_frames = max(
            1,
            round(fps / TARGET_IMAGES_PER_SECOND)
        )

    actual_images_per_second = fps / save_every_n_frames

    print(f"Resolution: {resolution.width} × {resolution.height}")
    print(f"Recorded FPS: {fps:g}")
    print(f"Total SVO frames: {total_frames}")
    print(f"Saving every {save_every_n_frames} frames")
    print(
        f"Output rate: approximately "
        f"{actual_images_per_second:.2f} images/second"
    )
    print(f"Output folder:\n{output_dir}")
    print("Press Ctrl+C to stop.\n")

    image = sl.Mat()

    saved_count = 0
    processed_count = 0

    try:
        while True:
            grab_status = zed.grab()

            if grab_status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                print("\nEnd of SVO reached.")
                break

            if grab_status != sl.ERROR_CODE.SUCCESS:
                print(f"\nFrame grab failed: {grab_status}")
                continue

            svo_position = zed.get_svo_position()
            processed_count += 1

            # Save only the selected frames
            if svo_position % save_every_n_frames != 0:
                continue

            retrieve_status = zed.retrieve_image(
                image,
                sl.VIEW.LEFT,
                sl.MEM.CPU
            )

            if retrieve_status != sl.ERROR_CODE.SUCCESS:
                print(
                    f"\nCould not retrieve frame {svo_position}: "
                    f"{retrieve_status}"
                )
                continue

            frame = image.get_data()

            if frame is None or frame.size == 0:
                print(f"\nFrame {svo_position} is empty.")
                continue

            # ZED normally returns BGRA; OpenCV PNG uses BGR
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame_to_save = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGRA2BGR
                )
            else:
                frame_to_save = frame

            timestamp_seconds = svo_position / fps

            filename = output_dir / (
                f"image_{saved_count:06d}_"
                f"frame_{svo_position:06d}_"
                f"time_{timestamp_seconds:010.3f}s.png"
            )

            success = cv2.imwrite(str(filename), frame_to_save)

            if not success:
                print(f"\nFailed to save image: {filename}")
                continue

            saved_count += 1

            print(
                f"\rSaved: {saved_count} images | "
                f"SVO frame: {svo_position}/{total_frames} | "
                f"Time: {timestamp_seconds:.2f} s",
                end="",
                flush=True
            )

    except KeyboardInterrupt:
        print("\n\nExtraction stopped by user.")

    finally:
        zed.close()

    print("\nExtraction completed.")
    print(f"Frames processed: {processed_count}")
    print(f"Images saved: {saved_count}")
    print(f"Images are stored in:\n{output_dir}")


if __name__ == "__main__":
    main()
