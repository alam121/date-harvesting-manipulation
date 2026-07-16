import pyzed.sl as sl
import cv2
from pathlib import Path
import sys


# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------

# Change this to your ZED X Mini SVO/SVO2 file
SVO_PATH = (
    "/home/datepalm2/manipulatorsdatepalm/ur_ws_new/src/"
    "HD1080_SN57931814_03-41-49.svo2"
)

# Main output folder
OUTPUT_DIR = "zed_x_mini_svo_images"

# Number of image pairs to save per second
TARGET_IMAGES_PER_SECOND = 3

# Save the right camera image as well
SAVE_RIGHT_IMAGE = True


def convert_zed_image_to_bgr(frame):
    """
    Convert a ZED image returned as BGRA into an OpenCV-compatible BGR image.
    """
    if frame is None or frame.size == 0:
        return None

    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    return frame


def main():
    svo_path = Path(SVO_PATH).expanduser().resolve()
    output_dir = Path(OUTPUT_DIR).expanduser().resolve()

    # -------------------------------------------------------
    # VALIDATE INPUT
    # -------------------------------------------------------

    if not svo_path.is_file():
        print(f"Error: SVO file not found:\n{svo_path}")
        sys.exit(1)

    if svo_path.suffix.lower() not in {".svo", ".svo2"}:
        print("Error: Input file must have a .svo or .svo2 extension.")
        sys.exit(1)

    if TARGET_IMAGES_PER_SECOND <= 0:
        print("Error: TARGET_IMAGES_PER_SECOND must be greater than zero.")
        sys.exit(1)

    # Separate folders keep stereo images organized
    left_output_dir = output_dir / "left"
    left_output_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_RIGHT_IMAGE:
        right_output_dir = output_dir / "right"
        right_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        right_output_dir = None

    # -------------------------------------------------------
    # INITIALIZE ZED X MINI
    # -------------------------------------------------------

    # ZED X Mini is a stereo camera
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_path))

    # Read the SVO without reproducing the original real-time delays
    init_params.svo_real_time_mode = False

    # Depth is not needed when extracting only RGB images
    init_params.depth_mode = sl.DEPTH_MODE.NONE

    print(f"Opening ZED X Mini SVO:\n{svo_path}")

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open SVO: {status}")
        zed.close()
        sys.exit(1)

    # -------------------------------------------------------
    # READ SVO INFORMATION
    # -------------------------------------------------------

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

    print(f"Camera model: {camera_info.camera_model}")
    print(f"Resolution: {resolution.width} × {resolution.height}")
    print(f"Recorded FPS: {fps:g}")
    print(f"Total SVO frames: {total_frames}")
    print(f"Saving every {save_every_n_frames} frames")
    print(
        f"Output rate: approximately "
        f"{actual_images_per_second:.2f} image pairs/second"
    )
    print(f"Left images:\n{left_output_dir}")

    if SAVE_RIGHT_IMAGE:
        print(f"Right images:\n{right_output_dir}")

    print("Press Ctrl+C to stop.\n")

    # -------------------------------------------------------
    # EXTRACTION
    # -------------------------------------------------------

    runtime_params = sl.RuntimeParameters()

    left_image = sl.Mat()
    right_image = sl.Mat()

    saved_count = 0
    processed_count = 0

    try:
        while True:
            grab_status = zed.grab(runtime_params)

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

            # ---------------------------------------------------
            # RETRIEVE LEFT IMAGE
            # ---------------------------------------------------

            left_status = zed.retrieve_image(
                left_image,
                sl.VIEW.LEFT,
                sl.MEM.CPU
            )

            if left_status != sl.ERROR_CODE.SUCCESS:
                print(
                    f"\nCould not retrieve left frame "
                    f"{svo_position}: {left_status}"
                )
                continue

            left_frame = convert_zed_image_to_bgr(
                left_image.get_data()
            )

            if left_frame is None:
                print(f"\nLeft frame {svo_position} is empty.")
                continue

            # ---------------------------------------------------
            # RETRIEVE RIGHT IMAGE
            # ---------------------------------------------------

            right_frame = None

            if SAVE_RIGHT_IMAGE:
                right_status = zed.retrieve_image(
                    right_image,
                    sl.VIEW.RIGHT,
                    sl.MEM.CPU
                )

                if right_status != sl.ERROR_CODE.SUCCESS:
                    print(
                        f"\nCould not retrieve right frame "
                        f"{svo_position}: {right_status}"
                    )
                    continue

                right_frame = convert_zed_image_to_bgr(
                    right_image.get_data()
                )

                if right_frame is None:
                    print(f"\nRight frame {svo_position} is empty.")
                    continue

            timestamp_seconds = svo_position / fps

            base_filename = (
                f"image_{saved_count:06d}_"
                f"frame_{svo_position:06d}_"
                f"time_{timestamp_seconds:010.3f}s.png"
            )

            left_filename = left_output_dir / base_filename

            left_saved = cv2.imwrite(
                str(left_filename),
                left_frame
            )

            if not left_saved:
                print(f"\nFailed to save left image: {left_filename}")
                continue

            if SAVE_RIGHT_IMAGE:
                right_filename = right_output_dir / base_filename

                right_saved = cv2.imwrite(
                    str(right_filename),
                    right_frame
                )

                if not right_saved:
                    print(
                        f"\nFailed to save right image: "
                        f"{right_filename}"
                    )
                    continue

            saved_count += 1

            print(
                f"\rSaved: {saved_count} stereo pairs | "
                f"SVO frame: {svo_position}/{total_frames} | "
                f"Time: {timestamp_seconds:.2f} s",
                end="",
                flush=True
            )

    except KeyboardInterrupt:
        print("\n\nExtraction stopped by user.")

    finally:
        zed.close()

    # -------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------

    print("\nExtraction completed.")
    print(f"Frames processed: {processed_count}")

    if SAVE_RIGHT_IMAGE:
        print(f"Stereo image pairs saved: {saved_count}")
        print(f"Total PNG files saved: {saved_count * 2}")
    else:
        print(f"Left images saved: {saved_count}")

    print(f"Images are stored in:\n{output_dir}")


if __name__ == "__main__":
    main()
