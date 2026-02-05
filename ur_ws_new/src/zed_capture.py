import os
import time
import cv2
import numpy as np
import pyzed.sl as sl


def main():
    # ------------------------------------------------------------
    # Initialize ZED camera
    # ------------------------------------------------------------
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED camera: {status}")
        return

    # ------------------------------------------------------------
    # Camera image settings
    # ------------------------------------------------------------
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 7)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 6)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAMMA, 2)

    # ------------------------------------------------------------
    # Prepare image container and output directory
    # ------------------------------------------------------------
    image = sl.Mat()

    output_dir = "zed_images"
    os.makedirs(output_dir, exist_ok=True)

    print("Starting image capture every 3 seconds (Ctrl+C to stop).")

    image_count = 0

    try:
        while True:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                img = image.get_data()

                filename = os.path.join(
                    output_dir, f"image_{image_count:03d}.png"
                )
                cv2.imwrite(filename, img)

                print(f"Saved: {filename}")
                image_count += 1

            time.sleep(3)

    except KeyboardInterrupt:
        print("\nImage capture stopped by user.")

    finally:
        zed.close()
        print("ZED camera closed.")


if __name__ == "__main__":
    main()

