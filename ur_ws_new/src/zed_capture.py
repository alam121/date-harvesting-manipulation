import pyzed.sl as sl
import numpy as np
import cv2
import time
import os

def main():
    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # You can change resolution if needed
    init_params.camera_fps = 30

    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Failed to open ZED camera: {status}")
        exit(1)

    # Set video settings after opening the camera
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 7)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 6)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAMMA, 2)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.DENOISING, 100)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE_COMPENSATION, 58)

    # Prepare image container
    image = sl.Mat()

    # Create output directory
    output_dir = "zed_images"
    os.makedirs(output_dir, exist_ok=True)

    print("Starting image capture every 3 seconds. Press Ctrl+C to stop.")
    count = 0
    try:
        while True:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                img = image.get_data()

                # Save image
                filename = os.path.join(output_dir, f"image_{count:03d}.png")
                cv2.imwrite(filename, img)
                print(f"Saved: {filename}")
                count += 1

            time.sleep(3)
    except KeyboardInterrupt:
        print("\nImage capture stopped by user.")
    finally:
        zed.close()

if __name__ == "__main__":
    main()

