"""ZED camera utility functions."""

import pyzed.sl as sl


def apply_zed_camera_settings(zed):
    """Apply custom camera settings for date fruit detection."""
    try:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.ENABLE_HDR, 1)
        print("[CAM] HDR enabled")
    except Exception:
        pass

    #zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 7)   # 0-8
    #zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 5)     # 0-8
    #zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 7)
   # zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 6)
    #zed.set_camera_settings(sl.VIDEO_SETTINGS.GAMMA, 7)         # 1-9

    # Disable AEC/AGC — lock manual exposure so HDR doesn't converge dark
    try:
        #zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)
        zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 19)  # 0-100
        #zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 70)      # 0-100
    except Exception:
        pass

    try:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.DENOISING, 100)
    except Exception:
        pass

    # Verify settings
    try:
        print("[CAM] sat:", zed.get_camera_settings(sl.VIDEO_SETTINGS.SATURATION))
        print("[CAM] sharp:", zed.get_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS))
        print("[CAM] gamma:", zed.get_camera_settings(sl.VIDEO_SETTINGS.GAMMA))
        print("[CAM] wb_auto:", zed.get_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO))
        print("[CAM] wb_temp:", zed.get_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE))
        print("[CAM] aec_agc:", zed.get_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC))
        print("[CAM] gain:", zed.get_camera_settings(sl.VIDEO_SETTINGS.GAIN))
        print("[CAM] exp:", zed.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE))
    except Exception:
        pass
