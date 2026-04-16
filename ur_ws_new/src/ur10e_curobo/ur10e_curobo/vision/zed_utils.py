"""ZED camera utility functions."""

import pyzed.sl as sl


def apply_zed_one_settings(zed):
    """Apply ZED X One Mono settings optimised for HDR date-fruit detection."""
    # HDR is enabled via InitParametersOne.enable_hdr = True before open() — no runtime call needed.
    print("[ZedOne] HDR active (set via InitParametersOne.enable_hdr)")

    #zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 7)   # 0-8
    #zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 5)     # 0-8
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 7)   # 0-8  (richer colour with HDR)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 6)    # 0-8
    # zed.set_camera_settings(sl.VIDEO_SETTINGS.GAMMA, 7)        # 1-9

    # Lock manual exposure so HDR tone-mapping doesn't converge dark
    try:
        #zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)
        zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 59)  # 0-100
        #zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 70)      # 0-100
    except Exception:
        pass

    try:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.DENOISING, 100)
    except Exception:
        pass

    # Verify settings
    try:
        print("[ZedOne] sat:", zed.get_camera_settings(sl.VIDEO_SETTINGS.SATURATION))
        print("[ZedOne] sharp:", zed.get_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS))
        print("[ZedOne] gamma:", zed.get_camera_settings(sl.VIDEO_SETTINGS.GAMMA))
        print("[ZedOne] wb_auto:", zed.get_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO))
        print("[ZedOne] wb_temp:", zed.get_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE))
        print("[ZedOne] aec_agc:", zed.get_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC))
        print("[ZedOne] gain:", zed.get_camera_settings(sl.VIDEO_SETTINGS.GAIN))
        print("[ZedOne] exp:", zed.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE))
        print("[ZedOne] denoise:", zed.get_camera_settings(sl.VIDEO_SETTINGS.DENOISING))
        print("[ZedOne] hdr:", zed.get_camera_settings(sl.VIDEO_SETTINGS.HDR))
        print("[ZedOne] brightness:", zed.get_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS))
        print("[ZedOne] contrast:", zed.get_camera_settings(sl.VIDEO_SETTINGS.CONTRAST))
    except Exception:
        pass


# Backward-compat alias (existing callers that use the old name still work)
apply_zed_camera_settings = apply_zed_one_settings


def apply_zed_mini_settings(zed):
    """Apply ZED X Mini settings for depth-only use (no HDR)."""
    try:
        #zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 4)
        zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 4)
    except Exception:
        pass
