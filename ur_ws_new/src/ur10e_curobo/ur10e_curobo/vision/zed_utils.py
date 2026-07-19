"""ZED camera utility functions."""

import pyzed.sl as sl


ZED_ONE_LAB_PRESET = {
    "aec_agc": 1,
    "exposure": 0,
    "gain": None,
    "description": "lab auto exposure",
}

ZED_ONE_OUTDOOR_PRESET = {
    "aec_agc": 0,
    "exposure": 8,
    "gain": 0,
    "description": "outdoor manual low exposure",
}


def _set_if_available(zed, setting, value):
    if value is None:
        return
    try:
        zed.set_camera_settings(setting, value)
    except Exception:
        pass


def _print_zed_one_settings(zed, label="[ZedOne]"):
    try:
        print(f"{label} sat:", zed.get_camera_settings(sl.VIDEO_SETTINGS.SATURATION))
        print(f"{label} sharp:", zed.get_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS))
        print(f"{label} gamma:", zed.get_camera_settings(sl.VIDEO_SETTINGS.GAMMA))
        print(f"{label} wb_auto:", zed.get_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO))
        print(f"{label} wb_temp:", zed.get_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE))
        print(f"{label} aec_agc:", zed.get_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC))
        print(f"{label} gain:", zed.get_camera_settings(sl.VIDEO_SETTINGS.GAIN))
        print(f"{label} exp:", zed.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE))
        print(f"{label} denoise:", zed.get_camera_settings(sl.VIDEO_SETTINGS.DENOISING))
        print(f"{label} hdr:", zed.get_camera_settings(sl.VIDEO_SETTINGS.HDR))
        print(f"{label} brightness:", zed.get_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS))
        print(f"{label} contrast:", zed.get_camera_settings(sl.VIDEO_SETTINGS.CONTRAST))
    except Exception:
        pass


def apply_zed_one_exposure_preset(zed, preset_name="lab", verify=True):
    """Apply a runtime exposure preset to the already-open ZED X One camera."""
    preset_key = str(preset_name or "lab").strip().lower()
    if preset_key == "outdoor":
        preset = ZED_ONE_OUTDOOR_PRESET
    else:
        preset_key = "lab"
        preset = ZED_ONE_LAB_PRESET

    try:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, preset["aec_agc"])
        zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, preset["exposure"])
        _set_if_available(zed, sl.VIDEO_SETTINGS.GAIN, preset["gain"])
    except Exception:
        pass

    print(
        f"[ZedOne] exposure preset: {preset_key} "
        f"({preset['description']}, exposure={preset['exposure']}, gain={preset['gain']})")
    if verify:
        _print_zed_one_settings(zed)
    return preset_key


def apply_zed_one_manual_exposure(zed, auto_exposure=False, exposure=8, gain=0, verify=True):
    """Apply live exposure controls from the RViz panel."""
    auto_value = 1 if auto_exposure else 0
    exposure_value = int(max(0, min(100, exposure)))
    gain_value = int(max(0, min(100, gain)))

    try:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, auto_value)
        if not auto_exposure:
            zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, exposure_value)
            _set_if_available(zed, sl.VIDEO_SETTINGS.GAIN, gain_value)
    except Exception:
        pass

    print(
        f"[ZedOne] manual exposure control: auto={auto_value} "
        f"exposure={exposure_value} gain={gain_value}")
    if verify:
        _print_zed_one_settings(zed)


def apply_zed_one_hdr(zed, enabled=True, verify=True):
    """Apply runtime HDR when supported by the active ZED camera."""
    enabled_value = 1 if enabled else 0
    ok = False
    try:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.HDR, enabled_value)
        ok = True
    except Exception:
        pass

    print(f"[ZedOne] HDR requested: {enabled_value} runtime_ok={int(ok)}")
    if verify:
        _print_zed_one_settings(zed)
    return ok


def apply_zed_one_settings(zed, preset_name="lab"):
    """Apply ZED X One Mono settings optimised for HDR date-fruit detection."""
    # HDR is controlled via InitParametersOne.enable_hdr before open, and may
    # also be toggled at runtime on SDK/camera combinations that support it.
    print("[ZedOne] applying colour/exposure settings")

    #zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 7)   # 0-8
    zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 5)     # 0-8
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 7)   # 0-8  (richer colour with HDR)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 4)    # 0-8
    # zed.set_camera_settings(sl.VIDEO_SETTINGS.GAMMA, 7)        # 1-9

    apply_zed_one_exposure_preset(zed, preset_name, verify=False)

    try:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.DENOISING, 100)
    except Exception:
        pass

    _print_zed_one_settings(zed)


# Backward-compat alias (existing callers that use the old name still work)
apply_zed_camera_settings = apply_zed_one_settings


def apply_zed_stereo_settings(zed):
    """Apply ZED X Mini settings when used as the main stereo camera (no ZED X One).
    Keeps auto-exposure active but applies EV compensation to prevent overexposure outdoors."""
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 7)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 6)
    zed.set_camera_settings(sl.VIDEO_SETTINGS.GAMMA, 2)

    try:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.DENOISING, 100)
    except Exception:
        pass

    try:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE_COMPENSATION, 58)
    except Exception:
        pass

    # Verify settings
    try:
        print("[ZedStereo] sat:", zed.get_camera_settings(sl.VIDEO_SETTINGS.SATURATION))
        print("[ZedStereo] sharp:", zed.get_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS))
        print("[ZedStereo] gamma:", zed.get_camera_settings(sl.VIDEO_SETTINGS.GAMMA))
        print("[ZedStereo] wb_auto:", zed.get_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO))
        print("[ZedStereo] wb_temp:", zed.get_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE))
        print("[ZedStereo] aec_agc:", zed.get_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC))
        print("[ZedStereo] gain:", zed.get_camera_settings(sl.VIDEO_SETTINGS.GAIN))
        print("[ZedStereo] exp:", zed.get_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE))
    except Exception:
        pass


def apply_zed_mini_settings(zed):
    """Apply ZED X Mini settings for depth-only use (no HDR)."""
    try:
        zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 7)
        zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 6)
        zed.set_camera_settings(sl.VIDEO_SETTINGS.GAMMA, 2)

    except Exception:
        pass
