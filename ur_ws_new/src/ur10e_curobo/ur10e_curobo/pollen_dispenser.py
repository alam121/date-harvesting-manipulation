"""Pollen dispenser: drive the blower over USB serial from the robot.

Hardware is an Arduino Nano holding two SG90 servos against the physical
buttons of a Wolfbox MF200 blower (firmware in hardware/pollen_dispenser/).
The Arduino owns the whole actuation cycle, so the host sends exactly one
line -- "spray <seconds>" -- and waits for "OK" or "ERR".

Open-loop by construction: mass is inferred from spray duration, never
measured. The characterisation behind SLOPE/INTERCEPT reported R^2 ~ 0.80,
with 4g targets under-delivering by roughly 30% and flow changing as the jar
empties. Treat a requested mass as a setpoint, not a guarantee.

Nothing here raises. A missing pyserial, an unplugged Arduino or a denied port
returns (False, reason) so a pollination run degrades to "visited the flower
but did not spray" instead of aborting mid-arc with the arm at the plant.
"""

import time

# Fitted on the rig: grams = SLOPE * seconds + INTERCEPT.
SLOPE = 1.7611
INTERCEPT = -1.5001

# MAX_MS in spray_controller.ino. Past this the firmware answers "ERR bad
# duration", so reject it here instead of waiting out a timeout for a refusal
# we can predict. (spray.py's own 60s ceiling is above the firmware's and lets
# that round trip happen.)
FIRMWARE_MAX_S = 30.0

# Fixed cost of one cycle, summed from the firmware constants:
#   power on   A_HOLD 2500 + MOVE 400
#   settle     SETTLE 1500
#   blow                     MOVE 400   (+ the requested duration)
#   power off  A_HOLD 2500 + MOVE 400
FIRMWARE_OVERHEAD_S = 7.7


def grams_to_seconds(grams):
    """Invert the fit. Raises ValueError below the model's usable floor."""
    grams = float(grams)
    if grams <= 0.0:
        raise ValueError("grams must be > 0")
    seconds = (grams - INTERCEPT) / SLOPE
    if seconds > FIRMWARE_MAX_S:
        raise ValueError(
            f"{grams:.2f}g needs {seconds:.2f}s, over the firmware's "
            f"{FIRMWARE_MAX_S:.0f}s limit (max ~{seconds_to_grams(FIRMWARE_MAX_S):.1f}g)")
    return round(seconds, 2)


def seconds_to_grams(seconds):
    return SLOPE * float(seconds) + INTERCEPT


def dispense(logger, *, grams=None, seconds=None,
             port="/dev/ttyUSB0", baud=9600, timeout_s=None):
    """Run one spray cycle. Returns (ok, message); never raises.

    Give either `grams` (converted through the fit) or `seconds` (sent as-is,
    which is how the calibration runs are done).
    """
    try:
        import serial  # noqa: F401  -- pyserial, optional at runtime
    except ImportError:
        return False, ("pyserial not installed (pip install 'pyserial>=3.5,<4')")

    import serial as _serial

    try:
        if seconds is None:
            if grams is None:
                return False, "neither grams nor seconds given"
            seconds = grams_to_seconds(grams)
        seconds = float(seconds)
        if seconds <= 0.0 or seconds > FIRMWARE_MAX_S:
            return False, (f"duration {seconds:.2f}s outside 0-{FIRMWARE_MAX_S:.0f}s")
    except ValueError as exc:
        return False, str(exc)

    # The board resets when the port opens, so every cycle costs a boot wait.
    budget = float(timeout_s) if timeout_s else (seconds + FIRMWARE_OVERHEAD_S + 10.0)
    t0 = time.time()

    try:
        with _serial.Serial(port, baud, timeout=1) as ser:
            ready = False
            while time.time() - t0 < 10.0:
                if b"ready" in ser.readline():
                    ready = True
                    break
            if not ready:
                return False, f"no 'ready' banner from {port} within 10s"

            ser.write(f"spray {seconds}\n".encode())
            if logger is not None:
                logger.info(
                    f"[SPRAY] sent 'spray {seconds}' to {port} "
                    f"(~{seconds_to_grams(seconds):.2f}g, "
                    f"~{seconds + FIRMWARE_OVERHEAD_S:.1f}s cycle)")

            deadline = time.time() + budget
            while time.time() < deadline:
                line = ser.readline().decode(errors="replace").strip()
                if not line:
                    continue
                if logger is not None:
                    logger.info(f"[SPRAY] {line}")
                if line.startswith("OK"):
                    return True, line
                if line.startswith("ERR"):
                    return False, line
            return False, f"no OK/ERR within {budget:.0f}s"

    except _serial.SerialException as exc:
        # Unplugged, wrong port, or the user is not in the dialout group.
        return False, f"serial error on {port}: {exc}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
