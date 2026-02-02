#!/usr/bin/env bash
set -euo pipefail

echo "[1/4] Killing processes that may hold the camera..."
sudo pkill -f "ur10e_curobo|zed|ZED|gst-launch|gstreamer|python3" || true

echo "[2/4] Restarting ZED X daemon..."
sudo systemctl restart zed_x_daemon

echo "[3/4] Restarting nvargus-daemon..."
sudo systemctl restart nvargus-daemon

# OPTIONAL: uncomment ONLY if you previously lowered lane rate on Orin NX devkit
# echo "[4/4] Re-applying MIPI lane rate (2.0 Gbps)..."
# sudo i2cset -y -f 30 0x29 0x04 0x15 0x34 i
# sudo i2cset -y -f 30 0x29 0x04 0x18 0x34 i
# sudo i2cset -y -f 30 0x29 0x04 0x1B 0x34 i
# sudo i2cset -y -f 30 0x29 0x04 0x1E 0x34 i

echo "Done. Try running: python3 -m ur10e_curobo.vision"

