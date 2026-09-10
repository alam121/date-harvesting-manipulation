#!/bin/bash

export SUDO_ASKPASS=/bin/false
echo risc | sudo -S true 2>/dev/null   # cache credentials for this session

# Restart camera daemons to clear stale state (pyzed sl.CameraOne needs both)
echo risc | sudo -S systemctl restart zed_x_daemon
echo risc | sudo -S systemctl restart nvargus-daemon
sleep 8

# ── Clock sanity (must run BEFORE any ROS node starts) ────────────────────
# This unit has no battery-backed RTC, so an offline boot comes up at the epoch
# (1970). A wrong-but-stable clock is harmless here: every ROS node runs on this
# one machine and shares it, so TF stays self-consistent. What is NOT harmless is
# a clock that STEPS while nodes are running -- every buffered transform becomes
# decades stale, stamped TF lookups start failing, and the vision node silently
# falls back to "latest", which quietly turns pipeline latency into position
# error. systemd-timesyncd always steps, never slews.
#
# fake-hwclock (installed 2026-09-10) normally restores the clock at boot, and
# chrony is configured "makestep 1 3" so it steps only in the first 3 updates
# after startup and slews from then on -- a mid-session WiFi reconnect nudges
# instead of jumping. This block is the belt-and-braces check in case
# fake-hwclock did not run: pull the clock forward while nothing is running.
if [ "$(date +%Y)" -lt 2025 ]; then
    _ref=/etc/fake-hwclock.data
    if [ -s "$_ref" ]; then
        echo "Clock is unset (no RTC battery). Restoring from $_ref ..."
        echo risc | sudo -S date -u -s "$(cat "$_ref")" >/dev/null 2>&1
    fi
fi
if [ "$(date +%Y)" -lt 2025 ]; then
    echo ""
    echo "*** WARNING: system clock is still unset ($(date))."
    echo "*** Logs, screenshots and any calibration saved this session will be"
    echo "*** stamped 1970, and connecting to a network mid-run will step the"
    echo "*** clock and invalidate the TF buffer."
    echo "*** Set it before launching:  sudo timedatectl set-time 'YYYY-MM-DD HH:MM:SS'"
    echo ""
else
    echo "Clock OK: $(date)"
fi

# Set ROS domain ID
export ROS_DOMAIN_ID=6
echo "ROS_DOMAIN_ID set to 6"

# Flush Ethernet IPs
echo "Flushing IP addresses on eth0..."
echo risc | sudo -S ip addr flush dev eth0

# Set primary static IP
echo "Adding 192.168.1.101/24 to eth0..."
echo risc | sudo -S ip addr add 192.168.1.101/24 dev eth0

# Set secondary link-local IP
echo "Adding 169.254.186.100/24 to eth0..."
echo risc | sudo -S ip addr add 169.254.186.100/24 dev eth0

# LiDAR host IP (Livox Mid-70 expects host at 192.168.1.50)
echo "Adding 192.168.1.50/24 to eth0 (LiDAR)..."
echo risc | sudo -S ip addr add 192.168.1.50/24 dev eth0

# Bring interface up
echo "Bringing eth0 up..."
echo risc | sudo -S ip link set eth0 up

# NetworkManager marks these addresses noprefixroute, so add the robot/LiDAR
# subnet route explicitly instead of allowing traffic to escape over Wi-Fi.
echo "Routing 192.168.1.0/24 through eth0..."
echo risc | sudo -S ip route replace 192.168.1.0/24 dev eth0 src 192.168.1.101

echo "eth0 configuration complete."
