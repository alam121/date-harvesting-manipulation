#!/bin/bash

export SUDO_ASKPASS=/bin/false
echo risc | sudo -S true 2>/dev/null   # cache credentials for this session

# Restart camera daemons to clear stale state (pyzed sl.CameraOne needs both)
echo risc | sudo -S systemctl restart zed_x_daemon
echo risc | sudo -S systemctl restart nvargus-daemon
sleep 8

# Set ROS domain ID
export ROS_DOMAIN_ID=6
echo "ROS_DOMAIN_ID set to 6"

# Flush ENO1 IPs
echo "Flushing IP addresses on eno1..."
echo risc | sudo -S ip addr flush dev eth0

# Set primary static IP
echo "Adding 192.168.1.101/24 to eno1..."
echo risc | sudo -S ip addr add 192.168.1.101/24 dev eth0

# Set secondary link-local IP
echo "Adding 169.254.186.100/24 to eno1..."
echo risc | sudo -S ip addr add 169.254.186.100/24 dev eth0

# LiDAR host IP (Livox Mid-70 expects host at 192.168.1.50)
echo "Adding 192.168.1.50/24 to eno1 (LiDAR)..."
echo risc | sudo -S ip addr add 192.168.1.50/24 dev eth0

# Bring interface up
echo "Bringing eno1 up..."
echo risc | sudo -S ip link set eth0 up

echo "eno1 configuration complete."

