#!/bin/bash

# Restart camera daemons to clear stale state (pyzed sl.CameraOne needs both)
sudo systemctl restart zed_x_daemon
sudo systemctl restart nvargus-daemon
sleep 8

# Set ROS domain ID
export ROS_DOMAIN_ID=5
echo "ROS_DOMAIN_ID set to 5"

# Flush ENO1 IPs
echo "Flushing IP addresses on eno1..."
sudo ip addr flush dev eth0

# Set primary static IP
echo "Adding 192.168.1.101/24 to eno1..."
sudo ip addr add 192.168.1.101/24 dev eth0

# Set secondary link-local IP
echo "Adding 169.254.186.100/24 to eno1..."
sudo ip addr add 169.254.186.100/24 dev eth0

# LiDAR host IP (Livox Mid-70 expects host at 192.168.1.50)
echo "Adding 192.168.1.50/24 to eno1 (LiDAR)..."
sudo ip addr add 192.168.1.50/24 dev eth0

# Bring interface up
echo "Bringing eno1 up..."
sudo ip link set eth0 up

echo "eno1 configuration complete."

