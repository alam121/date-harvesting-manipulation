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
