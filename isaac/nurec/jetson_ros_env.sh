# Source (do not execute) before talking to the Jetson camera stack over the network.
# Must match the Jetson side: same ROS_DOMAIN_ID and RMW (FastDDS, Humble default).
#
# Multicast discovery does not cross campus subnets, so the Jetson is added as a unicast
# peer (JETSON_IP). The Jetson needs the mirror profile pointing at this workstation.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JETSON_IP="${JETSON_IP:-10.85.49.36}"

source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-6}"
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

NUREC_FASTDDS_PROFILE="${TMPDIR:-/tmp}/nurec_fastdds_${USER}.xml"
cat > "$NUREC_FASTDDS_PROFILE" <<EOF
<?xml version="1.0" encoding="UTF-8" ?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
    <transport_descriptors>
        <transport_descriptor>
            <transport_id>udp_transport</transport_id>
            <type>UDPv4</type>
            <sendBufferSize>8388608</sendBufferSize>
            <receiveBufferSize>8388608</receiveBufferSize>
            <!-- Probe 32 participant ports on the peer (default 4); the camera nodes start after the UR stack. -->
            <maxInitialPeersRange>32</maxInitialPeersRange>
        </transport_descriptor>
    </transport_descriptors>
    <participant profile_name="jetson_peer" is_default_profile="true">
        <rtps>
            <builtin>
                <initialPeersList>
                    <locator><udpv4><address>${JETSON_IP}</address></udpv4></locator>
                </initialPeersList>
            </builtin>
            <userTransports>
                <transport_id>udp_transport</transport_id>
            </userTransports>
            <useBuiltinTransports>false</useBuiltinTransports>
        </rtps>
    </participant>
</profiles>
EOF
export FASTRTPS_DEFAULT_PROFILES_FILE="$NUREC_FASTDDS_PROFILE"

# Conda's python3 shadows the system interpreter that ROS (rosbag2_py, cv2) is built against.
export NUREC_ROS_PYTHON=/usr/bin/python3
