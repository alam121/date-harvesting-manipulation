#!/usr/bin/env bash
# Record the Jetson's stereo stream (shared over the network) for a NuRec reconstruction.
#
#   isaac/nurec/record_stereo.sh [output_bag_dir]
#
# Topics are auto-detected (compressed preferred: raw 1080p stereo does not fit in 1 GbE).
# Override with LEFT_IMAGE, RIGHT_IMAGE, LEFT_INFO, RIGHT_INFO. Stop with Ctrl+C.
set -eo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/jetson_ros_env.sh"

OUT="${1:-$HOME/nurec_data/bags/stereo_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$(dirname "$OUT")"

ros2 daemon stop >/dev/null 2>&1 || true

# The date-palm vision node only publishes /vision/nurec/* stereo while capture mode is on.
camera_command() {
    timeout 15 ros2 topic pub --once /camera_command std_msgs/msg/String "data: '$1'" >/dev/null 2>&1 || true
}
if [[ -z "$LEFT_IMAGE" ]] && timeout 15 ros2 topic list | grep -qx /camera_command; then
    echo "[INFO] Enabling vision-node NuRec capture (nurec on)"
    camera_command "nurec on"
    trap 'echo "[INFO] Disabling NuRec capture"; camera_command "nurec off"' EXIT
    for _ in $(seq 10); do
        timeout 10 ros2 topic list | grep -q "^/vision/nurec/" && break
        sleep 2
    done
fi
TOPICS="$(timeout 15 ros2 topic list)"

pick_image() {
    local side=$1
    local pattern
    for pattern in "${side}.*rect.*/compressed$" "${side}.*/compressed$" "${side}.*rect[^/]*$" "${side}.*image[^/]*$"; do
        local topic
        topic="$(grep -E "$pattern" <<<"$TOPICS" | grep -v -E "depth|disparity|confidence|gray|theora|compressedDepth|zstd" | head -1 || true)"
        if [[ -n "$topic" ]]; then
            echo "$topic"
            return
        fi
    done
}

# camera_info living next to the image topic (ZED wrapper 4.x and 5.x layouts both satisfy this).
pick_info() {
    local side=$1 image=$2
    local base="${image%/compressed}"
    local dir="${base%/*}"
    while [[ -n "$dir" ]]; do
        if grep -qx "$dir/camera_info" <<<"$TOPICS"; then
            echo "$dir/camera_info"
            return
        fi
        dir="${dir%/*}"
    done
    grep -E "${side}.*camera_info$" <<<"$TOPICS" | grep -v raw | head -1 || true
}

LEFT_IMAGE="${LEFT_IMAGE:-$(pick_image left)}"
RIGHT_IMAGE="${RIGHT_IMAGE:-$(pick_image right)}"

if [[ -z "$LEFT_IMAGE" || -z "$RIGHT_IMAGE" ]]; then
    echo "[ERROR] No left/right image topics visible on ROS_DOMAIN_ID=$ROS_DOMAIN_ID." >&2
    echo "        Visible topics:" >&2
    sed 's/^/          /' <<<"$TOPICS" >&2
    echo "        Check: Jetson uses the same ROS_DOMAIN_ID/RMW, both hosts are on one subnet," >&2
    echo "        and multicast is not blocked (see isaac/nurec/README.md)." >&2
    exit 1
fi

LEFT_INFO="${LEFT_INFO:-$(pick_info left "$LEFT_IMAGE")}"
RIGHT_INFO="${RIGHT_INFO:-$(pick_info right "$RIGHT_IMAGE")}"

RECORD=("$LEFT_IMAGE" "$RIGHT_IMAGE" "$LEFT_INFO" "$RIGHT_INFO")
grep -qx "/tf" <<<"$TOPICS" && RECORD+=(/tf)
grep -qx "/tf_static" <<<"$TOPICS" && RECORD+=(/tf_static)
# URDF (latched) lets replay_bag_joints.py recover joint angles from /tf without the robot.
grep -qx "/robot_description" <<<"$TOPICS" && RECORD+=(/robot_description)

echo "[INFO] Recording to $OUT"
printf '         %s\n' "${RECORD[@]}"
if [[ "$LEFT_IMAGE" != */compressed ]]; then
    echo "[WARN] Recording raw images over the network; expect drops. Prefer the Jetson's /compressed topics."
fi
echo "[INFO] Move slowly around the scene (keep ~70% overlap, avoid motion blur). Ctrl+C to stop."

ros2 bag record -o "$OUT" "${RECORD[@]}"
