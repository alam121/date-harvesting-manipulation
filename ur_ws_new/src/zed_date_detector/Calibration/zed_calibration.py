#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
import numpy as np
import cv2
import pyzed.sl as sl
import math

CHESSBOARD = (9, 6)
SQUARE_SIZE = 0.022   # meters

# --------------------------------------------
# Utility functions
# --------------------------------------------

def quaternion_to_rot(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])

def transform_to_matrix(tf):
    R = quaternion_to_rot([
        tf.transform.rotation.x,
        tf.transform.rotation.y,
        tf.transform.rotation.z,
        tf.transform.rotation.w
    ])
    t = np.array([
        tf.transform.translation.x,
        tf.transform.translation.y,
        tf.transform.translation.z
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def invert(T):
    R = T[:3,:3]
    t = T[:3,3]
    Ti = np.eye(4)
    Ti[:3,:3] = R.T
    Ti[:3,3] = -R.T @ t
    return Ti

def rot_to_quat(R):
    qw = math.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) / 2
    qx = (R[2,1] - R[1,2]) / (4*qw)
    qy = (R[0,2] - R[2,0]) / (4*qw)
    qz = (R[1,0] - R[0,1]) / (4*qw)
    return [qx, qy, qz, qw]

# --------------------------------------------
# Main class
# --------------------------------------------
class HandEyeCalibrator(Node):

    def __init__(self):
        super().__init__("zed_ur10_handeye")

        self.target_frame = "base_link"          # Stationary reference (e.g., robot base)
        self.source_frame = "tool0"             # Moving flange frame
        self.mount_frame = "tool0_to_camera_mount_link"  # Optional static link between tool and camera mount
        self.camera_frame = "zed_camera_frame"

        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self)

        print("\n=== Waiting for TF frames to become available ===")
        print(f"Target frame : {self.target_frame}")
        print(f"Source frame : {self.source_frame}\n")

        self.wait_for_tf_frames()

        # -------------------------------
        # ZED INITIALIZATION (same as before)
        # -------------------------------
        self.zed = sl.Camera()

        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.AUTO
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP

        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print("Camera open failed:", status)
            exit(1)

        self.runtime = sl.RuntimeParameters()
        self.image = sl.Mat()

        self.robot_poses = []
        self.board_poses = []

        print("\n=== Multi-Pose Hand-Eye Calibration ===")
        print("Move robot to different poses (8–12 recommended).")
        print("Press SPACE to capture a pose.")
        print("Press Q when done.\n")
        self.capture_loop()

        if len(self.robot_poses) >= 3:
            self.compute_handeye()
        else:
            print("Not enough samples. Need at least 3.")

    def detect_chessboard(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD)

        if not found:
            return None, False

        corners = cv2.cornerSubPix(
            gray, corners, (11,11), (-1,-1),
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        )
        return corners, True

    def get_board_pose_cam(self, corners):
        obj_pts = []
        for i in range(CHESSBOARD[1]):
            for j in range(CHESSBOARD[0]):
                obj_pts.append([j*SQUARE_SIZE, i*SQUARE_SIZE, 0])
        obj_pts = np.array(obj_pts, dtype=np.float32)

        cam_params = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
        K = np.array([[cam_params.fx, 0, cam_params.cx],
                      [0, cam_params.fy, cam_params.cy],
                      [0, 0, 1]])
        dist = np.array(cam_params.disto)

        ok, rvec, tvec = cv2.solvePnP(obj_pts, corners, K, dist)
        R, _ = cv2.Rodrigues(rvec)

        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = tvec.flatten()
        return T

    def capture_loop(self):
        cv2.namedWindow("Multi-Pose Calibration", cv2.WINDOW_NORMAL)

        while True:
            # Continuously process ROS callbacks so TF data stays fresh while GUI runs
            rclpy.spin_once(self, timeout_sec=0.0)
            if self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
                frame = self.image.get_data()

                corners, ok = self.detect_chessboard(frame)
                if ok:
                    cv2.drawChessboardCorners(frame, CHESSBOARD, corners, ok)

                cv2.imshow("Multi-Pose Calibration", frame)

                key = cv2.waitKey(10) & 0xFF
                # DEBUG: uncomment next line once to see what you're getting
                #print("key:", key)

                if key == ord(' '):  # SPACE
                    if ok:
                        print("Captured one pose.")
                        self.store_pose(corners)
                    else:
                        print("Chessboard not detected. Try again.")

                elif key == ord('q') or key == 27:  # 'q' or ESC
                    print("Quit requested.")
                    break

        cv2.destroyAllWindows()


    def store_pose(self, corners):
        # Board pose (camera frame)
        T_cam_board = self.get_board_pose_cam(corners)
        self.board_poses.append(T_cam_board)

        #print("\n--- DEBUG: Checking available TF frames ---")
        try:
            frames = self.tf_buffer.all_frames_as_yaml()
            #print(frames)
        except Exception as e:
            print("Could not list TF frames:", e)

        print(f"\nTrying TF lookup: {self.target_frame} -> {self.source_frame}")

        try:
            # Pump the TF buffer once more so the listener receives the latest pose
            rclpy.spin_once(self, timeout_sec=0.0)
            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                rclpy.time.Time())
        except Exception as e:
            print("TF ERROR:", e)
            print("\nYou need to change the TF frame names in the script.")
            print("Look at the frames printed above.\n")
            return

        T_base_tool = transform_to_matrix(tf)
        t = T_base_tool[:3, 3]
        print(f"Robot pose translation ({self.target_frame}->{self.source_frame}): {t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}")
        self.robot_poses.append(T_base_tool)
        print("Stored robot pose.")
        print(f"Total samples: {len(self.robot_poses)}")

    def wait_for_tf_frames(self):
        import time
        print("Waiting for valid TF transform...")

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

            try:
                tf = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    self.source_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.2)
                )

                # Accept only if transform exists AND timestamp is non-zero
                if tf.header.stamp.sec != 0:
                    print("TF is now valid.")
                    return

            except:
                pass

            print("Waiting for TF transform...", end="\r")
            time.sleep(0.2)


    # --------------------------------------------
    # SOLVE AX = XB
    # --------------------------------------------
    def compute_handeye(self):
        print("\n=== Solving Hand–Eye Calibration ===")

        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []

        for T_base_tool, T_cam_board in zip(self.robot_poses, self.board_poses):
            # We have:
            #   T_base_tool  : base -> tool
            #   T_cam_board  : cam  -> board
            #
            # OpenCV wants:
            #   R_gripper2base : gripper -> base
            #   R_target2cam   : target  -> cam
            #
            # So we invert both:

            T_tool_base = invert(T_base_tool)      # tool -> base
            T_board_cam = invert(T_cam_board)      # board -> cam

            R_gripper2base.append(T_tool_base[:3, :3])
            t_gripper2base.append(T_tool_base[:3, 3])

            R_target2cam.append(T_board_cam[:3, :3])
            t_target2cam.append(T_board_cam[:3, 3])

        # Solve using OpenCV (Tsai)
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base,
            t_gripper2base,
            R_target2cam,
            t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI
        )

        # X = T_cam2gripper
        T_cam2gripper = np.eye(4)
        T_cam2gripper[:3, :3] = R_cam2gripper
        T_cam2gripper[:3, 3] = t_cam2gripper.flatten()

        # We usually want gripper -> camera:
        T_gripper2cam = invert(T_cam2gripper)

        print("\nCamera → Gripper (tool0) transform (T_cam2gripper):")
        print(T_cam2gripper)
        print("\nGripper (tool0) → Camera transform (T_gripper2cam):")
        print(T_gripper2cam)

        # Use LAST robot pose (base -> tool0) to get base -> camera
        T_base_tool_last = self.robot_poses[-1]
        T_base_cam = T_base_tool_last @ T_gripper2cam

        print("\nCamera → Base transform (actually Base -> Camera matrix T_base_cam):")
        print(T_base_cam)

        print("\n=== YAML (static transform) ===")
        print(self.to_yaml(self.target_frame, self.camera_frame, T_base_cam))

        if self.mount_frame:
            try:
                rclpy.spin_once(self, timeout_sec=0.0)
                tf_mount = self.tf_buffer.lookup_transform(
                    self.source_frame,
                    self.mount_frame,
                    rclpy.time.Time())
                T_tool_mount = transform_to_matrix(tf_mount)
                T_mount_tool = invert(T_tool_mount)
                T_mount_cam = T_mount_tool @ T_gripper2cam

                print("\n=== YAML (tool mount -> camera) ===")
                print(self.to_yaml(self.mount_frame, self.camera_frame, T_mount_cam))
            except Exception as e:
                print(f"Could not retrieve {self.mount_frame} transform: {e}")


    def to_yaml(self, parent_frame, child_frame, T):
        t = T[:3,3].tolist()
        q = rot_to_quat(T[:3,:3])
        return f"""
static_transform_publisher:
  frame_id: "{parent_frame}"
  child_frame_id: "{child_frame}"
  translation: [{t[0]}, {t[1]}, {t[2]}]
  rotation: [{q[0]}, {q[1]}, {q[2]}, {q[3]}]
"""

# --------------------------------------------
def main():
    rclpy.init()
    HandEyeCalibrator()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
