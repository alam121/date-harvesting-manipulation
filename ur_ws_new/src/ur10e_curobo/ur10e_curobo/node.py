# ruff: noqa
import threading
import rclpy
import os
from rclpy.timer import Timer

from rclpy.node import Node
from sensor_msgs.msg import JointState as ROSJointState
from visualization_msgs.msg import InteractiveMarkerFeedback, Marker
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener

from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

from .config import JOINT_ORDER, WORLD_CONFIG, DEFAULT_QOS
from .utils import read_key
from . import fk as fk_mod
from . import markers as markers_mod
from . import motions as motions_mod
from . import goals as goals_mod
from . import gripper as gripper_mod
from .perception import ZedYoloPerception

class UR10eCuroboMoveIt(Node):
    def __init__(self):
        super().__init__("ur10e_curobo_moveit_node")
        # pubs
        from trajectory_msgs.msg import JointTrajectory  # type: ignore
        self.trajectory_pub = self.create_publisher(JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10)
        self.goal_marker_pub = self.create_publisher(Marker, "/goal_positions_marker", 10)
        self.path_marker_pub = self.create_publisher(Marker, "/robot_path_marker", 10)
        # subs
        self.create_subscription(ROSJointState, "/joint_states", self._joint_state_cb, 10)
        self.create_subscription(InteractiveMarkerFeedback,
                                 "/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/feedback",
                                 self._marker_cb, 10)
        self.create_subscription(Float32MultiArray, "/gripper/force", self._force_cb, 10)
        self.create_subscription(Bool, "/emergency_stop", self._stop_cb, 10)
        self.create_subscription(Bool, "/io_and_status_controller/robot_program_running", self._robot_running_cb, 10)
        # tf
        self.tf_buffer = Buffer(); self.tf_listener = TransformListener(self.tf_buffer, self)
        # timers
        self.create_timer(0.1, lambda: markers_mod.track_robot_path(self))
        self.create_timer(0.02, self._classifier_tick)
        self.timer_wait_js = self.create_timer(0.5, self._check_joint_states)
        # cuRobo
        self.motion_gen_config = MotionGenConfig.load_from_robot_config("ur10e.yml", WORLD_CONFIG, interpolation_dt=0.004)
        self.motion_gen = MotionGen(self.motion_gen_config); self.motion_gen.warmup()
        print('warming up done')
        # state
        self.qos = DEFAULT_QOS
        self.joint_order = JOINT_ORDER
        self.current_joint_positions = None
        self.current_joint_velocities = None
        self.latest_marker_pose = None
        self.goal_poses = []
        self.path_points = []
        self.running = True
        self.speed_scale = 1.8
        self.stop_requested = False
        self.robot_running = False
        # joints presets
        self.home_joints = [-1.5916569868670862, -1.649402920399801, 2.113215446472168, -4.4819199482547205, 4.604945659637451, -0.05743295351137334]
        self.dropoff_joints = [-2.16858417192568, -1.3347657362567347, 2.0885677337646484, -2.6394265333758753, 4.78283166885376, 0.013545919209718704]
        self.predropoff_joints = [-1.6832264105426233, -2.020153347645895, 2.238132953643799, -3.9681833426104944, 4.682962894439697, -0.010893646870748341]
        # goal capture
        self.goal_capture_active = False; self.goal_capture_timer = None; self.goal_capture_count = 0
        self.goal_sort_ref = None; self.goal_sort_ascending = True
        # gripper/classifier
        gripper_mod.init_gripper(self)
        # keyboard thread
        self.keyboard_thread = threading.Thread(target=self._wait_for_key_press, daemon=True); self.keyboard_thread.start()
        self.get_logger().info("UR10e cuRobo node initialized. Waiting for joint states…")
        
        # # Perception: ZED + YOLO
        #self._maybe_start_perception()

    # callbacks
    def _check_joint_states(self):
        if self.current_joint_positions is not None:
            self.get_logger().info("Initial joints received."); self.destroy_timer(self.timer_wait_js)

    def _joint_state_cb(self, msg):
        jm = dict(zip(msg.name, msg.position)); vm = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}
        self.current_joint_positions = [jm[j] for j in self.joint_order if j in jm]
        self.current_joint_velocities = [vm.get(j, 0.0) for j in self.joint_order]

    def _marker_cb(self, msg):
        self.latest_marker_pose = msg.pose

    def _robot_running_cb(self, msg):
        self.robot_running = msg.data
        self.get_logger().info("✅ Robot program is running." if msg.data else "⚠️ Robot program is NOT running.")

    def _stop_cb(self, msg):
        if not msg.data: return
        self.get_logger().warn("🛑 Emergency stop!"); self.stop_requested = True; from .motions import publish_stop_trajectory; publish_stop_trajectory(self)

    def _force_cb(self, msg):
        self.classifier.on_force(list(msg.data)[:3])

    def _classifier_tick(self):
        self.classifier.tick()

    # methods used by helpers (so helpers can call like node.get_end_effector_pose())
    def get_end_effector_pose(self):
        return fk_mod.get_end_effector_pose(self)

    # keyboard UI
    def _wait_for_key_press(self):
        print("Keys: y=save marker, m=manual, s=subscribe, p=capture, n=execute, o=open, c=close, h=home, d=dropoff, k=stop, q=quit")
        while self.running:
            key = read_key()
            if key is None: continue
            if key == 'y' and self.latest_marker_pose:
                from .goals import pose_to_vec7
                g = pose_to_vec7(self.latest_marker_pose); self.goal_poses.append(g); print(f"Saved goal: {g}"); markers_mod.publish_goal_marker(self, g[:3])
            elif key == 'm': self._manual_goal()
            elif key == 's': goals_mod.subscribe_to_goal_pose(self)
            elif key == 'p': self.start_goal_capture(10.0)
            elif key == 'n' and self.goal_poses: self._prep_and_execute()
            elif key == 'o': gripper_mod.control_gripper(self, 'OPEN')
            elif key == 'c': gripper_mod.control_gripper(self, 'CLOSE'); motions_mod.move_backward(self, -0.01); motions_mod.rotate_wrist(self, 120)
            elif key == 'h': motions_mod.move_to_home_position(self)
            elif key == 'd': motions_mod.move_to_dropoff_position(self); gripper_mod.control_gripper(self, 'OPEN')
            elif key == 'k': self.stop_requested = True; motions_mod.publish_stop_trajectory(self)
            elif key == 'q': self.running = False; break

    def _manual_goal(self):
        cur = self.get_end_effector_pose()
        try:
            x = float(input("Enter X: "))
            y = float(input("Enter Y: "))
            z = float(input("Enter Z: "))
        except ValueError:
            print("Invalid input."); return
        goal = [x, y, z] + (cur[3:] if cur else [1.0,0.0,0.0,0.0])
        self.goal_poses.append(goal); markers_mod.publish_goal_marker(self, goal[:3]); print("Manual goal saved.")

    # goal capture (timed)
    def start_goal_capture(self, duration: float = 30.0):
        if self.goal_capture_active: self.stop_goal_capture()
        self.goal_poses.clear(); self.goal_capture_count = 0
        ee = self.get_end_effector_pose(); self.goal_sort_ref = ee[:3] if ee else [0.0,0.0,0.0]
        self.goal_pose_sub = self.create_subscription(PoseStamped, '/external_goal_pose', self._capture_goal_cb, self.qos)
        self.goal_capture_active = True
        self.get_logger().info(f"Started goal capture for {duration:.0f}s")
        def _stop_once():
            if self.goal_capture_timer: self.goal_capture_timer.cancel()
            self.stop_goal_capture()
        self.goal_capture_timer = self.create_timer(duration, _stop_once)

    def stop_goal_capture(self):
        if not self.goal_capture_active: return
        self.goal_capture_active = False
        if hasattr(self, 'goal_pose_sub'):
            self.destroy_subscription(self.goal_pose_sub); del self.goal_pose_sub
        self.goal_poses.sort(key=lambda g: __import__('math').dist(g[:3], self.goal_sort_ref or [0,0,0]))
        self.get_logger().info(f"Goal capture stopped. Collected {self.goal_capture_count} goals.")

    def _capture_goal_cb(self, msg: PoseStamped):
        if goals_mod.is_robot_moving(self): return
        gx,gy,gz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        cur = self.get_end_effector_pose();
        qw,qx,qy,qz = (cur[3:] if cur else [1.0,0.0,0.0,0.0])
        import math
        if any(math.dist([gx,gy,gz], g[:3]) < 0.01 for g in (self.goal_poses or [])): return
        goal = [gx,gy,gz,qw,qx,qy,qz]; self.goal_poses.append(goal)
        self.goal_poses.sort(key=lambda g: math.dist(g[:3], self.goal_sort_ref or [0,0,0]))
        self.goal_capture_count += 1; markers_mod.publish_goal_marker(self, goal[:3])

    def _prep_and_execute(self):
        if self.goal_capture_active: self.stop_goal_capture()
        if hasattr(self, 'goal_pose_sub'): self.destroy_subscription(self.goal_pose_sub); del self.goal_pose_sub
        self.get_logger().info("Executing stored goals…"); goals_mod.plan_and_execute(self); self.get_logger().info("Done.")

    # expose some helpers for external callers
    def control_gripper(self, action: str):
        gripper_mod.control_gripper(self, action)


    def _start_perception_once(self):
        # run exactly once
        self.perception_timer.cancel()
        try:
            from .perception import ZedYoloPerception

            # resolve defaults safely (tiny model; no window)
            weights = os.getenv("UR10E_YOLO_WEIGHTS", "exp_aug.pt")
            imgsz   = int(os.getenv("UR10E_YOLO_IMGSZ", "640"))
            conf    = float(os.getenv("UR10E_YOLO_CONF", "0.45"))
            cam_fr  = os.getenv("UR10E_CAM_FRAME", "zed2_left_camera_frame")
            show    = bool(int(os.getenv("UR10E_SHOW_VIEW", "1")))
            cpu_only = os.getenv("CUDA_VISIBLE_DEVICES", "") == ""

            self.get_logger().info(
                f"Starting perception (weights={weights}, imgsz={imgsz}, conf={conf}, "
                f"cam={cam_fr}, show={int(show)}, cpu_only={cpu_only})"
            )

            self.perception = ZedYoloPerception(
                self,
                weights=weights,
                img_size=imgsz,
                conf_thres=conf,
                cam_frame=cam_fr,
                show_view=show,
            )
            self.perception.start()
            self.get_logger().info("Perception started.")
        except Exception as e:
            self.get_logger().error(f"Perception failed to start: [{type(e).__name__}] {e}")
            self.perception = None  # don’t crash the whole node

    def _maybe_start_perception(self):
        if os.getenv("UR10E_DISABLE_PERCEPTION", "0") == "1":
            self.get_logger().info("Perception disabled by UR10E_DISABLE_PERCEPTION=1")
            return
        # Delay startup to avoid RAM spikes colliding with cuRobo init
        self.perception_timer: Timer = self.create_timer(5.0, self._start_perception_once)
