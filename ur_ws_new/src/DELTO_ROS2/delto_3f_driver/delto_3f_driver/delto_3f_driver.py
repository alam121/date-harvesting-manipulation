#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import threading
import sys
import os
import time

import rclpy

from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.duration import Duration 
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Int32, Bool, Float32MultiArray, Int16MultiArray
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from delto_utility import delto_modbus_TCP as delto_TCP



class DeltoROSDriver(Node):

    def __init__(self):

        # ROS2 Node Initialize
        super().__init__('delto_3f_driver')
        qos_profile = QoSProfile(
            depth=2,
            reliability=QoSReliabilityPolicy.RELIABLE
        )

        fast_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            depth=1,
            lifespan=Duration(seconds=1),
            history=QoSHistoryPolicy.KEEP_LAST)
        
        self.declare_parameter('ip', "169.254.186.62")
        self.declare_parameter('port', 10000)
        self.declare_parameter('slaveID', 1)
        self.declare_parameter('dummy', False)
        self.declare_parameter('grasp_force', 50)
        # Motor-current -> contact-channel mapping, per finger. Defaults are the
        # values that were hardcoded in estimate_force(), so behaviour is
        # unchanged until someone deliberately measures better ones.
        #
        # force_baseline is the NO-LOAD current of each finger's sensed motor.
        # It was one shared -17.5 for all three, but the three motors do not
        # idle at the same current (different units, different gravity loading
        # by pose), and the value drifts with temperature. Measure per finger
        # with the hand open and unloaded, read /gripper/joint_states effort
        # x1000 for indices 3/7/11, and set them here.
        # force_baseline accepts 3 values (one per finger, applied to all four of
        # that finger's motors) or 12 (one per motor, the accurate form).
        #
        # MEASURED 2026-09-10 on this gripper: 667 samples over 8s, hand open and
        # unloaded, median per motor (stdev ~1.1mA on every channel).
        #
        # The previous default was a single -17.5 shared by all three fingers,
        # and it was simply wrong for this hardware: every motor actually idles
        # near ZERO. raw = current - (-17.5) was therefore always ~+23, so
        # /gripper/force read 7.35/7.05/7.95 N with the hand idle and open, and
        # the deadband could never engage because nothing was ever within 0.3 of
        # -17.5. DeltoGripperController re-baselines at closing time, which is
        # why contact detection worked at all -- it subtracted the bogus offset
        # back out -- but the published channel was meaningless in absolute
        # terms and had poor resolution.
        #
        # These are pose-dependent (gravity loads the fingers differently). They
        # were measured at the parked pose; re-measure at the grasp pose if you
        # want better. Procedure: hand open and unloaded, average
        # /gripper/joint_states effort x1000 per motor.
        self.declare_parameter(
            'force_baseline',
            [4.0, 2.0, 10.0, 5.0, 2.0, 0.0, 4.0, 6.0, 0.0, 1.0, 8.0, 2.0])
        # Which motor's current becomes each finger's contact channel.
        #
        #   "fixed"          -- read force_motor_indices, the long-standing
        #                       behaviour: the 4th motor of each finger.
        #   "max_per_finger" -- take whichever of that finger's four motors is
        #                       furthest from its baseline, matching what the
        #                       contact tracker in delto_gripper_controller
        #                       already does (it maxes over range(f*4, f*4+4)).
        #
        # The fixed form assumed one motor per finger carries the load. That held
        # for the OLD gripper, whose driving joint was a single index per finger.
        # The NEW gripper drives two joints per finger (2-3, 6-7, 10-11), so the
        # load can land on either -- observed 2026-09-10: finger 2 pulled a
        # 132mA delta and stopped 8.9deg short of its calibrated close while its
        # force channel read exactly 0.00N, because index 7 was idle and index 6
        # carried the load. That mislabelled a real grasp as WEAK.
        #
        # Switched to max_per_finger 2026-09-10 after three consecutive runs in
        # which finger 2 drew the highest current of all three (122-161mA) and
        # stopped 9-12deg short of its calibrated close, while its force channel
        # read ~0 and the grasp was labelled WEAK. Index 7 was idle; the load was
        # on another of that finger's motors.
        #
        # NOTE: max_per_finger reports a LARGER magnitude than "fixed", and
        # DeltoGripperController.force_threshold (4.0) was tuned against the old
        # signal. Set this back to 'fixed' if closures start stopping early.
        # REVERTED to 'fixed' 2026-09-10 after max_per_finger caused a false
        # contact confirmation on an EMPTY close: goals.py computes
        # _force_confirmed = sum(d > 1.5 for d in deltas) >= 2 and skips the
        # regrip when it fires. baseline_force is re-captured at each close, so
        # the delta measures open->closed change -- and driving the fingers
        # through their travel draws current even with nothing in the hand.
        # 'fixed' read one motor that was often idle; max_per_finger reads
        # whichever of the four works hardest, which is never zero, so empty
        # closes cleared 1.5. Observed: contacts=0/3, current_delta=[7,9,11]mA,
        # outcome=NO_GRAB -- and the regrip was skipped on
        # deltas=[3.00,1.80,3.30].
        #
        # The per-motor baselines below are unaffected and stay: baseline_force
        # is re-captured per close, so it cancels out of the delta either way.
        #
        # The real fix is for _force_confirmed to use the contact tracker
        # (contacts/current_delta from joint_states), which correctly reported
        # 0/3 on that close, rather than this channel. Until then, 'fixed'.
        self.declare_parameter('force_motor_mode', 'fixed')
        self.declare_parameter('force_motor_indices', [3, 7, 11])
        self.declare_parameter('force_scale', 0.3)
        # ~3 sigma of the measured 1.1mA per-channel noise floor, so an idle
        # hand reads a clean 0.0 instead of jittering. Was 0.3, which was sized
        # against the old (wrong) -17.5 baseline and never engaged anyway.
        self.declare_parameter('force_deadband', 3.5)

        self.joint_state_list = [0.0]*12
        self.current_joint_state = [0.0]*12
        self.current_joint_velocity = [0.0]*12
        self.current_joint_effort = [0.0]*12
        self.raw_current_state = [0]*12
        # When the values above were actually read off the gripper. joint_states
        # is published at publish_rate but refreshed at feedback_read_rate, so
        # stamping "now" at publish time claimed data was fresher than it was.
        self._joint_state_stamp = None
        self.target_joint_state = [0.0]*12
        self.fixed_joint_state = [0]*12

        self.joint_state_feedback = JointTrajectoryPoint()
        self.vel = []
        self.is_dummy = bool(self.get_parameter('dummy').value)
        self.delto_client = delto_TCP.Communication(dummy=self.is_dummy)
        self.stop_thread = False
        self.lock = threading.Lock()
        self.is_connected = False
        
        # Too high frequency will cause blocking sub/pub
        self.publish_rate = 100
        self.feedback_read_rate = 20
        print('publish late : '+str(self.publish_rate))
        self.get_logger().info(f"Delto feedback read rate: {self.feedback_read_rate} Hz")
        self.get_logger().info(f"Delto driver dummy mode: {self.is_dummy}")
        self.is_connected = False
        # Action Server
        self.jcm_action_server = ActionServer(
            self,
            FollowJointTrajectory,
            'delto_controller/follow_joint_trajectory',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self.joint_state_pub = self.create_publisher(
            JointState, 'gripper/joint_states', fast_qos)
        self.grasp_sub = self.create_subscription(
            Bool, 'gripper/grasp', callback=self.grasp_callback, qos_profile=qos_profile)
        self.write_register_sub = self.create_subscription(
            Int16MultiArray, 'gripper/write_register', self.write_register_callback, qos_profile=qos_profile)
        self.grasp_mode_sub = self.create_subscription(
            Int32, 'gripper/grasp_mode', callback=self.grasp_mode_callback, qos_profile=qos_profile)
        self.grasp_force_sub = self.create_subscription(
            Int32, 'gripper/set_grasp_force',
            callback=self.grasp_force_callback, qos_profile=qos_profile)
        self.holding_control_sub = self.create_subscription(
            Int16MultiArray, 'gripper/holding_control',
            callback=self.holding_control_callback, qos_profile=qos_profile)
        self.target_joint_sub = self.create_subscription(
            Float32MultiArray, 'gripper/target_joint', callback=self.target_joint_callback, qos_profile=qos_profile)
            
            
        # ROS Publisher for Force Data
        self.force_pub = self.create_publisher(Float32MultiArray, 'gripper/force', qos_profile)

        

        self.joint_state_timer = self.create_timer(
            1/self.publish_rate, self.timer_callback)
        self.read_joint_timer = self.create_timer(
            1/self.feedback_read_rate, self.read_joint_callback)
            
            # Start the force reading timer
        # Publish the motor-current-derived contact channels at the physical
        # feedback rate.  The previous 2 Hz stream was too slow to observe a
        # 0.5 s closing motion and caused the grasp classifier to use stale
        # samples.
        self.force_timer = self.create_timer(
            1.0 / self.feedback_read_rate, self.publish_force_data)
        
        self.fixed_joint_sub = self.create_subscription(
            Int16MultiArray, 'gripper/fixed_joint', self.fixed_joint_callback, qos_profile=qos_profile)
        self.set_gain_sub = self.create_subscription(
            Int16MultiArray, 'gripper/request/gain', self.set_gain_callback, qos_profile=qos_profile)
        self.gain_pub = self.create_publisher(
            Int16MultiArray, 'gripper/response/gain', qos_profile=qos_profile)
        
        self.load_pose_sub = self.create_subscription(
            Int32, 'gripper/load_pose', self.load_pose_callback, qos_profile=qos_profile)
        self.save_pose_sub = self.create_subscription(
            Int32, 'gripper/save_pose', self.save_pose_callback, qos_profile=qos_profile)

        self.reconnect_timer = self.create_timer(1.0, self.reconnect_callback)
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 100
    
    def load_pose_callback(self, msg):
        
        if msg.data < 1 or msg.data > 30:
            self.get_logger().error("pose index out of range")
            
        self.delto_client.load_pose(msg.data)
        
    def save_pose_callback(self, msg):
        
        if msg.data< 1 or msg.data > 30:
            self.get_logger().error("pose index out of range")
            return
        
        self.delto_client.save_pose(msg.data)
        
    def reconnect_callback(self):
        if not self.is_connected:
            if self.reconnect_attempts < self.max_reconnect_attempts:
                self.get_logger().info(f"Attempting to reconnect (attempt {self.reconnect_attempts + 1}/{self.max_reconnect_attempts})")
                
                try: 
                    self.connect()
                    self.reconnect_attempts += 1
                    
                    if self.is_connected:
                        self.get_logger().info("Reconnected successfully")
                        self.reconnect_attempts = 0
                        
                except Exception as e:
                    self.get_logger().error(f"Failed to reconnect: {e}")
                    self.reconnect_attempts += 1
            else:
                self.get_logger().error("Maximum reconnect attempts reached. Shutting down.")
                self.destroy_node()
        else:
            self.reconnect_attempts = 0
            
    # Connect to the delto gripper
    def set_gain_callback(self, msg):
        
        if len(msg.data) != 24:
            print(msg.data)
            self.get_logger().error("Invalid gain {0}".format(msg.data.size))    
            return
        
        self.delto_client.set_pgain(msg.data[0:12])
        self.delto_client.set_dgain(msg.data[12:24])
        pgain=self.delto_client.get_pgain()
        dgain=self.delto_client.get_dgain()
        #append pgain and dgain
        data= []
        data.extend(pgain)
        data.extend(dgain)
        print(data) 
        msg = Int16MultiArray()
        msg.data = data
        
        self.gain_pub.publish(msg)
        
        
        
    def connect(self) -> bool:

        if self.is_dummy:
            print("Dummy mode")
            return True

        print("Connecting to the delto gripper...")
        is_connected = self.delto_client.connect(self.get_parameter('ip').value,
                                         self.get_parameter('port').value,
                                         self.get_parameter('slaveID').value)
        self.is_connected = is_connected
        if is_connected:
            try:
                # DG-3F-M requires control-start before target register writes
                # execute. Keep the public ROS topics unchanged for the rest of
                # this application.
                response = self.delto_client.start_control()
                if hasattr(response, "isError") and response.isError():
                    self.get_logger().error(f"DG-3F-M control mode start failed: {response}")
                else:
                    self.get_logger().info("DG-3F-M control mode started")
                self._apply_grasp_force(
                    int(self.get_parameter('grasp_force').value))
            except Exception as e:
                self.get_logger().warn(f"Could not start DG-3F-M control mode: {e}")
        
        return is_connected

    def _apply_grasp_force(self, value: int):
        requested = max(0, min(200, int(value)))
        response = self.delto_client.set_grasp_force(requested)
        if hasattr(response, "isError") and response.isError():
            raise RuntimeError(f"grasp-force write failed: {response}")
        actual = self.delto_client.get_grasp_force()
        if actual != requested:
            raise RuntimeError(
                f"grasp-force verification mismatch: requested={requested}, read={actual}")
        self.get_logger().info(
            f"DG-3F-M grasp force set to {actual} ({actual * 0.1:.1f} N)")

    def grasp_force_callback(self, msg: Int32):
        if not self.is_connected:
            self.get_logger().error("Cannot set grasp force: connection lost")
            return
        try:
            self._apply_grasp_force(msg.data)
        except Exception as e:
            self.get_logger().error(f"Failed to set grasp force: {e}")

    @staticmethod
    def _checked_write(response, label):
        if hasattr(response, "isError") and response.isError():
            raise RuntimeError(f"{label} failed: {response}")

    def holding_control_callback(self, msg: Int16MultiArray):
        """Apply one ordered DG-3F-M hold command.

        Payload: [action, mode, force_0p1N, hold_M1, ..., hold_M12].
        A hold value of 1 locks that joint; 0 lets the grasp algorithm move it.
        """
        if len(msg.data) != 15:
            self.get_logger().error(
                f"Invalid holding control: expected 15 values, got {len(msg.data)}")
            return
        if not self.is_connected:
            self.get_logger().error("Cannot set holding control: connection lost")
            return
        action, mode, force = (int(msg.data[0]), int(msg.data[1]), int(msg.data[2]))
        hold_mask = [int(value) for value in msg.data[3:15]]
        if action not in (0, 1) or mode not in (1, 2, 3, 4, 5, 6):
            self.get_logger().error(
                f"Invalid holding control action={action} mode={mode}")
            return
        try:
            # Prevent the high-rate joint/current reads from being inserted
            # between the ordered setup writes and GRASP_ACTION=1.
            with self.delto_client.lock:
                if action == 0:
                    self._checked_write(
                        self.delto_client.set_grasp_action(False), "grasp release")
                    self.get_logger().info("DG-3F-M holding force released")
                    return
                self._checked_write(
                    self.delto_client.set_grasp_mode(mode), "grasp mode")
                self._checked_write(
                    self.delto_client.set_grasp_force(force), "grasp force")
                self._checked_write(
                    self.delto_client.set_grasp_hold_positions(hold_mask),
                    "grasp hold mask")
                self._checked_write(
                    self.delto_client.set_grasp_action(True), "grasp action")
            self.get_logger().info(
                f"DG-3F-M holding force active: mode={mode} "
                f"force={max(0, min(200, force)) * 0.1:.1f}N "
                f"free_motors={[i + 1 for i, held in enumerate(hold_mask) if not held]}")
        except Exception as e:
            self.get_logger().error(f"Failed to apply holding control: {e}")
        
    # Publish joint state
    def read_joint_callback(self):
        if self.reconnect_attempts > 1:
            return
        
        if self.is_connected == False:
            # self.get_logger().error("(read_joint_callback) Connection lost")
            return
        try:
            if self.is_dummy:
                # Preserve the dummy path exactly: get_position() echoes the
                # commanded state back, which get_state() cannot do.
                position_tmp = self.get_position()
                self.current_joint_state = [
                    float(self._deg2rad(x)) for x in position_tmp]
                self.raw_current_state = self.delto_client.get_current_raw()
                self.current_joint_effort = [
                    float(x * 0.001) for x in self.raw_current_state]
                self.current_joint_velocity = [
                    float(x) for x in self.delto_client.get_velocity()]
            else:
                # One Modbus round trip for all three, instead of three. Beyond
                # the 3x saving, position/current/velocity now come from the
                # SAME instant -- previously current was one round trip newer
                # than the position it was reported with, which matters when
                # deriving contact from motor current.
                position_deg, raw_current, velocity = (
                    self.delto_client.get_state())
                self.current_joint_state = [
                    float(self._deg2rad(x)) for x in position_deg]
                self.raw_current_state = raw_current
                self.current_joint_effort = [
                    float(x * 0.001) for x in raw_current]
                self.current_joint_velocity = [float(v) for v in velocity]
            self._joint_state_stamp = self.get_clock().now()
        except Exception as e:
            self.get_logger().error("Failed to read joint state: {0}".format(e))
            self.is_connected = False
            return
        
    def write_register_callback(self, msg):
        
        if self.is_connected == False:
            self.get_logger().error("Connection lost")
            return
        
        try:
            self.delto_client.write_registers(msg.data[0], msg.data[1:])
        except Exception as e:
            self.get_logger().error("Failed to write register: {0}".format(e))
            self.is_connected = False
            return
        
    def joint_state_publisher(self):

        joint_state_msg = JointState()
        # Stamp when the data was READ, not when this timer happened to fire.
        # publish_rate (100Hz) is 5x feedback_read_rate (20Hz), so most messages
        # repeat the previous sample; stamping "now" made every repeat claim to
        # be a fresh measurement, and any consumer differentiating position or
        # timing a contact onset was working from that false rate.
        joint_state_msg.header.stamp = (
            self._joint_state_stamp.to_msg()
            if self._joint_state_stamp is not None
            else self.get_clock().now().to_msg())

        joint_state_msg.name = ['F1M1', 'F1M2', 'F1M3', 'F1M4',
                                'F2M1', 'F2M2', 'F2M3', 'F2M4',
                                'F3M1', 'F3M2', 'F3M3', 'F3M4']

        joint_state_msg.position = self.current_joint_state
        joint_state_msg.velocity = self.current_joint_velocity
        joint_state_msg.effort = self.current_joint_effort
        self.joint_state_feedback.positions = joint_state_msg.position
        self.joint_state_pub.publish(joint_state_msg)
    # Get current position
    def get_position(self):

        if self.is_dummy:
            return self.current_joint_state

        if self.is_connected == False:
            self.get_logger().error("(get_position) Connection lost")
            return
        
        status = self.delto_client.get_position()
        return status

    def set_position(self, position):

        if self.is_dummy:
            self.current_joint_state = position
            return
        
        if self.is_connected == False:
            self.get_logger().error("Connection lost")
            return
        
        try:
            self.delto_client.set_position(position)
            
        except Exception as e:
            self.get_logger().error("(set_position) Failed to set position: {0}".format(e))
            self.is_connected = False
            return

    def set_motion_step(self, step):

        if self.is_dummy:
            return
        
        if self.is_connected == False:
            self.get_logger().error("Connection lost")
            return
        
        try:
            self.delto_client.set_step(step)
        except Exception as e:
            self.get_logger().error("Failed to set motion step: {0}".format(e))
            self.is_connected = False
            return

    def grasp_mode_callback(self, mode: Int32):
        
        if self.is_connected == False:
            self.get_logger().error("Connection lost")
            return
        
        try:
            self.delto_client.grasp_mode(mode.data)
        except Exception as e:
            self.get_logger().error("Failed to set grasp mode: {0}".format(e))
            self.is_connected = False
            return

    def grasp_callback(self, grasp: Bool):
        
        if self.is_connected == False:
            self.get_logger().error("Connection lost")
            return
        
        try:
            self.delto_client.grasp(grasp.data)
        except Exception as e:
            self.get_logger().error("Failed to grasp: {0}".format(e))
            self.is_connected = False
            return

    def timer_callback(self):
        
        if not self.is_connected and self.reconnect_attempts <= 1:
            self.get_logger().error("Connection lost")
            return
        
        if self.is_connected:
            self.joint_state_publisher()


    def goal_callback(self, goal_request):
        self.get_logger().info('Received goal')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel')
        return CancelResponse.ACCEPT

    # FollowJointTrajectory action server callback
    async def execute_callback(self, goal_handle):

        print('FollowJointTrajectory callback...')
        print(goal_handle.request.trajectory)
        goal = goal_handle.request.trajectory.points.copy()
        
        # download planned path from ros moveit
        self.joint_state_list = []

        if goal:
            self.joint_state_list = [p.positions for p in goal]
        else:
            self.stop_motion()
            return

        if self.joint_state_list:

            # print("joint_state_list: ", self.joint_state_list)
            # add first and last point to the trajectory
            new_array = [self.joint_state_list[0]]

            middle_point = self.joint_state_list[int(
                len(self.joint_state_list)/2)]
            new_array.append(middle_point)
            new_array.append(self.joint_state_list[-1])

            # set motion step by trajectory points
            self.set_motion_step(1)

            new_array = [[self._rad2deg(joints) for joints in subset]
                         for subset in new_array]
            # array = self.joint_state_list[-1]
            # array = [joint for joint in array]
            # print(new_array[-1])
            self._waypointMove(new_array, 0.5)
            # self.set_position(new_array[-1])

            # print("_waypointMoveSuccess")

            result = FollowJointTrajectory.Result()
            feedback_msg = FollowJointTrajectory.Feedback()

            feedback_msg.desired.positions = self.joint_state_feedback.positions
            feedback_msg.actual.positions = self.joint_state_feedback.positions

            goal_handle.publish_feedback(feedback_msg)

            time.sleep(0.01)

        goal_handle.succeed()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL

        print("success")
        return result

    def target_joint_callback(self, msg):

        if len(msg.data) != 12:
            self.get_logger().error("Invalid target joint state")
            return

        if self.is_connected == False:
            self.get_logger().error("Connection lost")
            return
        
        target_deg = [self._rad2deg(x) for x in msg.data]
        self.target_joint_state = target_deg
        try:
            response = self.delto_client.set_position(self.target_joint_state)
            if hasattr(response, "isError") and response.isError():
                self.get_logger().error(f"DG-3F-M target write failed: {response}")
        except Exception as e:
            self.get_logger().error("Failed to set target joint state: {0}".format(e))
            self.is_connected = False
            return

    def fixed_joint_callback(self, msg):
        if len(msg.data) != 12:
            self.get_logger().error("Invalid fixed joint state")
            return
    
        if self.is_connected == False:
            self.get_logger().error("Connection lost")
            return
    
    # Int16MultiArray의 data를 리스트로 변환
        self.fixed_joint_state = list(msg.data)
        print(self.fixed_joint_state)
    
        try:
            self.delto_client.fix_position(self.fixed_joint_state)
        except Exception as e:
            self.get_logger().error("Failed to set fixed joint state: {0}".format(e))
            self.is_connected = False
            return
    
    def waypointMove(self, waypointList, threshold):
        self.stop_thread = False
        self.waypoint_thread = threading.Thread(
            target=self._waypointMove, args=(waypointList, threshold))
        self.waypoint_thread.start()

    def stop_motion(self):
        
        if self.is_connected == False:
            self.get_logger().error("Connection lost")
            return
        
        self.delto_client.set_position(self.delto_client.get_position())
        self.stop_thread = True

    def _waypointMove(self, waypointList, threshold=0.3):

        self.stop_thread = False
        i = 0
        move_flag = False
        print("_waypointMove1")
        print("self.stop_thread: " + str(self.stop_thread))
        first_input_flag = True

        first_current_position = [0.0]*12  # float
        current_position = [0.0] * 12

        while not self.stop_thread:

            if (move_flag == False):
                # move to the next waypoint
                self.delto_client.set_position(waypointList[i])
                move_flag = True
                first_input_flag = True

            current_position = self.delto_client.get_position()

            if (first_input_flag):
                first_current_position = current_position
                first_input_flag = False

            error = []
            for a, b, c in zip(current_position, waypointList[i], first_current_position):

                if (b-c) < 0.0001:
                    error.append(0.0)
                else:
                    error.append(abs(((a-b)/(b-c))))

            if all(e < threshold for e in error):
                # arrived at the waypoint
                move_flag = False
                i += 1

            if (i >= len(waypointList)):
                self.stop_thread = True
                return

    def _deg2rad(self, deg):
        return deg * math.pi / 180.0

    def _rad2deg(self, rad):
        return rad * 180.0 / math.pi
        
        
    def convert_to_signed(self, value):
        """ Convert unsigned 16-bit integer to signed 16-bit integer """
        return value if value < 32768 else value - 65536

    def estimate_force(self, current, finger=0, motor=None):
        """Motor current -> contact channel for one finger.

        Not a calibrated force in N: force_scale is a display gain, not a
        transfer function to Newtons. See publish_force_data's docstring.

        `current - baseline` removes that finger's no-load current, so the
        result is deviation from unloaded and the deadband sits around zero
        load. DeltoGripperController additionally captures its own live
        baseline and thresholds on change from it, which absorbs whatever
        offset error remains here -- so a wrong baseline degrades resolution
        rather than breaking contact detection.
        """
        baselines = self.get_parameter('force_baseline').value
        if motor is not None and len(baselines) >= 12:
            baseline = float(baselines[motor])          # per-motor calibration
        elif finger < len(baselines):
            baseline = float(baselines[finger])         # per-finger
        else:
            baseline = float(baselines[0])
        k = float(self.get_parameter('force_scale').value)
        deadband = float(self.get_parameter('force_deadband').value)

        raw = current - baseline

        # Clamp very small noise
        if abs(raw) < deadband:
            raw = 0.0

        return round(k * raw, 3)
        
        
    def publish_force_data(self):
        """Publish three motor-current-derived contact channels.

        These values are estimates, not calibrated fingertip forces in N.
        Full per-motor current feedback is also available in the ``effort``
        field of ``/gripper/joint_states``.
        """
        if not self.is_connected:
            self.get_logger().error("Connection lost")
            return
        
        try:
            raw_current = list(self.raw_current_state)
            if not any(raw_current):
                raw_current = self.delto_client.get_current_raw()
                self.raw_current_state = list(raw_current)
                self.current_joint_effort = [float(x * 0.001) for x in raw_current]

            mode = str(self.get_parameter('force_motor_mode').value).strip().lower()
            if mode == 'max_per_finger':
                # Whichever of this finger's four motors is furthest from its
                # baseline -- the load can sit on any of them.
                estimated_forces = []
                for finger in range(3):
                    vals = [
                        self.estimate_force(raw_current[i], finger=finger, motor=i)
                        for i in range(finger * 4, finger * 4 + 4)
                        if i < len(raw_current)]
                    estimated_forces.append(max(vals, key=abs) if vals else 0.0)
            else:
                indices = [int(v) for v in
                           self.get_parameter('force_motor_indices').value]
                estimated_forces = [
                    self.estimate_force(raw_current[idx], finger=i, motor=idx)
                    for i, idx in enumerate(indices)
                    if 0 <= idx < len(raw_current)]

            # Publish as a Float32MultiArray ROS message
            force_msg = Float32MultiArray()
            force_msg.data = estimated_forces
            self.force_pub.publish(force_msg)

            #self.get_logger().info(f"Published Force Data: {estimated_forces}")

        except Exception as e:
            self.get_logger().error(f"Failed to read and publish force data: {e}")


def main(args=None):
    rclpy.init(args=args)

    delto_driver = DeltoROSDriver()
    connect = delto_driver.connect()

    if connect == False:
        delto_driver.get_logger().error("Init network connection failed.")

    time.sleep(0.1)
    delto_driver.get_logger().info("delto_driver initialized")

    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(delto_driver)
    executor.spin()
    executor.shutdown()


if __name__ == '__main__':
    main()
