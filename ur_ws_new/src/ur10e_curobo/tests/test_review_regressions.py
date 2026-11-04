"""Hardware-free regressions; extract ROS-bound functions without importing drivers.

Run: python -m unittest discover -s ur_ws_new/src/ur10e_curobo/tests -v
"""
import ast
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import threading
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'ur10e_curobo'
sys.path.insert(0, str(ROOT.parent))
from ur10e_curobo.observations import FruitObservation


def load(path, name, env, owner=None):
    env.setdefault("threading", threading)
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body if owner is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner).body
    node = next(n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    if isinstance(node, ast.FunctionDef):
        node.body = [n for n in node.body if not (isinstance(n, ast.ImportFrom) and n.module == 'motions')]
    mod = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), str(ROOT / path), 'exec'), env)
    return env[name]


class Clock:
    def __init__(self):
        self.t = 1.0
        self.callback = lambda: None
    def monotonic(self): return self.t
    def time(self): return self.t
    def sleep(self, dt):
        self.t += dt
        self.callback()
    def now(self): return types.SimpleNamespace(nanoseconds=int(self.t*1e9))


def logger():
    return types.SimpleNamespace(**{k: lambda *args: None for k in ('info', 'warn', 'error', 'debug')})


class JointStateTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.fn = load('managers/state_manager.py', '_joint_state_cb', dict(math=math,time=self.clock), 'StateManager')
        self.state = types.SimpleNamespace(_js_index_cache=None, _config=types.SimpleNamespace(joint_order=['a','b']), _state_lock=threading.Lock())
        self.send(['b','a'], [2,1], [.2,.1])
    def send(self, names, positions, velocities):
        self.fn(self.state, types.SimpleNamespace(name=names,position=positions,velocity=velocities))
    def test_finger_only_preserves_arm(self):
        self.send(['finger'],[5],[1])
        self.assertEqual(self.state._current_joint_positions,[1,2])
        self.assertEqual(self.state._current_joint_velocities,[.1,.2])
    def test_partial_preserves_arm(self):
        self.send(['a'],[9],[9])
        self.assertEqual(self.state._current_joint_positions,[1,2])
    def test_malformed_or_nonfinite_preserves_arm(self):
        for positions in ([5], [math.nan,5], [math.inf,5]):
            self.send(['a','b'],positions,[1,1])
            self.assertEqual(self.state._current_joint_positions,[1,2])
    def test_missing_velocity_is_unknown(self):
        self.send(['a','b'],[3,4],[])
        self.assertIsNone(self.state._current_joint_velocities)
    def test_duplicate_names_rejected(self):
        self.send(['a','a','b'],[8,8,8],[0,0,0])
        self.assertEqual(self.state._current_joint_positions,[1,2])


class WaitTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.stops = []
        self.fn = load('utils.py','wait_until_xyz',dict(time=self.clock,math=math,publish_stop_trajectory=lambda n:self.stops.append(n)))
        self.node = types.SimpleNamespace(cfg=types.SimpleNamespace(planner=types.SimpleNamespace(concise_console_logs=True)),
            get_logger=logger, running=True, stop_requested=False, current_joint_velocities=[0]*6,
            get_end_effector_pose=lambda:[0,0,0,1,0,0,0])
    def test_stationary_feedback_has_no_added_dwell(self):
        self.assertTrue(self.fn(self.node,[0,0,0],timeout=.5))
        self.assertEqual(self.clock.t,1.0)
    def test_short_move_can_reach_without_one_centimeter_departure(self):
        poses=iter([[.009,0,0,1,0,0,0]])
        self.node.get_end_effector_pose=lambda:next(poses,[0,0,0,1,0,0,0])
        self.assertTrue(self.fn(self.node,[0,0,0],tol=.004,timeout=.5))
    def test_moving_at_target_does_not_succeed(self):
        self.node.current_joint_velocities=[.5]*6
        self.assertFalse(self.fn(self.node,[0,0,0],timeout=.3))
    def test_unknown_velocity_does_not_succeed(self):
        self.node.current_joint_velocities=None
        self.assertFalse(self.fn(self.node,[0,0,0],timeout=.3))
    def test_stop_remains_latched(self):
        self.node.stop_requested=True
        self.assertFalse(self.fn(self.node,[0,0,0]))
        self.assertTrue(self.node.stop_requested)
        self.assertEqual(len(self.stops),1)
    def test_shutdown_does_not_succeed(self):
        self.node.running=False
        self.assertFalse(self.fn(self.node,[0,0,0]))
    def test_exception_does_not_succeed(self):
        def fail(): raise RuntimeError('FK unavailable')
        self.node.get_end_effector_pose=fail
        self.assertFalse(self.fn(self.node,[0,0,0]))
    def test_near_target_at_timeout_does_not_succeed(self):
        self.node.get_end_effector_pose=lambda:[.007,0,0,1,0,0,0]
        self.assertFalse(self.fn(self.node,[0,0,0],tol=.005,timeout=.2))
    def test_wrong_orientation_does_not_succeed(self):
        self.assertFalse(self.fn(self.node,[0,0,0],target_quat=[0,1,0,0],timeout=.2))
    def test_stale_feedback_does_not_succeed(self):
        self.node._state_mgr=types.SimpleNamespace(_joint_state_received_at=0)
        self.assertFalse(self.fn(self.node,[0,0,0],timeout=.3))
    def test_stop_for_found_goal_ends_after_settling_not_stall(self):
        # Arm stopped 3.8 cm short because a goal was found mid-scan.
        self.node.get_end_effector_pose=lambda:[.038,0,0,1,0,0,0]
        self.node.current_joint_velocities=[.2]*6
        def tick():
            if self.clock.t>1.2 and not hasattr(self.node,'_goal_stop_t'):
                self.node._goal_stop_t=self.clock.t          # goal callback stops the arm
            if self.clock.t>1.5:
                self.node.current_joint_velocities=[0]*6     # settled
        self.clock.callback=tick
        self.assertFalse(self.fn(self.node,[0,0,0],timeout=10))
        self.assertLess(self.clock.t,1.7)                    # not the 2 s stall timeout
        self.assertEqual(getattr(self.node,'_goal_stall_count',0),0)
    def test_goal_stop_before_this_wait_is_ignored(self):
        self.node._goal_stop_t=0.5                           # from an earlier motion
        self.assertTrue(self.fn(self.node,[0,0,0],timeout=.5))


class TrackerTests(unittest.TestCase):
    def setUp(self):
        cls=load('vision/tracking.py','FruitTracker',dict(MIN_FRAMES_TO_SHOW=2,MAX_FRAMES_TO_KEEP=2))
        self.tracker=cls(); self.d={'fruit_id':1}
    def counts(self, frames): return [len(self.tracker.stabilize_detections(f)) for f in frames]
    def test_single_detection_never_promoted_on_miss(self):
        self.assertEqual(self.counts([[self.d],[]]),[0,0])
    def test_expired_track_removed_before_output(self):
        self.assertEqual(self.counts([[self.d],[self.d],[],[],[]]),[0,1,1,1,0])
    def test_confirmation_is_consecutive(self):
        self.assertEqual(self.counts([[self.d],[],[self.d],[self.d]]),[0,0,0,1])


class CalibrationTests(unittest.TestCase):
    def test_missing_tf_or_too_few_stations_inconclusive(self):
        fn=load('vision/verify_hand_eye.py','report',dict(math=math,MIN_ROTATION_SPREAD_DEG=20,MIN_TRANSLATION_SPREAD_MM=50,PASS_MM=5,MARGINAL_MM=10))
        for goals,poses in [([([0,0,0],0)]*3,[None]*3),([],[])]:
            with contextlib.redirect_stdout(io.StringIO()): verdict=fn(goals,poses)
            self.assertEqual(verdict,2)


class ObservationTests(unittest.TestCase):
    def payload(self):
        return dict(frame_id='base_link',stamp_ns=1_000_000_000,sequence=1,poses=[[0,0,0]],qualities=[[1,.001,.9,1,1]])
    def test_roundtrip_and_age(self):
        packet=FruitObservation.decode(json.dumps(self.payload()))
        self.assertTrue(packet.fresh(1_100_000_000))
        self.assertFalse(packet.fresh(2_000_000_000))
        self.assertFalse(packet.fresh(0))
    def test_rejects_unpaired_and_nonfinite_data(self):
        for key,value in [('qualities',[]),('poses',[[math.nan,0,0]]),('frame_id','camera')]:
            d=self.payload();d[key]=value
            with self.assertRaises(ValueError): FruitObservation.decode(json.dumps(d))


class ReacquireTests(unittest.TestCase):
    def setUp(self):
        self.clock=Clock();self.env=dict(math=math,time=self.clock)
        for name in ['_reacquire_candidate_quality','_reacquire_quality_ok','_reacquire_goal_pose_impl','active_closure_offset']:
            load('goals.py',name,self.env)
        self.node=types.SimpleNamespace(cfg=types.SimpleNamespace(planner=types.SimpleNamespace()),get_clock=lambda:self.clock,get_logger=logger,stop_requested=False,running=True,_fruit_observation=None)
    def publish(self, pose):
        self.node._fruit_observation=FruitObservation(int(self.clock.t*1e9),1,(tuple(pose),),((1,.001,.9,1,1),))
    def run_reacquire(self,needed=1):
        return self.env['_reacquire_goal_pose_impl'](self.node,[0,0,0],timeout=.15,depth_settle_s=0,stable_needed=needed)
    def test_old_packet_cannot_refine_seed(self):
        self.publish([0,0,0])
        self.assertIsNone(self.run_reacquire())
    def test_identical_positions_in_new_frames_count(self):
        self.clock.callback=lambda:self.publish([.001,0,0])
        self.assertEqual(self.run_reacquire(2),(.001,0,0))
    def test_same_height_far_neighbor_rejected(self):
        self.clock.callback=lambda:self.publish([.5,0,0])
        self.assertIsNone(self.run_reacquire())
    def test_same_xy_far_depth_rejected(self):
        self.clock.callback=lambda:self.publish([0,0,.5])
        self.assertIsNone(self.run_reacquire())
    def test_quality_snapshot_does_not_follow_later_frame(self):
        self.publish([0,0,0]); snapshot=self.node._fruit_observation
        self.node._fruit_observation=FruitObservation(2,2,((0,0,0),),((0,1,0,0,0),))
        self.assertTrue(self.env['_reacquire_quality_ok'](self.node,[0,0,0],snapshot)[0])
        self.assertFalse(self.env['_reacquire_quality_ok'](self.node,[0,0,0])[0])
    def test_envelop_offset_selected(self):
        p=self.node.cfg.planner;p.closure_center_offset_tcp_m=[1,2,3];p.envelop_closure_center_offset_tcp_m=[4,5,6]
        self.node.active_goal_grasp_mode='ENVELOP'
        self.assertEqual(self.env['active_closure_offset'](self.node),[4,5,6])
        self.node.active_goal_grasp_mode='NORMAL'
        self.assertEqual(self.env['active_closure_offset'](self.node),[1,2,3])


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        class Trajectory:
            def __init__(self): self.points=[]
        class Point:
            def __init__(self): self.time_from_start=types.SimpleNamespace(sec=0,nanosec=0);self.accelerations=[]
        self.fn=load('utils.py','build_trajectory',dict(math=math,np=np,JointTrajectory=Trajectory,JointTrajectoryPoint=Point))
    def test_interpolated_derivatives_obey_limits(self):
        rng=np.random.default_rng(14)
        for include_acc in [False,True]:
            for q in [np.array([[0.],[1.]]),rng.normal(size=(8,3))]:
                traj=self.fn(['j']*q.shape[1],q,dt=.01,max_vel=.5,max_acc=1,include_acc=include_acc)
                t=lambda p:p.time_from_start.sec+p.time_from_start.nanosec/1e9
                for left,right in zip(traj.points,traj.points[1:]):
                    dt=t(right)-t(left)
                    # Independently solve Hermite polynomial constraints and sample derivatives.
                    for j in range(q.shape[1]):
                        c0=left.positions[j];c1=left.velocities[j]*dt
                        if include_acc:
                            c2=left.accelerations[j]*dt*dt/2
                            rhs=[right.positions[j]-c0-c1-c2,right.velocities[j]*dt-c1-2*c2,right.accelerations[j]*dt*dt-2*c2]
                            tail=np.linalg.solve([[1,1,1],[3,4,5],[6,12,20]],rhs)
                            coeff=np.r_[c0,c1,c2,tail]
                        else:
                            tail=np.linalg.solve([[1,1],[2,3]],[right.positions[j]-c0-c1,right.velocities[j]*dt-c1])
                            coeff=np.r_[c0,c1,tail]
                        u=np.linspace(0,1,101)
                        v=np.polynomial.polynomial.polyval(u,np.polynomial.polynomial.polyder(coeff))/dt
                        a=np.polynomial.polynomial.polyval(u,np.polynomial.polynomial.polyder(coeff,2))/dt**2
                        self.assertLessEqual(np.abs(v).max(),.5+1e-8)
                        self.assertLessEqual(np.abs(a).max(),1+1e-8)
    def test_degenerate_lengths_survive_end_ramps(self):
        # The end-interval ramp indexes a per-interval list; a single-point
        # trajectory has no intervals, which used to raise IndexError from
        # blend_motion's one-point path.
        for n in range(0, 9):
            q=[[.1*k]*3 for k in range(n)]
            for ramp in (0, 8, 15):
                traj=self.fn(['j']*3,q,dt=.01,max_vel=.5,max_acc=1,ramp_points=ramp)
                self.assertEqual(len(traj.points),n)
                t=[p.time_from_start.sec+p.time_from_start.nanosec/1e9 for p in traj.points]
                self.assertTrue(all(b>a for a,b in zip(t,t[1:])),(n,ramp,t))
                self.assertTrue(all(x>0 for x in t),(n,ramp,t))

    def test_cancel_discards_partial_trajectory(self):
        calls=iter([False,True])
        self.assertFalse(self.fn(['j'],[[0],[1]],stop_flag=lambda:next(calls)).points)
    def test_invalid_input_rejected(self):
        for states in [[[math.nan]], [[1,2]]]:
            with self.assertRaises(ValueError):self.fn(['j'],states)


class AssociationTests(unittest.TestCase):
    def setUp(self):
        env=dict(math=math,time=lambda:1.0,MIN_FRAMES_TO_SHOW=2,MAX_FRAMES_TO_KEEP=2,
                 FRUIT_ID_TIMEOUT=10,MAX_FRUIT_ATTEMPTS=3,
                 # Mirror tracking.py's module-level constants; load() execs the
                 # class alone, so anything it closes over has to be supplied.
                 ASSOCIATION_RADIUS_M=0.025,AMBIGUITY_MARGIN_M=0.003)
        self.tracker=load('vision/tracking.py','FruitTracker',env)()
    def test_rounding_boundary_retains_identity(self):
        self.tracker.begin_frame(); first=self.tracker.associate_fruit([.104,0,0])
        self.tracker.begin_frame(); second=self.tracker.associate_fruit([.106,0,0])
        self.assertEqual(first,second)
    def test_same_frame_cannot_assign_id_twice(self):
        self.tracker.begin_frame(); first=self.tracker.associate_fruit([0,0,0])
        second=self.tracker.associate_fruit([.001,0,0])
        self.assertNotEqual(first,second)
    def test_attempts_follow_base_frame_position(self):
        self.tracker.begin_frame(); fid=self.tracker.associate_fruit([.1,.2,.3])
        with contextlib.redirect_stdout(io.StringIO()):
            self.tracker.increment_fruit_attempt([.102,.2,.3])
        self.assertEqual(self.tracker.fruit_id_registry[fid]['attempt_count'],1)


class HandoffTests(unittest.TestCase):
    def test_queued_image_is_busy_immediately(self):
        fn=load('vision/yolo_thread.py','set_image',dict(time=lambda:1), 'YoloThread')
        worker=types.SimpleNamespace(lock=threading.Lock(),paused=False,run_event=threading.Event(),
            idle_event=threading.Event(),_next_frame_id=0,_dropped_frames=0)
        worker.idle_event.set()
        self.assertTrue(fn(worker,np.zeros((2,2,3)),capture_time=2))
        self.assertFalse(worker.idle_event.is_set())
        self.assertFalse(fn(worker,np.zeros((2,2,3))))
    def test_old_pause_ack_is_ignored(self):
        # Match the actual motion-node class name without importing ROS.
        tree=ast.parse((ROOT/'node.py').read_text())
        owner=next(n.name for n in tree.body if isinstance(n,ast.ClassDef)
                   and any(isinstance(f,ast.FunctionDef) and f.name=='_vision_mode_state_cb' for f in n.body))
        fn=load('node.py','_vision_mode_state_cb',{},owner)
        node=types.SimpleNamespace(_vision_pause_request='new',_vision_paused_event=threading.Event())
        fn(node,types.SimpleNamespace(data='paused:old'))
        self.assertFalse(node._vision_paused_event.is_set())
        fn(node,types.SimpleNamespace(data='paused:new'))
        self.assertTrue(node._vision_paused_event.is_set())


class JointCompletionTests(unittest.TestCase):
    def test_joint_target_requires_stationary_feedback(self):
        clock=Clock()
        fn=load('motions.py','_wait_for_joint_target',dict(time=clock,math=math))
        node=types.SimpleNamespace(current_joint_positions=[0,0],current_joint_velocities=[.2,0],running=True,stop_requested=False)
        self.assertFalse(fn(node,[0,0],timeout=.2))
        node.current_joint_velocities=[0,0]
        self.assertTrue(fn(node,[0,0],timeout=.2))
    def test_preplan_rejects_moving_start_without_publishing(self):
        fn=load('motions.py','execute_preplan',dict(math=math))
        node=types.SimpleNamespace(stop_requested=False,current_joint_positions=[0],current_joint_velocities=[.2])
        self.assertFalse(fn(node,types.SimpleNamespace(points=[object()]),[[0],[1]],'TEST'))


class TipFreshnessTests(unittest.TestCase):
    def test_republished_tip_cannot_gain_confirmation(self):
        clock=Clock()
        fn=load('managers/state_manager.py','_date_tip_cb',dict(math=math,time=clock),'StateManager')
        state=types.SimpleNamespace(_node=types.SimpleNamespace(get_clock=lambda:clock),_date_tip_history=[])
        msg=types.SimpleNamespace(header=types.SimpleNamespace(frame_id='base_link',stamp=types.SimpleNamespace(sec=1,nanosec=0)),point=types.SimpleNamespace(x=0,y=0,z=0))
        for _ in range(3):fn(state,msg)
        self.assertEqual(len(state._date_tip_history),1)
        self.assertFalse(state.date_tip_stable)
    def test_old_tip_is_not_refreshed_on_receipt(self):
        clock=Clock();clock.t=2
        fn=load('managers/state_manager.py','_date_tip_cb',dict(math=math,time=clock),'StateManager')
        state=types.SimpleNamespace(_node=types.SimpleNamespace(get_clock=lambda:clock),_date_tip_history=[])
        msg=types.SimpleNamespace(header=types.SimpleNamespace(frame_id='base_link',stamp=types.SimpleNamespace(sec=1,nanosec=0)),point=types.SimpleNamespace(x=0,y=0,z=0))
        fn(state,msg)
        self.assertFalse(state._date_tip_history)


class FinalAttemptBudgetTests(unittest.TestCase):
    def run_monitor(self, candidate, *, step=1.0, frames=6, live=True):
        clock = [10.0]
        calls = []
        planner = types.SimpleNamespace(final_inflight_apply=True,
            final_inflight_max_attempts=2, final_inflight_min_interval_s=.25)
        observation = types.SimpleNamespace(poses=[candidate], fresh=lambda stamp: True)
        node = types.SimpleNamespace(cfg=types.SimpleNamespace(planner=planner),
            all_fruits_seq=0, _fruit_observation=observation, _level_final_z=0.,
            _final_live_wait_xyz=[0., 0., 0.] if live else None,
            get_clock=lambda: types.SimpleNamespace(now=lambda: types.SimpleNamespace(nanoseconds=1)))
        class Stop:
            count = 0
            def is_set(self): return self.count > frames
            def wait(self, timeout):
                self.count += 1
                clock[0] += step
                node.all_fruits_seq += 1
                return self.is_set()
        report = dict(frames=0, rejected=0, no_candidate=0, matched=0,
                      below_min=0, above_max=0, usable=0, applied=0,
                      last_apply_t=0., applied_total=0.)
        def apply(*args):
            calls.append(clock[0])
            return False
        fn = load('goals.py', '_final_inflight_monitor', dict(math=math,
            time=types.SimpleNamespace(time=lambda: clock[0], monotonic=lambda: clock[0],
                                       perf_counter=lambda: clock[0]),
            _reacquire_quality_ok=lambda *args: (True, ''),
            _apply_final_inflight_correction=apply))
        fn(node, [0., 0., 0.], Stop(), report)
        return calls, report

    def test_height_projection_skips_negligible_executable_delta(self):
        calls, report = self.run_monitor([.001, 0., .01])
        self.assertEqual(calls, [])
        self.assertEqual(report['below_min_projected'], 6)

    def test_failed_plans_consume_attempt_budget(self):
        calls, report = self.run_monitor([.006, 0., 0.])
        self.assertEqual(len(calls), 2)
        self.assertEqual(report['attempts'], 2)
        self.assertEqual(report['applied'], 0)

    def test_failed_attempts_are_spaced_and_nonlive_target_is_skipped(self):
        calls, _ = self.run_monitor([.006, 0., 0.], step=.05, frames=10)
        self.assertEqual(len(calls), 2)
        self.assertGreaterEqual(calls[1] - calls[0], .25)
        calls, _ = self.run_monitor([.006, 0., 0.], live=False)
        self.assertEqual(calls, [])


class GripperOpenFeedbackTests(unittest.TestCase):
    def test_open_requires_fresh_complete_matching_feedback(self):
        fn = load('delto_gripper_controller.py', 'is_at_open_position',
                  dict(math=math, time=types.SimpleNamespace(time=lambda: 10.0)),
                  owner='DeltoGripperController')
        obj = types.SimpleNamespace(open_position=[0.] * 12,
                                    actual_joint_position=[0.] * 12,
                                    actual_joint_stamp=9.9)
        self.assertTrue(fn(obj))
        obj.actual_joint_stamp = 9.0
        self.assertFalse(fn(obj))
        obj.actual_joint_stamp = 9.9
        obj.actual_joint_position[0] = .1
        self.assertFalse(fn(obj))
        obj.actual_joint_position = [0.] * 11
        self.assertFalse(fn(obj))
        obj.actual_joint_position = [float('nan')] * 12
        self.assertFalse(fn(obj))
        obj.actual_joint_position = None
        self.assertFalse(fn(obj))


class SpeedRegressionTests(unittest.TestCase):
    def test_cubic_uses_exact_peak_instead_of_control_polygon_bound(self):
        class Trajectory:
            def __init__(self): self.points=[]
        class Point:
            def __init__(self): self.time_from_start=types.SimpleNamespace(sec=0,nanosec=0)
        fn=load('utils.py','build_trajectory',dict(math=math,np=np,JointTrajectory=Trajectory,JointTrajectoryPoint=Point))
        trajectory=fn(['j'],[[0.],[1.]],dt=.01,max_vel=.5,max_acc=100)
        stamp=trajectory.points[0].time_from_start
        # Rest-to-rest cubic's actual peak speed is 1.5/dt, not 3/dt.
        self.assertAlmostEqual(stamp.sec+stamp.nanosec/1e9,3.0,places=8)

    def test_fk_reuses_identical_state_across_facades(self):
        import torch
        calls=[]
        class Kin:
            def forward(self,q):
                calls.append(q)
                return torch.zeros((1,3)),torch.tensor([[1.,0.,0.,0.]]),None,None,None,None,None
        model=[Kin()]
        env=dict(torch=types.SimpleNamespace(device=torch.device,tensor=torch.tensor,
            float32=torch.float32,no_grad=torch.no_grad,cat=torch.cat,
            cuda=types.SimpleNamespace(is_available=lambda:False)),
            _fk_lock=threading.RLock(),_get_kin_model=lambda n:model[0],_is_planning_active=lambda n:False)
        fn=load('fk.py','get_end_effector_pose',env)
        owner=types.SimpleNamespace()
        def facade(joints):return types.SimpleNamespace(current_joint_positions=joints,_fk_cache_owner=owner,get_logger=logger)
        for _ in range(100):
            result=fn(facade([0]*6))
        self.assertEqual(len(calls),1)
        result[0]=99
        self.assertEqual(fn(facade([0]*6))[0],0)
        fn(facade([.01]*6))
        self.assertEqual(len(calls),2)
        model[0]=Kin()
        fn(facade([.01]*6))
        self.assertEqual(len(calls),3)


class BestTargetPublicationTests(unittest.TestCase):
    def test_tip_and_goal_preserve_acquisition_stamp(self):
        from collections import deque
        from geometry_msgs.msg import PointStamped, PoseStamped
        from std_msgs.msg import Float32, Float32MultiArray, String
        import json
        tree=ast.parse((ROOT/'vision/node.py').read_text())
        owner=next(n.name for n in tree.body if isinstance(n,ast.ClassDef)
                   and any(isinstance(f,ast.FunctionDef) and f.name=='_process_best_target' for f in n.body))
        env=dict(np=np,math=math,StdString=String,json=json,SHOW_FINGER_CONTACTS=False,Z_MAX=2.,
                 PointStamped=PointStamped,PoseStamped=PoseStamped,Float32=Float32,
                 Float32MultiArray=Float32MultiArray,time=lambda:99.,
                 estimate_fruit_radius=lambda *args,**kwargs:.015)
        fn=load('vision/node.py','_process_best_target',env,owner)
        for tip in (np.array([.01,.02,.03]), None, np.array([math.nan,0.,0.])):
            with self.subTest(tip=tip):
                tips=[]
                traces=[]
                point=PointStamped()
                point.header.frame_id='base_link'
                point.header.stamp.sec=12
                point.header.stamp.nanosec=345
                point.point.x=.1;point.point.y=.2;point.point.z=.3
                quiet=types.SimpleNamespace(publish=lambda msg:None)
                vision=types.SimpleNamespace(_cached_tf_base=(np.eye(3),np.array([.2,.3,.4])),
                    date_tip_pub=types.SimpleNamespace(publish=tips.append),prev_heat_point=None,
                    best_history=deque(maxlen=5),pub_lock=threading.Lock(),
                    depth_trace_pub=types.SimpleNamespace(publish=traces.append),
                    radius_pub=quiet,gap_info_pub=quiet,target_lock_active=False)
                target=dict(pt_base=point,best_point_3d=tip,fruit_id=1,bb=(1,1,4,4))
                fn(vision,[target],0,{'fx':500.})
                self.assertEqual(vision.latest_goal_msg.header.stamp,point.header.stamp)
                trace=json.loads(traces[0].data)
                self.assertEqual(trace["source_stamp_ns"],12000000345)
                self.assertEqual(trace["published_goal_base_xyz_m"],[.1,.2,.3])
                if tip is not None and np.isfinite(tip).all():
                    self.assertEqual(len(tips),1)
                    self.assertEqual(tips[0].header.stamp,point.header.stamp)
                    self.assertAlmostEqual(tips[0].point.x,.21)
                    self.assertAlmostEqual(tips[0].point.y,.32)
                    self.assertAlmostEqual(tips[0].point.z,.43)
                else:
                    self.assertFalse(tips)


class DetectionDisplayTests(unittest.TestCase):
    def setUp(self):
        tree=ast.parse((ROOT/'vision/node.py').read_text())
        function=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)
                      and n.name=='_enqueue_raw_viz_frame')
        self.clock=Clock();self.frames=[]
        self.env=dict(time=self.clock.time,_last_detection_result_t=1.0,
                      _enqueue_viz_frame=lambda *a:self.frames.append(a))
        module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
        exec(compile(module,'raw_display_gate','exec'),self.env)
        self.enqueue=self.env['_enqueue_raw_viz_frame']
    def test_rgb_frames_do_not_erase_recent_detections(self):
        for offset in [0,.033,.066,.1,.2,.4]:
            self.clock.t=1+offset
            self.enqueue()
        self.assertFalse(self.frames)
    def test_pause_startup_and_stall_show_live_video(self):
        self.enqueue(force=True)
        self.assertEqual(len(self.frames),1)
        self.env['_last_detection_result_t']=None
        self.enqueue()
        self.assertEqual(len(self.frames),2)
        self.env['_last_detection_result_t']=1.
        self.clock.t=1.6
        self.enqueue()
        self.assertEqual(len(self.frames),3)


class SpliceOntoMeasuredTest(unittest.TestCase):
    # In-flight retarget: the arm keeps moving while the leg is planned.
    def setUp(self):
        self.splice = load('goals.py', '_splice_onto_measured', {})
        self.states = [[0.01 * i, 0.0] for i in range(10)]

    def test_starts_at_measured_and_drops_passed_states(self):
        out, why = self.splice(self.states, [0.032, 0.001])
        self.assertIsNone(why)
        self.assertEqual(out[0], [0.032, 0.001])
        self.assertEqual(out[1], [0.04, 0.0])   # nearest was index 3; resume after it
        self.assertEqual(out[-1], self.states[-1])

    def test_rejects_jump_off_planned_line(self):
        self.assertEqual(self.splice(self.states, [0.03, 0.02])[1], "ARM_MOVED_DURING_BUILD")

    def test_rejects_when_nearly_done(self):
        self.assertEqual(self.splice(self.states, [0.08, 0.0])[1], "FINAL_NEARLY_DONE")

    def test_stored_path_is_the_one_flown(self):
        # Reverse replays this; after a retarget it must follow the new leg,
        # not jump back to the original line (2.21deg gap, 2026-10-08).
        join = load('goals.py', '_join_at', {})
        tail = [[0.032, 0.001], [0.04, 0.01], [0.05, 0.02]]
        flown = join(self.states, tail)
        self.assertEqual(flown, self.states[:3] + tail)


class CubicSpeedupTests(unittest.TestCase):
    def trajectory(self):
        def point(t, q):
            return types.SimpleNamespace(positions=[q], velocities=[0.], accelerations=[],
                time_from_start=types.SimpleNamespace(sec=t, nanosec=0))
        return types.SimpleNamespace(points=[point(1, 0.), point(3, .2)])

    def test_speedup_preserves_positions_and_obeys_cubic_acceleration(self):
        fn = load('utils.py', 'speed_up_cubic_trajectory', dict(np=np, math=math))
        original = self.trajectory()
        faster, scale = fn(original, 1.2, 1.5, 1.5)
        self.assertAlmostEqual(scale, 1.2)
        self.assertEqual([p.positions for p in faster.points], [[0.], [.2]])
        self.assertEqual(original.points[-1].time_from_start.sec, 3)
        def stamp(p): return p.time_from_start.sec + p.time_from_start.nanosec / 1e9
        h = stamp(faster.points[1]) - stamp(faster.points[0])
        self.assertLessEqual(6 * .2 / h**2, 1.5)
        # With acceleration near its original peak, only a bounded gain is legal.
        limited, scale = fn(original, 2., 1.5, .31)
        self.assertGreater(scale, 1.)
        self.assertLess(scale, 1.02)
        h = stamp(limited.points[1]) - stamp(limited.points[0])
        self.assertLessEqual(6 * .2 / h**2, .31)

    def test_no_margin_or_quintic_keeps_original(self):
        fn = load('utils.py', 'speed_up_cubic_trajectory', dict(np=np, math=math))
        original = self.trajectory()
        self.assertIs(fn(original, 1.2, .1, .1)[0], original)
        original.points[0].accelerations = [0.]
        self.assertIs(fn(original, 1.2, 10., 10.)[0], original)


class SubsampleStatesTest(unittest.TestCase):
    # Drives drop-off speed and reverse duration: fewer samples at fixed dt.
    def test_keeps_ends_and_count(self):
        sub = load('utils.py', 'subsample_states', {})
        states = [[float(i)] for i in range(281)]
        out = sub(states, 187)
        self.assertEqual(out[0], [0.0])
        self.assertEqual(out[-1], [280.0])
        self.assertEqual(len(out), 187)
        self.assertEqual(out, sorted(out))
        self.assertEqual(sub(states, 400), states)   # never adds samples


class NoLocalTorchImportTest(unittest.TestCase):
    # A function-local `import torch` makes torch local to the whole function, so
    # any earlier use raises UnboundLocalError (plan_and_send, 2026-10-08).
    def test_goals_has_no_function_local_torch_import(self):
        tree = ast.parse((ROOT / 'goals.py').read_text())
        bad = [(f.name, n.lineno) for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
               for n in ast.walk(f) if isinstance(n, ast.Import)
               and any((a.asname or a.name) == 'torch' for a in n.names)]
        self.assertEqual(bad, [])


class GripperCurrentLimitTests(unittest.TestCase):
    def test_trips_on_first_fresh_sample_over_limit(self):
        import time as _time
        fn = load('delto_gripper_controller.py', 'close_current_exceeded',
                  dict(time=_time), owner='DeltoGripperController')
        cfg = types.SimpleNamespace(close_current_limit_a=1.5,
                                    nontactile_feedback_max_age_s=0.3)
        g = types.SimpleNamespace(
            node=types.SimpleNamespace(cfg=types.SimpleNamespace(gripper=cfg)),
            _over_current_count=0, _current_sample_stamp=None,
            close_peak_current_a=0.0)

        def sample(amps_each):
            g.actual_joint_effort = [amps_each] * 12
            g.actual_joint_stamp = _time.time()
            _time.sleep(0.001)          # distinct stamp per sample
            return fn(g)

        self.assertFalse(sample(0.05))  # 0.6A total: under limit
        self.assertTrue(sample(0.20))   # 2.4A: first over-limit sample trips
        self.assertTrue(fn(g))          # same sample re-read: still tripped
        self.assertAlmostEqual(g.close_peak_current_a, 2.4)
        g.actual_joint_stamp -= 1.0     # stale feedback never trips
        g._over_current_count = 5
        self.assertFalse(fn(g))

    def test_close_command_never_leads_measured_by_more_than_limit(self):
        import time as _time
        fn = load('delto_gripper_controller.py', '_lead_limited',
                  dict(time=_time), owner='DeltoGripperController')
        cfg = types.SimpleNamespace(close_max_lead_rad=0.25)
        g = types.SimpleNamespace(
            node=types.SimpleNamespace(cfg=types.SimpleNamespace(gripper=cfg)),
            actual_joint_position=[0.0, 1.0, 0.5],
            actual_joint_stamp=_time.time())
        # Far target clamped both ways; a near one passes through.
        self.assertEqual(fn(g, [2.0, 0.0, 0.6]), [0.25, 0.75, 0.6])
        g.actual_joint_stamp -= 1.0     # stale feedback: stop advancing
        self.assertIsNone(fn(g, [2.0, 0.0, 0.6]))
        g.actual_joint_position = None  # no feedback yet: same
        self.assertIsNone(fn(g, [2.0, 0.0, 0.6]))
        g.actual_joint_position = [0.0, 1.0, 0.5]
        g.active_grasp_mode = "ENVELOP"  # 12 joints: its own, smaller lead
        cfg.close_max_lead_rad_envelop = 0.10
        g.actual_joint_stamp = _time.time()
        self.assertEqual(fn(g, [2.0, 0.0, 0.6]), [0.10, 0.90, 0.6])
        g.active_grasp_mode = "NORMAL"
        cfg.close_max_lead_rad = 0.0    # 0 disables
        g.actual_joint_stamp = _time.time()
        self.assertEqual(fn(g, [2.0, 0.0, 0.6]), [2.0, 0.0, 0.6])


class AxisWorldTiltTests(unittest.TestCase):
    def test_tilt_is_measured_from_gravity_not_the_image(self):
        fn = load('vision/node.py', '_axis_world_tilt_deg', dict(math=math, np=np),
                  owner='VisionNode')
        # Upright camera looking along base +x: optical x -> -base y,
        # optical y (down) -> -base z, optical z -> +base x.
        R = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
        vn = types.SimpleNamespace(_cached_tf_base=(R, np.zeros(3)))
        self.assertAlmostEqual(fn(vn, 90.0), 0.0, places=6)    # hanging date
        self.assertAlmostEqual(fn(vn, 0.0), 90.0, places=6)    # lying sideways
        self.assertAlmostEqual(fn(vn, 45.0), 45.0, places=6)
        # Same hanging date, camera rolled 40 deg about its view axis: the image
        # shows it at 90-40 = 50 deg, yet it is still vertical in the world.
        a = math.radians(40.0)
        roll = np.array([[math.cos(a), -math.sin(a), 0.], [math.sin(a), math.cos(a), 0.], [0., 0., 1.]])
        vn._cached_tf_base = (R @ roll, np.zeros(3))
        self.assertAlmostEqual(fn(vn, 50.0), 0.0, places=6)
        vn._cached_tf_base = None
        self.assertIsNone(fn(vn, 90.0))


class HoldCurrentGuardTests(unittest.TestCase):
    def _gripper(self, limit=1.15):
        import time as _time
        fn = load('delto_gripper_controller.py', '_watch_hold_current',
                  dict(time=_time, Int16MultiArray=lambda: types.SimpleNamespace(data=None)),
                  owner='DeltoGripperController')
        sent, logs = [], []
        g = types.SimpleNamespace(
            node=types.SimpleNamespace(
                running=True,
                cfg=types.SimpleNamespace(gripper=types.SimpleNamespace(
                    holding_current_limit_a=limit, holding_current_limit_hold_s=0.1)),
                get_logger=lambda: types.SimpleNamespace(warn=logs.append)),
            holding_control_publisher=types.SimpleNamespace(publish=lambda m: sent.append(m.data)),
            published=[], _hold_seq=1, hold_peak_current_a=0.0, hold_current_limited=False,
            actual_joint_position=[0.5] * 12, current_position=None,
            actual_joint_effort=[0.0] * 12, actual_joint_stamp=_time.time())
        g.publish_position = g.published.append
        return fn, g, sent, logs

    def _feed(self, g, amps, seconds):
        import time as _time
        end = _time.time() + seconds
        while _time.time() < end:
            g.actual_joint_effort = [amps / 12.0] * 12
            g.actual_joint_stamp = _time.time()
            _time.sleep(0.05)

    def test_empty_close_hold_is_released_to_position_hold(self):
        fn, g, sent, logs = self._gripper()
        t = threading.Thread(target=fn, args=(g, 1, [1, 70, 0, 0])); t.start()
        self._feed(g, 1.24, 0.4)                     # the outdoor trip level
        t.join(1.0)
        self.assertFalse(t.is_alive())
        self.assertEqual(sent, [[0, 1, 70, 0, 0]])   # force-hold released
        self.assertEqual(g.published, [[0.5] * 12])  # held at the measured pose
        self.assertTrue(g.hold_current_limited)
        self.assertGreater(g.hold_peak_current_a, 1.2)

    def test_held_date_level_and_brief_spike_keep_the_hold(self):
        fn, g, sent, logs = self._gripper()
        t = threading.Thread(target=fn, args=(g, 1, [1, 70, 0, 0])); t.start()
        self._feed(g, 1.04, 0.3)                     # the held date's level
        self._feed(g, 1.30, 0.05)                    # one-sample spike
        self._feed(g, 1.04, 0.2)
        g._hold_seq = 2                              # hold released by OPEN
        t.join(1.0)
        self.assertFalse(t.is_alive())
        self.assertEqual(sent, [])
        self.assertFalse(g.hold_current_limited)


class AfterReverseSnapshotTests(unittest.TestCase):
    def test_snapshot_lands_in_episode_recorded_after_it(self):
        import os
        import tempfile
        import time as _time
        from ur10e_curobo.grasp_episode_recorder import snapshot_after_reverse
        logger = types.SimpleNamespace(warn=lambda *a, **k: None)
        node = types.SimpleNamespace(
            cfg=types.SimpleNamespace(planner=types.SimpleNamespace(
                grasp_episode_recording=True)),
            get_logger=lambda: logger,
            capture_grasp_pair_frame=lambda *a, **k: np.zeros((8, 8, 3), np.uint8),
            gripper_controller=types.SimpleNamespace(
                actual_joint_position=[0.1] * 12, actual_joint_effort=[0.02] * 12),
            _last_episode=("/old/episode", _time.time() - 60.0))  # previous cycle
        with tempfile.TemporaryDirectory() as path:
            with open(os.path.join(path, "episode.json"), "w") as fh:
                json.dump({"outcome": "GRABBED"}, fh)
            t_close = _time.time()
            snapshot_after_reverse(node, t_close)
            _time.sleep(0.3)                    # outcome decided after the reverse
            node._last_episode = (path, _time.time())
            for _ in range(30):
                if os.path.isfile(os.path.join(path, "after_reverse.jpg")):
                    break
                _time.sleep(0.1)
            _time.sleep(0.1)
            with open(os.path.join(path, "episode.json")) as fh:
                ep = json.load(fh)
            self.assertEqual(ep["outcome"], "GRABBED")
            self.assertEqual(ep["after_reverse"]["image"], "after_reverse.jpg")
            self.assertEqual(ep["after_reverse"]["gripper_joint_position"], [0.1] * 12)


class FinalInflightTimelineTests(unittest.TestCase):
    def test_timeline_records_nearest_candidates_with_quality(self):
        import time as _time
        env = dict(time=_time, math=math)
        for name in ("_reacquire_candidate_quality", "_reacquire_quality_ok",
                     "_final_inflight_monitor"):
            fn = load('goals.py', name, env)
        planner = types.SimpleNamespace(
            final_inflight_min_delta_m=0.002, final_inflight_max_delta_m=0.025,
            final_inflight_apply=False)
        now_ns = lambda: int(_time.time() * 1e9)
        obs = types.SimpleNamespace(
            stamp_ns=now_ns(), fresh=lambda ns: True,
            poses=((0.0, 0.0, 0.0), (0.010, 0.0, 0.0), (0.200, 0.0, 0.0),
                   (0.050, 0.0, 0.0)),
            qualities=((0.9, 0.002, 0.8, 1, 0.1), (0.9, 0.002, 0.10, 1, 0.1),
                       (0.9, 0.002, 0.8, 1, 0.1), (0.9, 0.002, 0.8, 1, 0.1)))
        node = types.SimpleNamespace(
            cfg=types.SimpleNamespace(planner=planner), all_fruits_seq=0,
            _fruit_observation=obs,
            get_clock=lambda: types.SimpleNamespace(
                now=lambda: types.SimpleNamespace(nanoseconds=now_ns())))
        stop = threading.Event()
        report = dict(frames=0, rejected=0, no_candidate=0, matched=0,
                      below_min=0, above_max=0, usable=0, applied=0,
                      applied_total=0.0, last_apply_t=0.0)
        t = threading.Thread(target=fn, args=(node, [0.0, 0.0, 0.0], stop, report))
        t.start()
        node.all_fruits_seq = 1         # one new frame
        _time.sleep(0.2)
        stop.set()
        t.join(1.0)
        self.assertEqual(len(report["timeline"]), 1)
        cands = report["timeline"][0]["c"]
        self.assertEqual([c[3] for c in cands], [0.0, 10.0, 50.0])  # nearest 3
        self.assertEqual([c[7] for c in cands], [True, False, True])  # conf 0.10 fails
        self.assertEqual(cands[1][6], 0.1)
        self.assertGreaterEqual(report["timeline"][0]["age"], 0.0)


class FinalTrackAssociationTests(unittest.TestCase):
    def test_hidden_date_does_not_switch_to_neighbour(self):
        import time as _time
        from ur10e_curobo.fruit_track import SelectedFruitTrack
        env = dict(time=_time, math=math, SelectedFruitTrack=SelectedFruitTrack)
        for name in ("_reacquire_candidate_quality", "_reacquire_quality_ok",
                     "_final_inflight_monitor"):
            fn = load('goals.py', name, env)
        planner = types.SimpleNamespace(
            final_inflight_min_delta_m=0.002, final_inflight_max_delta_m=0.025,
            final_inflight_apply=False)
        now_ns = lambda: int(_time.time() * 1e9)
        good = (0.9, 0.003, 0.8, 1, 0.1)
        date_f, other_f = (22, 50, 20, 40, 0), (22, 80, 0, 10, 0)

        def obs(poses, feats):
            return types.SimpleNamespace(
                stamp_ns=now_ns(), fresh=lambda ns: True, poses=poses,
                qualities=(good,) * len(poses), features=feats)
        node = types.SimpleNamespace(
            cfg=types.SimpleNamespace(planner=planner), all_fruits_seq=0,
            _fruit_observation=obs(((0.001, 0, 0), (0, 0.022, 0)), (date_f, other_f)),
            get_clock=lambda: types.SimpleNamespace(
                now=lambda: types.SimpleNamespace(nanoseconds=now_ns())))
        stop = threading.Event()
        report = dict(frames=0, rejected=0, no_candidate=0, matched=0,
                      below_min=0, above_max=0, usable=0, applied=0,
                      applied_total=0.0, last_apply_t=0.0)
        t = threading.Thread(target=fn, args=(node, [0.0, 0.0, 0.0], stop, report))
        t.start()
        node.all_fruits_seq = 1                      # date + neighbour
        _time.sleep(0.15)
        # Date hidden: only the neighbour, 22 mm away -- inside the 25 mm
        # correction bound, so nearest-to-target would have retargeted to it.
        node._fruit_observation = obs(((0, 0.022, 0),), (other_f,))
        node.all_fruits_seq = 2
        _time.sleep(0.15)
        stop.set()
        t.join(1.0)
        self.assertNotIn("track_error", report)
        self.assertEqual(report["matched"], 1)       # the date, frame 1
        self.assertEqual(report.get("track_missed"), 1)
        self.assertEqual(report["usable"], 0)        # never offered as a correction
        self.assertEqual(report["timeline"][1]["trk"][-1], "missed")


class FinalFrameCaptureTests(unittest.TestCase):
    def test_main_requests_exact_frames_for_misses_and_hits(self):
        import time as _time
        from ur10e_curobo.fruit_track import SelectedFruitTrack
        env = dict(time=_time, math=math, SelectedFruitTrack=SelectedFruitTrack)
        for name in ("_reacquire_candidate_quality", "_reacquire_quality_ok",
                     "_final_frame_capture", "_final_inflight_monitor"):
            fn = load('goals.py', name, env)
        planner = types.SimpleNamespace(
            final_inflight_min_delta_m=0.002, final_inflight_max_delta_m=0.025,
            final_inflight_apply=False, final_frame_capture=True,
            final_frame_dir="/tmp/ff", final_frame_max_misses=2,
            final_frame_max_hits=1, final_frame_min_spacing_s=0.0)
        now_ns = lambda: int(_time.time() * 1e9)
        good = (0.9, 0.003, 0.8, 1, 0.1)
        sent = []

        def obs(poses, stamp):
            return types.SimpleNamespace(
                stamp_ns=stamp, fresh=lambda ns: True, poses=poses,
                qualities=(good,) * len(poses), features=(), ids=(7,) * len(poses))
        node = types.SimpleNamespace(
            cfg=types.SimpleNamespace(planner=planner), all_fruits_seq=0,
            current_joint_positions=[0.0] * 6,
            _request_camera_raw_stream=sent.append,
            get_clock=lambda: types.SimpleNamespace(
                now=lambda: types.SimpleNamespace(nanoseconds=now_ns())))
        stop = threading.Event()
        report = dict(frames=0, rejected=0, no_candidate=0, matched=0,
                      below_min=0, above_max=0, usable=0, applied=0,
                      applied_total=0.0, last_apply_t=0.0)
        t = threading.Thread(target=fn, args=(node, [0.0, 0.0, 0.0], stop, report))
        stamps = []
        # hit, hit (over the cap), miss, miss, miss (over the cap)
        frames = [((0.001, 0, 0),), ((0.001, 0, 0),), (), (), ()]
        node._fruit_observation = obs(frames[0], now_ns())
        t.start()
        for seq, poses in enumerate(frames, 1):
            stamps.append(now_ns())
            node._fruit_observation = obs(poses, stamps[-1])
            node.all_fruits_seq = seq
            _time.sleep(0.12)
        stop.set()
        t.join(1.0)
        self.assertTrue(sent[0].startswith("final_frames arm"))
        self.assertEqual(sent[-1], "final_frames off")
        saves = [json.loads(s.split(" ", 2)[2]) for s in sent
                 if s.startswith("final_frames save ")]
        self.assertEqual([s["kind"] for s in saves], ["hit", "miss", "miss"])
        # each save names the exact frame that produced that observation
        self.assertEqual([s["stamp_ns"] for s in saves],
                         [stamps[0], stamps[2], stamps[3]])
        self.assertEqual(saves[1]["target_id"], 7)          # remembered from the hit
        self.assertIn("position", saves[1]["track"])        # predicted, for the label
        self.assertEqual(len(saves[1]["track"]["cov_pos_m2"]), 3)
        self.assertEqual(len(report["frame_capture"]["saved"]), 3)

    def test_vision_saves_buffered_frame_with_projected_prediction(self):
        import os
        import tempfile
        from collections import deque
        import cv2
        env = dict(np=np, json=json, os=os, cv2=cv2, Thread=threading.Thread)
        buf = load('vision/node.py', '_buffer_final_frame', env, owner='VisionNode')
        save = load('vision/node.py', '_save_final_frame', env, owner='VisionNode')
        logger = types.SimpleNamespace(warn=lambda *a, **k: None)
        vn = types.SimpleNamespace(
            _final_frames=deque(maxlen=6),
            _cached_tf_base=(np.eye(3), np.zeros(3)),   # camera == base
            node=types.SimpleNamespace(get_logger=lambda: logger))
        img = np.zeros((100, 200, 4), np.uint8)
        raw = [types.SimpleNamespace(bounding_box_2d=[[10, 10], [20, 10], [20, 30], [10, 30]],
                                     probability=0.05)]
        k = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 25.0}
        buf(vn, 111, img, k, [2.0, 2.0], raw, [])
        buf(vn, 222, img + 1, k, [2.0, 2.0], [], [])
        with tempfile.TemporaryDirectory() as d:
            save(vn, json.dumps({"stamp_ns": 111, "dir": d, "name": "f",
                                 "track": {"position": [0.1, 0.05, 1.0],
                                           "pos_std_m": 0.01}}))
            for _ in range(30):
                if os.path.isfile(os.path.join(d, "f.json")):
                    break
                threading.Event().wait(0.05)
            meta = json.load(open(os.path.join(d, "f.json")))
            self.assertEqual(meta["stamp_ns"], 111)
            self.assertEqual(meta["predicted_px"], [120.0, 60.0])  # (100*.1+50)*2, (100*.05+25)*2
            self.assertEqual(meta["predicted_sigma_px"], 2.0)
            self.assertEqual(meta["raw_detections_xyxy_conf"], [[20.0, 20.0, 40.0, 60.0, 0.05]])
            self.assertEqual(int(cv2.imread(os.path.join(d, "f.jpg")).max()), 0)  # frame 111, not 222


class AutoHarvestGoalFailureTests(unittest.TestCase):
    def test_repeatedly_failing_date_is_excluded_once(self):
        fn = load('node.py', 'record_goal_failure', dict(math=math))
        counts, excluded = [], []
        date = [0.546, -1.233, -0.336]          # the VERY LOW date of 2026-11-02
        self.assertEqual(fn(counts, excluded, date, 2), 0)          # one retry allowed
        self.assertEqual(fn(counts, excluded, [0.548, -1.232, -0.335], 2), 2)  # same date
        self.assertEqual(len(excluded), 1)
        self.assertEqual(fn(counts, excluded, date, 2), 0)          # not re-added
        self.assertEqual(len(excluded), 1)
        self.assertEqual(fn(counts, excluded, [0.40, -1.20, 0.0], 2), 0)  # other date
        self.assertEqual(len(counts), 2)


class LabelFinalFramesTests(unittest.TestCase):
    def test_crop_prefill_and_label_geometry(self):
        from ur10e_curobo.label_final_frames import crop_window, prefill, label_lines
        self.assertEqual(crop_window([1000, 800], 1920, 1080), (680, 440))
        self.assertEqual(crop_window([10, 1070], 1920, 1080), (0, 440))   # kept inside
        self.assertEqual(crop_window(None, 1920, 1080), (640, 220))       # no prediction
        boxes = prefill([[700, 460, 760, 540, 0.6],     # inside -> kept
                         [0, 0, 50, 50, 0.9],           # outside the crop -> dropped
                         [700, 460, 760, 540, 0.1]],    # below threshold -> dropped
                        680, 440)
        self.assertEqual(boxes, [[1, 20.0, 20.0, 80.0, 100.0]])
        seg, det = label_lines(boxes + [[2, 0.0, 0.0, 64.0, 640.0]])
        self.assertEqual(det[0], "1 0.078125 0.093750 0.093750 0.125000")
        pts = [float(v) for v in seg[0].split()[1:]]
        self.assertEqual(len(pts), 32)                  # 16-point ellipse for a date
        xs, ys = pts[0::2], pts[1::2]
        self.assertAlmostEqual(min(xs), 20 / 640, places=4)
        self.assertAlmostEqual(max(ys), 100 / 640, places=4)
        self.assertEqual(seg[1].split()[0], "2")        # trunk as a rectangle
        self.assertEqual(len(seg[1].split()), 9)

    def test_hints_are_uncovered_weak_detections(self):
        from ur10e_curobo.label_final_frames import hints
        boxes = [[1, 10.0, 10.0, 60.0, 60.0]]
        raw = [[110, 110, 150, 150, 0.12],   # covered by the box (crop at 100,100)
               [200, 200, 240, 250, 0.08],   # uncovered -> hint
               [300, 300, 340, 340, 0.02],   # too weak
               [395, 395, 520, 520, 0.10]]   # clipped to 5 px -> too small
        self.assertEqual(hints(raw, 100, 100, boxes, size=300),
                         [[100.0, 100.0, 140.0, 150.0, 0.08]])

    def test_review_reloads_saved_boxes(self):
        import os, tempfile
        from ur10e_curobo.label_final_frames import label_lines, load_box_labels
        boxes = [[1, 20.0, 20.0, 80.0, 100.0], [2, 0.0, 0.0, 64.0, 400.0]]
        _, det = label_lines(boxes, size=400)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.txt")
            open(p, "w").write("\n".join(det) + "\n")
            back = load_box_labels(p, 400)
        for a, b in zip(back, boxes):
            self.assertEqual(a[0], b[0])
            for u, v in zip(a[1:], b[1:]):
                self.assertAlmostEqual(u, v, places=3)


class HandEyeRefinementTests(unittest.TestCase):
    def test_reprojection_refinement_recovers_known_transform(self):
        import cv2
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        env = dict(np=np, cv2=cv2, Rotation=Rotation, least_squares=least_squares)
        path = 'vision/hand_eye_calibration.py'
        to_T = load(path, '_to_T', env)
        solve = load(path, 'solve_board_pose', env)
        refine = load(path, 'refine_hand_eye_reprojection', env)

        rng = np.random.default_rng(0)
        K = np.array([[1000., 0, 960], [0, 1000., 540], [0, 0, 1]])
        dist = np.zeros(5)
        gx, gy = np.meshgrid(np.arange(8) * .04, np.arange(13) * .04)
        obj = np.stack([gx.ravel() - .14, gy.ravel() - .24, 0 * gx.ravel()], 1)
        Y = to_T(np.eye(3), [.6, -.4, 0.])
        X = to_T(Rotation.from_euler('xyz', [2, -3, 90], degrees=True).as_matrix(),
                 [.03, -.07, .05])
        G_list, C_list, objs, imgs = [], [], [], []
        for _ in range(15):
            pos = Y[:3, 3] + [rng.uniform(-.25, .25), rng.uniform(-.25, .25),
                              .5 + rng.uniform(-.05, .1)]
            z = (Y[:3, 3] - pos) / np.linalg.norm(Y[:3, 3] - pos)
            x = np.cross([0, 1, 0], z); x /= np.linalg.norm(x)
            R = np.column_stack([x, np.cross(z, x), z]) @ Rotation.from_rotvec(
                [0, 0, rng.uniform(-1, 1)]).as_matrix()
            C = to_T(R, pos)                      # base -> camera
            T_ct = np.linalg.inv(C) @ Y
            rvec, _ = cv2.Rodrigues(T_ct[:3, :3])
            px, _ = cv2.projectPoints(obj, rvec, T_ct[:3, 3], K, dist)
            px = px.reshape(-1, 2) + rng.normal(0, .3, (len(obj), 2))
            r, t, _ = solve(obj, px, K, dist)
            G_list.append(C @ np.linalg.inv(X))
            C_list.append(to_T(cv2.Rodrigues(r)[0], t))
            objs.append(obj); imgs.append(px)

        X0 = X.copy()
        X0[:3, 3] += [.003, -.002, .002]
        X0[:3, :3] = X0[:3, :3] @ Rotation.from_rotvec([.008, 0, .005]).as_matrix()
        X_hat, stats = refine(X0, G_list, C_list, objs, imgs, K, dist)

        self.assertTrue(stats['success'])
        self.assertLess(stats['after_px'], stats['before_px'])
        self.assertLess(stats['after_px'], .6)
        self.assertLess(np.linalg.norm(X_hat[:3, 3] - X[:3, 3]), .001)
        self.assertLess(np.degrees(Rotation.from_matrix(
            X[:3, :3].T @ X_hat[:3, :3]).magnitude()), .1)


if __name__=='__main__': unittest.main()

class IncrementalDiscoveryTests(unittest.TestCase):
    def setup_scan(self, plan, execute, found=lambda: False):
        clock = Clock()
        node = types.SimpleNamespace(
            cfg=types.SimpleNamespace(lidar_scan=types.SimpleNamespace(
                discovery_plan_budget_s=5., speed_factor=.08)),
            current_joint_positions=[0.] * 6, stop_requested=False,
            get_logger=lambda: types.SimpleNamespace(info=lambda msg: None, warn=lambda msg: None))
        env = dict(time=clock, PlanOptions=lambda **kw: types.SimpleNamespace(**kw),
                   _semicircle_scan_poses=lambda *a, **kw: [[1, 2], [3]],
                   _cartesian_scan_candidate=plan, _execute_scan_target=execute)
        run = load('lidar_scan.py', '_run_discovery_scan', env)
        return run, node, clock

    def test_detected_during_move_skips_remaining_plans(self):
        calls, found = [], [False]
        def plan(node, joints, pose, label, **kw):
            calls.append(pose)
            self.assertEqual(kw['plan_cfg'].max_attempts, 1)
            return {'pose': pose}, 'OK'
        def execute(*a, **kw):
            found[0] = True
            return True
        run, node, _ = self.setup_scan(plan, execute)
        self.assertTrue(run(node, lambda: found[0]))
        self.assertEqual(calls, [1])

    def test_detection_during_planning_prevents_movement(self):
        found = [False]
        def plan(*a, **kw):
            found[0] = True
            return {}, 'OK'
        run, node, _ = self.setup_scan(
            plan, lambda *a, **kw: self.fail('must not move'))
        self.assertTrue(run(node, lambda: found[0]))

    def test_budget_stops_rejected_candidate_search(self):
        calls = []
        def plan(*a, **kw):
            calls.append(1)
            clock.t += 6.
            return None, 'PLAN_OR_IK_FAIL'
        run, node, clock = self.setup_scan(
            plan, lambda *a, **kw: self.fail('must not move'))
        self.assertFalse(run(node, lambda: False))
        self.assertEqual(len(calls), 1)

    def test_next_viewpoint_uses_measured_start_and_horizontal_guard(self):
        starts = []
        def plan(node, joints, pose, label, **kw):
            starts.append((list(joints), kw['require_horizontal']))
            return {'pose': pose}, 'OK'
        def execute(node, *a, **kw):
            node.current_joint_positions = [1.] * 6
            return True
        run, node, _ = self.setup_scan(plan, execute)
        self.assertFalse(run(node, lambda: False))
        self.assertEqual(starts, [([0.] * 6, False), ([1.] * 6, True)])

class SectionScanTests(unittest.TestCase):
    def run_scan(self, reject=None, plan_seconds=0):
        clock = Clock()
        events = []
        node = types.SimpleNamespace(
            cfg=types.SimpleNamespace(lidar_scan=types.SimpleNamespace(
                scan_section_plan_budget_s=5., speed_factor=.08)),
            current_joint_positions=[0], stop_requested=False,
            get_logger=lambda: types.SimpleNamespace(info=lambda msg: events.append(msg)))
        def plan(node, start, pose, label, **kw):
            events.append(('plan', pose, list(start), kw['require_horizontal']))
            self.assertEqual(kw['plan_cfg'].max_attempts, 1)
            clock.t += plan_seconds
            if pose == reject:
                return None, 'JOINT_DELTA'
            return dict(pose=pose, end_joints=[pose]), 'OK'
        def execute(node, target, *args, **kw):
            events.append(('move', target['pose']))
            node.current_joint_positions = [target['pose'] + .01]
            return True
        run = load('lidar_scan.py', '_run_section_scan', dict(
            time=clock, PlanOptions=lambda **kw: types.SimpleNamespace(**kw),
            _semicircle_scan_poses=lambda *a, **kw: [[1], [2], [3], [4]],
            _cartesian_scan_candidate=plan, _execute_scan_target=execute))
        run(node)
        return events

    def test_sections_start_from_measured_joints(self):
        e = self.run_scan()
        actions = [x for x in e if isinstance(x, tuple)]
        self.assertEqual([x[0] for x in actions],
                         ['plan', 'plan', 'move', 'move', 'plan', 'plan', 'move', 'move'])
        self.assertEqual(actions[4][2], [2.01])
        self.assertFalse(actions[0][3])
        self.assertTrue(actions[1][3])

    def test_boundary_executes_valid_prefix_then_stops(self):
        e = self.run_scan(reject=2)
        self.assertEqual([x for x in e if isinstance(x, tuple) and x[0] == 'move'],
                         [('move', 1)])
        self.assertIn('partial coverage=1/4', e[-1])

    def test_budget_does_not_start_another_plan(self):
        e = self.run_scan(plan_seconds=6)
        self.assertEqual(len([x for x in e if isinstance(x, tuple) and x[0] == 'plan']), 1)
        self.assertIn('reason=PLAN_BUDGET', e[-1])


class ImageStampTfLookupTests(unittest.TestCase):
    def test_waits_briefly_for_tf_at_the_image_stamp(self):
        from tf2_ros import ExtrapolationException
        tree = ast.parse((ROOT / 'vision/node.py').read_text())
        owner = next(n.name for n in tree.body if isinstance(n, ast.ClassDef)
                     and any(isinstance(f, ast.FunctionDef) and f.name == '_lookup_tf_at_image' for f in n.body))
        import time as _time
        env = dict(time=_time.time, sleep=_time.sleep, ExtrapolationException=ExtrapolationException,
                   _tf_stamped_to_Rt=lambda t: t, rclpyTime=lambda: 'latest',
                   rclpyDuration=lambda **kw: None)
        fn = load('vision/node.py', '_lookup_tf_at_image', env, owner)

        class Buf:
            def __init__(self, misses):
                self.misses, self.calls = misses, []
            def lookup_transform(self, target, source, stamp, timeout=None):
                self.calls.append(stamp)
                if stamp != 'latest' and len(self.calls) <= self.misses:
                    raise ExtrapolationException('future')
                return stamp

        self_ = types.SimpleNamespace(_latest_image_stamp='image', cam_frame='cam',
                                      node=types.SimpleNamespace(get_logger=lambda: types.SimpleNamespace(warn=lambda m: None)))
        self_.tf_buffer = Buf(misses=3)          # listener catches up after ~3ms
        self.assertEqual(fn(self_, 'base_link'), 'image')
        self_.tf_buffer = Buf(misses=10**6)      # never arrives: old fallback
        t0 = _time.time()
        self.assertEqual(fn(self_, 'base_link'), 'latest')
        self.assertLess(_time.time() - t0, 0.1)


class WiderCorridorSkipTests(unittest.TestCase):
    def test_wrong_side_flip_drops_wider_same_side_only(self):
        import re
        fn = load('goals.py', '_wider_same_side_corridors', dict(re=re))
        labels = ['FROM_RIGHT_45', 'FROM_RIGHT_30', 'FROM_RIGHT_20', 'FROM_RIGHT_10',
                  'STRAIGHT', 'FROM_LEFT_10', 'FROM_LEFT_45']
        flip = 'APPROACH_BRANCH_SWITCH(j1=165deg>75deg)'
        self.assertEqual(fn('FROM_RIGHT_20', flip, labels), {'FROM_RIGHT_45', 'FROM_RIGHT_30'})
        self.assertEqual(fn('FROM_RIGHT_20', 'APPROACH_BRANCH_SWITCH(j1=90deg>75deg)', labels), set())
        self.assertEqual(fn('FROM_RIGHT_20', 'FINAL_WAYPOINT_1', labels), set())
        self.assertEqual(fn('STRAIGHT', flip, labels), set())


class BlindCloseTests(unittest.TestCase):
    def test_only_far_unmatched_closes_are_skipped(self):
        fn = load('goals.py', '_blind_close', {})
        r0 = {"frames": 13, "matched": 0}
        self.assertTrue(fn(r0, {"nearest_date_mm": 114.0}, 40.0))   # 2026-11-04 miss
        self.assertFalse(fn(r0, {"nearest_date_mm": 20.0}, 40.0))
        self.assertFalse(fn(r0, {}, 40.0))                          # no dates in view (ENVELOP held)
        self.assertFalse(fn({"frames": 14, "matched": 1}, {"nearest_date_mm": 114.0}, 40.0))
        self.assertFalse(fn({"frames": 3, "matched": 0}, {"nearest_date_mm": 114.0}, 40.0))
        self.assertFalse(fn(None, None, 40.0))
