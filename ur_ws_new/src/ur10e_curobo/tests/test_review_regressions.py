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
