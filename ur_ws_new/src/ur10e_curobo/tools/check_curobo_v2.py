"""GPU integration check; never creates a ROS node or commands hardware."""
import json
import torch
from curobo.scene import Cuboid
from curobo.types import JointState, Pose
from ur10e_curobo.planner_v2 import PlannerV2, RobotKinematics, load_robot_config
from ur10e_curobo.planner_options import PlanOptions


def main():
    p = PlannerV2(load_robot_config(), {}, interpolation_dt=.004, num_trajopt_seeds=2)
    print('v2 planner constructed (CUDA graphs enabled)', flush=True)
    names = p.planner.joint_names
    q = p.planner.default_joint_state.position.reshape(1, -1)
    js = JointState.from_position(q, joint_names=names)
    kin = RobotKinematics(p.fk_config)
    pos, quat, *_, spheres = kin.forward(q)
    assert spheres.shape == (1, 28, 4)
    assert p.check_constraints(js).feasible.all()
    # Check different batch lengths and reject invalid joint states.
    for count in (2, 5, 1):
        assert p.check_constraints(JointState.from_position(q.repeat(count, 1), joint_names=names)).feasible.all()
    invalid = q.clone(); invalid[0, 0] = 100.
    assert not p.check_constraints(JointState.from_position(invalid, joint_names=names)).feasible.all()
    pose = Pose(position=pos, quaternion=quat)
    ik = p.solve_ik(pose, seed_config=q.unsqueeze(0), retract_config=q, return_seeds=4)
    assert ik.success.any()
    print('FK, batched constraints, joint-limit rejection and IK passed', flush=True)
    target = q.clone(); target[0, 0] += .08
    target_js = JointState.from_position(target, joint_names=names)
    result = p.plan_joint(js, target_js, PlanOptions(max_attempts=3))
    assert result.success, result.status
    path = result.get_interpolated_plan().position
    assert path.ndim == 2 and path.shape[1] == 6
    assert (path[-1] - target[0]).abs().max() < .02
    assert p.check_constraints(JointState.from_position(path, joint_names=names)).feasible.all()
    print('Joint plan passed', path.shape, flush=True)
    tpos, tquat, *_ = kin.forward(target)
    cart = p.plan_pose(js, Pose(position=tpos, quaternion=tquat), PlanOptions(max_attempts=3))
    assert cart.success, cart.status
    cart_path = cart.get_interpolated_plan().position
    end_pos, end_quat, *_ = kin.forward(cart_path[-1:])
    assert torch.linalg.vector_norm(end_pos - tpos) < .005
    print('Cartesian plan passed', cart_path.shape, flush=True)
    # Put a cuboid around the full arm and prove collision rejection, then remove it.
    p.world_model.cuboid.append(Cuboid(name='test_blocker', dims=[4., 4., 4.],
                                      pose=[0., 0., 0., 1., 0., 0., 0.]))
    p.update_world(p.world_model)
    assert not p.check_constraints(js).feasible.all()
    p.world_model.cuboid.clear()
    p.update_world(p.world_model)
    assert p.check_constraints(js).feasible.all()
    print(json.dumps({'gpu': torch.cuda.get_device_name(0), 'collision_spheres': 28,
                      'joint_waypoints': len(path), 'pose_waypoints': len(cart_path),
                      'interpolation_dt_s': result.interpolation_dt,
                      'obstacle_add_remove': 'passed', 'status': 'passed'}), flush=True)
    p.planner.destroy()


if __name__ == '__main__':
    main()
