from setuptools import setup
from pathlib import Path

package_name = "ur10e_curobo"

setup(
    name=package_name,
    version="0.1.0",
    packages=[
        package_name,
        f"{package_name}.managers",
        f"{package_name}.vision",
        f"{package_name}.teleop",
        f"{package_name}.isaac_sim",
    ],
    package_data={
        package_name: [str(p.relative_to(package_name)) for p in
                       Path(package_name, "robot_assets").rglob("*") if p.is_file()],
        f"{package_name}.vision": [
            "charuco_7x24_30mm_22mm_dict5x5_100.png",
            "calibration_profiles/*.yaml",
        ],
    },
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        # optional if you have a launch file
        (f"share/{package_name}/launch", [
            "launch/bringup.launch.py",
            "launch/combined.launch.py",
            "launch/isaac_sim_mirror.launch.py",
            "launch/isaac_jetson_control.launch.py",
        ]),
        (f"share/{package_name}/meshes", [
            "meshes/Golfcart_pallet.stl",
            "meshes/Golfcart_pallet_front.stl",
            "meshes/Golfcart_pallet_mount.stl",
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="You",
    maintainer_email="you@example.com",
    description="UR10e control with cuRobo + perception",
    license="MIT",
    entry_points={
        "console_scripts": [
            "main = ur10e_curobo.runtime_entry:main",   # <- REQUIRED
            "gui = ur10e_curobo.gui:main",     # Desktop GUI control panel
            "vision = ur10e_curobo.vision:main",  # ZED/YOLO vision node
            "teleop = ur10e_curobo.teleop:main",  # Joystick teleop control
            "calibrate = ur10e_curobo.grasp_calibrate:main",  # Grasp force calibration
            "hand_eye = ur10e_curobo.vision.hand_eye_calibration:main",  # Hand-eye calibration
            # Isaac Sim digital twin, cherry-picked from the isaac-sim branch.
            # These modules import no cuRobo, so they are unaffected by the v2
            # migration -- unlike the planning files on that branch, which are
            # still on the v1 API and were deliberately left behind.
            "isaac_mirror = ur10e_curobo.isaac_sim.bridge:main",
            "isaac_twin_recorder = ur10e_curobo.isaac_sim.twin_recorder:main",
            "hardware_demo_recorder = ur10e_curobo.isaac_sim.twin_recorder:hardware_demo_main",
            "one_shot_grasp = ur10e_curobo.isaac_sim.one_shot_grasp:main",
            "isaac_joint_relay = ur10e_curobo.isaac_sim.joint_relay:main",
            "zed_extrinsic = ur10e_curobo.vision.zed_extrinsic_calibration:main",
            "dg3fm_finger_test = ur10e_curobo.dg3fm_finger_test:main",
            "verify_grasp = ur10e_curobo.visual_grasp_verifier:main",
        ],
    },
)
