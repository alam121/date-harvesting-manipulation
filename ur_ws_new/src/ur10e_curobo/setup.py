from setuptools import setup

package_name = "ur10e_curobo"

setup(
    name=package_name,
    version="0.1.0",
    packages=[
        package_name,
        f"{package_name}.managers",
        f"{package_name}.vision",
        f"{package_name}.teleop",
    ],
    package_data={
        f"{package_name}.vision": ["charuco_7x24_30mm_22mm_dict5x5_100.png"],
    },
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        # optional if you have a launch file
        (f"share/{package_name}/launch", [
            "launch/bringup.launch.py",
            "launch/combined.launch.py",
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
            "main = ur10e_curobo.main:main",   # <- REQUIRED
            "gui = ur10e_curobo.gui:main",     # Desktop GUI control panel
            "vision = ur10e_curobo.vision:main",  # ZED/YOLO vision node
            "teleop = ur10e_curobo.teleop:main",  # Joystick teleop control
            "calibrate = ur10e_curobo.grasp_calibrate:main",  # Grasp force calibration
            "hand_eye = ur10e_curobo.vision.hand_eye_calibration:main",  # Hand-eye calibration
            "dg3fm_finger_test = ur10e_curobo.dg3fm_finger_test:main",
        ],
    },
)
