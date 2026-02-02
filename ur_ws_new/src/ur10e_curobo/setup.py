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
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        # optional if you have a launch file
        (f"share/{package_name}/launch", ["launch/bringup.launch.py"]),
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
            # "perception = ur10e_curobo.perception:cli",  # optional
        ],
    },
)

