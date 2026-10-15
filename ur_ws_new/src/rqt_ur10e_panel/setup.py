from setuptools import setup
from glob import glob

package_name = "rqt_ur10e_panel"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, ["plugin.xml"]),
        ("share/" + package_name + "/resource", glob("resource/*.*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "rqt_ur10e_panel = rqt_ur10e_panel.main:main",
        ],
    },
)
