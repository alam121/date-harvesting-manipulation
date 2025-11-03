from setuptools import setup

package_name = 'ur10e_curobo_rviz'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['plugin_description.xml']),  # 👈 Add this line
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='datepalm',
    maintainer_email='you@example.com',
    description='RViz plugin for UR10e CuRobo',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [],
    },
)

