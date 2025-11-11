from setuptools import setup
import sys
# Remove unsupported flags that older colcon may pass to setuptools
sys.argv = [arg for arg in sys.argv if arg not in ("--editable", "--no-editable")]

package_name = 'zed_date_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dateplam',
    maintainer_email='syed.alam@kaust.edu.sa',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            # Example: 'detect = zed_date_detector.date_detector:main',
        ],
    },
)

