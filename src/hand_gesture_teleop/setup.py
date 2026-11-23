from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'hand_gesture_teleop'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files (simulation and future hand-teleop nodes)
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Simulation robot description (URDF / Xacro)
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf.xacro')),
        # Simulation controller and other configuration files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # MediaPipe model assets
        (os.path.join('share', package_name, 'models'), glob('models/*.task')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='santosh',
    maintainer_email='santosh@example.com',
    description='Hand-gesture teleoperation package for the SO-101 robotic arm.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hand_teleop_node = hand_gesture_teleop.hand_teleop_node:main',
        ],
    },
)
