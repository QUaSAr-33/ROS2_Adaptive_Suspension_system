from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'semi_active_suspension_system'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='admirer',
    maintainer_email='admirer@todo.todo',
    description='Semi-Active Suspension System Pipeline: Pothole/Bump Detection, Tracking, Estimation, PID Control',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'camera_node         = semi_active_suspension_system.camera_node:main',
            'preprocessing_node  = semi_active_suspension_system.preprocessing_node:main',
            'optical_flow        = semi_active_suspension_system.optical_flow:main',
            'yolo_detection_node = semi_active_suspension_system.yolo_detection_node:main',
            'fusion_node         = semi_active_suspension_system.fusion_node:main',
            'tracking_node       = semi_active_suspension_system.tracking_node:main',
            'estimation_node     = semi_active_suspension_system.estimation_node:main',
            'controller_node     = semi_active_suspension_system.controller_node:main',
            'visualization_node  = semi_active_suspension_system.visualization_node:main',
        ],
    },
)