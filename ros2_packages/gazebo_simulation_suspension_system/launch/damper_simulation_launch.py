"""
damper_simulation.launch.py
============================
Uses RViz2 (QT_QPA_PLATFORM=xcb) instead of Gazebo.
Works perfectly on Intel integrated GPU without any driver setup.

What you see in RViz2:
  - Full 3D damper assembly (cylinder, piston rod, solenoid coil)
  - Piston rod moves up/down in real time as PID runs
  - Copper coil sections visible around the cylinder
  - Chrome piston rod extending from bottom
  - Live joint state animation driven by controller

Usage:
  ros2 launch em_damper damper_simulation.launch.py
  ros2 launch em_damper damper_simulation.launch.py mode:=passive
  ros2 launch em_damper damper_simulation.launch.py mode:=max
  ros2 launch em_damper damper_simulation.launch.py kp:=30.0 kd:=8.0
  ros2 launch em_damper damper_simulation.launch.py excitation_amp:=0.06
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    pkg  = get_package_share_directory('gazebo_simulation_suspension_system')
    urdf = os.path.join(pkg, 'urdf', 'em_damper.urdf')
    rviz = os.path.join(pkg, 'config', 'damper_view.rviz')

    with open(urdf, 'r') as f:
        robot_desc = f.read()

    # ── Launch arguments ──────────────────────────────────────
    args = [
        DeclareLaunchArgument('kp',              default_value='22.0'),
        DeclareLaunchArgument('ki',              default_value='1.2'),
        DeclareLaunchArgument('kd',              default_value='5.0'),
        DeclareLaunchArgument('excitation_amp',  default_value='0.04'),
        DeclareLaunchArgument('excitation_freq', default_value='0.8'),
        DeclareLaunchArgument('mode',            default_value='pid'),
    ]

    # ── 1. Robot state publisher ──────────────────────────────
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_desc,
            'publish_frequency': 100.0,
        }]
    )

    # ── 2. Joint state publisher ──────────────────────────────
    # Forwards /damper/joint_command → joint_states so RViz2
    # animates the piston rod in real time
    joint_state_pub = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_desc,
            'rate': 100,
            'source_list': ['/damper/joint_command'],
        }]
    )

    # ── 3. Damper PID controller ──────────────────────────────
    controller = Node(
        package='gazebo_simulation_suspension_system',
        executable='damper_controller_node.py',
        name='damper_controller',
        output='screen',
        parameters=[{
            'kp':              LaunchConfiguration('kp'),
            'ki':              LaunchConfiguration('ki'),
            'kd':              LaunchConfiguration('kd'),
            'excitation_amp':  LaunchConfiguration('excitation_amp'),
            'excitation_freq': LaunchConfiguration('excitation_freq'),
            'mode':            LaunchConfiguration('mode'),
        }]
    )

    # ── 4. RViz2 — 3D visualiser ──────────────────────────────
    rviz2 = ExecuteProcess(
        cmd=[
            'bash', '-c',
            f'export QT_QPA_PLATFORM=xcb; '
            f'export LIBGL_ALWAYS_INDIRECT=0; '
            f'rviz2 -d {rviz}'
        ],
        output='screen'
    )

    # ── 5. ASCII live dashboard ───────────────────────────────
    dashboard = Node(
        package='gazebo_simulation_suspension_system',
        executable='dashboard_node.py',
        name='dashboard',
        output='screen',
        prefix='xterm -fa "Monospace" -fs 10 -bg "#0a0a12" -fg "#e0e0e0" -e'
    )

    # ── 6. rqt_plot — live graphs (current, damping, accel) ───
    rqt_plot = TimerAction(period=3.0, actions=[
        ExecuteProcess(
            cmd=[
                'bash', '-c',
                'export QT_QPA_PLATFORM=xcb; '
                'ros2 run rqt_plot rqt_plot '
                '/damper/status/data[0] '
                '/damper/status/data[1] '
                '/damper/status/data[4]'
            ],
            output='screen'
        )
    ])

    return LaunchDescription(args + [
        rsp,
        joint_state_pub,
        controller,
        rviz2,
        dashboard,
        rqt_plot,
    ])
