#!/usr/bin/env python3
"""
hand_teleop.launch.py
=====================
Compact launch file for the SO-101 hand-gesture teleoperation node.

Runs hand_teleop_node inside the project virtual environment (.venv_hand_gesture)
with the standard simulation 4-direction continuous position-following settings,
while keeping fail-closed defaults (enable_arm_motion=False) intact in source.

Usage:
    ROS_DOMAIN_ID=10 ros2 launch hand_gesture_teleop hand_teleop.launch.py
    ROS_DOMAIN_ID=10 ros2 launch hand_gesture_teleop hand_teleop.launch.py enable_vertical_z:=true
"""

import os
import sys

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _find_venv_python() -> str:
    """Locate .venv_hand_gesture/bin/python3 by walking up workspace roots."""
    candidates = []
    for base in (os.path.realpath(__file__), os.path.abspath(__file__)):
        cur = os.path.dirname(base)
        for _ in range(8):
            candidates.append(os.path.join(cur, ".venv_hand_gesture", "bin", "python3"))
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    try:
        pkg_prefix = get_package_prefix("hand_gesture_teleop")
        ws_root = os.path.dirname(os.path.dirname(pkg_prefix))
        candidates.append(os.path.join(ws_root, ".venv_hand_gesture", "bin", "python3"))
    except Exception:
        pass
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return sys.executable


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("enable_motion", default_value="true"),
        DeclareLaunchArgument("enable_arm_motion", default_value="true"),
        DeclareLaunchArgument("enable_ik_dry_run", default_value="false"),
        DeclareLaunchArgument("single_goal_test_mode", default_value="false"),
        DeclareLaunchArgument("max_goals_per_session", default_value="0"),
        DeclareLaunchArgument("continuous_arm_control", default_value="true"),
        DeclareLaunchArgument("enable_collision_checking", default_value="true"),
        DeclareLaunchArgument("enable_vertical_z", default_value="false"),
        DeclareLaunchArgument("continuous_cartesian_speed", default_value="0.04"),
        DeclareLaunchArgument("continuous_position_span", default_value="0.05"),
        DeclareLaunchArgument("continuous_max_joint_velocity", default_value="0.20"),
        DeclareLaunchArgument("camera_prefer_fixed_fps", default_value="false"),
        DeclareLaunchArgument("camera_index", default_value="0"),
    ]

    venv_python = _find_venv_python()

    teleop_node = Node(
        package="hand_gesture_teleop",
        executable="hand_teleop_node",
        name="hand_teleop_node",
        output="screen",
        emulate_tty=True,
        prefix=[venv_python],
        parameters=[
            {
                "enable_motion": ParameterValue(LaunchConfiguration("enable_motion"), value_type=bool),
                "enable_arm_motion": ParameterValue(LaunchConfiguration("enable_arm_motion"), value_type=bool),
                "enable_ik_dry_run": ParameterValue(LaunchConfiguration("enable_ik_dry_run"), value_type=bool),
                "single_goal_test_mode": ParameterValue(LaunchConfiguration("single_goal_test_mode"), value_type=bool),
                "max_goals_per_session": ParameterValue(LaunchConfiguration("max_goals_per_session"), value_type=int),
                "continuous_arm_control": ParameterValue(LaunchConfiguration("continuous_arm_control"), value_type=bool),
                "enable_collision_checking": ParameterValue(
                    LaunchConfiguration("enable_collision_checking"), value_type=bool
                ),
                "enable_vertical_z": ParameterValue(LaunchConfiguration("enable_vertical_z"), value_type=bool),
                "continuous_cartesian_speed": ParameterValue(
                    LaunchConfiguration("continuous_cartesian_speed"), value_type=float
                ),
                "continuous_position_span": ParameterValue(
                    LaunchConfiguration("continuous_position_span"), value_type=float
                ),
                "continuous_max_joint_velocity": ParameterValue(
                    LaunchConfiguration("continuous_max_joint_velocity"), value_type=float
                ),
                "camera_prefer_fixed_fps": ParameterValue(
                    LaunchConfiguration("camera_prefer_fixed_fps"), value_type=bool
                ),
                "camera_index": ParameterValue(LaunchConfiguration("camera_index"), value_type=int),
            }
        ],
    )

    return LaunchDescription([*args, teleop_node])
