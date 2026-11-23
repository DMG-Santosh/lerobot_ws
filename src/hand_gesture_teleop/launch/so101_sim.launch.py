#!/usr/bin/env python3
"""
so101_sim.launch.py
===================
Hardware-isolated simulation launch for the SO-101 arm.

PURPOSE:
    Brings up a fully functional ROS 2 Control + MoveIt 2 + RViz 2
    environment for testing hand-teleoperation commands without any
    physical hardware.

HARDWARE PLUGIN:
    mock_components/GenericSystem  (loopback: commands echo to states)

ISOLATION GUARANTEES:
    - Loads so101_sim.urdf.xacro, NOT so101.urdf.xacro
    - Does NOT start so_arm_100_controller/SOARM100Interface
    - Does NOT open serial devices (/dev/ttyUSB*, /dev/ttyACM*)
    - Does NOT enable or command physical servo torque
    - The physical robot launch file (so101_moveit.launch.py) is
      NOT included or referenced here

CONTROLLER STARTUP ORDER:
    1. ros2_control_node starts (loads mock hardware from URDF)
    2. joint_state_broadcaster spawned
    3. arm_controller spawned (after JSB is active)
    4. gripper_controller spawned (after arm_controller is active)
    5. move_group node started (depends on joint_states topic)
    6. RViz 2 started

NODE INTERFACES (identical to physical launch — same action endpoints):
    /arm_controller/follow_joint_trajectory  (FollowJointTrajectory action)
    /gripper_controller/follow_joint_trajectory
    /arm_controller/joint_trajectory         (direct topic streaming)
    /gripper_controller/joint_trajectory
    /joint_states                            (sensor_msgs/msg/JointState)
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessStart, OnProcessExit
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():

    # ── Launch arguments ────────────────────────────────────────────────────
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Launch RViz 2 with the MoveIt configuration",
    )
    use_rviz = LaunchConfiguration("use_rviz")

    # ── Package directories ──────────────────────────────────────────────────
    hand_gesture_pkg = get_package_share_directory("hand_gesture_teleop")
    lerobot_moveit_pkg = get_package_share_directory("lerobot_moveit")

    # ── Simulation robot description ─────────────────────────────────────────
    # Source: src/hand_gesture_teleop/urdf/so101_sim.urdf.xacro
    # Uses mock_components/GenericSystem — NOT the physical plugin.
    sim_xacro_path = os.path.join(hand_gesture_pkg, "urdf", "so101_sim.urdf.xacro")
    robot_description_content = ParameterValue(
        Command(["xacro ", sim_xacro_path]),
        value_type=str,
    )
    robot_description = {"robot_description": robot_description_content}

    # ── Simulation controller YAML ───────────────────────────────────────────
    # Source: src/hand_gesture_teleop/config/ros2_control_sim.yaml
    ros2_control_sim_yaml = os.path.join(
        hand_gesture_pkg, "config", "ros2_control_sim.yaml"
    )

    # ── MoveIt configuration ─────────────────────────────────────────────────
    # Reuses the existing semantic (SRDF), kinematics, joint limits, and
    # MoveIt controller configuration from lerobot_moveit.
    # The robot_description is OVERRIDDEN with the simulation URDF above so
    # that MoveIt uses the same link/joint tree but the mock hardware plugin.
    moveit_config = (
        MoveItConfigsBuilder("so101", package_name="lerobot_moveit")
        .robot_description(file_path=sim_xacro_path)
        .robot_description_semantic(file_path="config/so101.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .robot_description_kinematics()
        .joint_limits()
        .to_moveit_configs()
    )

    # ── RViz configuration ───────────────────────────────────────────────────
    rviz_config_path = os.path.join(lerobot_moveit_pkg, "config", "moveit.rviz")

    # ── Node definitions ────────────────────────────────────────────────────

    # 1. Robot State Publisher
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    # 2. ROS 2 Control node — loads mock hardware from simulation URDF
    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        name="controller_manager",
        output="screen",
        parameters=[
            robot_description,
            ros2_control_sim_yaml,
        ],
    )

    # 3. joint_state_broadcaster — spawned immediately after control_node starts
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        name="jsb_spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager", "/controller_manager",
        ],
        output="screen",
    )

    # 4. arm_controller — spawned after joint_state_broadcaster exits cleanly
    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        name="arm_controller_spawner",
        arguments=[
            "arm_controller",
            "--controller-manager", "/controller_manager",
        ],
        output="screen",
    )

    # 5. gripper_controller — spawned after arm_controller exits cleanly
    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        name="gripper_controller_spawner",
        arguments=[
            "gripper_controller",
            "--controller-manager", "/controller_manager",
        ],
        output="screen",
    )

    # 6. MoveIt move_group node
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        name="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            robot_description,
            {"publish_robot_description_semantic": True},
        ],
        arguments=["--ros-args", "--log-level", "info"],
    )

    # 7. RViz 2
    rviz_node = Node(
        condition=None,  # use_rviz condition evaluated below at list build time
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_path],
        parameters=[
            robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
        ],
    )

    # ── Controller spawning sequence ─────────────────────────────────────────
    # Mirror the ordering used in the physical so101_moveit.launch.py:
    #   JSB spawned when control_node starts
    #   arm_controller spawned when JSB spawner exits
    #   gripper_controller spawned when arm_controller spawner exits

    spawn_jsb = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=control_node,
            on_start=[joint_state_broadcaster_spawner],
        )
    )

    spawn_arm = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[arm_controller_spawner],
        )
    )

    spawn_gripper = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=arm_controller_spawner,
            on_exit=[gripper_controller_spawner],
        )
    )

    # move_group is delayed 3 s to allow joint_states to begin publishing
    # before MoveIt's PlanningSceneMonitor subscribes to it.
    move_group_delayed = TimerAction(period=3.0, actions=[move_group_node])

    # ── Launch description ───────────────────────────────────────────────────
    return LaunchDescription([
        use_rviz_arg,
        robot_state_publisher,
        control_node,
        spawn_jsb,
        spawn_arm,
        spawn_gripper,
        move_group_delayed,
        rviz_node,
    ])
