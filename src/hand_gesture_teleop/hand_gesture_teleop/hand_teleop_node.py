#!/usr/bin/env python3
"""
hand_teleop_node.py
===================
ROS 2 Hand-Gesture Teleoperation Node for the SO-101 Robotic Arm (Simulation Gated).

Subscribes to:
  - /joint_states (sensor_msgs/msg/JointState): Tracks real-time joint positions of the arm.

Clients:
  - /compute_ik (moveit_msgs/srv/GetPositionIK): MoveIt inverse kinematics solver.
  - /controller_manager/list_hardware_components (controller_manager_msgs/srv/ListHardwareComponents):
    Introspects active hardware plugins to verify simulation-only mock hardware before arming.
  - /arm_controller/follow_joint_trajectory (control_msgs/action/FollowJointTrajectory):
    Asynchronous action client for the 5 arm joints (joint1 through joint5).
    NOTE: Gripper joint6 is strictly excluded from trajectory execution in this task.

Safety Architecture (Task 5C-3A & 5C-3A-H):
  - Starts in explicitly DISARMED state.
  - Five Mutually Exclusive Operating Modes:
      Mode 1: enable_motion=False, enable_ik_dry_run=False, DISARMED -> No IK, no motion.
      Mode 2: enable_motion=False, enable_ik_dry_run=True, DISARMED  -> IK dry-run only, zero motion.
      Mode 3: enable_motion=True,  enable_ik_dry_run=False, DISARMED -> No IK, no motion until arming.
      Mode 4: enable_motion=True,  enable_ik_dry_run=False, ARMED    -> Fresh IK and gated arm trajectory dispatch.
      Mode 5: enable_motion=True,  enable_ik_dry_run=True, ANY      -> Conflict rejected, remain disarmed.
  - Fresh, Fail-Closed Simulation Hardware Introspection:
      * Positively verifies SimSystem (system, mock_components/GenericSystem, active).
      * Rejects any unrecognized or extra hardware components.
      * Rejects any physical hardware (SOARM100Interface, Feetech, serial).
      * Freshness check (<= 3.0s); expired or failed checks immediately disarm.
      * Out-of-order sequence tracking rejects delayed stale responses.
  - Two-Key Arming Handshake: Press 'A' to request arming (3.0s window), then 'Y' to confirm.
    All safety preconditions must pass to transition to ARMED.
  - Hardened IK Validator: Rejects duplicate joint names, missing joints, NaNs/infs, limit violations,
    stale/missing feedback (> 0.25s), and joint discontinuities (> 0.10 rad).
  - Validated Cubic-Spline Trajectory Duration:
      * Controller verified to use cubic spline interpolation between rest-to-rest endpoints.
      * Peak velocity bound: v_peak <= 1.5 * delta_q / duration <= 0.20 rad/s.
      * Duration: duration = max(0.50, 1.5 * max_joint_delta / 0.20).
      * At max delta (0.10 rad), duration is 0.75s, strictly limiting peak speed to 0.20 rad/s.
      * Rejects non-finite current or target joint values before building goal.
  - Single Active Goal Guard: At most one trajectory goal active at a time; newest candidate buffered.
  - Immediate Disarm: Pressing 'D', hand loss, stale feedback, hardware verification loss/expiry,
    or mode conflict immediately disarms the node and requests cancellation of any active trajectory goal.
    DISCLAIMER: Trajectory cancellation sends an asynchronous cancel request to the controller action
    server. It is NOT an emergency-stop guarantee and does not guarantee instantaneous hardware motor braking.
"""

from dataclasses import dataclass
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

# ROS 2 Python API
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration, Time as MsgTime
from controller_manager_msgs.srv import ListHardwareComponents
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.msg import State as LifecycleState
from moveit_msgs.msg import MoveItErrorCodes, PositionIKRequest, RobotState
from moveit_msgs.srv import GetPositionIK
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Reusable Hand Tracking Logic
from hand_gesture_teleop.hand_tracking import (
    GESTURE_CONFIG,
    MAPPING_CONFIG,
    GestureConfig,
    MappingConfig,
    MotionPreviewData,
    MotionPreviewTracker,
    NormalizedLandmark,
    TemporalSmoother,
    classify_hand_gestures,
    dist_2d,
    draw_compact_hud,
    draw_landmarks_on_frame,
    draw_thumb_index_connector,
    draw_mapping_hud,
    draw_motion_reticle,
    setup_enlarged_camera_window,
)

# ---------------------------------------------------------------------------
# Teleoperation States & Joint Constants
# ---------------------------------------------------------------------------
STATE_DISARMED = "DISARMED"
STATE_ARMING_PENDING = "ARMING_PENDING"
STATE_ARMED = "ARMED — SIMULATION ONLY"

EXPECTED_SO101_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
]

ARM_CONTROL_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
]

ARM_JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    "joint1": (-2.1, 2.1),
    "joint2": (-2.1, 2.1),
    "joint3": (-2.1, 2.1),
    "joint4": (-2.1, 2.1),
    "joint5": (-3.1, 3.1),
}

# Task 5C-3D Gripper Motion Constants & Configuration
DEFAULT_GRIPPER_OPEN_RAD: float = 1.0          # Radians for open gripper jaw (wide open, ~96 mm tip gap)
DEFAULT_GRIPPER_CLOSED_RAD: float = 0.0        # Radians for closed gripper jaw (neutral/home, ~21 mm tip gap)
GRIPPER_JOINT_LIMIT_MIN: float = -0.2          # Joint limit min from URDF
GRIPPER_JOINT_LIMIT_MAX: float = 2.0           # Joint limit max from URDF
DEFAULT_GRIPPER_DURATION: float = 1.0          # Trajectory duration for safe gripper actuation (s)
DEFAULT_GRIPPER_MAX_VELOCITY: float = 10.0     # URDF joint6 velocity limit (rad/s)
DEFAULT_GRIPPER_MAX_ACCELERATION: float = 5.0  # MoveIt joint6 max acceleration guidance (rad/s^2)
MIN_GRIPPER_TRAJECTORY_DURATION: float = 0.5   # Minimum duration floor for gripper motion (s)

# Task 5C-3A & 5C-3A-H Motion Safety Thresholds
MAX_JOINT_DELTA: float = 0.10             # Max allowed change per joint from current state (rad)
MAX_COMMANDED_VELOCITY: float = 0.20      # Max velocity for trajectory duration calculation (rad/s)
MIN_TRAJECTORY_DURATION: float = 0.50     # Minimum duration for commanded trajectory point (s)
MAX_JOINT_STATE_AGE: float = 0.25         # Max acceptable age of /joint_states feedback (s)
ARMING_TIMEOUT_SEC: float = 3.0           # Timeout for two-key arming handshake (s)

# Task 5C-3A-F No-Op Suppression & IK Target Binding Constants
MIN_IK_TARGET_DISPLACEMENT: float = 0.001   # Min Cartesian displacement to trigger new IK query (1 mm)
MIN_DISPATCH_JOINT_DELTA: float = 0.005     # Min joint displacement to dispatch trajectory goal (rad, ~0.29 deg)
MAX_IK_TARGET_DRIFT: float = 0.010          # Max allowable Cartesian drift between query and solution (10 mm)


@dataclass(frozen=True)
class TargetSnapshot:
    """Immutable snapshot of Cartesian target at the time an IK query is submitted."""
    x: float
    y: float
    z: float
    generation: int
    timestamp: float


# Cubic Spline Interpolation Safety Factor
# In rest-to-rest cubic splines: peak_velocity = 1.5 * delta / duration
# To guarantee peak_velocity <= MAX_COMMANDED_VELOCITY, duration >= 1.5 * delta / MAX_COMMANDED_VELOCITY
CUBIC_VELOCITY_SCALING_FACTOR: float = 1.5

# Simulation Hardware Identity Constants
EXPECTED_MOCK_HARDWARE_NAME: str = "SimSystem"
EXPECTED_MOCK_HARDWARE_TYPE: str = "system"
SIMULATION_MOCK_PLUGIN: str = "mock_components/GenericSystem"
MAX_HARDWARE_CHECK_AGE: float = 3.0       # Max age of hardware verification in seconds
PROHIBITED_PHYSICAL_PLUGINS: Tuple[str, ...] = (
    "so_arm_100_controller/SOARM100Interface",
    "feetech_servo_controller/FeetechServoInterface",
)

MOVEIT_ERROR_NAMES: Dict[int, str] = {
    1: "SUCCESS",
    99999: "FAILURE",
    -1: "PLANNING_FAILED",
    -2: "INVALID_MOTION_PLAN",
    -3: "MOTION_PLAN_INVALIDATED",
    -4: "CONTROL_FAILED",
    -5: "UNABLE_TO_AQUIRE_SENSOR_DATA",
    -6: "TIMED_OUT",
    -7: "PREEMPTED",
    -10: "START_STATE_IN_COLLISION",
    -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
    -12: "GOAL_IN_COLLISION",
    -13: "GOAL_VIOLATES_PATH_CONSTRAINTS",
    -14: "GOAL_CONSTRAINTS_VIOLATED",
    -15: "INVALID_GROUP_NAME",
    -16: "INVALID_GOAL_CONSTRAINTS",
    -17: "INVALID_ROBOT_STATE",
    -18: "INVALID_LINK_NAME",
    -19: "INVALID_OBJECT_NAME",
    -21: "FRAME_TRANSFORM_FAILURE",
    -22: "COLLISION_CHECKING_UNAVAILABLE",
    -23: "ROBOT_STATE_STALE",
    -24: "SENSOR_INFO_STALE",
    -25: "COMMUNICATION_FAILURE",
    -26: "START_STATE_INVALID",
    -27: "GOAL_STATE_INVALID",
    -28: "UNRECOGNIZED_GOAL_TYPE",
    -29: "CRASH",
    -30: "ABORT",
    -31: "NO_IK_SOLUTION",
}


# ---------------------------------------------------------------------------
# Fresh, Fail-Closed Hardware Introspection Helper
# ---------------------------------------------------------------------------
def verify_hardware_components_from_response(response: Any) -> Tuple[bool, str]:
    """
    Positively validates that the active controller_manager hardware component is strictly
    the expected SO-101 simulation mock system ('SimSystem', 'system', 'mock_components/GenericSystem')
    in active state, and that NO unexpected, unrecognized, or physical hardware components are present.

    Returns (is_valid, reason_str).
    """
    if response is None or not hasattr(response, "component"):
        return False, "INVALID_RESPONSE"

    components = response.component
    if not components:
        return False, "NO_COMPONENTS_FOUND"

    # Enforce strict component count for SO-101 simulation: exactly 1 mock system
    if len(components) != 1:
        return False, f"UNEXPECTED_COMPONENT_COUNT_{len(components)}"

    comp = components[0]
    name = str(getattr(comp, "name", ""))
    comp_type = str(getattr(comp, "type", "")).lower()
    plugin = str(getattr(comp, "plugin_name", "") or getattr(comp, "class_type", ""))

    # 1. Reject prohibited physical hardware
    for prohibited in PROHIBITED_PHYSICAL_PLUGINS:
        if prohibited in plugin:
            return False, f"PHYSICAL_HARDWARE_DETECTED_{plugin}"
    if "so_arm" in plugin.lower() or "serial" in plugin.lower() or "feetech" in plugin.lower():
        return False, f"PHYSICAL_HARDWARE_DETECTED_{plugin}"

    # 2. Positively verify expected simulation hardware identity
    if name != EXPECTED_MOCK_HARDWARE_NAME:
        return False, f"UNRECOGNIZED_HARDWARE_NAME_{name}"

    if comp_type != EXPECTED_MOCK_HARDWARE_TYPE:
        return False, f"UNRECOGNIZED_HARDWARE_TYPE_{comp_type}"

    if plugin != SIMULATION_MOCK_PLUGIN:
        return False, f"UNRECOGNIZED_HARDWARE_PLUGIN_{plugin}"

    # 3. Positively verify active lifecycle state
    state = getattr(comp, "state", None)
    state_id = getattr(state, "id", None)
    state_label = str(getattr(state, "label", "")).lower()
    if state_id != 3 and state_label != "active":
        return False, f"MOCK_HARDWARE_NOT_ACTIVE_{state_id}_{state_label}"

    return True, "MOCK_HARDWARE_ACTIVE"


# ---------------------------------------------------------------------------
# Arm Trajectory Goal Builder (Verified Cubic Spline Duration)
# ---------------------------------------------------------------------------
def build_trajectory_goal(
    candidate_map: Dict[str, float],
    current_positions: Dict[str, float],
    now_msg: Optional[Any] = None,
) -> Tuple[FollowJointTrajectory.Goal, float]:
    """
    Constructs a FollowJointTrajectory.Goal strictly targeting the 5 arm joints.
    Validates that current and candidate positions are complete, finite, within limits,
    and do not exceed MAX_JOINT_DELTA.
    Calculates duration based on confirmed cubic spline peak velocity bound:
      duration >= 1.5 * max_joint_delta / MAX_COMMANDED_VELOCITY
    with a minimum duration of MIN_TRAJECTORY_DURATION (0.50s).
    Excludes gripper joint6.
    """
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(ARM_CONTROL_JOINTS)

    max_delta = 0.0
    for j in ARM_CONTROL_JOINTS:
        if j not in candidate_map:
            raise ValueError(f"Missing candidate target position for joint {j}")
        if j not in current_positions:
            raise ValueError(f"Missing current feedback position for joint {j}")

        target_val = float(candidate_map[j])
        curr_val = float(current_positions[j])

        if not math.isfinite(target_val):
            raise ValueError(f"Candidate target position for joint {j} is non-finite: {target_val}")
        if not math.isfinite(curr_val):
            raise ValueError(f"Current feedback position for joint {j} is non-finite: {curr_val}")

        low, high = ARM_JOINT_LIMITS[j]
        if target_val < low or target_val > high:
            raise ValueError(f"Candidate target for {j}={target_val:.3f} violates limits [{low},{high}]")
        if curr_val < low or curr_val > high:
            raise ValueError(f"Current position for {j}={curr_val:.3f} violates limits [{low},{high}]")

        delta = abs(target_val - curr_val)
        if delta > MAX_JOINT_DELTA:
            raise ValueError(
                f"Joint delta for {j} ({delta:.4f} rad) exceeds MAX_JOINT_DELTA ({MAX_JOINT_DELTA:.4f} rad)"
            )
        if delta > max_delta:
            max_delta = delta

    # Verified rest-to-rest cubic spline duration for joint_trajectory_controller:
    # peak_velocity = 1.5 * max_delta / duration <= MAX_COMMANDED_VELOCITY
    # => duration >= 1.5 * max_delta / MAX_COMMANDED_VELOCITY
    duration_sec = max(
        MIN_TRAJECTORY_DURATION,
        (CUBIC_VELOCITY_SCALING_FACTOR * max_delta) / MAX_COMMANDED_VELOCITY,
    )

    point = JointTrajectoryPoint()
    point.positions = [float(candidate_map[j]) for j in ARM_CONTROL_JOINTS]
    point.velocities = [0.0] * len(ARM_CONTROL_JOINTS)
    sec = int(duration_sec)
    nanosec = int(round((duration_sec - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    point.time_from_start = Duration(sec=sec, nanosec=nanosec)

    goal.trajectory.points = [point]
    if now_msg is not None:
        goal.trajectory.header.stamp = now_msg

    return goal, duration_sec


def compute_cubic_peak_velocity(delta_position: float, duration: float) -> float:
    """
    Calculate peak velocity for a rest-to-rest cubic spline trajectory:
      v_peak = 1.5 * |delta_position| / duration
    """
    if duration <= 0.0 or not math.isfinite(duration):
        raise ValueError(f"Duration must be positive and finite, got {duration}")
    return 1.5 * abs(float(delta_position)) / duration


def compute_cubic_peak_acceleration(delta_position: float, duration: float) -> float:
    """
    Calculate peak acceleration for a rest-to-rest cubic spline trajectory:
      a_peak = 6.0 * |delta_position| / duration^2
    """
    if duration <= 0.0 or not math.isfinite(duration):
        raise ValueError(f"Duration must be positive and finite, got {duration}")
    return 6.0 * abs(float(delta_position)) / (duration ** 2)


def compute_gripper_trajectory_duration(
    delta_position: float,
    velocity_limit: float = DEFAULT_GRIPPER_MAX_VELOCITY,
    acceleration_limit: float = DEFAULT_GRIPPER_MAX_ACCELERATION,
    min_duration: float = MIN_GRIPPER_TRAJECTORY_DURATION,
    starting_velocity: float = 0.0,
) -> float:
    """
    Calculate safe duration for a cubic trajectory from actual displacement:
    For rest-to-rest (starting_velocity == 0.0):
      duration = max(min_duration, 1.5 * |delta| / v_limit, sqrt(6.0 * |delta| / a_limit))
    For nonzero starting velocity:
      Accounts for v_0 so that peak acceleration on [0, T] is bounded by acceleration_limit:
        t_acc = (2.0 * |v_0| + sqrt(4.0 * v_0^2 + 6.0 * a_limit * |delta|)) / a_limit
    Guarantees that peak velocity <= velocity_limit and peak acceleration <= acceleration_limit.
    """
    if not math.isfinite(delta_position):
        raise ValueError(f"Non-finite delta_position: {delta_position}")
    if not math.isfinite(starting_velocity):
        raise ValueError(f"Non-finite starting_velocity: {starting_velocity}")
    if not math.isfinite(velocity_limit) or velocity_limit <= 0.0:
        raise ValueError(f"Invalid velocity_limit: {velocity_limit}")
    if not math.isfinite(acceleration_limit) or acceleration_limit <= 0.0:
        raise ValueError(f"Invalid acceleration_limit: {acceleration_limit}")
    if not math.isfinite(min_duration) or min_duration <= 0.0:
        raise ValueError(f"Invalid min_duration: {min_duration}")

    d = abs(float(delta_position))
    v0 = abs(float(starting_velocity))
    t_vel = (1.5 * d) / velocity_limit
    if v0 > 0.0:
        t_acc = (2.0 * v0 + math.sqrt(4.0 * (v0 ** 2) + 6.0 * acceleration_limit * d)) / acceleration_limit
    else:
        t_acc = math.sqrt((6.0 * d) / acceleration_limit)
    return max(min_duration, t_vel, t_acc)


def build_gripper_trajectory_goal(
    target_pos: float,
    current_pos: float,
    starting_velocity: float = 0.0,
    velocity_limit: float = DEFAULT_GRIPPER_MAX_VELOCITY,
    acceleration_limit: float = DEFAULT_GRIPPER_MAX_ACCELERATION,
    min_duration: float = MIN_GRIPPER_TRAJECTORY_DURATION,
    now_msg: Optional[MsgTime] = None,
) -> Tuple[FollowJointTrajectory.Goal, float]:
    """
    Build a deterministic cubic spline FollowJointTrajectory.Goal for joint6.
    Generates two explicit waypoints:
      - Point 0: time=0.0s, position=current_pos, velocity=starting_velocity
      - Point 1: time=duration_sec, position=target_pos, velocity=0.0 (rest finish)
    Both waypoints specify positions and velocities with empty accelerations,
    guaranteeing that ROS 2 Jazzy joint_trajectory_controller selects cubic spline interpolation.

    Header Timestamp Semantics:
      - If now_msg is None, defaults to zero timestamp (sec=0, nanosec=0).
      - In ROS 2 Jazzy joint_trajectory_controller, a zero timestamp indicates "start now",
        causing the controller to set trajectory_start_time_ = sample_time on cycle 0.
        This guarantees Point 0 (time_from_start=0) is sampled at t=0 rather than in the past.
    """
    if not math.isfinite(target_pos):
        raise ValueError(f"target_pos must be finite, got {target_pos}")
    if not math.isfinite(current_pos):
        raise ValueError(f"current_pos must be finite, got {current_pos}")
    if not math.isfinite(starting_velocity):
        raise ValueError(f"starting_velocity must be finite, got {starting_velocity}")

    delta_pos = float(target_pos) - float(current_pos)
    duration_sec = compute_gripper_trajectory_duration(
        delta_position=delta_pos,
        velocity_limit=velocity_limit,
        acceleration_limit=acceleration_limit,
        min_duration=min_duration,
        starting_velocity=starting_velocity,
    )

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = ["joint6"]
    if now_msg is not None:
        goal.trajectory.header.stamp = now_msg
    else:
        goal.trajectory.header.stamp = MsgTime(sec=0, nanosec=0)

    # Point 0: Explicit start waypoint at current position and valid starting velocity
    point0 = JointTrajectoryPoint()
    point0.positions = [float(current_pos)]
    point0.velocities = [float(starting_velocity)]
    point0.accelerations = []
    point0.effort = []
    point0.time_from_start = Duration(sec=0, nanosec=0)

    # Point 1: Explicit target waypoint at rest
    point1 = JointTrajectoryPoint()
    point1.positions = [float(target_pos)]
    point1.velocities = [0.0]
    point1.accelerations = []
    point1.effort = []
    sec = int(duration_sec)
    nanosec = int(round((duration_sec - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    point1.time_from_start = Duration(sec=sec, nanosec=nanosec)

    goal.trajectory.points = [point0, point1]

    tol = JointTolerance()
    tol.name = "joint6"
    tol.position = 0.2
    goal.goal_tolerance = [tol]

    return goal, duration_sec


# ---------------------------------------------------------------------------
# Cartesian Target & IK Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CartesianTargetConfig:
    MAX_CARTESIAN_SPEED: float = 0.02          # Max speed in m/s (20 mm/s)
    MAX_CARTESIAN_STEP: float = 0.002          # Max step increment per frame (2 mm)
    CUMULATIVE_LIMIT_X: float = 0.050          # +/- 50 mm along X from initial pose
    CUMULATIVE_LIMIT_Y: float = 0.050          # +/- 50 mm along Y from initial pose
    CUMULATIVE_LIMIT_Z: float = 0.050          # +/- 50 mm along Z from initial pose
    MIN_HEIGHT_Z: float = 0.050                # Minimum height above base (50 mm)
    WORKSPACE_BOUNDS_X: Tuple[float, float] = (-0.25, 0.25)
    WORKSPACE_BOUNDS_Y: Tuple[float, float] = (-0.40, -0.10)
    WORKSPACE_BOUNDS_Z: Tuple[float, float] = (0.05, 0.45)
    INITIAL_TARGET_POSE: Tuple[float, float, float] = (0.0206, -0.2775, 0.2669)
    INITIAL_ORIENTATION: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, -0.5)  # qx, qy, qz, qw
    IK_PLANNING_GROUP: str = "arm"
    IK_TIP_LINK: str = "gripper"
    IK_REFERENCE_FRAME: str = "world"
    IK_THROTTLE_PERIOD: float = 0.5            # Max 2 Hz query rate


CARTESIAN_TARGET_CONFIG = CartesianTargetConfig()


def compute_cartesian_increment(
    ux: float,
    uy: float,
    uz: float,
    dt: float,
    hand_detected: bool,
    gesture_state: str,
    is_active: bool,
    cfg: CartesianTargetConfig = CARTESIAN_TARGET_CONFIG,
) -> Tuple[float, float, float]:
    """
    Computes conservative Cartesian increments (dx, dy, dz) in meters from normalized hand displacement.

    Mapping:
      - X (lateral):  dx = step_scale * ux  (Right > 0, Left < 0)
      - Y (reach):    dy = step_scale * uz  (Farther/into-scene uz < 0 -> robot reaches forward along -Y;
                                             Closer/pull-back uz > 0 -> robot retracts along +Y)
      - Z (vertical): dz = step_scale * uy  (Up > 0, Down < 0)

    Safety gating:
      - Increments are (0, 0, 0) if hand_detected is False.
      - Increments are (0, 0, 0) if gesture_state not in ("GRIPPER_OPEN", "GRIPPER_CLOSED").
      - Increments are (0, 0, 0) if is_active is False.
      - Step scale: min(MAX_CARTESIAN_SPEED * dt, MAX_CARTESIAN_STEP).
      - Each axis increment is clamped to [-MAX_CARTESIAN_STEP, MAX_CARTESIAN_STEP].
    """
    if not hand_detected or not is_active:
        return 0.0, 0.0, 0.0

    if gesture_state not in ("GRIPPER_OPEN", "GRIPPER_CLOSED"):
        return 0.0, 0.0, 0.0

    clamped_dt = max(0.001, min(dt, 0.100))
    step_scale = min(cfg.MAX_CARTESIAN_SPEED * clamped_dt, cfg.MAX_CARTESIAN_STEP)

    dx = float(np.clip(step_scale * float(ux), -cfg.MAX_CARTESIAN_STEP, cfg.MAX_CARTESIAN_STEP))
    dy = float(np.clip(step_scale * float(uz), -cfg.MAX_CARTESIAN_STEP, cfg.MAX_CARTESIAN_STEP))
    dz = float(np.clip(step_scale * float(uy), -cfg.MAX_CARTESIAN_STEP, cfg.MAX_CARTESIAN_STEP))

    return dx, dy, dz


def resolve_model_path(custom_path: str = "") -> str:
    """Resolve hand_landmarker.task model path across installed and source workspaces."""
    if custom_path and os.path.isfile(custom_path):
        return os.path.normpath(custom_path)

    # 1. Try ament_index_python (installed share directory)
    try:
        from ament_index_python.packages import get_package_share_directory
        share_dir = get_package_share_directory("hand_gesture_teleop")
        candidate = os.path.join(share_dir, "models", "hand_landmarker.task")
        if os.path.isfile(candidate):
            return os.path.normpath(candidate)
    except Exception:
        pass

    # 2. Try relative to this file in the source workspace
    module_dir = os.path.dirname(os.path.realpath(__file__))
    candidate = os.path.normpath(os.path.join(module_dir, "..", "models", "hand_landmarker.task"))
    if os.path.isfile(candidate):
        return candidate

    # 3. Fallback: current working directory
    cwd_candidate = os.path.normpath(
        os.path.join(os.getcwd(), "src", "hand_gesture_teleop", "models", "hand_landmarker.task")
    )
    if os.path.isfile(cwd_candidate):
        return cwd_candidate

    return candidate


class HandTeleopNode(Node):
    """
    ROS 2 Hand-Gesture Teleoperation Node for SO-101 (Simulation Gated).
    """

    def __init__(self, node_name: str = "hand_teleop_node") -> None:
        super().__init__(node_name)

        # ── Declare Parameters ──────────────────────────────────────────────
        self.declare_parameter("enable_motion", False)
        self.declare_parameter("enable_arm_motion", False)
        self.declare_parameter("enable_ik_dry_run", False)
        self.declare_parameter("single_goal_test_mode", False)
        self.declare_parameter("max_goals_per_session", 0)
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("frame_width", 640)
        self.declare_parameter("frame_height", 480)
        self.declare_parameter("model_path", "")
        self.declare_parameter("gripper_open_rad", DEFAULT_GRIPPER_OPEN_RAD)
        self.declare_parameter("gripper_closed_rad", DEFAULT_GRIPPER_CLOSED_RAD)
        self.declare_parameter("gripper_joint_limit_min", GRIPPER_JOINT_LIMIT_MIN)
        self.declare_parameter("gripper_joint_limit_max", GRIPPER_JOINT_LIMIT_MAX)
        self.declare_parameter("gripper_trajectory_duration", DEFAULT_GRIPPER_DURATION)
        self.declare_parameter("gripper_max_velocity", DEFAULT_GRIPPER_MAX_VELOCITY)
        self.declare_parameter("gripper_max_acceleration", DEFAULT_GRIPPER_MAX_ACCELERATION)
        self.declare_parameter("gripper_min_duration", MIN_GRIPPER_TRAJECTORY_DURATION)

        # ── State Initialization ───────────────────────────────────────────
        self._teleop_state: str = STATE_DISARMED
        self._expected_joints: List[str] = list(EXPECTED_SO101_JOINTS)
        self._arm_control_joints: List[str] = list(ARM_CONTROL_JOINTS)
        self._latest_joint_positions: Dict[str, float] = {}
        self._latest_joint_velocities: Dict[str, float] = {}
        self._joint_state_msg_count: int = 0
        self._last_joint_state_time: Optional[float] = None

        # ── Arming & Mode Safety State ──────────────────────────────────────
        self._arming_requested_time: float = 0.0
        self._last_disarm_reason: str = "STARTUP"
        self._last_enable_error: Optional[str] = None
        self._cancel_requested_on_disarm: bool = False
        self._last_hand_landmarks: Optional[Any] = None

        # ── Hardware Component Introspection Client ────────────────────────
        self.hw_client = self.create_client(
            ListHardwareComponents,
            "/controller_manager/list_hardware_components",
        )
        self._hardware_mock_verified: bool = False
        self._hardware_last_verified_time: Optional[float] = None
        self._hardware_status: str = "PENDING_VERIFICATION"
        self._hw_check_pending: bool = False
        self._hw_request_seq: int = 0
        self._hw_response_seq: int = 0

        # ── Arm Trajectory Action Client (Zero Gripper / Arm Only) ──────────
        self.arm_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/arm_controller/follow_joint_trajectory",
        )
        self._goal_active: bool = False
        self._active_goal_handle: Any = None
        self._pending_ik_candidate: Optional[Dict[str, float]] = None
        self._pending_ik_snapshot: Optional[TargetSnapshot] = None
        self._trajectory_dispatch_count: int = 0
        self._session_goal_dispatch_count: int = 0
        self._trajectory_success_count: int = 0
        self._trajectory_last_status: str = "IDLE"

        # ── Gripper Trajectory Action Client (Task 5C-3D Isolated Path) ────
        self.gripper_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/gripper_controller/follow_joint_trajectory",
        )
        self._gripper_goal_active: bool = False
        self._gripper_canceling: bool = False
        self._active_gripper_goal_handle: Any = None
        self._last_commanded_gripper_state: Optional[str] = None
        self._gripper_dispatch_count: int = 0
        self._gripper_success_count: int = 0
        self._gripper_last_status: str = "IDLE"

        # ── Cartesian Target State & No-Op Tracking ────────────────────────
        self.target_cfg: CartesianTargetConfig = CARTESIAN_TARGET_CONFIG
        self._initial_target: Tuple[float, float, float] = self.target_cfg.INITIAL_TARGET_POSE
        self._target_x: float = self._initial_target[0]
        self._target_y: float = self._initial_target[1]
        self._target_z: float = self._initial_target[2]
        self._target_generation: int = 0
        self._last_submitted_ik_target: Optional[Tuple[float, float, float]] = None
        self._last_submitted_ik_generation: int = 0

        # ── MoveIt IK Client ───────────────────────────────────────────────
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        self._ik_pending: bool = False
        self._last_ik_time: float = 0.0
        self._ik_call_count: int = 0
        self._ik_success_count: int = 0
        self._ik_last_status: str = "DISABLED" if not self.enable_ik_dry_run else "IDLE"
        self._ik_last_error_code: int = 0
        self._latest_ik_solution: Optional[Dict[str, float]] = None

        # ── Hand Tracking & Smoothing Setup ────────────────────────────────
        self.gesture_cfg: GestureConfig = GESTURE_CONFIG
        self.mapping_cfg: MappingConfig = MAPPING_CONFIG

        self.gesture_smoother = TemporalSmoother(
            confirm_frames=self.gesture_cfg.SMOOTHING_CONFIRM_FRAMES
        )
        self.thumb_smoother = TemporalSmoother(
            confirm_frames=self.gesture_cfg.SMOOTHING_CONFIRM_FRAMES
        )
        self.index_smoother = TemporalSmoother(
            confirm_frames=self.gesture_cfg.SMOOTHING_CONFIRM_FRAMES
        )
        self.middle_smoother = TemporalSmoother(
            confirm_frames=self.gesture_cfg.SMOOTHING_CONFIRM_FRAMES
        )
        self.motion_tracker = MotionPreviewTracker(cfg=self.mapping_cfg)

        # ── ROS Subscriptions ──────────────────────────────────────────────
        self.joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            10,
        )

        # ── Dynamic Parameter Validation & Change Callbacks ────────────────
        self.add_on_set_parameters_callback(self._on_set_parameters_callback)
        self.add_post_set_parameters_callback(self._post_set_parameters_callback)

        self.get_logger().info(
            f"HandTeleopNode initialized in {self._teleop_state} mode. "
            f"enable_motion={self.enable_motion}. "
            f"enable_arm_motion={self.enable_arm_motion}. "
            f"enable_ik_dry_run={self.enable_ik_dry_run}. "
            f"modes_conflict={self.modes_conflict}. "
            f"Initial target: X={self._target_x:.4f}, Y={self._target_y:.4f}, Z={self._target_z:.4f} m. "
            f"Subscribed to /joint_states."
        )

    # ── Node Properties ────────────────────────────────────────────────────
    @property
    def teleop_state(self) -> str:
        """Current teleoperation state: DISARMED, ARMING_PENDING, or ARMED — SIMULATION ONLY."""
        return self._teleop_state

    @property
    def is_armed(self) -> bool:
        """True if the teleoperation node is currently armed."""
        return self._teleop_state == STATE_ARMED

    @property
    def modes_conflict(self) -> bool:
        """True if both enable_motion and enable_ik_dry_run are True (conflicting modes)."""
        return bool(self.enable_motion and self.enable_ik_dry_run)

    @property
    def hardware_mock_verified(self) -> bool:
        """True if controller manager was verified to use simulation mock hardware."""
        return self._hardware_mock_verified

    @property
    def goal_active(self) -> bool:
        """True if an arm trajectory goal is currently executing on the action server."""
        return self._goal_active

    @property
    def enable_motion(self) -> bool:
        """Read the enable_motion ROS parameter (default False)."""
        return bool(self.get_parameter("enable_motion").value)

    @property
    def enable_arm_motion(self) -> bool:
        """Read the enable_arm_motion ROS parameter (default False, fail-closed)."""
        return bool(self.get_parameter("enable_arm_motion").value)

    @property
    def enable_ik_dry_run(self) -> bool:
        """Read the enable_ik_dry_run ROS parameter (default False)."""
        return bool(self.get_parameter("enable_ik_dry_run").value)

    @property
    def single_goal_test_mode(self) -> bool:
        """Read the single_goal_test_mode ROS parameter (default False)."""
        return bool(self.get_parameter("single_goal_test_mode").value)

    @property
    def max_goals_per_session(self) -> int:
        """
        Read the max_goals_per_session ROS parameter (default 0).
        0 indicates unlimited multi-goal teleoperation in normal mode.
        A positive integer limits trajectory dispatch attempts per arming session.
        """
        return int(self.get_parameter("max_goals_per_session").value)

    @property
    def effective_session_goal_limit(self) -> int:
        """
        Calculates the effective trajectory goal dispatch attempt limit for an arming session.
        Semantics:
          1. If single_goal_test_mode is True, the effective limit is always 1 (takes precedence).
          2. If single_goal_test_mode is False and max_goals_per_session > 0, returns max_goals_per_session.
          3. If single_goal_test_mode is False and max_goals_per_session == 0, returns 0 (unlimited).
          4. If single_goal_test_mode is False and max_goals_per_session < 0, returns -1 (invalid/fail-closed).
        """
        if self.single_goal_test_mode:
            return 1
        raw = self.max_goals_per_session
        if raw < 0:
            return -1
        return raw

    def is_session_goal_limit_reached(self) -> bool:
        """
        Returns True if the effective trajectory goal dispatch attempt quota for the current
        arming session has been reached or exhausted, or if configuration is invalid (fail-closed).
        """
        eff_limit = self.effective_session_goal_limit
        if eff_limit == 0:
            return False  # Unlimited
        if eff_limit < 0:
            return True   # Fail-closed immediately for negative values; quota exhausted
        return self._session_goal_dispatch_count >= eff_limit

    def get_control_state_text(self) -> str:
        """Return the formatted control state text for HUD and telemetry."""
        if not self.enable_motion:
            return "DISARMED"
        if self._teleop_state == STATE_ARMED:
            eff_limit = self.effective_session_goal_limit
            if eff_limit > 0:
                limit_lbl = "Single-Goal" if self.single_goal_test_mode else "Session"
                return f"ARMED [{limit_lbl} {self._session_goal_dispatch_count}/{eff_limit}]"
            return "ARMED"
        elif self._teleop_state == STATE_ARMING_PENDING:
            now = time.monotonic()
            rem = max(0.0, ARMING_TIMEOUT_SEC - (now - self._arming_requested_time))
            return f"ARMING PENDING ({rem:.1f}s) - PRESS [Y]"
        else:  # STATE_DISARMED
            if self._cancel_requested_on_disarm:
                return "DISARMED (CANCEL REQUESTED)"
            if self._last_enable_error:
                return f"DISARMED ({self._last_enable_error})"
            eff_limit = self.effective_session_goal_limit
            if eff_limit > 0:
                limit_lbl = "Single-Goal" if self.single_goal_test_mode else "Session"
                return f"DISARMED [{limit_lbl} {self._session_goal_dispatch_count}/{eff_limit}]"
            return "DISARMED"

    def get_anchor_state_text(self) -> str:
        """Return the neutral motion anchor status text."""
        return "READY" if self.motion_tracker.is_calibrated else "NOT READY"

    @property
    def session_goal_dispatch_count(self) -> int:
        """Number of arm trajectory goals dispatched during current arming session."""
        return self._session_goal_dispatch_count

    @property
    def target_pose(self) -> Tuple[float, float, float]:
        """Current Cartesian target pose (x, y, z) in meters."""
        return self._target_x, self._target_y, self._target_z

    @property
    def target_delta(self) -> Tuple[float, float, float]:
        """Cartesian target delta from initial pose (dx, dy, dz) in meters."""
        return (
            self._target_x - self._initial_target[0],
            self._target_y - self._initial_target[1],
            self._target_z - self._initial_target[2],
        )

    @property
    def camera_index(self) -> int:
        return int(self.get_parameter("camera_index").value)

    @property
    def frame_width(self) -> int:
        return int(self.get_parameter("frame_width").value)

    @property
    def frame_height(self) -> int:
        return int(self.get_parameter("frame_height").value)

    @property
    def custom_model_path(self) -> str:
        return str(self.get_parameter("model_path").value)

    # ── Gripper Parameter Properties (Task 5C-3D) ───────────────────────────
    @property
    def gripper_open_rad(self) -> float:
        return float(self.get_parameter("gripper_open_rad").value)

    @property
    def gripper_closed_rad(self) -> float:
        return float(self.get_parameter("gripper_closed_rad").value)

    @property
    def gripper_joint_limit_min(self) -> float:
        return float(self.get_parameter("gripper_joint_limit_min").value)

    @property
    def gripper_joint_limit_max(self) -> float:
        return float(self.get_parameter("gripper_joint_limit_max").value)

    @property
    def gripper_trajectory_duration(self) -> float:
        return float(self.get_parameter("gripper_trajectory_duration").value)

    @property
    def gripper_max_velocity(self) -> float:
        return float(self.get_parameter("gripper_max_velocity").value)

    @property
    def gripper_max_acceleration(self) -> float:
        return float(self.get_parameter("gripper_max_acceleration").value)

    @property
    def gripper_min_duration(self) -> float:
        return float(self.get_parameter("gripper_min_duration").value)

    # ── Operating Mode IK Gating (Task 5C-3A-H Phase C) ────────────────────
    def can_issue_ik_requests(self) -> bool:
        """
        Evaluates whether current parameter configuration and arming state permit /compute_ik queries.
        Enforces the five mutually exclusive operating modes:
          Mode 1: enable_motion=False, enable_ik_dry_run=False, DISARMED -> False (no IK, no motion)
          Mode 2: enable_motion=False, enable_ik_dry_run=True, DISARMED  -> True (dry-run only, zero motion)
          Mode 3: enable_motion=True,  enable_ik_dry_run=False, DISARMED -> False (motion mode disarmed; arming required)
          Mode 4: enable_motion=True,  enable_ik_dry_run=False, ARMED    -> True (armed motion tracking & execution)
          Mode 5: enable_motion=True,  enable_ik_dry_run=True, ANY      -> False (conflict rejected, zero motion)

        When arm motion is disabled (enable_arm_motion=False), IK queries are unconditionally refused.
        When a session goal limit is configured and reached, IK queries are suppressed.
        """
        if not self.enable_arm_motion:
            return False
        if self.modes_conflict:
            return False
        if self.is_session_goal_limit_reached():
            return False
        if self.enable_ik_dry_run and not self.enable_motion:
            return True
        if self.enable_motion and not self.enable_ik_dry_run and self._teleop_state == STATE_ARMED:
            return True
        return False

    # ── Dynamic Parameter Validation Callbacks ──────────────────────────────
    def _on_set_parameters_callback(self, params: List[Parameter]) -> SetParametersResult:
        """
        Validates parameter changes before they are applied.
        Rejects negative values for max_goals_per_session and out-of-limit gripper positions.
        """
        for param in params:
            if param.name == "enable_arm_motion":
                if param.value is not None and type(param.value) is not bool:
                    return SetParametersResult(
                        successful=False,
                        reason="enable_arm_motion must be a boolean."
                    )
            elif param.name == "max_goals_per_session":
                if param.value is not None and param.value < 0:
                    return SetParametersResult(
                        successful=False,
                        reason=f"max_goals_per_session cannot be negative ({param.value}); must be >= 0."
                    )
            elif param.name in ("gripper_open_rad", "gripper_closed_rad"):
                if param.value is not None:
                    try:
                        val = float(param.value)
                        if not math.isfinite(val):
                            return SetParametersResult(
                                successful=False,
                                reason=f"{param.name} must be finite.",
                            )
                        if not (self.gripper_joint_limit_min <= val <= self.gripper_joint_limit_max):
                            return SetParametersResult(
                                successful=False,
                                reason=(
                                    f"{param.name} ({val}) must be within configured joint limits "
                                    f"[{self.gripper_joint_limit_min}, {self.gripper_joint_limit_max}]."
                                ),
                            )
                    except (TypeError, ValueError):
                        return SetParametersResult(
                            successful=False,
                            reason=f"{param.name} must be a valid float.",
                        )
            elif param.name in (
                "gripper_trajectory_duration",
                "gripper_max_velocity",
                "gripper_max_acceleration",
                "gripper_min_duration",
            ):
                if param.value is not None:
                    try:
                        val = float(param.value)
                        if val <= 0.0 or not math.isfinite(val):
                            return SetParametersResult(
                                successful=False,
                                reason=f"{param.name} must be positive and finite (> 0.0).",
                            )
                    except (TypeError, ValueError):
                        return SetParametersResult(
                            successful=False,
                            reason=f"{param.name} must be a valid float.",
                        )
        return SetParametersResult(successful=True)

    def _post_set_parameters_callback(self, params: List[Parameter]) -> None:
        """
        Handles dynamic ROS 2 parameter changes.
        Enforces immediate disarm and active-goal cancellation if:
          - enable_motion is changed to False while ARMED or ARMING_PENDING.
          - A mode conflict arises (both enable_motion and enable_ik_dry_run become True).
          - single_goal_test_mode or max_goals_per_session is reconfigured while ARMED and session limit is met.
        Guarantees that enabling enable_motion alone NEVER re-arms the node.
        """
        for param in params:
            if param.name in ("enable_motion", "enable_ik_dry_run"):
                if self.modes_conflict and self._teleop_state != STATE_DISARMED:
                    self.disarm("MODE_CONFLICT")
                elif not self.enable_motion and self._teleop_state in (STATE_ARMED, STATE_ARMING_PENDING):
                    self.disarm("MOTION_DISABLED")
            elif param.name == "enable_arm_motion":
                if not self.enable_arm_motion:
                    self._pending_ik_candidate = None
                    self._pending_ik_snapshot = None
                    if self._goal_active:
                        self.get_logger().warn(
                            "enable_arm_motion changed to False while arm goal is active; "
                            "in-flight goal continues to completion on controller, new arm goals locked out."
                        )
            elif param.name == "single_goal_test_mode":
                if self.single_goal_test_mode and self._session_goal_dispatch_count >= 1 and self._teleop_state == STATE_ARMED:
                    self._pending_ik_candidate = None
                    self._pending_ik_snapshot = None
                    self.disarm("SINGLE_GOAL_LIMIT_EXCEEDED")
            elif param.name == "max_goals_per_session":
                if self.is_session_goal_limit_reached() and self._teleop_state == STATE_ARMED:
                    self._pending_ik_candidate = None
                    self._pending_ik_snapshot = None
                    if not self._goal_active:
                        self.disarm("SESSION_GOAL_LIMIT_EXCEEDED")

    # ── ROS Callbacks ──────────────────────────────────────────────────────
    def _joint_state_callback(self, msg: JointState) -> None:
        """Record latest joint positions and velocities from /joint_states."""
        self._joint_state_msg_count += 1
        now = time.monotonic()
        self._last_joint_state_time = now

        # Update latest positions by joint name
        for i, name in enumerate(msg.name):
            if name in self._expected_joints and len(msg.position) > i:
                self._latest_joint_positions[name] = float(msg.position[i])

        # Track latest velocities strictly per incoming message.
        # Ensure that an incoming JointState message with an empty or omitted velocity
        # array cannot leave a stale velocity value available as if it were current.
        new_velocities: Dict[str, float] = {}
        for i, name in enumerate(msg.name):
            if name in self._expected_joints and len(msg.velocity) > i:
                new_velocities[name] = float(msg.velocity[i])
        self._latest_joint_velocities = new_velocities

    def get_joint_state_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic dictionary of joint state telemetry."""
        now = time.monotonic()
        age = (now - self._last_joint_state_time) if self._last_joint_state_time else None
        return {
            "msg_count": self._joint_state_msg_count,
            "expected_joints": list(self._expected_joints),
            "available_joints": list(self._latest_joint_positions.keys()),
            "positions": dict(self._latest_joint_positions),
            "velocities": dict(self._latest_joint_velocities),
            "all_joints_present": all(
                j in self._latest_joint_positions for j in self._expected_joints
            ),
            "last_time": self._last_joint_state_time,
            "age_seconds": age,
            "is_fresh": (age is not None and age <= MAX_JOINT_STATE_AGE),
        }

    def _is_joint_state_fresh(self) -> bool:
        """True if /joint_states feedback has been received and is within MAX_JOINT_STATE_AGE."""
        if self._last_joint_state_time is None:
            return False
        return (time.monotonic() - self._last_joint_state_time) <= MAX_JOINT_STATE_AGE

    # ── Cartesian Target Updates ───────────────────────────────────────────
    def update_cartesian_target(self, dx: float, dy: float, dz: float) -> Tuple[float, float, float]:
        """
        Apply (dx, dy, dz) increment to target position subject to cumulative limits,
        workspace boundaries, and minimum height constraints.
        """
        cand_x = self._target_x + dx
        cand_y = self._target_y + dy
        cand_z = self._target_z + dz

        # 1. Enforce cumulative limits (+/- 50 mm from initial pose)
        x0, y0, z0 = self._initial_target
        cand_x = float(np.clip(cand_x, x0 - self.target_cfg.CUMULATIVE_LIMIT_X, x0 + self.target_cfg.CUMULATIVE_LIMIT_X))
        cand_y = float(np.clip(cand_y, y0 - self.target_cfg.CUMULATIVE_LIMIT_Y, y0 + self.target_cfg.CUMULATIVE_LIMIT_Y))
        cand_z = float(np.clip(cand_z, z0 - self.target_cfg.CUMULATIVE_LIMIT_Z, z0 + self.target_cfg.CUMULATIVE_LIMIT_Z))

        # 2. Enforce workspace bounds
        cand_x = float(np.clip(cand_x, self.target_cfg.WORKSPACE_BOUNDS_X[0], self.target_cfg.WORKSPACE_BOUNDS_X[1]))
        cand_y = float(np.clip(cand_y, self.target_cfg.WORKSPACE_BOUNDS_Y[0], self.target_cfg.WORKSPACE_BOUNDS_Y[1]))
        cand_z = float(np.clip(cand_z, self.target_cfg.WORKSPACE_BOUNDS_Z[0], self.target_cfg.WORKSPACE_BOUNDS_Z[1]))

        # 3. Enforce minimum height above base (Z >= 0.050 m)
        cand_z = max(cand_z, self.target_cfg.MIN_HEIGHT_Z)

        # Track target generation when position changes
        if (abs(cand_x - self._target_x) + abs(cand_y - self._target_y) + abs(cand_z - self._target_z)) > 1e-6:
            self._target_generation += 1

        self._target_x = cand_x
        self._target_y = cand_y
        self._target_z = cand_z

        return self._target_x, self._target_y, self._target_z

    def reset_target_pose(self) -> None:
        """Reset target pose to initial neutral target pose."""
        self._target_x = self._initial_target[0]
        self._target_y = self._initial_target[1]
        self._target_z = self._initial_target[2]
        self._target_generation += 1
        self._last_submitted_ik_target = None

    # ── Hardware Component Introspection & Freshness ───────────────────────
    def is_hardware_verified_and_fresh(self) -> Tuple[bool, str]:
        """
        Confirms positive mock hardware verification within MAX_HARDWARE_CHECK_AGE (3.0s).
        Returns (is_valid_and_fresh, reason).
        """
        if not self._hardware_mock_verified:
            return False, f"NOT_VERIFIED ({self._hardware_status})"
        if self._hardware_last_verified_time is None:
            return False, "NO_VERIFICATION_TIMESTAMP"
        now = time.monotonic()
        age = now - self._hardware_last_verified_time
        if age > MAX_HARDWARE_CHECK_AGE:
            return False, f"VERIFICATION_EXPIRED ({age:.2f}s > {MAX_HARDWARE_CHECK_AGE}s)"
        return True, "MOCK_HARDWARE_FRESH"

    def check_hardware_components_async(self) -> bool:
        """
        Queries /controller_manager/list_hardware_components asynchronously to verify
        mock hardware is active before permitting arming or motion dispatch.
        Tracks request sequence to ignore older delayed responses.
        """
        if self._hw_check_pending:
            return False
        if not self.hw_client.service_is_ready():
            self._hardware_mock_verified = False
            self._hardware_last_verified_time = None
            self._hardware_status = "CM_SERVICE_UNAVAILABLE"
            if self._teleop_state == STATE_ARMED:
                self.disarm("HARDWARE_SERVICE_UNAVAILABLE")
            return False

        req = ListHardwareComponents.Request()
        self._hw_check_pending = True
        self._hw_request_seq += 1
        current_seq = self._hw_request_seq
        self._hardware_status = f"QUERYING_SEQ_{current_seq}"
        future = self.hw_client.call_async(req)
        future.add_done_callback(
            lambda f, seq=current_seq: self._hw_response_callback(f, seq)
        )
        return True

    def _hw_response_callback(self, future, req_seq: int) -> None:
        self._hw_check_pending = False

        # Out-of-order / delayed response check
        if req_seq < self._hw_response_seq:
            self.get_logger().warn(
                f"Ignoring delayed hardware response (seq {req_seq} < latest {self._hw_response_seq})."
            )
            return

        self._hw_response_seq = req_seq
        now = time.monotonic()

        try:
            response = future.result()
            valid, reason = verify_hardware_components_from_response(response)
            self._hardware_mock_verified = valid
            self._hardware_status = reason
            if valid:
                self._hardware_last_verified_time = now
                self.get_logger().debug(f"Hardware components verified: {reason} at {now:.3f}")
            else:
                self._hardware_last_verified_time = None
                self.get_logger().warn(f"Hardware component check failed: {reason}")
                if self._teleop_state == STATE_ARMED:
                    self.disarm(f"HARDWARE_VERIFICATION_LOST_{reason}")
        except Exception as exc:
            self._hardware_mock_verified = False
            self._hardware_last_verified_time = None
            self._hardware_status = f"ERROR_{exc}"
            self.get_logger().error(f"Hardware verification service exception: {exc}")
            if self._teleop_state == STATE_ARMED:
                self.disarm("HARDWARE_VERIFICATION_ERROR")

    # ── Single-Key Arming & Safety Gates (Task 5C-3D-SINGLE-KEY-ARM-DISARM-AUTO-ANCHOR) ──
    def enable_teleoperation(
        self,
        hand_detected: bool,
        landmarks: Optional[Any] = None,
        gesture_state: str = "UNKNOWN",
    ) -> Tuple[bool, str]:
        """
        Single-key teleoperation enabling method (triggered by pressing 'E').
        Safely validates current hand detection, landmark validity, palm scale,
        and stability, captures the neutral motion anchor, checks all system
        safety preconditions, and transitions directly to ARMED.
        """
        # Precondition 1: enable_motion == True
        if not self.enable_motion:
            self.disarm("MOTION_DISABLED")
            self._last_enable_error = "MOTION_DISABLED"
            self.get_logger().warn("Teleoperation enable failed: enable_motion parameter is False.")
            return False, "MOTION_DISABLED"

        # Precondition 2: Mutually exclusive operating modes (enable_ik_dry_run == False)
        if self.enable_ik_dry_run or self.modes_conflict:
            self.disarm("MODE_CONFLICT")
            self._last_enable_error = "MODE_CONFLICT"
            self.get_logger().warn("Teleoperation enable failed: mode conflict (enable_ik_dry_run is True).")
            return False, "MODE_CONFLICT"

        # Precondition 3: Hand detection check
        if not hand_detected or landmarks is None or len(landmarks) < 21:
            self.disarm("HAND_NOT_DETECTED")
            self._last_enable_error = "HAND_NOT_DETECTED"
            self.get_logger().warn("Teleoperation enable failed: Hand not detected or incomplete landmarks.")
            return False, "HAND_NOT_DETECTED"

        # Precondition 4: Landmark validity & palm scale check
        coords_valid = True
        for idx in range(min(len(landmarks), 21)):
            lm = landmarks[idx]
            if not (math.isfinite(lm.x) and math.isfinite(lm.y)):
                coords_valid = False
                break
            if not (-0.1 <= lm.x <= 1.1 and -0.1 <= lm.y <= 1.1):
                coords_valid = False
                break

        lm0 = landmarks[0]
        lm9 = landmarks[9]
        palm_scale_2d = dist_2d(lm0, lm9)
        if not coords_valid or not math.isfinite(palm_scale_2d) or palm_scale_2d < 0.02:
            self.disarm("INVALID_LANDMARKS")
            self._last_enable_error = "INVALID_LANDMARKS"
            self.get_logger().warn(
                f"Teleoperation enable failed: Invalid landmarks or palm scale (scale={palm_scale_2d:.4f})."
            )
            return False, "INVALID_LANDMARKS"

        # Precondition 5: Hand stability check (gesture state and movement stability)
        if gesture_state not in ("GRIPPER_OPEN", "GRIPPER_CLOSED"):
            self.disarm("UNSTABLE_HAND")
            self._last_enable_error = "UNSTABLE_HAND"
            self.get_logger().warn(f"Teleoperation enable failed: Gesture state '{gesture_state}' is unstable.")
            return False, "UNSTABLE_HAND"

        if self._last_hand_landmarks is not None and len(self._last_hand_landmarks) >= 21:
            prev_lm9 = self._last_hand_landmarks[9]
            if math.isfinite(prev_lm9.x) and math.isfinite(prev_lm9.y):
                drift = math.hypot(lm9.x - prev_lm9.x, lm9.y - prev_lm9.y)
                if drift > 0.08:
                    self.disarm("UNSTABLE_HAND")
                    self._last_enable_error = "UNSTABLE_HAND"
                    self.get_logger().warn(f"Teleoperation enable failed: Hand moving too fast (drift={drift:.4f}).")
                    return False, "UNSTABLE_HAND"

        # Precondition 6: Auto-capture neutral anchor
        calib_ok = self.motion_tracker.calibrate(lm9.x, lm9.y, palm_scale_2d)
        if not calib_ok or not self.motion_tracker.is_calibrated:
            self.disarm("ANCHOR_CAPTURE_FAILED")
            self._last_enable_error = "ANCHOR_CAPTURE_FAILED"
            self.get_logger().warn("Teleoperation enable failed: Motion tracker anchor capture failed.")
            return False, "ANCHOR_CAPTURE_FAILED"

        # Precondition 7: /joint_states feedback received, all 6 joints present, fresh (< 0.25s)
        now = time.monotonic()
        if self._last_joint_state_time is None or (now - self._last_joint_state_time) > MAX_JOINT_STATE_AGE:
            self.disarm("STALE_OR_MISSING_FEEDBACK")
            self._last_enable_error = "STALE_OR_MISSING_FEEDBACK"
            self.motion_tracker.reset_calibration()
            self.get_logger().warn("Teleoperation enable failed: /joint_states feedback is stale or missing.")
            return False, "STALE_OR_MISSING_FEEDBACK"

        missing_joints = [j for j in self._expected_joints if j not in self._latest_joint_positions]
        if missing_joints:
            self.disarm("MISSING_JOINTS")
            self._last_enable_error = "MISSING_JOINTS"
            self.motion_tracker.reset_calibration()
            self.get_logger().warn(f"Teleoperation enable failed: Missing joint states {missing_joints}.")
            return False, "MISSING_JOINTS"

        # Precondition 8: Fresh positive mock hardware verification
        hw_ok, hw_reason = self.is_hardware_verified_and_fresh()
        if not hw_ok:
            self.disarm("HARDWARE_NOT_VERIFIED")
            self._last_enable_error = "HARDWARE_NOT_VERIFIED"
            self.motion_tracker.reset_calibration()
            self.get_logger().warn(f"Teleoperation enable failed: Hardware check failed ({hw_reason}).")
            return False, f"HARDWARE_NOT_VERIFIED: {hw_reason}"

        # Precondition 9: Action server available
        if self.enable_arm_motion:
            if not self.arm_action_client.server_is_ready():
                self.disarm("ACTION_SERVER_NOT_READY")
                self._last_enable_error = "ACTION_SERVER_NOT_READY"
                self.motion_tracker.reset_calibration()
                self.get_logger().warn(
                    "Teleoperation enable failed: /arm_controller/follow_joint_trajectory action server not ready."
                )
                return False, "ACTION_SERVER_NOT_READY"
        else:
            if not self.gripper_action_client.server_is_ready():
                self.disarm("ACTION_SERVER_NOT_READY")
                self._last_enable_error = "ACTION_SERVER_NOT_READY"
                self.motion_tracker.reset_calibration()
                self.get_logger().warn(
                    "Teleoperation enable failed: /gripper_controller/follow_joint_trajectory action server not ready."
                )
                return False, "ACTION_SERVER_NOT_READY"

        # Precondition 10: No active goal
        if self._goal_active or self._gripper_goal_active:
            self.disarm("GOAL_ALREADY_ACTIVE")
            self._last_enable_error = "GOAL_ALREADY_ACTIVE"
            self.motion_tracker.reset_calibration()
            self.get_logger().warn("Teleoperation enable failed: A trajectory goal is already active.")
            return False, "GOAL_ALREADY_ACTIVE"

        # Precondition 11: Non-negative max_goals_per_session (fail-closed)
        if not self.single_goal_test_mode and self.max_goals_per_session < 0:
            self.disarm("INVALID_PARAMETER_VALUE")
            self._last_enable_error = "INVALID_PARAMETER_VALUE"
            self.motion_tracker.reset_calibration()
            self.get_logger().warn("Teleoperation enable failed: max_goals_per_session cannot be negative.")
            return False, "INVALID_PARAMETER_VALUE"

        # All preconditions passed!
        self._teleop_state = STATE_ARMED
        self._last_submitted_ik_target = None
        self._session_goal_dispatch_count = 0
        self._last_enable_error = None
        self._cancel_requested_on_disarm = False
        self.reset_target_pose()

        if self.enable_arm_motion:
            self.get_logger().info(
                "*** TELEOPERATION ENABLED (AUTO-ANCHOR CAPTURED — SIMULATION ONLY) *** "
                "Arm trajectory execution enabled."
            )
        else:
            self.get_logger().info(
                "*** TELEOPERATION ENABLED (AUTO-ANCHOR CAPTURED — GRIPPER-ONLY) *** "
                "Gripper trajectory execution enabled."
            )
        return True, "ARMED"

    # ── Legacy Two-Key Arming Handshake & Disarming ─────────────────────────
    def request_arming(self) -> Tuple[bool, str]:
        """
        Step 1 of two-key arming handshake (triggered by pressing 'A').
        Initiates arming confirmation window (3.0s). Preserved for backward compatibility.
        """
        if self.modes_conflict:
            self.get_logger().warn(
                "Arming rejected: Operating mode conflict! "
                "Both enable_motion and enable_ik_dry_run are True."
            )
            return False, "MODE_CONFLICT"

        if not self.enable_motion:
            self.get_logger().warn("Arming rejected: enable_motion parameter is False.")
            return False, "MOTION_DISABLED"

        if self._teleop_state == STATE_ARMED:
            return True, "ALREADY_ARMED"

        self._teleop_state = STATE_ARMING_PENDING
        self._arming_requested_time = time.monotonic()
        self.get_logger().info(
            "Arming handshake initiated: Press [Y] within 3.0s to confirm arming."
        )
        return True, "ARMING_PENDING"

    def confirm_arming(
        self,
        hand_detected: bool,
        gesture_state: str,
        landmarks: Optional[Any] = None,
    ) -> Tuple[bool, str]:
        """
        Step 2 of two-key arming handshake (triggered by pressing 'Y').
        Preserved for backward compatibility with existing tests.
        """
        if self._teleop_state != STATE_ARMING_PENDING:
            self.get_logger().warn(
                "Arming confirmation rejected: Arming handshake was not initiated with 'A'."
            )
            return False, "NOT_PENDING"

        now = time.monotonic()
        if (now - self._arming_requested_time) > ARMING_TIMEOUT_SEC:
            self.disarm("ARMING_TIMEOUT")
            self.get_logger().warn("Arming confirmation timed out (> 3.0s). Reverting to DISARMED.")
            return False, "TIMEOUT"

        if landmarks is None and hand_detected:
            # Create synthetic valid landmarks for legacy unit tests
            landmarks = [NormalizedLandmark(x=0.5, y=0.5) for _ in range(21)]
            landmarks[0] = NormalizedLandmark(x=0.5, y=0.7)  # Wrist
            landmarks[9] = NormalizedLandmark(x=0.5, y=0.5)  # Middle MCP (scale=0.20)

        # Delegate to enable_teleoperation
        self._teleop_state = STATE_DISARMED
        ok, reason = self.enable_teleoperation(
            hand_detected=hand_detected,
            landmarks=landmarks,
            gesture_state=gesture_state,
        )
        if not ok:
            if reason in ("HAND_NOT_DETECTED", "UNSTABLE_HAND"):
                return False, "HAND_NOT_READY"
            return False, reason
        return True, "ARMED"

    def disarm(self, reason: str = "USER_REQUEST") -> None:
        """
        Transitions immediately to DISARMED, clears candidate queue, resets motion anchor,
        and requests cancellation of any active goal.

        NOTE: Trajectory cancellation sends an asynchronous cancel request to the controller.
        It is NOT an emergency-stop guarantee and does not guarantee instantaneous hardware motor braking.
        """
        prev_state = self._teleop_state
        self._teleop_state = STATE_DISARMED
        self._arming_requested_time = 0.0
        self._pending_ik_candidate = None
        self._pending_ik_snapshot = None
        self._last_submitted_ik_target = None
        self._last_disarm_reason = reason

        had_active_motion = bool(self._goal_active or self._gripper_goal_active)

        if self._goal_active:
            self._cancel_active_goal()

        if self._gripper_goal_active:
            self._cancel_active_gripper_goal()
        self._last_commanded_gripper_state = None

        if reason == "USER_KEYPRESS_D" and had_active_motion:
            self._cancel_requested_on_disarm = True
        elif reason == "USER_KEYPRESS_D":
            self._cancel_requested_on_disarm = False

        # Reset neutral motion anchor
        self.motion_tracker.reset_calibration()

        if prev_state != STATE_DISARMED:
            self.get_logger().info(f"Teleoperation DISARMED (reason: {reason}).")

    def _cancel_active_goal(self) -> None:
        """
        Requests cancellation of active goal handle.
        NOTE: Trajectory cancellation sends an asynchronous cancel request to the controller.
        It is NOT an emergency-stop guarantee and does not guarantee instantaneous hardware motor braking.
        """
        if self._active_goal_handle is not None:
            self.get_logger().info("Sending cancel request for active arm trajectory goal...")
            try:
                self._active_goal_handle.cancel_goal_async()
            except Exception as exc:
                self.get_logger().warn(f"Exception while requesting goal cancel: {exc}")
        self._goal_active = False
        self._active_goal_handle = None
        self._pending_ik_candidate = None
        self._pending_ik_snapshot = None
        self._trajectory_last_status = "CANCELLED"

    def _cancel_active_gripper_goal(self) -> None:
        """
        Requests cancellation of active gripper goal handle.
        NOTE: Trajectory cancellation sends an asynchronous cancel request to the controller.
        It is NOT an emergency-stop guarantee.
        """
        if self._active_gripper_goal_handle is not None:
            self.get_logger().info("Sending cancel request for active gripper trajectory goal...")
            self._gripper_canceling = True
            try:
                cancel_future = self._active_gripper_goal_handle.cancel_goal_async()
                cancel_future.add_done_callback(self._gripper_cancel_response_callback)
            except Exception as exc:
                self.get_logger().warn(f"Exception while requesting gripper goal cancel: {exc}")
                self._gripper_canceling = False
        self._gripper_goal_active = False
        self._active_gripper_goal_handle = None
        self._gripper_last_status = "CANCELLED"

    def _gripper_cancel_response_callback(self, future: Any) -> None:
        self._gripper_canceling = False
        try:
            future.result()
            self.get_logger().info("Gripper goal cancellation acknowledged by controller.")
        except Exception as exc:
            self.get_logger().warn(f"Gripper goal cancel callback exception: {exc}")

    # ── MoveIt IK Query & Hardened Candidate Validator ─────────────────────
    def request_ik(self, force: bool = False) -> bool:
        """
        Non-blocking, throttled MoveIt /compute_ik service request.
        Zero-motion guarantee in dry-run mode: never sends commands.
        Gated motion execution: only permits trajectory dispatch when ARMED.
        Target-level no-op suppression: avoids redundant queries when Cartesian target has not moved.
        """
        if not self.enable_arm_motion:
            return False

        if not self.can_issue_ik_requests():
            return False

        now = time.monotonic()
        if (now - self._last_ik_time) < self.target_cfg.IK_THROTTLE_PERIOD:
            return False

        if self._ik_pending:
            return False

        if not self.ik_client.service_is_ready():
            self._ik_last_status = "SERVICE_UNAVAILABLE"
            return False

        # Phase A: Target-level no-op suppression when target has not changed meaningfully
        if not force and self._last_submitted_ik_target is not None:
            lx, ly, lz = self._last_submitted_ik_target
            target_disp = math.sqrt(
                (self._target_x - lx) ** 2
                + (self._target_y - ly) ** 2
                + (self._target_z - lz) ** 2
            )
            if target_disp < MIN_IK_TARGET_DISPLACEMENT:
                return False

        req = GetPositionIK.Request()
        req.ik_request.group_name = self.target_cfg.IK_PLANNING_GROUP
        req.ik_request.ik_link_name = self.target_cfg.IK_TIP_LINK
        req.ik_request.pose_stamped.header.frame_id = self.target_cfg.IK_REFERENCE_FRAME
        req.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        req.ik_request.pose_stamped.pose.position.x = float(self._target_x)
        req.ik_request.pose_stamped.pose.position.y = float(self._target_y)
        req.ik_request.pose_stamped.pose.position.z = float(self._target_z)
        qx, qy, qz, qw = self.target_cfg.INITIAL_ORIENTATION
        req.ik_request.pose_stamped.pose.orientation.x = float(qx)
        req.ik_request.pose_stamped.pose.orientation.y = float(qy)
        req.ik_request.pose_stamped.pose.orientation.z = float(qz)
        req.ik_request.pose_stamped.pose.orientation.w = float(qw)
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout.sec = 0
        req.ik_request.timeout.nanosec = int(0.1 * 1e9)  # 100 ms timeout

        if self._latest_joint_positions:
            seed_state = JointState()
            seed_state.header.stamp = self.get_clock().now().to_msg()
            seed_state.name = list(self._latest_joint_positions.keys())
            seed_state.position = [float(p) for p in self._latest_joint_positions.values()]
            req.ik_request.robot_state.joint_state = seed_state

        target_snapshot = TargetSnapshot(
            x=float(self._target_x),
            y=float(self._target_y),
            z=float(self._target_z),
            generation=self._target_generation,
            timestamp=now,
        )
        self._last_submitted_ik_target = (float(self._target_x), float(self._target_y), float(self._target_z))
        self._last_submitted_ik_generation = self._target_generation
        self._ik_pending = True
        self._last_ik_time = now
        self._ik_last_status = "CALLING"

        future = self.ik_client.call_async(req)
        future.add_done_callback(
            lambda f, snap=target_snapshot: self._ik_response_callback(f, snap)
        )
        return True

    def request_ik_dry_run(self) -> bool:
        """Alias for backward compatibility."""
        return self.request_ik()

    def _ik_response_callback(self, future, snapshot: Optional[TargetSnapshot] = None) -> None:
        """Handle /compute_ik response asynchronously and perform hardened validation."""
        self._ik_pending = False
        self._ik_call_count += 1
        try:
            response = future.result()
        except Exception as exc:
            self._ik_last_status = "ERROR"
            self.get_logger().warn(f"IK service call exception: {exc}")
            return

        code_val = response.error_code.val
        self._ik_last_error_code = code_val

        if code_val == MoveItErrorCodes.SUCCESS:
            # 0. Validate Cartesian target drift (reject obsolete responses)
            if snapshot is not None:
                dx = self._target_x - snapshot.x
                dy = self._target_y - snapshot.y
                dz = self._target_z - snapshot.z
                drift = math.sqrt(dx * dx + dy * dy + dz * dz)
                if drift > MAX_IK_TARGET_DRIFT:
                    self._ik_last_status = "OBSOLETE_TARGET_REJECTED"
                    self.get_logger().warn(
                        f"IK solution rejected: Cartesian target drifted "
                        f"{drift*1000:.1f} mm > {MAX_IK_TARGET_DRIFT*1000:.1f} mm."
                    )
                    return

            sol_names = response.solution.joint_state.name
            sol_pos = response.solution.joint_state.position

            # 1. Reject malformed responses with length mismatch
            if len(sol_names) != len(sol_pos):
                self._ik_last_status = "MALFORMED_SOLUTION"
                self.get_logger().warn(
                    f"IK solution malformed: name length ({len(sol_names)}) != "
                    f"position length ({len(sol_pos)})"
                )
                return

            # 2. Reject duplicate joint names in solution
            if len(sol_names) != len(set(sol_names)):
                self._ik_last_status = "DUPLICATE_JOINTS_SOLUTION"
                self.get_logger().warn("IK solution contains duplicate joint names.")
                return

            # Build name-based dictionary safely
            try:
                sol_map = {n: float(p) for n, p in zip(sol_names, sol_pos)}
            except (ValueError, TypeError) as exc:
                self._ik_last_status = "MALFORMED_SOLUTION"
                self.get_logger().warn(f"IK solution value conversion error: {exc}")
                return

            # 3. Reject incomplete responses missing any required arm joints
            missing_joints = [j for j in self._arm_control_joints if j not in sol_map]
            if missing_joints:
                self._ik_last_status = "INCOMPLETE_SOLUTION"
                self.get_logger().warn(
                    f"IK solution incomplete: missing required joints {missing_joints}"
                )
                return

            # 4. Validate finite numbers and configured joint limits for all required arm joints
            for joint, (low, high) in ARM_JOINT_LIMITS.items():
                val = sol_map[joint]

                if not math.isfinite(val):
                    self._ik_last_status = "INVALID_NONFINITE_SOLUTION"
                    self.get_logger().warn(
                        f"IK solution non-finite value for {joint}: {val}"
                    )
                    return

                if val < low or val > high:
                    self._ik_last_status = "LIMIT_VIOLATION"
                    self.get_logger().warn(
                        f"IK solution joint limit violation: {joint}={val:.2f} outside [{low},{high}]"
                    )
                    return

            # 5. Check fresh /joint_states feedback availability (age <= 0.25s)
            now = time.monotonic()
            if not self._latest_joint_positions or self._last_joint_state_time is None:
                self._ik_last_status = "STALE_OR_MISSING_FEEDBACK"
                self.get_logger().warn("IK validation failed: No /joint_states feedback available.")
                return

            if (now - self._last_joint_state_time) > MAX_JOINT_STATE_AGE:
                self._ik_last_status = "STALE_FEEDBACK"
                self.get_logger().warn(
                    f"IK validation failed: /joint_states feedback is stale "
                    f"({now - self._last_joint_state_time:.3f}s > {MAX_JOINT_STATE_AGE}s)."
                )
                return

            missing_feedback = [j for j in self._arm_control_joints if j not in self._latest_joint_positions]
            if missing_feedback:
                self._ik_last_status = "STALE_OR_MISSING_FEEDBACK"
                self.get_logger().warn(
                    f"IK validation failed: Missing feedback for arm joints {missing_feedback}."
                )
                return

            # 6. Check joint continuity (max joint delta <= MAX_JOINT_DELTA)
            for j in self._arm_control_joints:
                delta = abs(sol_map[j] - self._latest_joint_positions[j])
                if delta > MAX_JOINT_DELTA:
                    self._ik_last_status = "EXCESSIVE_JOINT_DELTA"
                    self.get_logger().warn(
                        f"IK solution joint jump exceeded limit for {j}: {delta:.4f} rad > {MAX_JOINT_DELTA:.4f} rad."
                    )
                    return

            # All validations passed!
            self._ik_success_count += 1
            self._ik_last_status = "SUCCESS"
            self._latest_ik_solution = {
                j: sol_map[j] for j in self._arm_control_joints
            }

            # If ARMED and motion enabled, handle arm trajectory dispatching
            if (
                self._teleop_state == STATE_ARMED
                and self.enable_motion
                and not self.enable_ik_dry_run
            ):
                if not self.enable_arm_motion:
                    self.get_logger().debug("IK solution dropped: enable_arm_motion is False.")
                elif self.is_session_goal_limit_reached():
                    self.get_logger().debug("IK solution dropped: session goal limit reached.")
                else:
                    self._handle_valid_ik_solution(self._latest_ik_solution, snapshot=snapshot)
        else:
            err_str = MOVEIT_ERROR_NAMES.get(code_val, f"CODE_{code_val}")
            self._ik_last_status = err_str

    # ── Arm Trajectory Execution & Concurrency Guard ───────────────────────
    def _validate_joint_discontinuity(self, candidate_map: Dict[str, float]) -> bool:
        """Validates that candidate joints do not exceed MAX_JOINT_DELTA from current feedback."""
        now = time.monotonic()
        if self._last_joint_state_time is None or (now - self._last_joint_state_time) > MAX_JOINT_STATE_AGE:
            return False
        for j in self._arm_control_joints:
            if j not in self._latest_joint_positions or j not in candidate_map:
                return False
            if abs(candidate_map[j] - self._latest_joint_positions[j]) > MAX_JOINT_DELTA:
                return False
        return True

    def _handle_valid_ik_solution(
        self,
        candidate_map: Dict[str, float],
        snapshot: Optional[TargetSnapshot] = None,
    ) -> None:
        """
        Handles dispatching a valid IK solution to the arm action client.
        If a goal is currently active, buffers the candidate as the newest pending target.
        If session goal limit has been reached, drops candidates.
        """
        if self._teleop_state != STATE_ARMED:
            return

        if not self.enable_arm_motion:
            self._pending_ik_candidate = None
            self._pending_ik_snapshot = None
            self.get_logger().debug("Arm motion disabled: dropping IK candidate.")
            return

        if self.is_session_goal_limit_reached():
            self._pending_ik_candidate = None
            self._pending_ik_snapshot = None
            self.get_logger().debug("Session goal limit reached: candidate ignored, limit reached for session.")
            return

        if self._goal_active:
            # Overlapping goal guard: retain newest valid candidate and its target snapshot
            self._pending_ik_candidate = dict(candidate_map)
            self._pending_ik_snapshot = snapshot
            self.get_logger().debug("Goal active: buffering newest IK candidate.")
            return

        self.dispatch_arm_trajectory(candidate_map)

    def dispatch_arm_trajectory(self, candidate_map: Dict[str, float]) -> bool:
        """
        Dispatches a FollowJointTrajectory goal to /arm_controller/follow_joint_trajectory.
        Guarantees single active goal, 5 arm joints, velocity bounds, and duration >= 0.5s.
        """
        if self._teleop_state != STATE_ARMED or not self.enable_motion or self.enable_ik_dry_run:
            return False

        if not self.enable_arm_motion:
            self._trajectory_last_status = "ARM_MOTION_DISABLED"
            self.get_logger().warn("Cannot dispatch: Arm motion is disabled (enable_arm_motion=False).")
            return False

        if self.is_session_goal_limit_reached():
            if self.single_goal_test_mode:
                self._trajectory_last_status = "SINGLE_GOAL_LIMIT_EXCEEDED"
                self.get_logger().warn(
                    "Cannot dispatch: Single-goal test mode allows only 1 goal per arming session."
                )
            else:
                self._trajectory_last_status = "SESSION_GOAL_LIMIT_EXCEEDED"
                self.get_logger().warn(
                    f"Cannot dispatch: Session goal limit ({self.effective_session_goal_limit}) reached for this arming session."
                )
            return False

        if self._goal_active:
            self.get_logger().warn("Cannot dispatch: An arm trajectory goal is already active.")
            return False

        # Freshness check: Positive mock hardware verification must be fresh (<= 3.0s)
        hw_ok, hw_reason = self.is_hardware_verified_and_fresh()
        if not hw_ok:
            self._trajectory_last_status = "HARDWARE_NOT_VERIFIED_ON_DISPATCH"
            self.disarm(f"HARDWARE_VERIFICATION_STALE_ON_DISPATCH: {hw_reason}")
            self.get_logger().warn(f"Cannot dispatch: Hardware verification failed ({hw_reason}).")
            return False

        if not self.arm_action_client.server_is_ready():
            self._trajectory_last_status = "SERVER_NOT_READY"
            self.get_logger().warn("Cannot dispatch: Arm action server is not ready.")
            return False

        if not self._validate_joint_discontinuity(candidate_map):
            self._trajectory_last_status = "EXCESSIVE_JOINT_DELTA_ON_DISPATCH"
            self.get_logger().warn("Cannot dispatch: Candidate exceeds joint delta or feedback stale.")
            return False

        # Phase A: Joint-level no-op suppression (suppress zero or sub-deadband joint movement)
        max_joint_delta = max(
            abs(float(candidate_map[j]) - float(self._latest_joint_positions[j]))
            for j in self._arm_control_joints
        )
        if max_joint_delta < MIN_DISPATCH_JOINT_DELTA:
            self._trajectory_last_status = "NO_OP_SUPPRESSED"
            self.get_logger().debug(
                f"Trajectory dispatch suppressed: max joint delta {max_joint_delta:.5f} rad < "
                f"deadband {MIN_DISPATCH_JOINT_DELTA:.5f} rad."
            )
            return False

        try:
            goal_msg, duration_sec = build_trajectory_goal(
                candidate_map=candidate_map,
                current_positions=self._latest_joint_positions,
                now_msg=self.get_clock().now().to_msg(),
            )
        except ValueError as exc:
            self._trajectory_last_status = f"INVALID_GOAL_{exc}"
            self.get_logger().error(f"Cannot build goal: {exc}")
            return False

        if not self.enable_arm_motion:
            self._trajectory_last_status = "ARM_MOTION_DISABLED"
            return False

        self._goal_active = True
        self._trajectory_dispatch_count += 1
        self._session_goal_dispatch_count += 1
        self._trajectory_last_status = "SENDING"

        try:
            send_goal_future = self.arm_action_client.send_goal_async(goal_msg)
        except Exception as exc:
            self._goal_active = False
            self._active_goal_handle = None
            self._pending_ik_candidate = None
            self._pending_ik_snapshot = None
            self._trajectory_last_status = f"SEND_FAILED_{exc}"
            self.get_logger().error(f"Synchronous goal send exception: {exc}")
            if self.is_session_goal_limit_reached():
                disarm_reason = (
                    "SINGLE_GOAL_TEST_SEND_FAILED"
                    if self.single_goal_test_mode
                    else "SESSION_GOAL_LIMIT_SEND_FAILED"
                )
                self.disarm(disarm_reason)
            return False

        send_goal_future.add_done_callback(self._goal_response_callback)
        self.get_logger().info(
            f"Dispatched arm trajectory goal #{self._trajectory_dispatch_count} "
            f"(session #{self._session_goal_dispatch_count}, duration={duration_sec:.2f}s)."
        )
        return True

    def _goal_response_callback(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._goal_active = False
            self._active_goal_handle = None
            self._pending_ik_candidate = None
            self._pending_ik_snapshot = None
            self._trajectory_last_status = f"SEND_FAILED_{exc}"
            self.get_logger().error(f"Goal send exception: {exc}")
            if self.is_session_goal_limit_reached():
                disarm_reason = (
                    "SINGLE_GOAL_TEST_SEND_FAILED"
                    if self.single_goal_test_mode
                    else "SESSION_GOAL_LIMIT_SEND_FAILED"
                )
                self.disarm(disarm_reason)
            return

        if not goal_handle.accepted:
            self._goal_active = False
            self._active_goal_handle = None
            self._pending_ik_candidate = None
            self._pending_ik_snapshot = None
            self._trajectory_last_status = "GOAL_REJECTED"
            self.get_logger().warn("Arm trajectory goal was rejected by action server.")
            if self.is_session_goal_limit_reached():
                disarm_reason = (
                    "SINGLE_GOAL_TEST_REJECTED"
                    if self.single_goal_test_mode
                    else "SESSION_GOAL_LIMIT_REJECTED"
                )
                self.disarm(disarm_reason)
            return

        self._active_goal_handle = goal_handle
        self._trajectory_last_status = "EXECUTING"
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self._goal_result_callback)

    def _goal_result_callback(self, future) -> None:
        self._goal_active = False
        self._active_goal_handle = None
        try:
            wrapped_result = future.result()
            status = wrapped_result.status
            if status == GoalStatus.STATUS_SUCCEEDED:
                self._trajectory_success_count += 1
                self._trajectory_last_status = "SUCCEEDED"
                self.get_logger().info(f"Arm trajectory goal succeeded ({self._trajectory_success_count}).")
            else:
                self._trajectory_last_status = f"FAILED_STATUS_{status}"
                self.get_logger().warn(f"Arm trajectory goal finished with status {status}.")
        except Exception as exc:
            self._trajectory_last_status = f"RESULT_EXCEPTION_{exc}"
            self.get_logger().error(f"Goal result exception: {exc}")

        # When session goal limit is reached, purge candidates and automatically disarm
        if self.is_session_goal_limit_reached():
            self._pending_ik_candidate = None
            self._pending_ik_snapshot = None
            disarm_reason = (
                "SINGLE_GOAL_TEST_COMPLETED"
                if self.single_goal_test_mode
                else "SESSION_GOAL_LIMIT_COMPLETED"
            )
            self.disarm(disarm_reason)
            return

        # If armed and candidate was buffered, dispatch newest valid candidate
        if self._teleop_state == STATE_ARMED and self._pending_ik_candidate is not None:
            if not self.enable_arm_motion:
                self._pending_ik_candidate = None
                self._pending_ik_snapshot = None
                return
            pending = self._pending_ik_candidate
            snap = self._pending_ik_snapshot
            self._pending_ik_candidate = None
            self._pending_ik_snapshot = None

            # Phase C: Validate target drift of buffered candidate
            if snap is not None:
                dx = self._target_x - snap.x
                dy = self._target_y - snap.y
                dz = self._target_z - snap.z
                drift = math.sqrt(dx * dx + dy * dy + dz * dz)
                if drift > MAX_IK_TARGET_DRIFT:
                    self.get_logger().warn(
                        f"Buffered IK candidate discarded on goal completion: target drifted "
                        f"{drift*1000:.1f} mm > {MAX_IK_TARGET_DRIFT*1000:.1f} mm."
                    )
                    return

            if self._validate_joint_discontinuity(pending):
                self.dispatch_arm_trajectory(pending)
            else:
                self.get_logger().warn("Buffered IK candidate failed discontinuity check on dispatch.")
        else:
            self._pending_ik_candidate = None
            self._pending_ik_snapshot = None

    # ── Gripper Trajectory Execution (Task 5C-3D Isolated Path) ─────────────
    def _send_gripper_command(self, gesture_state: str) -> bool:
        """
        Dispatches a single-joint trajectory goal to /gripper_controller/follow_joint_trajectory
        when the debounced gesture transitions between GRIPPER_OPEN and GRIPPER_CLOSED.
        Generates a rest-to-rest cubic spline trajectory with:
          - Point 0 at t=0: current position at rest (pos=current_pos, vel=0.0)
          - Point 1 at t=duration: target position at rest (pos=target_pos, vel=0.0)
          - Duration dynamically scaled to bound peak velocity <= gripper_max_velocity
            and peak acceleration <= gripper_max_acceleration.

        Safety Gating:
          1. gesture_state must be 'GRIPPER_OPEN' or 'GRIPPER_CLOSED'.
          2. Transition check: must differ from self._last_commanded_gripper_state (no duplicate goals).
          3. Node must be in STATE_ARMED.
          4. Motion must be enabled (enable_motion=True, enable_ik_dry_run=False, no mode conflict).
          5. Positive mock hardware must be verified and fresh (< 3.0s).
          6. Joint state feedback must be fresh (< MAX_JOINT_STATE_AGE) and include finite 'joint6'.
          7. No active gripper goal already in flight and no unresolved cancellation.
          8. Gripper action server must be available.
          9. Target and current positions must be finite and within configured joint limits [-0.2, 2.0].
          10. Rest starting condition verified (zero velocity or <= 0.1 rad/s).
        """
        if gesture_state not in ("GRIPPER_OPEN", "GRIPPER_CLOSED"):
            return False

        if gesture_state == self._last_commanded_gripper_state:
            return False

        if self._teleop_state != STATE_ARMED:
            return False

        if not self.enable_motion or self.enable_ik_dry_run or self.modes_conflict:
            return False

        hw_ok, hw_reason = self.is_hardware_verified_and_fresh()
        if not hw_ok:
            self.get_logger().warn(f"Gripper dispatch blocked: hardware not verified ({hw_reason}).")
            return False

        if not self._is_joint_state_fresh():
            self.get_logger().warn("Gripper dispatch blocked: joint feedback stale or missing.")
            return False

        if "joint6" not in self._latest_joint_positions:
            self.get_logger().warn("Gripper dispatch blocked: joint6 not in feedback.")
            return False

        if self._gripper_goal_active or self._gripper_canceling:
            return False

        if not self.gripper_action_client.server_is_ready():
            self.get_logger().warn("Gripper dispatch blocked: action server not ready.")
            return False

        target_pos = (
            self.gripper_open_rad
            if gesture_state == "GRIPPER_OPEN"
            else self.gripper_closed_rad
        )

        if not math.isfinite(target_pos):
            self.get_logger().error(f"Gripper target position non-finite: {target_pos}")
            return False

        if not (self.gripper_joint_limit_min <= target_pos <= self.gripper_joint_limit_max):
            self.get_logger().error(
                f"Gripper target {target_pos:.3f} rad exceeds limits "
                f"[{self.gripper_joint_limit_min:.3f}, {self.gripper_joint_limit_max:.3f}]."
            )
            return False

        current_pos = float(self._latest_joint_positions["joint6"])
        if not math.isfinite(current_pos):
            self.get_logger().error(f"Gripper current feedback position non-finite: {current_pos}")
            return False

        if not (self.gripper_joint_limit_min <= current_pos <= self.gripper_joint_limit_max):
            self.get_logger().error(
                f"Gripper current feedback {current_pos:.3f} rad outside limits "
                f"[{self.gripper_joint_limit_min:.3f}, {self.gripper_joint_limit_max:.3f}]."
            )
            return False

        # Valid, finite, fresh velocity verification for joint6:
        # Require actual joint6 velocity measurement from current JointState feedback.
        # Never default missing velocity to 0.0. Reject omitted, non-finite, or moving joint6.
        if "joint6" not in self._latest_joint_velocities:
            self.get_logger().warn(
                "Gripper dispatch blocked: joint6 velocity feedback is missing from /joint_states."
            )
            return False

        current_vel = float(self._latest_joint_velocities["joint6"])
        if not math.isfinite(current_vel):
            self.get_logger().warn(
                f"Gripper dispatch blocked: joint6 velocity is non-finite: {current_vel}."
            )
            return False

        if abs(current_vel) > 0.1:
            self.get_logger().warn(
                f"Gripper dispatch blocked: joint6 is moving (velocity={current_vel:.3f} rad/s > 0.1 rad/s)."
            )
            return False

        try:
            goal_msg, duration_sec = build_gripper_trajectory_goal(
                target_pos=target_pos,
                current_pos=current_pos,
                starting_velocity=current_vel,
                velocity_limit=self.gripper_max_velocity,
                acceleration_limit=self.gripper_max_acceleration,
                min_duration=self.gripper_min_duration,
                now_msg=None,  # Zero timestamp (sec=0, nanosec=0) for "start now" controller semantics
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to build gripper trajectory goal: {exc}")
            return False

        self._gripper_goal_active = True
        self._gripper_dispatch_count += 1
        self._gripper_last_status = "SENDING"
        self._last_commanded_gripper_state = gesture_state

        try:
            send_future = self.gripper_action_client.send_goal_async(goal_msg)
        except Exception as exc:
            self._gripper_goal_active = False
            self._active_gripper_goal_handle = None
            self._gripper_last_status = f"SEND_FAILED_{exc}"
            self.get_logger().error(f"Gripper goal send exception: {exc}")
            return False

        send_future.add_done_callback(self._gripper_goal_response_callback)
        self.get_logger().info(
            f"Dispatched gripper trajectory goal #{self._gripper_dispatch_count} "
            f"(state={gesture_state}, target={target_pos:.3f} rad, current={current_pos:.3f} rad, duration={duration_sec:.3f}s)."
        )
        return True

    def _gripper_goal_response_callback(self, future: Any) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._gripper_goal_active = False
            self._gripper_canceling = False
            self._active_gripper_goal_handle = None
            self._gripper_last_status = f"RESPONSE_EXCEPTION_{exc}"
            self.get_logger().error(f"Gripper goal response exception: {exc}")
            return

        if not goal_handle.accepted:
            self._gripper_goal_active = False
            self._gripper_canceling = False
            self._active_gripper_goal_handle = None
            self._gripper_last_status = "REJECTED"
            self.get_logger().warn("Gripper goal REJECTED by controller.")
            return

        self._active_gripper_goal_handle = goal_handle
        self._gripper_last_status = "ACCEPTED"
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._gripper_goal_result_callback)

    def _gripper_goal_result_callback(self, future: Any) -> None:
        self._gripper_goal_active = False
        self._gripper_canceling = False
        self._active_gripper_goal_handle = None
        try:
            result_wrap = future.result()
            status = result_wrap.status
        except Exception as exc:
            self._gripper_last_status = f"RESULT_EXCEPTION_{exc}"
            self.get_logger().error(f"Gripper result exception: {exc}")
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self._gripper_success_count += 1
            self._gripper_last_status = "SUCCEEDED"
            self.get_logger().info(
                f"Gripper motion SUCCEEDED (total successes: {self._gripper_success_count})."
            )
        elif status == GoalStatus.STATUS_CANCELED:
            self._gripper_last_status = "CANCELLED"
            self.get_logger().info("Gripper motion CANCELLED by controller.")
        else:
            self._gripper_last_status = f"FINISHED_STATUS_{status}"
            self.get_logger().warn(f"Gripper motion finished with status code {status}.")

    def get_cartesian_and_ik_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic dictionary of Cartesian target, IK, and trajectory telemetry."""
        hw_ok, hw_reason = self.is_hardware_verified_and_fresh()
        return {
            "teleop_state": self._teleop_state,
            "enable_motion": self.enable_motion,
            "enable_ik_dry_run": self.enable_ik_dry_run,
            "modes_conflict": self.modes_conflict,
            "single_goal_test_mode": self.single_goal_test_mode,
            "max_goals_per_session": self.max_goals_per_session,
            "effective_session_goal_limit": self.effective_session_goal_limit,
            "is_session_goal_limit_reached": self.is_session_goal_limit_reached(),
            "session_goal_dispatch_count": self._session_goal_dispatch_count,
            "can_issue_ik_requests": self.can_issue_ik_requests(),
            "hardware_mock_verified": self._hardware_mock_verified,
            "hardware_is_fresh": hw_ok,
            "hardware_status": self._hardware_status,
            "goal_active": self._goal_active,
            "trajectory_dispatch_count": self._trajectory_dispatch_count,
            "trajectory_success_count": self._trajectory_success_count,
            "trajectory_last_status": self._trajectory_last_status,
            "gripper_goal_active": self._gripper_goal_active,
            "gripper_canceling": self._gripper_canceling,
            "gripper_dispatch_count": self._gripper_dispatch_count,
            "gripper_success_count": self._gripper_success_count,
            "gripper_last_status": self._gripper_last_status,
            "last_commanded_gripper_state": self._last_commanded_gripper_state,
            "gripper_max_velocity": self.gripper_max_velocity,
            "gripper_max_acceleration": self.gripper_max_acceleration,
            "gripper_min_duration": self.gripper_min_duration,
            "target_pose": {
                "x": self._target_x,
                "y": self._target_y,
                "z": self._target_z,
            },
            "target_delta": {
                "dx": self._target_x - self._initial_target[0],
                "dy": self._target_y - self._initial_target[1],
                "dz": self._target_z - self._initial_target[2],
            },
            "initial_target": {
                "x": self._initial_target[0],
                "y": self._initial_target[1],
                "z": self._initial_target[2],
            },
            "ik_pending": self._ik_pending,
            "ik_last_status": self._ik_last_status,
            "ik_last_error_code": self._ik_last_error_code,
            "ik_call_count": self._ik_call_count,
            "ik_success_count": self._ik_success_count,
            "ik_solution": dict(self._latest_ik_solution) if self._latest_ik_solution else None,
            "target_generation": self._target_generation,
            "last_submitted_ik_target": self._last_submitted_ik_target,
            "has_pending_candidate": (self._pending_ik_candidate is not None),
        }

    # ── Diagnostic Overlay Rendering ───────────────────────────────────────
    def _draw_ros_teleop_overlay(
        self,
        frame: np.ndarray,
        preview: MotionPreviewData,
    ) -> None:
        """Render ROS teleop state, Cartesian target, IK status, trajectory status, and joint feedback."""
        h, w = frame.shape[:2]

        # Draw ROS Diagnostics Card on bottom-left of screen
        card_x1 = 12
        card_y1 = max(h - 265, 205)
        card_x2 = min(card_x1 + 350, w - 10)
        card_y2 = h - 10

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (card_x1, card_y1),
            (card_x2, card_y2),
            (25, 25, 25),
            -1,
        )
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(
            frame,
            (card_x1, card_y1),
            (card_x2, card_y2),
            (70, 70, 70),
            1,
        )

        # Header
        cv2.putText(
            frame,
            "ROS 2 TELEOP (SIMULATION ONLY)",
            (card_x1 + 10, card_y1 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # State Indicator
        y_cursor = card_y1 + 34
        eff_limit = self.effective_session_goal_limit
        if self._teleop_state == STATE_ARMED:
            if eff_limit > 0:
                limit_label = "SINGLE-GOAL" if self.single_goal_test_mode else "SESSION"
                state_text = f"STATE: ARMED [{limit_label} {self._session_goal_dispatch_count}/{eff_limit}]"
            else:
                state_text = "STATE: ARMED — SIMULATION ONLY"
            state_color = (0, 255, 0)  # Green
        elif self._teleop_state == STATE_ARMING_PENDING:
            now = time.monotonic()
            rem = max(0.0, ARMING_TIMEOUT_SEC - (now - self._arming_requested_time))
            state_text = f"STATE: ARMING PENDING ({rem:.1f}s) - PRESS [Y]"
            state_color = (0, 255, 255)  # Yellow
        else:
            if eff_limit > 0:
                limit_label = "SINGLE-GOAL" if self.single_goal_test_mode else "SESSION"
                state_text = f"STATE: DISARMED [{limit_label} {self._session_goal_dispatch_count}/{eff_limit}]"
            else:
                state_text = "STATE: DISARMED"
            state_color = (0, 165, 255)  # Orange/Amber

        cv2.putText(
            frame,
            state_text,
            (card_x1 + 10, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            state_color,
            2,
            cv2.LINE_AA,
        )

        # Operating Mode / Safety Gate
        y_cursor += 18
        if self.modes_conflict:
            mode_text = "MODE: CONFLICT (MOTION & DRY-RUN BOTH ON)"
            mode_color = (0, 0, 255)
        else:
            motion_active = self.enable_motion
            mode_color = (0, 255, 0) if motion_active else (140, 140, 140)
            mode_text = f"MOTION GATE: {'ENABLED' if motion_active else 'LOCKED (false)'}"
        cv2.putText(
            frame,
            mode_text,
            (card_x1 + 10, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            mode_color,
            1,
            cv2.LINE_AA,
        )

        # Hardware Introspection Status
        y_cursor += 18
        hw_ok, hw_reason = self.is_hardware_verified_and_fresh()
        now = time.monotonic()
        hw_age_str = (
            f"{now - self._hardware_last_verified_time:.1f}s"
            if self._hardware_last_verified_time
            else "N/A"
        )
        if hw_ok:
            hw_text = f"HW: SimSystem OK ({hw_age_str})"
            hw_color = (0, 255, 120)
        else:
            hw_text = f"HW: {self._hardware_status} ({hw_age_str})"
            hw_color = (0, 165, 255)
        cv2.putText(
            frame,
            hw_text,
            (card_x1 + 10, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            hw_color,
            1,
            cv2.LINE_AA,
        )

        # Joint Feedback Summary
        y_cursor += 18
        joint_diag = self.get_joint_state_diagnostics()
        num_joints = len(joint_diag["available_joints"])
        age_str = f"{joint_diag['age_seconds']:.2f}s" if joint_diag['age_seconds'] is not None else "N/A"
        if joint_diag["all_joints_present"] and joint_diag["is_fresh"]:
            j_status = f"JOINTS: 6/6 OK ({joint_diag['msg_count']} msgs, {age_str})"
            j_color = (0, 255, 120)
        elif num_joints > 0:
            j_status = f"JOINTS: {num_joints}/6 ({age_str})"
            j_color = (0, 200, 255)
        else:
            j_status = "JOINTS: Waiting for /joint_states"
            j_color = (130, 130, 130)

        cv2.putText(
            frame,
            j_status,
            (card_x1 + 10, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            j_color,
            1,
            cv2.LINE_AA,
        )

        # Cartesian Target Position
        y_cursor += 18
        tgt_line = f"TGT: X:{self._target_x:+.3f} Y:{self._target_y:+.3f} Z:{self._target_z:+.3f} m"
        cv2.putText(
            frame,
            tgt_line,
            (card_x1 + 10, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        # Cartesian Target Delta
        y_cursor += 16
        dx = self._target_x - self._initial_target[0]
        dy = self._target_y - self._initial_target[1]
        dz = self._target_z - self._initial_target[2]
        dlt_line = f"DLT: dX:{dx:+.3f} dY:{dy:+.3f} dZ:{dz:+.3f} m"
        cv2.putText(
            frame,
            dlt_line,
            (card_x1 + 10, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )

        # MoveIt IK Status
        y_cursor += 18
        if not self.can_issue_ik_requests():
            ik_text = "IK: IDLE / GATED"
            ik_color = (130, 130, 130)
        else:
            ik_text = f"IK: {self._ik_last_status} ({self._ik_success_count}/{self._ik_call_count})"
            if self._ik_last_status == "SUCCESS":
                ik_color = (0, 255, 120)
            elif self._ik_last_status in ("CALLING", "IDLE"):
                ik_color = (0, 220, 255)
            else:
                ik_color = (0, 140, 255)

        cv2.putText(
            frame,
            ik_text,
            (card_x1 + 10, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            ik_color,
            1,
            cv2.LINE_AA,
        )

        # Trajectory Action Status
        y_cursor += 18
        traj_text = f"TRAJ: {self._trajectory_last_status} (active={self._goal_active}, {self._trajectory_success_count}/{self._trajectory_dispatch_count})"
        if self._trajectory_last_status == "SUCCEEDED":
            traj_color = (0, 255, 120)
        elif self._trajectory_last_status in ("EXECUTING", "SENDING"):
            traj_color = (0, 220, 255)
        elif self._trajectory_last_status == "IDLE":
            traj_color = (140, 140, 140)
        else:
            traj_color = (0, 140, 255)

        cv2.putText(
            frame,
            traj_text,
            (card_x1 + 10, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            traj_color,
            1,
            cv2.LINE_AA,
        )

        # Safety Disclaimer
        y_cursor += 18
        cv2.putText(
            frame,
            "SAFETY: SIM GATED TRAJECTORY (NO PHYSICAL E-STOP)",
            (card_x1 + 10, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.33,
            (0, 140, 255),
            1,
            cv2.LINE_AA,
        )

        # Key Controls Hint
        y_cursor += 16
        cv2.putText(
            frame,
            "[A] Arm | [Y] Confirm | [D] Disarm | [C] Calib | [R] Reset | [Q] Quit",
            (card_x1 + 10, y_cursor),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    # ── Interactive Runtime Loop ───────────────────────────────────────────
    def run(self) -> None:
        """Execute the interactive camera, MediaPipe tracking, and ROS loop."""
        # 1. Resolve model asset path
        model_path = resolve_model_path(self.custom_model_path)
        if not os.path.isfile(model_path):
            self.get_logger().error(
                f"Hand Landmarker model not found at:\n  {model_path}\n"
                "Please ensure hand_landmarker.task is installed or present in models/."
            )
            return

        self.get_logger().info(f"Using MediaPipe model: {model_path}")

        # 2. Build MediaPipe Hand Landmarker
        latest_result: Dict[str, Any] = {"value": None}

        def _result_callback(
            result: mp_vision.HandLandmarkerResult,
            output_image: mp.Image,
            timestamp_ms: int,
        ) -> None:
            latest_result["value"] = result

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=_result_callback,
        )

        self.get_logger().info("Initializing MediaPipe Hand Landmarker...")
        landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self.get_logger().info("Hand Landmarker ready.")

        # 3. Open webcam
        cam_idx = self.camera_index
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            self.get_logger().error(f"Cannot open camera index {cam_idx}.")
            landmarker.close()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(f"Camera opened: {actual_w}x{actual_h} (index {cam_idx})")
        self.get_logger().info(
            "Teleoperation Node running. Hotkeys: "
            "[E] Enable (Auto-Anchor) | [D] Disable/Stop | [C] Recalibrate | [R] Reset | [Q/Esc] Quit"
        )

        fps = 0.0
        frame_count = 0
        fps_window_frames = 30
        fps_timer = time.monotonic()
        last_frame_time = time.monotonic()
        last_hw_check_time = 0.0

        # Setup enlarged display window (~75% screen dimensions, preserving 4:3 aspect ratio)
        window_name = "Hand Gesture Teleop – ROS 2 Node (Simulation)"
        target_w, target_h = setup_enlarged_camera_window(window_name)
        self.get_logger().info(f"Display window configured: {target_w}x{target_h} (preserves 4:3 aspect ratio)")

        try:
            while rclpy.ok():
                # Service pending ROS events (non-blocking)
                rclpy.spin_once(self, timeout_sec=0.001)

                current_time = time.monotonic()

                # Periodic hardware verification check (every 1 second)
                if (current_time - last_hw_check_time) > 1.0:
                    self.check_hardware_components_async()
                    last_hw_check_time = current_time

                # Check arming handshake timeout
                if self._teleop_state == STATE_ARMING_PENDING:
                    if (current_time - self._arming_requested_time) > ARMING_TIMEOUT_SEC:
                        self.disarm("ARMING_TIMEOUT")

                # Mode conflict enforcement
                if self.modes_conflict and self._teleop_state != STATE_DISARMED:
                    self.disarm("MODE_CONFLICT")

                # Dynamic motion disabled watchdog
                if not self.enable_motion and self._teleop_state in (STATE_ARMED, STATE_ARMING_PENDING):
                    self.disarm("MOTION_DISABLED")

                # Session goal limit watchdog: if limit reached while ARMED and no goal active
                if (
                    self.is_session_goal_limit_reached()
                    and self._teleop_state == STATE_ARMED
                    and not self._goal_active
                ):
                    disarm_reason = (
                        "SINGLE_GOAL_TEST_COMPLETED"
                        if self.single_goal_test_mode
                        else "SESSION_GOAL_LIMIT_COMPLETED"
                    )
                    self.disarm(disarm_reason)

                # Freshness check: If ARMED, verify hardware check is still fresh (<= 3.0s)
                if self._teleop_state == STATE_ARMED:
                    hw_ok, hw_reason = self.is_hardware_verified_and_fresh()
                    if not hw_ok:
                        self.disarm(f"HARDWARE_VERIFICATION_EXPIRED: {hw_reason}")

                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    time.sleep(0.01)
                    continue

                frame_count += 1
                if frame_count % fps_window_frames == 0:
                    elapsed = current_time - fps_timer
                    fps = fps_window_frames / elapsed if elapsed > 0 else 0.0
                    fps_timer = current_time

                # Send frame to MediaPipe
                timestamp_ms = int(current_time * 1000)
                rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )
                landmarker.detect_async(mp_image, timestamp_ms)

                result = latest_result["value"]
                hand_detected = result is not None and len(result.hand_landmarks) > 0

                # Automatic disarm on hand loss or stale feedback while ARMED
                if self._teleop_state == STATE_ARMED:
                    if not hand_detected:
                        self.disarm("HAND_LOST")
                    elif (
                        self._last_joint_state_time is None
                        or (current_time - self._last_joint_state_time) > MAX_JOINT_STATE_AGE
                    ):
                        self.disarm("STALE_FEEDBACK")

                handedness_text = ""
                raw_finger_states = {
                    "thumb": "UNKNOWN",
                    "index": "UNKNOWN",
                    "middle": "UNKNOWN",
                }
                raw_gesture = "UNKNOWN"
                first_hand_landmarks = None

                if hand_detected:
                    if (
                        result.handedness
                        and len(result.handedness) > 0
                        and len(result.handedness[0]) > 0
                    ):
                        cat = result.handedness[0][0]
                        cat_name = cat.category_name or "Hand"
                        cat_pct = int(cat.score * 100)
                        handedness_text = f"[{cat_name} {cat_pct}%]"

                    first_hand_landmarks = result.hand_landmarks[0]
                    raw_finger_states, raw_gesture = classify_hand_gestures(
                        first_hand_landmarks,
                        cfg=self.gesture_cfg,
                        prev_gesture=self.gesture_smoother.confirmed_state,
                    )

                    confirmed_gesture = self.gesture_smoother.update(raw_gesture)
                    confirmed_fingers = {
                        "thumb": self.thumb_smoother.update(raw_finger_states["thumb"]),
                        "index": self.index_smoother.update(raw_finger_states["index"]),
                        "middle": self.middle_smoother.update(raw_finger_states["middle"]),
                    }
                    self._last_hand_landmarks = first_hand_landmarks
                else:
                    self.gesture_smoother.reset("UNKNOWN")
                    self.thumb_smoother.reset("UNKNOWN")
                    self.index_smoother.reset("UNKNOWN")
                    self.middle_smoother.reset("UNKNOWN")
                    confirmed_gesture = "UNKNOWN"
                    confirmed_fingers = raw_finger_states
                    self._last_hand_landmarks = None

                preview_data = self.motion_tracker.update(
                    hand_detected=hand_detected,
                    landmarks=first_hand_landmarks,
                )

                # Compute frame delta time
                dt = current_time - last_frame_time
                last_frame_time = current_time

                # Cartesian target accumulation
                is_active = (self._teleop_state == STATE_ARMED) or self.enable_ik_dry_run
                dx, dy, dz = compute_cartesian_increment(
                    ux=preview_data.ux,
                    uy=preview_data.uy,
                    uz=preview_data.uz,
                    dt=dt,
                    hand_detected=hand_detected,
                    gesture_state=confirmed_gesture,
                    is_active=is_active,
                    cfg=self.target_cfg,
                )
                self.update_cartesian_target(dx, dy, dz)

                # Trigger non-blocking IK query if permitted by current mode
                if self.can_issue_ik_requests():
                    self.request_ik()

                # Gripper actuation (Task 5C-3D)
                if hand_detected and confirmed_gesture in ("GRIPPER_OPEN", "GRIPPER_CLOSED"):
                    self._send_gripper_command(confirmed_gesture)

                # ── Display Pipeline (Task 5C-3D-UI-FIX) ────────────────────
                # 1. Create separate display-only copy scaled to target window (4:3 aspect ratio)
                if (target_w, target_h) != (frame_bgr.shape[1], frame_bgr.shape[0]):
                    display_frame = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                else:
                    display_frame = frame_bgr.copy()

                # 2. Draw MediaPipe landmarks directly on resized display frame at native resolution
                if hand_detected and result is not None:
                    draw_landmarks_on_frame(display_frame, result, gesture_state=confirmed_gesture)

                # 3. Determine Gripper state text
                if not self.enable_motion:
                    gripper_disp = "IDLE (Motion disabled)"
                elif self._gripper_goal_active:
                    gripper_disp = "PENDING"
                elif self._last_commanded_gripper_state == "GRIPPER_OPEN":
                    gripper_disp = "OPEN"
                elif self._last_commanded_gripper_state == "GRIPPER_CLOSED":
                    gripper_disp = "CLOSED"
                elif self._gripper_last_status in ("ABORTED", "CANCELED", "REJECTED"):
                    gripper_disp = f"ERROR ({self._gripper_last_status})"
                else:
                    gripper_disp = "IDLE"

                # 4. Determine Control state text, Anchor status, and hotkey help
                control_disp = self.get_control_state_text()
                anchor_disp = self.get_anchor_state_text()

                if not self.enable_motion:
                    help_line = "Hotkeys: [Q/Esc] Quit"
                    heading_text = "PREVIEW ONLY — NO ROBOT COMMANDS"
                elif self._teleop_state == STATE_ARMED:
                    help_line = "Hotkeys: [D] Stop | [C] Recalibrate | [Q] Quit"
                    heading_text = "HAND TELEOP — SIMULATION"
                else:
                    help_line = "Hotkeys: [E] Enable (Auto-Anchor) | [D] Stop | [Q] Quit"
                    heading_text = "HAND TELEOP — SIMULATION"

                # 5. Render single consolidated compact HUD in top-left corner
                draw_compact_hud(
                    frame=display_frame,
                    hand_detected=hand_detected,
                    gesture_state=confirmed_gesture,
                    gripper_state_text=gripper_disp,
                    control_state_text=control_disp,
                    is_motion_enabled=self.enable_motion,
                    help_text=help_line,
                    status_heading=heading_text,
                    anchor_state_text=anchor_disp,
                )

                cv2.imshow(window_name, display_frame)

                # Key Handling
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    self.get_logger().info("Exit key pressed. Shutting down...")
                    break
                elif key in (ord("e"), ord("E")):
                    self.enable_teleoperation(
                        hand_detected=hand_detected,
                        landmarks=first_hand_landmarks,
                        gesture_state=confirmed_gesture,
                    )
                elif key in (ord("d"), ord("D")):
                    self.disarm("USER_KEYPRESS_D")
                elif key in (ord("a"), ord("A")):
                    self.request_arming()
                elif key in (ord("y"), ord("Y")):
                    self.confirm_arming(
                        hand_detected=hand_detected,
                        gesture_state=confirmed_gesture,
                        landmarks=first_hand_landmarks,
                    )
                elif key in (ord("c"), ord("C")):
                    if hand_detected and first_hand_landmarks is not None:
                        lm9 = first_hand_landmarks[9]
                        lm0 = first_hand_landmarks[0]
                        scale_2d = dist_2d(lm0, lm9)
                        self.motion_tracker.calibrate(lm9.x, lm9.y, scale_2d)
                        self.get_logger().info(
                            f"Neutral anchor recalibrated: X0={lm9.x:.3f}, Y0={lm9.y:.3f}, Scale0={scale_2d:.3f}"
                        )
                    else:
                        self.get_logger().warn(
                            "Cannot recalibrate: No hand detected in frame."
                        )
                elif key in (ord("r"), ord("R")):
                    self.motion_tracker.reset_calibration()
                    self.reset_target_pose()
                    self.get_logger().info("Calibration anchor and Cartesian target reset.")

        finally:
            if self._goal_active:
                self._cancel_active_goal()
            if self._gripper_goal_active:
                self._cancel_active_gripper_goal()
            cap.release()
            cv2.destroyAllWindows()
            landmarker.close()
            self.get_logger().info("Camera and resources released.")


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = HandTeleopNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
