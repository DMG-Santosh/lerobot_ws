#!/usr/bin/env python3
"""
test_cartesian_ik.py
====================
Comprehensive headless unit & integration test suite for:
  - Task 5C-2C-A: Cartesian increment formulas, direction signs, speed scaling, and workspace limits.
  - Task 5C-2C-A-H: Hardened IK response validation (missing joints, non-finite values, limits).
  - Task 5C-3A: Action client goal construction, single-active-goal guard, and two-key arming handshake.
  - Task 5C-3A-H: Hardened trajectory timing (cubic spline duration), fail-closed hardware introspection
    (freshness, identity, out-of-order rejection), and 5-mode IK gating.
"""

import math
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, PropertyMock, patch

import cv2
import numpy as np

# Setup ROS 2 and package environment
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from action_msgs.msg import GoalStatus
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetPositionIK

from hand_gesture_teleop.hand_teleop_node import (
    HandTeleopNode,
    CartesianTargetConfig,
    CARTESIAN_TARGET_CONFIG,
    compute_cartesian_increment,
    verify_hardware_components_from_response,
    build_trajectory_goal,
    STATE_DISARMED,
    STATE_ARMING_PENDING,
    STATE_ARMED,
    ARM_JOINT_LIMITS,
    ARM_CONTROL_JOINTS,
    MAX_JOINT_DELTA,
    MAX_COMMANDED_VELOCITY,
    MIN_TRAJECTORY_DURATION,
    MAX_JOINT_STATE_AGE,
    ARMING_TIMEOUT_SEC,
    MAX_HARDWARE_CHECK_AGE,
    EXPECTED_SO101_JOINTS,
    EXPECTED_MOCK_HARDWARE_NAME,
    EXPECTED_MOCK_HARDWARE_TYPE,
    SIMULATION_MOCK_PLUGIN,
    CUBIC_VELOCITY_SCALING_FACTOR,
    MIN_IK_TARGET_DISPLACEMENT,
    MIN_DISPATCH_JOINT_DELTA,
    MAX_IK_TARGET_DRIFT,
    TargetSnapshot,
    DEFAULT_GRIPPER_OPEN_RAD,
    DEFAULT_GRIPPER_CLOSED_RAD,
    GRIPPER_JOINT_LIMIT_MIN,
    GRIPPER_JOINT_LIMIT_MAX,
    DEFAULT_GRIPPER_DURATION,
    DEFAULT_GRIPPER_MAX_VELOCITY,
    DEFAULT_GRIPPER_MAX_ACCELERATION,
    MIN_GRIPPER_TRAJECTORY_DURATION,
    compute_cubic_peak_velocity,
    compute_cubic_peak_acceleration,
    compute_gripper_trajectory_duration,
    build_gripper_trajectory_goal,
)
from hand_gesture_teleop.hand_tracking import (
    classify_hand_gestures,
    TemporalSmoother,
    GestureConfig,
    GESTURE_CONFIG,
    NormalizedLandmark,
    dist_2d,
    calculate_aspect_preserving_window_size,
    get_screen_resolution,
    setup_enlarged_camera_window,
    draw_compact_hud,
    draw_mapping_hud,
    draw_thumb_index_connector,
    draw_landmarks_on_frame,
)


class TestCartesianTargetCalculations(unittest.TestCase):
    """Test Cartesian increment mapping, directions, scaling, and gating."""

    def setUp(self):
        self.cfg = CARTESIAN_TARGET_CONFIG

    def test_config_defaults(self):
        """Verify safety constants adhere strictly to task specifications."""
        self.assertAlmostEqual(self.cfg.MAX_CARTESIAN_SPEED, 0.02)
        self.assertAlmostEqual(self.cfg.MAX_CARTESIAN_STEP, 0.002)
        self.assertAlmostEqual(self.cfg.CUMULATIVE_LIMIT_X, 0.050)
        self.assertAlmostEqual(self.cfg.CUMULATIVE_LIMIT_Y, 0.050)
        self.assertAlmostEqual(self.cfg.CUMULATIVE_LIMIT_Z, 0.050)
        self.assertAlmostEqual(self.cfg.MIN_HEIGHT_Z, 0.050)
        self.assertEqual(self.cfg.INITIAL_TARGET_POSE, (0.0206, -0.2775, 0.2669))
        self.assertEqual(self.cfg.INITIAL_ORIENTATION, (0.5, 0.5, 0.5, -0.5))
        self.assertEqual(self.cfg.IK_PLANNING_GROUP, "arm")
        self.assertEqual(self.cfg.IK_TIP_LINK, "gripper")
        self.assertEqual(self.cfg.IK_REFERENCE_FRAME, "world")
        self.assertAlmostEqual(self.cfg.IK_THROTTLE_PERIOD, 0.5)

    def test_axis_directions(self):
        """Verify exact sign conventions for all 3 axes."""
        dt = 0.033  # ~30 FPS

        # Vertical (uy -> Z): UP is positive, DOWN is negative
        dx, dy, dz_up = compute_cartesian_increment(0.0, 1.0, 0.0, dt, True, "GRIPPER_OPEN", True, self.cfg)
        self.assertGreater(dz_up, 0.0)
        self.assertAlmostEqual(dx, 0.0)
        self.assertAlmostEqual(dy, 0.0)

        dx, dy, dz_down = compute_cartesian_increment(0.0, -1.0, 0.0, dt, True, "GRIPPER_OPEN", True, self.cfg)
        self.assertLess(dz_down, 0.0)

        # Lateral (ux -> X): RIGHT is positive, LEFT is negative
        dx_right, dy, dz = compute_cartesian_increment(1.0, 0.0, 0.0, dt, True, "GRIPPER_OPEN", True, self.cfg)
        self.assertGreater(dx_right, 0.0)
        self.assertAlmostEqual(dy, 0.0)
        self.assertAlmostEqual(dz, 0.0)

        dx_left, dy, dz = compute_cartesian_increment(-1.0, 0.0, 0.0, dt, True, "GRIPPER_OPEN", True, self.cfg)
        self.assertLess(dx_left, 0.0)

        # Reach (uz -> Y): FARTHER into scene (uz < 0) reaches forward along -Y
        dx, dy_farther, dz = compute_cartesian_increment(0.0, 0.0, -1.0, dt, True, "GRIPPER_OPEN", True, self.cfg)
        self.assertLess(dy_farther, 0.0)

        # CLOSER toward body (uz > 0) retracts backward along +Y
        dx, dy_closer, dz = compute_cartesian_increment(0.0, 0.0, 1.0, dt, True, "GRIPPER_OPEN", True, self.cfg)
        self.assertGreater(dy_closer, 0.0)

    def test_step_clamping(self):
        """Verify single-step increment never exceeds MAX_CARTESIAN_STEP (2 mm)."""
        dx, dy, dz = compute_cartesian_increment(1.0, 1.0, 1.0, 10.0, True, "GRIPPER_OPEN", True, self.cfg)
        self.assertAlmostEqual(dx, 0.002)
        self.assertAlmostEqual(dy, 0.002)
        self.assertAlmostEqual(dz, 0.002)

        dx, dy, dz = compute_cartesian_increment(-1.0, -1.0, -1.0, 10.0, True, "GRIPPER_OPEN", True, self.cfg)
        self.assertAlmostEqual(dx, -0.002)
        self.assertAlmostEqual(dy, -0.002)
        self.assertAlmostEqual(dz, -0.002)

    def test_speed_scaling(self):
        """Verify speed scaling produces delta proportional to MAX_CARTESIAN_SPEED * dt."""
        dt = 0.020  # 20 ms
        expected_step = 0.02 * 0.020  # 0.0004 m
        dx, dy, dz = compute_cartesian_increment(1.0, 0.5, -1.0, dt, True, "GRIPPER_OPEN", True, self.cfg)
        self.assertAlmostEqual(dx, expected_step, places=6)
        self.assertAlmostEqual(dy, -expected_step, places=6)
        self.assertAlmostEqual(dz, expected_step * 0.5, places=6)

    def test_deadzone_gating(self):
        """Verify zero increment when displacement inputs are 0.0."""
        dx, dy, dz = compute_cartesian_increment(0.0, 0.0, 0.0, 0.033, True, "GRIPPER_OPEN", True, self.cfg)
        self.assertEqual((dx, dy, dz), (0.0, 0.0, 0.0))

    def test_hand_detection_gating(self):
        """Verify zero increment when hand_detected is False."""
        dx, dy, dz = compute_cartesian_increment(1.0, 1.0, 1.0, 0.033, False, "GRIPPER_OPEN", True, self.cfg)
        self.assertEqual((dx, dy, dz), (0.0, 0.0, 0.0))

    def test_gesture_gating(self):
        """Verify motion only accumulates for GRIPPER_OPEN and GRIPPER_CLOSED gestures."""
        dx1, dy1, dz1 = compute_cartesian_increment(1.0, 1.0, 1.0, 0.033, True, "GRIPPER_OPEN", True, self.cfg)
        self.assertGreater(dx1, 0.0)

        dx2, dy2, dz2 = compute_cartesian_increment(1.0, 1.0, 1.0, 0.033, True, "GRIPPER_CLOSED", True, self.cfg)
        self.assertGreater(dx2, 0.0)

        for prohibited in ("UNKNOWN", "POINTING", "PEACE", "THUMBS_UP", "NEUTRAL"):
            dx, dy, dz = compute_cartesian_increment(1.0, 1.0, 1.0, 0.033, True, prohibited, True, self.cfg)
            self.assertEqual((dx, dy, dz), (0.0, 0.0, 0.0))

    def test_active_flag_gating(self):
        """Verify zero increment when is_active is False."""
        dx, dy, dz = compute_cartesian_increment(1.0, 1.0, 1.0, 0.033, True, "GRIPPER_OPEN", False, self.cfg)
        self.assertEqual((dx, dy, dz), (0.0, 0.0, 0.0))


class TestTargetAccumulationAndNodeIK(unittest.TestCase):
    """Test node target state accumulation, cumulative limits, reset, and IK handling."""

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = HandTeleopNode("test_teleop_node")
        self.node._latest_joint_positions = {
            "joint1": 0.1,
            "joint2": -0.2,
            "joint3": 0.3,
            "joint4": -0.4,
            "joint5": 0.5,
            "joint6": 0.0,
        }
        self.node._last_joint_state_time = time.monotonic()

    def tearDown(self):
        self.node.destroy_node()

    def test_initial_state(self):
        """Verify node initial state, parameters, and properties."""
        self.assertEqual(self.node.teleop_state, STATE_DISARMED)
        self.assertFalse(self.node.enable_motion)
        self.assertFalse(self.node.enable_ik_dry_run)
        x0, y0, z0 = CARTESIAN_TARGET_CONFIG.INITIAL_TARGET_POSE
        self.assertAlmostEqual(self.node.target_pose[0], x0)
        self.assertAlmostEqual(self.node.target_pose[1], y0)
        self.assertAlmostEqual(self.node.target_pose[2], z0)
        self.assertEqual(self.node.target_delta, (0.0, 0.0, 0.0))

    def test_cumulative_limits(self):
        """Verify cumulative movement from initial pose is bounded within +/- 50 mm."""
        x0, y0, z0 = CARTESIAN_TARGET_CONFIG.INITIAL_TARGET_POSE

        for _ in range(100):
            self.node.update_cartesian_target(0.002, 0.002, 0.002)

        tx, ty, tz = self.node.target_pose
        self.assertAlmostEqual(tx, x0 + 0.050, places=4)
        self.assertAlmostEqual(ty, y0 + 0.050, places=4)
        self.assertAlmostEqual(tz, z0 + 0.050, places=4)

        for _ in range(200):
            self.node.update_cartesian_target(-0.002, -0.002, -0.002)

        tx, ty, tz = self.node.target_pose
        self.assertAlmostEqual(tx, x0 - 0.050, places=4)
        self.assertAlmostEqual(ty, y0 - 0.050, places=4)
        self.assertAlmostEqual(tz, max(z0 - 0.050, 0.050), places=4)

    def test_minimum_height_enforcement(self):
        """Verify target height is strictly bounded by MIN_HEIGHT_Z (0.050 m)."""
        self.node.update_cartesian_target(0.0, 0.0, -1.0)
        _, _, tz = self.node.target_pose
        self.assertGreaterEqual(tz, 0.050)

    def test_reset_target_pose(self):
        """Verify reset_target_pose restores initial pose exactly."""
        x0, y0, z0 = CARTESIAN_TARGET_CONFIG.INITIAL_TARGET_POSE
        self.node.update_cartesian_target(0.02, 0.02, 0.02)
        self.assertNotEqual(self.node.target_delta, (0.0, 0.0, 0.0))

        self.node.reset_target_pose()
        self.assertAlmostEqual(self.node.target_pose[0], x0)
        self.assertAlmostEqual(self.node.target_pose[1], y0)
        self.assertAlmostEqual(self.node.target_pose[2], z0)
        self.assertEqual(self.node.target_delta, (0.0, 0.0, 0.0))

    def test_ik_response_success(self):
        """Verify complete valid 5-joint solution is accepted and mapped (with gripper excluded)."""
        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.SUCCESS
        mock_resp.solution.joint_state.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        mock_resp.solution.joint_state.position = [0.1, -0.2, 0.3, -0.4, 0.5, 0.0]
        mock_future.result.return_value = mock_resp

        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future)

        self.assertFalse(self.node._ik_pending)
        self.assertEqual(self.node._ik_last_status, "SUCCESS")
        self.assertEqual(self.node._ik_success_count, 1)
        self.assertEqual(self.node._ik_call_count, 1)
        self.assertIsNotNone(self.node._latest_ik_solution)
        self.assertAlmostEqual(self.node._latest_ik_solution["joint1"], 0.1)
        self.assertAlmostEqual(self.node._latest_ik_solution["joint5"], 0.5)
        self.assertNotIn("joint6", self.node._latest_ik_solution)

    def test_ik_response_missing_joint1(self):
        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.SUCCESS
        mock_resp.solution.joint_state.name = ["joint2", "joint3", "joint4", "joint5"]
        mock_resp.solution.joint_state.position = [0.0, 0.0, 0.0, 0.0]
        mock_future.result.return_value = mock_resp

        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future)
        self.assertEqual(self.node._ik_last_status, "INCOMPLETE_SOLUTION")
        self.assertIsNone(self.node._latest_ik_solution)

    def test_ik_response_missing_joint2(self):
        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.SUCCESS
        mock_resp.solution.joint_state.name = ["joint1", "joint3", "joint4", "joint5"]
        mock_resp.solution.joint_state.position = [0.0, 0.0, 0.0, 0.0]
        mock_future.result.return_value = mock_resp

        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future)
        self.assertEqual(self.node._ik_last_status, "INCOMPLETE_SOLUTION")
        self.assertIsNone(self.node._latest_ik_solution)

    def test_ik_response_missing_joint3(self):
        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.SUCCESS
        mock_resp.solution.joint_state.name = ["joint1", "joint2", "joint4", "joint5"]
        mock_resp.solution.joint_state.position = [0.0, 0.0, 0.0, 0.0]
        mock_future.result.return_value = mock_resp

        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future)
        self.assertEqual(self.node._ik_last_status, "INCOMPLETE_SOLUTION")
        self.assertIsNone(self.node._latest_ik_solution)

    def test_ik_response_missing_joint4(self):
        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.SUCCESS
        mock_resp.solution.joint_state.name = ["joint1", "joint2", "joint3", "joint5"]
        mock_resp.solution.joint_state.position = [0.0, 0.0, 0.0, 0.0]
        mock_future.result.return_value = mock_resp

        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future)
        self.assertEqual(self.node._ik_last_status, "INCOMPLETE_SOLUTION")
        self.assertIsNone(self.node._latest_ik_solution)

    def test_ik_response_missing_joint5(self):
        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.SUCCESS
        mock_resp.solution.joint_state.name = ["joint1", "joint2", "joint3", "joint4"]
        mock_resp.solution.joint_state.position = [0.0, 0.0, 0.0, 0.0]
        mock_future.result.return_value = mock_resp

        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future)
        self.assertEqual(self.node._ik_last_status, "INCOMPLETE_SOLUTION")
        self.assertIsNone(self.node._latest_ik_solution)

    def test_ik_response_length_mismatch(self):
        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.SUCCESS
        mock_resp.solution.joint_state.name = ["joint1", "joint2", "joint3", "joint4", "joint5"]
        mock_resp.solution.joint_state.position = [0.0, 0.0, 0.0]
        mock_future.result.return_value = mock_resp

        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future)
        self.assertEqual(self.node._ik_last_status, "MALFORMED_SOLUTION")
        self.assertIsNone(self.node._latest_ik_solution)

    def test_ik_response_nan_value(self):
        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.SUCCESS
        mock_resp.solution.joint_state.name = ["joint1", "joint2", "joint3", "joint4", "joint5"]
        mock_resp.solution.joint_state.position = [0.0, float("nan"), 0.0, 0.0, 0.0]
        mock_future.result.return_value = mock_resp

        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future)
        self.assertEqual(self.node._ik_last_status, "INVALID_NONFINITE_SOLUTION")
        self.assertIsNone(self.node._latest_ik_solution)

    def test_ik_response_pos_inf_value(self):
        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.SUCCESS
        mock_resp.solution.joint_state.name = ["joint1", "joint2", "joint3", "joint4", "joint5"]
        mock_resp.solution.joint_state.position = [0.0, 0.0, float("inf"), 0.0, 0.0]
        mock_future.result.return_value = mock_resp

        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future)
        self.assertEqual(self.node._ik_last_status, "INVALID_NONFINITE_SOLUTION")
        self.assertIsNone(self.node._latest_ik_solution)

    def test_ik_response_neg_inf_value(self):
        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.SUCCESS
        mock_resp.solution.joint_state.name = ["joint1", "joint2", "joint3", "joint4", "joint5"]
        mock_resp.solution.joint_state.position = [0.0, 0.0, 0.0, float("-inf"), 0.0]
        mock_future.result.return_value = mock_resp

        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future)
        self.assertEqual(self.node._ik_last_status, "INVALID_NONFINITE_SOLUTION")
        self.assertIsNone(self.node._latest_ik_solution)

    def test_ik_response_below_lower_limit(self):
        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.SUCCESS
        mock_resp.solution.joint_state.name = ["joint1", "joint2", "joint3", "joint4", "joint5"]
        mock_resp.solution.joint_state.position = [-2.5, 0.0, 0.0, 0.0, 0.0]
        mock_future.result.return_value = mock_resp

        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future)
        self.assertEqual(self.node._ik_last_status, "LIMIT_VIOLATION")
        self.assertIsNone(self.node._latest_ik_solution)

    def test_ik_response_above_upper_limit(self):
        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.SUCCESS
        mock_resp.solution.joint_state.name = ["joint1", "joint2", "joint3", "joint4", "joint5"]
        mock_resp.solution.joint_state.position = [0.0, 2.5, 0.0, 0.0, 0.0]
        mock_future.result.return_value = mock_resp

        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future)
        self.assertEqual(self.node._ik_last_status, "LIMIT_VIOLATION")
        self.assertIsNone(self.node._latest_ik_solution)

    def test_ik_response_joint5_limits(self):
        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.SUCCESS
        mock_resp.solution.joint_state.name = ["joint1", "joint2", "joint3", "joint4", "joint5"]
        mock_resp.solution.joint_state.position = [0.0, 0.0, 0.0, 0.0, 3.5]
        mock_future.result.return_value = mock_resp

        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future)
        self.assertEqual(self.node._ik_last_status, "LIMIT_VIOLATION")
        self.assertIsNone(self.node._latest_ik_solution)

    def test_ik_response_no_solution(self):
        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.NO_IK_SOLUTION
        mock_future.result.return_value = mock_resp

        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future)
        self.assertFalse(self.node._ik_pending)
        self.assertEqual(self.node._ik_last_status, "NO_IK_SOLUTION")
        self.assertIsNone(self.node._latest_ik_solution)

    def test_ik_response_timed_out(self):
        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.TIMED_OUT
        mock_future.result.return_value = mock_resp

        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future)
        self.assertEqual(self.node._ik_last_status, "TIMED_OUT")
        self.assertIsNone(self.node._latest_ik_solution)

    def test_ik_response_collision(self):
        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.GOAL_IN_COLLISION
        mock_future.result.return_value = mock_resp

        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future)
        self.assertEqual(self.node._ik_last_status, "GOAL_IN_COLLISION")
        self.assertIsNone(self.node._latest_ik_solution)

    def test_rejected_solutions_never_counted(self):
        # 1. Missing joint
        f1 = MagicMock()
        r1 = GetPositionIK.Response()
        r1.error_code.val = MoveItErrorCodes.SUCCESS
        r1.solution.joint_state.name = ["joint1"]
        r1.solution.joint_state.position = [0.0]
        f1.result.return_value = r1
        self.node._ik_response_callback(f1)

        # 2. NaN value
        f2 = MagicMock()
        r2 = GetPositionIK.Response()
        r2.error_code.val = MoveItErrorCodes.SUCCESS
        r2.solution.joint_state.name = ["joint1", "joint2", "joint3", "joint4", "joint5"]
        r2.solution.joint_state.position = [0.0, float("nan"), 0.0, 0.0, 0.0]
        f2.result.return_value = r2
        self.node._ik_response_callback(f2)

        # 3. Limit violation
        f3 = MagicMock()
        r3 = GetPositionIK.Response()
        r3.error_code.val = MoveItErrorCodes.SUCCESS
        r3.solution.joint_state.name = ["joint1", "joint2", "joint3", "joint4", "joint5"]
        r3.solution.joint_state.position = [5.0, 0.0, 0.0, 0.0, 0.0]
        f3.result.return_value = r3
        self.node._ik_response_callback(f3)

        self.assertEqual(self.node._ik_success_count, 0)
        self.assertEqual(self.node._ik_call_count, 3)
        self.assertIsNone(self.node._latest_ik_solution)

    def test_zero_motion_guarantee_and_publisher_audit(self):
        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertFalse(self.node.enable_motion)

        all_publishers = self.node.publishers
        topic_names = [pub.topic_name for pub in all_publishers]

        self.assertNotIn("/arm_controller/joint_trajectory", topic_names)
        self.assertNotIn("/gripper_controller/joint_trajectory", topic_names)
        self.assertNotIn("/joint_trajectory_controller/joint_trajectory", topic_names)

        self.assertFalse(hasattr(self.node, "trajectory_pub"))
        self.assertFalse(hasattr(self.node, "arm_pub"))
        self.assertFalse(hasattr(self.node, "gripper_pub"))
        self.assertTrue(hasattr(self.node, "gripper_action_client"))
        self.assertTrue(hasattr(self.node, "arm_action_client"))


class TestHardwareIntrospection(unittest.TestCase):
    """Test controller_manager hardware component introspection validation (Task 5C-3A & 5C-3A-H Phase B)."""

    def test_mock_hardware_active(self):
        """Verify mock SimSystem active returns True."""
        mock_resp = MagicMock()
        comp1 = MagicMock()
        comp1.name = "SimSystem"
        comp1.type = "system"
        comp1.plugin_name = "mock_components/GenericSystem"
        comp1.state.id = 3
        comp1.state.label = "active"
        mock_resp.component = [comp1]

        valid, reason = verify_hardware_components_from_response(mock_resp)
        self.assertTrue(valid)
        self.assertEqual(reason, "MOCK_HARDWARE_ACTIVE")

    def test_physical_hardware_rejected(self):
        """Verify physical hardware interface returns False immediately."""
        mock_resp = MagicMock()
        comp1 = MagicMock()
        comp1.name = "SimSystem"
        comp1.type = "system"
        comp1.plugin_name = "so_arm_100_controller/SOARM100Interface"
        comp1.state.id = 3
        comp1.state.label = "active"
        mock_resp.component = [comp1]

        valid, reason = verify_hardware_components_from_response(mock_resp)
        self.assertFalse(valid)
        self.assertIn("PHYSICAL_HARDWARE_DETECTED", reason)

    def test_unrecognized_hardware_name_rejected(self):
        """Verify component with unrecognized name is rejected."""
        mock_resp = MagicMock()
        comp1 = MagicMock()
        comp1.name = "UnrecognizedArm"
        comp1.type = "system"
        comp1.plugin_name = "mock_components/GenericSystem"
        comp1.state.id = 3
        comp1.state.label = "active"
        mock_resp.component = [comp1]

        valid, reason = verify_hardware_components_from_response(mock_resp)
        self.assertFalse(valid)
        self.assertIn("UNRECOGNIZED_HARDWARE_NAME", reason)

    def test_unrecognized_hardware_type_rejected(self):
        """Verify component with unrecognized type is rejected."""
        mock_resp = MagicMock()
        comp1 = MagicMock()
        comp1.name = "SimSystem"
        comp1.type = "actuator"
        comp1.plugin_name = "mock_components/GenericSystem"
        comp1.state.id = 3
        comp1.state.label = "active"
        mock_resp.component = [comp1]

        valid, reason = verify_hardware_components_from_response(mock_resp)
        self.assertFalse(valid)
        self.assertIn("UNRECOGNIZED_HARDWARE_TYPE", reason)

    def test_unexpected_multiple_components_rejected(self):
        """Verify multiple hardware components cause verification failure."""
        mock_resp = MagicMock()
        comp1 = MagicMock()
        comp1.name = "SimSystem"
        comp1.type = "system"
        comp1.plugin_name = "mock_components/GenericSystem"
        comp1.state.id = 3
        comp1.state.label = "active"

        comp2 = MagicMock()
        comp2.name = "ExtraSystem"
        comp2.type = "system"
        comp2.plugin_name = "mock_components/GenericSystem"
        comp2.state.id = 3
        comp2.state.label = "active"

        mock_resp.component = [comp1, comp2]

        valid, reason = verify_hardware_components_from_response(mock_resp)
        self.assertFalse(valid)
        self.assertIn("UNEXPECTED_COMPONENT_COUNT", reason)

    def test_inactive_mock_hardware_returns_false(self):
        """Verify mock component not in active state returns False."""
        mock_resp = MagicMock()
        comp1 = MagicMock()
        comp1.name = "SimSystem"
        comp1.type = "system"
        comp1.plugin_name = "mock_components/GenericSystem"
        comp1.state.id = 1
        comp1.state.label = "unconfigured"
        mock_resp.component = [comp1]

        valid, reason = verify_hardware_components_from_response(mock_resp)
        self.assertFalse(valid)
        self.assertIn("MOCK_HARDWARE_NOT_ACTIVE", reason)

    def test_malformed_response_returns_false(self):
        valid1, reason1 = verify_hardware_components_from_response(None)
        self.assertFalse(valid1)
        self.assertEqual(reason1, "INVALID_RESPONSE")

        mock_resp = MagicMock()
        mock_resp.component = []
        valid2, reason2 = verify_hardware_components_from_response(mock_resp)
        self.assertFalse(valid2)
        self.assertEqual(reason2, "NO_COMPONENTS_FOUND")


class TestHardwareFreshnessAndAsyncHandling(unittest.TestCase):
    """Test hardware check freshness, sequence tracking, and fail-closed state (Task 5C-3A-H Phase B)."""

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = HandTeleopNode("test_hw_freshness_node")

    def tearDown(self):
        self.node.destroy_node()

    def test_freshness_check_passes_when_recent(self):
        """Verify is_hardware_verified_and_fresh() returns True when verified within 3.0s."""
        self.node._hardware_mock_verified = True
        self.node._hardware_last_verified_time = time.monotonic() - 1.0  # 1s old
        fresh, reason = self.node.is_hardware_verified_and_fresh()
        self.assertTrue(fresh)
        self.assertEqual(reason, "MOCK_HARDWARE_FRESH")

    def test_freshness_check_fails_when_expired(self):
        """Verify is_hardware_verified_and_fresh() returns False when older than 3.0s."""
        self.node._hardware_mock_verified = True
        self.node._hardware_last_verified_time = time.monotonic() - 4.0  # 4s old
        fresh, reason = self.node.is_hardware_verified_and_fresh()
        self.assertFalse(fresh)
        self.assertIn("VERIFICATION_EXPIRED", reason)

    def test_freshness_check_fails_when_unverified(self):
        """Verify is_hardware_verified_and_fresh() returns False when _hardware_mock_verified is False."""
        self.node._hardware_mock_verified = False
        self.node._hardware_last_verified_time = time.monotonic()
        fresh, reason = self.node.is_hardware_verified_and_fresh()
        self.assertFalse(fresh)
        self.assertIn("NOT_VERIFIED", reason)

    def test_out_of_order_delayed_response_ignored(self):
        """Verify an older response (seq 1) arriving after newer response (seq 2) is discarded."""
        self.node._hw_response_seq = 2  # Already processed seq 2
        self.node._hardware_mock_verified = True
        self.node._hardware_status = "MOCK_HARDWARE_ACTIVE"
        valid_time = time.monotonic()
        self.node._hardware_last_verified_time = valid_time

        # Delayed response for seq 1 (failed/stale)
        mock_future = MagicMock()
        mock_resp = MagicMock()
        mock_resp.component = []  # would fail if accepted
        mock_future.result.return_value = mock_resp

        self.node._hw_response_callback(mock_future, req_seq=1)

        # State must remain unchanged (seq 1 was ignored)
        self.assertTrue(self.node._hardware_mock_verified)
        self.assertEqual(self.node._hardware_last_verified_time, valid_time)
        self.assertEqual(self.node._hw_response_seq, 2)

    def test_hardware_service_exception_fails_closed(self):
        """Verify service exception clears verified state and disarms node."""
        self.node._teleop_state = STATE_ARMED
        self.node._hardware_mock_verified = True
        self.node._hardware_last_verified_time = time.monotonic()

        mock_future = MagicMock()
        mock_future.result.side_effect = RuntimeError("Service connection lost")

        self.node._hw_response_callback(mock_future, req_seq=1)

        self.assertFalse(self.node._hardware_mock_verified)
        self.assertIsNone(self.node._hardware_last_verified_time)
        self.assertEqual(self.node.teleop_state, "DISARMED")


class TestHardenedIKValidation(unittest.TestCase):
    """Test duplicate joint names, stale feedback, and joint jumps (Task 5C-3A Phase B)."""

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = HandTeleopNode("test_hardened_ik_node")
        self.node._latest_joint_positions = {
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
        }
        self.node._last_joint_state_time = time.monotonic()

    def tearDown(self):
        self.node.destroy_node()

    def test_duplicate_joint_names_rejected(self):
        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.SUCCESS
        mock_resp.solution.joint_state.name = ["joint1", "joint2", "joint2", "joint3", "joint4", "joint5"]
        mock_resp.solution.joint_state.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        mock_future.result.return_value = mock_resp

        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future)
        self.assertEqual(self.node._ik_last_status, "DUPLICATE_JOINTS_SOLUTION")
        self.assertEqual(self.node._ik_success_count, 0)
        self.assertIsNone(self.node._latest_ik_solution)

    def test_missing_feedback_rejected(self):
        self.node._latest_joint_positions = {}
        self.node._last_joint_state_time = None

        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.SUCCESS
        mock_resp.solution.joint_state.name = ["joint1", "joint2", "joint3", "joint4", "joint5"]
        mock_resp.solution.joint_state.position = [0.0, 0.0, 0.0, 0.0, 0.0]
        mock_future.result.return_value = mock_resp

        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future)
        self.assertEqual(self.node._ik_last_status, "STALE_OR_MISSING_FEEDBACK")
        self.assertEqual(self.node._ik_success_count, 0)

    def test_stale_feedback_rejected(self):
        self.node._last_joint_state_time = time.monotonic() - 0.50

        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.SUCCESS
        mock_resp.solution.joint_state.name = ["joint1", "joint2", "joint3", "joint4", "joint5"]
        mock_resp.solution.joint_state.position = [0.0, 0.0, 0.0, 0.0, 0.0]
        mock_future.result.return_value = mock_resp

        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future)
        self.assertEqual(self.node._ik_last_status, "STALE_FEEDBACK")
        self.assertEqual(self.node._ik_success_count, 0)

    def test_excessive_joint_delta_rejected(self):
        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.SUCCESS
        mock_resp.solution.joint_state.name = ["joint1", "joint2", "joint3", "joint4", "joint5"]
        mock_resp.solution.joint_state.position = [0.15, 0.0, 0.0, 0.0, 0.0]
        mock_future.result.return_value = mock_resp

        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future)
        self.assertEqual(self.node._ik_last_status, "EXCESSIVE_JOINT_DELTA")
        self.assertEqual(self.node._ik_success_count, 0)

    def test_valid_joint_delta_accepted(self):
        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.SUCCESS
        mock_resp.solution.joint_state.name = ["joint1", "joint2", "joint3", "joint4", "joint5"]
        mock_resp.solution.joint_state.position = [0.05, -0.04, 0.08, -0.02, 0.09]
        mock_future.result.return_value = mock_resp

        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future)
        self.assertEqual(self.node._ik_last_status, "SUCCESS")
        self.assertEqual(self.node._ik_success_count, 1)
        self.assertIsNotNone(self.node._latest_ik_solution)
        self.assertAlmostEqual(self.node._latest_ik_solution["joint1"], 0.05)


class TestArmingHandshakeAndSafetyGates(unittest.TestCase):
    """Test two-key arming handshake, preconditions, and disarming gates (Task 5C-3A & 5C-3A-H Phase D)."""

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = HandTeleopNode("test_arming_node")
        self.node._latest_joint_positions = {
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
        }
        self.node._last_joint_state_time = time.monotonic()
        # Default fresh hardware state
        self.node._hardware_mock_verified = True
        self.node._hardware_last_verified_time = time.monotonic()

    def tearDown(self):
        self.node.destroy_node()

    def test_default_startup_disarmed(self):
        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertFalse(self.node.enable_motion)
        self.assertFalse(self.node.enable_arm_motion)
        self.assertFalse(self.node.enable_ik_dry_run)
        self.assertFalse(self.node.is_armed)

    def test_cannot_arm_by_a_alone(self):
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        ok, reason = self.node.request_arming()
        self.assertTrue(ok)
        self.assertEqual(reason, "ARMING_PENDING")
        self.assertEqual(self.node.teleop_state, "ARMING_PENDING")
        self.assertFalse(self.node.is_armed)

    def test_cannot_arm_by_y_alone(self):
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        ok, reason = self.node.confirm_arming(hand_detected=True, gesture_state="GRIPPER_OPEN")
        self.assertFalse(ok)
        self.assertEqual(reason, "NOT_PENDING")
        self.assertEqual(self.node.teleop_state, "DISARMED")

    def test_arming_timeout_reverts_to_disarmed(self):
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node.request_arming()
        self.node._arming_requested_time = time.monotonic() - 3.5

        ok, reason = self.node.confirm_arming(hand_detected=True, gesture_state="GRIPPER_OPEN")
        self.assertFalse(ok)
        self.assertEqual(reason, "TIMEOUT")
        self.assertEqual(self.node.teleop_state, "DISARMED")

    def test_cannot_arm_when_motion_disabled(self):
        ok_req, reason_req = self.node.request_arming()
        self.assertFalse(ok_req)
        self.assertEqual(reason_req, "MOTION_DISABLED")
        self.assertEqual(self.node.teleop_state, "DISARMED")

        self.node._teleop_state = STATE_ARMING_PENDING
        self.node._arming_requested_time = time.monotonic()
        ok_conf, reason_conf = self.node.confirm_arming(hand_detected=True, gesture_state="GRIPPER_OPEN")
        self.assertFalse(ok_conf)
        self.assertEqual(reason_conf, "MOTION_DISABLED")
        self.assertEqual(self.node.teleop_state, "DISARMED")

    def test_cannot_arm_when_modes_conflict(self):
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_ik_dry_run", value=True),
        ])
        self.assertTrue(self.node.modes_conflict)
        ok1, reason1 = self.node.request_arming()
        self.assertFalse(ok1)
        self.assertEqual(reason1, "MODE_CONFLICT")

    def test_cannot_arm_when_mock_hardware_unverified(self):
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node.request_arming()
        self.node._hardware_mock_verified = False

        ok, reason = self.node.confirm_arming(hand_detected=True, gesture_state="GRIPPER_OPEN")
        self.assertFalse(ok)
        self.assertIn("HARDWARE_NOT_VERIFIED", reason)
        self.assertEqual(self.node.teleop_state, "DISARMED")

    def test_cannot_arm_when_mock_hardware_stale(self):
        """Verify arming fails if hardware check is older than 3.0s."""
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node.request_arming()
        self.node._hardware_mock_verified = True
        self.node._hardware_last_verified_time = time.monotonic() - 5.0  # 5s old

        ok, reason = self.node.confirm_arming(hand_detected=True, gesture_state="GRIPPER_OPEN")
        self.assertFalse(ok)
        self.assertIn("HARDWARE_NOT_VERIFIED", reason)
        self.assertIn("VERIFICATION_EXPIRED", reason)
        self.assertEqual(self.node.teleop_state, "DISARMED")

    def test_cannot_arm_when_hand_not_detected(self):
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node.request_arming()

        ok, reason = self.node.confirm_arming(hand_detected=False, gesture_state="UNKNOWN")
        self.assertFalse(ok)
        self.assertEqual(reason, "HAND_NOT_READY")

    def test_cannot_arm_when_feedback_is_stale(self):
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node.request_arming()
        self.node._last_joint_state_time = time.monotonic() - 0.50

        ok, reason = self.node.confirm_arming(hand_detected=True, gesture_state="GRIPPER_OPEN")
        self.assertFalse(ok)
        self.assertEqual(reason, "STALE_OR_MISSING_FEEDBACK")

    def test_arming_succeeds_when_all_preconditions_met(self):
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_arm_motion", value=True),
        ])
        self.node.request_arming()
        self.node._hardware_mock_verified = True
        self.node._hardware_last_verified_time = time.monotonic()
        self.node.arm_action_client.server_is_ready = MagicMock(return_value=True)

        ok, reason = self.node.confirm_arming(hand_detected=True, gesture_state="GRIPPER_OPEN")
        self.assertTrue(ok)
        self.assertEqual(reason, "ARMED")
        self.assertEqual(self.node.teleop_state, "ARMED — SIMULATION ONLY")
        self.assertTrue(self.node.is_armed)

    def test_disarm_via_d_key(self):
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        self.node.disarm("USER_KEYPRESS_D")
        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertFalse(self.node.is_armed)


class TestTrajectoryTimingAndInterpolation(unittest.TestCase):
    """Test cubic spline duration calculation and input validation (Task 5C-3A-H Phase A)."""

    def setUp(self):
        self.curr = {"joint1": 0.0, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}

    def test_duration_at_max_joint_delta(self):
        """
        Verify that at max allowed joint delta (0.10 rad), duration is 0.75s,
        guaranteeing rest-to-rest cubic peak velocity <= 0.20 rad/s.
        """
        cand = {"joint1": 0.10, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        goal, duration = build_trajectory_goal(cand, self.curr)

        # 1.5 * 0.10 / 0.20 = 0.75 s
        self.assertAlmostEqual(duration, 0.75, places=4)
        peak_vel = 1.5 * 0.10 / duration
        self.assertAlmostEqual(peak_vel, 0.20, places=4)
        self.assertLessEqual(peak_vel, 0.20)

        # Check goal timestamp
        pt = goal.trajectory.points[0]
        self.assertEqual(pt.time_from_start.sec, 0)
        self.assertEqual(pt.time_from_start.nanosec, 750000000)

    def test_duration_at_half_joint_delta(self):
        """
        Verify that for smaller deltas (0.05 rad), duration enforces MIN_TRAJECTORY_DURATION (0.50s),
        and peak velocity is strictly below 0.20 rad/s.
        """
        cand = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        goal, duration = build_trajectory_goal(cand, self.curr)

        # 1.5 * 0.05 / 0.20 = 0.375s < 0.50s minimum
        self.assertAlmostEqual(duration, 0.50, places=4)
        peak_vel = 1.5 * 0.05 / duration  # 0.15 rad/s
        self.assertAlmostEqual(peak_vel, 0.15, places=4)
        self.assertLess(peak_vel, 0.20)

    def test_rejection_of_nan_in_candidate(self):
        cand = {"joint1": float("nan"), "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        with self.assertRaises(ValueError):
            build_trajectory_goal(cand, self.curr)

    def test_rejection_of_inf_in_candidate(self):
        cand = {"joint1": float("inf"), "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        with self.assertRaises(ValueError):
            build_trajectory_goal(cand, self.curr)

    def test_rejection_of_nan_in_current_feedback(self):
        cand = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        curr = {"joint1": float("nan"), "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        with self.assertRaises(ValueError):
            build_trajectory_goal(cand, curr)

    def test_rejection_of_excessive_delta(self):
        """Verify joint delta > 0.10 rad raises ValueError."""
        cand = {"joint1": 0.12, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        with self.assertRaises(ValueError):
            build_trajectory_goal(cand, self.curr)

    def test_rejection_of_missing_joint(self):
        cand = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0}  # missing joint5
        with self.assertRaises(ValueError):
            build_trajectory_goal(cand, self.curr)


class TestOperatingModeIKGating(unittest.TestCase):
    """Test /compute_ik request gating across all five operating modes (Task 5C-3A-H Phase C)."""

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = HandTeleopNode("test_mode_gating_node")
        self.node.set_parameters([rclpy.parameter.Parameter("enable_arm_motion", value=True)])

    def tearDown(self):
        self.node.destroy_node()

    def test_mode_1_disabled_startup(self):
        """Mode 1: enable_motion=False, enable_ik_dry_run=False, DISARMED -> No IK, no motion."""
        self.assertFalse(self.node.enable_motion)
        self.assertFalse(self.node.enable_ik_dry_run)
        self.assertEqual(self.node.teleop_state, "DISARMED")

        self.assertFalse(self.node.can_issue_ik_requests())
        self.assertFalse(self.node.request_ik())
        self.assertEqual(self.node._trajectory_dispatch_count, 0)

    def test_mode_2_ik_dry_run(self):
        """Mode 2: enable_motion=False, enable_ik_dry_run=True, DISARMED -> IK dry-run permitted, zero motion."""
        self.node.set_parameters([rclpy.parameter.Parameter("enable_ik_dry_run", value=True)])
        self.assertFalse(self.node.enable_motion)
        self.assertTrue(self.node.enable_ik_dry_run)
        self.assertEqual(self.node.teleop_state, "DISARMED")

        self.assertTrue(self.node.can_issue_ik_requests())

        # When IK succeeds, zero motion goals must be dispatched
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._last_joint_state_time = time.monotonic()
        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.SUCCESS
        mock_resp.solution.joint_state.name = list(ARM_CONTROL_JOINTS)
        mock_resp.solution.joint_state.position = [0.05, 0.0, 0.0, 0.0, 0.0]
        mock_future.result.return_value = mock_resp

        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future)

        self.assertEqual(self.node._ik_last_status, "SUCCESS")
        self.assertEqual(self.node._trajectory_dispatch_count, 0)
        self.assertFalse(self.node._goal_active)

    def test_mode_3_motion_disarmed(self):
        """Mode 3: enable_motion=True, enable_ik_dry_run=False, DISARMED -> No IK, arming required."""
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.assertTrue(self.node.enable_motion)
        self.assertFalse(self.node.enable_ik_dry_run)
        self.assertEqual(self.node.teleop_state, "DISARMED")

        self.assertFalse(self.node.can_issue_ik_requests())
        self.assertFalse(self.node.request_ik())
        self.assertEqual(self.node._trajectory_dispatch_count, 0)

    def test_mode_4_armed_motion(self):
        """Mode 4: enable_motion=True, enable_ik_dry_run=False, ARMED -> IK permitted, motion execution enabled."""
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node._teleop_state = STATE_ARMED
        self.assertTrue(self.node.enable_motion)
        self.assertFalse(self.node.enable_ik_dry_run)
        self.assertTrue(self.node.is_armed)

        self.assertTrue(self.node.can_issue_ik_requests())

    def test_mode_5_conflict(self):
        """Mode 5: enable_motion=True, enable_ik_dry_run=True -> Conflict rejected, remain disarmed, zero IK."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_ik_dry_run", value=True),
        ])
        self.assertTrue(self.node.modes_conflict)
        self.assertFalse(self.node.can_issue_ik_requests())
        self.assertFalse(self.node.request_ik())
        self.assertEqual(self.node.teleop_state, "DISARMED")


class TestTrajectoryActionClient(unittest.TestCase):
    """Test goal construction, single active goal, queuing, and cancellation (Task 5C-3A & 5C-3A-H)."""

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = HandTeleopNode("test_action_node")
        self.node.set_parameters([rclpy.parameter.Parameter("enable_arm_motion", value=True)])
        self.node._latest_joint_positions = {
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
        }
        self.node._last_joint_state_time = time.monotonic()
        # Default fresh hardware state
        self.node._hardware_mock_verified = True
        self.node._hardware_last_verified_time = time.monotonic()

    def tearDown(self):
        self.node.destroy_node()

    def test_goal_construction_joint_names_and_positions(self):
        candidate = {"joint1": 0.05, "joint2": -0.05, "joint3": 0.08, "joint4": -0.02, "joint5": 0.09}
        current = {"joint1": 0.0, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}

        goal, duration = build_trajectory_goal(candidate, current)
        self.assertEqual(goal.trajectory.joint_names, ["joint1", "joint2", "joint3", "joint4", "joint5"])
        self.assertNotIn("joint6", goal.trajectory.joint_names)
        self.assertEqual(len(goal.trajectory.points), 1)

        pt = goal.trajectory.points[0]
        self.assertEqual(list(pt.positions), [0.05, -0.05, 0.08, -0.02, 0.09])
        self.assertEqual(list(pt.velocities), [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_no_goal_sent_when_disarmed(self):
        candidate = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.assertEqual(self.node.teleop_state, "DISARMED")
        res = self.node.dispatch_arm_trajectory(candidate)
        self.assertFalse(res)
        self.assertEqual(self.node._trajectory_dispatch_count, 0)

    def test_single_active_goal_guard(self):
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node._goal_active = True

        candidate = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        res = self.node.dispatch_arm_trajectory(candidate)
        self.assertFalse(res)
        self.assertEqual(self.node._trajectory_dispatch_count, 0)

    def test_candidate_buffered_during_active_goal(self):
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        self.node._goal_active = True

        candidate1 = {"joint1": 0.04, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node._handle_valid_ik_solution(candidate1)
        self.assertEqual(self.node._pending_ik_candidate, candidate1)

        candidate2 = {"joint1": 0.07, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node._handle_valid_ik_solution(candidate2)
        self.assertEqual(self.node._pending_ik_candidate, candidate2)

    def test_dispatch_fails_and_disarms_when_hardware_check_is_stale(self):
        """Verify dispatch_arm_trajectory aborts and disarms if hardware check age > 3.0s."""
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node.arm_action_client.server_is_ready = MagicMock(return_value=True)

        self.node._hardware_mock_verified = True
        self.node._hardware_last_verified_time = time.monotonic() - 5.0  # 5s old

        candidate = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        res = self.node.dispatch_arm_trajectory(candidate)

        self.assertFalse(res)
        self.assertEqual(self.node._trajectory_dispatch_count, 0)
        self.assertEqual(self.node.teleop_state, "DISARMED")

    def test_disarm_requests_cancellation_of_active_goal(self):
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        self.node._goal_active = True
        mock_handle = MagicMock()
        self.node._active_goal_handle = mock_handle
        self.node._pending_ik_candidate = {"joint1": 0.05}

        self.node.disarm("USER_REQUEST")

        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertFalse(self.node._goal_active)
        self.assertIsNone(self.node._active_goal_handle)
        self.assertIsNone(self.node._pending_ik_candidate)
        mock_handle.cancel_goal_async.assert_called_once()


class TestNoOpSuppressionAndAccumulation(unittest.TestCase):
    """Test target-level and joint-level no-op suppression and accumulation (Task 5C-3A-F Phase A)."""

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = HandTeleopNode("test_noop_node")
        self.node.set_parameters([rclpy.parameter.Parameter("enable_arm_motion", value=True)])
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._last_joint_state_time = time.monotonic()
        self.node._hardware_mock_verified = True
        self.node._hardware_last_verified_time = time.monotonic()
        self.node.arm_action_client.server_is_ready = MagicMock(return_value=True)
        self.node.arm_action_client.send_goal_async = MagicMock()
        self.node.ik_client.service_is_ready = MagicMock(return_value=True)
        self.node.ik_client.call_async = MagicMock()

    def tearDown(self):
        self.node.destroy_node()

    def test_stationary_target_suppresses_repeated_ik_requests(self):
        """Stationary target does not generate repeated IK requests."""
        self.node.set_parameters([rclpy.parameter.Parameter("enable_ik_dry_run", value=True)])
        self.assertTrue(self.node.can_issue_ik_requests())

        # First request should proceed
        self.node._last_ik_time = 0.0
        res1 = self.node.request_ik()
        self.assertTrue(res1)
        self.assertEqual(self.node._ik_call_count, 0)
        self.assertIsNotNone(self.node._last_submitted_ik_target)

        # Immediate second request while stationary and throttled -> False
        self.node._ik_pending = False
        res2 = self.node.request_ik()
        self.assertFalse(res2)

        # Second request after throttle period, but target has not changed (< 1 mm) -> False
        self.node._last_ik_time = time.monotonic() - 1.0
        res3 = self.node.request_ik()
        self.assertFalse(res3)

    def test_stationary_target_suppresses_zero_displacement_goals(self):
        """Stationary target does not generate repeated zero-displacement trajectory goals."""
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])

        # Candidate identical to current position (zero delta)
        candidate_zero = {j: 0.0 for j in ARM_CONTROL_JOINTS}
        res = self.node.dispatch_arm_trajectory(candidate_zero)
        self.assertFalse(res)
        self.assertEqual(self.node._trajectory_last_status, "NO_OP_SUPPRESSED")
        self.assertEqual(self.node._trajectory_dispatch_count, 0)
        self.node.arm_action_client.send_goal_async.assert_not_called()

        # Candidate with sub-deadband delta (0.002 rad < 0.005 rad deadband)
        candidate_sub = {j: 0.002 for j in ARM_CONTROL_JOINTS}
        res_sub = self.node.dispatch_arm_trajectory(candidate_sub)
        self.assertFalse(res_sub)
        self.assertEqual(self.node._trajectory_last_status, "NO_OP_SUPPRESSED")
        self.assertEqual(self.node._trajectory_dispatch_count, 0)
        self.node.arm_action_client.send_goal_async.assert_not_called()

    def test_meaningful_target_change_generates_valid_candidate(self):
        """Moving Cartesian target above MIN_IK_TARGET_DISPLACEMENT allows fresh IK request."""
        self.node.set_parameters([rclpy.parameter.Parameter("enable_ik_dry_run", value=True)])
        self.node._last_ik_time = 0.0
        self.assertTrue(self.node.request_ik())

        # Reset pending flag and throttle timer
        self.node._ik_pending = False
        self.node._last_ik_time = time.monotonic() - 1.0

        # Displace target by 2 mm (> 1 mm threshold)
        self.node.update_cartesian_target(0.002, 0.0, 0.0)
        res = self.node.request_ik()
        self.assertTrue(res)

    def test_tiny_increments_accumulate_to_trigger_ik(self):
        """Tiny increments (0.2 mm) accumulate across frames until >= 1 mm, then trigger IK."""
        self.node.set_parameters([rclpy.parameter.Parameter("enable_ik_dry_run", value=True)])
        self.node._last_ik_time = 0.0
        self.assertTrue(self.node.request_ik())
        self.node._ik_pending = False

        # Add 3 tiny steps of 0.2 mm = 0.6 mm total (still < 1.0 mm)
        for _ in range(3):
            self.node.update_cartesian_target(0.0002, 0.0, 0.0)
        self.node._last_ik_time = time.monotonic() - 1.0
        self.assertFalse(self.node.request_ik())

        # Add 3 more steps of 0.2 mm = 1.2 mm total (now >= 1.0 mm)
        for _ in range(3):
            self.node.update_cartesian_target(0.0002, 0.0, 0.0)
        self.assertTrue(self.node.request_ik())

    def test_meaningful_joint_delta_dispatches_successfully(self):
        """Candidate with delta >= MIN_DISPATCH_JOINT_DELTA dispatches trajectory goal."""
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])

        candidate_valid = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        res = self.node.dispatch_arm_trajectory(candidate_valid)
        self.assertTrue(res)
        self.assertEqual(self.node._trajectory_dispatch_count, 1)
        self.assertEqual(self.node._trajectory_last_status, "SENDING")
        self.assertTrue(self.node._goal_active)
        self.node.arm_action_client.send_goal_async.assert_called_once()


class TestDynamicDisarmingAndWatchdog(unittest.TestCase):
    """Test dynamic parameter disarming and watchdog behavior (Task 5C-3A-F Phase B)."""

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = HandTeleopNode("test_dynamic_node")
        self.node.set_parameters([rclpy.parameter.Parameter("enable_arm_motion", value=True)])
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._last_joint_state_time = time.monotonic()
        self.node._hardware_mock_verified = True
        self.node._hardware_last_verified_time = time.monotonic()

    def tearDown(self):
        self.node.destroy_node()

    def test_dynamic_enable_motion_false_forces_disarmed(self):
        """Setting enable_motion=False while ARMED immediately transitions to DISARMED."""
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        self.assertTrue(self.node.is_armed)

        # Dynamic parameter update
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=False)])
        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertEqual(self.node._last_disarm_reason, "MOTION_DISABLED")

    def test_dynamic_disarm_clears_pending_and_cancels_goal(self):
        """Disarming clears pending candidates, snapshots, and cancels active goal."""
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        self.node._goal_active = True
        mock_handle = MagicMock()
        self.node._active_goal_handle = mock_handle
        self.node._pending_ik_candidate = {"joint1": 0.05}
        self.node._pending_ik_snapshot = TargetSnapshot(0.0, 0.0, 0.0, 1, time.monotonic())

        # Trigger dynamic disarm
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=False)])
        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertFalse(self.node._goal_active)
        self.assertIsNone(self.node._active_goal_handle)
        self.assertIsNone(self.node._pending_ik_candidate)
        self.assertIsNone(self.node._pending_ik_snapshot)
        mock_handle.cancel_goal_async.assert_called_once()

    def test_reenabling_motion_does_not_auto_arm(self):
        """Re-enabling enable_motion leaves node DISARMED; does not restore ARMED state."""
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"

        # Disable
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=False)])
        self.assertEqual(self.node.teleop_state, "DISARMED")

        # Re-enable
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertFalse(self.node.is_armed)

    def test_dynamic_mode_conflict_forces_disarm(self):
        """Setting enable_ik_dry_run=True while enable_motion=True forces immediate DISARMED."""
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"

        # Introduce conflict
        self.node.set_parameters([rclpy.parameter.Parameter("enable_ik_dry_run", value=True)])
        self.assertTrue(self.node.modes_conflict)
        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertEqual(self.node._last_disarm_reason, "MODE_CONFLICT")
        self.assertFalse(self.node.can_issue_ik_requests())


class TestTargetSnapshotAndObsoleteIKRejection(unittest.TestCase):
    """Test target snapshot tracking and obsolete IK rejection (Task 5C-3A-F Phase C)."""

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = HandTeleopNode("test_snapshot_node")
        self.node.set_parameters([rclpy.parameter.Parameter("enable_arm_motion", value=True)])
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._last_joint_state_time = time.monotonic()
        self.node._hardware_mock_verified = True
        self.node._hardware_last_verified_time = time.monotonic()
        self.node.arm_action_client.server_is_ready = MagicMock(return_value=True)
        self.node.arm_action_client.send_goal_async = MagicMock()

    def tearDown(self):
        self.node.destroy_node()

    def _create_mock_ik_future(self, joint1_pos=0.05):
        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.SUCCESS
        mock_resp.solution.joint_state.name = list(ARM_CONTROL_JOINTS)
        mock_resp.solution.joint_state.position = [joint1_pos, 0.0, 0.0, 0.0, 0.0]
        mock_future.result.return_value = mock_resp
        return mock_future

    def test_ik_response_for_unchanged_target_handled_normally(self):
        """IK response for matching target snapshot is accepted."""
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])

        snapshot = TargetSnapshot(
            x=self.node._target_x,
            y=self.node._target_y,
            z=self.node._target_z,
            generation=self.node._target_generation,
            timestamp=time.monotonic(),
        )
        mock_future = self._create_mock_ik_future(0.05)
        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future, snapshot=snapshot)

        self.assertEqual(self.node._ik_last_status, "SUCCESS")
        self.assertIsNotNone(self.node._latest_ik_solution)
        self.assertEqual(self.node._latest_ik_solution["joint1"], 0.05)

    def test_stale_ik_target_response_rejected(self):
        """IK response for materially outdated target snapshot (> 10 mm drift) is rejected."""
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])

        # Snapshot for old target
        snapshot = TargetSnapshot(
            x=self.node._target_x - 0.020,  # 20 mm away (> 10 mm MAX_IK_TARGET_DRIFT)
            y=self.node._target_y,
            z=self.node._target_z,
            generation=self.node._target_generation - 5,
            timestamp=time.monotonic() - 1.0,
        )
        mock_future = self._create_mock_ik_future(0.05)
        self.node._ik_pending = True
        self.node._ik_response_callback(mock_future, snapshot=snapshot)

        self.assertEqual(self.node._ik_last_status, "OBSOLETE_TARGET_REJECTED")
        self.assertIsNone(self.node._latest_ik_solution)
        self.assertEqual(self.node._trajectory_dispatch_count, 0)

    def test_buffered_candidate_outdated_target_discarded_on_goal_completion(self):
        """Buffered candidate is discarded on goal completion if target drifted > 10 mm."""
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node._goal_active = True

        # Buffer a candidate with snapshot at initial target
        candidate = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        snapshot = TargetSnapshot(
            x=self.node._target_x,
            y=self.node._target_y,
            z=self.node._target_z,
            generation=self.node._target_generation,
            timestamp=time.monotonic(),
        )
        self.node._handle_valid_ik_solution(candidate, snapshot=snapshot)
        self.assertEqual(self.node._pending_ik_candidate, candidate)
        self.assertEqual(self.node._pending_ik_snapshot, snapshot)

        # Move Cartesian target by 15 mm (> 10 mm MAX_IK_TARGET_DRIFT)
        self.node.update_cartesian_target(0.015, 0.0, 0.0)

        # Goal completes
        mock_future = MagicMock()
        wrapped_res = MagicMock()
        wrapped_res.status = GoalStatus.STATUS_SUCCEEDED
        mock_future.result.return_value = wrapped_res

        self.node._goal_result_callback(mock_future)

        # Candidate must be discarded, not dispatched
        self.assertFalse(self.node._goal_active)
        self.assertIsNone(self.node._pending_ik_candidate)
        self.assertIsNone(self.node._pending_ik_snapshot)
        self.assertEqual(self.node._trajectory_dispatch_count, 0)

    def test_disarm_and_hand_loss_clears_target_pending_state(self):
        """Disarming or hand loss clears candidate, snapshot, and target submission state."""
        self.node._pending_ik_candidate = {"joint1": 0.05}
        self.node._pending_ik_snapshot = TargetSnapshot(0.0, 0.0, 0.0, 1, time.monotonic())
        self.node._last_submitted_ik_target = (0.02, -0.27, 0.26)

        self.node.disarm("TEST_DISARM")

        self.assertIsNone(self.node._pending_ik_candidate)
        self.assertIsNone(self.node._pending_ik_snapshot)
        self.assertIsNone(self.node._last_submitted_ik_target)

    def test_callbacks_cannot_dispatch_goals_after_disarming(self):
        """Goal result callback cannot dispatch buffered candidate if node is disarmed."""
        self.node._teleop_state = "DISARMED"
        self.node._pending_ik_candidate = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node._pending_ik_snapshot = TargetSnapshot(self.node._target_x, self.node._target_y, self.node._target_z, 1, time.monotonic())

        mock_future = MagicMock()
        wrapped_res = MagicMock()
        wrapped_res.status = GoalStatus.STATUS_SUCCEEDED
        mock_future.result.return_value = wrapped_res

        self.node._goal_result_callback(mock_future)

        self.assertEqual(self.node._trajectory_dispatch_count, 0)
        self.assertIsNone(self.node._pending_ik_candidate)
        self.assertIsNone(self.node._pending_ik_snapshot)


class TestSingleGoalTestMode(unittest.TestCase):
    """Test suite for single-trajectory test mode safeguards (Task 5C-3A-SG)."""

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = HandTeleopNode("test_single_goal_node")
        self.node.set_parameters([rclpy.parameter.Parameter("enable_arm_motion", value=True)])
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._last_joint_state_time = time.monotonic()
        self.node._hardware_mock_verified = True
        self.node._hardware_last_verified_time = time.monotonic()
        self.node.arm_action_client.server_is_ready = MagicMock(return_value=True)
        self.mock_send_future = MagicMock()
        self.node.arm_action_client.send_goal_async = MagicMock(return_value=self.mock_send_future)

    def tearDown(self):
        self.node.destroy_node()

    def test_default_parameter_and_counter(self):
        """single_goal_test_mode defaults to False and session_goal_dispatch_count to 0."""
        self.assertFalse(self.node.single_goal_test_mode)
        self.assertEqual(self.node.session_goal_dispatch_count, 0)
        diag = self.node.get_cartesian_and_ik_diagnostics()
        self.assertIn("single_goal_test_mode", diag)
        self.assertIn("session_goal_dispatch_count", diag)
        self.assertFalse(diag["single_goal_test_mode"])
        self.assertEqual(diag["session_goal_dispatch_count"], 0)

    def test_multi_goal_behavior_unchanged_when_disabled(self):
        """When single_goal_test_mode=False, multiple goals can be dispatched sequentially."""
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"

        cand1 = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.assertTrue(self.node.dispatch_arm_trajectory(cand1))
        self.assertEqual(self.node._trajectory_dispatch_count, 1)
        self.assertEqual(self.node.session_goal_dispatch_count, 1)

        # Goal completes
        mock_future = MagicMock()
        wrapped_res = MagicMock()
        wrapped_res.status = GoalStatus.STATUS_SUCCEEDED
        mock_future.result.return_value = wrapped_res
        self.node._goal_result_callback(mock_future)

        # Still armed, can dispatch second goal
        self.assertEqual(self.node.teleop_state, "ARMED — SIMULATION ONLY")
        cand2 = {"joint1": 0.10, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.assertTrue(self.node.dispatch_arm_trajectory(cand2))
        self.assertEqual(self.node._trajectory_dispatch_count, 2)
        self.assertEqual(self.node.session_goal_dispatch_count, 2)

    def test_disarmed_node_cannot_dispatch_in_single_goal_mode(self):
        """Disarmed node cannot dispatch goals or issue IK in single-goal mode."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("single_goal_test_mode", value=True),
            rclpy.parameter.Parameter("enable_motion", value=True),
        ])
        self.node._teleop_state = "DISARMED"
        cand = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.assertFalse(self.node.dispatch_arm_trajectory(cand))
        self.assertEqual(self.node.session_goal_dispatch_count, 0)

        self.assertFalse(self.node.can_issue_ik_requests())

    def test_single_goal_limit_enforced_per_session(self):
        """When single_goal_test_mode=True, second dispatch in the same session is rejected."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("single_goal_test_mode", value=True),
            rclpy.parameter.Parameter("enable_motion", value=True),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"

        cand1 = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.assertTrue(self.node.dispatch_arm_trajectory(cand1))
        self.assertEqual(self.node._trajectory_dispatch_count, 1)
        self.assertEqual(self.node.session_goal_dispatch_count, 1)

        # Reset active goal flag artificially to simulate finished goal without disarm
        self.node._goal_active = False

        cand2 = {"joint1": 0.10, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.assertFalse(self.node.dispatch_arm_trajectory(cand2))
        self.assertEqual(self.node._trajectory_last_status, "SINGLE_GOAL_LIMIT_EXCEEDED")
        self.assertEqual(self.node.session_goal_dispatch_count, 1)

    def test_candidate_dropped_during_active_goal_in_single_goal_mode(self):
        """New IK candidate is dropped (not buffered) when session goal limit is reached."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("single_goal_test_mode", value=True),
            rclpy.parameter.Parameter("enable_motion", value=True),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"

        cand1 = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.assertTrue(self.node.dispatch_arm_trajectory(cand1))
        self.assertTrue(self.node._goal_active)
        self.assertEqual(self.node.session_goal_dispatch_count, 1)

        # Arriving candidate during active goal
        cand2 = {"joint1": 0.08, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        snap = TargetSnapshot(self.node._target_x, self.node._target_y, self.node._target_z, 1, time.monotonic())
        self.node._handle_valid_ik_solution(cand2, snapshot=snap)

        self.assertIsNone(self.node._pending_ik_candidate)
        self.assertIsNone(self.node._pending_ik_snapshot)

    def test_ik_queries_suppressed_after_single_goal_dispatch(self):
        """can_issue_ik_requests returns False once session goal is dispatched."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("single_goal_test_mode", value=True),
            rclpy.parameter.Parameter("enable_motion", value=True),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"

        self.assertTrue(self.node.can_issue_ik_requests())

        cand1 = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.assertTrue(self.node.dispatch_arm_trajectory(cand1))
        self.assertEqual(self.node.session_goal_dispatch_count, 1)

        # Now IK queries must be suppressed
        self.assertFalse(self.node.can_issue_ik_requests())

    def test_accepted_goal_completion_disarms_with_reason(self):
        """Goal succeeding in single_goal_test_mode automatically disarms with SINGLE_GOAL_TEST_COMPLETED."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("single_goal_test_mode", value=True),
            rclpy.parameter.Parameter("enable_motion", value=True),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        cand = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node.dispatch_arm_trajectory(cand)

        mock_future = MagicMock()
        wrapped_res = MagicMock()
        wrapped_res.status = GoalStatus.STATUS_SUCCEEDED
        mock_future.result.return_value = wrapped_res

        self.node._goal_result_callback(mock_future)

        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertEqual(self.node._last_disarm_reason, "SINGLE_GOAL_TEST_COMPLETED")
        self.assertFalse(self.node.is_armed)
        self.assertFalse(self.node._goal_active)
        self.assertEqual(self.node.session_goal_dispatch_count, 1)

    def test_aborted_goal_completion_disarms_with_reason(self):
        """Goal aborting in single_goal_test_mode automatically disarms with SINGLE_GOAL_TEST_COMPLETED."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("single_goal_test_mode", value=True),
            rclpy.parameter.Parameter("enable_motion", value=True),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        cand = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node.dispatch_arm_trajectory(cand)

        mock_future = MagicMock()
        wrapped_res = MagicMock()
        wrapped_res.status = GoalStatus.STATUS_ABORTED
        mock_future.result.return_value = wrapped_res

        self.node._goal_result_callback(mock_future)

        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertEqual(self.node._last_disarm_reason, "SINGLE_GOAL_TEST_COMPLETED")
        self.assertFalse(self.node.is_armed)

    def test_rejected_goal_disarms_with_reason(self):
        """Rejected goal in single_goal_test_mode immediately disarms with SINGLE_GOAL_TEST_REJECTED."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("single_goal_test_mode", value=True),
            rclpy.parameter.Parameter("enable_motion", value=True),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        cand = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node.dispatch_arm_trajectory(cand)

        mock_future = MagicMock()
        mock_handle = MagicMock()
        mock_handle.accepted = False
        mock_future.result.return_value = mock_handle

        self.node._goal_response_callback(mock_future)

        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertEqual(self.node._last_disarm_reason, "SINGLE_GOAL_TEST_REJECTED")
        self.assertEqual(self.node._trajectory_last_status, "GOAL_REJECTED")

    def test_send_failed_disarms_with_reason(self):
        """Exception during goal sending in single_goal_test_mode disarms with SINGLE_GOAL_TEST_SEND_FAILED."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("single_goal_test_mode", value=True),
            rclpy.parameter.Parameter("enable_motion", value=True),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        cand = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node.dispatch_arm_trajectory(cand)

        mock_future = MagicMock()
        mock_future.result.side_effect = RuntimeError("Transport failure")

        self.node._goal_response_callback(mock_future)

        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertEqual(self.node._last_disarm_reason, "SINGLE_GOAL_TEST_SEND_FAILED")

    def test_buffered_candidate_purged_and_never_dispatched_in_single_goal_mode(self):
        """Even if candidate was somehow buffered, it is purged and not dispatched on completion."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("single_goal_test_mode", value=True),
            rclpy.parameter.Parameter("enable_motion", value=True),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        self.node._goal_active = True
        self.node._session_goal_dispatch_count = 1

        # Force a buffered candidate
        self.node._pending_ik_candidate = {"joint1": 0.08, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node._pending_ik_snapshot = TargetSnapshot(self.node._target_x, self.node._target_y, self.node._target_z, 1, time.monotonic())

        mock_future = MagicMock()
        wrapped_res = MagicMock()
        wrapped_res.status = GoalStatus.STATUS_SUCCEEDED
        mock_future.result.return_value = wrapped_res

        self.node._goal_result_callback(mock_future)

        self.assertIsNone(self.node._pending_ik_candidate)
        self.assertIsNone(self.node._pending_ik_snapshot)
        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertEqual(self.node._trajectory_dispatch_count, 0)  # No new dispatches occurred

    def test_disarm_retains_session_counter(self):
        """disarm() does not reset session_goal_dispatch_count; preserves until next arming handshake."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("single_goal_test_mode", value=True),
            rclpy.parameter.Parameter("enable_motion", value=True),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        cand = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node.dispatch_arm_trajectory(cand)
        self.assertEqual(self.node.session_goal_dispatch_count, 1)

        self.node.disarm("TEST_DISARM")
        self.assertEqual(self.node.session_goal_dispatch_count, 1)

    def test_new_arming_session_resets_counter_and_allows_one_new_goal(self):
        """Completing a fresh A-then-Y arming handshake resets session counter to 0."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("single_goal_test_mode", value=True),
            rclpy.parameter.Parameter("enable_motion", value=True),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        cand1 = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node.dispatch_arm_trajectory(cand1)
        self.assertEqual(self.node.session_goal_dispatch_count, 1)

        # Disarm
        self.node.disarm("SINGLE_GOAL_TEST_COMPLETED")
        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertEqual(self.node.session_goal_dispatch_count, 1)

        # Re-arm via handshake
        ok_a, _ = self.node.request_arming()
        self.assertTrue(ok_a)
        ok_y, state = self.node.confirm_arming(hand_detected=True, gesture_state="GRIPPER_OPEN")
        self.assertTrue(ok_y)
        self.assertEqual(state, "ARMED")
        self.assertEqual(self.node.session_goal_dispatch_count, 0)

        # Exactly 1 new goal can be dispatched
        self.assertTrue(self.node.can_issue_ik_requests())
        cand2 = {"joint1": 0.06, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.assertTrue(self.node.dispatch_arm_trajectory(cand2))
        self.assertEqual(self.node.session_goal_dispatch_count, 1)

        # Further goals in this 2nd session are rejected
        self.node._goal_active = False
        cand3 = {"joint1": 0.07, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.assertFalse(self.node.dispatch_arm_trajectory(cand3))

    def test_dynamic_enable_single_goal_test_mode_while_limit_exceeded(self):
        """Enabling single_goal_test_mode dynamically when session count >= 1 forces disarm."""
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        cand = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node.dispatch_arm_trajectory(cand)
        self.assertEqual(self.node.session_goal_dispatch_count, 1)

        # Dynamically enable single_goal_test_mode
        self.node.set_parameters([rclpy.parameter.Parameter("single_goal_test_mode", value=True)])
        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertEqual(self.node._last_disarm_reason, "SINGLE_GOAL_LIMIT_EXCEEDED")

    def test_synchronous_send_goal_exception_single_goal_mode(self):
        """Synchronous exception from send_goal_async in single-goal mode disarms, resets active flag, and clears candidates."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("single_goal_test_mode", value=True),
            rclpy.parameter.Parameter("enable_motion", value=True),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        self.node._pending_ik_candidate = {"joint1": 0.08, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node._pending_ik_snapshot = TargetSnapshot(self.node._target_x, self.node._target_y, self.node._target_z, 1, time.monotonic())

        # Cause send_goal_async to raise synchronously
        self.node.arm_action_client.send_goal_async.side_effect = RuntimeError("Synchronous socket error")

        cand = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        result = self.node.dispatch_arm_trajectory(cand)

        self.assertFalse(result)
        self.assertFalse(self.node._goal_active)
        self.assertIsNone(self.node._active_goal_handle)
        self.assertIsNone(self.node._pending_ik_candidate)
        self.assertIsNone(self.node._pending_ik_snapshot)
        self.assertEqual(self.node._trajectory_last_status, "SEND_FAILED_Synchronous socket error")
        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertEqual(self.node._last_disarm_reason, "SINGLE_GOAL_TEST_SEND_FAILED")
        self.assertEqual(self.node.session_goal_dispatch_count, 1)

        # Confirm failed candidate cannot be retried
        self.assertFalse(self.node.dispatch_arm_trajectory(cand))

    def test_synchronous_send_goal_exception_normal_mode(self):
        """Synchronous exception from send_goal_async in normal mode does not wedge node and clears pending candidate."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("single_goal_test_mode", value=False),
            rclpy.parameter.Parameter("enable_motion", value=True),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        self.node._pending_ik_candidate = {"joint1": 0.08, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node._pending_ik_snapshot = TargetSnapshot(self.node._target_x, self.node._target_y, self.node._target_z, 1, time.monotonic())
        self.node._last_joint_state_time = time.monotonic()
        self.node._hardware_last_verified_time = time.monotonic()

        # Cause send_goal_async to raise synchronously
        self.node.arm_action_client.send_goal_async.side_effect = RuntimeError("Direct client crash")

        cand = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        result = self.node.dispatch_arm_trajectory(cand)

        self.assertFalse(result)
        self.assertFalse(self.node._goal_active)
        self.assertIsNone(self.node._active_goal_handle)
        self.assertIsNone(self.node._pending_ik_candidate)
        self.assertIsNone(self.node._pending_ik_snapshot)
        self.assertEqual(self.node._trajectory_last_status, "SEND_FAILED_Direct client crash")
        self.assertEqual(self.node.teleop_state, "ARMED — SIMULATION ONLY")

        # Restore client: confirm node is NOT wedged and can dispatch a new valid candidate
        self.node.arm_action_client.send_goal_async.side_effect = None
        self.node.arm_action_client.send_goal_async.return_value = MagicMock()
        self.node._last_joint_state_time = time.monotonic()
        self.node._hardware_last_verified_time = time.monotonic()
        cand2 = {"joint1": 0.06, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.assertTrue(self.node.dispatch_arm_trajectory(cand2))
        self.assertTrue(self.node._goal_active)

    def test_synchronous_exception_conservative_session_counter_blocks_retry(self):
        """Even if state is artificially set to ARMED, conservative counter blocks second dispatch attempt."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("single_goal_test_mode", value=True),
            rclpy.parameter.Parameter("enable_motion", value=True),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"

        self.node.arm_action_client.send_goal_async.side_effect = RuntimeError("Network error")
        cand = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node.dispatch_arm_trajectory(cand)

        self.assertEqual(self.node.session_goal_dispatch_count, 1)

        # Artificially set ARMED without fresh handshake: dispatch must STILL be rejected by session count limit
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        self.node.arm_action_client.send_goal_async.side_effect = None
        cand2 = {"joint1": 0.06, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.assertFalse(self.node.dispatch_arm_trajectory(cand2))
        self.assertEqual(self.node._trajectory_last_status, "SINGLE_GOAL_LIMIT_EXCEEDED")


class TestBoundedSessionGoalLimit(unittest.TestCase):
    """Test suite for configurable bounded per-session goal limits (Task 5C-3C-SG)."""

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = HandTeleopNode("test_bounded_session_node")
        self.node.set_parameters([rclpy.parameter.Parameter("enable_arm_motion", value=True)])
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._last_joint_state_time = time.monotonic()
        self.node._hardware_mock_verified = True
        self.node._hardware_last_verified_time = time.monotonic()
        self.node.arm_action_client.server_is_ready = MagicMock(return_value=True)
        self.mock_send_future = MagicMock()
        self.node.arm_action_client.send_goal_async = MagicMock(return_value=self.mock_send_future)

    def tearDown(self):
        self.node.destroy_node()

    def test_1_default_parameter_and_effective_limit(self):
        """max_goals_per_session defaults to 0 (unlimited), effective limit is 0."""
        self.assertEqual(self.node.max_goals_per_session, 0)
        self.assertEqual(self.node.effective_session_goal_limit, 0)
        self.assertFalse(self.node.is_session_goal_limit_reached())
        diag = self.node.get_cartesian_and_ik_diagnostics()
        self.assertEqual(diag["max_goals_per_session"], 0)
        self.assertEqual(diag["effective_session_goal_limit"], 0)
        self.assertFalse(diag["is_session_goal_limit_reached"])

    def test_2_negative_limit_handling_fails_closed(self):
        """Negative limit is rejected by on_set parameter callback, fails closed in properties and arming."""
        # Pre-set parameter callback rejects negative value
        param = rclpy.parameter.Parameter("max_goals_per_session", value=-2)
        res = self.node._on_set_parameters_callback([param])
        self.assertFalse(res.successful)
        self.assertIn("cannot be negative", res.reason)

        # Confirm set_parameters with negative value returns unsuccessful result
        results = self.node.set_parameters([param])
        self.assertFalse(results[0].successful)

        # If forced/configured negative, fails closed immediately
        with patch.object(HandTeleopNode, "max_goals_per_session", new_callable=PropertyMock) as mock_limit:
            mock_limit.return_value = -1
            self.assertEqual(self.node.effective_session_goal_limit, -1)
            self.assertTrue(self.node.is_session_goal_limit_reached())
            self.assertFalse(self.node.can_issue_ik_requests())

            # Arming handshake must reject negative parameter
            self.node._teleop_state = "ARMING_PENDING"
            self.node._arming_requested_time = time.monotonic()
            self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
            ok_y, reason = self.node.confirm_arming(hand_detected=True, gesture_state="GRIPPER_OPEN")
            self.assertFalse(ok_y)
            self.assertEqual(reason, "INVALID_PARAMETER_VALUE")
            self.assertEqual(self.node.teleop_state, "DISARMED")

    def test_3_single_goal_test_mode_takes_precedence(self):
        """single_goal_test_mode=True forces effective limit to 1 regardless of max_goals_per_session."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("single_goal_test_mode", value=True),
            rclpy.parameter.Parameter("max_goals_per_session", value=10),
        ])
        self.assertEqual(self.node.effective_session_goal_limit, 1)

        self.node.set_parameters([
            rclpy.parameter.Parameter("max_goals_per_session", value=0),
        ])
        self.assertEqual(self.node.effective_session_goal_limit, 1)

        with patch.object(HandTeleopNode, "max_goals_per_session", new_callable=PropertyMock) as mock_limit:
            mock_limit.return_value = -3
            self.assertEqual(self.node.effective_session_goal_limit, 1)

    def test_4_normal_mode_limit_zero_preserves_unlimited_multigoal(self):
        """Normal mode with limit 0 preserves existing multi-goal behavior without disarming."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("max_goals_per_session", value=0),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"

        for i in range(1, 4):
            cand = {"joint1": 0.02 * i, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
            self.assertTrue(self.node.dispatch_arm_trajectory(cand))
            self.assertEqual(self.node.session_goal_dispatch_count, i)

            # Complete goal
            mock_future = MagicMock()
            wrapped_res = MagicMock()
            wrapped_res.status = GoalStatus.STATUS_SUCCEEDED
            mock_future.result.return_value = wrapped_res
            self.node._goal_result_callback(mock_future)

            # Node must remain armed after each goal
            self.assertEqual(self.node.teleop_state, "ARMED — SIMULATION ONLY")

    def test_5_normal_mode_limit_one_blocks_second_attempt(self):
        """Normal mode with max_goals_per_session=1 disallows a second goal dispatch."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("max_goals_per_session", value=1),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"

        cand1 = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.assertTrue(self.node.dispatch_arm_trajectory(cand1))
        self.assertEqual(self.node.session_goal_dispatch_count, 1)
        self.assertTrue(self.node.is_session_goal_limit_reached())

        # Simulated active goal clearing
        self.node._goal_active = False
        cand2 = {"joint1": 0.08, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.assertFalse(self.node.dispatch_arm_trajectory(cand2))
        self.assertEqual(self.node._trajectory_last_status, "SESSION_GOAL_LIMIT_EXCEEDED")

    def test_6_normal_mode_limit_two_permits_two_dispatches(self):
        """Normal mode with max_goals_per_session=2 permits exactly two trajectory dispatches."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("max_goals_per_session", value=2),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"

        # Goal 1
        cand1 = {"joint1": 0.04, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.assertTrue(self.node.dispatch_arm_trajectory(cand1))
        self.assertEqual(self.node.session_goal_dispatch_count, 1)
        self.assertFalse(self.node.is_session_goal_limit_reached())

        # Goal 1 completes
        mock_future = MagicMock()
        wrapped_res = MagicMock()
        wrapped_res.status = GoalStatus.STATUS_SUCCEEDED
        mock_future.result.return_value = wrapped_res
        self.node._goal_result_callback(mock_future)
        self.assertEqual(self.node.teleop_state, "ARMED — SIMULATION ONLY")

        # Goal 2
        cand2 = {"joint1": 0.08, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.assertTrue(self.node.dispatch_arm_trajectory(cand2))
        self.assertEqual(self.node.session_goal_dispatch_count, 2)
        self.assertTrue(self.node.is_session_goal_limit_reached())

    def test_7_third_candidate_rejected_at_two_goal_limit(self):
        """When limit=2 is reached, a 3rd dispatch attempt is rejected."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("max_goals_per_session", value=2),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"

        cand1 = {"joint1": 0.03, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        cand2 = {"joint1": 0.06, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        cand3 = {"joint1": 0.09, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}

        self.assertTrue(self.node.dispatch_arm_trajectory(cand1))
        self.node._goal_active = False
        self.assertTrue(self.node.dispatch_arm_trajectory(cand2))
        self.node._goal_active = False

        # Attempt 3 must fail
        self.assertFalse(self.node.dispatch_arm_trajectory(cand3))
        self.assertEqual(self.node._trajectory_last_status, "SESSION_GOAL_LIMIT_EXCEEDED")
        self.assertEqual(self.node.session_goal_dispatch_count, 2)

    def test_8_ik_requests_suppressed_at_limit(self):
        """can_issue_ik_requests returns False as soon as session limit is reached."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("max_goals_per_session", value=2),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"

        self.assertTrue(self.node.can_issue_ik_requests())

        cand1 = {"joint1": 0.03, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node.dispatch_arm_trajectory(cand1)
        self.assertTrue(self.node.can_issue_ik_requests())  # 1 < 2

        self.node._goal_active = False
        cand2 = {"joint1": 0.06, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node.dispatch_arm_trajectory(cand2)
        self.assertFalse(self.node.can_issue_ik_requests())  # 2 >= 2

    def test_9_buffered_candidates_cleared_at_limit(self):
        """When limit is reached, incoming candidates are dropped and not buffered."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("max_goals_per_session", value=1),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"

        cand1 = {"joint1": 0.04, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node.dispatch_arm_trajectory(cand1)
        self.assertTrue(self.node._goal_active)

        # Candidate arrives while goal 1 is active, but limit (1) was already reached
        cand2 = {"joint1": 0.08, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        snap = TargetSnapshot(self.node._target_x, self.node._target_y, self.node._target_z, 1, time.monotonic())
        self.node._handle_valid_ik_solution(cand2, snapshot=snap)

        self.assertIsNone(self.node._pending_ik_candidate)
        self.assertIsNone(self.node._pending_ik_snapshot)

    def test_10_stale_ik_responses_cannot_bypass_limit(self):
        """Delayed /compute_ik response arriving after limit is reached is dropped."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("max_goals_per_session", value=1),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        self.node._session_goal_dispatch_count = 1  # Limit consumed

        # Construct mock IK response
        mock_future = MagicMock()
        mock_resp = MagicMock()
        mock_resp.error_code.val = MoveItErrorCodes.SUCCESS
        mock_resp.solution.joint_state.name = list(ARM_CONTROL_JOINTS)
        mock_resp.solution.joint_state.position = [0.05, 0.0, 0.0, 0.0, 0.0]
        mock_future.result.return_value = mock_resp

        snap = TargetSnapshot(self.node._target_x, self.node._target_y, self.node._target_z, 0, time.monotonic())
        self.node._ik_response_callback(mock_future, snapshot=snap)

        self.assertIsNone(self.node._pending_ik_candidate)
        self.assertEqual(self.node._trajectory_dispatch_count, 0)

    def test_11_auto_disarm_after_terminal_result_at_limit(self):
        """Node automatically disarms with SESSION_GOAL_LIMIT_COMPLETED after final allowed goal finishes."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("max_goals_per_session", value=2),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"

        cand1 = {"joint1": 0.03, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node.dispatch_arm_trajectory(cand1)
        mock_future1 = MagicMock()
        wrapped_res1 = MagicMock()
        wrapped_res1.status = GoalStatus.STATUS_SUCCEEDED
        mock_future1.result.return_value = wrapped_res1
        self.node._goal_result_callback(mock_future1)
        self.assertEqual(self.node.teleop_state, "ARMED — SIMULATION ONLY")

        cand2 = {"joint1": 0.06, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node.dispatch_arm_trajectory(cand2)
        mock_future2 = MagicMock()
        wrapped_res2 = MagicMock()
        wrapped_res2.status = GoalStatus.STATUS_SUCCEEDED
        mock_future2.result.return_value = wrapped_res2
        self.node._goal_result_callback(mock_future2)

        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertEqual(self.node._last_disarm_reason, "SESSION_GOAL_LIMIT_COMPLETED")

    def test_12_rejection_at_limit_cannot_trigger_retry(self):
        """When final allowed goal is rejected by action server, node disarms with SESSION_GOAL_LIMIT_REJECTED."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("max_goals_per_session", value=1),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"

        cand = {"joint1": 0.04, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node.dispatch_arm_trajectory(cand)

        mock_future = MagicMock()
        mock_handle = MagicMock()
        mock_handle.accepted = False
        mock_future.result.return_value = mock_handle
        self.node._goal_response_callback(mock_future)

        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertEqual(self.node._last_disarm_reason, "SESSION_GOAL_LIMIT_REJECTED")
        self.assertIsNone(self.node._pending_ik_candidate)

    def test_13_aborted_or_canceled_goal_at_limit_cannot_retry(self):
        """Aborted final allowed goal disarms with SESSION_GOAL_LIMIT_COMPLETED and clears pending candidates."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("max_goals_per_session", value=1),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"

        cand = {"joint1": 0.04, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node.dispatch_arm_trajectory(cand)

        mock_future = MagicMock()
        wrapped_res = MagicMock()
        wrapped_res.status = GoalStatus.STATUS_ABORTED
        mock_future.result.return_value = wrapped_res
        self.node._goal_result_callback(mock_future)

        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertEqual(self.node._last_disarm_reason, "SESSION_GOAL_LIMIT_COMPLETED")
        self.assertIsNone(self.node._pending_ik_candidate)

    def test_14_synchronous_send_exception_consumes_attempt_and_disarms_at_limit(self):
        """Synchronous send_goal_async exception consumes an attempt and disarms with SESSION_GOAL_LIMIT_SEND_FAILED."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("max_goals_per_session", value=1),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        self.node.arm_action_client.send_goal_async.side_effect = RuntimeError("Immediate send error")

        cand = {"joint1": 0.04, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        result = self.node.dispatch_arm_trajectory(cand)

        self.assertFalse(result)
        self.assertEqual(self.node.session_goal_dispatch_count, 1)
        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertEqual(self.node._last_disarm_reason, "SESSION_GOAL_LIMIT_SEND_FAILED")
        self.assertIsNone(self.node._pending_ik_candidate)

    def test_15_asynchronous_send_failure_cannot_retry(self):
        """Async exception in goal future disarms with SESSION_GOAL_LIMIT_SEND_FAILED and clears candidate."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("max_goals_per_session", value=1),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"

        cand = {"joint1": 0.04, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node.dispatch_arm_trajectory(cand)

        mock_future = MagicMock()
        mock_future.result.side_effect = RuntimeError("Async future failure")
        self.node._goal_response_callback(mock_future)

        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertEqual(self.node._last_disarm_reason, "SESSION_GOAL_LIMIT_SEND_FAILED")
        self.assertIsNone(self.node._pending_ik_candidate)

    def test_16_fresh_handshake_resets_counter_and_permits_new_bounded_session(self):
        """Completing an A-then-Y handshake resets session counter to 0 for a new bounded session."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("max_goals_per_session", value=2),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"

        # Dispatch 2 goals to exhaust session
        cand1 = {"joint1": 0.02, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        cand2 = {"joint1": 0.04, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node.dispatch_arm_trajectory(cand1)
        self.node._goal_active = False
        self.node.dispatch_arm_trajectory(cand2)
        self.assertEqual(self.node.session_goal_dispatch_count, 2)
        self.assertTrue(self.node.is_session_goal_limit_reached())

        # Disarm
        self.node.disarm("SESSION_GOAL_LIMIT_COMPLETED")
        self.assertEqual(self.node.session_goal_dispatch_count, 2)

        # Re-arm via A-then-Y
        ok_a, _ = self.node.request_arming()
        self.assertTrue(ok_a)
        ok_y, _ = self.node.confirm_arming(hand_detected=True, gesture_state="GRIPPER_OPEN")
        self.assertTrue(ok_y)
        self.assertEqual(self.node.session_goal_dispatch_count, 0)
        self.assertFalse(self.node.is_session_goal_limit_reached())
        self.assertTrue(self.node.can_issue_ik_requests())

        # Dispatches up to 2 goals in new session
        cand3 = {"joint1": 0.06, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.assertTrue(self.node.dispatch_arm_trajectory(cand3))
        self.assertEqual(self.node.session_goal_dispatch_count, 1)

    def test_17_dynamic_reduction_of_limit_while_armed(self):
        """Dynamically reducing limit to <= current session count forces immediate disarm if idle, or on goal completion."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("max_goals_per_session", value=5),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"

        cand1 = {"joint1": 0.02, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        cand2 = {"joint1": 0.04, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.node.dispatch_arm_trajectory(cand1)
        self.node._goal_active = False
        self.node.dispatch_arm_trajectory(cand2)
        self.node._goal_active = False
        self.assertEqual(self.node.session_goal_dispatch_count, 2)

        # Reduce limit to 2 dynamically when idle: must disarm immediately
        self.node.set_parameters([rclpy.parameter.Parameter("max_goals_per_session", value=2)])
        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertEqual(self.node._last_disarm_reason, "SESSION_GOAL_LIMIT_EXCEEDED")

    def test_18_existing_motion_gates_and_single_goal_diagnostics_correct(self):
        """single_goal_test_mode diagnostics and reasons remain unchanged; motion disabled blocks dispatch."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=False),
            rclpy.parameter.Parameter("max_goals_per_session", value=2),
        ])
        self.node._teleop_state = "ARMED — SIMULATION ONLY"
        cand = {"joint1": 0.05, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0, "joint5": 0.0}
        self.assertFalse(self.node.dispatch_arm_trajectory(cand))
        self.assertEqual(self.node.session_goal_dispatch_count, 0)


# ===========================================================================
# Task 5C-3D: Two-Finger Gesture Classification Tests
# ===========================================================================
class TestTwoFingerGestureClassification(unittest.TestCase):
    """
    Focused unit tests for two-finger gripper gesture classification (Task 5C-3D).
    Verifies that:
      - Thumb EXTENDED + Index EXTENDED -> GRIPPER_OPEN (regardless of Middle finger).
      - Thumb FOLDED + Index FOLDED -> GRIPPER_CLOSED (regardless of Middle finger).
      - Thumb and Index disagreement -> UNKNOWN.
      - Middle finger state has ZERO influence on the outcome.
      - Ring and pinky fingers are ignored.
      - 4-frame debounce smoothing is preserved.
    """

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        class MockLM:
            def __init__(self, x=0.0, y=0.0, z=0.0):
                self.x = x
                self.y = y
                self.z = z

        self.landmarks = [MockLM(0.0, 0.0, 0.0) for _ in range(21)]
        self.landmarks[0] = MockLM(0.0, 0.0, 0.0)   # wrist
        self.landmarks[9] = MockLM(0.0, 1.0, 0.0)   # middle_mcp (palm_scale = 1.0)

    @patch("hand_gesture_teleop.hand_tracking.classify_single_finger")
    @patch("hand_gesture_teleop.hand_tracking.classify_thumb")
    def test_open_requires_thumb_and_index_extended_middle_neutral(self, mock_thumb, mock_single):
        """Thumb EXTENDED + Index EXTENDED yields GRIPPER_OPEN for any Middle finger state."""
        for mid_state in ("EXTENDED", "FOLDED", "UNKNOWN", "OTHER"):
            mock_thumb.return_value = "EXTENDED"
            mock_single.side_effect = ["EXTENDED", mid_state]
            states, gesture = classify_hand_gestures(self.landmarks)
            self.assertEqual(gesture, "GRIPPER_OPEN", f"Failed with middle={mid_state}")
            self.assertEqual(states["thumb"], "EXTENDED")
            self.assertEqual(states["index"], "EXTENDED")
            self.assertEqual(states["middle"], mid_state)

    @patch("hand_gesture_teleop.hand_tracking.classify_single_finger")
    @patch("hand_gesture_teleop.hand_tracking.classify_thumb")
    def test_closed_requires_thumb_and_index_folded_middle_neutral(self, mock_thumb, mock_single):
        """Thumb FOLDED + Index FOLDED yields GRIPPER_CLOSED for any Middle finger state."""
        for mid_state in ("EXTENDED", "FOLDED", "UNKNOWN", "OTHER"):
            mock_thumb.return_value = "FOLDED"
            mock_single.side_effect = ["FOLDED", mid_state]
            states, gesture = classify_hand_gestures(self.landmarks)
            self.assertEqual(gesture, "GRIPPER_CLOSED", f"Failed with middle={mid_state}")
            self.assertEqual(states["thumb"], "FOLDED")
            self.assertEqual(states["index"], "FOLDED")
            self.assertEqual(states["middle"], mid_state)

    @patch("hand_gesture_teleop.hand_tracking.classify_single_finger")
    @patch("hand_gesture_teleop.hand_tracking.classify_thumb")
    def test_disagreement_and_ambiguity_yields_unknown(self, mock_thumb, mock_single):
        """Any disagreement between thumb and index or presence of UNKNOWN yields gesture UNKNOWN."""
        disagree_pairs = [
            ("EXTENDED", "FOLDED"),
            ("FOLDED", "EXTENDED"),
            ("EXTENDED", "UNKNOWN"),
            ("UNKNOWN", "EXTENDED"),
            ("FOLDED", "UNKNOWN"),
            ("UNKNOWN", "FOLDED"),
            ("UNKNOWN", "UNKNOWN"),
        ]
        for th, idx in disagree_pairs:
            for mid in ("EXTENDED", "FOLDED", "UNKNOWN"):
                mock_thumb.return_value = th
                mock_single.side_effect = [idx, mid]
                states, gesture = classify_hand_gestures(self.landmarks)
                self.assertEqual(
                    gesture, "UNKNOWN",
                    f"Expected UNKNOWN for thumb={th}, index={idx}, middle={mid}"
                )

    def test_degenerate_landmarks_yield_all_unknown(self):
        """Palm scale < 1e-4 fails closed to UNKNOWN for all fingers and gesture."""
        class MockLM:
            def __init__(self, x=0.0, y=0.0, z=0.0):
                self.x = x
                self.y = y
                self.z = z

        lms = [MockLM(0.0, 0.0, 0.0) for _ in range(21)]
        states, gesture = classify_hand_gestures(lms)
        self.assertEqual(gesture, "UNKNOWN")
        self.assertEqual(states["thumb"], "UNKNOWN")
        self.assertEqual(states["index"], "UNKNOWN")
        self.assertEqual(states["middle"], "UNKNOWN")

    def test_four_frame_debounce_temporal_smoother(self):
        """TemporalSmoother requires exactly 4 consecutive frames before confirming a new gesture."""
        smoother = TemporalSmoother(confirm_frames=4, default_state="UNKNOWN")
        self.assertEqual(smoother.confirmed_state, "UNKNOWN")

        # 3 frames of GRIPPER_OPEN: still UNKNOWN
        for _ in range(3):
            self.assertEqual(smoother.update("GRIPPER_OPEN"), "UNKNOWN")

        # 4th frame: transitions to GRIPPER_OPEN
        self.assertEqual(smoother.update("GRIPPER_OPEN"), "GRIPPER_OPEN")

        # 3 frames of GRIPPER_CLOSED: still GRIPPER_OPEN
        for _ in range(3):
            self.assertEqual(smoother.update("GRIPPER_CLOSED"), "GRIPPER_OPEN")

        # 4th frame of GRIPPER_CLOSED: transitions to GRIPPER_CLOSED
        self.assertEqual(smoother.update("GRIPPER_CLOSED"), "GRIPPER_CLOSED")

        # Reset returns to UNKNOWN
        smoother.reset("UNKNOWN")
        self.assertEqual(smoother.confirmed_state, "UNKNOWN")


# ===========================================================================
# Task 5C-3D: Enlarged Window Sizing Tests
# ===========================================================================
class TestEnlargedWindowSizing(unittest.TestCase):
    """
    Focused unit tests for display window dimension calculations (Task 5C-3D).
    Verifies that:
      - 4:3 aspect ratio is strictly preserved across arbitrary screen sizes.
      - Width <= 75% screen width and Height <= 75% screen height.
      - Centering calculations are correct.
      - Fallback behavior operates headlessly without raising errors.
    """

    def test_window_sizing_1080p(self):
        """1920x1080 standard desktop -> 1080x810 (75% height bound, 4:3 preserved)."""
        tw, th = calculate_aspect_preserving_window_size(1920, 1080, aspect_ratio=4.0/3.0, scale_factor=0.75)
        self.assertEqual((tw, th), (1080, 810))
        self.assertAlmostEqual(tw / th, 4.0 / 3.0, places=4)
        self.assertLessEqual(tw, int(1920 * 0.75))
        self.assertLessEqual(th, int(1080 * 0.75))
        self.assertEqual(th, int(1080 * 0.75))

    def test_window_sizing_1440p(self):
        """2560x1440 QHD desktop -> 1440x1080 (75% height bound, 4:3 preserved)."""
        tw, th = calculate_aspect_preserving_window_size(2560, 1440, aspect_ratio=4.0/3.0, scale_factor=0.75)
        self.assertEqual((tw, th), (1440, 1080))
        self.assertAlmostEqual(tw / th, 4.0 / 3.0, places=4)
        self.assertLessEqual(tw, int(2560 * 0.75))
        self.assertLessEqual(th, int(1440 * 0.75))

    def test_window_sizing_4k(self):
        """3840x2160 4K UHD desktop -> 2160x1620 (75% height bound, 4:3 preserved)."""
        tw, th = calculate_aspect_preserving_window_size(3840, 2160, aspect_ratio=4.0/3.0, scale_factor=0.75)
        self.assertEqual((tw, th), (2160, 1620))
        self.assertAlmostEqual(tw / th, 4.0 / 3.0, places=4)
        self.assertLessEqual(tw, int(3840 * 0.75))
        self.assertLessEqual(th, int(2160 * 0.75))

    def test_window_sizing_768p(self):
        """1366x768 laptop desktop preserves 4:3 and remains within 75% bounds."""
        tw, th = calculate_aspect_preserving_window_size(1366, 768, aspect_ratio=4.0/3.0, scale_factor=0.75)
        self.assertAlmostEqual(tw / th, 4.0 / 3.0, delta=0.01)
        self.assertLessEqual(tw, int(1366 * 0.75))
        self.assertLessEqual(th, int(768 * 0.75))

    def test_window_sizing_portrait_screen(self):
        """1080x1920 portrait display is constrained by width, preserving 4:3."""
        tw, th = calculate_aspect_preserving_window_size(1080, 1920, aspect_ratio=4.0/3.0, scale_factor=0.75)
        self.assertEqual(tw, int(1080 * 0.75))  # 810
        self.assertEqual(th, int(round(810 / (4.0 / 3.0))))  # 608
        self.assertAlmostEqual(tw / th, 4.0 / 3.0, delta=0.01)
        self.assertLessEqual(tw, int(1080 * 0.75))
        self.assertLessEqual(th, int(1920 * 0.75))

    def test_get_screen_resolution_headless_fallback(self):
        """When DISPLAY is absent, fallback dimensions are returned cleanly."""
        with patch.dict(os.environ, {}, clear=True):
            w, h = get_screen_resolution(fallback_width=1600, fallback_height=900)
            self.assertEqual((w, h), (1600, 900))

    def test_setup_enlarged_camera_window_headless_safe(self):
        """setup_enlarged_camera_window executes without unhandled exceptions headlessly."""
        tw, th = setup_enlarged_camera_window("test_headless_win")
        self.assertGreater(tw, 0)
        self.assertGreater(th, 0)
        self.assertAlmostEqual(tw / th, 4.0 / 3.0, delta=0.02)


# ===========================================================================
# Task 5C-3D: Gripper Trajectory Actuation Tests
# ===========================================================================
class TestGripperActuation(unittest.TestCase):
    """
    Focused unit tests for gripper trajectory actuation path (Task 5C-3D).
    Verifies that:
      - Trajectory action client uses /gripper_controller/follow_joint_trajectory and joint6.
      - Disarmed state prevents any gripper action.
      - enable_motion=False prevents any gripper action.
      - Stale or missing joint feedback prevents any gripper action.
      - Stale or unverified mock hardware prevents any gripper action.
      - Repeated identical gesture states are deduplicated (no duplicate goals).
      - Transition from OPEN to CLOSED commands 0.0 rad.
      - Transition from CLOSED to OPEN commands 1.0 rad.
      - UNKNOWN gesture commands no gripper goal.
      - Targets respect configured limits [-0.2, 2.0].
      - No overlapping gripper goals while one is in flight.
      - Fail-closed handling on send exception or rejection.
      - Cancels active goal on disarm.
      - Arm counters and trajectory client remain completely separate.
    """

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = HandTeleopNode("test_gripper_actuation_node")
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._latest_joint_velocities = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._last_joint_state_time = time.monotonic()
        self.node._hardware_mock_verified = True
        self.node._hardware_last_verified_time = time.monotonic()
        self.node.arm_action_client.server_is_ready = MagicMock(return_value=True)
        self.node.gripper_action_client.server_is_ready = MagicMock(return_value=True)
        self.mock_future = MagicMock()
        self.node.gripper_action_client.send_goal_async = MagicMock(return_value=self.mock_future)

    def tearDown(self):
        self.node.destroy_node()

    def test_client_endpoints_and_isolation(self):
        """Gripper client is FollowJointTrajectory on /gripper_controller/follow_joint_trajectory."""
        self.assertTrue(hasattr(self.node, "gripper_action_client"))
        self.assertFalse(hasattr(self.node, "gripper_pub"))
        self.assertEqual(
            self.node.gripper_action_client._action_name,
            "/gripper_controller/follow_joint_trajectory",
        )
        self.assertEqual(
            self.node.arm_action_client._action_name,
            "/arm_controller/follow_joint_trajectory",
        )

    def test_disarmed_blocks_gripper_dispatch(self):
        """Gripper goal dispatch rejected when teleop node is DISARMED."""
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node._teleop_state = STATE_DISARMED
        res = self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertFalse(res)
        self.assertEqual(self.node._gripper_dispatch_count, 0)
        self.node.gripper_action_client.send_goal_async.assert_not_called()

    def test_motion_disabled_blocks_gripper_dispatch(self):
        """Gripper goal dispatch rejected when enable_motion is False."""
        self.node._teleop_state = STATE_ARMED
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=False)])
        res = self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertFalse(res)
        self.assertEqual(self.node._gripper_dispatch_count, 0)
        self.node.gripper_action_client.send_goal_async.assert_not_called()

    def test_mode_conflict_blocks_gripper_dispatch(self):
        """Gripper goal dispatch rejected when both enable_motion and enable_ik_dry_run are True."""
        self.node._teleop_state = STATE_ARMED
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_ik_dry_run", value=True),
        ])
        res = self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertFalse(res)
        self.assertEqual(self.node._gripper_dispatch_count, 0)
        self.node.gripper_action_client.send_goal_async.assert_not_called()

    def test_unverified_hardware_blocks_gripper_dispatch(self):
        """Gripper goal dispatch rejected when mock hardware is not positively verified."""
        self.node._teleop_state = STATE_ARMED
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node._hardware_mock_verified = False
        res = self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertFalse(res)
        self.assertEqual(self.node._gripper_dispatch_count, 0)
        self.node.gripper_action_client.send_goal_async.assert_not_called()

    def test_stale_joint_feedback_blocks_gripper_dispatch(self):
        """Gripper goal dispatch rejected when /joint_states feedback is older than 0.25s."""
        self.node._teleop_state = STATE_ARMED
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node._last_joint_state_time = time.monotonic() - 1.0
        res = self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertFalse(res)
        self.assertEqual(self.node._gripper_dispatch_count, 0)
        self.node.gripper_action_client.send_goal_async.assert_not_called()

    def test_missing_joint6_blocks_gripper_dispatch(self):
        """Gripper goal dispatch rejected when joint6 is missing from feedback positions."""
        self.node._teleop_state = STATE_ARMED
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        del self.node._latest_joint_positions["joint6"]
        res = self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertFalse(res)
        self.assertEqual(self.node._gripper_dispatch_count, 0)
        self.node.gripper_action_client.send_goal_async.assert_not_called()

    def test_unknown_gesture_blocks_gripper_dispatch(self):
        """Gesture state UNKNOWN never commands a gripper trajectory."""
        self.node._teleop_state = STATE_ARMED
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        res = self.node._send_gripper_command("UNKNOWN")
        self.assertFalse(res)
        self.assertEqual(self.node._gripper_dispatch_count, 0)
        self.node.gripper_action_client.send_goal_async.assert_not_called()

    def test_successful_gripper_open_dispatch_target_and_point(self):
        """GRIPPER_OPEN dispatches FollowJointTrajectory with joint6 at 1.0 rad."""
        self.node._teleop_state = STATE_ARMED
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        res = self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertTrue(res)
        self.assertEqual(self.node._gripper_dispatch_count, 1)
        self.assertTrue(self.node._gripper_goal_active)
        self.assertEqual(self.node._last_commanded_gripper_state, "GRIPPER_OPEN")

        self.node.gripper_action_client.send_goal_async.assert_called_once()
        sent_goal = self.node.gripper_action_client.send_goal_async.call_args[0][0]
        self.assertEqual(sent_goal.trajectory.joint_names, ["joint6"])
        self.assertEqual(len(sent_goal.trajectory.points), 2)
        pt0 = sent_goal.trajectory.points[0]
        pt1 = sent_goal.trajectory.points[1]
        self.assertEqual(list(pt0.positions), [0.0])
        self.assertEqual(list(pt0.velocities), [0.0])
        self.assertEqual(pt0.time_from_start.sec, 0)
        self.assertEqual(pt0.time_from_start.nanosec, 0)
        self.assertEqual(list(pt1.positions), [1.0])
        self.assertEqual(list(pt1.velocities), [0.0])
        self.assertEqual(pt1.time_from_start.sec, 1)

    def test_successful_gripper_closed_dispatch_target(self):
        """GRIPPER_CLOSED dispatches FollowJointTrajectory with joint6 at 0.0 rad."""
        self.node._teleop_state = STATE_ARMED
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        res = self.node._send_gripper_command("GRIPPER_CLOSED")
        self.assertTrue(res)
        self.assertEqual(self.node._gripper_dispatch_count, 1)
        sent_goal = self.node.gripper_action_client.send_goal_async.call_args[0][0]
        self.assertEqual(len(sent_goal.trajectory.points), 2)
        self.assertEqual(list(sent_goal.trajectory.points[-1].positions), [0.0])

    def test_deduplication_suppresses_identical_repeated_frames(self):
        """Repeated frames with the same debounced gesture do NOT submit duplicate goals."""
        self.node._teleop_state = STATE_ARMED
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        res1 = self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertTrue(res1)
        self.node._gripper_goal_active = False

        # Next frame with same gesture: must be suppressed
        res2 = self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertFalse(res2)
        self.assertEqual(self.node._gripper_dispatch_count, 1)

        # Transition to CLOSED: permitted!
        res3 = self.node._send_gripper_command("GRIPPER_CLOSED")
        self.assertTrue(res3)
        self.assertEqual(self.node._gripper_dispatch_count, 2)

    def test_no_overlapping_goals(self):
        """Cannot dispatch a new gripper goal while a previous goal is active in flight."""
        self.node._teleop_state = STATE_ARMED
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        res1 = self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertTrue(res1)
        self.assertTrue(self.node._gripper_goal_active)

        # Transition to CLOSED while OPEN goal is still in flight: blocked
        res2 = self.node._send_gripper_command("GRIPPER_CLOSED")
        self.assertFalse(res2)
        self.assertEqual(self.node._gripper_dispatch_count, 1)

    def test_target_exceeding_joint_limits_fails_closed(self):
        """Target exceeding configured joint limits is rejected and fails closed."""
        self.node._teleop_state = STATE_ARMED
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("gripper_joint_limit_max", value=0.8),
        ])
        res = self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertFalse(res)
        self.assertEqual(self.node._gripper_dispatch_count, 0)
        self.node.gripper_action_client.send_goal_async.assert_not_called()

    def test_goal_rejection_clears_active_state_no_retry(self):
        """Goal rejection by controller clears active flag and fails closed without retrying."""
        self.node._teleop_state = STATE_ARMED
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertTrue(self.node._gripper_goal_active)

        mock_future = MagicMock()
        mock_handle = MagicMock()
        mock_handle.accepted = False
        mock_future.result.return_value = mock_handle

        self.node._gripper_goal_response_callback(mock_future)
        self.assertFalse(self.node._gripper_goal_active)
        self.assertEqual(self.node._gripper_last_status, "REJECTED")

    def test_goal_success_increments_counter(self):
        """Goal success callback increments _gripper_success_count and sets status to SUCCEEDED."""
        self.node._teleop_state = STATE_ARMED
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node._send_gripper_command("GRIPPER_OPEN")

        mock_future = MagicMock()
        mock_wrap = MagicMock()
        mock_wrap.status = GoalStatus.STATUS_SUCCEEDED
        mock_future.result.return_value = mock_wrap

        self.node._gripper_goal_result_callback(mock_future)
        self.assertFalse(self.node._gripper_goal_active)
        self.assertEqual(self.node._gripper_success_count, 1)
        self.assertEqual(self.node._gripper_last_status, "SUCCEEDED")

    def test_disarm_cancels_active_gripper_goal(self):
        """Disarming triggers cancel request on active gripper goal and clears state."""
        self.node._teleop_state = STATE_ARMED
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node._send_gripper_command("GRIPPER_OPEN")
        mock_handle = MagicMock()
        self.node._active_gripper_goal_handle = mock_handle

        self.node.disarm("TEST_DISARM")
        mock_handle.cancel_goal_async.assert_called_once()
        self.assertFalse(self.node._gripper_goal_active)
        self.assertIsNone(self.node._last_commanded_gripper_state)

    def test_arm_counters_isolated_from_gripper_dispatch(self):
        """Gripper command dispatch does NOT increment arm session or total dispatch counters."""
        self.node._teleop_state = STATE_ARMED
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node.arm_action_client.send_goal_async = MagicMock()
        self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertEqual(self.node._gripper_dispatch_count, 1)
        self.assertEqual(self.node._trajectory_dispatch_count, 0)
        self.assertEqual(self.node.session_goal_dispatch_count, 0)
        self.node.arm_action_client.send_goal_async.assert_not_called()


class TestCompactHUDLayout(unittest.TestCase):
    """
    Focused unit tests for Task 5C-3D-UI-FIX:
    Compact top-left HUD layout, native display resolution rendering,
    clear central view, and accurate status and quota presentation.
    """

    def test_consolidated_hud_single_panel(self):
        """Verify that exactly one consolidated HUD panel is rendered in the top-left."""
        frame = np.zeros((810, 1080, 3), dtype=np.uint8)
        card_bbox = draw_compact_hud(
            frame=frame,
            hand_detected=True,
            gesture_state="GRIPPER_OPEN",
            gripper_state_text="OPEN",
            control_state_text="DISARMED",
            is_motion_enabled=False,
            help_text="Hotkeys: [Q/Esc] Quit",
        )
        x1, y1, x2, y2 = card_bbox
        # Verify card is strictly in the top-left
        self.assertLess(x1, 50)
        self.assertLess(y1, 50)
        self.assertGreater(x2, x1)
        self.assertGreater(y2, y1)

        # Verify that draw_mapping_hud delegates to the same consolidated single panel
        frame2 = np.zeros((810, 1080, 3), dtype=np.uint8)
        card_bbox2 = draw_mapping_hud(
            frame=frame2,
            hand_detected=True,
            gesture_state="GRIPPER_OPEN",
            gripper_state_text="OPEN",
            control_state_text="DISARMED",
            is_motion_enabled=False,
            help_text="Hotkeys: [Q/Esc] Quit",
        )
        self.assertEqual(card_bbox, card_bbox2)
        # Pixel rendering must match
        self.assertTrue(np.array_equal(frame, frame2))

    def test_aspect_ratio_preservation(self):
        """Verify that the camera display window strictly preserves 4:3 aspect ratio across screen resolutions."""
        screen_resolutions = [
            (1920, 1080),
            (1366, 768),
            (2560, 1440),
            (1280, 720),
            (3840, 2160),
            (800, 600),
        ]
        target_ratio = 4.0 / 3.0
        for sw, sh in screen_resolutions:
            tw, th = calculate_aspect_preserving_window_size(
                screen_width=sw,
                screen_height=sh,
                aspect_ratio=target_ratio,
                scale_factor=0.75,
            )
            # Fits within screen bounds with 75% scale factor
            self.assertLessEqual(tw, int(sw * 0.75))
            self.assertLessEqual(th, int(sh * 0.75))
            # Preserves 4:3 aspect ratio
            actual_ratio = tw / th
            self.assertAlmostEqual(actual_ratio, target_ratio, delta=0.03)

    def test_hud_scaling_with_display_copy(self):
        """Verify that HUD dimensions scale proportionally with display resolution."""
        frame_large = np.zeros((810, 1080, 3), dtype=np.uint8)
        x1_l, y1_l, x2_l, y2_l = draw_compact_hud(
            frame=frame_large,
            hand_detected=True,
            gesture_state="GRIPPER_OPEN",
        )
        w_l = x2_l - x1_l
        h_l = y2_l - y1_l

        frame_small = np.zeros((480, 640, 3), dtype=np.uint8)
        x1_s, y1_s, x2_s, y2_s = draw_compact_hud(
            frame=frame_small,
            hand_detected=True,
            gesture_state="GRIPPER_OPEN",
        )
        w_s = x2_s - x1_s
        h_s = y2_s - y1_s

        # Large frame panel should be strictly larger than small frame panel
        self.assertGreater(w_l, w_s)
        self.assertGreater(h_l, h_s)

    def test_source_frame_and_landmarks_unmutated(self):
        """Verify that source 640x480 frame and source landmarks are not mutated by display scaling."""
        source_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        source_copy = source_frame.copy()

        class DummyLandmark:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z

        landmarks = [DummyLandmark(0.5, 0.5, 0.0), DummyLandmark(0.3, 0.4, -0.1)]
        orig_coords = [(lm.x, lm.y, lm.z) for lm in landmarks]

        # Display pipeline creates separate copy resized to 1080x810
        display_frame = cv2.resize(source_frame, (1080, 810), interpolation=cv2.INTER_LINEAR)
        draw_compact_hud(
            frame=display_frame,
            hand_detected=True,
            gesture_state="GRIPPER_CLOSED",
        )

        # Source frame must be completely untouched
        self.assertTrue(np.array_equal(source_frame, source_copy))
        # Landmark coordinates must be completely unchanged
        current_coords = [(lm.x, lm.y, lm.z) for lm in landmarks]
        self.assertEqual(orig_coords, current_coords)

    def test_panel_stays_within_upper_left_bounds(self):
        """Verify that HUD card stays within top-left constraints and leaves center clear."""
        resolutions = [(640, 480), (800, 600), (1080, 810), (1280, 960), (1920, 1080)]
        for w, h in resolutions:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            x1, y1, x2, y2 = draw_compact_hud(
                frame=frame,
                hand_detected=True,
                gesture_state="UNKNOWN",
            )
            card_w = x2 - x1
            card_h = y2 - y1

            # Margin checks
            self.assertLess(x1, w * 0.05)
            self.assertLess(y1, h * 0.05)

            # Viewport width fraction: approximately 25-30%
            w_frac = card_w / w
            self.assertLessEqual(w_frac, 0.32)

            # Viewport height fraction: <= 22%
            h_frac = card_h / h
            self.assertLessEqual(h_frac, 0.22)

            # Central hand region (e.g. x in [0.35w, 0.65w], y in [0.25h, 0.75h]) must remain 0
            cx1, cx2 = int(w * 0.35), int(w * 0.65)
            cy1, cy2 = int(h * 0.25), int(h * 0.75)
            center_region = frame[cy1:cy2, cx1:cx2]
            self.assertEqual(np.count_nonzero(center_region), 0, f"Central view occluded on {w}x{h}")

    def test_hud_excludes_removed_debug_fields(self):
        """Verify that removed low-priority debug elements are not rendered."""
        frame = np.zeros((810, 1080, 3), dtype=np.uint8)
        # Using draw_mapping_hud (which delegates to draw_compact_hud)
        draw_mapping_hud(
            frame=frame,
            hand_detected=True,
            gesture_state="GRIPPER_OPEN",
            status_banner="PREVIEW ONLY — NO ROBOT COMMANDS",
        )
        # Bottom status bar area (y > 780) must remain completely untouched
        bottom_area = frame[780:810, :]
        self.assertEqual(np.count_nonzero(bottom_area), 0, "Stray bottom overlay detected")

        # Bottom-left diagnostics area (x in [10, 400], y in [500, 800]) must remain clean
        bottom_left_area = frame[500:800, 10:400]
        self.assertEqual(np.count_nonzero(bottom_left_area), 0, "Duplicate bottom-left ROS overlay detected")

    def test_motion_disabled_mode_heading(self):
        """Verify that motion-disabled mode explicitly displays PREVIEW ONLY — NO ROBOT COMMANDS."""
        frame = np.zeros((810, 1080, 3), dtype=np.uint8)
        draw_compact_hud(
            frame=frame,
            hand_detected=True,
            gesture_state="UNKNOWN",
            is_motion_enabled=False,
        )
        # If motion is disabled and heading is None, default heading is PREVIEW ONLY — NO ROBOT COMMANDS
        # Test with custom heading override
        frame2 = np.zeros((810, 1080, 3), dtype=np.uint8)
        draw_compact_hud(
            frame=frame2,
            hand_detected=True,
            gesture_state="UNKNOWN",
            status_heading="PREVIEW ONLY — NO ROBOT COMMANDS",
        )
        self.assertTrue(np.array_equal(frame, frame2))

    def test_motion_enabled_mode_control_state_and_quota(self):
        """Verify motion-enabled mode displays actual control state and quota without stale data."""
        frame = np.zeros((810, 1080, 3), dtype=np.uint8)
        draw_compact_hud(
            frame=frame,
            hand_detected=True,
            gesture_state="GRIPPER_OPEN",
            gripper_state_text="OPEN",
            control_state_text="ARMED [Session 1/2]",
            is_motion_enabled=True,
            help_text="Hotkeys: [A] Arm | [Y] Confirm | [D] Disarm | [Q] Quit",
        )
        # Heading when motion enabled must differ from motion disabled
        frame_disabled = np.zeros((810, 1080, 3), dtype=np.uint8)
        draw_compact_hud(
            frame=frame_disabled,
            hand_detected=True,
            gesture_state="GRIPPER_OPEN",
            gripper_state_text="IDLE",
            control_state_text="DISARMED",
            is_motion_enabled=False,
        )
        self.assertFalse(np.array_equal(frame, frame_disabled))


# ===========================================================================
# Task 5C-3D-GRIPPER-DYNAMICS-FIX: Trajectory Dynamics & Interpolation Tests
# ===========================================================================
class TestGripperDynamicsAndInterpolation(unittest.TestCase):
    """
    Focused unit & headless regression tests for verified gripper trajectory dynamics (Task 5C-3D-GRIPPER-DYNAMICS-FIX).
    Verifies:
      1. Actual generated gripper trajectory point fields and timestamp layout (2 points, rest-to-rest).
      2. The controller's selected interpolation method and the evidence supporting it.
      3. Correct use of actual joint6 current position as starting waypoint and displacement reference.
      4. Dynamic duration calculation from actual displacement.
      5. Peak velocity and acceleration formulas for rest-to-rest cubic spline trajectory.
      6. A 1.0 rad transition strictly adhering to software velocity and acceleration bounds (v<=10.0, a<=5.0).
      7. Smaller displacements receiving an appropriate duration scaled by acceleration limits.
      8. Duration not falling below configured minimum duration floor (0.5s).
      9. Invalid velocity or acceleration parameter configuration rejected safely (non-positive, non-finite).
      10. Missing, stale (>0.25s), or non-finite joint6 feedback preventing command submission.
      11. Out-of-limit (outside [-0.2, 2.0]) and non-finite targets rejected.
      12. Absence of overlapping gripper goals (active goal or unresolved cancellation prevents dispatch).
      13. Failed submission, rejected goal, canceled goal, and terminal failure not triggering automatic retry.
      14. Existing single-goal, bounded-session, and continuous-mode arm behavior remaining completely unchanged.
      15. Existing two-finger gesture and compact-UI tests continuing to pass.
      16. Arm trajectory goals and arm session counter remaining completely unaffected by gripper changes.
    """

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = HandTeleopNode("test_gripper_dynamics_node")
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._latest_joint_velocities = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._last_joint_state_time = time.monotonic()
        self.node._hardware_mock_verified = True
        self.node._hardware_last_verified_time = time.monotonic()
        self.node.arm_action_client.server_is_ready = MagicMock(return_value=True)
        self.node.gripper_action_client.server_is_ready = MagicMock(return_value=True)
        self.mock_future = MagicMock()
        self.node.gripper_action_client.send_goal_async = MagicMock(return_value=self.mock_future)

    def tearDown(self):
        self.node.destroy_node()

    def test_01_actual_generated_trajectory_point_fields_and_timestamp_layout(self):
        """1. Verify generated trajectory has exact 2 waypoints, field completeness, and monotonic timestamps."""
        goal, duration = build_gripper_trajectory_goal(
            target_pos=1.0,
            current_pos=0.0,
            velocity_limit=10.0,
            acceleration_limit=5.0,
            min_duration=0.5,
        )
        self.assertEqual(goal.trajectory.joint_names, ["joint6"])
        self.assertEqual(goal.trajectory.header.stamp.sec, 0)
        self.assertEqual(goal.trajectory.header.stamp.nanosec, 0)
        self.assertEqual(len(goal.trajectory.points), 2)

        pt0 = goal.trajectory.points[0]
        self.assertEqual(list(pt0.positions), [0.0])
        self.assertEqual(list(pt0.velocities), [0.0])
        self.assertEqual(len(pt0.accelerations), 0)
        self.assertEqual(len(pt0.effort), 0)
        self.assertEqual(pt0.time_from_start.sec, 0)
        self.assertEqual(pt0.time_from_start.nanosec, 0)

        pt1 = goal.trajectory.points[1]
        self.assertEqual(list(pt1.positions), [1.0])
        self.assertEqual(list(pt1.velocities), [0.0])
        self.assertEqual(len(pt1.accelerations), 0)
        self.assertEqual(len(pt1.effort), 0)
        t0 = pt0.time_from_start.sec + pt0.time_from_start.nanosec * 1e-9
        t1 = pt1.time_from_start.sec + pt1.time_from_start.nanosec * 1e-9
        self.assertGreater(t1, t0)
        self.assertAlmostEqual(t1, duration, places=4)

        self.assertEqual(len(goal.goal_tolerance), 1)
        self.assertEqual(goal.goal_tolerance[0].name, "joint6")
        self.assertAlmostEqual(goal.goal_tolerance[0].position, 0.2)

    def test_02_controller_selected_interpolation_method_and_evidence(self):
        """2. Document and assert the controller's interpolation selection mechanism."""
        goal, _ = build_gripper_trajectory_goal(target_pos=1.0, current_pos=0.0)
        self.assertEqual(len(goal.trajectory.points[0].velocities), 1)
        self.assertEqual(len(goal.trajectory.points[1].velocities), 1)
        self.assertEqual(len(goal.trajectory.points[0].accelerations), 0)
        self.assertEqual(len(goal.trajectory.points[1].accelerations), 0)

    def test_03_correct_use_of_actual_joint6_current_position(self):
        """3. Verify target displacement and initial waypoint derive from actual joint6 position."""
        self.node._teleop_state = STATE_ARMED
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node._latest_joint_positions["joint6"] = 0.35

        res = self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertTrue(res)
        sent_goal = self.node.gripper_action_client.send_goal_async.call_args[0][0]

        pt0 = sent_goal.trajectory.points[0]
        pt1 = sent_goal.trajectory.points[1]
        self.assertAlmostEqual(pt0.positions[0], 0.35)
        self.assertAlmostEqual(pt1.positions[0], 1.0)
        expected_duration = compute_gripper_trajectory_duration(0.65, 10.0, 5.0, 0.5)
        actual_duration = pt1.time_from_start.sec + pt1.time_from_start.nanosec * 1e-9
        self.assertAlmostEqual(actual_duration, expected_duration, places=4)

    def test_04_dynamic_duration_calculation_from_displacement(self):
        """4. Verify dynamic duration formula scales monotonically with displacement."""
        d_small = compute_gripper_trajectory_duration(0.1, 10.0, 5.0, 0.5)
        d_med = compute_gripper_trajectory_duration(0.5, 10.0, 5.0, 0.5)
        d_full = compute_gripper_trajectory_duration(1.0, 10.0, 5.0, 0.5)
        d_large = compute_gripper_trajectory_duration(1.5, 10.0, 5.0, 0.5)

        self.assertAlmostEqual(d_small, 0.5)
        self.assertAlmostEqual(d_med, math.sqrt(6.0 * 0.5 / 5.0), places=4)
        self.assertAlmostEqual(d_full, math.sqrt(6.0 * 1.0 / 5.0), places=4)
        self.assertAlmostEqual(d_large, math.sqrt(6.0 * 1.5 / 5.0), places=4)
        self.assertLess(d_small, d_med)
        self.assertLess(d_med, d_full)
        self.assertLess(d_full, d_large)

    def test_05_velocity_and_acceleration_calculations_for_cubic_interpolation(self):
        """5. Verify rest-to-rest cubic peak velocity and acceleration equations."""
        delta = 1.0
        dur_1s = 1.0
        v_peak_1s = compute_cubic_peak_velocity(delta, dur_1s)
        a_peak_1s = compute_cubic_peak_acceleration(delta, dur_1s)
        self.assertAlmostEqual(v_peak_1s, 1.50)
        self.assertAlmostEqual(a_peak_1s, 6.00)

        dur_safe = math.sqrt(6.0 * 1.0 / 5.0)
        v_peak_safe = compute_cubic_peak_velocity(delta, dur_safe)
        a_peak_safe = compute_cubic_peak_acceleration(delta, dur_safe)
        self.assertAlmostEqual(v_peak_safe, 1.5 * 1.0 / dur_safe)
        self.assertAlmostEqual(a_peak_safe, 5.0000, places=4)

        dur_cons = 1.2
        v_peak_cons = compute_cubic_peak_velocity(delta, dur_cons)
        a_peak_cons = compute_cubic_peak_acceleration(delta, dur_cons)
        self.assertAlmostEqual(v_peak_cons, 1.25)
        self.assertAlmostEqual(a_peak_cons, 6.0 / (1.2 ** 2), places=3)

    def test_06_1rad_stroke_within_software_motion_bounds(self):
        """6. Verify 1.0 rad transition strictly satisfies software velocity (10.0) and acceleration (5.0) bounds."""
        delta = 1.0
        duration = compute_gripper_trajectory_duration(delta, 10.0, 5.0, 0.5)
        self.assertGreaterEqual(duration, 1.0954)

        v_peak = compute_cubic_peak_velocity(delta, duration)
        a_peak = compute_cubic_peak_acceleration(delta, duration)
        self.assertLessEqual(v_peak, 10.0)
        self.assertLessEqual(a_peak, 5.00001)

    def test_07_smaller_displacements_receive_appropriate_duration(self):
        """7. Verify smaller displacement (e.g. 0.5 rad) scales duration proportionally to acceleration bound."""
        delta = 0.5
        duration = compute_gripper_trajectory_duration(delta, 10.0, 5.0, 0.5)
        self.assertAlmostEqual(duration, math.sqrt(6.0 * 0.5 / 5.0), places=4)
        self.assertLess(duration, 1.0)
        a_peak = compute_cubic_peak_acceleration(delta, duration)
        self.assertAlmostEqual(a_peak, 5.0, places=4)

    def test_08_duration_not_falling_below_minimum_floor(self):
        """8. Verify duration floor prevents sub-0.5s execution for arbitrarily small displacements."""
        for tiny_delta in (0.0, 0.001, 0.05, 0.1):
            dur = compute_gripper_trajectory_duration(tiny_delta, 10.0, 5.0, 0.5)
            self.assertGreaterEqual(dur, 0.5)

    def test_09_invalid_velocity_and_acceleration_configuration_rejected(self):
        """9. Verify non-positive or non-finite velocity and acceleration parameters are rejected."""
        with self.assertRaises(ValueError):
            compute_gripper_trajectory_duration(1.0, velocity_limit=-1.0)
        with self.assertRaises(ValueError):
            compute_gripper_trajectory_duration(1.0, velocity_limit=0.0)
        with self.assertRaises(ValueError):
            compute_gripper_trajectory_duration(1.0, velocity_limit=float('nan'))
        with self.assertRaises(ValueError):
            compute_gripper_trajectory_duration(1.0, acceleration_limit=-5.0)
        with self.assertRaises(ValueError):
            compute_gripper_trajectory_duration(1.0, acceleration_limit=0.0)
        with self.assertRaises(ValueError):
            compute_gripper_trajectory_duration(1.0, acceleration_limit=float('inf'))
        with self.assertRaises(ValueError):
            compute_gripper_trajectory_duration(1.0, min_duration=0.0)

        res_v_neg = self.node._on_set_parameters_callback([
            rclpy.parameter.Parameter("gripper_max_velocity", value=-5.0)
        ])
        self.assertFalse(res_v_neg.successful)

        res_a_zero = self.node._on_set_parameters_callback([
            rclpy.parameter.Parameter("gripper_max_acceleration", value=0.0)
        ])
        self.assertFalse(res_a_zero.successful)

        res_d_neg = self.node._on_set_parameters_callback([
            rclpy.parameter.Parameter("gripper_min_duration", value=-0.5)
        ])
        self.assertFalse(res_d_neg.successful)

    def test_10_missing_or_stale_joint6_feedback_prevents_command(self):
        """10. Verify missing, stale, or non-finite feedback prevents gripper goal submission."""
        self.node._teleop_state = STATE_ARMED
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])

        del self.node._latest_joint_positions["joint6"]
        self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))

        self.node._latest_joint_positions["joint6"] = float('nan')
        self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))

        self.node._latest_joint_positions["joint6"] = 0.0
        self.node._last_joint_state_time = time.monotonic() - 1.0
        self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))

    def test_11_out_of_limit_and_nonfinite_targets_rejected(self):
        """11. Verify out-of-limit (< -0.2, > 2.0) or non-finite targets are rejected."""
        self.node._teleop_state = STATE_ARMED
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])

        # 1. Parameter callback rejects out-of-limit values
        res_high = self.node._on_set_parameters_callback([
            rclpy.parameter.Parameter("gripper_open_rad", value=2.5)
        ])
        self.assertFalse(res_high.successful)

        res_low = self.node._on_set_parameters_callback([
            rclpy.parameter.Parameter("gripper_closed_rad", value=-0.5)
        ])
        self.assertFalse(res_low.successful)

        res_nan = self.node._on_set_parameters_callback([
            rclpy.parameter.Parameter("gripper_open_rad", value=float('nan'))
        ])
        self.assertFalse(res_nan.successful)

        # 2. Direct dispatch-level target limit and non-finite rejection
        with patch.object(type(self.node), 'gripper_open_rad', new_callable=PropertyMock) as mock_open:
            mock_open.return_value = 2.5
            self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))

            mock_open.return_value = float('nan')
            self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))

        with patch.object(type(self.node), 'gripper_closed_rad', new_callable=PropertyMock) as mock_closed:
            mock_closed.return_value = -0.5
            self.assertFalse(self.node._send_gripper_command("GRIPPER_CLOSED"))

            mock_closed.return_value = float('inf')
            self.assertFalse(self.node._send_gripper_command("GRIPPER_CLOSED"))

        # 3. Trajectory builder rejection
        with self.assertRaises(ValueError):
            build_gripper_trajectory_goal(target_pos=float('nan'), current_pos=0.0)
        with self.assertRaises(ValueError):
            build_gripper_trajectory_goal(target_pos=1.0, current_pos=float('inf'))

    def test_12_absence_of_overlapping_and_unresolved_cancelling_goals(self):
        """12. Verify active goal or unresolved cancellation prevents duplicate or overlapping dispatches."""
        self.node._teleop_state = STATE_ARMED
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])

        self.node._gripper_goal_active = True
        self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))
        self.node._gripper_goal_active = False

        self.node._gripper_canceling = True
        self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))
        self.node._gripper_canceling = False

    def test_13_failures_rejections_and_cancels_do_not_trigger_auto_retry(self):
        """13. Verify fail-closed terminal transitions never schedule or execute automatic retries."""
        self.node._teleop_state = STATE_ARMED
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])

        self.node.gripper_action_client.send_goal_async.side_effect = RuntimeError("Transport disconnected")
        self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))
        self.assertFalse(self.node._gripper_goal_active)
        self.assertEqual(self.node._gripper_dispatch_count, 1)

        self.node.gripper_action_client.send_goal_async.side_effect = None
        self.node.gripper_action_client.send_goal_async.return_value = self.mock_future

        self.node._last_commanded_gripper_state = None
        self.assertTrue(self.node._send_gripper_command("GRIPPER_OPEN"))
        mock_future = MagicMock()
        mock_handle = MagicMock()
        mock_handle.accepted = False
        mock_future.result.return_value = mock_handle
        self.node._gripper_goal_response_callback(mock_future)
        self.assertFalse(self.node._gripper_goal_active)
        self.assertEqual(self.node._gripper_last_status, "REJECTED")
        self.assertEqual(self.node._gripper_dispatch_count, 2)

        self.node._last_commanded_gripper_state = None
        self.assertTrue(self.node._send_gripper_command("GRIPPER_OPEN"))
        mock_future_res = MagicMock()
        mock_wrap = MagicMock()
        mock_wrap.status = GoalStatus.STATUS_ABORTED
        mock_future_res.result.return_value = mock_wrap
        self.node._gripper_goal_result_callback(mock_future_res)
        self.assertFalse(self.node._gripper_goal_active)
        self.assertEqual(self.node._gripper_last_status, f"FINISHED_STATUS_{GoalStatus.STATUS_ABORTED}")
        self.assertEqual(self.node._gripper_dispatch_count, 3)

    def test_14_existing_arm_single_goal_and_bounded_session_behavior_preserved(self):
        """14. Verify arm single-goal, session quota, and continuous modes remain fully intact."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("single_goal_test_mode", value=True),
            rclpy.parameter.Parameter("max_goals_per_session", value=5),
        ])
        self.assertEqual(self.node.effective_session_goal_limit, 1)

        self.node.set_parameters([
            rclpy.parameter.Parameter("single_goal_test_mode", value=False),
            rclpy.parameter.Parameter("max_goals_per_session", value=2),
        ])
        self.assertEqual(self.node.effective_session_goal_limit, 2)

        self.node.set_parameters([
            rclpy.parameter.Parameter("single_goal_test_mode", value=False),
            rclpy.parameter.Parameter("max_goals_per_session", value=0),
        ])
        self.assertEqual(self.node.effective_session_goal_limit, 0)
        self.assertFalse(self.node.is_session_goal_limit_reached())

    def test_15_existing_two_finger_gesture_and_compact_ui_preserved(self):
        """15. Verify two-finger gesture classifier and compact HUD rendering remain regression-free."""
        class MockLM:
            def __init__(self, x=0.0, y=0.0, z=0.0):
                self.x = x
                self.y = y
                self.z = z

        landmarks = [MockLM(0.0, 0.0, 0.0) for _ in range(21)]
        landmarks[0] = MockLM(0.0, 0.0, 0.0)
        landmarks[9] = MockLM(0.0, 1.0, 0.0)

        with patch("hand_gesture_teleop.hand_tracking.classify_thumb") as mock_thumb, \
             patch("hand_gesture_teleop.hand_tracking.classify_single_finger") as mock_single:
            # Thumb EXT + Index EXT -> GRIPPER_OPEN (middle ignored)
            mock_thumb.return_value = "EXTENDED"
            mock_single.side_effect = ["EXTENDED", "FOLDED"]
            _, gesture = classify_hand_gestures(landmarks)
            self.assertEqual(gesture, "GRIPPER_OPEN")

            # Thumb FOLDED + Index FOLDED -> GRIPPER_CLOSED (middle ignored)
            mock_thumb.return_value = "FOLDED"
            mock_single.side_effect = ["FOLDED", "EXTENDED"]
            _, gesture = classify_hand_gestures(landmarks)
            self.assertEqual(gesture, "GRIPPER_CLOSED")

            # Thumb EXTENDED + Index FOLDED -> UNKNOWN
            mock_thumb.return_value = "EXTENDED"
            mock_single.side_effect = ["FOLDED", "FOLDED"]
            _, gesture = classify_hand_gestures(landmarks)
            self.assertEqual(gesture, "UNKNOWN")

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        card_bbox = draw_compact_hud(
            frame=frame,
            hand_detected=True,
            gesture_state="GRIPPER_OPEN",
            gripper_state_text="OPEN",
            control_state_text="ARMED",
            is_motion_enabled=True,
        )
        self.assertEqual(len(card_bbox), 4)
        self.assertGreater(card_bbox[2], card_bbox[0])
        self.assertGreater(card_bbox[3], card_bbox[1])

    def test_16_arm_trajectory_goals_and_session_counter_unaffected_by_gripper(self):
        """16. Verify gripper goal dispatches have zero influence on arm goal quotas or arm client."""
        self.node._teleop_state = STATE_ARMED
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("max_goals_per_session", value=2),
        ])
        self.node.arm_action_client.send_goal_async = MagicMock()

        res1 = self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertTrue(res1)
        self.node._gripper_goal_active = False

        res2 = self.node._send_gripper_command("GRIPPER_CLOSED")
        self.assertTrue(res2)
        self.node._gripper_goal_active = False

        self.assertEqual(self.node._gripper_dispatch_count, 2)
        self.assertEqual(self.node._trajectory_dispatch_count, 0)
        self.assertEqual(self.node.session_goal_dispatch_count, 0)
        self.node.arm_action_client.send_goal_async.assert_not_called()



class TestGripperStartStateAndTiming(unittest.TestCase):
    """
    Focused unit tests for joint6 velocity feedback and trajectory start timing (Task 5C-3D-GRIPPER-START-STATE-FIX).
    Verifies:
      1. Missing joint6 velocity feedback blocks gripper dispatch.
      2. An empty JointState velocity array cannot be interpreted as zero.
      3. A JointState message that omits joint6 velocity cannot leave stale velocity data available.
      4. NaN velocity blocks gripper dispatch.
      5. Positive or negative infinite velocity blocks gripper dispatch.
      6. Stale feedback blocks gripper dispatch.
      7. Valid current position and velocity data are mapped by joint name.
      8. A near-zero valid velocity satisfies the existing rest check.
      9. A velocity above the allowed rest threshold blocks dispatch.
      10. The first trajectory waypoint uses the validated current position.
      11. The first waypoint velocity matches the established starting-state policy.
      12. The target waypoint has the correct target position and zero ending velocity.
      13. Both waypoints have the correct and ordered time_from_start values.
      14. Header timestamp follows the verified installed-controller start-time semantics.
      15. Zero header timestamp and first waypoint at t=0 are covered by tests.
      16. Dynamic duration calculations continue to satisfy the declared interpolation assumptions.
      17. Existing gripper cancellation and no-overlap guards continue to work.
      18. Existing two-finger gesture tests pass.
      19. Existing arm safety and bounded session-goal tests pass unchanged.
    """

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = HandTeleopNode("test_gripper_start_state_node")
        self.node._teleop_state = STATE_ARMED
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node._hardware_mock_verified = True
        self.node._hardware_last_verified_time = time.monotonic()
        self.node.arm_action_client.server_is_ready = MagicMock(return_value=True)
        self.node.gripper_action_client.server_is_ready = MagicMock(return_value=True)
        self.mock_future = MagicMock()
        self.node.gripper_action_client.send_goal_async = MagicMock(return_value=self.mock_future)

    def tearDown(self):
        self.node.destroy_node()

    def test_01_missing_joint6_velocity_feedback_blocks_gripper_dispatch(self):
        """1. Missing joint6 velocity feedback blocks gripper dispatch."""
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._latest_joint_velocities = {j: 0.0 for j in EXPECTED_SO101_JOINTS if j != "joint6"}
        self.node._last_joint_state_time = time.monotonic()

        res = self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertFalse(res)
        self.assertEqual(self.node._gripper_dispatch_count, 0)
        self.node.gripper_action_client.send_goal_async.assert_not_called()

    def test_02_empty_joint_state_velocity_array_cannot_be_interpreted_as_zero(self):
        """2. An incoming JointState message with an empty velocity array cannot be treated as zero."""
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(EXPECTED_SO101_JOINTS)
        msg.position = [0.0] * len(EXPECTED_SO101_JOINTS)
        msg.velocity = []

        self.node._joint_state_callback(msg)
        self.assertNotIn("joint6", self.node._latest_joint_velocities)
        self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))
        self.assertEqual(self.node._gripper_dispatch_count, 0)

    def test_03_joint_state_omitting_velocity_clears_stale_data(self):
        """3. A JointState message that omits joint6 velocity cannot leave stale velocity data available."""
        # Step 1: Arrive with valid velocity
        msg1 = JointState()
        msg1.header.stamp = self.node.get_clock().now().to_msg()
        msg1.name = list(EXPECTED_SO101_JOINTS)
        msg1.position = [0.0] * len(EXPECTED_SO101_JOINTS)
        msg1.velocity = [0.0] * len(EXPECTED_SO101_JOINTS)
        self.node._joint_state_callback(msg1)
        self.assertIn("joint6", self.node._latest_joint_velocities)
        self.assertEqual(self.node._latest_joint_velocities["joint6"], 0.0)

        # Step 2: Arrive with empty velocity array
        msg2 = JointState()
        msg2.header.stamp = self.node.get_clock().now().to_msg()
        msg2.name = list(EXPECTED_SO101_JOINTS)
        msg2.position = [0.0] * len(EXPECTED_SO101_JOINTS)
        msg2.velocity = []
        self.node._joint_state_callback(msg2)
        self.assertNotIn("joint6", self.node._latest_joint_velocities)

        # Step 3: Gripper command must fail closed
        res = self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertFalse(res)
        self.assertEqual(self.node._gripper_dispatch_count, 0)

    def test_04_nan_velocity_blocks_gripper_dispatch(self):
        """4. NaN velocity value in JointState blocks gripper dispatch."""
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(EXPECTED_SO101_JOINTS)
        msg.position = [0.0] * len(EXPECTED_SO101_JOINTS)
        msg.velocity = [0.0 if j != "joint6" else float('nan') for j in EXPECTED_SO101_JOINTS]

        self.node._joint_state_callback(msg)
        self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))
        self.assertEqual(self.node._gripper_dispatch_count, 0)

    def test_05_infinite_velocity_blocks_gripper_dispatch(self):
        """5. Positive or negative infinite velocity blocks gripper dispatch."""
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._last_joint_state_time = time.monotonic()

        # Positive infinity
        self.node._latest_joint_velocities = {"joint6": float('inf')}
        self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))

        # Negative infinity
        self.node._latest_joint_velocities = {"joint6": float('-inf')}
        self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))
        self.assertEqual(self.node._gripper_dispatch_count, 0)

    def test_06_stale_feedback_blocks_gripper_dispatch(self):
        """6. Stale joint feedback (> 0.25s) blocks gripper dispatch."""
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._latest_joint_velocities = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._last_joint_state_time = time.monotonic() - 1.0  # 1.0s > 0.25s

        self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))
        self.assertEqual(self.node._gripper_dispatch_count, 0)

    def test_07_valid_position_and_velocity_mapped_by_joint_name(self):
        """7. Valid current position and velocity data are correctly mapped by joint name."""
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = ["joint4", "joint6", "joint2", "joint1", "joint5", "joint3"]
        msg.position = [0.1, 0.42, 0.2, 0.3, 0.5, 0.6]
        msg.velocity = [0.01, 0.03, 0.02, 0.04, 0.05, 0.06]

        self.node._joint_state_callback(msg)
        self.assertAlmostEqual(self.node._latest_joint_positions["joint6"], 0.42)
        self.assertAlmostEqual(self.node._latest_joint_velocities["joint6"], 0.03)

    def test_08_near_zero_valid_velocity_satisfies_rest_check(self):
        """8. A near-zero valid velocity (|v| <= 0.1 rad/s) satisfies the rest check."""
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._latest_joint_velocities = {j: 0.05 for j in EXPECTED_SO101_JOINTS}
        self.node._last_joint_state_time = time.monotonic()

        res = self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertTrue(res)
        self.assertEqual(self.node._gripper_dispatch_count, 1)

    def test_09_velocity_above_rest_threshold_blocks_dispatch(self):
        """9. A velocity above the allowed rest threshold (|v| > 0.1 rad/s) blocks dispatch."""
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._last_joint_state_time = time.monotonic()

        # Positive violation
        self.node._latest_joint_velocities = {"joint6": 0.12}
        self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))

        # Negative violation
        self.node._latest_joint_velocities = {"joint6": -0.15}
        self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))
        self.assertEqual(self.node._gripper_dispatch_count, 0)

    def test_10_first_trajectory_waypoint_uses_validated_current_position(self):
        """10. The first trajectory waypoint uses the exact validated current joint6 position."""
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._latest_joint_positions["joint6"] = 0.28
        self.node._latest_joint_velocities = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._last_joint_state_time = time.monotonic()

        res = self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertTrue(res)
        sent_goal = self.node.gripper_action_client.send_goal_async.call_args[0][0]
        pt0 = sent_goal.trajectory.points[0]
        self.assertAlmostEqual(pt0.positions[0], 0.28)

    def test_11_first_waypoint_velocity_matches_starting_state_policy(self):
        """11. The first waypoint velocity matches the established starting-state policy."""
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._latest_joint_velocities = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._latest_joint_velocities["joint6"] = 0.04
        self.node._last_joint_state_time = time.monotonic()

        res = self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertTrue(res)
        sent_goal = self.node.gripper_action_client.send_goal_async.call_args[0][0]
        pt0 = sent_goal.trajectory.points[0]
        self.assertAlmostEqual(pt0.velocities[0], 0.04)

    def test_12_target_waypoint_has_correct_position_and_zero_ending_velocity(self):
        """12. The target waypoint has the correct target position and zero ending velocity."""
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._latest_joint_velocities = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._last_joint_state_time = time.monotonic()

        res = self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertTrue(res)
        sent_goal = self.node.gripper_action_client.send_goal_async.call_args[0][0]
        pt1 = sent_goal.trajectory.points[1]
        self.assertAlmostEqual(pt1.positions[0], 1.0)
        self.assertEqual(pt1.velocities[0], 0.0)

    def test_13_both_waypoints_have_correct_and_ordered_time_from_start(self):
        """13. Both waypoints have valid and monotonically ordered time_from_start values."""
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._latest_joint_velocities = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._last_joint_state_time = time.monotonic()

        res = self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertTrue(res)
        sent_goal = self.node.gripper_action_client.send_goal_async.call_args[0][0]
        pt0 = sent_goal.trajectory.points[0]
        pt1 = sent_goal.trajectory.points[1]
        self.assertEqual(pt0.time_from_start.sec, 0)
        self.assertEqual(pt0.time_from_start.nanosec, 0)
        t1 = pt1.time_from_start.sec + pt1.time_from_start.nanosec * 1e-9
        self.assertGreater(t1, 0.0)

    def test_14_header_timestamp_follows_installed_controller_start_semantics(self):
        """14. Header timestamp uses zero stamp (sec=0, nanosec=0) for 'start now' controller semantics."""
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._latest_joint_velocities = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._last_joint_state_time = time.monotonic()

        res = self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertTrue(res)
        sent_goal = self.node.gripper_action_client.send_goal_async.call_args[0][0]
        self.assertEqual(sent_goal.trajectory.header.stamp.sec, 0)
        self.assertEqual(sent_goal.trajectory.header.stamp.nanosec, 0)

    def test_15_zero_header_timestamp_and_first_waypoint_at_t0_covered(self):
        """15. Explicitly assert zero header timestamp and Point 0 at t=0 on built goal."""
        goal, _ = build_gripper_trajectory_goal(target_pos=1.0, current_pos=0.0)
        self.assertEqual(goal.trajectory.header.stamp.sec, 0)
        self.assertEqual(goal.trajectory.header.stamp.nanosec, 0)
        self.assertEqual(goal.trajectory.points[0].time_from_start.sec, 0)
        self.assertEqual(goal.trajectory.points[0].time_from_start.nanosec, 0)

    def test_16_dynamic_duration_calculations_satisfy_assumptions(self):
        """16. Dynamic duration calculation accounts for starting velocity and satisfies limits."""
        # Rest-to-rest
        d_rest = compute_gripper_trajectory_duration(1.0, 10.0, 5.0, 0.5, starting_velocity=0.0)
        self.assertAlmostEqual(d_rest, math.sqrt(6.0 / 5.0), places=4)

        # With nonzero starting velocity: duration scales to maintain peak acceleration bound
        d_nonzero = compute_gripper_trajectory_duration(1.0, 10.0, 5.0, 0.5, starting_velocity=0.08)
        self.assertGreater(d_nonzero, d_rest)

        # Non-finite starting velocity raises ValueError
        with self.assertRaises(ValueError):
            compute_gripper_trajectory_duration(1.0, 10.0, 5.0, 0.5, starting_velocity=float('nan'))

    def test_17_existing_gripper_cancellation_and_no_overlap_guards(self):
        """17. Existing gripper cancellation and no-overlap guards continue to block concurrent dispatches."""
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._latest_joint_velocities = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._last_joint_state_time = time.monotonic()

        self.node._gripper_goal_active = True
        self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))
        self.node._gripper_goal_active = False

        self.node._gripper_canceling = True
        self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))
        self.node._gripper_canceling = False

    def test_18_existing_two_finger_gesture_tests_pass(self):
        """18. Existing two-finger gesture classifier functions properly without middle finger."""
        class MockLM:
            def __init__(self, x=0.0, y=0.0, z=0.0):
                self.x = x
                self.y = y
                self.z = z

        landmarks = [MockLM(0.0, 0.0, 0.0) for _ in range(21)]
        landmarks[0] = MockLM(0.0, 0.0, 0.0)
        landmarks[9] = MockLM(0.0, 1.0, 0.0)

        with patch("hand_gesture_teleop.hand_tracking.classify_thumb") as mock_thumb, \
             patch("hand_gesture_teleop.hand_tracking.classify_single_finger") as mock_single:
            # Thumb EXT + Index EXT -> GRIPPER_OPEN (middle finger ignored)
            mock_thumb.return_value = "EXTENDED"
            mock_single.side_effect = ["EXTENDED", "FOLDED"]
            _, gesture = classify_hand_gestures(landmarks)
            self.assertEqual(gesture, "GRIPPER_OPEN")

            # Thumb FOLDED + Index FOLDED -> GRIPPER_CLOSED (middle finger ignored)
            mock_thumb.return_value = "FOLDED"
            mock_single.side_effect = ["FOLDED", "EXTENDED"]
            _, gesture = classify_hand_gestures(landmarks)
            self.assertEqual(gesture, "GRIPPER_CLOSED")

    def test_19_existing_arm_safety_and_bounded_session_goals_pass_unchanged(self):
        """19. Existing arm safety, mode conflict, and bounded session quota behavior are unaffected."""
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._latest_joint_velocities = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._last_joint_state_time = time.monotonic()

        # Mode conflict check
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_ik_dry_run", value=True),
        ])
        self.assertTrue(self.node.modes_conflict)
        self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))


class TestMockGripperDynamicVelocityConfig(unittest.TestCase):
    """
    Focused unit & configuration regression tests for Task 5C-3D-SIM-DYNAMICS-CONFIG.
    Covers:
      1. Simulation hardware configuration enables calculate_dynamics=true.
      2. Joint6 exposes position and velocity state interfaces (position command interface preserved).
      3. Gripper controller requests both position and velocity state interfaces in ros2_control_sim.yaml.
      4. Missing, empty, and non-finite velocity data blocks gripper dispatch.
      5. Finite velocity above the rest threshold (0.1 rad/s) blocks dispatch.
      6. Finite near-zero velocity (<= 0.1 rad/s) permits eligibility checks to continue.
      7. Existing gripper duration and interpolation calculations remain verified and passing.
      8. Existing two-finger gesture and arm safety invariants remain verified and passing.
    """

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = HandTeleopNode("test_mock_gripper_dyn_cfg_node")
        self.node._teleop_state = STATE_ARMED
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node._hardware_mock_verified = True
        self.node._hardware_last_verified_time = time.monotonic()
        self.node.arm_action_client.server_is_ready = MagicMock(return_value=True)
        self.node.gripper_action_client.server_is_ready = MagicMock(return_value=True)
        self.mock_future = MagicMock()
        self.node.gripper_action_client.send_goal_async = MagicMock(return_value=self.mock_future)

    def tearDown(self):
        self.node.destroy_node()

    def test_01_simulation_hardware_enables_calculate_dynamics(self):
        """1. Simulation hardware configuration enables verified calculate_dynamics=true option."""
        import xml.etree.ElementTree as ET
        urdf_path = "/home/santosh/lerobot_ws/src/hand_gesture_teleop/urdf/so101_sim.urdf.xacro"
        self.assertTrue(os.path.exists(urdf_path), f"URDF not found: {urdf_path}")
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # Find ros2_control SimSystem
        sim_system = None
        for r2c in root.findall("ros2_control"):
            if r2c.attrib.get("name") == "SimSystem":
                sim_system = r2c
                break
        self.assertIsNotNone(sim_system, "SimSystem ros2_control tag not found in so101_sim.urdf.xacro")

        hardware = sim_system.find("hardware")
        self.assertIsNotNone(hardware, "hardware tag not found under SimSystem")
        plugin = hardware.find("plugin")
        self.assertIsNotNone(plugin, "plugin tag not found under hardware")
        self.assertEqual(plugin.text.strip(), "mock_components/GenericSystem")

        # Find calculate_dynamics param
        calc_dyn = None
        for p in hardware.findall("param"):
            if p.attrib.get("name") == "calculate_dynamics":
                calc_dyn = p.text.strip()
                break
        self.assertEqual(calc_dyn, "true", "calculate_dynamics must be set to 'true' in SimSystem hardware")

    def test_02_joint6_exposes_position_and_velocity_state_interfaces(self):
        """2. Joint6 exposes position and velocity state interfaces, preserving position command interface."""
        import xml.etree.ElementTree as ET
        urdf_path = "/home/santosh/lerobot_ws/src/hand_gesture_teleop/urdf/so101_sim.urdf.xacro"
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        sim_system = None
        for r2c in root.findall("ros2_control"):
            if r2c.attrib.get("name") == "SimSystem":
                sim_system = r2c
                break
        self.assertIsNotNone(sim_system)

        joint6 = None
        for j in sim_system.findall("joint"):
            if j.attrib.get("name") == "joint6":
                joint6 = j
                break
        self.assertIsNotNone(joint6, "joint6 not found in SimSystem")

        # Check command interfaces
        cmd_interfaces = [c.attrib.get("name") for c in joint6.findall("command_interface")]
        self.assertIn("position", cmd_interfaces)
        self.assertNotIn("velocity", cmd_interfaces, "Command interface must remain position control")

        # Check state interfaces
        state_interfaces = {s.attrib.get("name"): s for s in joint6.findall("state_interface")}
        self.assertIn("position", state_interfaces)
        self.assertIn("velocity", state_interfaces)

        # Velocity initial value
        vel_init = None
        for p in state_interfaces["velocity"].findall("param"):
            if p.attrib.get("name") == "initial_value":
                vel_init = p.text.strip()
        self.assertEqual(vel_init, "0.0")

    def test_03_gripper_controller_requests_position_and_velocity_state_interfaces(self):
        """3. Gripper controller requests both position and velocity state interfaces in ros2_control_sim.yaml."""
        import yaml
        yaml_path = "/home/santosh/lerobot_ws/src/hand_gesture_teleop/config/ros2_control_sim.yaml"
        self.assertTrue(os.path.exists(yaml_path), f"YAML not found: {yaml_path}")
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.assertIn("gripper_controller", cfg)
        gripper_params = cfg["gripper_controller"]["ros__parameters"]
        self.assertEqual(gripper_params["joints"], ["joint6"])
        self.assertEqual(gripper_params["command_interfaces"], ["position"])
        self.assertIn("position", gripper_params["state_interfaces"])
        self.assertIn("velocity", gripper_params["state_interfaces"])

    def test_04_missing_empty_and_non_finite_velocity_blocks_gripper_dispatch(self):
        """4. Missing, empty and non-finite velocity data still blocks gripper dispatch."""
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._last_joint_state_time = time.monotonic()

        # 4a. Missing joint6
        self.node._latest_joint_velocities = {j: 0.0 for j in EXPECTED_SO101_JOINTS if j != "joint6"}
        self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))
        self.assertEqual(self.node._gripper_dispatch_count, 0)

        # 4b. Empty JointState velocity array
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = list(EXPECTED_SO101_JOINTS)
        msg.position = [0.0] * len(EXPECTED_SO101_JOINTS)
        msg.velocity = []
        self.node._joint_state_callback(msg)
        self.assertNotIn("joint6", self.node._latest_joint_velocities)
        self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))
        self.assertEqual(self.node._gripper_dispatch_count, 0)

        # 4c. NaN velocity
        self.node._latest_joint_velocities = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._latest_joint_velocities["joint6"] = float("nan")
        self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))
        self.assertEqual(self.node._gripper_dispatch_count, 0)

        # 4d. Positive infinity
        self.node._latest_joint_velocities["joint6"] = float("inf")
        self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))
        self.assertEqual(self.node._gripper_dispatch_count, 0)

        # 4e. Negative infinity
        self.node._latest_joint_velocities["joint6"] = float("-inf")
        self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))
        self.assertEqual(self.node._gripper_dispatch_count, 0)

    def test_05_finite_velocity_above_rest_threshold_blocks_dispatch(self):
        """5. A finite velocity above the existing rest threshold (0.10 rad/s) blocks dispatch."""
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._last_joint_state_time = time.monotonic()

        for vel in [0.101, 0.15, 0.50, -0.101, -0.25]:
            self.node._latest_joint_velocities = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
            self.node._latest_joint_velocities["joint6"] = vel
            res = self.node._send_gripper_command("GRIPPER_OPEN")
            self.assertFalse(res, f"Expected dispatch blocked for vel={vel}")
            self.assertEqual(self.node._gripper_dispatch_count, 0)

    def test_06_finite_near_zero_velocity_allows_eligibility_to_continue(self):
        """6. A finite near-zero velocity permits the existing eligibility checks to continue."""
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._last_joint_state_time = time.monotonic()

        # Zero velocity
        self.node._latest_joint_velocities = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.assertTrue(self.node._send_gripper_command("GRIPPER_OPEN"))
        self.assertEqual(self.node._gripper_dispatch_count, 1)

        # Finish first goal
        self.node._gripper_goal_active = False

        # Near-zero positive velocity <= 0.1
        self.node._latest_joint_positions["joint6"] = 1.0  # current open
        self.node._latest_joint_velocities["joint6"] = 0.05
        self.assertTrue(self.node._send_gripper_command("GRIPPER_CLOSED"))
        self.assertEqual(self.node._gripper_dispatch_count, 2)

        # Finish second goal
        self.node._gripper_goal_active = False

        # Near-zero negative velocity >= -0.1
        self.node._latest_joint_positions["joint6"] = 0.0  # current closed
        self.node._latest_joint_velocities["joint6"] = -0.08
        self.assertTrue(self.node._send_gripper_command("GRIPPER_OPEN"))
        self.assertEqual(self.node._gripper_dispatch_count, 3)

    def test_07_existing_gripper_duration_interpolation_preserved(self):
        """7. Existing gripper duration/interpolation tests remain unchanged and passing."""
        # Rest-to-rest duration
        dur_rest = compute_gripper_trajectory_duration(1.0, 10.0, 5.0, 0.5, starting_velocity=0.0)
        self.assertAlmostEqual(dur_rest, math.sqrt(6.0 * 1.0 / 5.0), places=3)

        # Nonzero starting velocity duration scaling
        dur_moving = compute_gripper_trajectory_duration(1.0, 10.0, 5.0, 0.5, starting_velocity=0.05)
        self.assertGreater(dur_moving, dur_rest)

        # Waypoint structure
        goal, d = build_gripper_trajectory_goal(
            target_pos=1.0,
            current_pos=0.0,
            starting_velocity=0.05,
            velocity_limit=10.0,
            acceleration_limit=5.0,
            min_duration=0.5,
        )
        self.assertEqual(len(goal.trajectory.points), 2)
        self.assertEqual(goal.trajectory.header.stamp.sec, 0)
        self.assertEqual(goal.trajectory.header.stamp.nanosec, 0)
        self.assertEqual(goal.trajectory.points[0].time_from_start.sec, 0)
        self.assertEqual(goal.trajectory.points[0].time_from_start.nanosec, 0)
        self.assertAlmostEqual(goal.trajectory.points[0].positions[0], 0.0)
        self.assertAlmostEqual(goal.trajectory.points[0].velocities[0], 0.05)
        self.assertAlmostEqual(goal.trajectory.points[1].positions[0], 1.0)
        self.assertAlmostEqual(goal.trajectory.points[1].velocities[0], 0.0)

    def test_08_existing_gesture_and_arm_safety_invariants_pass(self):
        """8. Existing two-finger, arm-safety, cancellation and bounded-session tests pass."""
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._latest_joint_velocities = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._last_joint_state_time = time.monotonic()

        # Arm safety: mode conflict
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_ik_dry_run", value=True),
        ])
        self.assertTrue(self.node.modes_conflict)
        self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))
        self.node.set_parameters([rclpy.parameter.Parameter("enable_ik_dry_run", value=False)])

        # Max goals per session limit
        self.node._gripper_dispatch_count = 10
        self.assertFalse(self.node._send_gripper_command("GRIPPER_OPEN"))


class TestArmMotionLockout(unittest.TestCase):
    """
    Unit & integration tests for Task 5C-3D-ARM-LOCKOUT-IMPLEMENT:
    Enforcing a complete, fail-closed arm-motion lockout while preserving gripper operation.
    """

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = HandTeleopNode("test_arm_lockout_node")
        self.node._latest_joint_positions = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._latest_joint_velocities = {j: 0.0 for j in EXPECTED_SO101_JOINTS}
        self.node._last_joint_state_time = time.monotonic()
        self.node._hardware_mock_verified = True
        self.node._hardware_last_verified_time = time.monotonic()
        self.node.arm_action_client.server_is_ready = MagicMock(return_value=True)
        self.node.gripper_action_client.server_is_ready = MagicMock(return_value=True)
        self.mock_arm_future = MagicMock()
        self.mock_gripper_future = MagicMock()
        self.node.arm_action_client.send_goal_async = MagicMock(return_value=self.mock_arm_future)
        self.node.gripper_action_client.send_goal_async = MagicMock(return_value=self.mock_gripper_future)
        self.node.ik_client.call_async = MagicMock()

    def tearDown(self):
        self.node.destroy_node()

    def test_01_default_parameter_enable_arm_motion_false_fail_closed(self):
        """1. Default parameter behaviour: enable_arm_motion defaults to False (fail-closed)."""
        self.assertFalse(self.node.enable_arm_motion)
        self.assertFalse(self.node.get_parameter("enable_arm_motion").value)

    def test_01b_default_startup_without_override_rejects_arm_ik_and_dispatch(self):
        """1b. Startup without explicit enable_arm_motion override keeps arm motion locked out."""
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        self.node._teleop_state = STATE_ARMED
        self.assertFalse(self.node.can_issue_ik_requests())
        self.assertFalse(self.node.request_ik(force=True))
        candidate = {j: 0.05 for j in ARM_CONTROL_JOINTS}
        self.assertFalse(self.node.dispatch_arm_trajectory(candidate))
        self.assertEqual(self.node._trajectory_last_status, "ARM_MOTION_DISABLED")
        self.node.arm_action_client.send_goal_async.assert_not_called()

    def test_01c_explicit_opt_in_enable_arm_motion_true_permits_arm_dispatch(self):
        """1c. Explicit opt-in via enable_arm_motion=True permits authorized arm trajectory dispatch."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_arm_motion", value=True),
        ])
        self.node._teleop_state = STATE_ARMED
        candidate = {j: 0.05 for j in ARM_CONTROL_JOINTS}
        self.assertTrue(self.node.dispatch_arm_trajectory(candidate))
        self.assertEqual(self.node._trajectory_last_status, "SENDING")
        self.node.arm_action_client.send_goal_async.assert_called_once()

    def test_02_lockout_prevents_new_arm_ik_requests(self):
        """2. Lockout prevents new arm IK requests."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_arm_motion", value=False),
        ])
        self.node._teleop_state = STATE_ARMED
        # can_issue_ik_requests must refuse
        self.assertFalse(self.node.can_issue_ik_requests())
        # request_ik must refuse and never call service
        res = self.node.request_ik(force=True)
        self.assertFalse(res)
        self.node.ik_client.call_async.assert_not_called()

        # In dry run mode as well
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=False),
            rclpy.parameter.Parameter("enable_ik_dry_run", value=True),
        ])
        self.assertFalse(self.node.can_issue_ik_requests())
        self.assertFalse(self.node.request_ik(force=True))
        self.node.ik_client.call_async.assert_not_called()

    def test_03_direct_dispatch_arm_trajectory_rejected(self):
        """3. A direct attempt to invoke dispatch_arm_trajectory() while arm motion is disabled cannot dispatch a goal."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_arm_motion", value=False),
        ])
        self.node._teleop_state = STATE_ARMED
        candidate = {j: 0.05 for j in ARM_CONTROL_JOINTS}
        res = self.node.dispatch_arm_trajectory(candidate)
        self.assertFalse(res)
        self.assertEqual(self.node._trajectory_last_status, "ARM_MOTION_DISABLED")
        self.node.arm_action_client.send_goal_async.assert_not_called()
        self.assertEqual(self.node._trajectory_dispatch_count, 0)

    def test_04_asynchronous_ik_response_dropped_without_motion(self):
        """4. A successful but stale/pending asynchronous IK response cannot bypass the lockout and submit an arm goal."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_arm_motion", value=False),
        ])
        self.node._teleop_state = STATE_ARMED

        mock_future = MagicMock()
        mock_resp = GetPositionIK.Response()
        mock_resp.error_code.val = MoveItErrorCodes.SUCCESS
        mock_resp.solution.joint_state.name = list(ARM_CONTROL_JOINTS)
        mock_resp.solution.joint_state.position = [0.05, 0.0, 0.0, 0.0, 0.0]
        mock_future.result.return_value = mock_resp

        # Invoke response callback directly
        self.node._ik_response_callback(mock_future)
        self.node.arm_action_client.send_goal_async.assert_not_called()
        self.assertIsNone(self.node._pending_ik_candidate)
        self.assertFalse(self.node._goal_active)

    def test_05_no_arm_action_send_goal_async_called_under_lockout(self):
        """5. No arm action client's send_goal_async() call occurs while the lockout is disabled."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_arm_motion", value=False),
        ])
        self.node._teleop_state = STATE_ARMED

        # Multiple dispatch attempts through different routes
        self.node.dispatch_arm_trajectory({j: 0.02 for j in ARM_CONTROL_JOINTS})
        self.node._handle_valid_ik_solution({j: 0.02 for j in ARM_CONTROL_JOINTS})
        # Buffered completion route
        self.node._pending_ik_candidate = {j: 0.02 for j in ARM_CONTROL_JOINTS}
        mock_res_future = MagicMock()
        mock_res = MagicMock()
        mock_res.status = GoalStatus.STATUS_SUCCEEDED
        mock_res_future.result.return_value = mock_res
        self.node._goal_result_callback(mock_res_future)

        self.node.arm_action_client.send_goal_async.assert_not_called()
        self.assertEqual(self.node._trajectory_dispatch_count, 0)

    def test_06_gripper_dispatch_permitted_under_arm_lockout(self):
        """6. Gripper dispatch remains permitted by the new parameter when all existing gripper prerequisites are satisfied."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_arm_motion", value=False),
        ])
        self.node._teleop_state = STATE_ARMED
        self.node._latest_joint_positions["joint6"] = 0.0
        self.node._latest_joint_velocities["joint6"] = 0.0

        res = self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertTrue(res)
        self.assertEqual(self.node._gripper_dispatch_count, 1)
        self.node.gripper_action_client.send_goal_async.assert_called_once()
        self.node.arm_action_client.send_goal_async.assert_not_called()

    def test_07_arming_with_arm_motion_disabled_requires_only_gripper_server(self):
        """7. With enable_arm_motion=false, arming does not fail solely due to arm action-server unavailability, but gripper server readiness is enforced."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_arm_motion", value=False),
        ])
        self.node._teleop_state = STATE_ARMING_PENDING
        self.node._arming_requested_time = time.monotonic()
        self.node.arm_action_client.server_is_ready = MagicMock(return_value=False)
        self.node.gripper_action_client.server_is_ready = MagicMock(return_value=True)

        # Handshake confirmation
        ok, reason = self.node.confirm_arming(hand_detected=True, gesture_state="GRIPPER_OPEN")
        self.assertTrue(ok)
        self.assertEqual(reason, "ARMED")
        self.assertEqual(self.node.teleop_state, STATE_ARMED)

        # If gripper action server is NOT ready, arming must fail-closed
        self.node._teleop_state = STATE_ARMING_PENDING
        self.node._arming_requested_time = time.monotonic()
        self.node.gripper_action_client.server_is_ready = MagicMock(return_value=False)
        ok, reason = self.node.confirm_arming(hand_detected=True, gesture_state="GRIPPER_OPEN")
        self.assertFalse(ok)
        self.assertEqual(reason, "ACTION_SERVER_NOT_READY")
        self.assertEqual(self.node.teleop_state, STATE_DISARMED)

    def test_08_existing_enable_arm_motion_true_preserves_arm_requirements(self):
        """8. Existing enable_arm_motion=true tests retain expected behaviour (arm server required for arming)."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_arm_motion", value=True),
        ])
        self.node._teleop_state = STATE_ARMING_PENDING
        self.node._arming_requested_time = time.monotonic()
        self.node.arm_action_client.server_is_ready = MagicMock(return_value=False)
        self.node.gripper_action_client.server_is_ready = MagicMock(return_value=True)

        ok, reason = self.node.confirm_arming(hand_detected=True, gesture_state="GRIPPER_OPEN")
        self.assertFalse(ok)
        self.assertEqual(reason, "ACTION_SERVER_NOT_READY")
        self.assertEqual(self.node.teleop_state, STATE_DISARMED)

    def test_09_parameter_validation_and_dynamic_updates(self):
        """9. Invalid parameter updates are handled according to existing validation convention."""
        # Reject non-bool
        res = self.node.set_parameters([
            rclpy.parameter.Parameter("enable_arm_motion", value="invalid_string"),
        ])
        self.assertFalse(res[0].successful)

        # Dynamic change to False while armed clears pending candidate/snapshot
        self.node._teleop_state = STATE_ARMED
        self.node._pending_ik_candidate = {j: 0.01 for j in ARM_CONTROL_JOINTS}
        self.node._pending_ik_snapshot = TargetSnapshot(x=0.0, y=0.0, z=0.0, generation=1, timestamp=time.monotonic())

        res = self.node.set_parameters([
            rclpy.parameter.Parameter("enable_arm_motion", value=False),
        ])
        self.assertTrue(res[0].successful)
        self.assertIsNone(self.node._pending_ik_candidate)
        self.assertIsNone(self.node._pending_ik_snapshot)

    def test_10_session_quota_and_gripper_counter_separation(self):
        """10. Session quota and gripper dispatch counter separation: _session_goal_dispatch_count vs _gripper_dispatch_count."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_arm_motion", value=True),
            rclpy.parameter.Parameter("max_goals_per_session", value=1),
        ])
        self.node._teleop_state = STATE_ARMED

        # Incrementing gripper dispatch counter alone does NOT trigger session goal limit
        self.node._gripper_dispatch_count = 5
        self.assertFalse(self.node.is_session_goal_limit_reached())

        # Incrementing session goal dispatch count (arm goals) DOES trigger session goal limit
        self.node._session_goal_dispatch_count = 1
        self.assertTrue(self.node.is_session_goal_limit_reached())
        # Arm IK is suppressed
        self.assertFalse(self.node.can_issue_ik_requests())


class TestThumbIndexGestureFeedback(unittest.TestCase):
    """
    Focused unit tests for thumb-index gesture recognition and visual feedback (Task 5C-3D-THUMB-INDEX-GESTURE-FEEDBACK).
    Validates:
      1. Pinch closure detection using scale-normalized fingertip distance.
      2. Spread open hand detection.
      3. Fist closure detection fallback.
      4. Hysteresis band between PINCH_CLOSED_MAX and PINCH_OPEN_MIN.
      5. draw_thumb_index_connector line rendering, color coding, and endpoint geometry.
    """

    class MockLandmark:
        def __init__(self, x: float, y: float, z: float = 0.0):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    def _create_hand(self, thumb_tip_pos, index_tip_pos, wrist_pos=(0.5, 0.9, 0.0), middle_mcp_pos=(0.5, 0.5, 0.0)):
        # 21 landmarks; palm scale = dist_3d(wrist, middle_mcp) = 0.40
        lms = [self.MockLandmark(0.5, 0.7, 0.0) for _ in range(21)]
        lms[0] = self.MockLandmark(*wrist_pos)
        lms[9] = self.MockLandmark(*middle_mcp_pos)
        lms[4] = self.MockLandmark(*thumb_tip_pos)
        lms[8] = self.MockLandmark(*index_tip_pos)
        return lms

    def test_01_pinch_closed_detection(self):
        """1. Closing thumb and index fingertips together (distance <= 0.40 * palm_scale) classifies as GRIPPER_CLOSED."""
        # palm_scale = 0.40. Pinch gap = 0.05 -> norm_dist = 0.05 / 0.40 = 0.125 <= 0.40
        hand = self._create_hand(thumb_tip_pos=(0.52, 0.50, 0.0), index_tip_pos=(0.50, 0.52, 0.0))
        states, gesture = classify_hand_gestures(hand)
        self.assertEqual(gesture, "GRIPPER_CLOSED")

    def test_02_open_fingers_detection(self):
        """2. Spreading thumb and index fingertips apart (distance >= 0.65 * palm_scale) with extension classifies as GRIPPER_OPEN."""
        # palm_scale = 0.40. Open gap = 0.32 -> norm_dist = 0.32 / 0.40 = 0.80 >= 0.65
        hand = self._create_hand(thumb_tip_pos=(0.34, 0.55, 0.0), index_tip_pos=(0.66, 0.55, 0.0))
        # Ensure index and thumb are extended away
        hand[5] = self.MockLandmark(0.55, 0.60, 0.0)  # index MCP
        hand[6] = self.MockLandmark(0.60, 0.58, 0.0)  # index PIP
        hand[1] = self.MockLandmark(0.45, 0.75, 0.0)  # thumb CMC
        hand[2] = self.MockLandmark(0.40, 0.70, 0.0)  # thumb MCP
        hand[17] = self.MockLandmark(0.62, 0.65, 0.0) # pinky MCP
        states, gesture = classify_hand_gestures(hand)
        self.assertEqual(gesture, "GRIPPER_OPEN")

    def test_03_fist_closure_fallback(self):
        """3. When fingers are curled into a fist, classification outputs GRIPPER_CLOSED even if tips are not touching."""
        with patch("hand_gesture_teleop.hand_tracking.classify_thumb") as mock_th, \
             patch("hand_gesture_teleop.hand_tracking.classify_single_finger") as mock_idx:
            mock_th.return_value = "FOLDED"
            mock_idx.side_effect = ["FOLDED", "FOLDED"]
            hand = self._create_hand(thumb_tip_pos=(0.40, 0.65, 0.0), index_tip_pos=(0.60, 0.65, 0.0))
            states, gesture = classify_hand_gestures(hand)
            self.assertEqual(gesture, "GRIPPER_CLOSED")

    def test_04_hysteresis_preserves_closed_in_deadband(self):
        """4. In deadband (0.40 < dist < 0.65), prev_gesture=GRIPPER_CLOSED is retained without flickering."""
        # palm_scale = 0.40. Gap = 0.20 -> norm_dist = 0.20 / 0.40 = 0.50 (deadband)
        hand = self._create_hand(thumb_tip_pos=(0.40, 0.50, 0.0), index_tip_pos=(0.60, 0.50, 0.0))
        states, gesture = classify_hand_gestures(hand, prev_gesture="GRIPPER_CLOSED")
        self.assertEqual(gesture, "GRIPPER_CLOSED")

    def test_05_hysteresis_preserves_open_in_deadband(self):
        """5. In deadband (0.40 < dist < 0.65), prev_gesture=GRIPPER_OPEN is retained without flickering."""
        # palm_scale = 0.40. Gap = 0.20 -> norm_dist = 0.50 (deadband)
        hand = self._create_hand(thumb_tip_pos=(0.40, 0.50, 0.0), index_tip_pos=(0.60, 0.50, 0.0))
        states, gesture = classify_hand_gestures(hand, prev_gesture="GRIPPER_OPEN")
        self.assertEqual(gesture, "GRIPPER_OPEN")

    def test_06_deadband_without_previous_gesture_yields_unknown(self):
        """6. In deadband without prior state, ambiguous pose yields UNKNOWN."""
        hand = self._create_hand(thumb_tip_pos=(0.40, 0.50, 0.0), index_tip_pos=(0.60, 0.50, 0.0))
        states, gesture = classify_hand_gestures(hand, prev_gesture=None)
        self.assertEqual(gesture, "UNKNOWN")

    def test_07_draw_thumb_index_connector_rendering(self):
        """7. draw_thumb_index_connector draws visible line, endpoint circles, and distance tag."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hand = self._create_hand(thumb_tip_pos=(0.3, 0.4, 0.0), index_tip_pos=(0.4, 0.4, 0.0))
        norm_dist, pt_th, pt_idx = draw_thumb_index_connector(
            frame=frame,
            landmarks=hand,
            gesture_state="GRIPPER_CLOSED",
        )
        self.assertAlmostEqual(norm_dist, 0.10 / 0.40, places=2)
        self.assertEqual(pt_th, (192, 192))
        self.assertEqual(pt_idx, (256, 192))
        # Ensure pixels were drawn on the frame
        self.assertGreater(int(np.sum(frame)), 0)
        # Check closed color (Amber/Orange in BGR: B=40, G=180, R=255)
        self.assertGreater(int(np.max(frame[:, :, 2])), 200)

    def test_08_draw_thumb_index_connector_open_color(self):
        """8. draw_thumb_index_connector uses vivid green for GRIPPER_OPEN."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hand = self._create_hand(thumb_tip_pos=(0.2, 0.4, 0.0), index_tip_pos=(0.6, 0.4, 0.0))
        norm_dist, pt_th, pt_idx = draw_thumb_index_connector(
            frame=frame,
            landmarks=hand,
            gesture_state="GRIPPER_OPEN",
        )
        self.assertGreater(norm_dist, 0.65)
        # Check open color (Green in BGR: G=240)
        self.assertGreater(int(np.max(frame[:, :, 1])), 200)


class TestSingleKeyArmDisarmAutoAnchor(unittest.TestCase):
    """
    Test suite for Task 5C-3D-SINGLE-KEY-ARM-DISARM-AUTO-ANCHOR:
    Single-key [E] enable with auto-anchor, [D] immediate stop/cancel,
    optional [C] recalibration, HUD state text, and preservation of all safety gates.
    Covers the 13 required verification items:
      1. Single E keypress captures neutral anchor and enables teleop (STATE_ARMED).
      2. A -> Y no longer required (enabling directly from STATE_DISARMED).
      3. E with no detected hand is rejected (HAND_NOT_DETECTED).
      4. E with invalid or non-finite landmarks is rejected (INVALID_LANDMARKS).
      5. E with unstable hand is rejected (UNSTABLE_HAND).
      6. Failed anchor capture leaves node DISARMED.
      7. D disarms immediately.
      8. D requests cancellation of active goals and sets DISARMED (CANCEL REQUESTED).
      9. C remains functional as optional manual recalibration.
      10. HAND_LOST still triggers automatic disarm.
      11. 3-goal session limit (max_goals_per_session=3) triggers auto-disarm and blocks new dispatches until explicit E rearm.
      12. Gripper open/closed gestures remain fully operational.
      13. IK and trajectory safety protections (MAX_JOINT_DELTA = 0.100 rad) remain unchanged.
    """

    @classmethod
    def setUpClass(cls):
        if not rclpy.ok():
            rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.node = HandTeleopNode()
        # Seed healthy /joint_states
        now = time.monotonic()
        self.node._last_joint_state_time = now
        for j in self.node._expected_joints:
            self.node._latest_joint_positions[j] = 0.0
            self.node._latest_joint_velocities[j] = 0.0

        # Seed mock hardware verification
        self.node._hardware_mock_verified = True
        self.node._hardware_last_verified_time = now

        # Mock action client readiness
        self.node.arm_action_client.server_is_ready = MagicMock(return_value=True)
        self.node.gripper_action_client.server_is_ready = MagicMock(return_value=True)

    def tearDown(self):
        self.node.destroy_node()

    def _create_valid_hand(self, wrist_pos=(0.5, 0.7), middle_mcp_pos=(0.5, 0.5)):
        """Create a list of 21 NormalizedLandmarks representing a valid stable hand."""
        landmarks = [NormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)]
        landmarks[0] = NormalizedLandmark(x=wrist_pos[0], y=wrist_pos[1], z=0.0)
        landmarks[9] = NormalizedLandmark(x=middle_mcp_pos[0], y=middle_mcp_pos[1], z=0.0)
        return landmarks

    def test_01_single_e_captures_anchor_and_enables_teleop(self):
        """1. Single E keypress captures neutral anchor and enables teleop (STATE_ARMED)."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_arm_motion", value=True),
        ])
        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertFalse(self.node.motion_tracker.is_calibrated)
        self.assertEqual(self.node.get_anchor_state_text(), "NOT READY")

        hand = self._create_valid_hand(wrist_pos=(0.5, 0.7), middle_mcp_pos=(0.5, 0.5))
        ok, reason = self.node.enable_teleoperation(hand_detected=True, landmarks=hand, gesture_state="GRIPPER_OPEN")

        self.assertTrue(ok)
        self.assertEqual(reason, "ARMED")
        self.assertEqual(self.node.teleop_state, "ARMED — SIMULATION ONLY")
        self.assertTrue(self.node.is_armed)
        self.assertTrue(self.node.motion_tracker.is_calibrated)
        self.assertAlmostEqual(self.node.motion_tracker.anchor_x, 0.5)
        self.assertAlmostEqual(self.node.motion_tracker.anchor_y, 0.5)
        self.assertAlmostEqual(self.node.motion_tracker.anchor_scale, 0.20, places=3)
        self.assertEqual(self.node.get_anchor_state_text(), "READY")
        self.assertIn("ARMED", self.node.get_control_state_text())

    def test_02_a_to_y_handshake_not_required_to_enable(self):
        """2. A -> Y no longer required (enabling directly from STATE_DISARMED)."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_arm_motion", value=True),
        ])
        # Ensure we are DISARMED without ever calling request_arming()
        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertEqual(self.node._arming_requested_time, 0.0)

        ok, reason = self.node.enable_teleoperation(
            hand_detected=True,
            landmarks=self._create_valid_hand(),
            gesture_state="GRIPPER_OPEN",
        )
        self.assertTrue(ok)
        self.assertEqual(self.node.teleop_state, "ARMED — SIMULATION ONLY")
        self.assertTrue(self.node.is_armed)

    def test_03_enable_rejected_when_no_hand_detected(self):
        """3. E with no detected hand is rejected (HAND_NOT_DETECTED)."""
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        ok, reason = self.node.enable_teleoperation(hand_detected=False, landmarks=None, gesture_state="UNKNOWN")

        self.assertFalse(ok)
        self.assertEqual(reason, "HAND_NOT_DETECTED")
        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertFalse(self.node.motion_tracker.is_calibrated)
        self.assertEqual(self.node.get_control_state_text(), "DISARMED (HAND_NOT_DETECTED)")
        self.assertEqual(self.node.get_anchor_state_text(), "NOT READY")

    def test_04_enable_rejected_on_invalid_or_non_finite_landmarks(self):
        """4. E with invalid or non-finite landmarks is rejected (INVALID_LANDMARKS)."""
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])

        # Subtest A: NaN landmark
        hand_nan = self._create_valid_hand()
        hand_nan[9] = NormalizedLandmark(x=float("nan"), y=0.5)
        ok_a, reason_a = self.node.enable_teleoperation(hand_detected=True, landmarks=hand_nan, gesture_state="GRIPPER_OPEN")
        self.assertFalse(ok_a)
        self.assertEqual(reason_a, "INVALID_LANDMARKS")
        self.assertEqual(self.node.teleop_state, "DISARMED")

        # Subtest B: Microscopic palm scale (< 0.02)
        hand_tiny = self._create_valid_hand(wrist_pos=(0.5, 0.505), middle_mcp_pos=(0.5, 0.500))
        ok_b, reason_b = self.node.enable_teleoperation(hand_detected=True, landmarks=hand_tiny, gesture_state="GRIPPER_OPEN")
        self.assertFalse(ok_b)
        self.assertEqual(reason_b, "INVALID_LANDMARKS")
        self.assertEqual(self.node.teleop_state, "DISARMED")

        # Subtest C: Out-of-bounds coordinate (> 1.1)
        hand_oob = self._create_valid_hand()
        hand_oob[9] = NormalizedLandmark(x=1.8, y=0.5)
        ok_c, reason_c = self.node.enable_teleoperation(hand_detected=True, landmarks=hand_oob, gesture_state="GRIPPER_OPEN")
        self.assertFalse(ok_c)
        self.assertEqual(reason_c, "INVALID_LANDMARKS")
        self.assertEqual(self.node.teleop_state, "DISARMED")

    def test_05_enable_rejected_on_unstable_hand(self):
        """5. E with unstable hand is rejected (UNSTABLE_HAND)."""
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])

        # Subtest A: Unknown/ambiguous gesture
        ok_a, reason_a = self.node.enable_teleoperation(
            hand_detected=True,
            landmarks=self._create_valid_hand(),
            gesture_state="UNKNOWN",
        )
        self.assertFalse(ok_a)
        self.assertEqual(reason_a, "UNSTABLE_HAND")
        self.assertEqual(self.node.teleop_state, "DISARMED")

        # Subtest B: Rapid hand movement across frames (> 0.08 normalized drift)
        self.node._last_hand_landmarks = self._create_valid_hand(middle_mcp_pos=(0.3, 0.5))
        current_hand = self._create_valid_hand(middle_mcp_pos=(0.5, 0.5))  # 0.20 drift
        ok_b, reason_b = self.node.enable_teleoperation(
            hand_detected=True,
            landmarks=current_hand,
            gesture_state="GRIPPER_OPEN",
        )
        self.assertFalse(ok_b)
        self.assertEqual(reason_b, "UNSTABLE_HAND")
        self.assertEqual(self.node.teleop_state, "DISARMED")

    def test_06_failed_anchor_capture_leaves_node_disarmed(self):
        """6. Failed anchor capture leaves node DISARMED."""
        self.node.set_parameters([rclpy.parameter.Parameter("enable_motion", value=True)])
        with patch.object(self.node.motion_tracker, "calibrate", return_value=False):
            ok, reason = self.node.enable_teleoperation(
                hand_detected=True,
                landmarks=self._create_valid_hand(),
                gesture_state="GRIPPER_OPEN",
            )
            self.assertFalse(ok)
            self.assertEqual(reason, "ANCHOR_CAPTURE_FAILED")
            self.assertEqual(self.node.teleop_state, "DISARMED")
            self.assertFalse(self.node.is_armed)
            self.assertFalse(self.node.motion_tracker.is_calibrated)

    def test_07_d_key_disarms_immediately(self):
        """7. D disarms immediately."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_arm_motion", value=True),
        ])
        self.node.enable_teleoperation(
            hand_detected=True,
            landmarks=self._create_valid_hand(),
            gesture_state="GRIPPER_OPEN",
        )
        self.assertTrue(self.node.is_armed)

        self.node.disarm("USER_KEYPRESS_D")
        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertFalse(self.node.is_armed)
        self.assertFalse(self.node.motion_tracker.is_calibrated)
        self.assertEqual(self.node.get_anchor_state_text(), "NOT READY")
        self.assertEqual(self.node.get_control_state_text(), "DISARMED")

    def test_08_d_key_requests_cancellation_and_sets_cancel_requested(self):
        """8. D requests cancellation of active goals and sets DISARMED (CANCEL REQUESTED)."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_arm_motion", value=True),
        ])
        self.node.enable_teleoperation(
            hand_detected=True,
            landmarks=self._create_valid_hand(),
            gesture_state="GRIPPER_OPEN",
        )
        self.node._goal_active = True
        mock_gh = MagicMock()
        self.node._active_goal_handle = mock_gh

        self.node.disarm("USER_KEYPRESS_D")
        mock_gh.cancel_goal_async.assert_called_once()
        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertTrue(self.node._cancel_requested_on_disarm)
        self.assertEqual(self.node.get_control_state_text(), "DISARMED (CANCEL REQUESTED)")

    def test_09_c_key_optional_manual_recalibration_while_armed(self):
        """9. C remains functional as optional manual recalibration."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_arm_motion", value=True),
        ])
        self.node.enable_teleoperation(
            hand_detected=True,
            landmarks=self._create_valid_hand(middle_mcp_pos=(0.5, 0.5)),
            gesture_state="GRIPPER_OPEN",
        )
        self.assertTrue(self.node.is_armed)
        self.assertAlmostEqual(self.node.motion_tracker.anchor_x, 0.5)

        # Recalibrate via [C]
        new_hand = self._create_valid_hand(wrist_pos=(0.6, 0.8), middle_mcp_pos=(0.6, 0.6))
        lm9 = new_hand[9]
        lm0 = new_hand[0]
        scale_2d = dist_2d(lm0, lm9)
        calib_ok = self.node.motion_tracker.calibrate(lm9.x, lm9.y, scale_2d)
        self.assertTrue(calib_ok)
        self.assertAlmostEqual(self.node.motion_tracker.anchor_x, 0.6)
        self.assertAlmostEqual(self.node.motion_tracker.anchor_y, 0.6)
        self.assertTrue(self.node.is_armed)
        self.assertEqual(self.node.get_anchor_state_text(), "READY")

    def test_10_hand_lost_triggers_automatic_disarm(self):
        """10. HAND_LOST still triggers automatic disarm."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_arm_motion", value=True),
        ])
        self.node.enable_teleoperation(
            hand_detected=True,
            landmarks=self._create_valid_hand(),
            gesture_state="GRIPPER_OPEN",
        )
        self.assertTrue(self.node.is_armed)

        self.node.disarm("HAND_LOST")
        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertFalse(self.node.is_armed)
        self.assertFalse(self.node.motion_tracker.is_calibrated)
        self.assertEqual(self.node._last_disarm_reason, "HAND_LOST")

    def test_11_three_goal_session_limit_auto_disarms_and_blocks_until_explicit_e_rearm(self):
        """11. 3-goal session limit (max_goals_per_session=3) triggers auto-disarm and blocks new dispatches until explicit E rearm."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_arm_motion", value=True),
            rclpy.parameter.Parameter("single_goal_test_mode", value=False),
            rclpy.parameter.Parameter("max_goals_per_session", value=3),
        ])
        self.node.enable_teleoperation(
            hand_detected=True,
            landmarks=self._create_valid_hand(),
            gesture_state="GRIPPER_OPEN",
        )
        self.assertEqual(self.node.effective_session_goal_limit, 3)
        self.assertEqual(self.node.session_goal_dispatch_count, 0)
        self.assertIn("ARMED [Session 0/3]", self.node.get_control_state_text())

        # Dispatch and complete 3 goals
        for i in range(1, 4):
            self.assertFalse(self.node.is_session_goal_limit_reached())
            mock_gh = MagicMock()
            mock_gh.accepted = True
            mock_res_future = MagicMock()
            mock_gh.get_result_async.return_value = mock_res_future
            self.node.arm_action_client.send_goal_async = MagicMock(return_value=mock_gh)

            cand = {f"joint{k}": 0.01 * i for k in range(1, 6)}
            dispatched = self.node.dispatch_arm_trajectory(cand)
            self.assertTrue(dispatched)
            self.assertEqual(self.node.session_goal_dispatch_count, i)

            wrapped_result = MagicMock()
            wrapped_result.status = GoalStatus.STATUS_SUCCEEDED
            res_future_mock = MagicMock()
            res_future_mock.result.return_value = wrapped_result
            self.node._goal_result_callback(res_future_mock)

        # 3 goals dispatched and finished -> limit reached and auto-disarmed
        self.assertTrue(self.node.is_session_goal_limit_reached())
        self.assertEqual(self.node.teleop_state, "DISARMED")
        self.assertFalse(self.node.is_armed)
        self.assertEqual(self.node._last_disarm_reason, "SESSION_GOAL_LIMIT_COMPLETED")

        # 4th goal dispatch is blocked
        cand4 = {f"joint{k}": 0.04 for k in range(1, 6)}
        dispatched4 = self.node.dispatch_arm_trajectory(cand4)
        self.assertFalse(dispatched4)

        # Explicit E rearm starts a new session (quota resets to 0/3)
        ok_rearm, _ = self.node.enable_teleoperation(
            hand_detected=True,
            landmarks=self._create_valid_hand(),
            gesture_state="GRIPPER_OPEN",
        )
        self.assertTrue(ok_rearm)
        self.assertTrue(self.node.is_armed)
        self.assertEqual(self.node.session_goal_dispatch_count, 0)
        self.assertIn("ARMED [Session 0/3]", self.node.get_control_state_text())

    def test_12_gripper_gestures_remain_operational(self):
        """12. Gripper open/closed gestures remain fully operational."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_arm_motion", value=False),
        ])
        self.node.enable_teleoperation(
            hand_detected=True,
            landmarks=self._create_valid_hand(),
            gesture_state="GRIPPER_OPEN",
        )
        mock_future = MagicMock()
        self.node.gripper_action_client.send_goal_async = MagicMock(return_value=mock_future)

        # Command OPEN
        dispatched_open = self.node._send_gripper_command("GRIPPER_OPEN")
        self.assertTrue(dispatched_open)
        self.assertEqual(self.node._last_commanded_gripper_state, "GRIPPER_OPEN")

        # Command CLOSED
        self.node._gripper_goal_active = False
        dispatched_closed = self.node._send_gripper_command("GRIPPER_CLOSED")
        self.assertTrue(dispatched_closed)
        self.assertEqual(self.node._last_commanded_gripper_state, "GRIPPER_CLOSED")

    def test_13_ik_and_trajectory_safety_protections_unchanged(self):
        """13. IK and trajectory safety protections (MAX_JOINT_DELTA = 0.100 rad) remain unchanged."""
        self.node.set_parameters([
            rclpy.parameter.Parameter("enable_motion", value=True),
            rclpy.parameter.Parameter("enable_arm_motion", value=True),
        ])
        self.node.enable_teleoperation(
            hand_detected=True,
            landmarks=self._create_valid_hand(),
            gesture_state="GRIPPER_OPEN",
        )
        self.assertEqual(MAX_JOINT_DELTA, 0.10)

        # Jump exceeding limit (> 0.10 rad)
        jump_candidate = {
            "joint1": 0.15,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
        }
        self.assertFalse(self.node._validate_joint_discontinuity(jump_candidate))

        # Safe jump (<= 0.10 rad)
        safe_candidate = {
            "joint1": 0.05,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
        }
        self.assertTrue(self.node._validate_joint_discontinuity(safe_candidate))


if __name__ == "__main__":
    unittest.main()

