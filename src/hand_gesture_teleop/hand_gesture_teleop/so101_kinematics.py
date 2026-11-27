#!/usr/bin/env python3
"""
so101_kinematics.py
===================
Pure-Python forward kinematics, translational Jacobian, and continuity-aware
damped least-squares (DLS) differential IK solver for the 5-DOF SO-101 arm.

Kinematic chain (matching src/hand_gesture_teleop/urdf/so101_sim.urdf.xacro
and src/lerobot_moveit/config/so101.srdf):
  world -> base (fixed) -> shoulder (joint1) -> upper_arm (joint2) ->
  lower_arm (joint3) -> wrist (joint4) -> gripper (joint5)

Why 3x5 position-only DLS for SO-101:
  SO-101 is a 5-DOF arm whose joint5 rolls the gripper around its own tip axis.
  Forcing full 6-DOF Cartesian twist constraints (including zero angular rate
  wz=0) over-constrains lateral X motion via base yaw joint1, severely
  attenuating lateral speed and raising the 6x5 condition number (~29-34).
  Operating in the 3x5 translational task space (matching position_only_ik=True
  in kinematics.yaml) keeps the manipulator well-conditioned (cond ~3.3-5.5),
  preserves exact continuous branch locality without KDL configuration flips,
  and naturally avoids spurious wrist-roll drift on joint5.
"""

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SO101_ARM_JOINTS: Tuple[str, ...] = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
)

SO101_JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    "joint1": (-2.1, 2.1),
    "joint2": (-2.1, 2.1),
    "joint3": (-2.1, 2.1),
    "joint4": (-2.1, 2.1),
    "joint5": (-3.1, 3.1),
}

# Verified transform chain fallback matching urdf/so101_sim.urdf.xacro lines 47-402
_DEFAULT_CHAIN_SPECS: Tuple[Tuple[str, str, Tuple[float, float, float], Tuple[float, float, float]], ...] = (
    ("base_joint", "fixed", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ("joint1", "revolute", (0.0207909, -0.0230745, 0.0948817), (-3.14159, 6.03684e-16, 1.5708)),
    ("joint2", "revolute", (-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0.0)),
    ("joint3", "revolute", (-0.11257, -0.028, 2.46331e-16), (-1.22818e-15, 5.75928e-16, 1.5708)),
    ("joint4", "revolute", (-0.1349, 0.0052, 1.65232e-16), (3.2474e-15, 2.86219e-15, -1.5708)),
    ("joint5", "revolute", (0.0, -0.0611, 0.0181), (1.5708, -9.38083e-08, 3.14159)),
)


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert URDF intrinsic XYZ fixed-axis RPY (Rz(yaw) @ Ry(pitch) @ Rx(roll)) to 3x3 matrix."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def _origin_to_transform(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    """Build 4x4 homogeneous transform from URDF origin xyz & rpy."""
    T = np.eye(4, dtype=float)
    T[:3, :3] = _rpy_to_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    T[:3, 3] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    return T


def _axis_angle_transform(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues 4x4 pure rotation matrix around unit axis vector."""
    ax = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(ax))
    if norm < 1e-12:
        return np.eye(4, dtype=float)
    ax = ax / norm
    kx, ky, kz = ax
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], dtype=float)
    c = math.cos(theta)
    s = math.sin(theta)
    R = np.eye(3, dtype=float) + s * K + (1.0 - c) * (K @ K)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    return T


def unwrap_joint_delta(target: float, current: float) -> float:
    """Shortest signed angular delta wrapped cleanly onto [-pi, pi]."""
    return float(math.atan2(math.sin(target - current), math.cos(target - current)))


def resolve_urdf_path(custom_path: str = "") -> Optional[str]:
    """Locate so101_sim.urdf.xacro across source and installed colcon workspaces."""
    if custom_path and os.path.isfile(custom_path):
        return os.path.normpath(custom_path)

    try:
        from ament_index_python.packages import get_package_share_directory
        share_dir = get_package_share_directory("hand_gesture_teleop")
        candidate = os.path.join(share_dir, "urdf", "so101_sim.urdf.xacro")
        if os.path.isfile(candidate):
            return os.path.normpath(candidate)
    except Exception:
        pass

    module_dir = os.path.dirname(os.path.realpath(__file__))
    candidate = os.path.normpath(os.path.join(module_dir, "..", "urdf", "so101_sim.urdf.xacro"))
    if os.path.isfile(candidate):
        return candidate

    cwd_candidate = os.path.normpath(
        os.path.join(os.getcwd(), "src", "hand_gesture_teleop", "urdf", "so101_sim.urdf.xacro")
    )
    if os.path.isfile(cwd_candidate):
        return cwd_candidate

    return None


@dataclass(frozen=True)
class DifferentialIKResult:
    """Result of a single continuous continuity-aware differential IK cycle."""
    q_next: Tuple[float, float, float, float, float]
    qdot_cmd: Tuple[float, float, float, float, float]
    ee_pos: Tuple[float, float, float]
    max_joint_delta: float
    max_joint_vel: float
    pos_error_m: float
    jacobian_cond: float
    min_singular_value: float
    clamped_to_limits: bool


class SO101Kinematics:
    """
    Exact rigid-body FK, translational Jacobian, and singularity-damped
    differential IK solver for SO-101 arm joints (joint1..joint5 -> gripper).
    """

    def __init__(self, urdf_path: str = "") -> None:
        self._chain: List[Tuple[str, str, np.ndarray, np.ndarray, Tuple[float, float]]] = []
        self._joint_limits: Dict[str, Tuple[float, float]] = dict(SO101_JOINT_LIMITS)
        loaded_from_urdf = False

        resolved = resolve_urdf_path(urdf_path)
        if resolved is not None:
            loaded_from_urdf = self._load_from_urdf(resolved)

        if not loaded_from_urdf:
            self._load_default_chain()

        # Precompute shoulder origin in world frame (origin after joint2 fixed offset)
        self._shoulder_origin = self._compute_shoulder_origin()

    def _load_default_chain(self) -> None:
        """Load compiled fallback transform chain matching so101_sim.urdf.xacro."""
        self._chain = []
        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        for name, jtype, xyz, rpy in _DEFAULT_CHAIN_SPECS:
            limits = self._joint_limits.get(name, (-math.pi, math.pi))
            self._chain.append((name, jtype, _origin_to_transform(xyz, rpy), z_axis.copy(), limits))

    def _load_from_urdf(self, path: str) -> bool:
        """Parse fixed base_joint and revolute joint1..joint5 from SO-101 URDF/xacro."""
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            joints_xml = {}
            for elem in root.findall("joint"):
                jname = elem.attrib.get("name", "")
                if jname in ("base_joint",) or jname in SO101_ARM_JOINTS:
                    jtype = elem.attrib.get("type", "fixed")
                    orig = elem.find("origin")
                    xyz = [float(v) for v in orig.attrib["xyz"].split()] if orig is not None else [0.0, 0.0, 0.0]
                    rpy = [float(v) for v in orig.attrib["rpy"].split()] if orig is not None else [0.0, 0.0, 0.0]
                    axis_el = elem.find("axis")
                    axis = (
                        np.array([float(v) for v in axis_el.attrib["xyz"].split()], dtype=float)
                        if axis_el is not None
                        else np.array([0.0, 0.0, 1.0], dtype=float)
                    )
                    lim_el = elem.find("limit")
                    if lim_el is not None and "lower" in lim_el.attrib and "upper" in lim_el.attrib:
                        lo = float(lim_el.attrib["lower"])
                        hi = float(lim_el.attrib["upper"])
                    else:
                        lo, hi = SO101_JOINT_LIMITS.get(jname, (-math.pi, math.pi))
                    joints_xml[jname] = (jname, jtype, _origin_to_transform(xyz, rpy), axis, (lo, hi))

            ordered_names = ("base_joint",) + SO101_ARM_JOINTS
            if not all(n in joints_xml for n in ordered_names):
                return False

            self._chain = [joints_xml[n] for n in ordered_names]
            for jname in SO101_ARM_JOINTS:
                self._joint_limits[jname] = joints_xml[jname][4]
            return True
        except Exception:
            return False

    def _compute_shoulder_origin(self) -> np.ndarray:
        """Return world position of shoulder pitch pivot (joint2 origin at q1=0)."""
        T = np.eye(4, dtype=float)
        for name, _, T_static, _, _ in self._chain:
            T = T @ T_static
            if name == "joint2":
                return T[:3, 3].copy()
        return np.array([0.02079, -0.04135, 0.14908], dtype=float)

    @property
    def shoulder_origin(self) -> Tuple[float, float, float]:
        """World-frame shoulder pitch pivot coordinates (m)."""
        return (
            float(self._shoulder_origin[0]),
            float(self._shoulder_origin[1]),
            float(self._shoulder_origin[2]),
        )

    @property
    def joint_limits(self) -> Dict[str, Tuple[float, float]]:
        """Joint limits mapping for joint1..joint5."""
        return dict(self._joint_limits)

    def _to_q_array(self, q: Sequence[float] | Dict[str, float]) -> np.ndarray:
        """Normalize joint dict or 5-element sequence into validated float64 vector."""
        if isinstance(q, dict):
            vals = [float(q[j]) for j in SO101_ARM_JOINTS]
        else:
            if len(q) != len(SO101_ARM_JOINTS):
                raise ValueError(f"Expected {len(SO101_ARM_JOINTS)} joint positions, got {len(q)}")
            vals = [float(v) for v in q]
        arr = np.asarray(vals, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite joint values in IK input: {vals}")
        return arr

    def forward_kinematics_and_jacobian(
        self,
        q: Sequence[float] | Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute world-frame gripper position (3,), rotation matrix (3,3),
        and 3x5 translational Jacobian J_pos (3,5).
        """
        q_vec = self._to_q_array(q)
        T = np.eye(4, dtype=float)
        origins: List[np.ndarray] = []
        axes_world: List[np.ndarray] = []

        rev_idx = 0
        for _, jtype, T_static, axis_local, _ in self._chain:
            T = T @ T_static
            if jtype == "revolute":
                origins.append(T[:3, 3].copy())
                axes_world.append((T[:3, :3] @ axis_local).copy())
                T = T @ _axis_angle_transform(axis_local, float(q_vec[rev_idx]))
                rev_idx += 1

        p_ee = T[:3, 3].copy()
        R_ee = T[:3, :3].copy()

        J_pos = np.zeros((3, 5), dtype=float)
        for i in range(5):
            J_pos[:, i] = np.cross(axes_world[i], p_ee - origins[i])

        return p_ee, R_ee, J_pos

    def forward_kinematics(
        self,
        q: Sequence[float] | Dict[str, float],
    ) -> Tuple[Tuple[float, float, float], np.ndarray]:
        """Compute world-frame gripper position tuple (x, y, z) and rotation matrix."""
        p_ee, R_ee, _ = self.forward_kinematics_and_jacobian(q)
        return (float(p_ee[0]), float(p_ee[1]), float(p_ee[2])), R_ee

    def clamp_target_to_reachable_workspace(
        self,
        target_xyz: Sequence[float],
        min_height_z: float = 0.050,
        workspace_bounds_x: Tuple[float, float] = (-0.25, 0.25),
        workspace_bounds_y: Tuple[float, float] = (-0.40, -0.08),
        workspace_bounds_z: Tuple[float, float] = (0.05, 0.45),
        min_shoulder_radius: float = 0.085,
        max_shoulder_radius: float = 0.295,
    ) -> Tuple[float, float, float]:
        """
        Clamp desired world Cartesian target strictly within SO-101 box bounds and
        conservative spherical arm reach centered at shoulder pivot.
        """
        tx = float(np.clip(float(target_xyz[0]), workspace_bounds_x[0], workspace_bounds_x[1]))
        ty = float(np.clip(float(target_xyz[1]), workspace_bounds_y[0], workspace_bounds_y[1]))
        tz = float(np.clip(float(target_xyz[2]), max(workspace_bounds_z[0], min_height_z), workspace_bounds_z[1]))

        pt = np.array([tx, ty, tz], dtype=float)
        rel = pt - self._shoulder_origin
        dist = float(np.linalg.norm(rel))
        if dist < 1e-6:
            pt = self._shoulder_origin + np.array([0.0, -min_shoulder_radius, 0.0])
        elif dist > max_shoulder_radius:
            pt = self._shoulder_origin + rel * (max_shoulder_radius / dist)
        elif dist < min_shoulder_radius:
            pt = self._shoulder_origin + rel * (min_shoulder_radius / dist)

        pt[2] = max(float(pt[2]), min_height_z)
        return float(pt[0]), float(pt[1]), float(pt[2])

    def solve_differential_step(
        self,
        q_curr: Sequence[float] | Dict[str, float],
        target_xyz: Sequence[float],
        v_ff_xyz: Sequence[float] = (0.0, 0.0, 0.0),
        dt: float = 0.05,
        prev_qdot: Optional[Sequence[float]] = None,
        pos_gain: float = 5.0,
        max_cartesian_speed: float = 0.06,
        damping_min: float = 0.015,
        damping_max: float = 0.06,
        sigma_threshold: float = 0.055,
        max_joint_delta: float = 0.05,
        max_joint_vel: float = 0.50,
        max_joint_acc: float = 2.5,
    ) -> DifferentialIKResult:
        """
        Compute a single continuity-preserving joint step (q_next, qdot_cmd) using
        3x5 position-only Damped Least Squares with singularity-adaptive damping,
        uniform direction-preserving velocity/step scaling, acceleration limiting,
        and joint-limit enforcement.
        """
        dt_safe = max(1e-3, min(float(dt), 0.25))
        q_vec = self._to_q_array(q_curr)
        tgt = np.asarray([float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])], dtype=float)
        v_ff = np.asarray([float(v_ff_xyz[0]), float(v_ff_xyz[1]), float(v_ff_xyz[2])], dtype=float)

        if not (np.all(np.isfinite(tgt)) and np.all(np.isfinite(v_ff))):
            raise ValueError("Target or feedforward velocity contains non-finite values.")

        if prev_qdot is None:
            qdot_prev = np.zeros(5, dtype=float)
        else:
            qdot_prev = np.asarray([float(v) for v in prev_qdot], dtype=float)
            if qdot_prev.shape != (5,) or not np.all(np.isfinite(qdot_prev)):
                qdot_prev = np.zeros(5, dtype=float)

        p_ee, _, J_pos = self.forward_kinematics_and_jacobian(q_vec)
        pos_err_vec = tgt - p_ee
        pos_err_m = float(np.linalg.norm(pos_err_vec))

        # Task-space desired velocity = closed-loop proportional correction + feedforward
        v_cmd = v_ff + float(pos_gain) * pos_err_vec
        v_norm = float(np.linalg.norm(v_cmd))
        if v_norm > float(max_cartesian_speed) and v_norm > 1e-12:
            v_cmd = v_cmd * (float(max_cartesian_speed) / v_norm)

        # Suppress sub-millimeter micro-jitter when stationary in deadzone
        if float(np.linalg.norm(v_ff)) < 1e-5 and pos_err_m < 0.0005:
            v_cmd = np.zeros(3, dtype=float)

        # Active translational joints are joint1..joint4 (joint5 only rotates wrist roll around tip origin)
        J4 = J_pos[:, :4]
        sing_vals = np.linalg.svd(J4, compute_uv=False)
        sigma_min = float(sing_vals[-1]) if len(sing_vals) > 0 else 0.0
        sigma_max = float(sing_vals[0]) if len(sing_vals) > 0 else 0.0
        cond = sigma_max / max(sigma_min, 1e-9)

        # Singularity-adaptive Levenberg-Marquardt damping
        if sigma_min < sigma_threshold:
            ratio = sigma_min / max(sigma_threshold, 1e-6)
            lam_sq = (damping_min ** 2) + (1.0 - ratio ** 2) * ((damping_max ** 2) - (damping_min ** 2))
        else:
            lam_sq = damping_min ** 2

        JJT = J4 @ J4.T + lam_sq * np.eye(3, dtype=float)
        qdot_4 = J4.T @ np.linalg.solve(JJT, v_cmd)

        # Gentle nullspace limit-avoidance push ONLY when actively moving near hard joint boundary
        if float(np.linalg.norm(v_cmd)) > 1e-4:
            grad_4 = np.zeros(4, dtype=float)
            margin = 0.25
            for idx in range(4):
                jname = SO101_ARM_JOINTS[idx]
                lo, hi = self._joint_limits[jname]
                if q_vec[idx] < lo + margin:
                    grad_4[idx] = (lo + margin - q_vec[idx]) / margin
                elif q_vec[idx] > hi - margin:
                    grad_4[idx] = -(q_vec[idx] - (hi - margin)) / margin
            if np.any(grad_4 != 0.0):
                J4_pinv = J4.T @ np.linalg.inv(JJT)
                null_proj = np.eye(4, dtype=float) - (J4_pinv @ J4)
                qdot_4 = qdot_4 + 0.35 * (null_proj @ grad_4)

        qdot = np.array([qdot_4[0], qdot_4[1], qdot_4[2], qdot_4[3], 0.0], dtype=float)

        # Joint-limit directional guard (never drive deeper into hit limit)
        clamped = False
        lim_eps = 0.005
        for idx, jname in enumerate(SO101_ARM_JOINTS):
            lo, hi = self._joint_limits[jname]
            if q_vec[idx] <= lo + lim_eps and qdot[idx] < 0.0:
                qdot[idx] = 0.0
                clamped = True
            elif q_vec[idx] >= hi - lim_eps and qdot[idx] > 0.0:
                qdot[idx] = 0.0
                clamped = True

        # Per-joint acceleration limiting relative to previous velocity
        max_dv = float(max_joint_acc) * dt_safe
        qdot = qdot_prev + np.clip(qdot - qdot_prev, -max_dv, max_dv)

        # Direction-preserving uniform velocity bound
        vel_peak = float(np.max(np.abs(qdot)))
        if vel_peak > float(max_joint_vel) and vel_peak > 1e-12:
            qdot = qdot * (float(max_joint_vel) / vel_peak)

        # Direction-preserving uniform step-delta bound (strictly <= max_joint_delta)
        dq = qdot * dt_safe
        delta_peak = float(np.max(np.abs(dq)))
        if delta_peak > float(max_joint_delta) and delta_peak > 1e-12:
            scale = float(max_joint_delta) / delta_peak
            dq = dq * scale
            qdot = dq / dt_safe

        q_next = q_vec + dq
        for idx, jname in enumerate(SO101_ARM_JOINTS):
            lo, hi = self._joint_limits[jname]
            if q_next[idx] < lo or q_next[idx] > hi:
                q_next[idx] = float(np.clip(q_next[idx], lo, hi))
                qdot[idx] = (q_next[idx] - q_vec[idx]) / dt_safe
                clamped = True

        actual_dq = np.abs(q_next - q_vec)
        return DifferentialIKResult(
            q_next=tuple(float(v) for v in q_next),
            qdot_cmd=tuple(float(v) for v in qdot),
            ee_pos=(float(p_ee[0]), float(p_ee[1]), float(p_ee[2])),
            max_joint_delta=float(np.max(actual_dq)),
            max_joint_vel=float(np.max(np.abs(qdot))),
            pos_error_m=pos_err_m,
            jacobian_cond=float(cond),
            min_singular_value=float(sigma_min),
            clamped_to_limits=clamped,
        )
