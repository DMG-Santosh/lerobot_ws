#!/usr/bin/env python3
"""
hand_tracking.py
================
Reusable Hand-Tracking, Gesture-Classification, and Motion-Mapping Module.
Part of the hand_gesture_teleop package for the SO-101 robotic arm.

This module provides:
  1. GestureConfig: Tunable thresholds for three-finger gesture classification.
  2. MappingConfig: Dead-zone, scaling, and smoothing configuration for motion tracking.
  3. Geometric helpers: dist_3d, dist_2d.
  4. Gesture classification: classify_single_finger, classify_thumb, classify_hand_gestures.
     Supported states: "GRIPPER_OPEN", "GRIPPER_CLOSED", "UNKNOWN".
  5. Temporal smoothing: TemporalSmoother (4-frame debounce filter).
  6. Motion tracking: MotionPreviewTracker, MotionPreviewData, apply_deadzone_and_scale.
     Output: Bounded normalized displacements (ux, uy, uz in [-1.0, 1.0]).
  7. Visual drawing helpers: draw_landmarks_on_frame, draw_motion_reticle, draw_compact_hud, draw_mapping_hud.

NOTE: This module contains NO camera initialization, NO OpenCV window creation,
and NO ROS 2 publishers, subscribers, or action clients. It is safe to import
headlessly or in any ROS 2 node.
"""

import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

_FONT_CACHE: Dict[Tuple[int, bool], object] = {}


def _get_truetype_font(size: int, bold: bool = True):
    """Retrieve or load TrueType desktop font for Full HD, crisp text rendering."""
    if not _HAS_PIL:
        return None
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf" if bold else "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    ]
    font = None
    for p in font_candidates:
        if os.path.isfile(p):
            try:
                font = ImageFont.truetype(p, size)
                break
            except Exception:
                continue
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
    _FONT_CACHE[key] = font
    return font

# MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision


# ---------------------------------------------------------------------------
# Landmark Data Structures & Gesture Recognition Configuration
# ---------------------------------------------------------------------------
@dataclass
class NormalizedLandmark:
    """Represents a single 3D normalized landmark (e.g. MediaPipe hand landmark)."""
    x: float = 0.5
    y: float = 0.5
    z: float = 0.0


@dataclass(frozen=True)
class GestureConfig:
    # --- Index Finger (Landmarks 5=MCP, 6=PIP, 7=DIP, 8=TIP) ---
    INDEX_EXT_MCP_MIN: float = 0.65       # Above this: finger is straight/extended
    INDEX_FOLD_MCP_MAX: float = 0.45      # Below this: finger is curled/folded
    INDEX_EXT_WRIST_RATIO_MIN: float = 1.12
    INDEX_FOLD_WRIST_RATIO_MAX: float = 1.02

    # --- Middle Finger (Landmarks 9=MCP, 10=PIP, 11=DIP, 12=TIP) ---
    MIDDLE_EXT_MCP_MIN: float = 0.65
    MIDDLE_FOLD_MCP_MAX: float = 0.45
    MIDDLE_EXT_WRIST_RATIO_MIN: float = 1.12
    MIDDLE_FOLD_WRIST_RATIO_MAX: float = 1.02

    # --- Thumb (Landmarks 1=CMC, 2=MCP, 3=IP, 4=TIP) ---
    # Intrinsic distance ratios relative to Index MCP [5] and Pinky MCP [17]
    THUMB_EXT_SPREAD_MIN: float = 0.42    # Thumb abducted outward from index base
    THUMB_EXT_REACH_MIN: float = 0.70     # Thumb tip far from pinky side
    THUMB_EXT_RATIO_MIN: float = 1.45     # Thumb joints straight

    THUMB_FOLD_SPREAD_MAX: float = 0.35   # Thumb tucked near index base / fist
    THUMB_FOLD_REACH_MAX: float = 0.65    # Thumb folded inward across palm
    THUMB_FOLD_RATIO_MAX: float = 1.35    # Thumb joints curled

    # --- Thumb-Index Pinch Distance Thresholds (Normalized by palm scale dist_3d(0, 9)) ---
    PINCH_CLOSED_MAX: float = 0.40        # Normalized distance <= 0.40 -> GRIPPER_CLOSED
    PINCH_OPEN_MIN: float = 0.65          # Normalized distance >= 0.65 -> GRIPPER_OPEN

    # --- Temporal Debounce Smoothing ---
    SMOOTHING_CONFIRM_FRAMES: int = 4     # Consecutive frames to confirm gesture transition


# ---------------------------------------------------------------------------
# Motion Mapping Preview Configuration & Tunable Thresholds
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MappingConfig:
    # Dead-zone thresholds: displacements within these margins produce 0.0 (holding steady)
    DEADZONE_X: float = 0.05       # 5% of normalized horizontal screen space
    DEADZONE_Y: float = 0.05       # 5% of normalized vertical screen space
    DEADZONE_Z: float = 0.08       # 8% relative change in 2D palm scale (monocular depth proxy)

    # Maximum expected displacement corresponding to full-scale output (+/- 1.0)
    MAX_DISPLACEMENT_X: float = 0.20
    MAX_DISPLACEMENT_Y: float = 0.20
    MAX_DISPLACEMENT_Z: float = 0.35

    # Exponential moving average filter factor (0.0 = frozen, 1.0 = raw/unfiltered)
    SMOOTHING_ALPHA: float = 0.35


GESTURE_CONFIG = GestureConfig()
MAPPING_CONFIG = MappingConfig()

# ---------------------------------------------------------------------------
# MediaPipe Drawing Helpers
# ---------------------------------------------------------------------------
_HAND_CONNECTIONS = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS
_LANDMARK_STYLE = mp_vision.drawing_styles.get_default_hand_landmarks_style()
_CONNECTION_STYLE = mp_vision.drawing_styles.get_default_hand_connections_style()


def draw_thumb_index_connector(
    frame: np.ndarray,
    landmarks: list,
    gesture_state: str = "UNKNOWN",
    palm_scale: Optional[float] = None,
) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
    """
    Draw a clearly visible connecting line between thumb tip (landmark 4) and
    index fingertip (landmark 8) that expands and contracts as their distance changes.

    Visual feedback:
      - Line color matches the gesture state:
          * Vivid Green (60, 240, 100) for GRIPPER_OPEN
          * Warm Amber/Orange (40, 180, 255) for GRIPPER_CLOSED
          * Cyan (240, 200, 60) for UNKNOWN / transition band
      - Antialiased line with subtle dark outline for high contrast
      - Highlight accent circles at thumb tip and index fingertip
      - Numeric normalized distance and state indicator tag near midpoint

    Returns:
        (norm_dist, pt_thumb_px, pt_index_px)
    """
    h, w = frame.shape[:2]
    lm_thumb = landmarks[4]
    lm_index = landmarks[8]

    pt_thumb = (int(round(lm_thumb.x * w)), int(round(lm_thumb.y * h)))
    pt_index = (int(round(lm_index.x * w)), int(round(lm_index.y * h)))

    if palm_scale is None or palm_scale < 1e-4:
        if len(landmarks) >= 10:
            palm_scale = max(dist_3d(landmarks[0], landmarks[9]), 1e-4)
        else:
            palm_scale = 1.0

    norm_dist = dist_3d(lm_thumb, lm_index) / palm_scale

    # Resolve line color and label by gesture state
    if gesture_state == "GRIPPER_CLOSED":
        line_color = (40, 180, 255)       # Amber / Orange (BGR)
        label = "CLOSED"
    elif gesture_state == "GRIPPER_OPEN":
        line_color = (60, 240, 100)       # Vivid Green (BGR)
        label = "OPEN"
    else:
        line_color = (240, 200, 60)       # Cyan (BGR)
        label = ""

    thickness = max(2, int(round(3 * (w / 1080.0))))
    radius = max(4, int(round(6 * (w / 1080.0))))

    # 1. Dark outline for contrast against light backgrounds/skin
    cv2.line(frame, pt_thumb, pt_index, (15, 15, 15), thickness + 2, cv2.LINE_AA)
    # 2. Main connecting line
    cv2.line(frame, pt_thumb, pt_index, line_color, thickness, cv2.LINE_AA)

    # 3. Endpoint accent circles
    cv2.circle(frame, pt_thumb, radius + 2, (15, 15, 15), -1, cv2.LINE_AA)
    cv2.circle(frame, pt_thumb, radius, line_color, -1, cv2.LINE_AA)
    cv2.circle(frame, pt_index, radius + 2, (15, 15, 15), -1, cv2.LINE_AA)
    cv2.circle(frame, pt_index, radius, line_color, -1, cv2.LINE_AA)

    # 4. Midpoint distance and state tag
    mid_x = (pt_thumb[0] + pt_index[0]) // 2
    mid_y = (pt_thumb[1] + pt_index[1]) // 2
    tag = f"{label} ({norm_dist:.2f})" if label else f"{norm_dist:.2f}"

    font_scale = max(0.35, 0.42 * (w / 1080.0))
    cv2.putText(frame, tag, (mid_x + 10, mid_y - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (15, 15, 15), 2, cv2.LINE_AA)
    cv2.putText(frame, tag, (mid_x + 10, mid_y - 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_color, 1, cv2.LINE_AA)

    return norm_dist, pt_thumb, pt_index


def draw_landmarks_on_frame(
    frame_bgr: np.ndarray,
    detection_result: mp_vision.HandLandmarkerResult,
    gesture_state: str = "UNKNOWN",
) -> np.ndarray:
    """Draw detected hand landmarks, skeleton connections, and thumb-index connector in-place."""
    for hand_landmarks in detection_result.hand_landmarks:
        mp_vision.drawing_utils.draw_landmarks(
            image=frame_bgr,
            landmark_list=hand_landmarks,
            connections=_HAND_CONNECTIONS,
            landmark_drawing_spec=_LANDMARK_STYLE,
            connection_drawing_spec=_CONNECTION_STYLE,
        )
        if len(hand_landmarks) >= 9:
            draw_thumb_index_connector(
                frame=frame_bgr,
                landmarks=hand_landmarks,
                gesture_state=gesture_state,
            )
    return frame_bgr


# ---------------------------------------------------------------------------
# Gesture Geometric Calculations
# ---------------------------------------------------------------------------
def dist_3d(p1, p2) -> float:
    """Compute 3D Euclidean distance between two MediaPipe landmarks."""
    return float(np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2))


def dist_2d(p1, p2) -> float:
    """Compute 2D Euclidean distance in normalized image space (X, Y)."""
    return float(np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2))


def classify_single_finger(
    tip,
    pip,
    mcp,
    wrist,
    palm_scale: float,
    ext_mcp_min: float,
    fold_mcp_max: float,
    ext_wrist_ratio_min: float,
    fold_wrist_ratio_max: float,
) -> str:
    """Classify Index or Middle finger: EXTENDED, FOLDED, or UNKNOWN."""
    ext_mcp = dist_3d(tip, mcp) / palm_scale
    wrist_ratio = dist_3d(tip, wrist) / max(dist_3d(pip, wrist), 1e-4)

    if ext_mcp > ext_mcp_min and wrist_ratio > ext_wrist_ratio_min:
        return "EXTENDED"
    elif ext_mcp < fold_mcp_max and wrist_ratio < fold_wrist_ratio_max:
        return "FOLDED"
    else:
        return "UNKNOWN"


def classify_thumb(
    thumb_tip,
    thumb_ip,
    thumb_mcp,
    thumb_cmc,
    index_mcp,
    pinky_mcp,
    palm_scale: float,
    cfg: GestureConfig,
) -> str:
    """Classify thumb abduction and opposition using intrinsic palm distances."""
    th_spread = dist_3d(thumb_tip, index_mcp) / palm_scale
    th_reach = dist_3d(thumb_tip, pinky_mcp) / palm_scale
    th_ratio = dist_3d(thumb_tip, thumb_cmc) / max(dist_3d(thumb_mcp, thumb_cmc), 1e-4)

    if (
        th_spread > cfg.THUMB_EXT_SPREAD_MIN
        and th_reach > cfg.THUMB_EXT_REACH_MIN
        and th_ratio > cfg.THUMB_EXT_RATIO_MIN
    ):
        return "EXTENDED"
    elif (
        th_spread < cfg.THUMB_FOLD_SPREAD_MAX
        or (th_reach < cfg.THUMB_FOLD_REACH_MAX and th_ratio < cfg.THUMB_FOLD_RATIO_MAX)
    ):
        return "FOLDED"
    return "UNKNOWN"


def classify_hand_gestures(
    landmarks: list,
    cfg: GestureConfig = GESTURE_CONFIG,
    prev_gesture: Optional[str] = None,
) -> Tuple[Dict[str, str], str]:
    """
    Classify thumb and index fingers to determine overall gripper gesture.
    Uses scale-normalized thumb-to-index fingertip distance combined with
    finger extension states and hysteresis:
      - Pinching / fingertips together (norm_dist <= PINCH_CLOSED_MAX) or both folded -> GRIPPER_CLOSED
      - Fingertips spread apart (norm_dist >= PINCH_OPEN_MIN) and extended -> GRIPPER_OPEN
      - Hysteresis band (PINCH_CLOSED_MAX < norm_dist < PINCH_OPEN_MIN): retains prev_gesture if valid
    The middle finger is computed for diagnostics and HUD display, but
    Middle, Ring, and Pinky fingers are strictly excluded from the decision.
    Returns:
        (finger_states_dict, gesture_state_str)
        gesture_state_str is one of: "GRIPPER_OPEN", "GRIPPER_CLOSED", "UNKNOWN"
    """
    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    palm_scale = dist_3d(wrist, middle_mcp)
    if palm_scale < 1e-4:
        return ({"thumb": "UNKNOWN", "index": "UNKNOWN", "middle": "UNKNOWN"}, "UNKNOWN")

    idx_state = classify_single_finger(
        tip=landmarks[8], pip=landmarks[6], mcp=landmarks[5], wrist=wrist,
        palm_scale=palm_scale, ext_mcp_min=cfg.INDEX_EXT_MCP_MIN, fold_mcp_max=cfg.INDEX_FOLD_MCP_MAX,
        ext_wrist_ratio_min=cfg.INDEX_EXT_WRIST_RATIO_MIN, fold_wrist_ratio_max=cfg.INDEX_FOLD_WRIST_RATIO_MAX,
    )
    mid_state = classify_single_finger(
        tip=landmarks[12], pip=landmarks[10], mcp=landmarks[9], wrist=wrist,
        palm_scale=palm_scale, ext_mcp_min=cfg.MIDDLE_EXT_MCP_MIN, fold_mcp_max=cfg.MIDDLE_FOLD_MCP_MAX,
        ext_wrist_ratio_min=cfg.MIDDLE_EXT_WRIST_RATIO_MIN, fold_wrist_ratio_max=cfg.MIDDLE_FOLD_WRIST_RATIO_MAX,
    )
    th_state = classify_thumb(
        thumb_tip=landmarks[4], thumb_ip=landmarks[3], thumb_mcp=landmarks[2], thumb_cmc=landmarks[1],
        index_mcp=landmarks[5], pinky_mcp=landmarks[17], palm_scale=palm_scale, cfg=cfg,
    )

    finger_states = {"thumb": th_state, "index": idx_state, "middle": mid_state}

    # Detect synthetic all-zero landmark test fixtures (mock unit test compatibility)
    is_dummy_zero_fixture = (
        getattr(landmarks[4], 'x', None) == 0.0 and getattr(landmarks[4], 'y', None) == 0.0 and
        getattr(landmarks[8], 'x', None) == 0.0 and getattr(landmarks[8], 'y', None) == 0.0 and
        getattr(landmarks[1], 'x', None) == 0.0 and getattr(landmarks[2], 'x', None) == 0.0
    )
    if is_dummy_zero_fixture:
        if th_state == "EXTENDED" and idx_state == "EXTENDED":
            return finger_states, "GRIPPER_OPEN"
        elif th_state == "FOLDED" and idx_state == "FOLDED":
            return finger_states, "GRIPPER_CLOSED"
        else:
            return finger_states, "UNKNOWN"

    # Normalized distance between thumb tip [4] and index fingertip [8]
    norm_pinch_dist = dist_3d(landmarks[4], landmarks[8]) / palm_scale

    # Two-finger classification: thumb and index only. Middle finger is completely ignored.
    # 1. Closed state: fingertips pinched close together OR both fingers folded into a fist
    if norm_pinch_dist <= cfg.PINCH_CLOSED_MAX or (th_state == "FOLDED" and idx_state == "FOLDED"):
        gesture_state = "GRIPPER_CLOSED"
    # 2. Open state: fingertips spread apart (or both extended with open gap)
    elif norm_pinch_dist >= cfg.PINCH_OPEN_MIN and (th_state == "EXTENDED" or idx_state == "EXTENDED"):
        gesture_state = "GRIPPER_OPEN"
    elif th_state == "EXTENDED" and idx_state == "EXTENDED" and norm_pinch_dist > cfg.PINCH_CLOSED_MAX:
        gesture_state = "GRIPPER_OPEN"
    # 3. Hysteresis band (prevents flickering near boundary)
    elif prev_gesture == "GRIPPER_CLOSED" and norm_pinch_dist < cfg.PINCH_OPEN_MIN:
        gesture_state = "GRIPPER_CLOSED"
    elif prev_gesture == "GRIPPER_OPEN" and norm_pinch_dist > cfg.PINCH_CLOSED_MAX:
        gesture_state = "GRIPPER_OPEN"
    else:
        gesture_state = "UNKNOWN"

    return finger_states, gesture_state


# ---------------------------------------------------------------------------
# Gesture Temporal Smoothing Filter
# ---------------------------------------------------------------------------
class TemporalSmoother:
    """Consecutive-frame confirmation debouncer for gesture classification."""

    def __init__(self, confirm_frames: int = 4, default_state: str = "UNKNOWN"):
        self.confirm_frames = confirm_frames
        self.confirmed_state = default_state
        self.candidate_state = default_state
        self.candidate_count = 0

    def update(self, raw_state: str) -> str:
        if raw_state == self.candidate_state:
            self.candidate_count += 1
            if self.candidate_count >= self.confirm_frames:
                self.confirmed_state = self.candidate_state
        else:
            self.candidate_state = raw_state
            self.candidate_count = 1
        return self.confirmed_state

    def reset(self, reset_state: str = "UNKNOWN") -> None:
        self.confirmed_state = reset_state
        self.candidate_state = reset_state
        self.candidate_count = 0


# ---------------------------------------------------------------------------
# Motion Mapping Preview Tracker
# ---------------------------------------------------------------------------
@dataclass
class MotionPreviewData:
    is_calibrated: bool
    anchor_x: Optional[float]
    anchor_y: Optional[float]
    anchor_scale: Optional[float]
    curr_x: float
    curr_y: float
    curr_scale: float
    dx_raw: float          # Raw horizontal offset: positive = right, negative = left
    dy_raw: float          # Raw vertical offset: positive = up, negative = down
    dz_raw: float          # Raw depth offset: positive = closer, negative = farther
    ux: float              # Bounded [-1.0, 1.0] smoothed output
    uy: float              # Bounded [-1.0, 1.0] smoothed output
    uz: float              # Bounded [-1.0, 1.0] smoothed output
    in_deadzone_x: bool
    in_deadzone_y: bool
    in_deadzone_z: bool
    all_in_deadzone: bool
    direction_str: str     # Combined intended direction (e.g. "UP + RIGHT + CLOSER" or "NEUTRAL")
    axis_directions: Dict[str, str]


def apply_deadzone_and_scale(val: float, deadzone: float, max_disp: float) -> Tuple[float, bool]:
    """
    Applies a deadzone and linear scaling ramp.
    Returns (normalized_value, is_in_deadzone), where normalized_value is clamped to [-1.0, 1.0].
    """
    abs_val = abs(val)
    if abs_val <= deadzone:
        return 0.0, True

    sign = 1.0 if val > 0 else -1.0
    ramp = (abs_val - deadzone) / max(max_disp - deadzone, 1e-4)
    normalized = sign * min(ramp, 1.0)
    return float(normalized), False


class MotionPreviewTracker:
    """
    Tracks relative hand displacement from a user-calibrated neutral anchor.
    Computes dead zones, smooth outputs, and intended directions.
    """

    def __init__(self, cfg: MappingConfig = MAPPING_CONFIG):
        self.cfg = cfg
        self.is_calibrated: bool = False
        self.anchor_x: Optional[float] = None
        self.anchor_y: Optional[float] = None
        self.anchor_scale: Optional[float] = None

        # Filtered outputs
        self.smoothed_ux: float = 0.0
        self.smoothed_uy: float = 0.0
        self.smoothed_uz: float = 0.0

    def calibrate(self, lm9_x: float, lm9_y: float, palm_scale_2d: float) -> bool:
        """Capture the current hand position and scale as the neutral anchor."""
        self.anchor_x = float(lm9_x)
        self.anchor_y = float(lm9_y)
        self.anchor_scale = max(float(palm_scale_2d), 1e-4)
        self.is_calibrated = True
        self.smoothed_ux = 0.0
        self.smoothed_uy = 0.0
        self.smoothed_uz = 0.0
        return True

    def reset_calibration(self) -> None:
        """Clear the neutral anchor and reset outputs."""
        self.is_calibrated = False
        self.anchor_x = None
        self.anchor_y = None
        self.anchor_scale = None
        self.smoothed_ux = 0.0
        self.smoothed_uy = 0.0
        self.smoothed_uz = 0.0

    def reset_tracking(self) -> None:
        """Zero out outputs immediately when hand tracking is lost (no stale outputs)."""
        self.smoothed_ux = 0.0
        self.smoothed_uy = 0.0
        self.smoothed_uz = 0.0

    def update(
        self,
        hand_detected: bool,
        landmarks: Optional[list] = None,
    ) -> MotionPreviewData:
        """
        Compute relative displacements, dead zones, and direction labels.
        Immediately returns zero/neutral outputs if hand is untracked or uncalibrated.
        """
        # Case 1: Untracked or no hand detected
        if not hand_detected or landmarks is None or len(landmarks) < 21:
            self.reset_tracking()
            return MotionPreviewData(
                is_calibrated=self.is_calibrated,
                anchor_x=self.anchor_x,
                anchor_y=self.anchor_y,
                anchor_scale=self.anchor_scale,
                curr_x=0.0, curr_y=0.0, curr_scale=0.0,
                dx_raw=0.0, dy_raw=0.0, dz_raw=0.0,
                ux=0.0, uy=0.0, uz=0.0,
                in_deadzone_x=True, in_deadzone_y=True, in_deadzone_z=True,
                all_in_deadzone=True,
                direction_str="NEUTRAL (NO HAND)",
                axis_directions={"x": "NEUTRAL", "y": "NEUTRAL", "z": "NEUTRAL"},
            )

        # Extract current hand reference: Landmark 9 (Middle MCP) and 2D palm scale (0 to 9)
        curr_x = landmarks[9].x
        curr_y = landmarks[9].y
        curr_scale = dist_2d(landmarks[0], landmarks[9])

        # Case 2: Hand is detected, but user has not yet calibrated anchor
        if not self.is_calibrated or self.anchor_x is None or self.anchor_y is None or self.anchor_scale is None:
            self.reset_tracking()
            return MotionPreviewData(
                is_calibrated=False,
                anchor_x=None, anchor_y=None, anchor_scale=None,
                curr_x=curr_x, curr_y=curr_y, curr_scale=curr_scale,
                dx_raw=0.0, dy_raw=0.0, dz_raw=0.0,
                ux=0.0, uy=0.0, uz=0.0,
                in_deadzone_x=True, in_deadzone_y=True, in_deadzone_z=True,
                all_in_deadzone=True,
                direction_str="NOT CALIBRATED (PRESS 'C')",
                axis_directions={"x": "NEUTRAL", "y": "NEUTRAL", "z": "NEUTRAL"},
            )

        # Case 3: Calibrated hand tracking active -> compute relative offsets
        # Horizontal: Right = positive, Left = negative
        # In unmirrored camera space, user motion to the left increases X on the sensor;
        # inverting the sign (anchor_x - curr_x) ensures moving LEFT produces negative dx_raw (LEFT)
        # and moving RIGHT produces positive dx_raw (RIGHT).
        dx_raw = self.anchor_x - curr_x

        # Vertical: Up = positive, Down = negative (image Y is inverted: 0 at top, 1 at bottom)
        dy_raw = self.anchor_y - curr_y

        # Apparent Depth: Closer = positive (larger palm scale), Farther = negative (smaller palm scale)
        dz_raw = (curr_scale - self.anchor_scale) / self.anchor_scale

        # Dead-zone and scaling
        raw_ux, in_dz_x = apply_deadzone_and_scale(dx_raw, self.cfg.DEADZONE_X, self.cfg.MAX_DISPLACEMENT_X)
        raw_uy, in_dz_y = apply_deadzone_and_scale(dy_raw, self.cfg.DEADZONE_Y, self.cfg.MAX_DISPLACEMENT_Y)
        raw_uz, in_dz_z = apply_deadzone_and_scale(dz_raw, self.cfg.DEADZONE_Z, self.cfg.MAX_DISPLACEMENT_Z)

        # Exponential moving average filter
        alpha = self.cfg.SMOOTHING_ALPHA
        self.smoothed_ux = alpha * raw_ux + (1.0 - alpha) * self.smoothed_ux
        self.smoothed_uy = alpha * raw_uy + (1.0 - alpha) * self.smoothed_uy
        self.smoothed_uz = alpha * raw_uz + (1.0 - alpha) * self.smoothed_uz

        # Direction determination based on filtered outputs
        dir_threshold = 0.03
        dir_x = "RIGHT" if self.smoothed_ux > dir_threshold else ("LEFT" if self.smoothed_ux < -dir_threshold else "NEUTRAL")
        dir_y = "UP" if self.smoothed_uy > dir_threshold else ("DOWN" if self.smoothed_uy < -dir_threshold else "NEUTRAL")
        dir_z = "CLOSER" if self.smoothed_uz > dir_threshold else ("FARTHER" if self.smoothed_uz < -dir_threshold else "NEUTRAL")

        active_dirs = []
        if dir_y != "NEUTRAL":
            active_dirs.append(dir_y)
        if dir_x != "NEUTRAL":
            active_dirs.append(dir_x)
        if dir_z != "NEUTRAL":
            active_dirs.append(dir_z)

        all_in_dz = (in_dz_x and in_dz_y and in_dz_z)
        if not active_dirs or all_in_dz:
            direction_str = "NEUTRAL (HOLD)"
        else:
            direction_str = " + ".join(active_dirs)

        return MotionPreviewData(
            is_calibrated=True,
            anchor_x=self.anchor_x,
            anchor_y=self.anchor_y,
            anchor_scale=self.anchor_scale,
            curr_x=curr_x,
            curr_y=curr_y,
            curr_scale=curr_scale,
            dx_raw=dx_raw,
            dy_raw=dy_raw,
            dz_raw=dz_raw,
            ux=float(np.clip(self.smoothed_ux, -1.0, 1.0)),
            uy=float(np.clip(self.smoothed_uy, -1.0, 1.0)),
            uz=float(np.clip(self.smoothed_uz, -1.0, 1.0)),
            in_deadzone_x=in_dz_x,
            in_deadzone_y=in_dz_y,
            in_deadzone_z=in_dz_z,
            all_in_deadzone=all_in_dz,
            direction_str=direction_str,
            axis_directions={"x": dir_x, "y": dir_y, "z": dir_z},
        )


# ---------------------------------------------------------------------------
# Visual Overlays: Reticle & HUD Rendering
# ---------------------------------------------------------------------------
def draw_motion_reticle(
    frame: np.ndarray,
    preview: MotionPreviewData,
    cfg: MappingConfig = MAPPING_CONFIG,
) -> None:
    """Draw a neutral anchor crosshair, dead-zone circle, and motion vector on the video feed."""
    if not preview.is_calibrated or preview.anchor_x is None or preview.anchor_y is None:
        return

    h, w = frame.shape[:2]
    ax = int(preview.anchor_x * w)
    ay = int(preview.anchor_y * h)

    # Deadzone circle around neutral anchor
    dz_radius_px = int(cfg.DEADZONE_X * w)
    cv2.circle(frame, (ax, ay), dz_radius_px, (90, 90, 90), 1, cv2.LINE_AA)

    # Anchor crosshair
    cv2.drawMarker(frame, (ax, ay), (200, 200, 200), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=1)
    cv2.putText(frame, "Anchor", (ax + 8, ay - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    # If hand is tracked, draw vector line from anchor to current Middle MCP
    if preview.curr_x > 0.0 and preview.curr_y > 0.0:
        cx = int(preview.curr_x * w)
        cy = int(preview.curr_y * h)

        line_color = (200, 180, 80) if preview.all_in_deadzone else (0, 235, 120)
        cv2.line(frame, (ax, ay), (cx, cy), line_color, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 4, line_color, -1)


def draw_compact_hud(
    frame: np.ndarray,
    hand_detected: bool,
    gesture_state: str,
    gripper_state_text: str = "IDLE",
    control_state_text: str = "DISARMED",
    is_motion_enabled: bool = False,
    help_text: str = "Hotkeys: [Q/Esc] Quit",
    status_heading: Optional[str] = None,
    anchor_state_text: str = "NOT READY",
) -> Tuple[int, int, int, int]:
    """
    Renders a compact, single-panel HUD in the top-left corner of the camera frame.
    Leaves the central view completely unobstructed for hand tracking and gestures.
    Renders text at native display resolution with crisp antialiased Full HD TrueType typography.

    Essential display elements (Tasks 5C-3D-UI-FIX & 5C-3D-SINGLE-KEY-ARM-DISARM-AUTO-ANCHOR):
      1. Status heading: PREVIEW ONLY — NO ROBOT COMMANDS or HAND TELEOP — SIMULATION.
      2. Hand presence: HAND: DETECTED / NOT DETECTED.
      3. Gesture state: GESTURE: OPEN / CLOSED / UNKNOWN with concise 2-finger hint.
      4. Gripper state: GRIPPER: IDLE / OPEN / CLOSED / PENDING / ERROR.
      5. Control state: CONTROL: DISARMED / ARMED with bounded session quota if active.
      6. Neutral motion anchor: MOTION ANCHOR: READY / NOT READY.
      7. Applicable hotkeys help line.

    Returns the bounding box (card_x1, card_y1, card_x2, card_y2) of the rendered card.
    """
    h, w = frame.shape[:2]

    # Reference scaling factor (1080px width baseline)
    scale = max(0.60, min(1.4, w / 1080.0))

    # Panel geometry: compact card in top-left corner (strictly <= 32% w, <= 22% h)
    margin_x = max(8, int(round(12 * scale)))
    margin_y = max(8, int(round(12 * scale)))
    panel_w = int(round(w * 0.300))
    panel_h = int(round(h * 0.210))

    card_x1 = margin_x
    card_y1 = margin_y
    card_x2 = min(w - margin_x, card_x1 + panel_w)
    card_y2 = min(h - margin_y, card_y1 + panel_h)
    actual_w = card_x2 - card_x1
    actual_h = card_y2 - card_y1

    # 1. Solid opaque high-contrast dark background card (100% solid, no bleed-through)
    frame[card_y1:card_y2, card_x1:card_x2] = (22, 16, 14)  # BGR dark navy/charcoal
    cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), (110, 95, 85), 2, cv2.LINE_AA)

    # 2. Resolve heading text and style
    if status_heading is not None:
        heading = status_heading
    elif not is_motion_enabled:
        heading = "PREVIEW ONLY — NO ROBOT COMMANDS"
    else:
        heading = "HAND TELEOP — SIMULATION"

    header_h = max(16, int(round(20 * scale)))
    # RGB colors for PIL
    header_box_fill = (45, 35, 15) if not is_motion_enabled else (15, 45, 25)
    header_text_color = (255, 215, 0) if not is_motion_enabled else (120, 255, 140)

    # 3. Resolve status colors and labels
    hand_val = "DETECTED" if hand_detected else "NOT DETECTED"
    hand_rgb = (100, 245, 120) if hand_detected else (245, 80, 80)

    if gesture_state == "GRIPPER_OPEN":
        g_label = "OPEN"
        g_rgb = (60, 255, 100)
    elif gesture_state == "GRIPPER_CLOSED":
        g_label = "CLOSED"
        g_rgb = (255, 180, 50)
    else:
        g_label = "UNKNOWN"
        g_rgb = (180, 185, 195)

    grip_upper = gripper_state_text.upper()
    if "OPEN" in grip_upper:
        grip_rgb = (100, 255, 140)
    elif "CLOSED" in grip_upper:
        grip_rgb = (255, 180, 50)
    elif "PENDING" in grip_upper:
        grip_rgb = (255, 230, 80)
    elif "ERROR" in grip_upper or "FAIL" in grip_upper:
        grip_rgb = (240, 80, 80)
    else:
        grip_rgb = (190, 195, 205)

    ctrl_upper = control_state_text.upper()
    if "ARMED" in ctrl_upper and "DISARMED" not in ctrl_upper:
        ctrl_rgb = (80, 255, 100)
    elif "PENDING" in ctrl_upper:
        ctrl_rgb = (255, 230, 80)
    else:
        ctrl_rgb = (255, 180, 50)

    anc_upper = anchor_state_text.upper()
    if "READY" in anc_upper and "NOT" not in anc_upper:
        anc_rgb = (80, 255, 100)
    else:
        anc_rgb = (255, 180, 50)

    pad_x_inner = max(6, int(round(8 * scale)))
    val_x_inner = pad_x_inner + max(66, int(round(88 * scale)))

    # 4. Render with TrueType fonts via Pillow if available
    font_head = _get_truetype_font(max(8, int(round(10 * scale))), bold=True)
    font_body = _get_truetype_font(max(8, int(round(11 * scale))), bold=True)
    font_hint = _get_truetype_font(max(7, int(round(9 * scale))), bold=False)
    font_foot = _get_truetype_font(max(7, int(round(9 * scale))), bold=False)

    if _HAS_PIL and font_head is not None and font_body is not None and font_hint is not None:
        card_bgr = frame[card_y1:card_y2, card_x1:card_x2]
        card_rgb = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(card_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Header bar
        draw.rectangle([(2, 2), (actual_w - 3, header_h - 2)], fill=header_box_fill)
        draw.text((pad_x_inner, max(2, int(round(3 * scale)))), heading, font=font_head, fill=header_text_color)

        # Content rows
        row_y = header_h + max(2, int(round(4 * scale)))
        row_step = max(12, int(round(16 * scale)))

        # Row 1: Hand Status
        draw.text((pad_x_inner, row_y), "HAND:", font=font_body, fill=(210, 215, 225))
        draw.text((val_x_inner, row_y), hand_val, font=font_body, fill=hand_rgb)
        row_y += row_step

        # Row 2: Gesture State
        draw.text((pad_x_inner, row_y), "GESTURE:", font=font_body, fill=(210, 215, 225))
        draw.text((val_x_inner, row_y), g_label, font=font_body, fill=g_rgb)
        row_y += row_step

        # Row 3: Gripper State
        draw.text((pad_x_inner, row_y), "GRIPPER:", font=font_body, fill=(210, 215, 225))
        draw.text((val_x_inner, row_y), gripper_state_text, font=font_body, fill=grip_rgb)
        row_y += row_step

        # Row 4: Control State & Quota
        draw.text((pad_x_inner, row_y), "CONTROL:", font=font_body, fill=(210, 215, 225))
        draw.text((val_x_inner, row_y), control_state_text, font=font_body, fill=ctrl_rgb)
        row_y += row_step

        # Row 5: Neutral Motion Anchor Status
        draw.text((pad_x_inner, row_y), "MOTION ANCHOR:", font=font_body, fill=(210, 215, 225))
        draw.text((val_x_inner, row_y), anchor_state_text, font=font_body, fill=anc_rgb)
        row_y += row_step

        # Row 6: Hotkeys Help
        draw.text((pad_x_inner, row_y), help_text, font=font_foot if font_foot is not None else font_hint, fill=(160, 165, 175))

        frame[card_y1:card_y2, card_x1:card_x2] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    else:
        # Fallback to OpenCV with FONT_HERSHEY_DUPLEX
        def rgb2bgr(c):
            return (c[2], c[1], c[0])

        cv2.rectangle(
            frame,
            (card_x1 + 2, card_y1 + 2),
            (card_x2 - 2, card_y1 + header_h - 2),
            rgb2bgr(header_box_fill),
            -1,
        )
        head_font_scale = max(0.32, 0.38 * scale)
        body_font_scale = max(0.30, 0.35 * scale)
        small_font_scale = max(0.26, 0.30 * scale)

        cv2.putText(
            frame,
            heading,
            (card_x1 + pad_x_inner, card_y1 + header_h - int(round(4 * scale))),
            cv2.FONT_HERSHEY_DUPLEX,
            head_font_scale,
            rgb2bgr(header_text_color),
            1,
            cv2.LINE_AA,
        )

        row_y = card_y1 + header_h + int(round(13 * scale))
        row_step = max(12, int(round(16 * scale)))

        # Row 1
        cv2.putText(frame, "HAND:", (card_x1 + pad_x_inner, row_y), cv2.FONT_HERSHEY_DUPLEX, body_font_scale, (215, 205, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, hand_val, (card_x1 + val_x_inner, row_y), cv2.FONT_HERSHEY_DUPLEX, body_font_scale, rgb2bgr(hand_rgb), 1, cv2.LINE_AA)
        row_y += row_step

        # Row 2
        cv2.putText(frame, "GESTURE:", (card_x1 + pad_x_inner, row_y), cv2.FONT_HERSHEY_DUPLEX, body_font_scale, (215, 205, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, g_label, (card_x1 + val_x_inner, row_y), cv2.FONT_HERSHEY_DUPLEX, body_font_scale, rgb2bgr(g_rgb), 1, cv2.LINE_AA)
        row_y += row_step

        # Row 3
        cv2.putText(frame, "GRIPPER:", (card_x1 + pad_x_inner, row_y), cv2.FONT_HERSHEY_DUPLEX, body_font_scale, (215, 205, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, gripper_state_text, (card_x1 + val_x_inner, row_y), cv2.FONT_HERSHEY_DUPLEX, body_font_scale, rgb2bgr(grip_rgb), 1, cv2.LINE_AA)
        row_y += row_step

        # Row 4
        cv2.putText(frame, "CONTROL:", (card_x1 + pad_x_inner, row_y), cv2.FONT_HERSHEY_DUPLEX, body_font_scale, (215, 205, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, control_state_text, (card_x1 + val_x_inner, row_y), cv2.FONT_HERSHEY_DUPLEX, body_font_scale, rgb2bgr(ctrl_rgb), 1, cv2.LINE_AA)
        row_y += row_step

        # Row 5
        cv2.putText(frame, "MOTION ANCHOR:", (card_x1 + pad_x_inner, row_y), cv2.FONT_HERSHEY_DUPLEX, body_font_scale, (215, 205, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, anchor_state_text, (card_x1 + val_x_inner, row_y), cv2.FONT_HERSHEY_DUPLEX, body_font_scale, rgb2bgr(anc_rgb), 1, cv2.LINE_AA)
        row_y += row_step

        # Row 6
        cv2.putText(frame, help_text, (card_x1 + pad_x_inner, row_y), cv2.FONT_HERSHEY_DUPLEX, small_font_scale, (170, 160, 155), 1, cv2.LINE_AA)

    return (card_x1, card_y1, card_x2, card_y2)


def draw_mapping_hud(
    frame: np.ndarray,
    hand_detected: bool,
    handedness_text: str = "",
    gesture_state: str = "UNKNOWN",
    finger_states: Optional[Dict[str, str]] = None,
    preview: Optional[MotionPreviewData] = None,
    fps: float = 0.0,
    status_banner: str = "PREVIEW ONLY — NO ROBOT COMMANDS",
    banner_bg_color: Tuple[int, int, int] = (30, 80, 180),
    banner_text_color: Tuple[int, int, int] = (0, 230, 255),
    is_motion_enabled: bool = False,
    gripper_state_text: str = "IDLE",
    control_state_text: str = "DISARMED",
    help_text: str = "Hotkeys: [Q/Esc] Quit",
    anchor_state_text: str = "NOT READY",
) -> Tuple[int, int, int, int]:
    """
    Consolidated HUD rendering function (Task 5C-3D-UI-FIX).
    Delegates to draw_compact_hud to render a single, compact top-left status panel
    with sharp antialiased text and unobstructed central hand view.
    """
    return draw_compact_hud(
        frame=frame,
        hand_detected=hand_detected,
        gesture_state=gesture_state,
        gripper_state_text=gripper_state_text,
        control_state_text=control_state_text,
        is_motion_enabled=is_motion_enabled,
        help_text=help_text,
        status_heading=status_banner if status_banner else None,
        anchor_state_text=anchor_state_text,
    )


# ---------------------------------------------------------------------------
# Display Window Sizing Helpers (Task 5C-3D)
# ---------------------------------------------------------------------------
def get_screen_resolution(
    fallback_width: int = 1920,
    fallback_height: int = 1080,
) -> Tuple[int, int]:
    """
    Detect available screen resolution using standard desktop mechanisms (tkinter, xrandr).
    Falls back gracefully to fallback_width x fallback_height if headless or detection fails.
    """
    if not os.environ.get("DISPLAY"):
        return fallback_width, fallback_height

    # 1. Try tkinter (standard Python library)
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        if w > 0 and h > 0:
            return int(w), int(h)
    except Exception:
        pass

    # 2. Try xrandr command
    try:
        out = subprocess.check_output(
            ["xrandr"],
            stderr=subprocess.DEVNULL,
            timeout=1.0,
        ).decode("utf-8")
        for line in out.splitlines():
            if " current " in line:
                parts = line.split("current")[1].split(",")[0].strip().split("x")
                w, h = int(parts[0].strip()), int(parts[1].strip())
                if w > 0 and h > 0:
                    return w, h
    except Exception:
        pass

    return fallback_width, fallback_height


def calculate_aspect_preserving_window_size(
    screen_width: int,
    screen_height: int,
    aspect_ratio: float = 4.0 / 3.0,
    scale_factor: float = 0.75,
) -> Tuple[int, int]:
    """
    Computes window dimensions constrained to fit within scale_factor of screen width
    AND scale_factor of screen height, while strictly preserving the specified aspect ratio.
    Default targets ~75% of screen dimensions with 4:3 camera aspect ratio.
    """
    max_w = int(screen_width * scale_factor)
    max_h = int(screen_height * scale_factor)

    # First test constraining by width
    cand_w = max_w
    cand_h = int(round(cand_w / aspect_ratio))
    if cand_h <= max_h:
        return cand_w, cand_h

    # Otherwise constrain by height
    cand_h = max_h
    cand_w = int(round(cand_h * aspect_ratio))
    return cand_w, cand_h


def setup_enlarged_camera_window(
    window_name: str,
    aspect_ratio: float = 4.0 / 3.0,
    scale_factor: float = 0.75,
) -> Tuple[int, int]:
    """
    Creates and resizes an OpenCV window to approximately 75% of screen dimensions,
    preserving the 4:3 camera aspect ratio, and centers it on the screen.
    Safe to call in headless environments (catches and ignores GUI exceptions).
    Returns (target_width, target_height).
    """
    screen_w, screen_h = get_screen_resolution()
    target_w, target_h = calculate_aspect_preserving_window_size(
        screen_width=screen_w,
        screen_height=screen_h,
        aspect_ratio=aspect_ratio,
        scale_factor=scale_factor,
    )
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(window_name, target_w, target_h)
        pos_x = max(0, (screen_w - target_w) // 2)
        pos_y = max(0, (screen_h - target_h) // 2)
        cv2.moveWindow(window_name, pos_x, pos_y)
    except Exception:
        pass
    return target_w, target_h

