#!/usr/bin/env python3
"""
hand_tracking_demo.py
=====================
Phase 2 & 5A – Hand-Gesture & Motion-Mapping Preview Prototype.
Part of the hand_gesture_teleop package for the SO-101 robotic arm project.

Objective:
  1. Captures frames from the laptop webcam using OpenCV.
  2. Tracks 21 hand landmarks in real time via MediaPipe Tasks API (LIVE_STREAM mode).
  3. Classifies two-finger gesture states (Thumb, Index) into:
       - GRIPPER_OPEN   : Thumb and Index are EXTENDED (Middle ignored).
       - GRIPPER_CLOSED : Thumb and Index are FOLDED (Middle ignored).
       - UNKNOWN        : Any other pose, disagreement, transition, or no hand.
     (Middle, Ring, and Pinky fingers are strictly excluded from gesture decision).
  4. Neutral Hand Calibration (Task 5A):
       - User presses 'C' to capture current hand position as the neutral anchor.
       - User presses 'R' to reset the calibration.
  5. Hand-to-Robot Motion Mapping Preview:
       - Computes relative displacements:
           * Horizontal (dX) : Left (-) / Right (+)
           * Vertical (dY)   : Down (-) / Up (+)  [inverted from image Y]
           * Depth (dZ)      : Farther (-) / Closer (+) [approx 2D palm-scale change]
       - Applies configurable dead zones (~5%) to prevent jitter when holding steady.
       - Applies conservative linear scaling and exponential smoothing.
       - Outputs bounded normalized preview values in range [-1.0, +1.0].
       - Identifies intended directional commands (UP, DOWN, LEFT, RIGHT, CLOSER, FARTHER, NEUTRAL).
  6. Visual-Only Safety:
       - Strictly a display preview: NO ROS 2 motion topics, NO robot actions, NO serial communication.
       - Prominent on-screen warning: "PREVIEW ONLY — NO ROBOT COMMANDS".
       - Immediate zeroing of outputs if tracking is lost or hand leaves frame.

Usage (activate virtual environment first):
    source /home/santosh/lerobot_ws/.venv_hand_gesture/bin/activate
    python3 /home/santosh/lerobot_ws/src/hand_gesture_teleop/scripts/hand_tracking_demo.py

Keyboard Controls:
    C       : Calibrate / capture current hand pose as neutral anchor
    R       : Reset calibration anchor
    Q / Esc : Exit cleanly and release camera
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_MODEL_PATH = os.path.normpath(
    os.path.join(_SCRIPT_DIR, "..", "models", "hand_landmarker.task")
)

# ---------------------------------------------------------------------------
# Import Reusable Hand-Tracking & Mapping Logic
# ---------------------------------------------------------------------------
# Allow direct script execution from source directory
_PKG_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from hand_gesture_teleop.hand_tracking import (
    GESTURE_CONFIG,
    MAPPING_CONFIG,
    GestureConfig,
    MappingConfig,
    MotionPreviewData,
    MotionPreviewTracker,
    TemporalSmoother,
    classify_hand_gestures,
    dist_2d,
    draw_compact_hud,
    draw_landmarks_on_frame,
    draw_mapping_hud,
    draw_motion_reticle,
    setup_enlarged_camera_window,
)



# ---------------------------------------------------------------------------
# Main Application Loop
# ---------------------------------------------------------------------------
def main() -> None:
    # ── 1. Verify model asset ─────────────────────────────────────────────
    if not os.path.isfile(_MODEL_PATH):
        print(
            f"\n[ERROR] Hand Landmarker model not found at:\n  {_MODEL_PATH}\n\n"
            "Download it with:\n"
            "  cd /home/santosh/lerobot_ws/src/hand_gesture_teleop\n"
            "  curl -L -o models/hand_landmarker.task \\\n"
            "    https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[INFO] Model: {_MODEL_PATH}")

    # ── 2. Build MediaPipe Hand Landmarker in LIVE_STREAM mode ────────────
    latest_result: dict = {"value": None}

    def _result_callback(
        result: mp_vision.HandLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        latest_result["value"] = result

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=_result_callback,
    )

    print("[INFO] Initialising MediaPipe Hand Landmarker …")
    landmarker = mp_vision.HandLandmarker.create_from_options(options)
    print("[INFO] Hand Landmarker ready.")

    # ── 3. Open webcam ───────────────────────────────────────────────────
    cam_index = 0
    cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print(
            f"\n[ERROR] Cannot open camera at index {cam_index}.\n"
            "Troubleshooting:\n"
            "  ls -l /dev/video*       # verify device exists\n"
            "  groups                  # check permissions\n",
            file=sys.stderr,
        )
        landmarker.close()
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera opened: {actual_w}×{actual_h} (index {cam_index})")
    print("[INFO] Task 5A Display-Only Motion Mapping Preview running.")
    print("[INFO] Controls: [C] Calibrate Anchor | [R] Reset Anchor | [Q/Esc] Quit\n")

    # ── 4. Initialize Smoothers & Motion Tracker ──────────────────────────
    gesture_smoother = TemporalSmoother(confirm_frames=GESTURE_CONFIG.SMOOTHING_CONFIRM_FRAMES)
    thumb_smoother = TemporalSmoother(confirm_frames=GESTURE_CONFIG.SMOOTHING_CONFIRM_FRAMES)
    index_smoother = TemporalSmoother(confirm_frames=GESTURE_CONFIG.SMOOTHING_CONFIRM_FRAMES)
    middle_smoother = TemporalSmoother(confirm_frames=GESTURE_CONFIG.SMOOTHING_CONFIRM_FRAMES)

    motion_tracker = MotionPreviewTracker(cfg=MAPPING_CONFIG)

    frame_count = 0
    fps = 0.0
    fps_window_frames = 30
    fps_timer = time.monotonic()

    # Configure enlarged window (~75% screen dimensions, preserving 4:3 camera aspect ratio)
    window_name = "Hand Gesture Teleop – Motion Mapping Preview"
    target_w, target_h = setup_enlarged_camera_window(window_name)
    print(f"[INFO] Display window configured: {target_w}x{target_h} (preserves 4:3 aspect ratio)")

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                time.sleep(0.033)
                continue

            frame_count += 1

            # ── FPS Calculation ───────────────────────────────────────
            if frame_count % fps_window_frames == 0:
                elapsed = time.monotonic() - fps_timer
                fps = fps_window_frames / elapsed if elapsed > 0 else 0.0
                fps_timer = time.monotonic()

            # ── Send frame to MediaPipe (non-blocking) ─────────────────
            timestamp_ms = int(time.monotonic() * 1000)
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )
            landmarker.detect_async(mp_image, timestamp_ms)

            # ── Retrieve latest result from callback ───────────────────
            result = latest_result["value"]
            hand_detected = result is not None and len(result.hand_landmarks) > 0

            handedness_text = ""
            raw_finger_states = {
                "thumb": "UNKNOWN",
                "index": "UNKNOWN",
                "middle": "UNKNOWN",
            }
            raw_gesture = "UNKNOWN"
            first_hand_landmarks = None

            if hand_detected:
                # Extract Handedness label if available
                if (
                    result.handedness
                    and len(result.handedness) > 0
                    and len(result.handedness[0]) > 0
                ):
                    cat = result.handedness[0][0]
                    cat_name = cat.category_name or "Hand"
                    cat_pct = int(cat.score * 100)
                    handedness_text = f"[{cat_name} {cat_pct}%]"

                # Classify raw finger and gesture states
                first_hand_landmarks = result.hand_landmarks[0]
                raw_finger_states, raw_gesture = classify_hand_gestures(
                    first_hand_landmarks,
                    cfg=GESTURE_CONFIG,
                    prev_gesture=gesture_smoother.confirmed_state,
                )

                # Apply temporal smoothing to reduce flickering
                confirmed_gesture = gesture_smoother.update(raw_gesture)
                confirmed_fingers = {
                    "thumb": thumb_smoother.update(raw_finger_states["thumb"]),
                    "index": index_smoother.update(raw_finger_states["index"]),
                    "middle": middle_smoother.update(raw_finger_states["middle"]),
                }
            else:
                # No hand: reset gesture smoothers to UNKNOWN immediately
                gesture_smoother.reset("UNKNOWN")
                thumb_smoother.reset("UNKNOWN")
                index_smoother.reset("UNKNOWN")
                middle_smoother.reset("UNKNOWN")
                confirmed_gesture = "UNKNOWN"
                confirmed_fingers = raw_finger_states

            # ── Motion Preview Update (Task 5A) ───────────────────────
            preview_data = motion_tracker.update(
                hand_detected=hand_detected,
                landmarks=first_hand_landmarks,
            )

            # ── Display Pipeline (Task 5C-3D-UI-FIX) ────────────────────
            # 1. Create display-only copy scaled to target window (4:3 aspect ratio)
            if (target_w, target_h) != (frame_bgr.shape[1], frame_bgr.shape[0]):
                display_frame = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            else:
                display_frame = frame_bgr.copy()

            # 2. Draw landmarks directly on resized display frame at native resolution
            if hand_detected and result is not None:
                draw_landmarks_on_frame(display_frame, result, gesture_state=confirmed_gesture)

            # 3. Draw single compact HUD in top-left corner
            draw_compact_hud(
                frame=display_frame,
                hand_detected=hand_detected,
                gesture_state=confirmed_gesture,
                gripper_state_text="IDLE (Preview)",
                control_state_text="DISARMED (Preview Mode)",
                is_motion_enabled=False,
                help_text="Hotkeys: [C] Anchor | [R] Reset | [Q/Esc] Quit",
                status_heading="PREVIEW ONLY — NO ROBOT COMMANDS",
            )

            cv2.imshow("Hand Gesture Teleop – Motion Mapping Preview", display_frame)

            # ── Key Handling ───────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):  # Q or Escape
                print("\n[INFO] Exit key pressed. Shutting down …")
                break

            elif key in (ord("c"), ord("C")):    # Calibrate Anchor
                if hand_detected and first_hand_landmarks is not None:
                    lm9 = first_hand_landmarks[9]
                    lm0 = first_hand_landmarks[0]
                    scale_2d = dist_2d(lm0, lm9)
                    motion_tracker.calibrate(lm9.x, lm9.y, scale_2d)
                    print(
                        f"[INFO] Neutral anchor captured: X0={lm9.x:.3f}, Y0={lm9.y:.3f}, Scale0={scale_2d:.3f}"
                    )
                else:
                    print("[WARN] Cannot calibrate: No hand detected in frame. Please show your hand and press 'C'.")

            elif key in (ord("r"), ord("R")):    # Reset Anchor
                motion_tracker.reset_calibration()
                print("[INFO] Calibration reset. Hold hand in comfortable position and press 'C' to set neutral anchor.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print("[INFO] Camera released. Demo finished cleanly.")


if __name__ == "__main__":
    main()
