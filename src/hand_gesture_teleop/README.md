# hand_gesture_teleop

ROS 2 (ament_python) package for real-time hand-gesture teleoperation of the SO-101 robotic arm.

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Vision prototype – webcam + MediaPipe landmarks | ✅ This phase |
| Phase 2 | Gesture → joint-angle mapping (no robot yet) | 🔜 |
| Phase 3 | Live arm control via `arm_controller` action | 🔜 |
| Phase 4 | Gripper control via three-finger gestures | 🔜 |
| Phase 5 | Safety features, smoothing, workspace limits | 🔜 |

## Prerequisites

- Ubuntu 24.04 LTS, ROS 2 Jazzy
- Python 3.12
- Virtual environment at `/home/santosh/lerobot_ws/.venv_hand_gesture`

## Setup

```bash
# Create venv (already done – do not repeat unless venv is missing)
python3 -m venv --system-site-packages /home/santosh/lerobot_ws/.venv_hand_gesture
source /home/santosh/lerobot_ws/.venv_hand_gesture/bin/activate
pip install "mediapipe==0.10.35" "numpy<2" "matplotlib>=3.8"
```

## Run the Phase 1 demo (vision only, no robot)

```bash
source /home/santosh/lerobot_ws/.venv_hand_gesture/bin/activate
python3 /home/santosh/lerobot_ws/src/hand_gesture_teleop/scripts/hand_tracking_demo.py
```

Press **Q** or **Escape** to exit.

## Model asset

File: `models/hand_landmarker.task`  
Source: <https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task>  
Size: ~7.5 MB (float16, MediaPipe Hand Landmarker v1)

## Package structure

```
hand_gesture_teleop/
├── hand_gesture_teleop/        # Python module (ROS nodes go here in later phases)
│   └── __init__.py
├── models/
│   └── hand_landmarker.task    # Official Google MediaPipe model
├── resource/
│   └── hand_gesture_teleop     # ament resource marker
├── scripts/
│   └── hand_tracking_demo.py   # Phase 1 standalone webcam demo
├── package.xml
├── setup.cfg
└── setup.py
```
