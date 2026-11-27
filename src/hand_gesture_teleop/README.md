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

## Run the Live Simulation Teleoperation (2 Terminals)

Make sure no browser tab or other app is using `/dev/video0`.

**Terminal 1 — Simulated SO-101 Arm, MoveIt 2 & RViz:**
```bash
cd /home/santosh/lerobot_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ROS_DOMAIN_ID=10 ros2 launch hand_gesture_teleop so101_sim.launch.py
```

**Terminal 2 — Hand Gesture Teleoperation & Camera Window:**
```bash
cd /home/santosh/lerobot_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ROS_DOMAIN_ID=10 ros2 launch hand_gesture_teleop hand_teleop.launch.py
```

Controls in the **Hand Gesture Teleop** window (`[E]` Enable/Auto-Anchor, `[D]` Disable/Stop, `[C]` Recalibrate, `[R]` Reset, `[Q/Esc]` Quit).
Optional overrides: `enable_vertical_z:=true` to enable vertical Z control.

---

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
├── hand_gesture_teleop/        # ROS 2 node, tracking, and SO-101 kinematics
│   ├── __init__.py
│   ├── hand_teleop_node.py
│   ├── hand_tracking.py
│   └── so101_kinematics.py
├── launch/
│   ├── hand_teleop.launch.py   # Compact teleoperation launch (.venv_hand_gesture)
│   └── so101_sim.launch.py     # Mock hardware + MoveIt 2 + RViz 2 launch
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
