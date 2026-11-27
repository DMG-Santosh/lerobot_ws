# SO-ARM 101 Hand Gesture Teleoperation

A ROS 2 Jazzy project that uses a laptop webcam and MediaPipe hand tracking to control an SO-ARM 101 robot-arm model and its gripper in an isolated ROS 2 mock simulation.

> **Current scope: simulation only.** The documented launch workflow uses `mock_components/GenericSystem` rather than a physical servo/serial hardware driver. It is intended for software development and RViz validation; it is not a physical robot bringup procedure.

## Contents

- [Project overview](#project-overview)
- [What the system does](#what-the-system-does)
- [Hand-to-robot mapping](#hand-to-robot-mapping)
- [How the software is organised](#how-the-software-is-organised)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Before cloning](#before-cloning)
- [Clone the repository](#clone-the-repository)
- [Install ROS package dependencies](#install-ros-package-dependencies)
- [Prepare the workspace](#prepare-the-workspace)
- [Build the package](#build-the-package)
- [Run the simulation and teleoperation](#run-the-simulation-and-teleoperation)
- [Using the hand-gesture controls](#using-the-hand-gesture-controls)
- [Verify the simulation](#verify-the-simulation)
- [Shut everything down](#shut-everything-down)
- [Troubleshooting](#troubleshooting)
- [Run the regression tests](#run-the-regression-tests)
- [Safety notes and limitations](#safety-notes-and-limitations)
- [Updating the repository](#updating-the-repository)

## Project overview

The project connects four main parts:

1. **Camera and computer vision** — OpenCV captures frames from the laptop webcam. MediaPipe Hand Landmarker detects and tracks hand landmarks.
2. **Hand interpretation** — the hand-tracking module interprets hand position and thumb-index gestures, maintains the neutral reference (anchor), and prepares smoothed input for teleoperation.
3. **Robot control** — the ROS 2 teleoperation node converts the hand input into bounded robot-arm movement and independent gripper commands. The SO-ARM kinematics module provides forward-kinematics and differential-IK calculations.
4. **Simulation and visualisation** — ROS 2 Control runs against mock hardware. MoveIt 2 provides the planning/robot-state environment, while RViz 2 displays the simulated robot.

The goal is intuitive, live control: moving a hand in front of the webcam moves the simulated arm, while thumb-index gestures control the gripper independently.

## What the system does

- Opens the webcam and displays a live camera window with hand landmarks and a compact status HUD.
- Detects whether a hand is visible and tracks its position relative to a captured neutral anchor.
- Converts lateral and depth movement into bounded Cartesian targets for the robot arm.
- Uses URDF-based SO-ARM kinematics and differential inverse kinematics to produce arm-joint updates.
- Streams bounded arm trajectories to the simulated arm controller and uses controller feedback to supervise movement.
- Sends separate gripper commands for the thumb-index gesture.
- Uses motion limits, input-freshness checks, hand-loss/watchdog handling, and a disarm control to reduce unintended movement.

The arm has five arm joints (`joint1`–`joint5`); the gripper is controlled separately through `joint6`.

## Hand-to-robot mapping

The intended mapping is:

| Human hand movement | Simulated robot response |
|---|---|
| Move hand left | Robot moves left |
| Move hand right | Robot moves right |
| Move hand toward the laptop webcam | Arm extends forward (robot world `-Y`) |
| Move hand away from the laptop webcam | Arm retracts backward (robot world `+Y`) |
| Keep hand stationary | Robot holds its current target; it should not drift |
| Thumb-index spread / open gesture | Gripper opens |
| Thumb-index pinch / closed gesture | Gripper closes |

Vertical robot movement (`Z`) is disabled in the initial mapping to avoid unwanted cross-axis motion. The hand should be visible and reasonably centred when the neutral anchor is captured.

## How the software is organised

```text
Laptop webcam
    │
    ▼
OpenCV frame capture
    │
    ▼
MediaPipe Hand Landmarker
    │  hand landmarks + hand presence
    ▼
hand_tracking.py
    │  tracking, gesture interpretation, smoothing, anchor-relative input, HUD
    ▼
hand_teleop_node.py
    │  enable/disarm state, input freshness, bounds and motion supervision
    ├───────────────────────────────┐
    ▼                               ▼
so101_kinematics.py                 Gripper command path
    │                               │
    │ FK + differential IK          │
    ▼                               ▼
/arm_controller/joint_trajectory   /gripper_controller/follow_joint_trajectory
    │                               │
    └─────────── ROS 2 Control ─────┘
                    │
                    ▼
       mock_components/GenericSystem
                    │
                    ▼
              /joint_states
                    │
                    ▼
             MoveIt 2 + RViz 2
```

The simulation launch starts the robot description/state publisher, mock `ros2_control` system, controllers, MoveIt 2 and RViz 2. The teleoperation launch starts the webcam/MediaPipe control interface and its ROS 2 node.

## Repository structure

The main application package is `src/hand_gesture_teleop/`.

```text
lerobot_ws/
├── src/
│   ├── hand_gesture_teleop/
│   │   ├── hand_gesture_teleop/
│   │   │   ├── hand_teleop_node.py       # ROS 2 teleoperation node
│   │   │   ├── hand_tracking.py          # Camera, landmarks and gesture tracking
│   │   │   └── so101_kinematics.py       # SO-ARM forward/differential kinematics
│   │   ├── launch/
│   │   │   ├── so101_sim.launch.py       # Mock robot, controllers, MoveIt and RViz
│   │   │   └── hand_teleop.launch.py     # Webcam and hand-teleoperation node
│   │   ├── config/                       # Simulation/controller configuration
│   │   ├── urdf/                         # Simulation robot description
│   │   ├── models/
│   │   │   └── hand_landmarker.task      # MediaPipe Hand Landmarker model
│   │   ├── scripts/                     # Supporting vision scripts
│   │   ├── package.xml
│   │   ├── setup.py
│   │   └── README.md
│   ├── lerobot_moveit/                  # Existing MoveIt configuration package
│   └── so_arm_100_hardware-main/        # Existing robot description/hardware package
├── scratch/
│   └── test_cartesian_ik.py             # Headless unit/regression tests
├── .venv_hand_gesture/                  # Local Python environment (create locally)
├── build/                               # Generated by colcon; do not edit
├── install/                             # Generated by colcon; do not edit
└── log/                                 # Generated build logs; do not edit
```

`build/`, `install/`, `log/`, Python caches and the local virtual environment are generated or machine-specific. They should not be treated as source files to commit. The MediaPipe model under `models/` is a project runtime asset and should remain in the repository.

## Requirements

### Operating system and robotics software

- Ubuntu 24.04 LTS
- ROS 2 Jazzy
- MoveIt 2 for ROS 2 Jazzy
- `ros2_control` and `ros2_controllers`
- `mock_components/GenericSystem`
- RViz 2 and Xacro
- `colcon` build tools

### Python and camera

- Python 3.12 (the project environment was developed with Python 3.12)
- OpenCV
- MediaPipe (Hand Landmarker Tasks API)
- NumPy compatible with the installed MediaPipe/OpenCV build
- A working webcam, accessible to the logged-in user
- A desktop session for the OpenCV camera window and RViz 2

The MediaPipe model file is included at `src/hand_gesture_teleop/models/hand_landmarker.task`.

For a fresh Ubuntu 24.04 machine, install the ROS 2 Jazzy desktop environment, MoveIt 2, ros2_control and the build tools before building this workspace. A typical binary-package setup is:

```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-desktop \
  ros-jazzy-moveit \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool \
  python3.12-venv
```

Refer to the [ROS 2 Jazzy Ubuntu installation guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html), the [MoveIt 2 binary installation guide](https://moveit.ai/install-moveit2/binary/), and the [ros2_control Jazzy installation guide](https://control.ros.org/jazzy/doc/getting_started/getting_started.html) for platform-specific setup and current package details.

## Before cloning

Complete these steps on the computer where you want to run the project:

1. Use **Ubuntu 24.04 LTS**.
2. Install **ROS 2 Jazzy**, **MoveIt 2**, `ros2_control`, the controllers, RViz 2, Xacro and the colcon build tools using the installation commands in [Requirements](#requirements).
3. Make sure Git is installed and that your GitHub account has been granted access to this private repository.
4. Configure GitHub authentication before cloning. If GitHub CLI is installed and is being used on this computer, sign in once:

```bash
gh auth login
gh auth setup-git
```

Do not run the workspace `rosdep install` command yet; it needs the repository's `src/` directory, which will be available after cloning.

## Clone the repository

Copy the clone URL from the repository's **GitHub → Code** menu (HTTPS or SSH). Then clone the repository into the expected workspace path:

```bash
git clone <YOUR_GITHUB_CLONE_URL> ~/lerobot_ws
cd ~/lerobot_ws
```

Replace `<YOUR_GITHUB_CLONE_URL>` with the actual private repository URL. Do not include the angle brackets. Keeping the workspace at `~/lerobot_ws` also keeps any workspace-specific paths consistent with the existing setup.

## Install ROS package dependencies

Now that the repository has been cloned and the `src/` directory exists, install any additional ROS package dependencies declared by its source packages:

```bash
cd ~/lerobot_ws
source /opt/ros/jazzy/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y --rosdistro jazzy
```

## Prepare the workspace

### 1. Load ROS 2 Jazzy

```bash
source /opt/ros/jazzy/setup.bash
cd ~/lerobot_ws
```

### 2. Create the project Python environment

The project uses a dedicated virtual environment with system site packages enabled so that Python can also access the ROS 2 Python modules installed by the ROS distribution.

```bash
python3 -m venv --system-site-packages .venv_hand_gesture
source .venv_hand_gesture/bin/activate
python -m pip install --upgrade pip
```

Install the computer-vision dependencies in that environment. These versions were used in the documented development environment:

```bash
python -m pip install "numpy==1.26.4" "mediapipe==0.10.35" "opencv-contrib-python==5.0.0.93"
```

If the project later adds a dependency manifest, prefer installing from that file instead of maintaining package versions manually.

### 3. Check the webcam

Close applications that may already own the webcam, such as Chrome/Google Meet, video-conferencing applications, or another camera preview. Ensure the webcam is available and provide good, even front lighting so MediaPipe can see the hand clearly.

## Build the package

With the virtual environment active, build the complete workspace after a fresh clone so all local packages required by the launch files are built:

```bash
cd ~/lerobot_ws
source /opt/ros/jazzy/setup.bash
source .venv_hand_gesture/bin/activate
python3 -m colcon build --symlink-install
```

For later changes limited to the hand-teleoperation package, a package-only rebuild is sufficient:

```bash
python3 -m colcon build --packages-select hand_gesture_teleop --symlink-install
```

After a successful build, source the workspace overlay:

```bash
source install/setup.bash
```

For a new terminal, source both the ROS underlay and the workspace overlay again. The `hand_teleop.launch.py` executable is installed by the package build; rebuild after changing package source or launch files.

## Run the simulation and teleoperation

**Important:** Use two separate terminals. Both terminals must use the same ROS domain ID. The following commands explicitly export `ROS_DOMAIN_ID=10` so every launched process inherits it.

Before starting, close any browser tab or video application that may be using `/dev/video0` (or another webcam device). Turn on good front lighting and position yourself so your hand is clearly visible.

### Terminal 1 — Robot simulation, MoveIt 2 and RViz 2

```bash
cd ~/lerobot_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=10
ros2 launch hand_gesture_teleop so101_sim.launch.py
```

Wait approximately **5–8 seconds** for RViz 2, the mock hardware system and the controllers to finish starting. Keep this terminal running.

### Terminal 2 — Hand-gesture teleoperation and camera window

Open a second terminal and run:

```bash
cd ~/lerobot_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=10
ros2 launch hand_gesture_teleop hand_teleop.launch.py
```

The OpenCV hand-teleoperation camera window should appear. Wait for the hand detector to initialise and for the HUD to show that a hand is detected.

> **Why `export ROS_DOMAIN_ID=10`?** A standalone `ROS_DOMAIN_ID=10` line only sets a shell variable unless it was already exported. Using `export` ensures the ROS 2 launch process and its child nodes receive the same domain ID in both terminals.

## Using the hand-gesture controls

The teleoperation interface starts in a **DISABLED / DISARMED** state. This is intentional. Make sure the webcam sees your hand before enabling control.

| Key / gesture | Function |
|---|---|
| `E` | Enable teleoperation and capture the neutral hand anchor |
| `D` | Disable teleoperation, cancel/clear pending motion as applicable, and return to a safe hold state |
| `C` | Recalibrate the neutral hand reference |
| `R` | Reset the teleoperation interaction state |
| `Q` or `Esc` | Close the camera/teleoperation window |
| Thumb-index spread | Request gripper open |
| Thumb-index pinch | Request gripper closed |

### Recommended first-use sequence

1. Keep your hand in a comfortable, visible position in front of the webcam.
2. Wait until the HUD shows `HAND: DETECTED`.
3. Press `E` once. The system enables control and captures the current hand position as its neutral anchor. Confirm that the HUD shows `CONTROL: ENABLED` and `MOTION ANCHOR: READY`.
4. Move your hand a small distance left or right and observe the corresponding movement in RViz.
5. Return near the anchor, then gently move your hand toward and away from the webcam to verify forward/backward movement.
6. Hold your hand still and confirm the robot holds its target without drift.
7. Try the thumb-index spread and pinch gestures to operate the gripper independently of arm movement.
8. Press `D` to disable teleoperation when finished.

Move slowly and deliberately during the first test. The controller applies bounds and motion limits; the robot is not intended to follow sudden large hand movements as unrestricted motion.

## Verify the simulation

Run these commands in another terminal after sourcing the same ROS environment and setting the same domain ID:

```bash
cd ~/lerobot_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=10
```

### Check controller status

```bash
ros2 control list_controllers
```

The simulation is expected to report these controllers as active:

- `joint_state_broadcaster`
- `arm_controller`
- `gripper_controller`

### Confirm mock hardware is active

```bash
ros2 control list_hardware_components --verbose
```

The simulation should report `SimSystem` using `mock_components/GenericSystem` in the active state. This is an important confirmation that the documented simulation is using mock hardware rather than the physical robot driver.

### Inspect joint feedback

```bash
ros2 topic echo /joint_states --once
```

This displays the latest simulated joint positions and available state values.

## Shut everything down

1. In the hand-teleoperation camera window, press `D` to disable control.
2. Press `Q` or `Esc` to close the camera window and stop the teleoperation node.
3. In Terminal 1, press `Ctrl+C` to shut down the simulation, MoveIt 2 and RViz 2.
4. If you opened extra diagnostic terminals, stop their commands as well.

Do not simply close terminal windows while ROS processes are still running. Shut down each launch process with `Ctrl+C` and check that the camera is released before starting another session.

## Troubleshooting

### The camera window does not open or the camera cannot be opened

- Close Chrome/Meet, Zoom, browser camera previews and other video applications.
- Check that the webcam is present: `ls /dev/video*`.
- Check camera permissions and make sure no other process owns the device.
- Relaunch the teleoperation launch after the other application has released the webcam.

### The HUD says `HAND: NOT DETECTED` or landmarks are unstable

- Improve front lighting and avoid strong backlighting.
- Keep the full hand inside the camera frame.
- Move the hand closer to a comfortable detection distance and hold it steady briefly.
- Recalibrate with `C` after improving the camera position or hand placement.

### ROS 2 cannot find the package or launch file

In each terminal, source the ROS distribution and workspace overlay in this order:

```bash
source /opt/ros/jazzy/setup.bash
cd ~/lerobot_ws
source install/setup.bash
```

If the package was changed or has not yet been built, rebuild it with the command in [Build the package](#build-the-package).

### RViz opens but the controllers are not active

Check the simulation launch terminal for startup errors, then inspect:

```bash
ros2 control list_controllers
ros2 control list_hardware_components --verbose
```

Do not enable hand teleoperation until the mock hardware component and required controllers are active.

### The camera sees the hand, but the robot does not move

- Confirm both terminals use `ROS_DOMAIN_ID=10`.
- Check that the HUD reports `CONTROL: ENABLED` and `MOTION ANCHOR: READY`.
- Make sure the neutral anchor was captured with `E` and the hand has moved beyond the input deadzone.
- Confirm the mock hardware and controllers are active.
- Press `D` to disarm, correct the issue, and re-enable with `E` rather than repeatedly moving the hand while the system is in an uncertain state.

### The hand appears to move the robot in the wrong direction

Stop the session with `D`. Check that the application is using the mapping documented in [Hand-to-robot mapping](#hand-to-robot-mapping), then recalibrate with `C`. Do not continue operating with an unexpected direction mapping.

## Run the regression tests

The headless regression suite is located at `scratch/test_cartesian_ik.py`. Run it with the project virtual environment and an isolated ROS domain so that it does not receive live messages from a running simulation:

```bash
cd ~/lerobot_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=99
.venv_hand_gesture/bin/python3 scratch/test_cartesian_ik.py
```

The test script reports the number of tests executed and whether they passed. Run tests when changing hand mapping, kinematics, trajectory generation, controller interaction, safety logic or gesture handling.

## Safety notes and limitations

- **Simulation only:** the documented `so101_sim.launch.py` workflow is designed around `mock_components/GenericSystem`. It does not establish or verify a physical robot connection.
- Do not replace the mock simulation launch with a physical hardware launch as part of these instructions.
- The source-level arm-motion parameter is designed to fail closed by default. Runtime enablement should only be used with the verified mock simulation and the intended launch configuration.
- The teleoperation node starts disabled. Use `E` only after checking the camera view and simulation status; use `D` to disarm.
- Motion limits, workspace bounds, input timeouts and hand-loss handling are software safeguards, not a certified safety system.
- Software cancellation/disarming is not an emergency stop and must never be treated as one for a physical robot.
- Do not connect, power, or command a physical SO-ARM 100/101 using this simulation-only procedure.

## Updating the repository

After changing source files or this README, review the changes and push them to the configured GitHub remote:

```bash
cd ~/lerobot_ws
git status
git add README.md
git commit -m "Document hand gesture teleoperation project"
git push
```

For future changes, stage the relevant files rather than generated folders such as `build/`, `install/`, `log/` or `.venv_hand_gesture/`.

---

**Project:** SO-ARM 101 Hand Gesture Teleoperation
**Middleware:** ROS 2 Jazzy
**Vision:** OpenCV + MediaPipe Hand Landmarker
**Simulation:** ROS 2 Control mock hardware + MoveIt 2 + RViz 2
