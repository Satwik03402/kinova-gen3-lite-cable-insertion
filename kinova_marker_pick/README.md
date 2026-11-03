# Kinova Gen3 Lite: cable peg pickup and insertion

ROS 2 Jazzy + Gazebo Harmonic + MoveIt 2. This package now starts the six-axis
Gen3 Lite, its two-finger gripper, controllers, MoveIt, calibrated camera, fixtures,
two calibrated cameras, marker detector, and insertion controller together. It replaces the earlier
pickup-only prototype; the external `kinova_camera_control.launch.py` now forwards
to the package launch.

The initial object is a **rigid 8 mm diameter peg with a 120 mm cable-like tail**.
Its 48 × 40 × 26 mm head carries ArUco 4×4 ID 7 (32 mm). ID 8 locates a green
upright hub with a genuine 14 mm square opening. The gray cradle holds the head
upright for pickup. The hub face is vertical (the YZ plane), facing the robot along
world −X; insertion proceeds horizontally along world +X. The task is open →
predefined pose 1 above peg → wrist IBVS on ID 7 → grasp → lift → retreat and turn
→ predefined pose 2 in front of wall → wrist IBVS on ID 8 → insert 26 mm.
The robot keeps holding the inserted connector at the end.

## Run in this workspace

ROS 2 Jazzy and the existing `~/kinova_ws/install` Kortex installation must be present.

```bash
cd /home/sri/vbm_sim/src
./run_sim.sh auto_execute:=true
```

The script builds just this package into `.sim/`, sources ROS and the Kinova overlay,
and opens Gazebo with the external workspace view and an **Eye in hand** image
panel. The wrist camera moves with `tool_frame` and is tilted toward the peg/port. Override the overlay with
`KINOVA_SETUP=/path/to/install/setup.bash ./run_sim.sh ...` if necessary.
It never starts a physical Kinova driver.

## Two predefined observation poses

Edit `config/observation_poses.yaml` to set the aligned reference poses and the
deliberate starting offsets. XYZ values are world metres; RPY values are radians.

| Stage | Aligned reference position | Orientation and feedback |
| --- | --- | --- |
| 1: above the peg | Tool XYZ `[0.300, -0.120, 0.230]` | Tool RPY `[pi, 0, pi/2]`; wrist camera looks down. ID 7 drives X/Y and yaw alignment before descent. |
| 2: in front of the wall | Connector XYZ `[0.308, 0.140, 0.250]`; nominal tool XYZ approximately `[0.288, 0.140, 0.250]` | Connector RPY `[0, -pi/2, 0]`; peg points into the wall and wrist camera faces the wall. ID 8 drives Y/Z and roll about the insertion axis. |

**Both actual observation approaches start off target by default.**
`observation_offset_m: 0.006` adds +6/-6 mm in the observation plane, and
`observation_angle_rad: 0.025` adds 1.43 degrees about the fixture normal:

- Pickup starts at tool XYZ `[0.306, -0.126, 0.230]` with the added world yaw.
- Wall approach adds +6 mm world Y, -6 mm world Z and the added world roll to
  the reference tool pose computed from the actual grasp.

The desired image corners stay at the aligned reference. IBVS must move the
detected marker into the target square before grasping or inserting. Change
the two offsets in the YAML file, or override them on the command line. To
disable the deliberate displacement, add
`observation_offset_m:=0 observation_angle_rad:=0` to the launch command.

Pose 2 is stored as a connector pose, so the tool pose can account for the actual
grasp offset. Its tip starts 65 mm in front of the hole. The clearance turn first
sets the peg perpendicular to the known vertical wall; image feedback then
corrects lateral position and rotation about that normal to match the marker.

At **both stages**, IBVS uses the four measured marker corners, a calibrated
point interaction matrix, and damped least squares. PnP supplies corner depths
for the matrix; reconstructed 3D pose error does not drive the corrections.
Camera rotation and its offset from the tool are included in the image model.

This is **sampled IBVS**: each image produces a bounded Cartesian correction
(at most 3 mm / 0.012 rad); MoveIt checks the entire correction for collisions,
executes it, then the loop waits for a new image after motion. It requires three
distinct frames within 0.9 pixel coordinate RMSE before proceeding. It is not a
high-rate velocity servo. Missing images/transforms, cancellation, incomplete
paths, excessive corrections or nonconvergence stop the sequence.

Insertion is a separate, collision-checked Cartesian motion that preserves the
position and rotation obtained at pose 2. IBVS runs at the two observation poses.

To inspect first, omit `auto_execute:=true`. For RViz as well, add `rviz:=true`.
For a headless run, add `gui:=false`. A headless camera still needs working EGL
rendering; it is not a sensor-free simulation.

In another terminal, source the same environments and start one attempt:

```bash
source /opt/ros/jazzy/setup.bash
source ~/kinova_ws/install/setup.bash
source /home/sri/vbm_sim/src/.sim/install/setup.bash
ros2 service call /marker_pick/start std_srvs/srv/Trigger '{}'
ros2 topic echo /marker_pick/status --qos-durability transient_local
```

Stop queued or running motion using:

```bash
ros2 service call /marker_pick/cancel std_srvs/srv/Trigger '{}'
```

Cancellation leaves a held connector attached. Every launch permits one attempt;
restart the launch after completion or an abort to reset the scene.

## What is simulated

- The robot joints and object move in Gazebo. The controller checks the actual
  connector pose after lifting and insertion; a successful arm trajectory alone
  does not count as task success.
- Grasp retention uses Gazebo's **idealized detachable fixed joint**. Robot/connector
  physical contact is disabled with collision masks; finger friction and cable
  deformation are not being evaluated. The object still collides with the fixtures.
- `target_source:=markers` (default) uses the wrist camera for both pickup ID 7
  and hub ID 8. The overhead stream remains available for viewing. Each stream
  has its own calibration and optical frame; control uses image-time TF.
- Known fixture locations supply the two initial observation poses. The cradle
  height and vertical wall normal are known. This is not an unknown-fixture search.
- The EIH sensor is a simulated RGB camera. A full D405 depth model, force-guided
  insertion and high-rate velocity IBVS remain subsequent project stages.
- `target_source:=fixture` uses known fixture poses to isolate motion/physics
  debugging from vision; it is explicitly a ground-truth baseline.
- Gazebo and MoveIt use the same cradle/socket geometry. Head, peg and tail are
  represented in MoveIt's scene and attached to the tool during transport.
- The held connector turns in a lateral clearance corridor before the separate
  approach to the hub. The arm planner uses dense collision checks and a 0.001 rad
  TOTG path tolerance; final trajectory validation stays enabled. Planning-only
  rejections can be retried up to three times. Controller failures, cancellation,
  and action timeouts abort without automatic re-execution.
- Pickup descent, lift, clearance retreat, alignment and insertion require full collision-checked
  Cartesian paths. Partial paths and joint discontinuities abort before execution.
- All nodes use simulation time. Detection uses sensor-compatible QoS and
  transforms observations at the image timestamp, which also supports moving cameras.

Success appears on `/marker_pick/status` as `stage: complete`, with measured lateral
error in the hub face and insertion depth along its normal. The limits are 2.5 mm lateral error and 4 mm depth error.
These are loose first-simulation criteria, not connector assembly specifications.

## Views and topics

Gazebo starts with an oblique view of the hub front, the robot, and the cradle.
Its floating **Eye in hand** panel displays the live alignment overlay from
`/marker_pick/eih_debug_image`, bridged back into Gazebo. RViz uses the same
annotated wrist image, `world`, a close orbit around the workspace,
and shows the planning scene, grasp target, and both image streams. Sensor image topics are independent
of the movable Gazebo viewer camera.

| Topic | Content |
| --- | --- |
| `/camera/image` | 1280 × 960 overhead RGB image, 15 Hz |
| `/camera/camera_info` | overhead intrinsics and optical frame |
| `/eih/image` | 1280 × 960 wrist RGB image, 15 Hz |
| `/eih/camera_info` | wrist intrinsics; `eih_camera_optical_frame` |
| `/marker_pick/eih_debug_image` | live wrist alignment overlay, preserved camera frame and timestamp |
| `/marker_pick/debug_image` | marker outlines when subscribed |
| `/marker_pick/head_grasp_pose` | estimated `tool_frame` pickup goal |
| `/marker_pick/port_pose` | estimated socket opening pose |
| `/marker_pick/status` | JSON stage and result, transient-local |
| `/marker_pick/ibvs` | per-stage corner error, desired/measured corners and proposed tool increment |
| `/model/marked_connector/pose` | actual Gazebo object pose for verification |

The gripper uses `/gen3_lite_2f_gripper_controller/gripper_cmd`
(`control_msgs/action/GripperCommand`), not a joint-trajectory action.
The Cartesian planner targets `tool_frame`, 130 mm forward of the wrist.

The camera overlay shows:

- **Cyan:** current marker corners and centre.
- **Yellow target square:** a fixed outline with bold corner brackets and a
  centre crosshair. It turns green when the marker falls within tolerance.
- **Pink arrows:** the direction from each measured corner toward its target.
- **Header:** pickup/port stage, marker ID and live corner RMSE in pixels.
- **Green:** within tolerance; `ALIGNED` appears after the controller confirms
  convergence and holds the confirmation for one second. If the current frame loses the marker, the cyan outline disappears
  and `MARKER LOST` replaces the alignment status.

The target automatically switches between ID 7 and ID 8. It is displayed during
the corresponding alignment stage and hidden during transfer or insertion, when
the camera is moving away from that observation pose. The desired port marker is
offset from the image centre by design; match the cyan corners to the target
square. The square follows the calibrated camera perspective, so its projection
may be slightly trapezoidal. The raw camera remains on `/eih/image` for vision
and recording.

## Standard colcon build

If building in your own existing workspace, source ROS and the Kortex overlay,
place the package in `src`, then:

```bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select kinova_marker_pick --symlink-install
source install/setup.bash
ros2 launch kinova_marker_pick cable_insertion.launch.py auto_execute:=true
```

Use the ROS system Python and distro `python3-opencv`, `python3-scipy`, and
`python3-numpy`. The detector supports both OpenCV 4.6's factory API and the newer
`ArucoDetector` API. The launch corrects upstream duplicate gripper interfaces and
the gripper simulation plugin spelling locally, without changing Kortex files.
Bullet Featherstone supplies the finger mimic-joint support.

## Development and checks

```bash
# From this repository's src directory, after sourcing ROS and Kortex:
PYTHONPATH="$PWD/kinova_marker_pick:$PYTHONPATH" python3 -m pytest -q kinova_marker_pick/test
python3 kinova_marker_pick/generate_model.py
```

The generator writes both model/world SDF assets and printable marker references.
Scene dimensions are centralized in `kinova_marker_pick/scene.py`; regenerate
assets and rebuild after changes. `config/initial_positions.yaml` controls the
initial arm posture; `config/controllers.yaml` controls simulated joints.

References for the implementation:
[Kinova ROS 2 Kortex](https://github.com/Kinovarobotics/ros2_kortex),
[Gazebo detachable joints](https://gazebosim.org/api/sim/8/detachablejoints.html), and
[Gazebo ROS model spawning](https://gazebosim.org/docs/harmonic/ros2_spawn_model/),
and [Gazebo image display configuration](https://gazebosim.org/api/gui/9/config.html).
The image control follows the point-feature formulation in
[Chaumette & Hutchinson, Visual Servo Control, Part I](https://web.mit.edu/amcp/OldFiles/drg/Chaumette_Part_I.pdf).
The supplied project slide motivates marker-guided alignment, the simplified
peg-in-hole task, and later wrist-camera visual servoing; this first demo does not
implement the learning-based methods from those papers.

For a reproducible full Gazebo check (after sourcing the built workspace):

```bash
python3 kinova_marker_pick/scripts/validate_sim.py --output .sim/validation
```

This starts its own simulation, saves `result.json`, `ibvs.json`, the launch log,
initial/final images from both cameras and annotated images for both IBVS stages,
then stops its processes. Marker-mode success requires convergence at both stages
as well as the physical insertion check. Its default ROS domain is 92, separate
from the normal interactive launch. The earlier upright-hub marker-guided baseline
measured 2.0 mm lateral error and 23.3 mm horizontal insertion depth. It also
verified both calibrated image streams and 0.38 m of wrist-camera movement. All
16 regression checks passed. Evidence is saved under `.sim/vertical_validation/`.
The launcher removes inherited Snap GUI module paths locally, fixing the
`GLIBC_PRIVATE` library error when starting Gazebo from Snap-installed VS Code.

## If the arm stops after lifting

Check `/marker_pick/status` or the launch terminal. `aborted` means the attempt
has ended, while the grasp attachment keeps holding the connector. Error messages
include the failed stage and MoveIt error name. The earlier direct turn-and-transfer
could produce a postprocessed path that clipped the hub's upper edge; the separate
clearance turn and transfer replace that motion. See MoveIt's
[time parameterization documentation](https://moveit.picknik.ai/main/doc/examples/time_parameterization/time_parameterization_tutorial.html)
for why retimed paths require collision validation.

After updating the package, restart the old process so it loads the new code:
press Ctrl+C in the launch terminal, then run:

```bash
cd /home/sri/vbm_sim/src
./run_sim.sh auto_execute:=true 2>&1 | tee /tmp/kinova_terminal.log
```

The launcher rebuilds automatically. Reuse the ROS domain and Gazebo partition
from your terminal; do not start another copy while the previous launch is active.

The transfer fix passed two fresh marker-guided Gazebo runs, each with about
2.0 mm lateral error and 23.3 mm insertion depth. Reports and camera frames are
in `.sim/transfer_validation_2/` and `.sim/transfer_validation_3/`; all 16
regression checks passed.

## Two-pose validation

The full marker-guided Gazebo trials passed, including the default offset / square-overlay GUI run:

| Trial | Approach displacement | Pose 1 error | Pose 2 error | Physical insertion |
| --- | --- | --- | --- | --- |
| `.sim/two_pose_validation_1` | +6/-6 mm in each observation plane | 27.59 → 0.83 px | 26.97 → 0.76 px | 1.77 mm lateral, 23.26 mm depth |
| `.sim/two_pose_validation_2` (GUI) | -6/+6 mm plus 0.025 rad rotation | 26.86 → 0.89 px | 33.10 → 0.32 px | 1.84 mm lateral, 23.38 mm depth |
| `.sim/square_target_validation` (GUI, current defaults) | +6/-6 mm plus 0.025 rad rotation | 27.72 → 0.81 px | 20.40 → 0.57 px | 2.08 mm lateral, 23.29 mm depth |

The current default run completed in 122 seconds and delivered 598 annotated
wrist-camera frames. Its confirmed green target squares are saved as
`.sim/square_target_validation/pose1_overlay.png` and `pose2_overlay.png`.

The insertion command is 26 mm; the table gives the actual Gazebo measurements,
evaluated against the existing 2.5 mm lateral / 4 mm depth-error criteria.
All 25 regression checks passed. The original working source and launch file are
saved in `.sim/checkpoints/before_two_pose_ibvs/`.

To repeat the pose/rotation recovery test after sourcing the built workspace:

```bash
python3 kinova_marker_pick/scripts/validate_sim.py --gui \
  --observation-offset -0.006 --observation-angle 0.025 \
  --output .sim/two_pose_repeat
```

Those arguments override the robot's observation approach offsets; the desired
image features remain unchanged. Normal `run_sim.sh` uses +6/-6 mm offsets and
0.025 rad rotation at both stages.
