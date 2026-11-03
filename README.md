# Kinova Gen3 Lite cable connector insertion

ROS 2 Jazzy, Gazebo Harmonic and MoveIt 2 simulation of a Kinova Gen3 Lite
picking up a peg with a rigid cable-like tail and inserting it into a vertical hub.

The wrist camera uses sampled image-based visual servoing (IBVS) at two
observation poses: above the pickup marker and in front of the port marker.
Both approaches deliberately start offset by 6 mm in each alignment axis and
1.43 degrees about the fixture normal. The camera overlay shows measured marker
corners, a target square and convergence status.

## Run

Requires Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic and an installed ROS 2 Kortex
workspace providing `kortex_description` and `kinova_gen3_lite_moveit_config`.
The launcher expects the Kortex overlay at `~/kinova_ws/install/setup.bash`.

From this repository directory:

```bash
./run_sim.sh auto_execute:=true
```

For another Kortex installation:

```bash
KINOVA_SETUP=/path/to/kinova_ws/install/setup.bash ./run_sim.sh auto_execute:=true
```

The script builds this package into `.sim/` and opens Gazebo with an eye-in-hand
camera panel. Stop the launch with Ctrl+C and relaunch for a fresh scene.
It starts the simulation only; no physical robot driver is launched.

## Results and scope

One completed GUI trial reduced pickup marker error from 27.72 to 0.81 pixels
and port marker error from 20.40 to 0.57 pixels. Measured insertion depth was
23.29 mm for a 26 mm target, with 2.08 mm lateral error; the sequence took
122 seconds. These are measurements from one trial, not a success-rate estimate.

The prototype uses a rigid tail, idealized grasp attachment and known fixture
geometry. IBVS handles observation-stage alignment. Final insertion is a planned
Cartesian motion, followed by verification using Gazebo object pose data.

- [Detailed setup, parameters and validation commands](kinova_marker_pick/README.md)
- [Observation poses and starting offsets](kinova_marker_pick/config/observation_poses.yaml)
- [Convergence graph](presentation_assets/ibvs_convergence.png)
- [Vector graph](presentation_assets/ibvs_convergence.svg)
- [Recorded convergence measurements](presentation_assets/ibvs_convergence.csv)

Generated builds, local validation reports and runtime logs are excluded from Git.
