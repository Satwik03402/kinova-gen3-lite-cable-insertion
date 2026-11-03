#!/usr/bin/env bash
# Build only this simulation package, then launch Gen3 Lite in Gazebo Harmonic.
set -eo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# VS Code installed as a Snap can export GTK modules linked to an older libc.
# Keep native Gazebo/Qt processes on the system GUI libraries.
for gui_variable in GTK_PATH GTK_EXE_PREFIX GTK_DATA_PREFIX GIO_MODULE_DIR GIO_EXTRA_MODULES QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH; do
  if [[ "${!gui_variable}" == *"/snap/"* ]]; then
    unset "$gui_variable"
  fi
done
source /opt/ros/jazzy/setup.bash
kinova_setup="${KINOVA_SETUP:-$HOME/kinova_ws/install/setup.bash}"
if [[ -f "$kinova_setup" ]]; then
  source "$kinova_setup"
fi
if ! ros2 pkg prefix kortex_description >/dev/null 2>&1 || ! ros2 pkg prefix kinova_gen3_lite_moveit_config >/dev/null 2>&1; then
  echo 'Source your ros2_kortex installation or set KINOVA_SETUP to its install/setup.bash.' >&2
  exit 1
fi
sim_state="$project_root/.sim"
colcon --log-base "$sim_state/log" build \
  --base-paths "$project_root/kinova_marker_pick" \
  --build-base "$sim_state/build" --install-base "$sim_state/install" \
  --packages-select kinova_marker_pick --symlink-install
source "$sim_state/install/setup.bash"
export ROS_LOG_DIR="${ROS_LOG_DIR:-$sim_state/ros_logs}"
exec ros2 launch kinova_marker_pick cable_insertion.launch.py "$@"
