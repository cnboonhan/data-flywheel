#!/usr/bin/env bash
# Run Isaac Sim (IsaacLab-Arena's venv, 6.1) with its bundled ROS 2 Jazzy, on the same DDS settings as the nav2
# container. No ROS install on the host.
#   bash sensors/real2sim/run_sim.sh            # headless Nova Carter warehouse
#   bash sensors/real2sim/run_sim.sh --gui
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARENA="$(cd "${HERE}/../.." && pwd)/eval/system2/IsaacLab-Arena"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export ROS_DISTRO=jazzy RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${HERE}/fastdds.xml"
# Without the bundled libraries on the path the bridge logs "ROS2 Bridge startup failed" and publishes nothing.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${ARENA}/.venv/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/jazzy/lib"
[[ $(uname -m) == aarch64 ]] && export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}/lib/aarch64-linux-gnu/libgomp.so.1"   # Isaac Sim on ARM refuses to start without it
cd "${ARENA}"
exec "${ARENA}/.venv/bin/python" "${HERE}/isaac_sim.py" "$@"
