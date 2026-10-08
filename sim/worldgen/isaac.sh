#!/usr/bin/env bash
# Run the Isaac Sim half of the world-generation loop (Nova Carter in a NuRec room, ROS 2 out) from Arena's
# Isaac Sim 6.1 venv, using Isaac Sim's bundled ROS 2 Jazzy: no ROS install on the host. The DDS profile is
# UDP-only so the worldgen container (sim/worldgen/compose.yml, host network) can see the topics.
#   bash sim/worldgen/isaac.sh                      # headless, until Ctrl-C
#   bash sim/worldgen/isaac.sh --gui --seconds 60
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARENA="${ROOT}/eval/system2/IsaacLab-Arena"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export ROS_DISTRO="${ROS_DISTRO:-jazzy}" RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
# Own DDS domain: this LAN also carries a real robot on domain 0 (its /hdas/* topics showed up in a probe).
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${ROOT}/sim/worldgen/ros/fastdds.xml"
# Isaac Sim's bundled ROS 2 libraries (the bridge refuses to start without this when launched from the venv).
ROS2_LIB="${ARENA}/.venv/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/${ROS_DISTRO}/lib"
[ -d "${ROS2_LIB}" ] || { echo "bundled ROS 2 libs not found: ${ROS2_LIB}" >&2; exit 1; }
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${ROS2_LIB}"
cd "${ARENA}"
exec .venv/bin/python "${ROOT}/sim/worldgen/isaac/carter_room.py" "$@"
