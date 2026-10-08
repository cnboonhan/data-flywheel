#!/usr/bin/env bash
# docker compose for the worldgen ROS 2 container, run as the calling user so files under runs/ stay yours.
# (Compose can't see bash's UID/GID unless they are exported.)
#   bash sim/worldgen/nav.sh build
#   bash sim/worldgen/nav.sh up explore                       # SLAM + Nav2 + frontier exploration
#   bash sim/worldgen/nav.sh run --rm nav ros2 topic list
#   bash sim/worldgen/nav.sh run --rm nav /worldgen/record.sh <name>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UID GID="$(id -g)"
mkdir -p "${HERE}/runs"
exec docker compose -f "${HERE}/compose.yml" "$@"
