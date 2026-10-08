#!/bin/bash
# Source ROS 2 Jazzy and the carter_navigation overlay, then run the given command.
set -e
source /opt/ros/jazzy/setup.bash
[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash
exec "$@"
