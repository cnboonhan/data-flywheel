#!/bin/bash
# Record the capture bag NuRec's stereo workflow needs: the front Hawk stereo pair (+ camera_info), /tf,
# odometry and the 3D lidar, as mcap under /runs/<name>/. Run inside the worldgen container while Isaac Sim
# publishes and the robot explores:
#   docker compose -f sim/worldgen/compose.yml run --rm nav /worldgen/record.sh <name> [extra topics...]
set -euo pipefail
name="${1:?usage: record.sh <name> [extra topics...]}"; shift
out="/runs/${name}/bag"
mkdir -p "/runs/${name}"
exec ros2 bag record --storage mcap --use-sim-time -o "${out}" \
  /tf /tf_static /clock \
  /chassis/odom \
  /front_3d_lidar/lidar_points \
  /front_stereo_camera/left/image_raw /front_stereo_camera/left/camera_info \
  /front_stereo_camera/right/image_raw /front_stereo_camera/right/camera_info \
  "$@"
