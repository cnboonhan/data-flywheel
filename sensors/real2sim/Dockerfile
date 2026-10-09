# ROS 2 Jazzy + Nav2 + NVIDIA's carter_navigation package (map, Nav2/AMCL params, launch) from
# IsaacSim-ros_workspaces at a pinned commit. Upstream is used unmodified; only carter_navigation is built.
FROM ros:jazzy-ros-base

ARG ROS_WS_COMMIT=a9e8471ee901bc2332c1e4aca94ac580713ca3ab
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git python3-colcon-common-extensions \
      ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-pointcloud-to-laserscan \
      ros-jazzy-robot-state-publisher ros-jazzy-rviz2 ros-jazzy-rmw-fastrtps-cpp ros-jazzy-tf2-tools \
      ros-jazzy-nav2-simple-commander ros-jazzy-cv-bridge ros-jazzy-tf2-sensor-msgs ros-jazzy-sensor-msgs-py \
      python3-scipy python3-opencv xvfb xauth \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
RUN git clone --filter=blob:none --sparse https://github.com/isaac-sim/IsaacSim-ros_workspaces.git /tmp/ros_ws \
    && git -C /tmp/ros_ws checkout ${ROS_WS_COMMIT} \
    && git -C /tmp/ros_ws sparse-checkout set jazzy_ws/src/navigation/carter_navigation \
    && mkdir -p src && cp -r /tmp/ros_ws/jazzy_ws/src/navigation/carter_navigation src/ && rm -rf /tmp/ros_ws \
    && . /opt/ros/jazzy/setup.sh && colcon build --packages-select carter_navigation

# Source ROS and the workspace for every command.
RUN printf '#!/bin/bash\nsource /opt/ros/jazzy/setup.bash\nsource /ws/install/setup.bash\nexec "$@"\n' > /entrypoint.sh \
    && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
