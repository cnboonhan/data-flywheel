# ROS 2 Jazzy navigation stack for the world-generation loop: Nav2, slam_toolbox, explore_lite and
# rosbag2 (mcap). Talks to Isaac Sim's bundled ROS 2 Jazzy over DDS
# on the host network; no ROS install on the host.
FROM ros:jazzy-ros-base

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ros-jazzy-navigation2 \
      ros-jazzy-nav2-bringup \
      ros-jazzy-slam-toolbox \
      ros-jazzy-pointcloud-to-laserscan \
      ros-jazzy-rosbag2-storage-mcap \
      ros-jazzy-rmw-fastrtps-cpp \
      ros-jazzy-rmw-cyclonedds-cpp \
      ros-jazzy-tf2-tools \
      ros-jazzy-rviz2 \
      python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# explore_lite (frontier exploration) has no Jazzy apt package: build it from the m-explore-ros2 port.
# NVIDIA's carter_navigation package isn't built; its Nav2 params were copied into ros/nav2_params.yaml.
WORKDIR /ws
RUN git clone --depth 1 https://github.com/robo-friends/m-explore-ros2.git src/m-explore-ros2 \
    && rm -rf src/m-explore-ros2/map_merge
RUN . /opt/ros/jazzy/setup.sh \
    && apt-get update && rosdep install -i --from-path src --rosdistro jazzy -y \
    && colcon build --symlink-install --packages-up-to explore_lite \
    && rm -rf /var/lib/apt/lists/*

# Our launch files and params are bind-mounted at /worldgen (see compose.yml).
COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
