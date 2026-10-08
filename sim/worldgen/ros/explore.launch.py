"""Autonomous exploration for a Nova Carter driven by Isaac Sim: slam_toolbox builds the map, Nav2 moves the
robot, explore_lite picks frontiers. No map server / AMCL: the map comes from SLAM.

    ros2 launch /worldgen/explore.launch.py              # inside the worldgen container
    ros2 launch /worldgen/explore.launch.py explore:=false   # SLAM + Nav2 only (send goals by hand)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    explore = LaunchConfiguration("explore")
    nav2_params = LaunchConfiguration("nav2_params")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="True"),
        DeclareLaunchArgument("explore", default_value="True", description="Start explore_lite"),
        DeclareLaunchArgument("explore_delay", default_value="90.0", description="Seconds (wall) before explore_lite starts"),
        DeclareLaunchArgument("nav2_params", default_value=os.path.join(HERE, "nav2_params.yaml")),

        # Carter's 3D lidar -> 2D scan for SLAM (same node and settings as NVIDIA's carter_navigation launch).
        Node(
            package="pointcloud_to_laserscan", executable="pointcloud_to_laserscan_node", name="pointcloud_to_laserscan",
            remappings=[("cloud_in", "/front_3d_lidar/lidar_points"), ("scan", "/scan")],
            parameters=[{
                "target_frame": "front_3d_lidar", "transform_tolerance": 0.01,
                "min_height": -0.25, "max_height": 1.5,   # lidar sits 0.53 m up: skip floor-mesh noise below ~0.28 m
                # 720 beams of 0.5 deg: (max - min) / increment must be an integer, or slam_toolbox (Karto)
                # rejects every scan as "contains N range readings, expected M".
                "angle_min": -3.14159265, "angle_max": 3.14159265, "angle_increment": 0.00872665,
                "scan_time": 0.1, "range_min": 0.05, "range_max": 30.0,
                "use_inf": True, "inf_epsilon": 1.0, "use_sim_time": use_sim_time,
            }],
        ),
        # slam_toolbox is a lifecycle node in Jazzy; its own launch file emits the configure/activate transitions.
        # Started as a plain Node it stays unconfigured and silent (no /map, no map->odom).
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory("slam_toolbox"), "launch", "online_async_launch.py")),
            launch_arguments={"use_sim_time": use_sim_time, "autostart": "true",
                              "slam_params_file": os.path.join(HERE, "slam_params.yaml")}.items(),
        ),
        # Nav2 after SLAM has had time to publish map->odom; its costmaps abort activation if the transform is late.
        TimerAction(
            period=20.0,
            actions=[IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(get_package_share_directory("nav2_bringup"), "launch", "navigation_launch.py")),
                launch_arguments={"use_sim_time": use_sim_time, "params_file": nav2_params, "autostart": "True"}.items(),
            )],
        ),
        # Started right after Nav2 activates, explore_lite sees an unpopulated costmap, reports "No frontiers
        # found" and quits for good. Give SLAM and the costmaps time to fill in first (wall-clock seconds).
        TimerAction(
            period=LaunchConfiguration("explore_delay"),
            actions=[Node(
                package="explore_lite", executable="explore", name="explore_node", output="screen",
                condition=IfCondition(explore),
                parameters=[os.path.join(HERE, "explore_params.yaml"), {"use_sim_time": use_sim_time}],
            )],
        ),
    ])
