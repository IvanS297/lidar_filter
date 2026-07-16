import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    package_name = "lidar_filter"
    package_dir = get_package_share_directory(package_name)
    params_file_path = os.path.join(package_dir, 'config', 'standart.yaml')

    return LaunchDescription([
        Node(
            package=package_name,
            executable='glass_node',
            name='glass_patcher',
            output='screen',
            parameters=[params_file_path]
        )
    ])
