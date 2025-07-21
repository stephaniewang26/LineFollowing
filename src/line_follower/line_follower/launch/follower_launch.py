from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    follower_node = Node(
        package='line_follower',
        executable='tracker',
        output='screen'
    )

    detector_node = Node(
        package='line_follower',
        executable='detector',
        output='screen'
    )

    return LaunchDescription([follower_node, detector_node])