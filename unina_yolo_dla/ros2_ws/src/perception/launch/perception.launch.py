import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Paths
    pkg_share = get_package_share_directory('perception')
    default_params_file = os.path.join(pkg_share, 'config', 'params.yaml')
    
    # Launch Arguments
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Full path to the ROS2 parameters file to use'
    )
    
    # Perception Node (Lifecycle)
    perception_node = LifecycleNode(
        package='perception',
        executable='perception_node',
        name='perception_node',
        namespace='',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
        arguments=['--ros-args', '--log-level', 'info']
    )
    
    return LaunchDescription([
        params_file_arg,
        perception_node
    ])
