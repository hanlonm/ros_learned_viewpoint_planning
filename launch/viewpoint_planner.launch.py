import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='home_directory',
            default_value='/workspace'
        ),
        launch.actions.DeclareLaunchArgument(
            name='map_name',
            default_value="map-DLAB_3"
        ),
        launch.actions.DeclareLaunchArgument(
            name='cam_string',
            default_value='PINHOLE 1280 720 609.5238037109375 610.1694946289062 640 360'
        ),
        launch.actions.DeclareLaunchArgument(
            name='mode_string',
            default_value='max_crit',
            choices=["mlp_clf", "mlp_reg", "trf_clf", "angle_crit", "max_crit", "fisher_info", "random"]
        ),
        launch.actions.DeclareLaunchArgument(
            name='occlusion',
            default_value='False'
        ),
        launch.actions.DeclareLaunchArgument(
            name='num_viewpoint_samples',
            default_value='100'
        ),
        launch_ros.actions.Node(
            package='learned_viewpoint_planning',
            executable='viewpoint_planner',
            output='screen',
            emulate_tty=True,
            parameters=[
                {
                    'home_directory': launch.substitutions.LaunchConfiguration('home_directory')
                },
                {
                    'map_name': launch.substitutions.LaunchConfiguration('map_name')
                },
                {
                    'cam_string': launch.substitutions.LaunchConfiguration('cam_string')
                },
                {
                    'mode_string': launch.substitutions.LaunchConfiguration('mode_string')
                },
                {
                    'occlusion': launch.substitutions.LaunchConfiguration('occlusion')
                },
                {
                    'num_viewpoint_samples': launch.substitutions.LaunchConfiguration('num_viewpoint_samples')
                },
            ]
        ),
        launch_ros.actions.Node(
            package='learned_viewpoint_planning',
            executable='pose_publisher',
            output='screen',
            emulate_tty=True,
        ),
        launch_ros.actions.Node(
            package='learned_viewpoint_planning',
            executable='spot_viewpoint_planning',
            output='screen',
            emulate_tty=True,
        )
        
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
