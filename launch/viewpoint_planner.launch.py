import launch
import launch_ros.actions


def generate_launch_description():
    ld = launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            name='environment',
            default_value='00195'
        ),
        launch.actions.DeclareLaunchArgument(
            name='home_directory',
            default_value='/workspace'
        ),
        launch.actions.DeclareLaunchArgument(
            name='cam_string',
            default_value='PINHOLE 1280 720 609.5238037109375 610.1694946289062 640 360'
        ),
        launch.actions.DeclareLaunchArgument(
            name='mode_string',
            default_value='mlp_clf'
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
                    'environment': launch.substitutions.LaunchConfiguration('environment')
                },
                {
                    'home_directory': launch.substitutions.LaunchConfiguration('home_directory')
                },
                {
                    'cam_string': launch.substitutions.LaunchConfiguration('cam_string')
                },
                {
                    'mode_string': launch.substitutions.LaunchConfiguration('cam_string')
                },
                {
                    'occlusion': launch.substitutions.LaunchConfiguration('occlusion')
                },
                {
                    'num_viewpoint_samples': launch.substitutions.LaunchConfiguration('num_viewpoint_samples')
                },
            ]
        )
    ])
    return ld


if __name__ == '__main__':
    generate_launch_description()
