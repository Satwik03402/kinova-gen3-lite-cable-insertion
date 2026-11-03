"""Complete Gen3 Lite / Gazebo Harmonic / MoveIt pick-and-insert demonstration."""
from pathlib import Path
import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            SetEnvironmentVariable, UnsetEnvironmentVariable,
                            RegisterEventHandler, EmitEvent, LogInfo)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from kinova_marker_pick.sim_description import robot_description, simulation_sdf
from kinova_marker_pick.scene import CAMERA


def setup(context):
    share = Path(get_package_share_directory('kinova_marker_pick'))
    urdf = robot_description(share)
    sdf = simulation_sdf(urdf)
    moveit = (MoveItConfigsBuilder('gen3_lite', package_name='kinova_gen3_lite_moveit_config')
              .robot_description_semantic(file_path='config/gen3_lite.srdf')
              .robot_description_kinematics(file_path='config/kinematics.yaml')
              .trajectory_execution(file_path='config/moveit_controllers.yaml')
              .planning_pipelines(pipelines=['ompl']).to_moveit_configs())
    moveit.robot_description = {'robot_description':urdf}
    # The rigid tail is thin: check joint-space edges densely and keep TOTG
    # blending close to the collision-checked path. ValidateSolution stays on.
    pipeline = moveit.planning_pipelines['ompl']
    pipeline.setdefault('arm', {})['longest_valid_segment_fraction'] = 0.002
    pipeline['arm']['enforce_joint_model_state_space'] = True
    pipeline['arm']['projection_evaluator'] = 'joints(joint_1,joint_2)'
    pipeline['totg'] = {'path_tolerance':0.001, 'resample_dt':0.02, 'min_angle_change':0.001}
    # Older upstream configs misidentify the end-effector group as the arm itself.
    semantic = moveit.robot_description_semantic['robot_description_semantic']
    semantic = semantic.replace('parent_link="end_effector_link" group="arm"',
                                'parent_link="end_effector_link" group="gripper"')
    moveit.robot_description_semantic['robot_description_semantic'] = semantic
    sim = {'use_sim_time':True}
    gz_args = (f'-r -v 3 {share / "worlds/cable_insertion.sdf"}'
               ' --physics-engine gz-physics-bullet-featherstone-plugin')
    if LaunchConfiguration('gui').perform(context).lower() != 'true':
        gz_args += ' -s --headless-rendering'
    bridge_args = [
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        '/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
        '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        '/eih/image@sensor_msgs/msg/Image[gz.msgs.Image',
        '/eih/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        '/marker_pick/eih_debug_image@sensor_msgs/msg/Image]gz.msgs.Image',
        '/grasp/attach@std_msgs/msg/Empty]gz.msgs.Empty',
        '/grasp/detach@std_msgs/msg/Empty]gz.msgs.Empty',
        '/grasp/state@std_msgs/msg/String[gz.msgs.StringMsg',
        '/model/marked_connector/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose',
    ]
    controllers = Node(package='controller_manager', executable='spawner', arguments=[
        'joint_state_broadcaster', 'joint_trajectory_controller',
        'gen3_lite_2f_gripper_controller', '-c', '/controller_manager',
        '--controller-manager-timeout', '120'], output='screen')
    motion_nodes = [
        Node(package='moveit_ros_move_group', executable='move_group',
             parameters=[moveit.to_dict(), sim, {'trajectory_execution.allowed_start_tolerance':0.02}],
             output='screen'),
        Node(package='kinova_marker_pick', executable='marker_pick', parameters=[
            str(share / 'config/observation_poses.yaml'), sim, {
            'auto_execute':LaunchConfiguration('auto_execute').perform(context).lower() == 'true',
            'target_source':LaunchConfiguration('target_source').perform(context),
            'observation_offset_m':float(LaunchConfiguration('observation_offset_m').perform(context)),
            'observation_angle_rad':float(LaunchConfiguration('observation_angle_rad').perform(context)),
        }], output='screen'),
        Node(package='rviz2', executable='rviz2', arguments=['-d',str(share / 'config/workspace.rviz')],
             parameters=[moveit.to_dict(), sim], condition=IfCondition(LaunchConfiguration('rviz'))),
    ]

    def controllers_ready(event, context):
        if event.returncode:
            return [LogInfo(msg='Controller startup failed; motion will not start.'),
                    EmitEvent(event=Shutdown(reason='Controller startup failed'))]
        return motion_nodes

    return [
        *[UnsetEnvironmentVariable(name) for name in (
            'GTK_PATH', 'GTK_EXE_PREFIX', 'GTK_DATA_PREFIX', 'GIO_MODULE_DIR',
            'GIO_EXTRA_MODULES', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH')
          if '/snap/' in os.environ.get(name, '')],
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', str(share / 'models')),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(
            Path(get_package_share_directory('ros_gz_sim')) / 'launch/gz_sim.launch.py')),
            launch_arguments={'gz_args':gz_args}.items()),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description':urdf}, sim], output='screen'),
        Node(package='ros_gz_sim', executable='create', arguments=[
            '-world', 'cable_insertion', '-name', 'gen3_lite',
            '-string', sdf, '-allow_renaming', 'false'], output='screen'),
        Node(package='ros_gz_bridge', executable='parameter_bridge',
             arguments=bridge_args, output='screen'),
        Node(package='tf2_ros', executable='static_transform_publisher', arguments=[
            '--x',str(CAMERA[0]),'--y',str(CAMERA[1]),'--z',str(CAMERA[2]),
            '--roll','3.141592653589793','--pitch','0','--yaw','-1.570796326794897',
            '--frame-id','world','--child-frame-id','overhead_camera_optical_frame'],
            parameters=[sim]),
        RegisterEventHandler(OnProcessExit(target_action=controllers, on_exit=controllers_ready)),
        controllers,
    ]


def generate_launch_description():
    share = Path(get_package_share_directory('kinova_marker_pick'))
    defaults = yaml.safe_load((share / 'config/observation_poses.yaml').read_text())['marker_pick']['ros__parameters']
    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('auto_execute', default_value='false'),
        DeclareLaunchArgument('target_source', default_value='markers', choices=['markers','fixture']),
        DeclareLaunchArgument('observation_offset_m', default_value=str(defaults['observation_offset_m']),
                             description='Offset each observation-plane axis in metres; desired image stays fixed'),
        DeclareLaunchArgument('observation_angle_rad', default_value=str(defaults['observation_angle_rad']),
                             description='Initial rotation about the fixture normal in radians'),
        OpaqueFunction(function=setup),
    ])
