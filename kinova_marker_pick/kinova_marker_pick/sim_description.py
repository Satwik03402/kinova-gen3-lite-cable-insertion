"""Build a simulation-only URDF without modifying the installed Kinova package."""
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from scipy.spatial.transform import Rotation
from xml.etree import ElementTree as ET
import xacro
from ament_index_python.packages import get_package_share_directory
from .scene import child, EIH_CAMERA_OFFSET, eih_camera_rotation


def robot_description(share):
    description = Path(get_package_share_directory('kortex_description'))
    document = xacro.process_file(str(description / 'robots/kinova.urdf.xacro'), mappings={
        'name':'gen3_lite_gen3_lite_2f', 'arm':'gen3_lite', 'dof':'6', 'gripper':'gen3_lite_2f',
        'gripper_joint_name':'right_finger_bottom_joint', 'sim_gazebo':'true',
        'initial_positions_file':str(share / 'config/initial_positions.yaml'),
        'simulation_controllers':str(share / 'config/controllers.yaml'),
    })
    robot = ET.fromstring(document.toxml())
    # The upstream arm and separate gripper blocks both export the driving
    # finger interface. Keep it in the dedicated gripper hardware block only.
    for control in robot.findall('ros2_control'):
        if control.get('name') != 'GripperHardwareInterface':
            for joint in list(control.findall('joint')):
                if 'finger' in joint.get('name', ''):
                    control.remove(joint)
    for gazebo in list(robot.findall('gazebo')):
        # The provided upstream description contains a camera referencing a
        # nonexistent link on the Lite and a second world Sensors system.
        for plugin in list(gazebo.findall('plugin')):
            if 'Sensors' in plugin.get('name', ''):
                gazebo.remove(plugin)
        if gazebo.find('sensor') is not None:
            robot.remove(gazebo)
    for plugin in robot.findall('./ros2_control/hardware/plugin'):
        if plugin.text == 'gz_ros2_control/GzSimSystem':
            plugin.text = 'gz_ros2_control/GazeboSimSystem'
    camera_link = child(robot, 'link', name='eih_camera_link')
    inertial = child(camera_link, 'inertial')
    child(inertial, 'mass', value='0.025')
    child(inertial, 'inertia', ixx='0.000005', iyy='0.000005', izz='0.000005',
          ixy='0', ixz='0', iyz='0')
    for kind in ('visual', 'collision'):
        body = child(camera_link, kind)
        child(child(body, 'geometry'), 'box', size='0.030 0.040 0.020')
        if kind == 'visual':
            child(child(body, 'material', name='eih_camera_dark'), 'color', rgba='0.08 0.10 0.12 1')
    joint = child(robot, 'joint', name='eih_camera_joint', type='fixed')
    child(joint, 'parent', link='tool_frame')
    child(joint, 'child', link='eih_camera_link')
    sensor_optical = np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
    sensor_rpy = Rotation.from_matrix(eih_camera_rotation() @ sensor_optical.T).as_euler('xyz')
    child(joint, 'origin', xyz=' '.join(map(str, EIH_CAMERA_OFFSET)),
          rpy=' '.join(map(str,sensor_rpy)))
    child(robot, 'link', name='eih_camera_optical_frame')
    joint = child(robot, 'joint', name='eih_camera_optical_joint', type='fixed')
    child(joint, 'parent', link='eih_camera_link')
    child(joint, 'child', link='eih_camera_optical_frame')
    child(joint, 'origin', xyz='0 0 0', rpy='-1.570796326794897 0 -1.570796326794897')
    gazebo = child(robot, 'gazebo', reference='eih_camera_link')
    sensor = child(gazebo, 'sensor', name='eih_camera', type='camera')
    child(sensor, 'topic', '/eih/image')
    child(sensor, 'gz_frame_id', 'eih_camera_optical_frame')
    child(sensor, 'update_rate', '15')
    child(sensor, 'always_on', 'true')
    camera = child(sensor, 'camera')
    child(camera, 'camera_info_topic', '/eih/camera_info')
    child(camera, 'horizontal_fov', '1.2')
    image = child(camera, 'image')
    child(image, 'width', '1280')
    child(image, 'height', '960')
    child(image, 'format', 'R8G8B8')
    clip = child(camera, 'clip')
    child(clip, 'near', '0.025')
    child(clip, 'far', '3.0')
    gazebo = child(robot, 'gazebo')
    plugin = child(gazebo, 'plugin', filename='gz-sim-detachable-joint-system',
                   name='gz::sim::systems::DetachableJoint')
    for tag, value in {
        'parent_link':'end_effector_link', 'child_model':'marked_connector',
        'child_link':'connector', 'attach_topic':'/grasp/attach',
        'detach_topic':'/grasp/detach', 'output_topic':'/grasp/state',
    }.items():
        child(plugin, tag, value)
    return ET.tostring(robot, encoding='unicode')


def simulation_sdf(urdf):
    """Apply collision masks after URDF conversion, at valid SDF locations."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf') as source:
        source.write(urdf)
        source.flush()
        converted = subprocess.run(['gz', 'sdf', '-p', source.name],
                                   check=True, capture_output=True, text=True)
    sdf = ET.fromstring(converted.stdout)
    for collision in sdf.findall('.//collision'):
        surface = collision.find('surface')
        if surface is None:
            surface = child(collision, 'surface')
        contact = surface.find('contact')
        if contact is None:
            contact = child(surface, 'contact')
        mask = contact.find('collide_bitmask')
        if mask is None:
            mask = child(contact, 'collide_bitmask')
        mask.text = '1'
    return ET.tostring(sdf, encoding='unicode')
