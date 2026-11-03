"""Metre-based fixture geometry shared by Gazebo, MoveIt and verification."""
from xml.etree import ElementTree as ET
import numpy as np
from .vision import marker_image

PICK = (0.30, -0.12, 0.120)
PORT = (0.42, 0.14, 0.25)  # opening centre on an upright YZ face
# Socket-local +Z points out of the port toward the robot (-world X).
PORT_ROTATION = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
HEAD = (0.048, 0.040, 0.026)
MARKER_SIZE = 0.032
MARKER_Z = 0.0134
PORT_MARKER_OFFSET = (0.055, 0.065, 0.0004)  # in the socket frame
TIP_Z = -0.047
INSERT_DEPTH = 0.026
GRASP_Z_OFFSET = 0.020  # keep fingertips above the socket rim at full insertion
TRANSFER_CLEAR = (0.30, -0.20, 0.34)
TURN_CORRIDOR_HALF_WIDTH = 0.05
CAMERA = (0.32, 0.0, 1.05)
EIH_CAMERA_OFFSET = (0.035, 0.075, -0.09)  # tool-frame XYZ
EIH_CAMERA_LOOK_AT = (0.0, 0.010, 0.090)


def eih_camera_rotation():
    """Tool <- optical frame, aimed below the marker toward the peg and bore."""
    forward = np.array(EIH_CAMERA_LOOK_AT)-EIH_CAMERA_OFFSET
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.,1.,0.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.column_stack((right, down, forward))


def port_transform():
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = PORT_ROTATION, PORT
    return pose


def object_at_port(port, depth):
    """Connector pose at signed tip depth; negative depth is outside the hub."""
    pose = port.copy()
    pose[:3, 3] += port[:3, :3] @ np.array([0, 0, -TIP_Z-depth])
    return pose


def insertion_metrics(connector_pose, port=None):
    port = port_transform() if port is None else port
    tip = connector_pose @ np.array([0, 0, TIP_Z, 1])
    error = port[:3, :3].T @ (tip[:3]-port[:3, 3])
    return float(np.linalg.norm(error[:2])), float(-error[2])


def child(parent, tag, text=None, **attrs):
    node = ET.SubElement(parent, tag, attrs)
    if text is not None:
        node.text = str(text)
    return node


def shape(link, name, xyz, dimensions, color, kind='box', collision=True, mask=65535):
    for tag in (['collision', 'visual'] if collision else ['visual']):
        node = child(link, tag, name=name)
        child(node, 'pose', ' '.join(map(str, (*xyz, 0, 0, 0))))
        geometry = child(child(node, 'geometry'), kind)
        if kind == 'box':
            child(geometry, 'size', ' '.join(map(str, dimensions)))
        else:
            child(geometry, 'radius', dimensions[0])
            child(geometry, 'length', dimensions[1])
        if tag == 'visual':
            material = child(node, 'material')
            child(material, 'ambient', color)
            child(material, 'diffuse', color)
        else:
            surface = child(node, 'surface')
            child(child(surface, 'contact'), 'collide_bitmask', mask)
            ode = child(child(surface, 'friction'), 'ode')
            child(ode, 'mu', 1.0)
            child(ode, 'mu2', 1.0)


def marker(link, marker_id, xyz, rotation=None):
    first = len(link)
    x, y, z = xyz
    shape(link, f'marker_{marker_id}_paper', (x, y, z - 0.0002),
          (0.042, 0.042, 0.0002), '1 1 1 1', collision=False)
    pixels = marker_image(marker_id, 600)
    for r in range(6):
        for c in range(6):
            color = '0.01 0.01 0.01 1' if pixels[r*100+50, c*100+50] < 128 else '1 1 1 1'
            shape(link, f'marker_{marker_id}_{r}_{c}',
                  (x+(c-2.5)*MARKER_SIZE/6, y+(2.5-r)*MARKER_SIZE/6, z),
                  (MARKER_SIZE/6, MARKER_SIZE/6, 0.0001), color, collision=False)
    if rotation is not None:
        origin = np.array(xyz)
        planar = np.hypot(rotation[0,0], rotation[1,0])
        pitch = np.arctan2(-rotation[2,0], planar)
        roll = np.arctan2(rotation[2,1], rotation[2,2]) if planar > 1e-8 else np.arctan2(-rotation[1,2],rotation[1,1])
        yaw = np.arctan2(rotation[1,0], rotation[0,0]) if planar > 1e-8 else 0.0
        angles = (roll, pitch, yaw)
        for visual in list(link)[first:]:
            pose = visual.find('pose')
            values = np.fromstring(pose.text, sep=' ')
            position = origin + rotation @ (values[:3]-origin)
            pose.text = ' '.join(map(str, (*position, *angles)))


def connector():
    model = ET.Element('model', name='marked_connector')
    link = child(model, 'link', name='connector')
    inertial = child(link, 'inertial')
    child(inertial, 'pose', '0.018 0 0 0 0 0')
    child(inertial, 'mass', 0.06)
    inertia = child(inertial, 'inertia')
    for tag, value in {'ixx':0.000015, 'iyy':0.00012, 'izz':0.00012,
                       'ixy':0, 'ixz':0, 'iyz':0}.items():
        child(inertia, tag, value)
    # Bit 2 excludes the robot (bit 1): fingers use idealized joint retention.
    shape(link, 'head', (0, 0, 0), HEAD, '0.12 0.35 0.55 1', mask=2)
    shape(link, 'peg', (0, 0, -0.030), (0.004, 0.034), '0.7 0.75 0.8 1', 'cylinder', mask=2)
    shape(link, 'cable', (0.082, 0, 0.003), (0.003, 0.120), '0.95 0.6 0.06 1',
          'cylinder', mask=2)
    for node in link:
        if node.get('name') == 'cable':
            node.find('pose').text = '0.082 0 0.003 0 1.57079632679 0'
    marker(link, 7, (0, 0, MARKER_Z))
    return model


def fixture_boxes():
    """Axis-aligned world boxes, identical in both collision engines."""
    px, py, _ = PICK
    boxes = [('floor', (0.25, 0, -0.025), (1.6, 1.4, 0.05))]
    for sign in (-1, 1):
        boxes.append((f'cradle_{sign}', (px+sign*0.021, py, 0.0535), (0.006, 0.032, 0.107)))
    def socket_box(name, position, size):
        position = np.array(PORT) + PORT_ROTATION @ position
        size = np.abs(PORT_ROTATION) @ size
        boxes.append((name, tuple(position), tuple(size)))
    # A 190 x 200 mm upright hub with a 14 mm square bore and a back stop.
    for sign in (-1, 1):
        socket_box(f'port_x_{sign}', [sign*0.051, 0, -0.020], [0.088, 0.200, 0.040])
        socket_box(f'port_y_{sign}', [0, sign*0.0535, -0.020], [0.014, 0.093, 0.040])
    socket_box('port_back', [0, 0, -0.042], [0.190, 0.200, 0.004])
    boxes.append(('hub_stand', (PORT[0]+0.035, PORT[1], 0.08), (0.035, 0.060, 0.160)))
    boxes.append(('hub_base', (PORT[0]+0.025, PORT[1], 0.012), (0.140, 0.240, 0.024)))
    return boxes


def write_xml(root, path):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)


def generate_assets(root):
    root.mkdir(parents=True, exist_ok=True)
    model_dir = root / 'models' / 'marked_connector'
    model_dir.mkdir(parents=True, exist_ok=True)
    sdf = ET.Element('sdf', version='1.9')
    sdf.append(connector())
    write_xml(sdf, model_dir / 'model.sdf')
    (model_dir / 'model.config').write_text(
        '<?xml version="1.0"?><model><name>marked_connector</name><version>2.0</version>'
        '<sdf version="1.9">model.sdf</sdf><description>Downward peg with rigid cable tail and ID 7</description></model>\n')
    sdf = ET.Element('sdf', version='1.9')
    world = child(sdf, 'world', name='cable_insertion')
    physics = child(world, 'physics', name='physics', type='ignored')
    child(physics, 'max_step_size', 0.001)
    child(physics, 'real_time_factor', 1)
    for filename, name in [('physics', 'Physics'), ('user-commands', 'UserCommands'),
                           ('scene-broadcaster', 'SceneBroadcaster'), ('sensors', 'Sensors')]:
        plugin = child(world, 'plugin', filename=f'gz-sim-{filename}-system',
                       name=f'gz::sim::systems::{name}')
        if name == 'Sensors':
            child(plugin, 'render_engine', 'ogre2')
    scene = child(world, 'scene')
    child(scene, 'ambient', '0.7 0.7 0.7 1')
    child(scene, 'background', '0.88 0.90 0.93 1')
    light = child(world, 'light', type='directional', name='sun')
    child(light, 'pose', '0 0 3 0 0 0')
    child(light, 'diffuse', '0.9 0.9 0.9 1')
    child(light, 'direction', '-0.3 0.2 -1')
    child(light, 'cast_shadows', 'false')
    gui = child(world, 'gui', fullscreen='false')
    plugin = child(gui, 'plugin', filename='MinimalScene', name='3D View')
    properties = child(plugin, 'gz-gui')
    child(properties, 'property', 'false', type='bool', key='showTitleBar')
    child(properties, 'property', 'docked', type='string', key='state')
    child(plugin, 'engine', 'ogre2')
    child(plugin, 'scene', 'scene')
    child(plugin, 'ambient_light', '0.7 0.7 0.7')
    child(plugin, 'background_color', '0.88 0.90 0.93')
    child(plugin, 'camera_pose', '-0.35 -0.65 0.70 0 0.49 0.85')
    for filename in ['GzSceneManager', 'InteractiveViewControl', 'CameraTracking']:
        plugin = child(gui, 'plugin', filename=filename, name=filename)
        properties = child(plugin, 'gz-gui')
        for key, value, kind in [('resizable','false','bool'), ('width','5','double'),
                                  ('height','5','double'), ('state','floating','string'),
                                  ('showTitleBar','false','bool')]:
            child(properties, 'property', value, type=kind, key=key)
    for filename in ['WorldControl', 'WorldStats']:
        plugin = child(gui, 'plugin', filename=filename, name=filename)
        properties = child(plugin, 'gz-gui')
        for key, value, kind in [('resizable','false','bool'), ('showTitleBar','false','bool'),
                                  ('height','72','double'), ('width','290','double'),
                                  ('z','1','double'), ('state','floating','string')]:
            child(properties, 'property', value, type=kind, key=key)
        anchors = child(properties, 'anchors', target='3D View')
        child(anchors, 'line', own='left' if filename == 'WorldControl' else 'right',
              target='left' if filename == 'WorldControl' else 'right')
        child(anchors, 'line', own='bottom', target='bottom')
        if filename == 'WorldControl':
            for tag in ['play_pause', 'step', 'use_event']:
                child(plugin, tag, 'true')
        else:
            child(plugin, 'real_time_factor', 'true')
            child(plugin, 'sim_time', 'true')
    plugin = child(gui, 'plugin', filename='ImageDisplay', name='Eye in hand')
    properties = child(plugin, 'gz-gui')
    child(properties, 'title', 'Eye in hand')
    for key, value, kind in [('width','540','double'), ('height','500','double'),
                              ('x','650','double'), ('y','12','double'), ('z','2','double'),
                              ('state','floating','string')]:
        child(properties, 'property', value, type=kind, key=key)
    child(plugin, 'topic', '/marker_pick/eih_debug_image')
    fixture = child(world, 'model', name='fixtures')
    child(fixture, 'static', 'true')
    link = child(fixture, 'link', name='fixtures')
    for name, pos, size in fixture_boxes():
        color = ('0.80 0.83 0.87 1' if name == 'floor' else
                 '0.14 0.55 0.32 1' if name.startswith('port') else '0.3 0.32 0.36 1')
        shape(link, name, pos, size, color)
    marker_position = np.array(PORT) + PORT_ROTATION @ PORT_MARKER_OFFSET
    marker(link, 8, marker_position, PORT_ROTATION)
    model = connector()
    child(model, 'pose', ' '.join(map(str, (*PICK, 0, 0, 0))))
    world.append(model)
    plugin = child(model, 'plugin', filename='gz-sim-pose-publisher-system',
                   name='gz::sim::systems::PosePublisher')
    child(plugin, 'publish_link_pose', 'false')
    child(plugin, 'publish_model_pose', 'true')
    child(plugin, 'use_pose_vector_msg', 'false')
    child(plugin, 'update_frequency', 30)
    camera = child(world, 'model', name='overhead_camera')
    child(camera, 'static', 'true')
    child(camera, 'pose', ' '.join(map(str, (*CAMERA, 0, 1.57079632679, 0))))
    link = child(camera, 'link', name='camera')
    sensor = child(link, 'sensor', name='camera', type='camera')
    child(sensor, 'topic', '/camera/image')
    child(sensor, 'gz_frame_id', 'overhead_camera_optical_frame')
    child(sensor, 'update_rate', 15)
    child(sensor, 'always_on', 'true')
    optics = child(sensor, 'camera')
    child(optics, 'camera_info_topic', '/camera/camera_info')
    child(optics, 'horizontal_fov', 1.04719755)
    image = child(optics, 'image')
    child(image, 'width', 1280)
    child(image, 'height', 960)
    child(image, 'format', 'R8G8B8')
    clip = child(optics, 'clip')
    child(clip, 'near', 0.05)
    child(clip, 'far', 3)
    worlds = root / 'worlds'
    worlds.mkdir(exist_ok=True)
    write_xml(sdf, worlds / 'cable_insertion.sdf')
