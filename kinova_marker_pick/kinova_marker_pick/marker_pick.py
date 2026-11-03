"""Simulation-only marker-guided pickup, lift, alignment and peg insertion."""
from copy import deepcopy
import json
import threading
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Time
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (Constraints, PositionConstraint, OrientationConstraint,
                             CollisionObject, AttachedCollisionObject, PlanningScene,
                             AllowedCollisionEntry, PlanningSceneComponents, MoveItErrorCodes)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetCartesianPath
from shape_msgs.msg import SolidPrimitive
from control_msgs.action import GripperCommand
from .vision import Detector, estimate_pose
from .image_servo import marker_points, project, increment, apply_increment
from .alignment_overlay import render_alignment
from .scene import (PICK, PORT, HEAD, MARKER_SIZE, MARKER_Z, PORT_MARKER_OFFSET,
                    TIP_Z, INSERT_DEPTH, GRASP_Z_OFFSET, fixture_boxes,
                    PORT_ROTATION, port_transform, object_at_port, insertion_metrics,
                    TRANSFER_CLEAR, TURN_CORRIDOR_HALF_WIDTH)


class ActionFailure(RuntimeError):
    """Preserve the action's domain error, instead of hiding it behind status 6."""
    def __init__(self, status, result):
        error = getattr(result, 'error_code', None)
        self.code = getattr(error, 'val', None)
        names = {getattr(MoveItErrorCodes, name):name for name in (
            'PLANNING_FAILED', 'INVALID_MOTION_PLAN', 'CONTROL_FAILED', 'TIMED_OUT',
            'NO_IK_SOLUTION', 'START_STATE_IN_COLLISION', 'GOAL_IN_COLLISION',
            'MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE')}
        label = names.get(self.code, str(self.code))
        detail = getattr(error, 'message', '')
        super().__init__(f'Action status {status}; MoveIt {label} ({self.code}) {detail}'.strip()
                         if error is not None else f'Action failed with status {status}')


def tf_matrix(transform):
    t, q = transform.transform.translation, transform.transform.rotation
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


def pose_matrix(pose):
    matrix = np.eye(4)
    q = pose.orientation
    matrix[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return matrix


def make_pose(matrix, frame, stamp):
    msg = PoseStamped()
    msg.header.frame_id, msg.header.stamp = frame, stamp
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, matrix[:3, 3])
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = map(float, q)
    return msg


def primitive(kind, dimensions, position, rotation=None):
    solid = SolidPrimitive(type=kind, dimensions=list(map(float, dimensions)))
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = map(float, position)
    q = [0, 0, 0, 1] if rotation is None else rotation
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, q)
    return solid, pose


class MarkerPick(Node):
    def __init__(self):
        super().__init__('marker_pick')
        defaults = {
            'image_topic':'/camera/image', 'camera_info_topic':'/camera/camera_info',
            'eih_image_topic':'/eih/image', 'eih_camera_info_topic':'/eih/camera_info',
            'base_frame':'world', 'ee_link':'tool_frame', 'planning_group':'arm',
            'move_group_action':'/move_action', 'planning_pipeline':'ompl',
            'gripper_action':'/gen3_lite_2f_gripper_controller/gripper_cmd',
            'gripper_open_rad':0.0, 'gripper_closed_rad':0.60,
            'marker_id':7, 'port_marker_id':8, 'marker_size_m':MARKER_SIZE,
            'target_source':'markers', 'auto_execute':False, 'max_marker_age_s':2.0,
            'approach_distance_m':0.09, 'lift_distance_m':0.12,
            'pose_tolerance_m':0.001, 'orientation_tolerance_rad':0.003,
            'velocity_scaling':0.15, 'insertion_speed_m_s':0.01,
            'pickup_observation_xyz':[.300, -.120, .230],
            'pickup_observation_rpy':[np.pi, 0., np.pi/2],
            'port_observation_xyz':[.308, .140, .250],
            'port_observation_rpy':[0., -np.pi/2, 0.],
            'pose1_ibvs_axes':[0, 1, 5], 'pose2_ibvs_axes':[1, 2, 3],
            'ibvs_pixel_tolerance':0.9, 'ibvs_gain':.7, 'ibvs_damping':.0002,
            'ibvs_max_step_m':.003, 'ibvs_max_step_rad':.012,
            'ibvs_max_iterations':30, 'ibvs_timeout_s':100.,
            'observation_offset_m':.006, 'observation_angle_rad':.025,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        if self.p('target_source') not in ('markers', 'fixture'):
            raise ValueError('target_source must be markers or fixture')
        for name in ('pickup_observation_xyz', 'pickup_observation_rpy',
                     'port_observation_xyz', 'port_observation_rpy'):
            value = np.asarray(self.p(name))
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f'{name} must have three finite values')
        if abs(self.p('observation_offset_m')) > .015 or not np.isfinite(self.p('observation_offset_m')):
            raise ValueError('observation_offset_m must be finite and within 15 mm')
        if abs(self.p('observation_angle_rad')) > .08 or not np.isfinite(self.p('observation_angle_rad')):
            raise ValueError('observation_angle_rad must be finite and within 0.08 rad')
        for name in ('ibvs_gain', 'ibvs_damping', 'ibvs_pixel_tolerance', 'ibvs_max_step_m',
                     'ibvs_max_step_rad', 'ibvs_timeout_s', 'ibvs_max_iterations'):
            if not np.isfinite(self.p(name)) or self.p(name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name, expected in (('pose1_ibvs_axes', [0, 1, 5]), ('pose2_ibvs_axes', [1, 2, 3])):
            if list(self.p(name)) != expected:
                raise ValueError(f'{name} must be {expected} for the known fixture plane')
        self.group = ReentrantCallbackGroup()
        self.vision_group = MutuallyExclusiveCallbackGroup()
        self.bridge, self.detector = CvBridge(), Detector()
        self.calibration = {}
        self.observations = {}
        self.features = {}
        self.ibvs_metrics = {}
        self.alignment_view = None
        self.lock = threading.Lock()
        self.busy = False
        self.tick_idle_detach = True
        self.attempted = False
        self.pending = bool(self.p('auto_execute'))
        self.cancel = threading.Event()
        self.active_handle = None
        self.worker = None
        self.grasp_state = None
        self.object_pose = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pose_pub = self.create_publisher(PoseStamped, '~/head_grasp_pose', 10)
        self.port_pub = self.create_publisher(PoseStamped, '~/port_pose', 10)
        self.debug_pub = self.create_publisher(Image, '~/debug_image', qos_profile_sensor_data)
        # The ROS->Gazebo display bridge uses reliable image subscriptions.
        self.eih_debug_pub = self.create_publisher(Image, '~/eih_debug_image', 1)
        self.ibvs_pub = self.create_publisher(String, '~/ibvs', 10)
        state_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, '~/status', state_qos)
        self.attach_pub = self.create_publisher(Empty, '/grasp/attach', 10)
        self.detach_pub = self.create_publisher(Empty, '/grasp/detach', 10)
        self.create_subscription(CameraInfo, self.p('camera_info_topic'), self.on_info, qos_profile_sensor_data)
        image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, self.p('image_topic'), self.on_image, image_qos,
                                 callback_group=self.vision_group)
        self.create_subscription(CameraInfo, self.p('eih_camera_info_topic'),
                                 lambda msg:self.on_info(msg, 'eih'), qos_profile_sensor_data)
        self.create_subscription(Image, self.p('eih_image_topic'),
                                 lambda msg:self.on_image(msg, 'eih'), image_qos,
                                 callback_group=self.vision_group)
        self.create_subscription(String, '/grasp/state', self.on_grasp_state, 10)
        self.create_subscription(PoseStamped, '/model/marked_connector/pose', self.on_object_pose, 10)
        self.arm = ActionClient(self, MoveGroup, self.p('move_group_action'), callback_group=self.group)
        self.execute = ActionClient(self, ExecuteTrajectory, '/execute_trajectory', callback_group=self.group)
        self.gripper = ActionClient(self, GripperCommand, self.p('gripper_action'), callback_group=self.group)
        self.apply_scene = self.create_client(ApplyPlanningScene, '/apply_planning_scene', callback_group=self.group)
        self.get_scene = self.create_client(GetPlanningScene, '/get_planning_scene', callback_group=self.group)
        self.cartesian = self.create_client(GetCartesianPath, '/compute_cartesian_path', callback_group=self.group)
        self.create_service(Trigger, '~/start', self.start, callback_group=self.group)
        self.create_service(Trigger, '~/cancel', self.stop, callback_group=self.group)
        self.create_timer(0.3, self.tick)
        self.status('ready', 'Waiting for simulation; call /marker_pick/start to execute once.')

    def p(self, name):
        return self.get_parameter(name).value

    def status(self, stage, detail='', **metrics):
        self.current_stage = stage
        if not rclpy.ok():
            return
        self.status_pub.publish(String(data=json.dumps({'stage':stage, 'detail':detail, **metrics})))
        self.get_logger().info(f'{stage}: {detail}')

    def on_grasp_state(self, msg):
        self.grasp_state = msg.data

    def on_object_pose(self, msg):
        self.object_pose = (deepcopy(msg.pose), self.get_clock().now())

    def on_info(self, msg, source='overhead'):
        matrix = np.asarray(msg.k, dtype=float).reshape(3, 3)
        if matrix[0, 0] > 0 and matrix[1, 1] > 0:
            self.calibration[source] = (matrix, np.asarray(msg.d, dtype=float), msg.header.frame_id)

    def on_image(self, msg, source='overhead'):
        calibration = self.calibration.get(source)
        if calibration is None or not msg.header.frame_id:
            return
        try:
            matrix, distortion, frame_id = calibration
            if frame_id and frame_id != msg.header.frame_id:
                return
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            corners, ids, _ = self.detector.detect(image)
            visible = {}
            if ids is not None:
                for index, marker_id in enumerate(ids.flatten()):
                    allowed = ([self.p('marker_id')] if source == 'overhead' else
                               [self.p('marker_id'), self.p('port_marker_id')])
                    if int(marker_id) not in allowed:
                        continue
                    estimate = estimate_pose(corners[index], self.p('marker_size_m'), matrix, distortion)
                    if estimate is not None:
                        visible[int(marker_id)] = np.asarray(corners[index], dtype=float).reshape(4, 2)
                        with self.lock:
                            self.features[(source, int(marker_id))] = (
                                np.asarray(corners[index], dtype=float).reshape(4, 2).copy(),
                                estimate, msg.header.frame_id, msg.header.stamp, time.monotonic())
                            if source == 'overhead' or int(marker_id) == self.p('port_marker_id'):
                                self.observations[int(marker_id)] = (estimate, msg.header.frame_id, msg.header.stamp)
                cv2.aruco.drawDetectedMarkers(image, corners, ids)
            publisher = self.debug_pub if source == 'overhead' else self.eih_debug_pub
            if publisher.get_subscription_count():
                if source == 'eih':
                    with self.lock:
                        view = self.alignment_view
                    stage = getattr(self, 'current_stage', 'ready')
                    active = view is not None and stage in (view['phase'], view['phase']+'_complete')
                    wall_stage = stage.startswith('pose2') or stage in ('insert', 'complete')
                    if stage == 'aborted' and view is not None:
                        wall_stage = view['marker_id'] == self.p('port_marker_id')
                    marker_id = view['marker_id'] if active else self.p('port_marker_id' if wall_stage else 'marker_id')
                    image = render_alignment(image, visible.get(marker_id),
                        view['desired'] if active else None, marker_id, stage,
                        self.p('ibvs_pixel_tolerance'), view.get('converged', False) if active else False)
                if source == 'eih':
                    # Gazebo ImageDisplay supports RGB_INT8, not BGR_INT8.
                    debug = self.bridge.cv2_to_imgmsg(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), encoding='rgb8')
                else:
                    debug = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
                debug.header = msg.header
                publisher.publish(debug)
        except (cv2.error, ValueError) as exc:
            self.get_logger().warning(f'Vision: {exc}', throttle_duration_sec=5)

    def marker_world(self, marker_id):
        with self.lock:
            observation = self.observations.get(marker_id)
        if observation is None:
            return None
        estimate, camera_frame, stamp = observation
        age = (self.get_clock().now() - Time.from_msg(stamp)).nanoseconds / 1e9
        if age < 0 or age > self.p('max_marker_age_s'):
            return None
        # Image-time TF is essential when the same detector is used eye-in-hand.
        try:
            transform = self.tf_buffer.lookup_transform(self.p('base_frame'), camera_frame, Time.from_msg(stamp))
        except Exception:
            return None
        camera_world = tf_matrix(transform)
        marker = camera_world @ estimate
        # Both fixture planes are known in this first simulation. Intersect the
        # observed ray with that plane to remove monocular PnP depth noise.
        if marker_id == self.p('marker_id'):
            normal = np.array([0.,0.,1.])
            plane_point = np.array(PICK) + [0,0,MARKER_Z]
        else:
            normal = PORT_ROTATION[:,2]
            plane_point = np.array(PORT) + PORT_ROTATION @ PORT_MARKER_OFFSET
        origin = camera_world[:3,3]
        ray = marker[:3,3]-origin
        denominator = float(normal @ ray)
        if abs(denominator) < 1e-6:
            return None
        distance = float(normal @ (plane_point-origin))/denominator
        if distance <= 0:
            return None
        marker[:3,3] = origin + distance*ray
        return marker

    def port_target(self):
        if self.p('target_source') == 'fixture':
            return port_transform()
        else:
            port = self.marker_world(self.p('port_marker_id'))
            if port is None:
                return None
            # The hub is constrained vertical. Its marker supplies lateral
            # position; its known normal sets the horizontal insertion axis.
            port[:3, 3] -= PORT_ROTATION @ PORT_MARKER_OFFSET
            port[:3,:3] = PORT_ROTATION
            normal = PORT_ROTATION[:,2]
            port[:3,3] += normal * (normal @ (np.array(PORT)-port[:3,3]))
        return port

    def targets(self):
        # A nominal fixture pose provides the observation approach; the wrist
        # camera acquires ID 8 after the held connector has been rotated.
        port = port_transform()
        # Start at the predefined observation pose before requiring visibility.
        pick = np.eye(4)
        pick[:3, 3] = PICK
        # The cradle constrains pickup height and upright orientation.
        yaw = np.arctan2(pick[1, 0], pick[0, 0])
        # Tool X is the jaw-closing direction. The cable leaves along marker X,
        # so close across marker Y to keep the tail out of the fingers.
        pick[:3, :3] = Rotation.from_euler('xyz', [np.pi, 0, yaw+np.pi/2]).as_matrix()
        pick[2, 3] = PICK[2] + GRASP_Z_OFFSET
        return pick, port

    def tick(self):
        if not self.busy and self.tick_idle_detach:
            # DetachableJoint starts attached; detach before any robot motion.
            self.detach_pub.publish(Empty())
        targets = self.targets()
        if targets is not None and not self.busy:
            now = self.get_clock().now().to_msg()
            self.pose_pub.publish(make_pose(targets[0], self.p('base_frame'), now))
            self.port_pub.publish(make_pose(targets[1], self.p('base_frame'), now))
        if self.p('auto_execute'):
            self.pending = True
            self.set_parameters([rclpy.parameter.Parameter('auto_execute', value=False)])
        if self.pending and targets is None and not self.attempted:
            self.get_logger().warning('Waiting for a fresh pickup marker observation', throttle_duration_sec=10)
        if self.pending and not self.busy and not self.attempted and targets is not None:
            self.pending = False
            self.attempted = True
            self.busy = True
            self.cancel.clear()
            self.worker = threading.Thread(target=self.run_sequence, args=targets, daemon=True)
            self.worker.start()

    def start(self, request, response):
        if self.attempted:
            response.success, response.message = False, 'Attempt already used; relaunch the scene to reset'
        elif self.busy:
            response.success, response.message = False, 'Sequence already running'
        else:
            self.pending = True
            response.success, response.message = True, 'One attempt queued; waiting for fresh targets and controllers'
        return response

    def stop(self, request, response):
        self.pending = False
        self.cancel.set()
        if self.active_handle is not None:
            self.active_handle.cancel_goal_async()
        response.success, response.message = True, 'Cancellation requested; held object stays attached'
        return response

    def wait(self, future, timeout=30):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            if self.cancel.wait(0.02):
                raise RuntimeError('Cancelled')
        if self.cancel.is_set():
            raise RuntimeError('Cancelled')
        if not future.done():
            if self.active_handle is not None:
                self.active_handle.cancel_goal_async()
            raise RuntimeError('Operation timed out')
        return future.result()

    def service(self, client, request):
        if not client.wait_for_service(timeout_sec=30):
            raise RuntimeError(f'Service unavailable: {client.srv_name}')
        return self.wait(client.call_async(request))

    def action(self, client, goal, timeout=120):
        if not client.wait_for_server(timeout_sec=30):
            raise RuntimeError('Action server unavailable')
        def feedback(message):
            if client is self.gripper:
                self.get_logger().info(f'Gripper position: {message.feedback.position:.3f}',
                                       throttle_duration_sec=5)
        handle = self.wait(client.send_goal_async(goal, feedback_callback=feedback))
        if not handle.accepted:
            raise RuntimeError('Action goal rejected')
        self.active_handle = handle
        try:
            result = self.wait(handle.get_result_async(), timeout)
            if result.status != 4:  # action_msgs/GoalStatus.STATUS_SUCCEEDED
                raise ActionFailure(result.status, result.result)
            return result.result
        finally:
            self.active_handle = None

    def move_to(self, target, label, lateral_corridor=None):
        self.status(label)
        goal = MoveGroup.Goal()
        request = goal.request
        request.group_name, request.pipeline_id = self.p('planning_group'), self.p('planning_pipeline')
        request.num_planning_attempts, request.allowed_planning_time = 10, 10.0
        request.max_velocity_scaling_factor = self.p('velocity_scaling')
        request.max_acceleration_scaling_factor = self.p('velocity_scaling')
        request.start_state.is_diff = True
        pose = make_pose(target, self.p('base_frame'), self.get_clock().now().to_msg())
        # Image references assume the fixed height/wall distance and tilt at
        # each observation pose. Keep those uncontrolled axes precise enough
        # that image alignment is not left with an irreducible scale/tilt error.
        observation = label in ('pose1_above_peg', 'pose2_in_front_of_wall')
        position_tolerance = min(self.p('pose_tolerance_m'), .0002) if observation else self.p('pose_tolerance_m')
        angle_tolerance = min(self.p('orientation_tolerance_rad'), .0005) if observation else self.p('orientation_tolerance_rad')
        constraints = Constraints(name=label)
        position = PositionConstraint(header=pose.header, link_name=self.p('ee_link'), weight=1.0)
        sphere, centre = primitive(SolidPrimitive.SPHERE, [position_tolerance], target[:3, 3])
        position.constraint_region.primitives = [sphere]
        position.constraint_region.primitive_poses = [centre]
        orientation = OrientationConstraint(header=pose.header, link_name=self.p('ee_link'),
                                             orientation=pose.pose.orientation, weight=1.0)
        orientation.absolute_x_axis_tolerance = angle_tolerance
        orientation.absolute_y_axis_tolerance = angle_tolerance
        orientation.absolute_z_axis_tolerance = angle_tolerance
        constraints.position_constraints, constraints.orientation_constraints = [position], [orientation]
        request.goal_constraints = [constraints]
        if lateral_corridor is not None:
            y, half_width = lateral_corridor
            box, centre = primitive(SolidPrimitive.BOX, [1.5, 2*half_width, 1.5], [0.25,y,0.5])
            corridor = PositionConstraint(header=pose.header, link_name=self.p('ee_link'), weight=1.0)
            corridor.constraint_region.primitives = [box]
            corridor.constraint_region.primitive_poses = [centre]
            request.path_constraints = Constraints(name='turn_away_from_hub', position_constraints=[corridor])
        for attempt in range(3):
            if self.cancel.is_set():
                raise RuntimeError('Cancelled')
            try:
                result = self.action(self.arm, goal)
                break
            except ActionFailure as exc:
                # Only planning rejections are retryable; controller failures,
                # cancellation and timeouts never cause another execution.
                if exc.code not in (MoveItErrorCodes.PLANNING_FAILED,
                                    MoveItErrorCodes.INVALID_MOTION_PLAN,
                                    MoveItErrorCodes.NO_IK_SOLUTION) or attempt == 2:
                    raise RuntimeError(f'{label}: {exc}') from exc
                self.get_logger().warning(f'{label}: {exc}; replanning {attempt+2}/3')
        if result.error_code.val != 1:
            raise RuntimeError(f'{label}: MoveIt error {result.error_code.val}')

    def straight_to(self, target, label, speed=0.04):
        self.status(label)
        request = GetCartesianPath.Request()
        request.header.frame_id = self.p('base_frame')
        request.start_state.is_diff = True
        request.group_name, request.link_name = self.p('planning_group'), self.p('ee_link')
        request.waypoints = [make_pose(target, self.p('base_frame'), self.get_clock().now().to_msg()).pose]
        request.max_step, request.revolute_jump_threshold = 0.002, 0.2
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = self.p('velocity_scaling')
        request.max_acceleration_scaling_factor = self.p('velocity_scaling')
        request.cartesian_speed_limited_link, request.max_cartesian_speed = self.p('ee_link'), speed
        result = self.service(self.cartesian, request)
        if result.error_code.val != 1 or result.fraction < 0.999:
            raise RuntimeError(f'{label}: incomplete Cartesian path ({result.fraction:.1%}); no motion executed')
        # Respect actual returned joint continuity too, across ROS/MoveIt versions.
        points = result.solution.joint_trajectory.points
        if not points or any(np.max(np.abs(np.array(b.positions)-a.positions)) > 0.2
                             for a, b in zip(points, points[1:])):
            raise RuntimeError(f'{label}: empty path or joint discontinuity')
        outcome = self.action(self.execute, ExecuteTrajectory.Goal(trajectory=result.solution))
        if outcome.error_code.val != 1:
            raise RuntimeError(f'{label}: execution error {outcome.error_code.val}')

    def observation_pose(self, name):
        result = np.eye(4)
        result[:3, 3] = self.p(name+'_observation_xyz')
        result[:3, :3] = Rotation.from_euler('xyz', self.p(name+'_observation_rpy')).as_matrix()
        return result

    def tool_world(self, stamp=None):
        return tf_matrix(self.tf_buffer.lookup_transform(
            self.p('base_frame'), self.p('ee_link'), Time() if stamp is None else Time.from_msg(stamp)))

    def image_align(self, marker_id, desired_tool_marker, axes, label):
        """Sampled IBVS: measure pixels, execute a small checked increment, remeasure.

        Each correction is computed from image error, not a 3D marker pose error.
        Fresh images after execution are mandatory. The known fixture plane fixes
        the remaining axes while the image loop corrects translation and rotation.
        """
        self.status(label, f'Wrist IBVS on marker {marker_id}; waiting for fresh corners')
        origin = self.tool_world()
        after = self.get_clock().now().nanoseconds
        deadline = time.monotonic()+self.p('ibvs_timeout_s')
        missing_since = time.monotonic()
        stable = 0
        updates = 0
        initial_error = None
        while time.monotonic() < deadline and updates < self.p('ibvs_max_iterations'):
            if self.cancel.is_set():
                raise RuntimeError('Cancelled')
            with self.lock:
                sample = self.features.get(('eih', marker_id))
            if sample is None:
                valid = False
            else:
                corners, estimate, frame, stamp, received = sample
                stamp_ns = Time.from_msg(stamp).nanoseconds
                age = (self.get_clock().now().nanoseconds-stamp_ns)/1e9
                valid = stamp_ns > after and -.05 <= age < .75 and time.monotonic()-received < .75
            if not valid:
                if time.monotonic()-missing_since > 4.:
                    raise RuntimeError(f'{label}: lost fresh wrist marker {marker_id}; no correction executed')
                self.cancel.wait(.03)
                continue
            try:
                world_tool = self.tool_world(stamp)
                world_camera = tf_matrix(self.tf_buffer.lookup_transform(self.p('base_frame'), frame, Time.from_msg(stamp)))
            except Exception:
                if time.monotonic()-missing_since > 4.:
                    raise RuntimeError(f'{label}: image-time transforms unavailable; no correction executed')
                self.cancel.wait(.03)
                continue
            missing_since = time.monotonic()
            matrix, distortion, _ = self.calibration['eih']
            tool_camera = np.linalg.inv(world_tool)@world_camera
            desired, _ = project(np.linalg.inv(tool_camera)@desired_tool_marker,
                                 self.p('marker_size_m'), matrix, distortion)
            xyz = marker_points(self.p('marker_size_m'))@estimate[:3, :3].T+estimate[:3, 3]
            delta, error = increment(corners, desired, xyz[:, 2], matrix, distortion,
                world_tool, world_camera, axes, gain=self.p('ibvs_gain'), damping=self.p('ibvs_damping'),
                max_translation=self.p('ibvs_max_step_m'), max_rotation=self.p('ibvs_max_step_rad'))
            updates += 1
            if initial_error is None:
                initial_error = error
            stable = stable+1 if error <= self.p('ibvs_pixel_tolerance') else 0
            with self.lock:
                self.alignment_view = {'phase':label, 'marker_id':marker_id,
                                       'desired':desired.copy(), 'converged':stable >= 3}
            self.ibvs_metrics.update({label+'_initial_error_px':initial_error,
                label+'_final_error_px':error, label+'_updates':updates, label+'_converged':stable >= 3})
            self.ibvs_pub.publish(String(data=json.dumps({'phase':label, 'marker_id':marker_id,
                'stamp_ns':stamp_ns, 'error_px':error, 'iteration':updates,
                'corners_px':corners.tolist(), 'desired_corners_px':desired.tolist(),
                'tool_increment':delta.tolist()})))
            self.get_logger().info(f'{label}: {error:.3f} px, iteration {updates}')
            if stable >= 3:
                self.status(label+'_complete', f'IBVS converged at {error:.3f} px', **self.ibvs_metrics)
                # Let the live camera show its confirmed ALIGNED state before
                # advancing to descent/transfer and hiding the observation goal.
                if self.cancel.wait(1.0):
                    raise RuntimeError('Cancelled')
                return self.tool_world()
            after = stamp_ns
            if stable:
                continue
            target = apply_increment(world_tool, delta)
            if (np.linalg.norm(target[:3, 3]-origin[:3, 3]) > .025 or
                    Rotation.from_matrix(origin[:3, :3].T@target[:3, :3]).magnitude() > .14):
                raise RuntimeError(f'{label}: correction exceeded the 25 mm / 8 degree observation envelope')
            self.straight_to(target, label, .015)
            after = self.get_clock().now().nanoseconds+50_000_000
            missing_since = time.monotonic()
        raise RuntimeError(f'{label}: IBVS did not converge; no subsequent task motion executed')

    def grip(self, close):
        goal = GripperCommand.Goal()
        goal.command.position = self.p('gripper_closed_rad' if close else 'gripper_open_rad')
        goal.command.max_effort = 10.0
        result = self.action(self.gripper, goal, 20)
        if not result.reached_goal and not (close and result.stalled):
            raise RuntimeError('Gripper did not reach its command')

    def set_attachment(self, attached):
        desired = 'attached' if attached else 'detached'
        publisher = self.attach_pub if attached else self.detach_pub
        deadline = time.monotonic() + 60
        while self.grasp_state != desired:
            publisher.publish(Empty())
            if self.cancel.wait(0.1) or time.monotonic() > deadline:
                raise RuntimeError(f'No Gazebo grasp confirmation: {desired}')

    def object_world(self):
        value = self.object_pose
        if value is None or (self.get_clock().now()-value[1]).nanoseconds > 2e9:
            raise RuntimeError('Fresh Gazebo object pose unavailable')
        return pose_matrix(value[0])

    def apply(self, scene):
        scene.is_diff = True
        scene.robot_state.is_diff = True
        if not self.service(self.apply_scene, ApplyPlanningScene.Request(scene=scene)).success:
            raise RuntimeError('Planning scene update rejected')

    def setup_scene(self):
        scene = PlanningScene()
        for name, position, dimensions in fixture_boxes():
            obj = CollisionObject(id=name)
            obj.header.frame_id = self.p('base_frame')
            obj.pose.orientation.w = 1.0
            solid, pose = primitive(SolidPrimitive.BOX, dimensions, position)
            obj.primitives, obj.primitive_poses = [solid], [pose]
            scene.world.collision_objects.append(obj)
        # Keep the existing SRDF collision permissions, adding only finger/part
        # and fixture support contacts needed by this task.
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        matrix = self.service(self.get_scene, request).scene.allowed_collision_matrix
        fingers = ['gripper_base_link','left_finger_dist_link','left_finger_prox_link',
                   'right_finger_dist_link','right_finger_prox_link','end_effector_link']
        pairs = [('marked_connector', name) for name in fingers]
        pairs += [('marked_connector', f'cradle_{s}') for s in (-1, 1)]
        pairs += [('base_link', 'floor')]
        for a, b in pairs:
            for name in (a, b):
                if name not in matrix.entry_names:
                    matrix.entry_names.append(name)
                    for row in matrix.entry_values:
                        row.enabled.append(False)
                    matrix.entry_values.append(AllowedCollisionEntry(enabled=[False]*len(matrix.entry_names)))
            i, j = matrix.entry_names.index(a), matrix.entry_names.index(b)
            matrix.entry_values[i].enabled[j] = matrix.entry_values[j].enabled[i] = True
        scene.allowed_collision_matrix = matrix
        self.apply(scene)
        self.update_object(False)

    def update_object(self, attached):
        scene = PlanningScene()
        obj = CollisionObject(id='marked_connector')
        obj.header.frame_id = self.p('base_frame')
        obj.pose.orientation.w = 1.0
        transform = self.object_world()
        # Include head, peg and rigid tail in the planner as well as Gazebo.
        for kind, dims, pos, rot in [
            (SolidPrimitive.BOX, HEAD, (0,0,0), None),
            (SolidPrimitive.CYLINDER, (0.034,0.004), (0,0,-0.030), None),
            (SolidPrimitive.CYLINDER, (0.120,0.003), (0.082,0,0.003),
             Rotation.from_euler('y', np.pi/2).as_quat()),
        ]:
            solid, pose = primitive(kind, dims, pos, rot)
            world_pose = make_pose(transform @ pose_matrix(pose), self.p('base_frame'),
                                   self.get_clock().now().to_msg()).pose
            obj.primitives.append(solid)
            obj.primitive_poses.append(world_pose)
        if attached:
            # MoveIt removes the world object as part of attaching it. A second
            # explicit REMOVE in the same diff makes ApplyPlanningScene fail.
            attachment = AttachedCollisionObject(link_name=self.p('ee_link'), object=obj)
            attachment.touch_links = ['end_effector_link','gripper_base_link','tool_frame',
                'left_finger_dist_link','left_finger_prox_link','right_finger_dist_link','right_finger_prox_link']
            scene.robot_state.attached_collision_objects = [attachment]
        else:
            scene.world.collision_objects = [obj]
        self.apply(scene)

    def run_sequence(self, pick, port):
        try:
            self.status('initializing', 'Waiting for controllers and Gazebo grasp state')
            self.set_attachment(False)
            self.setup_scene()
            self.grip(False)
            approach = self.observation_pose('pickup')
            commanded = approach.copy()
            offset = self.p('observation_offset_m')
            commanded[:2, 3] += [offset, -offset]
            angle = self.p('observation_angle_rad')
            commanded[:3, :3] = Rotation.from_rotvec([0., 0., angle]).as_matrix()@commanded[:3, :3]
            self.move_to(commanded, 'pose1_above_peg')
            if self.p('target_source') == 'markers':
                marker = np.eye(4)
                marker[:3, 3] = np.array(PICK)+[0., 0., MARKER_Z]
                aligned = self.image_align(self.p('marker_id'), np.linalg.inv(approach)@marker,
                                           self.p('pose1_ibvs_axes'), 'pose1_ibvs')
                pick = aligned.copy()
                pick[2, 3] = PICK[2]+GRASP_Z_OFFSET
            self.straight_to(pick, 'grasp')
            self.grip(True)
            actual_tool = tf_matrix(self.tf_buffer.lookup_transform(self.p('base_frame'), self.p('ee_link'), Time()))
            object_transform = self.object_world()
            grasp_centre = actual_tool[:3, 3] + actual_tool[:3, :3] @ np.array([0,0,GRASP_Z_OFFSET])
            if np.linalg.norm(grasp_centre-object_transform[:3, 3]) > 0.015:
                raise RuntimeError('Object outside the grasp region')
            tool_object = np.linalg.inv(actual_tool) @ object_transform
            self.set_attachment(True)
            self.update_object(True)
            lift = pick.copy()
            lift[2, 3] += self.p('lift_distance_m')
            self.straight_to(lift, 'lift')
            if self.object_world()[2, 3] < PICK[2] + 0.06:
                raise RuntimeError('Gazebo did not confirm object lift')
            # Move away from the upright hub before rotating the rigid tail.
            # Keep tool Y in a free corridor during the turn. Plan the transfer
            # separately so joint limits can be handled before final alignment.
            clear_object = self.object_world()
            clear_object[:3,3] = TRANSFER_CLEAR
            self.straight_to(clear_object @ np.linalg.inv(tool_object), 'transfer_clearance')
            clear_object[:3,:3] = port[:3,:3]
            self.move_to(clear_object @ np.linalg.inv(tool_object), 'orient_connector',
                         lateral_corridor=(TRANSFER_CLEAR[1], TURN_CORRIDOR_HALF_WIDTH))
            desired_object = self.observation_pose('port')
            preinsert = desired_object @ np.linalg.inv(tool_object)
            commanded = preinsert.copy()
            commanded[1:3, 3] += [offset, -offset]
            commanded[:3, :3] = Rotation.from_rotvec([angle, 0., 0.]).as_matrix()@commanded[:3, :3]
            self.move_to(commanded, 'pose2_in_front_of_wall')
            if self.p('target_source') == 'markers':
                marker = port.copy()
                marker[:3, 3] += port[:3, :3]@PORT_MARKER_OFFSET
                aligned = self.image_align(self.p('port_marker_id'), np.linalg.inv(preinsert)@marker,
                                           self.p('pose2_ibvs_axes'), 'pose2_ibvs')
            else:
                aligned = preinsert
                self.straight_to(aligned, 'align', .02)
            # Preserve the image-aligned lateral position and roll during the
            # final checked plunge; use the known wall plane for axial travel.
            tip = (aligned@tool_object)@np.array([0., 0., TIP_Z, 1.])
            current_depth = -float(port[:3, 2]@(tip[:3]-port[:3, 3]))
            inserted = aligned.copy()
            inserted[:3, 3] -= port[:3, 2]*(INSERT_DEPTH-current_depth)
            self.straight_to(inserted, 'insert', self.p('insertion_speed_m_s'))
            self.cancel.wait(0.3)
            measured = self.object_world()
            lateral, depth = insertion_metrics(measured)
            if lateral > 0.0025 or abs(depth-INSERT_DEPTH) > 0.004:
                raise RuntimeError(f'Insertion check failed: lateral={lateral:.4f} m, depth={depth:.4f} m')
            # Keep holding the connector in the port for this initial demo.
            self.status('complete', f'Gazebo verified lift and horizontal insertion: lateral={lateral:.4f} m, depth={depth:.4f} m; holding connector',
                        lateral_error_m=lateral, insertion_depth_m=depth, port_view='eye_in_hand',
                        **self.ibvs_metrics)
        except Exception as exc:
            failed_stage = getattr(self, 'current_stage', 'unknown')
            self.status('aborted', f'{failed_stage}: {exc}', failed_stage=failed_stage, **self.ibvs_metrics)
        finally:
            # Do not detach a held object on the next idle timer tick.
            self.tick_idle_detach = False
            self.busy = False


def main(args=None):
    rclpy.init(args=args)
    node = MarkerPick()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.cancel.set()
        if node.active_handle is not None and rclpy.ok():
            node.active_handle.cancel_goal_async()
        if node.worker is not None:
            node.worker.join(timeout=2)
        executor.shutdown(timeout_sec=2)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
