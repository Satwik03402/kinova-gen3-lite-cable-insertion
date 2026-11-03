"""Regression checks for calibration, asset consistency and simulation wiring."""
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from kinova_marker_pick.vision import Detector, marker_image, estimate_pose
from kinova_marker_pick.scene import (generate_assets, fixture_boxes, PORT, PICK,
                                      TIP_Z, INSERT_DEPTH, PORT_ROTATION, port_transform, object_at_port, insertion_metrics)
from kinova_marker_pick.marker_pick import MarkerPick, make_pose, pose_matrix


def test_local_opencv_detector_recognizes_both_markers():
    detector = Detector()
    for marker_id in (7, 8):
        image = np.full((400, 400), 255, dtype=np.uint8)
        image[100:300, 100:300] = marker_image(marker_id, 200)
        corners, ids, _ = detector.detect(image)
        assert ids.flatten().tolist() == [marker_id]
        assert len(corners) == 1


def test_pose_recovery_for_frontal_and_oblique_marker():
    matrix = np.array([[900.,0,640],[0,900.,480],[0,0,1]])
    size = .032
    half = size/2
    obj = np.array([[-half,half,0],[half,half,0],[half,-half,0],[-half,-half,0]])
    for rpy in ([np.pi,0,0], [2.9,.15,-.2]):
        rotation = Rotation.from_euler('xyz',rpy)
        translation = np.array([.04,-.08,.7])
        points, _ = cv2.projectPoints(obj,rotation.as_rotvec(),translation,matrix,np.zeros(5))
        result = estimate_pose(points,size,matrix,np.zeros(5))
        np.testing.assert_allclose(result[:3,3],translation,atol=1e-6)
        np.testing.assert_allclose(result[:3,:3],rotation.as_matrix(),atol=1e-6)


def test_pose_rejects_corrupted_corner_correspondence():
    points=np.array([[100.,100],[140,100],[140,140],[110,150]])
    matrix=np.array([[900.,0,640],[0,900.,480],[0,0,1]])
    assert estimate_pose(points,.032,matrix,np.zeros(5),max_error=.1) is None


def test_grasp_transform_roundtrip_preserves_actual_held_offset():
    tool=np.eye(4)
    tool[:3,:3]=Rotation.from_euler('xyz',[np.pi,0,.2]).as_matrix()
    tool[:3,3]=PICK
    obj=np.eye(4)
    obj[:3,3]=np.array(PICK)+[.001,-.002,0]
    held=np.linalg.inv(tool)@obj
    desired=object_at_port(port_transform(),INSERT_DEPTH)
    new_tool=desired@np.linalg.inv(held)
    np.testing.assert_allclose((new_tool@held)@np.array([0,0,TIP_Z,1]),
                               [*((np.array(PORT)-PORT_ROTATION[:,2]*INSERT_DEPTH).tolist()),1],atol=1e-9)


def test_scene_assets_and_planning_fixtures_match(tmp_path):
    generate_assets(tmp_path)
    world=ET.parse(tmp_path/'worlds/cable_insertion.sdf')
    link=world.find(".//model[@name='fixtures']/link")
    from kinova_marker_pick.scene import PORT_MARKER_OFFSET
    paper=link.find("visual[@name='marker_8_paper']/pose")
    np.testing.assert_allclose(np.fromstring(paper.text,sep=' ')[:2],
                               (np.array(PORT)+PORT_ROTATION@(np.array(PORT_MARKER_OFFSET)-[0,0,.0002]))[:2])
    for name,position,size in fixture_boxes():
        collision=link.find(f"collision[@name='{name}']")
        np.testing.assert_allclose(np.fromstring(collision.findtext('pose'),sep=' ')[:3],position)
        np.testing.assert_allclose(np.fromstring(collision.findtext('geometry/box/size'),sep=' '),size)
    # The vertical hub has an open 14 mm bore in its YZ face.
    for collision in link.findall('collision'):
        if collision.get('name').startswith(('port_x','port_y')):
            pos=np.fromstring(collision.findtext('pose'),sep=' ')
            size=np.fromstring(collision.findtext('geometry/box/size'),sep=' ')
            assert np.any(np.abs(pos[1:3]-PORT[1:3])-size[1:3]/2 > .0069)


def test_simulation_description_has_unique_interfaces_and_no_real_driver():
    from kinova_marker_pick.sim_description import robot_description
    share=Path(__file__).resolve().parents[1]
    robot=ET.fromstring(robot_description(share))
    interfaces=[]
    for control in robot.findall('ros2_control'):
        for joint in control.findall('joint'):
            for interface in joint.findall('command_interface'):
                interfaces.append((joint.get('name'),interface.get('name')))
        assert control.findtext('hardware/plugin') == 'gz_ros2_control/GazeboSimSystem'
    assert len(interfaces) == len(set(interfaces))
    assert robot.get('name') == 'gen3_lite_gen3_lite_2f'


def test_partial_cartesian_path_is_never_executed():
    calls=[]
    values={'base_frame':'world','planning_group':'arm','ee_link':'tool_frame','velocity_scaling':.1}
    fake=SimpleNamespace(status=lambda *args:None, p=values.get,
        get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(to_msg=lambda:__import__('builtin_interfaces.msg',fromlist=['Time']).Time())),
        cartesian=None, service=lambda *args:SimpleNamespace(error_code=SimpleNamespace(val=1),fraction=.8),
        action=lambda *args:calls.append(args))
    with pytest.raises(RuntimeError,match='incomplete Cartesian'):
        MarkerPick.straight_to(fake,np.eye(4),'insert')
    assert calls == []


def test_robot_masks_survive_urdf_to_sdf_conversion():
    from kinova_marker_pick.sim_description import robot_description, simulation_sdf
    share=Path(__file__).resolve().parents[1]
    root=ET.fromstring(simulation_sdf(robot_description(share)))
    collisions=root.findall('.//collision')
    assert collisions
    for collision in collisions:
        assert collision.findtext('surface/contact/collide_bitmask') == '1'
    assert not root.findall('.//link/collide_bitmask')


def test_grasp_closes_across_head_and_has_rim_clearance():
    from kinova_marker_pick.scene import GRASP_Z_OFFSET
    values={'target_source':'fixture'}
    fake=SimpleNamespace(p=values.get,port_target=lambda:np.eye(4))
    grasp,_=MarkerPick.targets(fake)
    # Jaw X and cable X are perpendicular in world coordinates.
    assert abs(grasp[:3,0] @ np.array([1,0,0])) < 1e-6
    assert grasp[2,3] == pytest.approx(PICK[2]+GRASP_Z_OFFSET)
    head=object_at_port(port_transform(),INSERT_DEPTH)
    tip_clearance=-TIP_Z-INSERT_DEPTH+GRASP_Z_OFFSET-.035
    assert tip_clearance > 0
    np.testing.assert_allclose(head[:3,:3]@np.array([0,0,-1]), [1,0,0],atol=1e-9)


def test_port_refresh_does_not_require_visible_pickup_marker():
    from kinova_marker_pick.scene import PORT_MARKER_OFFSET
    values={'target_source':'markers','port_marker_id':8}
    marker=port_transform()
    marker[:3,3]=np.array(PORT)+PORT_ROTATION@PORT_MARKER_OFFSET
    requested=[]
    def observe(marker_id):
        requested.append(marker_id)
        return marker.copy()
    fake=SimpleNamespace(p=values.get,marker_world=observe)
    port=MarkerPick.port_target(fake)
    np.testing.assert_allclose(port[:3,3],PORT)
    assert requested == [8]


def test_horizontal_insertion_metrics_reject_vertical_axis_assumption():
    pose=object_at_port(port_transform(),INSERT_DEPTH)
    lateral,depth=insertion_metrics(pose)
    assert lateral == pytest.approx(0,abs=1e-10)
    assert depth == pytest.approx(INSERT_DEPTH)
    pose[2,3] += .01
    lateral,depth=insertion_metrics(pose)
    assert lateral == pytest.approx(.01)
    assert depth == pytest.approx(INSERT_DEPTH)


def test_wrist_camera_aims_at_insertion_and_survives_sdf_conversion():
    from kinova_marker_pick.sim_description import robot_description,simulation_sdf
    share=Path(__file__).resolve().parents[1]
    urdf=robot_description(share)
    robot=ET.fromstring(urdf)
    mount=robot.find("joint[@name='eih_camera_joint']")
    assert mount.find('parent').get('link') == 'tool_frame'
    optical=robot.find("joint[@name='eih_camera_optical_joint']")
    r1=Rotation.from_euler('xyz',np.fromstring(mount.find('origin').get('rpy'),sep=' '))
    r2=Rotation.from_euler('xyz',np.fromstring(optical.find('origin').get('rpy'),sep=' '))
    from kinova_marker_pick.scene import EIH_CAMERA_OFFSET,EIH_CAMERA_LOOK_AT
    ray=np.array(EIH_CAMERA_LOOK_AT)-EIH_CAMERA_OFFSET
    np.testing.assert_allclose((r1*r2).as_matrix()[:,2],ray/np.linalg.norm(ray),atol=1e-9)
    sdf=ET.fromstring(simulation_sdf(urdf))
    sensor=sdf.find(".//sensor[@name='eih_camera']")
    assert sensor is not None
    assert sensor.findtext('topic') == '/eih/image'
    assert sensor.findtext('gz_frame_id') == 'eih_camera_optical_frame'
    assert sensor.findtext('camera/camera_info_topic') == '/eih/camera_info'


def test_turn_corridor_keeps_entire_rigid_tail_clear_of_hub():
    from kinova_marker_pick.scene import TRANSFER_CLEAR,TURN_CORRIDOR_HALF_WIDTH,GRASP_Z_OFFSET
    # Even a full rotation of the 145 mm proxy bounding sphere cannot reach
    # the panel from any tool position permitted by the turn corridor.
    maximum_y=TRANSFER_CLEAR[1]+TURN_CORRIDOR_HALF_WIDTH+GRASP_Z_OFFSET+.015+.145
    hub_min_y=min(pos[1]-size[1]/2 for name,pos,size in fixture_boxes() if name.startswith('port_'))
    assert maximum_y < hub_min_y-.005
    assert TRANSFER_CLEAR[2]-.145 > .107


def planning_harness(outcomes):
    import threading
    from builtin_interfaces.msg import Time
    values={'planning_group':'arm','planning_pipeline':'ompl','base_frame':'world',
            'ee_link':'tool_frame','velocity_scaling':.15,'pose_tolerance_m':.001,
            'orientation_tolerance_rad':.03}
    calls=[]
    def send(client,goal):
        calls.append(goal)
        outcome=outcomes[min(len(calls)-1,len(outcomes)-1)]
        if isinstance(outcome,Exception):raise outcome
        return outcome
    fake=SimpleNamespace(status=lambda *args:None,p=values.get,cancel=threading.Event(),arm=None,
        get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(to_msg=Time)),
        get_logger=lambda:SimpleNamespace(warning=lambda *args:None),action=send)
    return fake,calls


def test_rejected_collision_path_is_replanned_with_validation_enabled():
    from kinova_marker_pick.marker_pick import ActionFailure
    from moveit_msgs.msg import MoveItErrorCodes
    rejected=ActionFailure(6,SimpleNamespace(error_code=MoveItErrorCodes(val=MoveItErrorCodes.INVALID_MOTION_PLAN)))
    success=SimpleNamespace(error_code=MoveItErrorCodes(val=MoveItErrorCodes.SUCCESS))
    fake,calls=planning_harness([rejected,success])
    MarkerPick.move_to(fake,np.eye(4),'orient_connector',lateral_corridor=(-.20,.05))
    assert len(calls) == 2
    for goal in calls:
        assert goal.request.start_state.is_diff
        assert not goal.planning_options.plan_only
        box=goal.request.path_constraints.position_constraints[0].constraint_region
        assert box.primitives[0].dimensions[1] == pytest.approx(.10)
        assert box.primitive_poses[0].position.y == pytest.approx(-.20)


def test_planning_retries_are_bounded_and_report_moveit_error():
    from kinova_marker_pick.marker_pick import ActionFailure
    from moveit_msgs.msg import MoveItErrorCodes
    error=ActionFailure(6,SimpleNamespace(error_code=MoveItErrorCodes(val=MoveItErrorCodes.INVALID_MOTION_PLAN)))
    fake,calls=planning_harness([error])
    with pytest.raises(RuntimeError,match='orient_connector:.*INVALID_MOTION_PLAN'):
        MarkerPick.move_to(fake,np.eye(4),'orient_connector')
    assert len(calls) == 3


def test_controller_failure_is_not_automatically_reexecuted():
    from kinova_marker_pick.marker_pick import ActionFailure
    from moveit_msgs.msg import MoveItErrorCodes
    error=ActionFailure(6,SimpleNamespace(error_code=MoveItErrorCodes(val=MoveItErrorCodes.CONTROL_FAILED)))
    fake,calls=planning_harness([error])
    with pytest.raises(RuntimeError,match='CONTROL_FAILED'):
        MarkerPick.move_to(fake,np.eye(4),'orient_connector')
    assert len(calls) == 1


def test_observation_pose_constraints_keep_fixed_image_reference_axes_precise():
    from moveit_msgs.msg import MoveItErrorCodes
    success=SimpleNamespace(error_code=MoveItErrorCodes(val=MoveItErrorCodes.SUCCESS))
    for label in ('pose1_above_peg', 'pose2_in_front_of_wall'):
        fake,calls=planning_harness([success])
        MarkerPick.move_to(fake,np.eye(4),label)
        goal=calls[0].request.goal_constraints[0]
        assert goal.position_constraints[0].constraint_region.primitives[0].dimensions[0] <= .0002
        assert goal.orientation_constraints[0].absolute_x_axis_tolerance <= .0005
