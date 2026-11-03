"""IBVS frame geometry, correction bounds and two-stage convergence."""
import threading
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation
from kinova_marker_pick.image_servo import project, interaction, increment, apply_increment
from kinova_marker_pick.scene import (PICK, PORT_MARKER_OFFSET, MARKER_Z, MARKER_SIZE,
    port_transform, object_at_port, EIH_CAMERA_OFFSET, eih_camera_rotation)
from kinova_marker_pick.marker_pick import MarkerPick

K = np.array([[935., 0., 640.], [0., 935., 480.], [0., 0., 1.]])
D = np.zeros(5)
OBSERVATION = yaml.safe_load((Path(__file__).resolve().parents[1]/'config/observation_poses.yaml').read_text())['marker_pick']['ros__parameters']


@pytest.mark.parametrize('stage', [1, 2])
@pytest.mark.parametrize('sign', [-1, 1])
def test_two_observation_stages_correct_image_translation_and_rotation(stage, sign):
    tool = np.eye(4)
    tool[:3, 3] = [.30, -.12, .23]
    tool[:3, :3] = Rotation.from_euler('xyz', [np.pi, 0., np.pi/2]).as_matrix()
    marker = np.eye(4)
    marker[:3, 3] = np.array(PICK)+[0., 0., MARKER_Z]
    axes = [0, 1, 5]
    if stage == 2:
        held = np.eye(4)
        held[:3, :3] = tool[:3, :3].T
        held[:3, 3] = [0., 0., .02]
        tool = object_at_port(port_transform(), -.065)@np.linalg.inv(held)
        marker = port_transform()
        marker[:3, 3] += marker[:3, :3]@PORT_MARKER_OFFSET
        axes = [1, 2, 3]
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = eih_camera_rotation()
    extrinsic[:3, 3] = EIH_CAMERA_OFFSET
    desired, _ = project(np.linalg.inv(tool@extrinsic)@marker, MARKER_SIZE, K, D)
    disturbance = np.zeros(6)
    offset, angle = OBSERVATION['observation_offset_m'], OBSERVATION['observation_angle_rad']
    disturbance[axes] = np.array([offset, -offset, angle])*sign
    current = apply_increment(tool, disturbance)
    errors = []
    for _ in range(25):
        camera = current@extrinsic
        corners, depths = project(np.linalg.inv(camera)@marker, MARKER_SIZE, K, D)
        delta, error = increment(corners, desired, depths, K, D, current, camera, axes)
        assert np.linalg.norm(delta[:3]) <= .003000001
        assert np.linalg.norm(delta[3:]) <= .012000001
        errors.append(error)
        current = apply_increment(current, delta)
    assert errors[0] > 10
    assert errors[-1] < .1
    np.testing.assert_allclose(current[:3, 3], tool[:3, 3], atol=1e-5)
    assert Rotation.from_matrix(tool[:3, :3].T@current[:3, :3]).magnitude() < .0001


def test_invalid_depth_is_rejected_before_command_generation():
    with pytest.raises(ValueError, match='measurement'):
        interaction(np.zeros((4, 2)), [.2, .2, 0., .2])


def test_predefined_poses_keep_camera_facing_peg_and_wall():
    import yaml
    from pathlib import Path
    p = yaml.safe_load((Path(__file__).resolve().parents[1]/'config/observation_poses.yaml').read_text())['marker_pick']['ros__parameters']
    pickup = Rotation.from_euler('xyz', p['pickup_observation_rpy']).as_matrix()
    optical = pickup@eih_camera_rotation()
    assert optical[:, 2]@np.array([0., 0., -1.]) > .9
    wall_object = Rotation.from_euler('xyz', p['port_observation_rpy']).as_matrix()
    wall_tool = wall_object@pickup
    optical = wall_tool@eih_camera_rotation()
    assert optical[:, 2]@np.array([1., 0., 0.]) > .9


@pytest.mark.parametrize('cancelled', [False, True])
def test_missing_camera_or_cancellation_never_executes_a_correction(monkeypatch, cancelled):
    from kinova_marker_pick import marker_pick as module
    elapsed = [0.]
    calls = []
    values = {'ibvs_timeout_s':10., 'ibvs_max_iterations':30}
    def wait(delay):
        elapsed[0] += delay
    fake = SimpleNamespace(status=lambda *a:None, tool_world=lambda:np.eye(4), p=values.get,
        get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=0)),
        cancel=SimpleNamespace(is_set=lambda:cancelled, wait=wait), lock=threading.Lock(), features={},
        straight_to=lambda *a:calls.append(a))
    monkeypatch.setattr(module.time, 'monotonic', lambda:elapsed[0])
    with pytest.raises(RuntimeError, match='Cancelled' if cancelled else 'lost fresh wrist marker'):
        MarkerPick.image_align(fake, 7, np.eye(4), [0, 1, 5], 'pose1_ibvs')
    assert calls == []
