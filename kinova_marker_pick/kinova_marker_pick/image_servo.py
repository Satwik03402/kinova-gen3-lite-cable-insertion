"""Calibrated point-feature IBVS for two known fixture planes.

Feature errors are normalized image coordinates. PnP provides only depths for
the point interaction matrix. See Chaumette & Hutchinson, Visual Servo Control,
Part I (2006). Output is a bounded tool increment for collision-checked execution.
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def marker_points(size):
    h = size/2
    return np.array([[-h, h, 0.], [h, h, 0.], [h, -h, 0.], [-h, -h, 0.]])


def project(camera_marker, size, matrix, distortion):
    xyz = marker_points(size)@camera_marker[:3, :3].T+camera_marker[:3, 3]
    if not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= .025):
        raise ValueError('IBVS reference outside the usable camera depth')
    pixels, _ = cv2.projectPoints(marker_points(size),
        Rotation.from_matrix(camera_marker[:3, :3]).as_rotvec(), camera_marker[:3, 3], matrix, distortion)
    return pixels.reshape(4, 2), xyz[:, 2]


def interaction(points, depths):
    points = np.asarray(points, dtype=float).reshape(4, 2)
    depths = np.asarray(depths, dtype=float).reshape(4)
    if not np.isfinite(points).all() or not np.isfinite(depths).all() or np.any(depths <= .025):
        raise ValueError('Invalid IBVS measurement')
    rows = []
    for (x, y), z in zip(points, depths):
        rows.extend(([-1/z, 0, x/z, x*y, -(1+x*x), y],
                     [0, -1/z, y/z, 1+y*y, -x*y, -x]))
    return np.asarray(rows)


def increment(corners, desired, depths, matrix, distortion, world_tool, world_camera, axes,
              gain=.7, damping=.0002, max_translation=.003, max_rotation=.012):
    """Damped IBVS in selected WORLD tool axes; include the camera lever arm."""
    corners, desired = np.asarray(corners).reshape(4, 2), np.asarray(desired).reshape(4, 2)
    current = cv2.undistortPoints(corners.astype(float)[:, None], matrix, distortion).reshape(4, 2)
    target = cv2.undistortPoints(desired.astype(float)[:, None], matrix, distortion).reshape(4, 2)
    error = (current-target).ravel()
    camera_world = world_camera[:3, :3].T
    lever = world_camera[:3, 3]-world_tool[:3, 3]
    basis = np.zeros((6, 6))
    basis[:3, :3] = camera_world
    basis[3:, 3:] = camera_world
    for axis in range(3):
        basis[:3, 3+axis] = camera_world@np.cross(np.eye(3)[axis], lever)
    scales = np.array([.01 if axis < 3 else .08 for axis in axes])
    jacobian = (interaction(current, depths)@basis[:, axes])*scales
    if np.linalg.matrix_rank(jacobian) < len(axes):
        raise ValueError('IBVS task interaction matrix is singular')
    step = -gain*scales*np.linalg.solve(jacobian.T@jacobian+damping**2*np.eye(len(axes)), jacobian.T@error)
    delta = np.zeros(6)
    delta[axes] = step
    factor = min(1., max_translation/max(np.linalg.norm(delta[:3]), 1e-12),
                 max_rotation/max(np.linalg.norm(delta[3:]), 1e-12))
    delta *= factor
    if not np.isfinite(delta).all():
        raise ValueError('IBVS produced a nonfinite command')
    return delta, float(np.sqrt(np.mean((corners-desired)**2)))


def apply_increment(world_tool, delta):
    target = world_tool.copy()
    target[:3, 3] += delta[:3]
    target[:3, :3] = Rotation.from_rotvec(delta[3:]).as_matrix()@target[:3, :3]
    return target
