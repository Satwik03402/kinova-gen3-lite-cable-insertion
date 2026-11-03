"""OpenCV-version-independent marker detection and calibrated pose estimation."""
import cv2
import numpy as np


def dictionary():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def marker_image(marker_id, pixels=600):
    aruco = cv2.aruco
    if hasattr(aruco, 'generateImageMarker'):
        return aruco.generateImageMarker(dictionary(), marker_id, pixels)
    return aruco.drawMarker(dictionary(), marker_id, pixels)


class Detector:
    def __init__(self):
        aruco = cv2.aruco
        params = (aruco.DetectorParameters() if hasattr(aruco, 'ArucoDetector')
                  else aruco.DetectorParameters_create())
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.params = params
        self.detector = (aruco.ArucoDetector(dictionary(), params)
                         if hasattr(aruco, 'ArucoDetector') else None)

    def detect(self, image):
        if self.detector is not None:
            return self.detector.detectMarkers(image)
        return cv2.aruco.detectMarkers(image, dictionary(), parameters=self.params)


def estimate_pose(corners, size, camera_matrix, distortion, max_error=2.0):
    """Return optical-camera <- marker; reject bad/behind-camera solutions."""
    half = size / 2
    obj = np.array([[-half, half, 0], [half, half, 0],
                    [half, -half, 0], [-half, -half, 0]], dtype=np.float64)
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    # ITERATIVE also handles exactly frontal observations on OpenCV 4.6.
    ok, rvec, tvec = cv2.solvePnP(obj, pts, camera_matrix, distortion,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or not np.isfinite(tvec).all() or tvec[2, 0] <= 0:
        return None
    projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, distortion)
    if np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - pts)**2, axis=1))) > max_error:
        return None
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rvec)[0]
    transform[:3, 3] = tvec.ravel()
    return transform
