"""Live wrist-camera alignment guides; drawing never changes control images."""
import cv2
import numpy as np

CYAN = (255, 230, 0)
YELLOW = (0, 225, 255)
GREEN = (80, 235, 80)
RED = (80, 90, 255)


def alignment_state(corners, desired, tolerance, converged=False):
    if corners is None:
        return ('MARKER LOST' if desired is not None else 'WAITING FOR MARKER'), None, RED
    if desired is None:
        return 'MARKER DETECTED', None, CYAN
    error = float(np.sqrt(np.mean((np.asarray(corners)-desired)**2)))
    if error <= tolerance:
        return ('ALIGNED' if converged else 'IN TOLERANCE'), error, GREEN
    return 'ALIGNING', error, YELLOW


def render_alignment(image, corners, desired, marker_id, stage, tolerance, converged=False):
    """Cyan marker, a fixed target square and arrows toward its four corners.

    The target is a projected physical square: its perspective follows the
    calibrated desired camera view rather than an arbitrary screen rectangle.
    """
    result = image.copy()
    state, error, color = alignment_state(corners, desired, tolerance, converged)
    scale = max(.6, image.shape[1]/1280)
    thick = max(1, round(3*scale))
    point = lambda xy:tuple(np.rint(xy).astype(int))
    if desired is not None:
        target = np.asarray(desired).reshape(4, 2)
        polygon = np.rint(target).astype(np.int32).reshape(-1, 1, 2)
        target_color = GREEN if error is not None and error <= tolerance else YELLOW
        cv2.polylines(result, [polygon], True, target_color, max(1, thick-1), cv2.LINE_AA)
        # Bold L-shaped corners make the convergence square easy to see even
        # when the cyan marker outline overlaps the target at the end.
        for index, xy in enumerate(target):
            before = xy + .25*(target[(index-1) % 4]-xy)
            after = xy + .25*(target[(index+1) % 4]-xy)
            bracket = np.rint([before, xy, after]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(result, [bracket], False, target_color, thick+round(3*scale), cv2.LINE_AA)
        centre = point(np.mean(desired, axis=0))
        cv2.drawMarker(result, centre, target_color, cv2.MARKER_CROSS, round(30*scale), thick)
    if corners is not None:
        polygon = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(result, [polygon], True, CYAN, thick, cv2.LINE_AA)
        centre = point(np.mean(corners, axis=0))
        cv2.circle(result, centre, round(9*scale), CYAN, thick, cv2.LINE_AA)
        if desired is not None:
            for actual, target in zip(corners, desired):
                if np.linalg.norm(actual-target) > 2:
                    cv2.arrowedLine(result, point(actual), point(target), (255, 100, 240),
                                    max(1, thick-1), cv2.LINE_AA, tipLength=.3)
    # Keep guides at their real image coordinates; the HUD sits above them.
    hud_height = round(142*scale)
    tint = result[:hud_height].copy()
    tint[:] = (24, 28, 34)
    cv2.addWeighted(tint, .90, result[:hud_height], .10, 0, result[:hud_height])
    phase = 'PICKUP' if marker_id == 7 else 'WALL / PORT'
    lines = [f'{phase}  |  Marker {marker_id}  |  {state}',
             (f'Corner error: {error:.2f} px   Target: <= {tolerance:.2f} px'
              if error is not None else stage.replace('_', ' ').upper()),
             'Match CYAN marker to TARGET SQUARE (yellow / green)']
    for index, line in enumerate(lines):
        cv2.putText(result, line, (round(18*scale), round((36+43*index)*scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, (1.12 if index < 2 else .90)*scale,
                    color if index == 0 else (235, 235, 235), max(1, round(2*scale)), cv2.LINE_AA)
    return result
