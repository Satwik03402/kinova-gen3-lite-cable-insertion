"""Regenerate SDF assets using the local OpenCV version."""
from pathlib import Path
import cv2
from kinova_marker_pick.scene import generate_assets
from kinova_marker_pick.vision import marker_image

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    generate_assets(root)
    for marker_id in (7, 8):
        cv2.imwrite(str(root / f'marker_id{marker_id}.png'), marker_image(marker_id))
