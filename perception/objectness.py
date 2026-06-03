# ============================================
# OBJECTNESS
# ============================================

import cv2
import numpy as np

def compute_objectness(rgb):

    img = cv2.resize(rgb, (224, 224))

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    edges = cv2.Canny(gray, 80, 160)

    edge_score = edges.mean() / 255.0

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contour_areas = [
        cv2.contourArea(c)
        for c in contours
    ]

    large_contours = sum(
        a > 300
        for a in contour_areas
    )

    contour_score = min(
        large_contours / 5.0,
        1.0
    )

    texture_score = np.std(gray) / 64.0

    texture_score = np.clip(
        texture_score,
        0,
        1
    )

    obj_score = (
        0.4 * edge_score
        + 0.3 * contour_score
        + 0.3 * texture_score
    )

    return float(
        np.clip(obj_score, 0, 1)
    )
