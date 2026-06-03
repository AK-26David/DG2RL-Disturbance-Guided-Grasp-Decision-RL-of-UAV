# ============================================
# DRONE VELOCITY
# ============================================
import cv2
import numpy as np

def compute_drone_velocity(prev_img, curr_img):

    prev_gray = cv2.cvtColor(
        prev_img,
        cv2.COLOR_BGR2GRAY
    )

    curr_gray = cv2.cvtColor(
        curr_img,
        cv2.COLOR_BGR2GRAY
    )

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    dx = np.mean(flow[..., 0])
    dy = np.mean(flow[..., 1])

    return np.array([dx, dy])
