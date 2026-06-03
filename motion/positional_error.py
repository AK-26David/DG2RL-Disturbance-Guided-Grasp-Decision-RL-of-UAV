# ============================================
# POSITIONAL ERROR (FIXED)
# ============================================
import cv2
import numpy as np
def compute_positional_error(
    curr_pos,
    prev_pos,
    drone_vel
):

    if prev_pos is None:
        return 0.0

    # ========================================
    # NORMALIZE PIXEL COORDINATES
    # ========================================

    curr_norm = curr_pos / 224.0
    prev_norm = prev_pos / 224.0

    motion = curr_norm - prev_norm

    # ========================================
    # NORMALIZE OPTICAL FLOW
    # ========================================

    drone_vel = drone_vel / 10.0

    error_vec = motion - drone_vel

    e_p = np.linalg.norm(error_vec)

    return float(np.clip(e_p, 0, 1))
