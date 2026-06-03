import numpy as np
import cv2

from perception.grasp_extraction import compute_grasp_stats

from motion.disturbance import compute_disturbance

from motion.optical_flow import compute_drone_velocity

from motion.positional_error import compute_positional_error

from motion.temporal_features import compute_temporal_features

# ============================================
# GLOBAL POSITION MEMORY
# ============================================

prev_pos_global = None

# ============================================
# FINAL STATE BUILDER
# ============================================

def build_state_from_output(
    prev_img,
    curr_img,
    out
):

    global prev_pos_global

    # =====================================
    # GRASP FEATURES
    # =====================================

    q_max, sigma_max, curr_pos = \
        compute_grasp_stats(out)

    obj_score = out["objectness"]

    # =====================================
    # CONFIDENCE FUSION
    # =====================================

    q_max = (
        0.7 * q_max
        + 0.3 * obj_score
    )

    sigma_max = (
        0.7 * sigma_max
        + 0.3 * (1.0 - obj_score)
    )

    # =====================================
    # DISTURBANCE
    # =====================================

    A_t, f_t = compute_disturbance(
        prev_img,
        curr_img
    )

    # =====================================
    # MOTION
    # =====================================

    drone_vel = compute_drone_velocity(
        prev_img,
        curr_img
    )

    # =====================================
    # POSITIONAL ERROR
    # =====================================

    e_p = compute_positional_error(
        curr_pos,
        prev_pos_global,
        drone_vel
    )

    # =====================================
    # TEMPORAL FEATURES
    # =====================================

    S_t, delta_q, ep_dot = \
        compute_temporal_features(
            q_max,
            e_p
        )

    # =====================================
    # MOTION STATS
    # =====================================

    v_d = np.linalg.norm(drone_vel)

    theta = np.arctan2(
        drone_vel[1],
        drone_vel[0]
    )

    omega_sin = np.sin(theta)

    omega_cos = np.cos(theta)

    # =====================================
    # NORMALIZATION
    # =====================================

    q_max = np.clip(q_max, 0, 1)

    sigma_max = np.clip(
        sigma_max,
        0,
        1
    )

    A_t = np.tanh(A_t / 80.0)

    f_t = np.clip(f_t / 10.0, 0, 1)

    ep_dot = np.tanh(ep_dot / 0.1)

    delta_q = np.tanh(delta_q / 0.05)

    v_d = np.tanh(v_d / 2.0)

    # =====================================
    # FINAL STATE VECTOR
    # =====================================

    state = np.array([

        q_max,
        sigma_max,

        A_t,
        f_t,

        e_p,
        ep_dot,

        S_t,
        delta_q,

        v_d,
        omega_sin

    ], dtype=np.float32)

    prev_pos_global = curr_pos

    return state
