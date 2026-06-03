import numpy as np

from collections import deque

# ============================================
# TEMPORAL MEMORY
# ============================================

q_history = deque(maxlen=10)

ep_history = deque(maxlen=10)

prev_q_global = None

prev_ep_global = None

prev_stability_global = 1.0

# ============================================
# TEMPORAL FEATURES
# ============================================

def compute_temporal_features(
    q_max,
    e_p
):

    global prev_q_global
    global prev_ep_global
    global prev_stability_global

    q_history.append(q_max)

    ep_history.append(e_p)

    # ========================================
    # STABILITY
    # ========================================

    if len(q_history) > 1:

        q_std = np.std(q_history)

        ep_std = np.std(ep_history)

        raw_stability = np.exp(
            -(2.0*q_std + ep_std)
        )

        alpha = 0.90

        S_t = (
            alpha * prev_stability_global
            + (1 - alpha) * raw_stability
        )

    else:

        S_t = 1.0

    prev_stability_global = S_t

    # ========================================
    # DELTA Q
    # ========================================

    if prev_q_global is not None:

        delta_q = q_max - prev_q_global

    else:

        delta_q = 0.0

    # ========================================
    # ERROR RATE
    # ========================================

    if prev_ep_global is not None:

        ep_dot = e_p - prev_ep_global

    else:

        ep_dot = 0.0

    prev_q_global = q_max

    prev_ep_global = e_p

    return S_t, delta_q, ep_dot