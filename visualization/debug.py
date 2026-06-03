from config.config import *

# ============================================
# RL TELEMETRY DEBUG
# ============================================

def debug_policy(state, action, reward):

    q = state[0]

    sigma = state[1]

    A = state[2]

    f = state[3]

    e_p = state[4]

    ep_dot = state[5]

    S = state[6]

    delta_q = state[7]

    v = state[8]

    omega = state[9]

    print("\n==============================")

    print("DG²-RL POLICY DEBUG")

    print("==============================")

    print(f"Action      : {ACTION_MAP[action]}")

    print(f"Reward      : {reward:.3f}")

    print("\n--- GRASP ---")

    print(f"Confidence  : {q:.3f}")

    print(f"Uncertainty : {sigma:.3f}")

    print("\n--- DISTURBANCE ---")

    print(f"Amplitude   : {A:.3f}")

    print(f"Frequency   : {f:.3f}")

    print("\n--- POSITION ---")

    print(f"Pos Error   : {e_p:.3f}")

    print(f"Error Rate  : {ep_dot:.3f}")

    print("\n--- TEMPORAL ---")

    print(f"Stability   : {S:.3f}")

    print(f"Delta Q     : {delta_q:.3f}")

    print("\n--- MOTION ---")

    print(f"Velocity    : {v:.3f}")

    print(f"Omega       : {omega:.3f}")

    print("==============================")