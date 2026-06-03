# ============================================
# DG²-RL ENVIRONMENT
# ============================================
import numpy as np
from config.config import *
class DG2RLEnv:

    def __init__(self):

        self.reset()

    # ========================================
    # RESET
    # ========================================

    def reset(self):

        self.timestep = 0

        self.prev_state = np.zeros(
            STATE_DIM,
            dtype=np.float32
        )

        return self.prev_state

    # ========================================
    # REWARD FUNCTION
    # ========================================

    def compute_reward(self, state, action):

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

        reward = 0

        # ====================================
        # WAIT
        # ====================================

        if action == 0:

            if (
                A > 0.45
                or sigma > 0.35
                or S < 0.45
            ):
                reward += 3

            else:
                reward -= 1

        # ====================================
        # TRACK
        # ====================================

        elif action == 1:

            if e_p < 0.35:
                reward += 4
            else:
                reward -= 3

        # ====================================
        # REPOSITION
        # ====================================

        elif action == 2:

            if (
                e_p > 0.4
                and S > 0.5
            ):
                reward += 5
            else:
                reward -= 3

        # ====================================
        # EXECUTE
        # ====================================

        elif action == 3:

            stable_execute = (

                q > EXECUTE_CONF_THRESH
                and sigma < MAX_UNCERTAINTY
                and A < MAX_DISTURBANCE
                and S > MIN_STABILITY
            )

            if stable_execute:
                reward += 15
            else:
                reward -= 12

        # ====================================
        # ABORT
        # ====================================

        elif action == 4:

            catastrophic = (
                sigma > 0.75
                or A > 0.85
                or S < 0.15
            )

            if catastrophic:
                reward += 8
            else:
                reward -= 3

        # ====================================
        # TEMPORAL BONUSES
        # ====================================

        reward += 2 * delta_q

        reward -= 1.5 * ep_dot

        return reward

    # ========================================
    # STEP
    # ========================================

    def step(self, state, action):

        reward = self.compute_reward(
            state,
            action
        )

        self.timestep += 1

        done = False

        if self.timestep > 200:
            done = True

        self.prev_state = state

        return (
            state,
            reward,
            done,
            {}
        )
