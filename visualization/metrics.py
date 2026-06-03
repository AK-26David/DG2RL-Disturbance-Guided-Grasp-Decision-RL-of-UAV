import numpy as np

from config.config import *

# ============================================
# ACTION TRACKER
# ============================================

action_counter = {
    0:0,
    1:0,
    2:0,
    3:0,
    4:0
}

# ============================================
# METRIC STORAGE
# ============================================

reward_history = []

execute_success_history = []

stability_history = []

confidence_history = []

action_history = []

episode_rewards = []

# ============================================
# TRACK ACTIONS
# ============================================

def track_action(action):

    action_counter[action] += 1

    total = sum(action_counter.values())

    print("\n===== ACTION DISTRIBUTION =====")

    for k,v in action_counter.items():

        pct = 100 * v / total

        print(
            f"{ACTION_MAP[k]} : "
            f"{v} ({pct:.1f}%)"
        )

# ============================================
# TRACK REWARDS
# ============================================

def track_rewards(reward):

    reward_history.append(reward)

    if len(reward_history) > 50:

        avg_reward = np.mean(
            reward_history[-50:]
        )

        print(
            f"\n📈 Avg Reward (50): "
            f"{avg_reward:.3f}"
        )

# ============================================
# TRACK METRICS
# ============================================

def track_metrics(state, action, reward):

    q = state[0]

    S = state[6]

    reward_history.append(reward)

    stability_history.append(S)

    confidence_history.append(q)

    action_history.append(action)

    # ========================================
    # EXECUTE SUCCESS
    # ========================================

    if action == 3:

        success = int(

            q > EXECUTE_CONF_THRESH
            and S > MIN_STABILITY
            and state[2] < MAX_DISTURBANCE
            and state[1] < MAX_UNCERTAINTY
        )

        execute_success_history.append(success)

# ============================================
# PRINT FINAL SUMMARY
# ============================================

def print_summary():

    print("\n==============================")

    print("DG²-RL PERFORMANCE SUMMARY")

    print("==============================")

    # ========================================
    # REWARDS
    # ========================================

    if len(reward_history) > 0:

        print(
            f"\nAvg Reward        : "
            f"{np.mean(reward_history):.3f}"
        )

        print(
            f"Max Reward        : "
            f"{np.max(reward_history):.3f}"
        )

        print(
            f"Min Reward        : "
            f"{np.min(reward_history):.3f}"
        )

    # ========================================
    # CONFIDENCE
    # ========================================

    if len(confidence_history) > 0:

        print(
            f"\nAvg Confidence    : "
            f"{np.mean(confidence_history):.3f}"
        )

        print(
            f"Max Confidence    : "
            f"{np.max(confidence_history):.3f}"
        )

    # ========================================
    # STABILITY
    # ========================================

    if len(stability_history) > 0:

        print(
            f"\nAvg Stability     : "
            f"{np.mean(stability_history):.3f}"
        )

        print(
            f"Max Stability     : "
            f"{np.max(stability_history):.3f}"
        )

    # ========================================
    # EXECUTE SUCCESS
    # ========================================

    if len(execute_success_history) > 0:

        execute_acc = 100 * np.mean(
            execute_success_history
        )

        print(
            f"\nExecute Accuracy  : "
            f"{execute_acc:.2f}%"
        )

    else:

        print(
            "\nExecute Accuracy  : "
            "No execute attempts"
        )

    # ========================================
    # ACTION DISTRIBUTION
    # ========================================

    print("\n===== ACTION COUNTS =====")

    unique, counts = np.unique(
        action_history,
        return_counts=True
    )

    for u,c in zip(unique, counts):

        pct = 100 * c / len(action_history)

        print(
            f"{ACTION_MAP[u]} : "
            f"{c} ({pct:.1f}%)"
        )

    print("==============================")