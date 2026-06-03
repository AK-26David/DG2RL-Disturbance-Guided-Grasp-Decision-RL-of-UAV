import cv2
import time
import numpy as np

from picamera2 import Picamera2

from perception.module1_inference import module1_forward

from perception.state_builder import build_state_from_output

from rl.ppo_agent import PPOAgent

from rl.environment import DG2RLEnv

from hardware.execution import execute_action

from visualization.hud import draw_hud

from visualization.debug import debug_policy

from visualization.metrics import (
    track_action,
    track_rewards,
    track_metrics,
    print_summary
)

from config.config import *

# ============================================
# INITIALIZE RL SYSTEM
# ============================================

env = DG2RLEnv()

agent = PPOAgent()

print("✅ DG²-RL initialized")

# ============================================
# INITIALIZE PI CAMERA
# ============================================

print("📷 Initializing Pi Camera...")

picam2 = Picamera2()

picam2.configure(

    picam2.create_preview_configuration(

        main={"size": (640,480)}

    )
)

picam2.start()

time.sleep(2)

prev_frame = picam2.capture_array()

print("✅ Camera initialized")

# ============================================
# START DG²-RL LOOP
# ============================================

print("✅ Starting DG²-RL real-time inference")

step_counter = 0

prev_time = time.time()

# ============================================
# MAIN LOOP
# ============================================

while True:

    # ========================================
    # CAMERA FRAME
    # ========================================

    curr_frame = picam2.capture_array()

    # ========================================
    # RGB CONVERSION
    # ========================================

    curr_frame = cv2.cvtColor(
        curr_frame,
        cv2.COLOR_RGB2BGR
    )

    # ========================================
    # FAKE DEPTH
    # ========================================

    gray = cv2.cvtColor(
        curr_frame,
        cv2.COLOR_BGR2GRAY
    )

    depth = gray.astype(np.float32)

    # ========================================
    # MODULE 1
    # ========================================

    out = module1_forward(
        curr_frame,
        depth
    )

    # ========================================
    # MODULE 2
    # ========================================

    state = build_state_from_output(
        prev_frame,
        curr_frame,
        out
    )

    # ========================================
    # OBJECTNESS
    # ========================================

    objectness = out["objectness"]

    NO_OBJECT_THRESH = 0.25

    # ========================================
    # OBJECT-AWARE POLICY GATING
    # ========================================

    if objectness < NO_OBJECT_THRESH:

        action = 0

        print("\n🚫 NO OBJECT DETECTED")

    else:

        # ====================================
        # PPO ACTION
        # ====================================

        action = agent.select_action(state)

        # ====================================
        # DEBUG VALUES
        # ====================================

        print("\n================")

        print(f"Objectness : {objectness:.3f}")

        print(f"Q          : {state[0]:.3f}")

        print(f"Sigma      : {state[1]:.3f}")

        print(f"Disturbance: {state[2]:.3f}")

        print(f"Stability  : {state[6]:.3f}")

        print(f"ACTION     : {ACTION_MAP[action]}")

        print("================")

        # ====================================
        # HARD EXECUTE SAFETY GATE
        # ====================================

        q = state[0]

        sigma = state[1]

        A = state[2]

        S = state[6]

        # ====================================
        # BLOCK UNSAFE EXECUTE
        # ====================================

        if action == 3:

            safe_execute = (

                q > EXECUTE_CONF_THRESH
                and sigma < MAX_UNCERTAINTY
                and A < MAX_DISTURBANCE
                and S > MIN_STABILITY
                and objectness > NO_OBJECT_THRESH
            )

            if not safe_execute:

                print("\n🛑 EXECUTE BLOCKED")

                action = 0

    # ========================================
    # TRACK ACTIONS
    # ========================================

    track_action(action)

    # ========================================
    # EXECUTE ACTION
    # ========================================

    execute_action(action, state)

    # ========================================
    # ENVIRONMENT STEP
    # ========================================

    _, reward, done, _ = env.step(
        state,
        action
    )

    # ========================================
    # TRACK REWARDS
    # ========================================

    track_rewards(reward)

    # ========================================
    # STORE PPO BUFFER
    # ========================================

    agent.buffer.rewards.append(reward)

    agent.buffer.dones.append(done)

    # ========================================
    # DEBUG
    # ========================================

    debug_policy(state, action, reward)

    track_metrics(state, action, reward)

    # ========================================
    # PPO UPDATE
    # ========================================

    if len(agent.buffer.states) >= ROLLOUT_STEPS:

        loss = agent.update()

        print(
            f"\n🔥 PPO Updated | "
            f"Loss: {loss:.4f}"
        )

    # ========================================
    # HUD
    # ========================================

    draw_hud(
        curr_frame,
        state,
        action,
        reward,
        out,
        prev_time
    )

    # ========================================
    # DISPLAY
    # ========================================

    cv2.imshow(
        "DG2-RL Research HUD",
        curr_frame
    )

    # ========================================
    # UPDATE PREVIOUS FRAME
    # ========================================

    prev_frame = curr_frame.copy()

    step_counter += 1

    # ========================================
    # EXIT KEY
    # ========================================

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# ============================================
# CLEANUP
# ============================================

print("\n🛑 Shutting down DG²-RL")

cv2.destroyAllWindows()

print_summary()
