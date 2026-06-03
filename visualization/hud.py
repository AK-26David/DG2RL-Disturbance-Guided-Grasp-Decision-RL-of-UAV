import cv2
import time
import numpy as np

from config.config import *

# ============================================
# DRAW HUD
# ============================================

def draw_hud(
    curr_frame,
    state,
    action,
    reward,
    out,
    prev_time
):

    # ========================================
    # STATE VARIABLES
    # ========================================

    q = state[0]

    sigma = state[1]

    A = state[2]

    f = state[3]

    e_p = state[4]

    S = state[6]

    v = state[8]

    # ========================================
    # OBJECTNESS
    # ========================================

    objectness = out["objectness"]

    # ========================================
    # FPS + LATENCY
    # ========================================

    current_time = time.time()

    latency = current_time - prev_time

    fps = 1.0 / (latency + 1e-6)

    latency_ms = latency * 1000

    # ========================================
    # EXECUTE SAFETY
    # ========================================

    safe_execute = (

        q > EXECUTE_CONF_THRESH
        and sigma < MAX_UNCERTAINTY
        and A < MAX_DISTURBANCE
        and S > MIN_STABILITY
    )

    # ========================================
    # ACTION COLOR MAP
    # ========================================

    action_color_map = {

        0: (255,255,0),

        1: (0,255,0),

        2: (0,165,255),

        3: (255,0,255),

        4: (0,0,255)
    }

    hud_color = action_color_map[action]

    # ========================================
    # BEST GRASP
    # ========================================

    best_grasp = out["grasps"][0]

    x = int(best_grasp["x"] * (640 / 224))

    y = int(best_grasp["y"] * (480 / 224))

    # ========================================
    # DRAW GRASP POINT
    # ========================================

    cv2.circle(
        curr_frame,
        (x, y),
        10,
        hud_color,
        -1
    )

    cv2.circle(
        curr_frame,
        (x, y),
        18,
        (255,255,255),
        2
    )

    # ========================================
    # ACTION DISPLAY
    # ========================================

    cv2.putText(
        curr_frame,
        f"ACTION: {ACTION_MAP[action]}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        hud_color,
        3
    )

    # ========================================
    # STATUS DISPLAY
    # ========================================

    status_text = (
        "EXECUTE READY"
        if safe_execute
        else "MONITORING"
    )

    status_color = (
        (0,255,0)
        if safe_execute
        else (0,255,255)
    )

    cv2.putText(
        curr_frame,
        status_text,
        (20,75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2
    )

    # ========================================
    # METRICS PANEL
    # ========================================

    panel_x = 20

    panel_y = 120

    line_gap = 28

    metrics = [

        f"Objectness     : {objectness:.3f}",

        f"Confidence     : {q:.3f}",

        f"Uncertainty    : {sigma:.3f}",

        f"Stability      : {S:.3f}",

        f"Disturbance    : {A:.3f}",

        f"Frequency      : {f:.3f}",

        f"Pos Error      : {e_p:.3f}",

        f"Velocity       : {v:.3f}",

        f"Reward         : {reward:.2f}",

        f"FPS            : {fps:.1f}",

        f"Latency        : {latency_ms:.1f} ms"
    ]

    # ========================================
    # PANEL BACKGROUND
    # ========================================

    overlay = curr_frame.copy()

    cv2.rectangle(
        overlay,
        (10, 90),
        (360, 450),
        (0,0,0),
        -1
    )

    alpha = 0.45

    curr_frame[:] = cv2.addWeighted(
        overlay,
        alpha,
        curr_frame,
        1 - alpha,
        0
    )

    # ========================================
    # DRAW METRICS
    # ========================================

    for i, text in enumerate(metrics):

        color = (255,255,255)

        if "Objectness" in text and objectness > 0.20:
            color = (0,255,0)

        if "Objectness" in text and objectness <= 0.20:
            color = (0,0,255)

        if "Confidence" in text and q > 0.30:
            color = (0,255,0)

        if "Uncertainty" in text and sigma > 0.30:
            color = (0,0,255)

        if "Stability" in text and S > 0.75:
            color = (0,255,0)

        if "Pos Error" in text and e_p > 0.50:
            color = (0,0,255)

        cv2.putText(
            curr_frame,
            text,
            (panel_x, panel_y + i*line_gap),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )