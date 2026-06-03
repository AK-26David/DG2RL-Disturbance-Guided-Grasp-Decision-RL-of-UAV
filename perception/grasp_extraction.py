import numpy as np

# ============================================
# TOP-K GRASP EXTRACTION
# ============================================

def get_top_k_grasps(q, cos, sin, w, k=10):

    q = q.detach().cpu().numpy()[0,0]

    cos = cos.detach().cpu().numpy()[0,0]

    sin = sin.detach().cpu().numpy()[0,0]

    w = w.detach().cpu().numpy()[0,0]

    q = np.power(q, 1.5)

    # ========================================
    # CENTER PRIOR
    # ========================================

    H, W = q.shape

    center_x = W // 2

    center_y = H // 2

    Y, X = np.meshgrid(
        np.arange(H),
        np.arange(W),
        indexing='ij'
    )

    dist = np.sqrt(
        (X - center_x)**2
        + (Y - center_y)**2
    )

    dist = dist / dist.max()

    # ========================================
    # CENTER-WEIGHTED Q MAP
    # ========================================

    center_weight = 1.0 - dist

    q = q * center_weight

    # ========================================
    # THRESHOLD NOISE
    # ========================================

    q[q < 0.15] = 0

    grasps = []

    q_copy = q.copy()

    for _ in range(k):

        idx = np.argmax(q_copy)

        if q_copy.flat[idx] <= 0:
            break

        y, x = np.unravel_index(
            idx,
            q.shape
        )

        angle = np.arctan2(
            sin[y, x],
            cos[y, x]
        ) / 2

        width = w[y, x] * 224

        grasps.append({

            "x": float(x),

            "y": float(y),

            "angle": float(angle),

            "width": float(width),

            "score": float(q[y, x])
        })

        r = 12

        q_copy[
            max(0,y-r):min(q.shape[0],y+r),
            max(0,x-r):min(q.shape[1],x+r)
        ] = 0

    # ========================================
    # FALLBACK CENTER
    # ========================================

    if len(grasps) == 0:

        grasps.append({

            "x": 112.0,

            "y": 112.0,

            "angle": 0.0,

            "width": 50.0,

            "score": 0.0
        })

    return grasps

def compute_grasp_stats(out):

    grasps = out["grasps"]

    scores = np.array([
        g["score"]
        for g in grasps
    ])

    q_max = scores[0]

    sigma_max = np.std(scores)

    x = grasps[0]["x"]
    y = grasps[0]["y"]

    return (
        q_max,
        sigma_max,
        np.array([x, y])
    )
