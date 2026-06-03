# ============================================
# DISTURBANCE MODELING (UPDATED)
# ============================================

import cv2
import numpy as np

def compute_disturbance(prev_img, curr_img):

    prev_gray = cv2.cvtColor(
        prev_img,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    curr_gray = cv2.cvtColor(
        curr_img,
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    diff = curr_gray - prev_gray

    # ========================================
    # AMPLITUDE
    # ========================================

    A_t = np.mean(np.abs(diff))

    # ========================================
    # FREQUENCY
    # ========================================

    fft = np.fft.fft2(diff)

    fft_shift = np.fft.fftshift(fft)

    magnitude = np.abs(fft_shift)

    f_t = np.log1p(np.mean(magnitude))

    return A_t, f_t
