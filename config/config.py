STATE_DIM = 10
ACTION_DIM = 5

ACTION_MAP = {
    0: "WAIT",
    1: "TRACK",
    2: "REPOSITION",
    3: "EXECUTE",
    4: "ABORT"
}

GAMMA = 0.99
LAMBDA = 0.95

LR = 3e-4

CLIP_EPS = 0.2

K_EPOCHS = 10

ENTROPY_COEF = 0.01
VALUE_COEF = 0.5

MAX_GRAD_NORM = 0.5

ROLLOUT_STEPS = 512

BATCH_SIZE = 64

# ============================================
# EXECUTION SAFETY
# ============================================

EXECUTE_CONF_THRESH = 0.15

MAX_UNCERTAINTY = 0.35

MAX_DISTURBANCE = 0.80

MIN_STABILITY = 0.60

SERIAL_PORT = '/dev/ttyUSB0'

BAUD_RATE = 9600
