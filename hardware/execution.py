from config.config import *

from hardware.arduino_controller import ser

import time

# ============================================
# EXECUTION MEMORY
# ============================================

prev_action = None

last_execute_time = 0

EXECUTE_COOLDOWN = 5.0

# ============================================
# EXECUTION CONTROL
# ============================================

def execute_action(action, state):

    global prev_action
    global last_execute_time

    action_name = ACTION_MAP[action]

    print(f"\n🎯 ACTION: {action_name}")

    current_time = time.time()

    try:

        # ====================================
        # EXECUTE
        # ====================================

        if action == 3:

            # COOLDOWN CHECK

            if (
                current_time - last_execute_time
                > EXECUTE_COOLDOWN
            ):

                print("🚀 MOTOR START")

                ser.write(b'1')

                ser.flush()

                last_execute_time = current_time

        # ====================================
        # STOP
        # ====================================

        else:

            if prev_action == 3:

                print("🛑 MOTOR STOP")

                ser.write(b'0')

                ser.flush()

    except Exception as e:

        print(f"❌ Serial Error: {e}")

    prev_action = action
