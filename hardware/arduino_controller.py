import serial
import time

from config.config import *

print("Connecting to Arduino...")

ser = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE,
    timeout=1
)

time.sleep(3)

print("✅ Arduino connected")
