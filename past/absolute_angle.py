import time
import math
from icm20948 import ICM20948

# Constants for the complementary filter
ALPHA = 0.98
DT = 0.01  # Loop time (s)

def calculate_angle():
    # Initialize the IMU
    imu = ICM20948()

    # Initialize angles
    roll = 0.0
    pitch = 0.0

    while True:
        # Read accelerometer and gyroscope data
        accel_x, accel_y, accel_z,gyro_x, gyro_y, gyro_z = imu.read_accelerometer_gyro_data()
        # Calculate the angle from the accelerometer
        accel_roll = math.atan2(accel_y, accel_z) * 180 / math.pi
        accel_pitch = math.atan2(-accel_x, math.sqrt(accel_y**2 + accel_z**2)) * 180 / math.pi

        # Integrate the gyroscope data
        roll += gyro_x * DT
        pitch += gyro_y * DT

        # Apply the complementary filter
        roll = ALPHA * roll + (1 - ALPHA) * accel_roll
        pitch = ALPHA * pitch + (1 - ALPHA) * accel_pitch

        # Print the calculated angles
        print(f"Roll: {roll:.2f}, Pitch: {pitch:.2f}")

        time.sleep(DT)

if __name__ == "__main__":
    try:
        calculate_angle()
    except KeyboardInterrupt:
        print("\nProgram terminated.")
