import time
import math
from icm20948 import ICM20948  # ICM20948 Python package for IMU
from dynamixel_sdk import *  # Dynamixel SDK library for controlling the motor

# Dynamixel settings
ADDR_PRO_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRO_PRESENT_POSITION = 132
PROTOCOL_VERSION = 2.0
DXL_ID = 25  # Dynamixel ID
BAUDRATE = 57600
DEVICENAME = '/dev/ttyUSB0'  # Adjust to your port
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

DT = 0.01

imu = ICM20948()

ALPHA = 0.98

#Calculate angle based on imu accelerometer and gyroscope data
def calculate_angle(accel_x, accel_y, accel_z,gyro_x, gyro_y, gyro_z,roll):
  # Calculate the angle from the accelerometer
  accel_roll = math.atan2(accel_y, accel_z) * 180 / math.pi
  # Integrate the gyroscope data
  roll += gyro_x * DT
  # Apply the complementary filter
  roll = ALPHA * roll + (1 - ALPHA) * accel_roll
  return roll

# Convert degrees to Dynamixel position
def angle_to_position(angle):
  return int(((-angle+90) / 360.0) * 4095)

def initialize_dynamixel():
    port_handler = PortHandler(DEVICENAME)
    packet_handler = PacketHandler(PROTOCOL_VERSION)
    
    # Open port
    if not port_handler.openPort():
        raise Exception("Failed to open port")
    
    # Set baudrate
    if not port_handler.setBaudRate(BAUDRATE):
        raise Exception("Failed to set baudrate")
    
    # Enable torque
    dxl_comm_result, dxl_error = packet_handler.write1ByteTxRx(port_handler, DXL_ID, ADDR_PRO_TORQUE_ENABLE, TORQUE_ENABLE)
    if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
        raise Exception("Failed to enable torque")
    
    return port_handler, packet_handler

def set_dynamixel_position(packet_handler, port_handler, position):
    if position>1500:
       position = 1500
    elif position<478:
       position = 478
    dxl_comm_result, dxl_error = packet_handler.write4ByteTxRx(port_handler, DXL_ID, ADDR_GOAL_POSITION, position)
    if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
        raise Exception(f"Failed to set position: {packet_handler.getTxRxResult(dxl_comm_result)}")

def main():
    # Initialize Dynamixel
    port_handler, packet_handler = initialize_dynamixel()
    try:
        set_dynamixel_position(packet_handler, port_handler, 1000)
        """curr_angle = 0
        while True:
          start=time.clock_gettime(0)
          accel_x, accel_y, accel_z,gyro_x, gyro_y, gyro_z = imu.read_accelerometer_gyro_data()
          
          curr_angle = calculate_angle(accel_x, accel_y, accel_z,gyro_x, gyro_y, gyro_z,curr_angle)

          abs_g = math.sqrt(accel_x*accel_x+accel_y*accel_y+accel_z*accel_z)
          if accel_z>0.995 and accel_z<1.005 and abs_g>0.95 and abs_g<1.05:
              curr_angle = 0

          # Set the motor to the target position
          set_dynamixel_position(packet_handler, port_handler, angle_to_position(curr_angle))
          print(curr_angle)

          end = time.clock_gettime(0)
          interval = end-start
          if (DT-interval)>0:
            time.sleep(DT-interval)  # Small delay for IMU reading and motor adjustment"""
    
    except KeyboardInterrupt: 
        print("Stopping...")

    # Disable torque on the Dynamixel motor
    port_handler.closePort()

if __name__ == "__main__":
    main()