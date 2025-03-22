import time
import math
from icm20948 import ICM20948  # ICM20948 Python package for IMU
from dynamixel_sdk import *  # Dynamixel SDK library for controlling the motor

# Control table address for XC330
ADDR_TORQUE_ENABLE = 64
ADDR_PRO_TORQUE_ENABLE = 64
ADDR_GOAL_VELOCITY = 104
ADDR_PRESENT_POSITION = 132
ADDR_PRO_GOAL_POSITION = 116
ADDR_GOAL_POSITION = 116
ADDR_OPERATING_MODE = 11
POSITION_CONTROL_MODE = 3

# Protocol version
PROTOCOL_VERSION = 2.0

# Default setting
DXL_ID = 1                 # Dynamixel ID
BAUDRATE = 57600           # Dynamixel default baudrate
DEVICENAME = '/dev/ttyUSB0'  # Check which port is being used on your controller
TORQUE_ENABLE = 1          # Value for enabling the torque
TORQUE_DISABLE = 0         # Value for disabling the torque
VELOCITY_CONTROL_MODE = 1  # Value for setting velocity control mode

DT = 0.01

imu = ICM20948()

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

def initialize_dynamixel():
    # Open port
    if not portHandler.openPort():
        print("Failed to open the port")
        quit()

    # Set port baudrate
    if not portHandler.setBaudRate(BAUDRATE):
        print("Failed to change the baudrate")
        quit()

    # Set the operating mode to velocity control mode
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_OPERATING_MODE, VELOCITY_CONTROL_MODE)

    # Enable Dynamixel torque
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

def set_motor_velocity(velocity):
    # Set motor velocity
    velocity_value = int(velocity)
    packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_VELOCITY, velocity_value)

def set_dynamixel_position(packetHandler, port_handler, position):
    if position>1500:
       position = 1500
    elif position<478:
       position = 478
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(port_handler, DXL_ID, ADDR_PRO_GOAL_POSITION, position)
    if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
        raise Exception(f"Failed to set position: {packet_handler.getTxRxResult(dxl_comm_result)}")

def change_to_position_control_mode():
    # Disable torque before changing mode
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    
    # Set the operating mode to position control mode
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_OPERATING_MODE, POSITION_CONTROL_MODE)
    
    # Enable torque after changing mode
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

def change_to_velocity_control_mode():
    # Disable torque before changing mode
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    
    # Set the operating mode to position control mode
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_OPERATING_MODE, VELOCITY_CONTROL_MODE)
    
    # Enable torque after changing mode
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

def main():
    # Initialize Dynamixel
    try:
        initialize_dynamixel()
        while True:
          accel_x, accel_y, accel_z,gyro_x, gyro_y, gyro_z = imu.read_accelerometer_gyro_data()
          abs_g = math.sqrt(accel_x*accel_x+accel_y*accel_y+accel_z*accel_z)
          if accel_z>0.999 and accel_z<1.001 and abs_g>0.95 and abs_g<1.05:
              change_to_position_control_mode()
              set_dynamixel_position(packetHandler, portHandler,1024)
              change_to_velocity_control_mode()
          set_motor_velocity(-gyro_x)
          time.sleep(DT)
    except KeyboardInterrupt: 
        print("Stopping...")

    # Disable torque on the Dynamixel motor
    packet_handler.write1ByteTxRx(port_handler, DXL_ID, ADDR_PRO_TORQUE_ENABLE, TORQUE_DISABLE)
    port_handler.closePort()

if __name__ == "__main__":
    main()