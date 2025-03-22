import time
import numpy as np
from dynamixel_sdk import *  # Uses Dynamixel SDK library

# Control table address for XC330
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_VELOCITY = 104
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_OPERATING_MODE = 11

# Protocol version
PROTOCOL_VERSION = 2.0

# Default setting
DXL_ID = 25                 # Dynamixel ID
BAUDRATE = 57600           # Dynamixel default baudrate
DEVICENAME = '/dev/ttyUSB0'  # Check which port is being used on your controller
TORQUE_ENABLE = 1          # Value for enabling the torque
TORQUE_DISABLE = 0         # Value for disabling the torque
VELOCITY_CONTROL_MODE = 1  # Value for setting velocity control mode
POSITION_CONTROL_MODE = 3  # Value for setting position control mode

# Initialize the port handler and packet handler
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

    # Set the operating mode to position control mode
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_OPERATING_MODE, POSITION_CONTROL_MODE)

    # Enable Dynamixel torque
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

def set_motor_position(position):
    # Set motor position
    position_value = int(position)
    packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_POSITION, position_value)

def control_motor_with_position():
    try:
        while True:
            # Example: Set position to 1024
            print("Setting position to 1024")
            set_motor_position(2000)
            time.sleep(3)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    try:
        initialize_dynamixel()
        control_motor_with_position()
    except KeyboardInterrupt:
        # Disable Dynamixel torque before quitting
        packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        print("\nProgram terminated.")
    finally:
        # Close port
        portHandler.closePort()
