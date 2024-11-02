import time
import math
from icm20948 import ICM20948  # ICM20948 Python package for IMU
from dynamixel_sdk import *  # Dynamixel SDK library for controlling the motor

# Dynamixel settings
ADDR_PRO_TORQUE_ENABLE = 64
ADDR_PRO_GOAL_POSITION = 116
ADDR_PRO_PRESENT_POSITION = 132
PROTOCOL_VERSION = 2.0
DXL_ID = 1  # Dynamixel ID
BAUDRATE = 57600
DEVICENAME = '/dev/ttyUSB0'  # Adjust to your port
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

delay = 0.05

# Initialize the ICM20948 IMU
imu = ICM20948()

# Function to get accelerometer readings and calculate roll, pitch
def get_imu_angles():
    accel_data = imu.read_accelerometer_gyro_data()  # Read accelerometer data (g's)
    accel_x, accel_y, accel_z,gx,gy,gz = accel_data
    return gx, gy

def get_accel():
    accel_data = imu.read_accelerometer_gyro_data()  # Read accelerometer data (g's)
    accel_x, accel_y, accel_z,gx,gy,gz = accel_data
    return accel_x,accel_y,accel_z

# Convert degrees to Dynamixel position
def angle_to_position(angle):
    return int((angle / 360.0) * 4095)

def set_position(angle,curr_angle):
    new_angle = curr_angle+angle*(delay)
    if new_angle+90>132:
        return angle_to_position(132),new_angle
    elif new_angle+90<48:
        return angle_to_position(42),new_angle
    else:
        new_pos = angle_to_position(new_angle+90)
        return new_pos, new_angle

# Dynamixel initialization
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
    dxl_comm_result, dxl_error = packet_handler.write4ByteTxRx(port_handler, DXL_ID, ADDR_PRO_GOAL_POSITION, position)
    if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
        raise Exception(f"Failed to set position: {packet_handler.getTxRxResult(dxl_comm_result)}")

def main():
    # Initialize Dynamixel
    port_handler, packet_handler = initialize_dynamixel()
    
    try:
        curr_angle = 0
        while True:
            start=time.clock_gettime(0)
            # Get roll and pitch angles from IMU
            gx, gy = get_imu_angles()
            accel_x,accel_y,accel_z = get_accel()
            abs_g = math.sqrt(accel_x*accel_x+accel_y*accel_y+accel_z*accel_z)
            # Convert the pitch angle to a Dynamixel position
            target_position,curr_angle = set_position(-gx,curr_angle)
            
            if accel_z>0.995 and accel_z<1.005 and abs_g>0.95 and abs_g<1.05:
                curr_angle = 0

            # Set the motor to the target position
            set_dynamixel_position(packet_handler, port_handler, target_position)
            
            # Debugging: print roll, pitch, and target position
            #print(f"Roll: {roll:.2f}, Pitch: {pitch:.2f}, Target Position: {target_position}")
            
            end = time.clock_gettime(0)
            
            interval = end-start

            time.sleep(delay-interval)  # Small delay for IMU reading and motor adjustment
    
    except KeyboardInterrupt: 
        print("Stopping...")

    # Disable torque on the Dynamixel motor
    packet_handler.write1ByteTxRx(port_handler, DXL_ID, ADDR_PRO_TORQUE_ENABLE, TORQUE_DISABLE)
    port_handler.closePort()

if __name__ == "__main__":
    main()
