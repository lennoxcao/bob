import numpy as np
from icm20948 import ICM20948  # ICM20948 Python package for IMU
from dynamixel_sdk import *  # Dynamixel SDK library for controlling the motor
import math

# {left_hip_1:11,left_hip_2:12,left_hip_3:13,left_knee:14,left_ankle:15,right_hip_1:21,right_hip_2:22,right_hip_3:23,right_knee:24}
# neutral motor angle: {hip1:180,hip2:180}


class bob:
    # Dynamixel settings
    ADDR_PRO_TORQUE_ENABLE = 64
    ADDR_PRO_GOAL_POSITION = 116
    ADDR_PRO_PRESENT_POSITION = 132
    PROTOCOL_VERSION = 2.0
    BAUDRATE = 57600
    DEVICENAME = "/dev/ttyUSB0"  # Adjust to your port
    TORQUE_ENABLE = 1
    TORQUE_DISABLE = 0
    dynaindex = [21, 22, 23, 24, 25, 11, 12, 13, 14, 15]

    def __init__(self, dynaindex):
        self.dynaindex = dynaindex
        # update speed
        self.dt = 0.01
        # list of all the motorized joints
        self.joints_right = [
            dynaindex[0],
            dynaindex[1],
            dynaindex[2],
            dynaindex[3],
            dynaindex[4],
        ]
        self.joints_left = {
            dynaindex[5],
            dynaindex[6],
            dynaindex[7],
            dynaindex[8],
            dynaindex[9],
        }
        # Initialized motor positions
        self.joint_angles_right = [0, 0, 0, 0, 0]
        self.joint_angles_left = [0, 0, 0, 0, 0]
        # kinematics parameter (a, alpha, d, theta)
        joints_right = [
            [55.3, np.pi, 0, np.pi / 2],
            [35.175, np.pi / 2, -47.45, -np.pi / 2],
            [0, np.pi / 2, -55.55, 0],
            [-108.5, 0, 0, np.pi],
            [-99.5, 0, 0, 0],
        ]
        joints_left = joints_right * np.array([1, -1, -1, 1])
        # DH Parameters for 5 joints in radians (theta, d, a, alpha)
        # Replace these values with actual DH parameters for your robot arm.
        dh_params = [
            (np.pi / 2, 0.0, 55.3, np.pi),  # Joint 1
            (np.pi / 2, -35.175, -35.175, np.pi / 2),  # Joint 2
            (1.0472, 0.2, 0.3, -1.5708),  # Joint 3
            (1.5708, 0.3, 0.2, 1.5708),  # Joint 4
            (2.0944, 0.4, 0.1, 0.0),  # Joint 5
        ]
        # define dh_transformation matrices
        self.dh_transform_right = []
        self.dh_transform_left = []
        for i in range(5):
            a, alpha, d, theta = joints_right[i]
            self.dh_transform_right.append(
                np.array(
                    [
                        [
                            np.cos(theta),
                            -np.sin(theta) * np.cos(alpha),
                            np.sin(theta) * np.sin(alpha),
                            a * np.cos(theta),
                        ],
                        [
                            np.sin(theta),
                            np.cos(theta) * np.cos(alpha),
                            -np.cos(theta) * np.sin(alpha),
                            a * np.sin(theta),
                        ],
                        [0, np.sin(alpha), np.cos(alpha), d],
                        [0, 0, 0, 1],
                    ]
                )
            )
            a, alpha, d, theta = joints_left[i]
            self.dh_transform_left.append(
                np.array(
                    [
                        [
                            np.cos(theta),
                            -np.sin(theta) * np.cos(alpha),
                            np.sin(theta) * np.sin(alpha),
                            a * np.cos(theta),
                        ],
                        [
                            np.sin(theta),
                            np.cos(theta) * np.cos(alpha),
                            -np.cos(theta) * np.sin(alpha),
                            a * np.sin(theta),
                        ],
                        [0, np.sin(alpha), np.cos(alpha), d],
                        [0, 0, 0, 1],
                    ]
                )
            )

        def initialize_dynamixel(id):
            port_handler = PortHandler(self.DEVICENAME)
            packet_handler = PacketHandler(self.PROTOCOL_VERSION)

            # Open port
            if not port_handler.openPort():
                raise Exception("Failed to open port")

            # Set baudrate
            if not port_handler.setBaudRate(self.BAUDRATE):
                raise Exception("Failed to set baudrate")

            # Enable torque
            dxl_comm_result, dxl_error = packet_handler.write1ByteTxRx(
                port_handler, id, self.ADDR_PRO_TORQUE_ENABLE, self.TORQUE_ENABLE
            )
            if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                raise Exception("Failed to enable torque")
            return port_handler, packet_handler

        self.port_handler = []
        self.packet_handler = []
        for i in dynaindex:
            port, packet = initialize_dynamixel(i)
            self.port_handler.append(port)
            self.packet_handler.append(packet)
        self.imu = ICM20948()
        # roll, pitch, and yaw of system reference frame
        self.roll = 0
        self.pitch = 0
        self.yaw = 0

    # Calculate the transformation matrix of a given joint with regards to system reference frame
    def forward_kinematics(self, target_joint, side):
        if side == "right":
            dh_transform = self.dh_transform_right
            joint_angles = self.joint_angles_right
        else:
            dh_transform = self.dh_transform_left
            joint_angles = self.joint_angles_left
        T = np.eye(4)  # Initialize as identity matrix
        for i in range(target_joint):
            theta += joint_angles[i]  # Add the joint angle to the theta parameter
            T_i = dh_transform[i]
            T = np.dot(T, T_i)  # Multiply the current transformation
        return T

    # Set dynamixel position
    def set_dynamixel_position(self, packet_handler, port_handler, position, id):
        if position > 1500:
            position = 1500
        elif position < 478:
            position = 478
        dxl_comm_result, dxl_error = packet_handler.write4ByteTxRx(
            port_handler, id, self.ADDR_PRO_GOAL_POSITION, position
        )
        if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
            raise Exception(
                f"Failed to set position: {packet_handler.getTxRxResult(dxl_comm_result)}"
            )

    # Update system reference frame roll and pitch with imu data (need time delay in main loop)
    def update_reference_angle(self):
        alpha = 0.98
        try:
            ax, ay, az, gx, gy, gz = self.imu.read_accelerometer_gyro_data()
        except:
            return
        accel_roll = math.atan2(ay, az) * 180 / math.pi
        accel_pitch = math.atan2(-ax, math.sqrt(ay**2 + az**2)) * 180 / math.pi

        # Integrate gyroscope data to calculate roll and pitch rate
        gyro_roll_rate = gx
        gyro_pitch_rate = gy

        # Update roll and pitch using the complementary filter
        roll = alpha * (self.roll + gyro_roll_rate * self.dt) + (1 - alpha) * accel_roll
        pitch = (
            alpha * (self.pitch + gyro_pitch_rate * self.dt) + (1 - alpha) * accel_pitch
        )
        self.roll = roll
        self.pitch = pitch

    # Helper: Given angle of motor, return the position
    def angle_to_position(angle):
        return int(((-angle + 90) / 360.0) * 4095)

    # Helper: Given position of motor, return the angle
    def position_to_angle(position):
        return -int((position / 4095.0) * 360.0)

    # Update the angle of each motor
    def update_motor_angles(self):
        for i in range(len(self.dynaindex)):
            id = self.dynaindex[i]
            packetHandler = self.packet_handler[i]
            portHandler = self.port_handler[i]
            dxl_present_position, dxl_comm_result, dxl_error = (
                packetHandler.read4ByteTxRx(
                    portHandler, id, self.ADDR_PRO_PRESENT_POSITION
                )
            )
            if dxl_comm_result != COMM_SUCCESS:
                print(
                    "Communication error: %s"
                    % packetHandler.getTxRxResult(dxl_comm_result)
                )
            elif dxl_error != 0:
                print("Error: %s" % packetHandler.getRxPacketError(dxl_error))
            else:
                if i <= 4:
                    self.joint_angles_right[i] = self.position_to_angle(
                        dxl_present_position
                    )
                else:
                    self.joint_angles_left[i - 5] = self.position_to_angle(
                        dxl_present_position
                    )

    # Return the roll and pitch of a given joint relative to absolute reference frame
    def get_abs_joint_angle(self, joint, side):
        T = self.forward_kinematics(joint, side)
        roll = np.arctan2(T[2, 1], T[2, 2])
        pitch = np.arctan2(-T[2, 0], np.sqrt(T[2, 1] ** 2 + T[2, 2] ** 2))
        return roll + self.roll, pitch + self.pitch
