import numpy as np
from icm20948 import ICM20948  # ICM20948 Python package for IMU
from dynamixel_sdk import *  # Dynamixel SDK library for controlling the motor
import math

# {left_hip_1:11,left_hip_2:12,left_hip_3:13,left_knee:14,left_ankle:15,right_hip_1:21,right_hip_2:22,right_hip_3:23,right_knee:24}
# neutral motor angle: {hip1:180,hip2:180}


class bob:
    # Dynamixel settings
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_VELOCITY = 104
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_POSITION = 132
    ADDR_OPERATING_MODE = 11
    VELOCITY_CONTROL_MODE = 1  # Value for setting velocity control mode
    POSITION_CONTROL_MODE = 3  # Value for setting position control mode
    PROTOCOL_VERSION = 2.0
    BAUDRATE = 57600
    DEVICENAME = "/dev/ttyUSB0"  # Adjust to your port
    TORQUE_ENABLE = 1
    TORQUE_DISABLE = 0
    # right motors start with 2, left motors start with 1
    dynaindex = [21, 22, 23, 24, 25, 11, 12, 13, 14, 15]

    def __init__(self):
        # update speed
        self.dt = 0.01
        # list of all the motorized joints
        self.joints_right = self.dynaindex[:5]
        self.joints_left = self.dynaindex[5:]
        # Initialized motor positions
        self.joint_angles_right = [0, 0, 0, 0, 0]
        self.joint_angles_left = [0, 0, 0, 0, 0]
        # kinematics parameter (a, alpha, d, theta)
        right_joint1 = [np.pi, 0, self.joint_angles_right[0], 0, -55.3, 0]
        left_joint1 = [0, np.pi, self.joint_angles_left[0], 0, -55.3, 0]
        right_joint2 = [
            np.pi / 2,
            -np.pi / 2,
            self.joint_angles_right[1],
            -35.175,
            -59.25,
            0,
        ]
        left_joint2 = [
            np.pi / 2,
            np.pi / 2,
            self.joint_angles_left[1],
            35.175,
            -59.25,
            0,
        ]
        right_joint3 = [
            np.pi / 2,
            np.pi / 2,
            np.pi / 2 + self.joint_angles_right[2],
            0,
            -33.125,
            0,
        ]
        left_joint3 = [
            np.pi / 2,
            -np.pi / 2,
            -np.pi / 2 + self.joint_angles_left[2],
            0,
            -33.125,
            0,
        ]
        right_joint4 = [np.pi, 0, self.joint_angles_right[3], 0, 108.5, 0]
        left_joint4 = [np.pi, 0, self.joint_angles_left[3], 0, 108.5, 0]
        right_joint5 = [0, np.pi, self.joint_angles_right[4], 0, 97, 0]
        left_joint5 = [0, np.pi, self.joint_angles_left[4], 0, 97, 0]
        self.right_joints_para = [
            right_joint1,
            right_joint2,
            right_joint3,
            right_joint4,
            right_joint5,
        ]
        self.left_joints_para = [
            left_joint1,
            left_joint2,
            left_joint3,
            left_joint4,
            left_joint5,
        ]

        # Initialize Dynamixel
        self.port_handler = PortHandler(self.DEVICENAME)
        self.packet_handler = PacketHandler(self.PROTOCOL_VERSION)
        # Open port
        if not self.port_handler.openPort():
            raise Exception("Failed to open port")
        # Set baudrate
        if not self.port_handler.setBaudRate(self.BAUDRATE):
            raise Exception("Failed to set baudrate")
        for i in self.dynaindex:
            self.packet_handler.write1ByteTxRx(
                self.port_handler,
                i,
                self.ADDR_OPERATING_MODE,
                self.POSITION_CONTROL_MODE,
            )
            dxl_comm_result, dxl_error = (self.packet_handler).write1ByteTxRx(
                self.port_handler, i, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE
            )
            if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                raise Exception("Failed to enable torque")

        # Initialize dyanmixel
        self.imu = ICM20948()
        # roll, pitch, and yaw of system reference frame
        self.roll = 0
        self.pitch = 0
        self.yaw = 0

    # Calculate transformation matrix based on joint parameters
    def edh_transform(self, side, index):
        """
        Compute the Expanded Denavit-Hartenberg (EDH) transformation matrix
        using parameters theta, d, alpha, a, r, and b.
        """

        def Rz(theta, d):
            """Rotation and translation matrix about the Z-axis by angle theta."""
            return np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0, 0],
                    [np.sin(theta), np.cos(theta), 0, 0],
                    [0, 0, 1, d],
                    [0, 0, 0, 1],
                ]
            )

        def Rx(alpha, a):
            """Rotation and translation matrix about the X-axis by angle alpha."""
            return np.array(
                [
                    [1, 0, 0, a],
                    [0, np.cos(alpha), -np.sin(alpha), 0],
                    [0, np.sin(alpha), np.cos(alpha), 0],
                    [0, 0, 0, 1],
                ]
            )

        def Ry(r, b):
            """Additional rotation matrix about the Y-axis by angle r."""
            return np.array(
                [
                    [np.cos(r), 0, np.sin(r), 0],
                    [0, 1, 0, b],
                    [-np.sin(r), 0, np.cos(r), 0],
                    [0, 0, 0, 1],
                ]
            )

        if side == "left":
            joint = self.left_joints_para[index]
        else:
            joint = self.right_joints_para[index]
        [r, alpha, theta, b, a, d] = joint
        # Combine transformations in the order specified by EDH convention
        T = Ry(r, b) @ Rx(alpha, a) @ Rz(theta, d)
        if index == 0:
            return T
        else:
            return T @ self.edh_transform(joint, side, index - 1)

    # Set dynamixel position
    def set_dynamixel_position(self, position, id):
        """if position > 1500:
            position = 1500
        elif position < 478:
            position = 478"""
        print(position)
        dxl_comm_result, dxl_error = (self.packet_handler).write4ByteTxRx(
            self.port_handler, id, self.ADDR_GOAL_POSITION, position
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
    def angle_to_position(self, angle):
        return int(((-angle + 90) / 360.0) * 4095)

    # Helper: Given position of motor, return the angle
    def position_to_angle(self, position):
        return -int((position / 4095.0) * 360.0)

    # Update the angle of each motor
    def update_motor_angles(self):
        for i in range(len(self.dynaindex)):
            id = self.dynaindex[i]
            packetHandler = self.packet_handler
            portHandler = self.port_handler
            dxl_present_position, dxl_comm_result, dxl_error = (
                packetHandler.read4ByteTxRx(portHandler, id, self.ADDR_PRESENT_POSITION)
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

    # Disable torque and close port
    def terminate(self):
        for i in self.dynaindex:
            self.packet_handler.write1ByteTxRx(
                self.port_handler, i, self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE
            )
        self.port_handler.closePort()


try:
    bob1 = bob()
    bob1.set_dynamixel_position(2000, bob1.dynaindex[4])
    bob1.update_motor_angles()
except KeyboardInterrupt:
    # Disable Dynamixel torque before quitting
    bob1.terminate()
