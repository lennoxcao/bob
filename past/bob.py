import numpy as np
from dynamixel_sdk import *  # Dynamixel SDK library for controlling the motor
import math

imu = True
from icm20948 import ICM20948  # ICM20948 Python package for IMU

# {left_hip_1:11,left_hip_2:12,left_hip_3:13,left_knee:14,left_ankle:15,right_hip_1:21,right_hip_2:22,right_hip_3:23,right_knee:24}
# coordinate system is defined such that y points forward, z points upward


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
    DEVICENAME = "/dev/ttyUSB0"  # pi
    # DEVICENAME = "/dev/cu.usbserial-FT9BTH5F"  #mac
    TORQUE_ENABLE = 1
    TORQUE_DISABLE = 0
    # right motors start with 2, left motors start with 1 (top to bottom)
    dynaindex = [21, 22, 23, 24, 25, 11, 12, 13, 14, 15]

    def __init__(self):
        # update speed
        self.dt = 0.01
        # list of all the motorized joints
        self.joints_right = self.dynaindex[:5]
        self.joints_left = self.dynaindex[5:]
        # Initialized motor positions (degrees)
        self.joint_angles_right = [0, 0, 0, 0, 0]
        self.initial_angle_right = [180, 180, 180, 0, 90]
        self.joint_angles_left = [0, 0, 0, 0, 0]
        self.initial_angle_left = [180, 180, 180, 180, 270]
        # kinematics parameter (a, alpha, d, theta)
        right_joint0 = np.array([0, 0, 0, 0, 0, 2], dtype="float64")
        left_joint0 = np.array([0, 0, 0, 0, 0, 2], dtype="float64")
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
            right_joint0,
            right_joint1,
            right_joint2,
            right_joint3,
            right_joint4,
            right_joint5,
        ]
        self.left_joints_para = [
            left_joint0,
            left_joint1,
            left_joint2,
            left_joint3,
            left_joint4,
            left_joint5,
        ]
        # Mass parameters
        # [hip_body_connector, hip_joint_cover, top_hip_joint, lateral_hip_joint, femur, shin, ankle]
        self.part_mass = [17.2, 8.8, 22.2, 12.5, 79, 32.8, 11.3]
        # [x1, x2, x3, x4, x5]
        self.motor_mass = [23, 65, 65, 23, 23]
        self.total_mass = (np.sum(self.part_mass) + np.sum(self.motor_mass)) * 2

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
        if imu:
            self.imu = ICM20948()
        # roll, pitch, and yaw of system reference frame (degrees)
        self.roll = 0
        self.pitch = 0

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

        if index == 0:
            return Rz(theta, d) @ Ry(r, b) @ Rx(alpha, a)
        else:
            T = Ry(r, b) @ Rx(alpha, a) @ Rz(theta, d)
            return self.edh_transform(side, index - 1) @ T

    # Set dynamixel position
    def set_dynamixel_position(self, position, id):
        """if position > 1500:
            position = 1500
        elif position < 478:
            position = 478"""
        dxl_comm_result, dxl_error = (self.packet_handler).write4ByteTxRx(
            self.port_handler, id, self.ADDR_GOAL_POSITION, position
        )
        if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
            raise Exception(
                f"Failed to set position: {packet_handler.getTxRxResult(dxl_comm_result)}"
            )

    # Update system reference frame roll and pitch with imu data (need time delay in main loop)
    def update_reference_angle(self, t):
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
        roll = alpha * (self.roll + gyro_roll_rate * t) + (1 - alpha) * accel_roll * t
        pitch = (
            alpha * (self.pitch + gyro_pitch_rate * t) + (1 - alpha) * accel_pitch * t
        )
        self.roll = roll
        self.pitch = pitch
        self.left_joints_para[0] = [
            self.pitch / 180 * np.pi,
            self.roll / 180 * np.pi,
            0,
            0,
            0,
            2,
        ]
        self.right_joints_para[0] = [
            self.pitch / 180 * np.pi,
            self.roll / 180 * np.pi,
            0,
            0,
            0,
            2,
        ]

    # Helper: Given angle of motor, return the position
    def angle_to_position(self, angle):
        return int((self.normalize_angle(angle) / 360.0) * 4095)

    # Normalize angle
    def normalize_angle(self, angle):
        while angle >= 360:
            angle -= 360
        while angle < 0:
            angle += 360
        return angle

    # Helper: Given position of motor, return the angle
    def position_to_angle(self, position):
        angle = int((position / 4095.0) * 360.0)
        return self.normalize_angle(angle)

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
                new_angle = self.position_to_angle(dxl_present_position)
                if i <= 4:
                    new_angle -= self.initial_angle_right[i]
                    new_angle = self.normalize_angle(new_angle)
                    old_angle = self.joint_angles_right[i]
                    self.joint_angles_right[i] = new_angle
                    self.right_joints_para[i + 1] += np.array(
                        [0, 0, ((new_angle - old_angle) / 180) * np.pi, 0, 0, 0]
                    )
                else:
                    new_angle -= self.initial_angle_left[i - 5]
                    new_angle = -self.normalize_angle(new_angle)
                    old_angle = self.joint_angles_left[i - 5]
                    self.joint_angles_left[i - 5] = new_angle
                    self.left_joints_para[i - 5 + 1] += np.array(
                        [0, 0, ((new_angle - old_angle) / 180) * np.pi, 0, 0, 0]
                    )

    # Return the roll and pitch of a given joint relative to absolute reference frame
    def get_abs_joint_angle(self, joint, side):
        T = self.forward_kinematics(joint, side)
        roll = np.arctan2(T[2, 1], T[2, 2])
        pitch = np.arctan2(-T[2, 0], np.sqrt(T[2, 1] ** 2 + T[2, 2] ** 2))
        return roll + self.roll, pitch + self.pitch

    # Disable torque
    def disable_torque(self):
        for i in self.dynaindex:
            self.packet_handler.write1ByteTxRx(
                self.port_handler, i, self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE
            )

    def disable_torque_except_ankle(self):
        for i in self.dynaindex:
            if i != 15 and i != 25:
                self.packet_handler.write1ByteTxRx(
                    self.port_handler, i, self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE
                )

    # Enable torque
    def enable_torque(self):
        for i in self.dynaindex:
            self.packet_handler.write1ByteTxRx(
                self.port_handler, i, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE
            )

    # Update the joints and return the coordinates of the joints
    def get_coordinates(self):
        self.update_motor_angles()
        x_left = np.zeros(6)
        y_left = np.zeros(6)
        z_left = np.zeros(6)
        x_right = np.zeros(6)
        y_right = np.zeros(6)
        z_right = np.zeros(6)
        for i in range(6):
            [x_left[i], y_left[i], z_left[i], a] = self.edh_transform("left", i) @ [
                0,
                0,
                0,
                1,
            ]
            [x_right[i], y_right[i], z_right[i], a] = self.edh_transform("right", i) @ [
                0,
                0,
                0,
                1,
            ]
        return [[x_left, y_left, z_left], [x_right, y_right, z_right]]

    # Return the coordinates of parts based on the coordinate of joints
    def get_part_coordinates(self, joint_coordinates):
        joint_x_left = joint_coordinates[0][0]
        joint_y_left = joint_coordinates[0][1]
        joint_z_left = joint_coordinates[0][2]
        joint_x_right = joint_coordinates[1][0]
        joint_y_right = joint_coordinates[1][1]
        joint_z_right = joint_coordinates[1][2]
        # [hip_body_connector, hip_joint_cover, top_hip_joint, lateral_hip_joint, femur, shin, ankle]
        # [x1, x2, x3, x4, x5]
        x_left = np.array(
            [
                joint_x_left[0] + 30,
                joint_x_left[0],
                joint_x_left[1],
                joint_x_left[2],
                (joint_x_left[2] + joint_x_left[3]) / 2,
                (joint_x_left[3] + joint_x_left[4]) / 2,
                joint_x_left[4],
            ]
        )
        x_right = np.array(
            [
                joint_x_right[0] - 30,
                joint_x_right[0],
                joint_x_right[1],
                joint_x_right[2],
                (joint_x_right[2] + joint_x_right[3]) / 2,
                (joint_x_right[3] + joint_x_right[4]) / 2,
                joint_x_right[4],
            ]
        )
        y_left = np.array(
            [
                joint_y_left[0],
                joint_y_left[0],
                joint_y_left[1],
                joint_y_left[2],
                (joint_y_left[2] + joint_y_left[3]) / 2,
                (joint_y_left[3] + joint_y_left[4]) / 2,
                joint_y_left[4],
            ]
        )
        y_right = np.array(
            [
                joint_y_right[0],
                joint_y_right[0],
                joint_y_right[1],
                joint_y_right[2],
                (joint_y_right[2] + joint_y_right[3]) / 2,
                (joint_y_right[3] + joint_y_right[4]) / 2,
                joint_y_right[4],
            ]
        )
        z_left = np.array(
            [
                joint_z_left[0],
                joint_z_left[0],
                joint_z_left[1],
                joint_z_left[2],
                (joint_z_left[2] + joint_z_left[3]) / 2,
                (joint_z_left[2] + joint_z_left[3]) / 2,
                joint_z_left[4],
            ]
        )
        z_right = np.array(
            [
                joint_z_right[0],
                joint_z_right[0],
                joint_z_right[1],
                joint_z_right[2],
                (joint_z_right[2] + joint_z_right[3]) / 2,
                (joint_z_right[2] + joint_z_right[3]) / 2,
                joint_z_right[4],
            ]
        )
        return [[x_left, y_left, z_left], [x_right, y_right, z_right]]

    # Calculate COM
    def get_com(self):
        joints_coordinates = self.get_coordinates()
        joints_coordinates_left, joints_coordinates_right = joints_coordinates
        part_coordinates = self.get_part_coordinates(joints_coordinates)
        part_coordinates_left, part_coordinates_right = part_coordinates
        com = np.zeros((2, 3))
        for i in range(3):
            com[0][i] = (
                (
                    np.dot(self.motor_mass, joints_coordinates_left[i])
                    + np.dot(self.part_mass, part_coordinates_left[i])
                )
                / self.total_mass
                / 2
            )
            com[1][i] = (
                (
                    np.dot(self.motor_mass, joints_coordinates_right[i])
                    + np.dot(self.part_mass, part_coordinates_right[i])
                )
                / self.total_mass
                / 2
            )
        return com

    # Disable torque and close port
    def terminate(self):
        for i in self.dynaindex:
            self.packet_handler.write1ByteTxRx(
                self.port_handler, i, self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE
            )
        self.port_handler.closePort()

    # Sync ankle
    def sync_ankle(self):
        angle_left = self.normalize_angle(
            -self.roll + self.joint_angles_left[2] - self.joint_angles_left[3]
        )
        angle_right = self.normalize_angle(
            self.roll - self.joint_angles_right[2] + self.joint_angles_right[3]
        )
        if angle_right > 48 and angle_right <= 180:
            angle_right = 47
        elif angle_right < 312 and angle_right >= 180:
            angle_right = 311
        if angle_left > 47 and angle_left <= 180:
            angle_left = 47
        elif angle_left < 312 and angle_left >= 180:
            angle_left = 311
        self.set_dynamixel_position(
            self.angle_to_position(angle_right + self.initial_angle_right[4]), 25
        )
        self.set_dynamixel_position(
            self.angle_to_position(angle_left + self.initial_angle_left[4]), 15
        )


bob1 = bob()
"""bob1 = bob()
start = time.time()
#bob1.disable_torque_except_ankle()
while True:
    bob1.update_motor_angles()
    end = time.time()
    bob1.update_reference_angle(end-start)
    start = time.time()
    bob1.sync_ankle()
    time.sleep(0.0001)"""
