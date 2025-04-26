import numpy as np
from dynamixel_sdk import *  # Dynamixel SDK (includes GroupBulkRead and GroupBulkWrite)
import math
import pickle
import time
import bob_params
import bob_sim

imu = True
from icm20948 import ICM20948  # IMU package


class Bob:
    # Dynamixel settings.
    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_VELOCITY = 104
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_POSITION = 132
    ADDR_OPERATING_MODE = 11
    VELOCITY_CONTROL_MODE = 1
    POSITION_CONTROL_MODE = 3
    PROTOCOL_VERSION = 2.0
    BAUDRATE = 57600
    DEVICENAME = "/dev/ttyUSB0"  # e.g., Raspberry Pi port
    TORQUE_ENABLE = 1
    TORQUE_DISABLE = 0

    def __init__(self):
        bp = bob_params.Bob_params()
        self.bs = bob_sim.Bob_sim()
        self.dt = bp.dt
        self.motor_ids = bp.motor_ids

        self.initial_angles = bp.initial_angles + np.array(
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        )

        # joint_angles: 2×5 array holding the current measured motor angles (in degrees).
        self.joint_angles = np.zeros((2, 5), dtype=float)

        self.joints_para = bp.joints_para

        self.part_mass = bp.part_mass
        self.motor_mass = bp.motor_mass
        self.total_mass = bp.total_mass

        # --------------------------------------------------
        # Initialize Dynamixel communication.
        # --------------------------------------------------
        self.port_handler = PortHandler(self.DEVICENAME)
        self.packet_handler = PacketHandler(self.PROTOCOL_VERSION)
        if not self.port_handler.openPort():
            raise Exception("Failed to open port")
        if not self.port_handler.setBaudRate(self.BAUDRATE):
            raise Exception("Failed to set baudrate")
        for motor_id in self.motor_ids.flatten():
            self.packet_handler.write1ByteTxRx(
                self.port_handler,
                motor_id,
                self.ADDR_OPERATING_MODE,
                self.POSITION_CONTROL_MODE,
            )
            dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
                self.port_handler, motor_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE
            )
            if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                raise Exception(f"Failed to enable torque on motor {motor_id}")

        # --------------------------------------------------
        # Initialize persistent BulkRead and BulkWrite objects.
        # --------------------------------------------------
        self.bulk_read = GroupBulkRead(self.port_handler, self.packet_handler)
        self.bulk_write = GroupBulkWrite(self.port_handler, self.packet_handler)
        for motor_id in self.motor_ids.flatten():
            if not self.bulk_read.addParam(motor_id, self.ADDR_PRESENT_POSITION, 4):
                print(f"Failed to add motor {motor_id} to bulk read.")

        # --------------------------------------------------
        # Initialize IMU if available.
        # --------------------------------------------------
        if imu:
            self.imu = ICM20948()
        self.roll = 0.0  # System roll (degrees)
        self.pitch = 0.0  # System pitch (degrees)

        # Load precomputed Jacobian lookup table.
        with open("jacobian_grid.pkl", "rb") as f:
            self.jacobian_grid = pickle.load(f)

    # --------------------------------------------------
    # Transformation helper methods.
    # --------------------------------------------------
    @staticmethod
    def Rz(theta, d):
        """Rotation about Z-axis by theta and translation along Z by d."""
        return np.array(
            [
                [np.cos(theta), -np.sin(theta), 0, 0],
                [np.sin(theta), np.cos(theta), 0, 0],
                [0, 0, 1, d],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

    @staticmethod
    def Rx(alpha, a):
        """Rotation about X-axis by alpha and translation along X by a."""
        return np.array(
            [
                [1, 0, 0, a],
                [0, np.cos(alpha), -np.sin(alpha), 0],
                [0, np.sin(alpha), np.cos(alpha), 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

    @staticmethod
    def Ry(r, b):
        """Rotation about Y-axis by r and translation along Y by b."""
        return np.array(
            [
                [np.cos(r), 0, np.sin(r), 0],
                [0, 1, 0, b],
                [-np.sin(r), 0, np.cos(r), 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

    def edh_transform(self, side, index):
        """
        Compute the EDH transformation matrix up to joint 'index' for the given side.
        :param side: "left" or "right" (case insensitive)
        :param index: Joint index (0 to 5)
        :return: 4×4 homogeneous transformation matrix.
        """
        side_idx = 0 if side.lower() != "left" else 1
        joints = self.joints_para[side_idx]
        T = np.eye(4, dtype=float)
        for i in range(index + 1):
            r, alpha, theta, b, a, d = joints[i]
            if i == 0:
                T_i = self.Rz(theta, d) @ self.Ry(r, b) @ self.Rx(alpha, a)
            else:
                T_i = self.Ry(r, b) @ self.Rx(alpha, a) @ self.Rz(theta, d)
            T = T @ T_i
        return T

    def set_dynamixel_position(self, position, motor_id):
        """Set the goal position (0–4095 scale) for a motor."""
        dxl_comm_result, dxl_error = self.packet_handler.write4ByteTxRx(
            self.port_handler, motor_id, self.ADDR_GOAL_POSITION, int(position)
        )
        if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
            raise Exception(
                f"Failed to set position on motor {motor_id}: {self.packet_handler.getTxRxResult(dxl_comm_result)}"
            )

    def update_reference_angle(self, t):
        """
        Update system roll and pitch using IMU data.
        Also update the base joint (joint 0) parameters for both legs.
        """
        alpha = 0.98
        try:
            ax, ay, az, gx, gy, gz = self.imu.read_accelerometer_gyro_data()
        except Exception as e:
            print("IMU read error:", e)
            return
        # might need to add additional constraints to the magnitude of a
        if 0.99 < az and az < 1.01:
            self.roll = 0
            self.pitch = 0
        else:
            accel_roll = math.degrees(math.atan2(ay, az))
            accel_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2)))
            roll_rate = gx
            pitch_rate = gy
            self.roll = alpha * (self.roll + roll_rate * t) + (1 - alpha) * (
                self.roll + accel_roll * t
            )
            self.pitch = alpha * (self.pitch + pitch_rate * t) + (1 - alpha) * (
                self.pitch + accel_pitch * t
            )

        new_base = np.array(
            [np.radians(self.pitch), np.radians(self.roll), 0, 0, 0, 2], dtype=float
        )
        self.joints_para[0, 0, :] = new_base  # Right leg base
        self.joints_para[1, 0, :] = new_base  # Left leg base

    def normalize_angle(self, angle):
        """Normalize an angle (degrees) to [0, 360). Supports scalars or arrays."""
        angle = np.mod(angle, 360)
        return np.mod(angle, 360)

    def angle_to_position(self, angle):
        """
        Convert an angle in degrees to a motor position (0–4095).
        Works vectorized.
        """
        norm_angle = self.normalize_angle(angle)
        return ((norm_angle / 360.0) * 4095).astype(int)

    def position_to_angle(self, position):
        """
        Convert motor positions (0–4095) to angles (degrees).
        Works vectorized.
        """
        angle = (position / 4095.0) * 360.0
        return self.normalize_angle(angle)

    # --------------------------------------------------
    # Bulk-read and Bulk-write Methods
    # note that when updating motor positions and storing in self.joint_angles the initial position is taken into consideration
    # however, when writing motor positions, the initial angles are not taken into consideration
    # --------------------------------------------------
    def bulk_read_positions(self, motor_ids=None):
        """
        Bulk-read the present positions for the given motor IDs.
        If motor_ids is None, read for all motors in self.motor_ids.

        Returns:
          A numpy array of positions (in the same shape as motor_ids).
        """
        if motor_ids is None:
            motor_ids = self.motor_ids
        dxl_comm_result = self.bulk_read.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print(
                "Bulk read communication error:",
                self.packet_handler.getTxRxResult(dxl_comm_result),
            )
            return None
        get_data = np.vectorize(
            lambda motor_id: self.bulk_read.getData(
                int(motor_id), self.ADDR_PRESENT_POSITION, 4
            )
        )
        positions = get_data(motor_ids)
        return positions

    def set_init_pos(self):
        self.initial_angles = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        positions = self.bulk_read_positions(self.motor_ids)
        angles = np.round(self.position_to_angle(positions) / 90) * 90
        with open("init_angles.pkl", "wb") as file:
            pickle.dump(angles, file)

    def bulk_write_positions(self, motor_ids, positions):
        """
        Bulk-write goal positions to the specified motor IDs.

        Parameters:
          motor_ids: a 1D array or list of motor IDs.
          positions: a 1D array or list of goal positions (integers).
        """
        self.bulk_write.clearParam()
        for motor_id, pos in zip(motor_ids, positions):
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(pos)),
                DXL_HIBYTE(DXL_LOWORD(pos)),
                DXL_LOBYTE(DXL_HIWORD(pos)),
                DXL_HIBYTE(DXL_HIWORD(pos)),
            ]
            self.bulk_write.addParam(
                motor_id, self.ADDR_GOAL_POSITION, 4, param_goal_position
            )
        self.bulk_write.txPacket()
        self.bulk_write.clearParam()

    def update_motor_angles(self):
        """
        Update motor angles using the persistent bulk_read object.
        Data is retrieved in a vectorized manner.
        The unified joint_angles array and joints_para theta values (for joints 1–5) are updated.
        """
        dxl_comm_result = self.bulk_read.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print(
                "Bulk read communication error:",
                self.packet_handler.getTxRxResult(dxl_comm_result),
            )
            return
        get_data = np.vectorize(
            lambda motor_id: self.bulk_read.getData(
                int(motor_id), self.ADDR_PRESENT_POSITION, 4
            )
        )
        positions = get_data(self.motor_ids)  # shape (2,5)
        new_angles = self.position_to_angle(positions)
        # Right leg: subtract initial offsets.
        angles_right = self.normalize_angle(
            new_angles[0, :] - self.initial_angles[0, :]
        )
        # Left leg: subtract initial offsets then invert sign.
        angles_left = self.normalize_angle(new_angles[1, :] - self.initial_angles[1, :])
        delta_right = angles_right - self.joint_angles[0, :]
        delta_left = angles_left - self.joint_angles[1, :]
        self.joint_angles[0, :] = angles_right
        self.joint_angles[1, :] = angles_left
        # Update EDH theta parameters for joints 1–5.
        self.joints_para[0, 1:6, 2] += (delta_right / 180.0) * np.pi
        self.joints_para[1, 1:6, 2] += (delta_left / 180.0) * np.pi

    def get_coordinates(self):
        """
        Update motor angles and reference angles.
        Compute (x, y, z) coordinates for each joint using EDH transforms.
        Returns a list of two 3×6 arrays:
          - Index 0: Right leg joints.
          - Index 1: Left leg joints.
        """
        self.update_motor_angles()
        self.update_reference_angle(robot.dt)
        coords = [np.zeros((3, 6), dtype=float) for _ in range(2)]
        for side in range(2):
            side_str = "right" if side == 0 else "left"
            for i in range(6):
                T = self.edh_transform(side_str, i)
                coords[side][:, i] = (T @ np.array([0, 0, 0, 1], dtype=float))[:3]
        return coords

    def get_part_coordinates(self, joint_coordinates):
        """
        Compute (x, y, z) coordinates for body parts based on joint coordinates.
        Returns a list of two 3×7 arrays:
          - Index 0: Right leg parts.
          - Index 1: Left leg parts.
        """
        right_coords, left_coords = joint_coordinates
        x_right = np.array(
            [
                right_coords[0, 0] - 30,
                right_coords[0, 0],
                right_coords[0, 1],
                right_coords[0, 2],
                (right_coords[0, 2] + right_coords[0, 3]) / 2,
                (right_coords[0, 3] + right_coords[0, 4]) / 2,
                right_coords[0, 4],
            ]
        )
        y_right = np.array(
            [
                right_coords[1, 0],
                right_coords[1, 0],
                right_coords[1, 1],
                right_coords[1, 2],
                (right_coords[1, 2] + right_coords[1, 3]) / 2,
                (right_coords[1, 3] + right_coords[1, 4]) / 2,
                right_coords[1, 4],
            ]
        )
        z_right = np.array(
            [
                right_coords[2, 0],
                right_coords[2, 0],
                right_coords[2, 1],
                right_coords[2, 2],
                (right_coords[2, 2] + right_coords[2, 3]) / 2,
                (right_coords[2, 2] + right_coords[2, 3]) / 2,
                right_coords[2, 4],
            ]
        )
        right_parts = np.vstack([x_right, y_right, z_right])
        x_left = np.array(
            [
                left_coords[0, 0] + 30,
                left_coords[0, 0],
                left_coords[0, 1],
                left_coords[0, 2],
                (left_coords[0, 2] + left_coords[0, 3]) / 2,
                (left_coords[0, 3] + left_coords[0, 4]) / 2,
                left_coords[0, 4],
            ]
        )
        y_left = np.array(
            [
                left_coords[1, 0],
                left_coords[1, 0],
                left_coords[1, 1],
                left_coords[1, 2],
                (left_coords[1, 2] + left_coords[1, 3]) / 2,
                (left_coords[1, 3] + left_coords[1, 4]) / 2,
                left_coords[1, 4],
            ]
        )
        z_left = np.array(
            [
                left_coords[2, 0],
                left_coords[2, 0],
                left_coords[2, 1],
                left_coords[2, 2],
                (left_coords[2, 2] + left_coords[2, 3]) / 2,
                (left_coords[2, 2] + left_coords[2, 3]) / 2,
                left_coords[2, 4],
            ]
        )
        left_parts = np.vstack([x_left, y_left, z_left])
        return [right_parts, left_parts]

    def get_com(self):
        """
        Compute the center of mass for each leg.
        Returns a 2×3 array:
          - Row 0: COM for the right leg.
          - Row 1: COM for the left leg.
        """
        coords = self.get_coordinates()  # [right_coords, left_coords]
        parts = self.get_part_coordinates(coords)
        com_right = (
            coords[0][:, 1:6] @ self.motor_mass + parts[0] @ self.part_mass
        ) / (self.total_mass / 2)
        com_left = (coords[1][:, 1:6] @ self.motor_mass + parts[1] @ self.part_mass) / (
            self.total_mass / 2
        )
        return np.array([com_right, com_left])

    def disable_torque(self):
        """Disable torque for all motors."""
        for motor_id in self.motor_ids.flatten():
            self.packet_handler.write1ByteTxRx(
                self.port_handler,
                motor_id,
                self.ADDR_TORQUE_ENABLE,
                self.TORQUE_DISABLE,
            )

    def disable_torque_except_ankle(self):
        """
        Disable torque for all motors except the ankle motors.
        (Right ankle: self.motor_ids[0,4]; Left ankle: self.motor_ids[1,4])
        """
        for motor_id in self.motor_ids.flatten():
            if motor_id not in (self.motor_ids[0, 4], self.motor_ids[1, 4]):
                self.packet_handler.write1ByteTxRx(
                    self.port_handler,
                    motor_id,
                    self.ADDR_TORQUE_ENABLE,
                    self.TORQUE_DISABLE,
                )

    def enable_torque(self):
        """Enable torque for all motors."""
        for motor_id in self.motor_ids.flatten():
            self.packet_handler.write1ByteTxRx(
                self.port_handler, motor_id, self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE
            )

    def sync_ankle(self):
        """
        Synchronize ankle positions to compensate for body roll.
        For each leg, the computed ankle angle is clamped to a safe range.
        """
        angle_right = self.normalize_angle(
            -self.roll + self.joint_angles[0, 2] - self.joint_angles[0, 3]
        )
        pos_right = self.angle_to_position(angle_right + self.initial_angles[0, 4])
        angle_left = self.normalize_angle(
            self.roll - self.joint_angles[1, 2] + self.joint_angles[1, 3]
        )
        pos_left = self.angle_to_position(angle_left + self.initial_angles[1, 4])

        self.bulk_write_positions([25,15],[pos_right,pos_left])

    def terminate(self):
        """Disable torque for all motors and close the port."""
        for motor_id in self.motor_ids.flatten():
            self.packet_handler.write1ByteTxRx(
                self.port_handler,
                motor_id,
                self.ADDR_TORQUE_ENABLE,
                self.TORQUE_DISABLE,
            )
        self.port_handler.closePort()

    def lookup_jacobian(self, current_roll, current_pitch):
        """
        Look up the precomputed Jacobian matrix corresponding to the current base
        orientation (roll and pitch in degrees). The lookup table (self.jacobian_grid)
        is keyed by (roll, pitch) tuples, where roll and pitch are in {0, 5, 10, ..., 50}.
        """
        roll_key = int(round(current_roll / 5.0) * 5)
        pitch_key = int(round(current_pitch / 5.0) * 5)
        roll_key = max(0, min(roll_key, 50))
        pitch_key = max(0, min(pitch_key, 50))
        key = (roll_key, pitch_key)
        if key in self.jacobian_grid:
            return self.jacobian_grid[key]
        else:
            raise ValueError(
                f"No Jacobian available for roll={roll_key}°, pitch={pitch_key}°"
            )

    def balance_controller_with_jacobian(self, coords, alpha):
        """
        Balancing controller using the precomputed Jacobian lookup.

        This method computes the COM error (the difference between the overall COM
        and the support (foot) center), looks up the corresponding 3×8 Jacobian based
        on the current base orientation (roll, pitch), and solves for joint corrections
        (dtheta) via the pseudoinverse. The corrections are applied to the eight non–ankle
        joints (assumed to be joints 0–3 for each leg). The new goal positions are computed
        in a vectorized manner and sent via a single bulk-write packet.
        """
        # (a) Get forward kinematics: ankle (foot) positions.
        foot_right = coords[0][:, 5]
        foot_left = coords[1][:, 5]
        foot_center = (foot_right + foot_left) / 2.0

        # (b) Compute the overall COM.
        com_arr = self.get_com()  # 2×3 array (row 0: right, row 1: left)
        overall_com = (com_arr[0, :] + com_arr[1, :]) / 2.0

        # (c) Compute COM error.
        error = foot_center - overall_com  # 3-element vector

        # (d) Lookup the appropriate Jacobian (3×8) using current base orientation.
        J = self.lookup_jacobian(self.roll, self.pitch)

        # (e) Solve for joint corrections (dtheta) in radians.
        dtheta = np.linalg.pinv(J) @ error  # 8-element vector (radians)
        dtheta_deg = np.degrees(dtheta)  # Convert to degrees

        # (f) Compute new commanded angles for the eight non–ankle joints.
        # Right leg non–ankle joints: indices 0–3; Left leg non–ankle joints: indices 0–3.
        cmd_angles_right = self.joint_angles[0, 0:4] + dtheta_deg[0:4]
        # + self.initial_angles[0, 0:4]
        cmd_angles_left = self.joint_angles[1, 0:4] + dtheta_deg[4:8]
        # + self.initial_angles[1, 0:4]
        cmd_angles = np.concatenate([cmd_angles_right, cmd_angles_left]) * alpha
        # Convert commanded angles (in degrees) to motor units.
        command_positions = self.angle_to_position(cmd_angles)

        # (g) Build a 1D array of motor IDs for the non–ankle joints.
        motor_ids_right = self.motor_ids[0, 0:4]
        motor_ids_left = self.motor_ids[1, 0:4]
        motor_ids_non_ankle = np.concatenate([motor_ids_right, motor_ids_left])

        print("error:" + str(error))
        print("theta:" + str(dtheta))
        # (h) Use the bulk write abstraction to send all goal positions.
        # self.bulk_write_positions(motor_ids_non_ankle, command_positions)

    def balance_controller_realtime(self, alpha):
        """
        Balancing controller using the real-time Jacobian computation.

        This method computes the COM error (the difference between the overall COM
        and the support (foot) center), computes the Jacobian matrix in real-time

        using the current base orientation (roll, pitch), and solves for joint corrections
        (dtheta) via the pseudoinverse. The corrections are applied to the eight non–ankle
        joints (assumed to be joints 0–3 for each leg). The new goal positions are computed
        in a vectorized manner and sent via a single bulk-write packet.
        """
        dtheta = self.bs.compute_jacobian_realtime(
            self.joint_angles, self.roll, self.pitch, alpha
        )
        self.bulk_write_positions(robot.motor_ids.flatten(),self.angle_to_position(self.joint_angles.flatten()+self.initial_angles.flatten()+dtheta.flatten()))

    def test_init_pos(self):
        self.bulk_write_positions(
            self.motor_ids.flatten(),
            self.angle_to_position(self.initial_angles.flatten()),
        )

    def movement_sequence(self, sequence, dt):
        for i in range(len(sequence)):
            pos = robot.angle_to_position(robot.initial_angles.flatten()) + sequence[i]
            self.bulk_write_positions(robot.motor_ids.flatten(), pos)
            time.sleep(dt)


try:
    robot = Bob()
    robot.test_init_pos()
    while True:
        robot.update_motor_angles()
        robot.update_reference_angle(robot.dt)
        robot.balance_controller_realtime(0.5)
        robot.sync_ankle()
        time.sleep(robot.dt)
except KeyboardInterrupt:
    print("Terminating...")
finally:
    robot.terminate()

# -------------------------
# test motor position
# -------------------------
"""try:
    robot = Bob()
    iteration = 0
    robot.disable_torque()
    while True:
        robot.get_coordinates()
        if iteration % 10 == 0:
            print('right'+str(robot.joint_angles[0,1]))
            print('left'+str(robot.joint_angles[1,1]))
        time.sleep(robot.dt)
        iteration += 1
except KeyboardInterrupt:
    print("Terminating...")
finally:
    robot.terminate()"""

"""# -------------------------
# com animation
# -------------------------
if __name__ == "__main__":
    try:
        robot = Bob()
        robot.disable_torque()
        com = np.empty((0, 3))
        joints = np.empty((0, 3))
        while True:
            robot.update_reference_angle(robot.dt)
            com = np.vstack((com, robot.get_com()[1]))
            coordinates = robot.get_coordinates()
            coordinates = np.vstack((np.transpose(coordinates[1])))
            joints = np.vstack((joints, coordinates))
            # print(robot.get_com())
            # You may call other controllers (e.g., sync_ankle) as needed.
            # robot.balance_controller_with_jacobian()
            # robot.sync_ankle()
            # For demonstration, sleep for dt seconds.
            time.sleep(robot.dt)
    except KeyboardInterrupt:
        with open("com.pkl", "wb") as file:
            pickle.dump(com, file)
        with open("coordinates.pkl", "wb") as file:
            pickle.dump(joints, file)
        print(com)
        print(joints)
        print("Terminating...")
    finally:
        robot.terminate()"""
