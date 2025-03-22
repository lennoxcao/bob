import numpy as np
import math
import pickle
import bob_params


class BobSim:
    def __init__(self):
        bp = bob_params.Bob_params()
        self.dt = bp.dt
        self.motor_ids = bp.motor_ids
        self.initial_angles = bp.initial_angles

        self.joint_angles = np.zeros((2, 5), dtype=float)

        self.joints_para = bp.joints_para

        self.roll = 0

        self.pitch = 0

        # This dictionary will hold the precomputed Jacobian lookup table.
        self.jacobian_grid = {}

    # --- Kinematic Transformations ---
    @staticmethod
    def Rz(theta, d):
        """Rotation about the Z axis and translation along Z."""
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
        """Rotation about the X axis and translation along X."""
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
        """Rotation about the Y axis and translation along Y."""
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

    def update_reference_angle(self, roll, pitch):
        """
        Update system roll and pitch using IMU data.
        Also update the base joint (joint 0) parameters for both legs.
        """
        self.roll = roll
        self.pitch = pitch

        new_base = np.array(
            [np.radians(self.pitch), np.radians(self.roll), 0, 0, 0, 2], dtype=float
        )
        self.joints_para[0, 0, :] = new_base  # Right leg base
        self.joints_para[1, 0, :] = new_base  # Left leg base

    def update_motor_angles(self, joint_angles):
        """
        Update motor angles using the persistent bulk_read object.
        Data is retrieved in a vectorized manner.
        The unified joint_angles array and joints_para theta values (for joints 1–5) are updated.
        """
        new_angles = joint_angles
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

    def get_coordinates(self, roll, pitch, joint_angles):
        """
        Compute the (x, y, z) coordinates for each joint (using the EDH transforms).
        Returns:
          A list of two 3x6 numpy arrays:
            - Index 0: Right leg joint positions.
            - Index 1: Left leg joint positions.
        """
        self.update_motor_angles(joint_angles)
        self.update_reference_angle(roll, pitch)
        coords = [np.zeros((3, 6), dtype=float), np.zeros((3, 6), dtype=float)]
        for i in range(6):
            T_right = self.edh_transform("right", i)
            T_left = self.edh_transform("left", i)
            coords[0][:, i] = (T_right @ np.array([0, 0, 0, 1], dtype=float))[:3]
            coords[1][:, i] = (T_left @ np.array([0, 0, 0, 1], dtype=float))[:3]
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

    # --- Jacobian Grid Computation ---

    def compute_jacobian_realtime(self, joint_angles, roll, pitch):
        joint_angles = self.joint_angles.copy()
        com = self.get_com()
        target_position = (
            com[0][:2] + com[1][:2]
        ) / 2  # Midpoint of the two foot positions

        adjustments = np.zeros_like(joint_angles)

        for leg in range(2):  # 0 = right, 1 = left
            for joint in range(5):
                # Perturb the current joint slightly to compute the gradient
                delta = step_size
                joint_angles[leg, joint] += delta
                self.update_motor_angles(joint_angles)
                new_com = self.get_com()

                new_distance = np.linalg.norm(
                    (new_com[0][:2] + new_com[1][:2]) / 2 - target_position
                )
                original_distance = np.linalg.norm(target_position - com[leg][:2])

                if new_distance < original_distance:
                    # Move in the positive direction
                    adjustments[leg, joint] = delta
                else:
                    # Try the negative direction
                    joint_angles[leg, joint] -= 2 * delta
                    self.update_motor_angles(joint_angles)
                    new_com = self.get_com()
                    new_distance = np.linalg.norm(
                        (new_com[0][:2] + new_com[1][:2]) / 2 - target_position
                    )

                    if new_distance < original_distance:
                        adjustments[leg, joint] = -delta
                    else:
                        adjustments[leg, joint] = 0

                # Reset joint to original position after testing
                joint_angles[leg, joint] += delta
                self.update_motor_angles(joint_angles)

        return adjustments

    def compute_jacobian_grid(self):
        """
        Precompute the Jacobian matrix (3x8) of the overall COM (average of both legs)
        with respect to the non–ankle joint angles for various base (torso) orientations.

        We vary base roll and pitch from 0° to 50° (inclusive) in 5° increments.
        The base joint (joint 0 for each leg) is updated with:

            new_base = [radians(pitch), radians(roll), 0, 0, 0, 2]

        For each (roll, pitch) combination, we use finite differences (delta = 0.5°)
        to compute the sensitivity of the overall COM (3-vector) to each of the 8 non–ankle
        joint angles (joints 0–3 for each leg). The resulting 3×8 Jacobian is stored in a
        lookup dictionary keyed by the (roll, pitch) tuple (in degrees).

        Returns:
          A dictionary with keys (roll, pitch) and values a 3x8 numpy array (the Jacobian).
        """
        jacobian_table = {}
        # Backup the current joint angles and EDH parameters.
        backup_joint_angles = self.joint_angles.copy()
        backup_joints_para = self.joints_para.copy()

        # Perturbation: 0.5° in radians.
        delta = np.radians(0.5)

        # Loop over roll and pitch values (0, 5, 10, …, 50).
        for roll_deg in range(0, 55, 5):
            for pitch_deg in range(0, 55, 5):
                # Update the base joint (joint 0 for each leg) with new orientation.
                new_base = np.array(
                    [np.radians(pitch_deg), np.radians(roll_deg), 0, 0, 0, 2],
                    dtype=float,
                )
                self.joints_para[0, 0, :] = new_base  # right leg base
                self.joints_para[1, 0, :] = new_base  # left leg base

                # Compute the current overall COM.
                com_arr = self.get_com()
                overall_com = (com_arr[0, :] + com_arr[1, :]) / 2.0

                # Prepare an empty Jacobian: 3 (x,y,z) x 8 (4 joints per leg, non–ankle).
                num_joints = 8
                J = np.zeros((3, num_joints))

                # Backup joint angles for this (roll, pitch) configuration.
                backup_angles = self.joint_angles.copy()
                col = 0
                for side in range(2):  # side 0: right, side 1: left
                    for m in range(0, 4):  # non–ankle joints (indices 0,1,2,3)
                        # Perturb joint m for this leg by delta (adding 0.5°).
                        self.joint_angles[side, m] += np.degrees(delta)
                        # Update the corresponding EDH parameter (theta) for joint (m+1).
                        if side == 0:
                            self.joints_para[0, m + 1, 2] = np.radians(
                                self.joint_angles[0, m]
                            )
                        else:
                            self.joints_para[1, m + 1, 2] = np.radians(
                                self.joint_angles[1, m]
                            )

                        # Recompute COM after the perturbation.
                        new_com_arr = self.get_com()
                        new_overall_com = (new_com_arr[0, :] + new_com_arr[1, :]) / 2.0
                        # Approximate the partial derivative.
                        dCOM = (new_overall_com - overall_com) / delta
                        J[:, col] = dCOM

                        # Reset the joint angle.
                        self.joint_angles[side, m] = backup_angles[side, m]
                        if side == 0:
                            self.joints_para[0, m + 1, 2] = np.radians(
                                self.joint_angles[0, m]
                            )
                        else:
                            self.joints_para[1, m + 1, 2] = np.radians(
                                self.joint_angles[1, m]
                            )
                        col += 1

                # Store the Jacobian for this (roll, pitch) combination.
                jacobian_table[(roll_deg, pitch_deg)] = J.copy()

        # Restore original joint angles and EDH parameters.
        self.joint_angles = backup_joint_angles
        self.joints_para = backup_joints_para

        # Save the lookup table for later use.
        self.jacobian_grid = jacobian_table
        return jacobian_table


"""# --- Main: Compute the Jacobian grid without hardware connection ---
if __name__ == "__main__":
    robot_sim = BobSim()
    jacobian_grid = robot_sim.compute_jacobian_grid()

    with open("jacobian_grid.pkl", "wb") as file:
        pickle.dump(jacobian_grid, file)

    # For example, print the Jacobian at roll=10°, pitch=15°:
    key = (30, 15)
    if key in jacobian_grid:
        print(f"Jacobian at roll={key[0]}°, pitch={key[1]}°:")
        print(jacobian_grid[key])
    else:
        print(f"No Jacobian entry for roll={key[0]}°, pitch={key[1]}°")"""
