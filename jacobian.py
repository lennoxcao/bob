import numpy as np
import math
import pickle


class BobSim:
    def __init__(self):
        self.dt = 0.01  # control time step (seconds)

        # --- Unified (simulated) motor and joint data ---
        # motor_ids: 2x5 array (row 0: right leg, row 1: left leg)
        self.motor_ids = np.array(
            [[21, 22, 23, 24, 25], [11, 12, 13, 14, 15]], dtype=int
        )

        # initial_angles: 2x5 array (in degrees)
        self.initial_angles = np.array(
            [
                [180, 180, 180, 0, 90],  # Right leg offsets
                [180, 180, 180, 180, 270],  # Left leg offsets
            ],
            dtype=float,
        )

        # joint_angles: current measured angles (2x5, in degrees)
        self.joint_angles = np.zeros((2, 5), dtype=float)

        # --- EDH joint parameters ---
        # We use a 3-D array of shape (2, 6, 6):
        # For each leg (row 0: right, row 1: left):
        #   Row 0: Base joint parameters: [r, alpha, theta, b, a, d]
        #   Rows 1–5: Parameters for joints 1–5.
        # (The theta value is taken from self.joint_angles.)
        self.joints_para = np.zeros((2, 6, 6), dtype=float)
        # Set base joint (joint 0) for both legs.
        self.joints_para[0, 0, :] = np.array([0, 0, 0, 0, 0, 2], dtype=float)
        self.joints_para[1, 0, :] = np.array([0, 0, 0, 0, 0, 2], dtype=float)
        # Joint 1:
        self.joints_para[0, 1, :] = np.array(
            [np.pi, 0, self.joint_angles[0, 0], 0, -55.3, 0], dtype=float
        )
        self.joints_para[1, 1, :] = np.array(
            [0, np.pi, self.joint_angles[1, 0], 0, -55.3, 0], dtype=float
        )
        # Joint 2:
        self.joints_para[0, 2, :] = np.array(
            [np.pi / 2, -np.pi / 2, self.joint_angles[0, 1], -35.175, -59.25, 0],
            dtype=float,
        )
        self.joints_para[1, 2, :] = np.array(
            [np.pi / 2, np.pi / 2, self.joint_angles[1, 1], 35.175, -59.25, 0],
            dtype=float,
        )
        # Joint 3:
        self.joints_para[0, 3, :] = np.array(
            [np.pi / 2, np.pi / 2, np.pi / 2 + self.joint_angles[0, 2], 0, -33.125, 0],
            dtype=float,
        )
        self.joints_para[1, 3, :] = np.array(
            [
                np.pi / 2,
                -np.pi / 2,
                -np.pi / 2 + self.joint_angles[1, 2],
                0,
                -33.125,
                0,
            ],
            dtype=float,
        )
        # Joint 4:
        self.joints_para[0, 4, :] = np.array(
            [np.pi, 0, self.joint_angles[0, 3], 0, 108.5, 0], dtype=float
        )
        self.joints_para[1, 4, :] = np.array(
            [np.pi, 0, self.joint_angles[1, 3], 0, 108.5, 0], dtype=float
        )
        # Joint 5 (ankle; we assume this remains fixed for balancing)
        self.joints_para[0, 5, :] = np.array(
            [0, np.pi, self.joint_angles[0, 4], 0, 97, 0], dtype=float
        )
        self.joints_para[1, 5, :] = np.array(
            [0, np.pi, self.joint_angles[1, 4], 0, 97, 0], dtype=float
        )

        # This dictionary will hold the precomputed Jacobian lookup table.
        self.jacobian_grid = {}

        # For any derivative-based control (if needed)
        self.prev_error_pitch = 0.0

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
        Compute the homogeneous transformation matrix using the EDH parameters
        for joints 0 to index (inclusive) for the specified leg.

        Parameters:
          side: "right" or "left"
          index: Joint index (0 to 5)
        Returns:
          A 4x4 numpy array representing the transformation.
        """
        # Select the proper row from joints_para: row 0 → right, row 1 → left.
        if side.lower() == "right":
            joints = self.joints_para[0]
        else:
            joints = self.joints_para[1]
        T = np.eye(4, dtype=float)
        for i in range(index + 1):
            r, alpha, theta, b, a, d = joints[i]
            if i == 0:
                T_i = self.Rz(theta, d) @ self.Ry(r, b) @ self.Rx(alpha, a)
            else:
                T_i = self.Ry(r, b) @ self.Rx(alpha, a) @ self.Rz(theta, d)
            T = T @ T_i
        return T

    def get_coordinates(self):
        """
        Compute the (x, y, z) coordinates for each joint (using the EDH transforms).
        Returns:
          A list of two 3x6 numpy arrays:
            - Index 0: Right leg joint positions.
            - Index 1: Left leg joint positions.
        """
        coords = [np.zeros((3, 6), dtype=float), np.zeros((3, 6), dtype=float)]
        for i in range(6):
            T_right = self.edh_transform("right", i)
            T_left = self.edh_transform("left", i)
            coords[0][:, i] = (T_right @ np.array([0, 0, 0, 1], dtype=float))[:3]
            coords[1][:, i] = (T_left @ np.array([0, 0, 0, 1], dtype=float))[:3]
        return coords

    def get_com(self):
        """
        Compute a simple overall center-of-mass (COM) as the average of joints 1-5
        for each leg, then average the legs.
        Returns:
          A 2x3 numpy array where row 0 is the COM for the right leg and row 1 for the left leg.
        """
        coords = self.get_coordinates()
        com_right = np.mean(coords[0][:, 1:6], axis=1)
        com_left = np.mean(coords[1][:, 1:6], axis=1)
        return np.array([com_right, com_left])

    # --- Jacobian Grid Computation ---
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


# --- Main: Compute the Jacobian grid without hardware connection ---
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
        print(f"No Jacobian entry for roll={key[0]}°, pitch={key[1]}°")
