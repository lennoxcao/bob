import numpy as np
import pickle


class Bob_params:
    def __init__(self):
        # Time step (s) for updates
        self.dt = 0.01

        # motor_ids: 2×5 array.
        # Row 0 (index 0): Right leg motors (IDs 21, 22, 23, 24, 25)
        # Row 1 (index 1): Left leg motors  (IDs 11, 12, 13, 14, 15)
        self.motor_ids = np.array(
            [[21, 22, 23, 24, 25], [11, 12, 13, 14, 15]], dtype=int
        )

        # initial_angles: 2×5 array of motor "home" offsets (in degrees).
        # Row 0: Right leg; Row 1: Left leg.
        with open("init_angles.pkl", "rb") as file:
            init_angles = pickle.load(file)
        self.initial_angles = init_angles

        # joints_para is a 3-D array of shape (2, 6, 6):
        #   For each leg (first dimension):
        #     Row 0: Joint 0 (base) parameters [r, alpha, theta, b, a, d]
        #     Rows 1–5: Joints 1–5.
        # Convention: Row 0 → right leg, Row 1 → left leg.
        # (The theta value is taken from self.joint_angles.)
        self.joints_para = np.array(
            [
                [  # Right leg joint parameters.
                    [0, 0, 0, 0, 0, 2],  # Joint 0 (base)
                    [np.pi, 0, 0, 0, -55.3, 0],  # Joint 1
                    [
                        np.pi / 2,
                        -np.pi / 2,
                        0,
                        -35.175,
                        -59.25,
                        0,
                    ],  # Joint 2
                    [
                        np.pi / 2,
                        np.pi / 2,
                        np.pi / 2,
                        0,
                        -33.125,
                        0,
                    ],  # Joint 3
                    [np.pi, 0, 0, 0, 108.5, 0],  # Joint 4
                    [0, np.pi, 0, 0, 97, 0],  # Joint 5 (ankle)
                ],
                [  # Left leg joint parameters.
                    [0, 0, 0, 0, 0, 2],  # Joint 0 (base)
                    [0, np.pi, 0, 0, -55.3, 0],  # Joint 1
                    [
                        np.pi / 2,
                        np.pi / 2,
                        0,
                        35.175,
                        -59.25,
                        0,
                    ],  # Joint 2
                    [
                        np.pi / 2,
                        -np.pi / 2,
                        -np.pi / 2 + 0,
                        0,
                        -33.125,
                        0,
                    ],  # Joint 3
                    [np.pi, 0, 0, 0, 108.5, 0],  # Joint 4
                    [0, np.pi, 0, 0, 97, 0],  # Joint 5 (ankle)
                ],
            ],
            dtype=float,
        )

        # --------------------------------------------------
        # Mass parameters (example values)
        # --------------------------------------------------
        self.part_mass = np.array([17.2, 8.8, 22.2, 12.5, 79, 32.8, 11.3], dtype=float)
        self.motor_mass = np.array([23, 65, 65, 23, 23], dtype=float)
        self.total_mass = (np.sum(self.part_mass) + np.sum(self.motor_mass)) * 2
