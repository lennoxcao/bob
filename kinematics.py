import numpy as np


# Define the transformation matrix using DH parameters
def dh_transform(a, alpha, d, theta):
    return np.array(
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


# Function to calculate the forward kinematics
def forward_kinematics(dh_params, joint_angles, target_joint=None):
    T = np.eye(4)  # Initialize as identity matrix
    target_joint = target_joint if target_joint is not None else len(dh_params)
    for i in range(target_joint):
        a, alpha, d, theta = dh_params[i]
        theta += joint_angles[i]  # Add the joint angle to the theta parameter
        T_i = dh_transform(a, alpha, d, theta)
        T = np.dot(T, T_i)  # Multiply the current transformation
    return T


# DH parameters: [a, alpha, d, theta]
# Assuming we have 5 joints with arbitrary DH parameters for illustration
dh_params = np.transpose(
    [
        [0, 0, 0, -108.5, -99.5],
        [0, np.pi / 2, np.pi / 2, 0, 0],
        [0, -47.45, -55.55, 0, 0],
        [0, -np.pi / 2, 0, np.pi, 0],
    ]
)

print(dh_params * [1, -1, -1, 1])

joint_angles = [0, 0, 0, 0, 0]


T_05 = forward_kinematics(dh_params, joint_angles)

print(T_05)
