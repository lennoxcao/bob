import numpy as np


# y x z
def edh_transform(joint):
    """
    Compute the Expanded Denavit-Hartenberg (EDH) transformation matrix
    using parameters theta, d, alpha, a, r, and b.
    """
    [r, alpha, theta, b, a, d] = joint

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

    # Combine transformations in the order specified by EDH convention
    T = Ry(r, b) @ Rx(alpha, a) @ Rz(theta, d)
    return T

motor_angle = [0,0,0,0,0]

left_joint1 = [np.pi, 0, motor_angle[0], 0, 55.3, 0]
left_joint2 = [np.pi / 2, -np.pi / 2, motor_angle[1], -35.175, -59.25, 0]
left_joint3 = [np.pi / 2, np.pi / 2, np.pi / 2+motor_angle[2], 0, -33.125, 0]
left_joint4 = [np.pi, 0, motor_angle[3], 0, 108.5, 0]
left_joint5 = [0, np.pi, motor_angle[4], 0, 97, 0]

p0 = [0, 0, 0, 1]
p1 = edh_transform(left_joint1) @ p0
p2 = edh_transform(left_joint1) @ edh_transform(left_joint2) @ p0
p3 = (
    edh_transform(left_joint1)
    @ edh_transform(left_joint2)
    @ edh_transform(left_joint3)
    @ p0
)
p4 = (
    edh_transform(left_joint1)
    @ edh_transform(left_joint2)
    @ edh_transform(left_joint3)
    @ edh_transform(left_joint4)
    @ p0
)
p5 = (
    edh_transform(left_joint1)
    @ edh_transform(left_joint2)
    @ edh_transform(left_joint3)
    @ edh_transform(left_joint4)
    @ edh_transform(left_joint5)
    @ p0
)

points = np.array([p0, p1, p2, p3, p4, p5])




