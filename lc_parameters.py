import numpy as np


# y x z
def edh_transform(r, alpha, theta, b, a, d):
    """
    Compute the Expanded Denavit-Hartenberg (EDH) transformation matrix
    using parameters theta, d, alpha, a, r, and b.
    """

    def Rz(theta):
        """Rotation matrix about the Z-axis by angle theta."""
        return np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )

    def Rx(alpha):
        """Rotation matrix about the X-axis by angle alpha."""
        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(alpha), -np.sin(alpha)],
                [0, np.sin(alpha), np.cos(alpha)],
            ]
        )

    def Ry(r):
        """Additional rotation matrix about the Y-axis by angle r."""
        return np.array(
            [[np.cos(r), 0, np.sin(r)], [0, 1, 0], [-np.sin(r), 0, np.cos(r)]]
        )

    # Combine transformations in the order specified by EDH convention
    T = Rz(theta) @ Rx(alpha) @ Ry(r)
    T = np.concatenate((T, np.array([[a], [b], [d]])), axis=1)
    T = np.vstack((T, np.array([0, 0, 0, 1])))
    return T


joint1 = [np.pi, 0, 0, -55.3, 0, 0]
joint2 = [np.pi / 2, -np.pi / 2, 0, -35.175, 0, -59.25]
joint3 = [0, np.pi / 2, 0, 0, 0, 33.125]
joint4 = [np.pi, 0, 0, 0, -108.5, 0]
joint5 = [0, np.pi, 0, 0, 97, 0]

right = []
