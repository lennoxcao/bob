import numpy as np


def edh_transform(theta, d, alpha, a, r, b):
    """
    Compute the Expanded Denavit-Hartenberg (EDH) transformation matrix
    using parameters theta, d, alpha, a, r, and b.
    """

    def Rz(theta):
        """Rotation matrix about the Z-axis by angle theta."""
        return np.array(
            [
                [np.cos(theta), -np.sin(theta), 0, 0],
                [np.sin(theta), np.cos(theta), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

    def Tz(d):
        """Translation matrix along the Z-axis by distance d."""
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, d], [0, 0, 0, 1]])

    def Rx(alpha):
        """Rotation matrix about the X-axis by angle alpha."""
        return np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(alpha), -np.sin(alpha), 0],
                [0, np.sin(alpha), np.cos(alpha), 0],
                [0, 0, 0, 1],
            ]
        )

    def Tx(a):
        """Translation matrix along the X-axis by distance a."""
        return np.array([[1, 0, 0, a], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def Ry(r):
        """Additional rotation matrix about the Y-axis by angle r."""
        return np.array(
            [
                [np.cos(r), 0, np.sin(r), 0],
                [0, 1, 0, 0],
                [-np.sin(r), 0, np.cos(r), 0],
                [0, 0, 0, 1],
            ]
        )

    def Ty(b):
        """Additional translation matrix along the Y-axis by distance b."""
        return np.array([[1, 0, 0, 0], [0, 1, 0, b], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Combine transformations in the order specified by EDH convention
    T = Rz(theta) @ Tz(d) @ Rx(alpha) @ Tx(a) @ Ry(r) @ Ty(b)
    return T


# Example usage:
# Define EDH parameters for a link
theta = np.radians(30)  # rotation about z-axis in radians
d = 2.0  # translation along z-axis
alpha = np.radians(45)  # rotation about x-axis in radians
a = 3.0  # translation along x-axis
r = np.radians(15)  # additional rotation about y-axis in radians
b = 1.0  # additional translation along y-axis

# Compute the transformation matrix
T = edh_transform(theta, d, alpha, a, r, b)

print("EDH Transformation Matrix:")
print(T)
