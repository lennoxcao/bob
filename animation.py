import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from ipywidgets import interact, FloatSlider
from IPython.display import display
import ipywidgets as widgets
from IPython.display import display
import time
import pickle


def plot_2d_points_for_animation(ax, coordinates):
    x, y, z = coordinates
    ax.clear()  # Clear the previous frame
    ax.plot(-y, z, marker="o", linestyle="-", color="b", markerfacecolor="r")
    ax.set_xlim([-200, 200])
    ax.set_ylim([-300, 150])
    ax.set_xlabel("Y")
    ax.set_ylabel("Z")


def create_animation(coordinates_list, interval=500):
    """
    Create an animation from a list of y and z coordinate pairs.

    :param coordinates_list: List of tuples [(y1, z1), (y2, z2), ...].
    :param interval: Time between frames in milliseconds.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame):
        coordinate = coordinates_list[frame]
        plot_2d_points_for_animation(ax, coordinate)

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(coordinates_list), interval=interval, repeat=True
    )

    plt.show()


with open("sequence1.pkl", "rb") as file:
    coordinates,interval = pickle.load(file)
create_animation(coordinates, interval=np.average(np.array(interval))*1000)
