import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pickle

with open("com.pkl", "rb") as file:
    center_of_mass_data = pickle.load(file)
with open("coordinates.pkl", "rb") as file:
    joints_data = pickle.load(file)

num_frames = len(center_of_mass_data)
# Number of joints per frame
NUM_JOINTS = 12
FRAME_INTERVAL = 10  # Milliseconds (10ms = 0.01s)

# Number of joints per frame
NUM_JOINTS = 12

# Create figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Set plot limits (adjust based on actual data range)
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_zlim(-100, 100)

# Plot objects for updating
(joints_plot,) = ax.plot([], [], [], "o", color="blue", markersize=8, label="Joints")
(com1_plot,) = ax.plot(
    [], [], [], "o", color="red", markersize=10, label="Center of Mass 1"
)
(com2_plot,) = ax.plot(
    [], [], [], "o", color="green", markersize=10, label="Center of Mass 2"
)

# Add legend
ax.legend()

# State tracking for current frame
current_frame = [0]


def update_plot():
    frame = current_frame[0]

    # Extract joint positions for the current frame
    joints = joints_data[frame * NUM_JOINTS : (frame + 1) * NUM_JOINTS]

    # Extract two centers of mass for the current frame
    com1 = center_of_mass_data[frame * 2]
    com2 = center_of_mass_data[frame * 2 + 1]

    # Update joint positions
    joints_plot.set_data(joints[:, 0], joints[:, 1])
    joints_plot.set_3d_properties(joints[:, 2])

    # Update center of mass positions
    com1_plot.set_data(com1[0], com1[1])
    com1_plot.set_3d_properties(com1[2])

    com2_plot.set_data(com2[0], com2[1])
    com2_plot.set_3d_properties(com2[2])

    # Update plot title with current frame
    ax.set_title(f"Frame {frame + 1}/{num_frames}")

    fig.canvas.draw_idle()


def next_frame(event):
    if current_frame[0] < num_frames - 1:
        current_frame[0] += 1
        update_plot()


def prev_frame(event):
    if current_frame[0] > 0:
        current_frame[0] -= 1
        update_plot()


# Create "Next" button
ax_next = plt.axes([0.8, 0.01, 0.1, 0.05])
btn_next = Button(ax_next, "Next")
btn_next.on_clicked(next_frame)

# Create "Previous" button
ax_prev = plt.axes([0.65, 0.01, 0.1, 0.05])
btn_prev = Button(ax_prev, "Previous")
btn_prev.on_clicked(prev_frame)

# Initialize with the first frame
update_plot()

# Display plot
# plt.show()
