import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Load data
data = pd.read_csv("updated_grasp_data_cylinder_with_predictions.csv")

# Define color mapping
colors = {
    0: 'red',    # Failure
    1: 'green',  # Success
    2: 'blue'    # Almost
}

# Parameters for plotting the cylinder boundary
cylinder_radius = 0.06
cylinder_height = 0.8
theta = np.linspace(0, 2 * np.pi, 100)
z = np.linspace(0, cylinder_height, 100)
theta_grid, z_grid = np.meshgrid(theta, z)
x_cylinder = cylinder_radius * np.cos(theta_grid)
y_cylinder = cylinder_radius * np.sin(theta_grid)

# Plot Gripper Position


def plot_gripper_position(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(data['Position X'], data['Position Y'], data['Position Z'],
                         c=data['Success'].map(colors), s=10)

    # Adding legend
    labels = ['Failure', 'Success', 'Almost']
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=colors[i], markersize=10) for i in range(3)]
    ax.legend(handles, labels)

    ax.set_xlabel('Gripper X')
    ax.set_ylabel('Gripper Y')
    ax.set_zlabel('Gripper Z')
    ax.set_title('Gripper Visualization - Position Only')

    # Adding the URDF cylinder boundary with darker color and higher opacity
    ax.plot_surface(x_cylinder, y_cylinder, z_grid, color='black', alpha=0.4)

    plt.show()

# Plot Gripper Position and Orientation


def plot_gripper_position_orientation(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for _, row in data.iterrows():
        ax.quiver(row['Position X'], row['Position Y'], row['Position Z'],
                  row['Orientation Roll'], row['Orientation Pitch'], row['Orientation Yaw'],
                  color=colors[row['Success']], alpha=0.5, length=0.05)

    # Adding legend
    labels = ['Failure', 'Success', 'Almost']
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=colors[i], markersize=10) for i in range(3)]
    ax.legend(handles, labels)

    ax.set_xlabel('Gripper X')
    ax.set_ylabel('Gripper Y')
    ax.set_zlabel('Gripper Z')
    ax.set_title('Gripper Visualization - Position and Orientation')

    # Adding the URDF cylinder boundary with darker color and higher opacity
    ax.plot_surface(x_cylinder, y_cylinder, z_grid, color='black', alpha=0.4)

    plt.show()

# Plot Gripper Outcome


def plot_gripper_outcome(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data['Position X'], data['Position Y'], data['Position Z'],
               c=data['Success'].map(colors), s=10)

    # Adding legend
    labels = ['Failure', 'Success', 'Almost']
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=colors[i], markersize=10) for i in range(3)]
    ax.legend(handles, labels)

    ax.set_xlabel('Gripper X')
    ax.set_ylabel('Gripper Y')
    ax.set_zlabel('Gripper Z')
    ax.set_title('Gripper Visualization - Grasping Outcome')

    # Adding the URDF cylinder boundary with darker color and higher opacity
    ax.plot_surface(x_cylinder, y_cylinder, z_grid, color='black', alpha=0.4)

    plt.show()


# Run the functions to generate plots
plot_gripper_position(data)
plot_gripper_position_orientation(data)
plot_gripper_outcome(data)
