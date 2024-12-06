import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Load data
data = pd.read_csv("updated_grasp_data.csv")

# Define color mapping
colors = {
    0: 'red',    # Failure
    1: 'green',  # Success
    2: 'blue'    # Almost
}

# Common variables for plotting the cuboid boundary
boundary_x = [-0.1, -0.1, 0.1, 0.1, -0.1, -0.1, 0.1, 0.1]
boundary_y = [-0.1, 0.1, 0.1, -0.1, -0.1, 0.1, 0.1, -0.1]
boundary_z = [0, 0, 0, 0, 0.8, 0.8, 0.8, 0.8]

edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom square
    [4, 5], [5, 6], [6, 7], [7, 4],  # Top square
    [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
]

# Plot Gripper Position
def plot_gripper_position(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(data['Position X'], data['Position Y'], data['Position Z'], 
                         c=data['Success'].map(colors), s=10)

    # Adding legend
    labels = ['Failure', 'Success', 'Almost']
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in range(3)]
    ax.legend(handles, labels)

    ax.set_xlabel('Gripper X')
    ax.set_ylabel('Gripper Y')
    ax.set_zlabel('Gripper Z')
    ax.set_title('Group 10 Gripper Visualization - Position Only')

    # Adding the URDF cuboid boundary
    for edge in edges:
        ax.plot([boundary_x[edge[0]], boundary_x[edge[1]]],
                [boundary_y[edge[0]], boundary_y[edge[1]]],
                [boundary_z[edge[0]], boundary_z[edge[1]]], color='black', alpha=0.8)

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
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in range(3)]
    ax.legend(handles, labels)

    ax.set_xlabel('Gripper X')
    ax.set_ylabel('Gripper Y')
    ax.set_zlabel('Gripper Z')
    ax.set_title('Group 10 Gripper Visualization - Position and Orientation')

    # Adding the URDF cuboid boundary
    for edge in edges:
        ax.plot([boundary_x[edge[0]], boundary_x[edge[1]]],
                [boundary_y[edge[0]], boundary_y[edge[1]]],
                [boundary_z[edge[0]], boundary_z[edge[1]]], color='black', alpha=0.8)

    plt.show()

# Plot Gripper Outcome
def plot_gripper_outcome(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data['Position X'], data['Position Y'], data['Position Z'], 
               c=data['Success'].map(colors), s=10)

    # Adding legend
    labels = ['Failure', 'Success', 'Almost']
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in range(3)]
    ax.legend(handles, labels)

    ax.set_xlabel('Gripper X')
    ax.set_ylabel('Gripper Y')
    ax.set_zlabel('Gripper Z')
    ax.set_title('Gripper Visualization - Grasping Outcome')

    # Adding the URDF cuboid boundary
    for edge in edges:
        ax.plot([boundary_x[edge[0]], boundary_x[edge[1]]],
                [boundary_y[edge[0]], boundary_y[edge[1]]],
                [boundary_z[edge[0]], boundary_z[edge[1]]], color='black', alpha=0.8)

    plt.show()

# Run the functions to generate plots
plot_gripper_position(data)
plot_gripper_position_orientation(data)
plot_gripper_outcome(data)
