import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

cuboid_data = pd.read_csv(os.path.join(
    "data", "updated_grasp_data_cuboid_with_predictions.csv"))
cylinder_data = pd.read_csv(os.path.join(
    "data", "updated_grasp_data_cylinder_with_predictions.csv"))

# Define color mapping
colors = {
    0: 'red',    # Failure
    1: 'green',  # Success
    2: 'blue'    # Almost
}

# Cuboid boundary parameters
boundary_x = [-0.1, -0.1, 0.1, 0.1, -0.1, -0.1, 0.1, 0.1]
boundary_y = [-0.1, 0.1, 0.1, -0.1, -0.1, 0.1, 0.1, -0.1]
boundary_z = [0, 0, 0, 0, 0.8, 0.8, 0.8, 0.8]

edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom square
    [4, 5], [5, 6], [6, 7], [7, 4],  # Top square
    [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
]

# Cylinder boundary parameters
cylinder_radius = 0.06
cylinder_height = 0.8
theta = np.linspace(0, 2 * np.pi, 100)
z = np.linspace(0, cylinder_height, 100)
theta_grid, z_grid = np.meshgrid(theta, z)
x_cylinder = cylinder_radius * np.cos(theta_grid)
y_cylinder = cylinder_radius * np.sin(theta_grid)

# Common plot functions


def plot_gripper(data, boundary_type="cuboid"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of gripper positions
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
    ax.set_title(
        f'Gripper Visualization - {boundary_type.capitalize()} Boundary')

    # Adding the boundary
    if boundary_type == "cuboid":
        for edge in edges:
            ax.plot([boundary_x[edge[0]], boundary_x[edge[1]]],
                    [boundary_y[edge[0]], boundary_y[edge[1]]],
                    [boundary_z[edge[0]], boundary_z[edge[1]]], color='black', alpha=0.8)
    elif boundary_type == "cylinder":
        ax.plot_surface(x_cylinder, y_cylinder, z_grid,
                        color='black', alpha=0.4)

    plt.show()


def plot_gripper_orientation(data, boundary_type="cuboid"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Quiver plot of gripper orientations
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
    ax.set_title(
        f'Gripper Visualization - {boundary_type.capitalize()} Boundary with Orientation')

    # Adding the boundary
    if boundary_type == "cuboid":
        for edge in edges:
            ax.plot([boundary_x[edge[0]], boundary_x[edge[1]]],
                    [boundary_y[edge[0]], boundary_y[edge[1]]],
                    [boundary_z[edge[0]], boundary_z[edge[1]]], color='black', alpha=0.8)
    elif boundary_type == "cylinder":
        ax.plot_surface(x_cylinder, y_cylinder, z_grid,
                        color='black', alpha=0.4)

    plt.show()


# Run the functions for both scenarios
print("Visualizing Cuboid Boundary:")
plot_gripper(cuboid_data, boundary_type="cuboid")
plot_gripper_orientation(cuboid_data, boundary_type="cuboid")

print("Visualizing Cylinder Boundary:")
plot_gripper(cylinder_data, boundary_type="cylinder")
plot_gripper_orientation(cylinder_data, boundary_type="cylinder")
