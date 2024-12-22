"""
main.py

Top-level script to:
1) Ask user if they want to generate new data or use existing data.
2) Let them choose shape (cuboid/cylinder) and number of grasps (if new data).
3) Optionally run classification (train_model).
4) Models are saved in 'models' folder, data in 'data' folder.

Usage:
    python main.py
"""

import os
import pybullet as p
import pybullet_data
import time
import math
import random

# Import your training logic
import train_model
from robots.gripper import PR2Gripper, CustomGripper, select_gripper


def safe_step_simulation(num_steps=50, delay=0.01):
    """
    Safely step the simulation for num_steps, each with delay seconds,
    avoiding 'Not connected to physics server' errors if user closes PyBullet.
    """
    for _ in range(num_steps):
        if not p.isConnected():
            print("[INFO] PyBullet disconnected, stopping further steps.")
            return
        p.stepSimulation()
        time.sleep(delay)


class CustomObject:
    """
    Class to handle creation of both cuboid and cylinder objects
    in PyBullet for data generation.
    """

    def __init__(self, object_type="cuboid", size=None, height=None, radius=None):
        self.object_type = object_type.lower()
        self.size = size if size else [0.05, 0.05, 0.8]
        self.height = height if height else 0.8
        self.radius = radius if radius else 0.06
        self.object_id = None
        self._create_object()

    def _create_object(self):
        if self.object_type == "cuboid":
            self._create_cuboid()
        elif self.object_type == "cylinder":
            self._create_cylinder()
        else:
            raise ValueError(
                "Invalid object type. Supported: 'cuboid', 'cylinder'.")

    def _create_cuboid(self):
        half_extents = [s / 2 for s in self.size]
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[1, 1, 1, 1]
        )
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents
        )
        self.object_id = p.createMultiBody(
            baseMass=0.1,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[0, 0, self.size[2] / 2],
            useMaximalCoordinates=False
        )

    def _create_cylinder(self):
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=self.radius,
            length=self.height,
            rgbaColor=[1, 1, 1, 1]
        )
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=self.radius,
            height=self.height
        )
        self.object_id = p.createMultiBody(
            baseMass=0.1,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[0, 0, self.height / 2],
            useMaximalCoordinates=False
        )

    def get_id(self):
        return self.object_id


def generate_random_pose(object_position, height, step, total_steps, object_type):
    """
    Generate a random gripper pose for the specified object type, old format style.
    """
    midplane_line = [
        object_position[0],
        object_position[1],
        object_position[2] + height / 2
    ]

    if object_type == "cuboid":
        radius = 0.25
    else:  # cylinder
        radius = 0.22

    radius_variation = random.uniform(-0.05, 0.05)
    radius += radius_variation

    angle = (2 * math.pi) * (step / total_steps)
    gripper_x = midplane_line[0] + radius * math.cos(angle)
    gripper_y = midplane_line[1] + radius * \
        math.sin(angle) + random.uniform(-0.05, 0.05)
    gripper_z = midplane_line[2] - 0.1 + random.uniform(-0.1, 0.1)
    position = [gripper_x, gripper_y, gripper_z]

    direction_vec = [
        midplane_line[0] - position[0],
        midplane_line[1] - position[1],
        midplane_line[2] - position[2]
    ]
    mag = math.sqrt(sum(i**2 for i in direction_vec))
    if mag < 1e-8:
        direction_vec = [1.0, 0.0, 0.0]
        mag = 1.0
    else:
        direction_vec = [i / mag for i in direction_vec]

    yaw = math.atan2(direction_vec[1], direction_vec[0])
    pitch = math.asin(direction_vec[2])
    roll = random.uniform(-0.5, 0.5)

    orientation = p.getQuaternionFromEuler([roll, pitch, yaw])
    return position, orientation


def generate_data_for_shape(object_type, num_grasps=50):
    """
    Generate new data for the given shape (cuboid/cylinder) using old format CSV columns.
    Writes to data/grasp_data_{shape}.csv. 
    This time we use p.GUI so user can see the simulation.
    """

    # Connect in GUI mode so user sees simulation
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)

    p.loadURDF("plane.urdf")

    custom_obj = CustomObject(object_type=object_type)
    object_id = custom_obj.get_id()
    object_pos = [0, 0, custom_obj.height / 2]

    # Increase friction
    p.changeDynamics(object_id, -1, mass=0.2,
                     lateralFriction=1.2, spinningFriction=0.1)

    # Use PR2 for data gen
    gripper = select_gripper(
        "pr2", [0, 0, custom_obj.height + 0.2], [0, 0, 0, 1])

    rows = []

    for step in range(num_grasps):
        print(
            f"[INFO] Generating grasp {step+1}/{num_grasps} for {object_type}...")
        gripper.open_gripper()
        safe_step_simulation(30)

        # Reset object
        p.resetBasePositionAndOrientation(object_id, object_pos, [0, 0, 0, 1])
        safe_step_simulation(30)

        # Generate random pose
        position, orientation_quat = generate_random_pose(
            object_pos, custom_obj.height, step, num_grasps, object_type
        )
        gripper.set_position(position, orientation_quat)
        safe_step_simulation(30)

        gripper.close_gripper()
        safe_step_simulation(30)

        init_obj_pos, _ = p.getBasePositionAndOrientation(object_id)

        # Move up
        lift_target_z = position[2] + 0.3
        gripper.move_up_smoothly(
            target_z=lift_target_z, steps=100, delay=0.005)
        safe_step_simulation(100)

        final_obj_pos, _ = p.getBasePositionAndOrientation(object_id)
        init_z = init_obj_pos[2]
        final_z = final_obj_pos[2]
        delta_z = final_z - init_z

        # Simple success = 1 if delta_z > 0.1
        success = 1 if delta_z > 0.1 else 0

        # Orientation in RADIANS
        orientation_euler = p.getEulerFromQuaternion(orientation_quat)

        # old format columns
        row = [
            position[0],
            position[1],
            position[2],
            orientation_euler[0],
            orientation_euler[1],
            orientation_euler[2],
            init_z,
            final_z,
            delta_z,
            success
        ]
        rows.append(row)

        gripper.open_gripper()
        safe_step_simulation(30)

    # Disconnect
    p.disconnect()

    # Write CSV
    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)
    csv_file = os.path.join(data_folder, f"grasp_data_{object_type}.csv")

    import csv
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Position X", "Position Y", "Position Z",
                "Orientation Roll", "Orientation Pitch", "Orientation Yaw",
                "Initial Z", "Final Z", "Delta Z", "Success"
            ])
        for row in rows:
            writer.writerow(row)

    print(
        f"[INFO] Generated {num_grasps} grasps for shape '{object_type}' -> {csv_file}")


def main():
    """
    Main entry point:
    1) Ask user if they want new data or existing data
    2) If new data, ask shape & number of grasps, then optionally run classification
    3) If existing data, just ask shape to classify
    4) Classification logic is in 'train_model.py'
    """

    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)

    choice_data = input(
        "Do you want to generate new data? (y/n): ").strip().lower()
    if choice_data == 'y':
        shape = input("Enter object shape (cuboid/cylinder): ").strip().lower()
        if shape not in ["cuboid", "cylinder"]:
            print("[ERROR] Invalid shape. Exiting.")
            return
        try:
            num_grasps = int(input("Enter number of grasps to generate: "))
        except ValueError:
            print("[ERROR] Invalid integer. Exiting.")
            return

        # Generate data (GUI mode so user sees it)
        generate_data_for_shape(shape, num_grasps)
        # Ask if user wants classification
        choice_classify = input(
            "Do you want to run classification now? (y/n): ").strip().lower()
        if choice_classify == 'y':
            train_model.train_model(shape)
        else:
            print("[INFO] Classification was skipped. Exiting main.")
    else:
        # Use existing data
        shape = input(
            "Enter shape to classify (cuboid/cylinder): ").strip().lower()
        if shape not in ["cuboid", "cylinder"]:
            print("[ERROR] Invalid shape. Exiting.")
            return
        train_model.train_model(shape)


if __name__ == "__main__":
    main()
