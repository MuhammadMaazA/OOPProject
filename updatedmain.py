import pybullet as p
import pybullet_data
import time
import math
import csv
import os
import random
from gripper import Gripper
from evaluate import evaluate_grasp


class CustomObject:
    """
    Class to handle the creation of both cuboid and cylinder objects.
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
            raise ValueError("Invalid object type. Supported types: 'cuboid', 'cylinder'.")

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
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[0, 0, self.size[2] / 2]
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
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[0, 0, self.height / 2]
        )

    def get_id(self):
        return self.object_id


def generate_pose(object_position, height, step, total_steps, object_type):
    """
    Generate a gripper pose with variability, considering object type.
    """
    midplane_line = [object_position[0], object_position[1], object_position[2] + height / 2]

    # Different radii for cuboid and cylinder
    if object_type == "cuboid":
        radius = 0.25  # Larger radius for cuboid
    elif object_type == "cylinder":
        radius = 0.22  # Smaller radius for cylinder
    else:
        raise ValueError("Unsupported object type for pose generation.")

    # Add randomness to the radius for variability
    radius_variation = random.uniform(-0.05, 0.05)
    radius += radius_variation

    # Calculate the gripper position
    angle = (2 * math.pi) * (step / total_steps)
    gripper_x = midplane_line[0] + radius * math.cos(angle)
    gripper_y = midplane_line[1] + radius * math.sin(angle) + random.uniform(-0.05, 0.05)
    gripper_z = midplane_line[2] - 0.1 + random.uniform(-0.1, 0.1)
    gripper_position = [gripper_x, gripper_y, gripper_z]

    # Calculate the orientation
    direction_vector = [
        midplane_line[0] - gripper_position[0],
        midplane_line[1] - gripper_position[1],
        midplane_line[2] - gripper_position[2]
    ]
    magnitude = math.sqrt(sum(i ** 2 for i in direction_vector))
    direction_vector = [i / magnitude for i in direction_vector]

    yaw = math.atan2(direction_vector[1], direction_vector[0])
    pitch = math.asin(direction_vector[2])
    roll = random.uniform(-0.5, 0.5)
    orientation = p.getQuaternionFromEuler([roll, pitch, yaw])

    return gripper_position, orientation


def main():
    try:
        # PyBullet setup
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        # Select object type
        object_type = input("Enter object type (cuboid/cylinder): ").strip().lower()
        if object_type not in ["cuboid", "cylinder"]:
            raise ValueError("Invalid object type. Supported types: 'cuboid', 'cylinder'.")

        custom_object = CustomObject(object_type=object_type)
        object_height = custom_object.height
        object_id = custom_object.get_id()
        initial_object_position = [0, 0, object_height / 2]
        initial_object_orientation = [0, 0, 0, 1]

        # Gripper setup
        gripper = Gripper([0, 0, object_height + 0.2], [0, 0, 0, 1])
        gripper.open_gripper()

        # Data collection
        total_steps = 100
        iterations = 10
        output_file = f"grasp_data_{object_type}.csv"
        file_exists = os.path.isfile(output_file)

        # Append to the file if it already exists
        with open(output_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow([
                    "Step", "Position X", "Position Y", "Position Z",
                    "Orientation Roll", "Orientation Pitch", "Orientation Yaw",
                    "Initial Z", "Final Z", "Delta Z", "Success"
                ])

            for iteration in range(iterations):
                for step in range(total_steps):
                    # Reset the object's position and orientation
                    p.resetBasePositionAndOrientation(object_id, initial_object_position, initial_object_orientation)

                    # Generate a new pose for the gripper
                    position, orientation = generate_pose([0, 0, object_height / 2], object_height, step, total_steps, object_type)
                    gripper.set_position(position, orientation)
                    time.sleep(0.2)

                    # Perform grasping
                    gripper.close_gripper()
                    time.sleep(0.2)

                    # Move gripper up to evaluate the grasp
                    gripper.move_up_smoothly(target_z=position[2] + 0.3)
                    time.sleep(0.2)

                    # Evaluate grasp success
                    success, delta_z, final_position = evaluate_grasp(object_id, initial_object_position)
                    writer.writerow([
                        iteration * total_steps + step + 1,
                        position[0], position[1], position[2],
                        orientation[0], orientation[1], orientation[2],
                        initial_object_position[2], final_position[2], delta_z, success
                    ])

                    # Open the gripper for the next step
                    gripper.open_gripper()

        print(f"Data collection complete. Results saved to {output_file}.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
