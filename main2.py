import pybullet as p
import pybullet_data
import time
import math
import csv
import os
import random
from gripper import Gripper
from evaluate import evaluate_grasp


def create_cylinder_object(height=0.8, radius=0.06):
    """Create a tall cylinder object."""
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=[1, 1, 1, 1]
    )
    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        height=height
    )

    object_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=[0, 0, height / 2]
    )
    return object_id


def generate_pose(object_position, height, orientation_type, step, total_steps, radius):
    """Generate a gripper pose with variability."""
    midplane_line = [object_position[0],
                     object_position[1], object_position[2] + height / 2]

    # Randomized radius and angle
    radius_variation = random.uniform(-0.05, 0.05)
    angle = (2 * math.pi) * (step / total_steps)
    radius += radius_variation

    gripper_x = midplane_line[0] + radius * math.cos(angle)
    gripper_y = midplane_line[1] + radius * \
        math.sin(angle) + random.uniform(-0.05, 0.05)
    # Adjusted to start slightly below
    gripper_z = midplane_line[2] - 0.1 + random.uniform(-0.1, 0.1)

    gripper_position = [gripper_x, gripper_y, gripper_z]

    # Direction vector and orientation
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

    if orientation_type == "vertical":
        pitch += math.pi / 2

    orientation = p.getQuaternionFromEuler([roll, pitch, yaw])

    return gripper_position, orientation


def evaluate_grasp(object_id, initial_position):
    """
    Evaluate if the grasp was successful based on the change in the object's z-position.
    Also, return the change in z-position (delta_z) and the final position of the object.
    """
    time.sleep(0.5)
    final_position, _ = p.getBasePositionAndOrientation(object_id)

    # Extract z-components of the initial and final positions
    initial_z = initial_position[2]
    final_z = final_position[2]

    # Calculate the change in z (delta_z)
    delta_z = final_z - initial_z

    # Determine success if delta_z exceeds the threshold
    if delta_z > 0.1:
        success = 1  # Success
    elif 0.05 <= delta_z <= 0.1:
        success = 2  # Almost
    else:
        success = 0  # Failure

    return success, delta_z, final_position


def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)

    plane_id = p.loadURDF("plane.urdf")
    object_height = 0.8
    object_radius = 0.06  # Restored cylinder radius
    object_position = [0, 0, object_height / 2]
    object_id = create_cylinder_object(
        height=object_height, radius=object_radius)

    # Adjusted initial Z position
    gripper = Gripper([0, 0, object_height / 2],
                      p.getQuaternionFromEuler([0, 0, 0]))
    gripper.open_gripper()

    total_steps = 100
    iterations = 10
    output_file = "grasp_data_cylinder.csv"
    file_exists = os.path.isfile(output_file)

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
                p.resetBasePositionAndOrientation(
                    object_id, object_position, [0, 0, 0, 1])

                position, orientation = generate_pose(
                    object_position, object_height, "horizontal", step, total_steps, 0.22)  # Adjusted radius
                orientation_euler = p.getEulerFromQuaternion(orientation)
                initial_position, _ = p.getBasePositionAndOrientation(
                    object_id)

                gripper.set_position(position, orientation)
                time.sleep(0.5)
                gripper.close_gripper()
                time.sleep(0.5)
                gripper.move_up_smoothly(target_z=position[2] + 0.3)
                time.sleep(0.5)

                success, delta_z, final_position = evaluate_grasp(
                    object_id, initial_position)
                writer.writerow([
                    iteration * total_steps + step +
                    1, position[0], position[1], position[2],
                    orientation_euler[0], orientation_euler[1], orientation_euler[2],
                    initial_position[2], final_position[2], delta_z, success
                ])

                p.removeBody(object_id)
                object_id = create_cylinder_object(
                    height=object_height, radius=object_radius)
                gripper.open_gripper()
                time.sleep(0.5)

    p.disconnect()
    print("Data collection complete. Check 'grasp_data_cylinder.csv' for results.")


if __name__ == "__main__":
    main()
