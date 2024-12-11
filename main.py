import pybullet as p
import time
import pybullet_data
import math
import csv
import os
import random
from gripper import Gripper
from evaluate import evaluate_grasp

def create_custom_object(size=[0.05, 0.05, 0.8]):
    """Create a custom box-shaped object."""
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[s / 2 for s in size],
        rgbaColor=[1, 1, 1, 1]
    )
    
    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[s / 2 for s in size]
    )
    
    object_id = p.createMultiBody(
        baseMass=0.1,
        baseInertialFramePosition=[0, 0, 0],
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=[0, 0, size[2] / 2],
        useMaximalCoordinates=False
    )
    
    return object_id

def generate_pose(object_position, object_size, height, orientation_type, step, total_steps, radius):
    """Generate a gripper pose with improved variability."""
    obj_width, obj_depth, obj_height = object_size
    midplane_line = [object_position[0], object_position[1], object_position[2] + height]

    # Introduce randomness in the radius to explore different distances from the object
    radius_variability = random.uniform(-0.05, 0.05)  # Radius variation within ±0.05
    radius += radius_variability

    # Compute the angle for the current step
    angle = (2 * math.pi) * (step / total_steps)

    # Introduce variability in gripper positions along X, Y, and Z
    gripper_x = midplane_line[0] + radius * math.cos(angle)
    gripper_y = midplane_line[1] + radius * math.sin(angle) + random.uniform(-0.1, 0.1)  # Vary Y by ±0.1
    gripper_z = midplane_line[2] + random.uniform(-0.3, 0.3)  # Vary Z by ±0.3

    gripper_position = [gripper_x, gripper_y, gripper_z]

    # Compute direction vector from gripper to the midplane (pointing toward the cuboid)
    direction_vector = [
        midplane_line[0] - gripper_position[0],
        midplane_line[1] - gripper_position[1],
        midplane_line[2] - gripper_position[2],
    ]
    magnitude = math.sqrt(sum([i ** 2 for i in direction_vector]))
    direction_vector = [i / magnitude for i in direction_vector]

    # Introduce randomness in orientation angles for roll, pitch, and yaw
    roll = random.uniform(-0.7, 0.7)  # Roll variation between -0.7 and 0.7 radians (~ -40.11 to 40.11 degrees)
    pitch = random.uniform(-0.7, 0.7)  # Pitch variation between -0.7 and 0.7 radians
    yaw = math.atan2(direction_vector[1], direction_vector[0]) + random.uniform(-0.7, 0.7)  # Yaw with added randomness

    # Set orientation based on the specified type
    if orientation_type == "horizontal":
        orientation = p.getQuaternionFromEuler([roll, pitch, yaw])
    elif orientation_type == "vertical":
        orientation = p.getQuaternionFromEuler([roll, pitch + math.pi / 2, yaw])
    else:
        raise ValueError("Invalid orientation type")
    
    return gripper_position, orientation

def main():
    # Connect to PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    
    plane_id = p.loadURDF("plane.urdf")
    object_size = [0.05, 0.05, 0.8]
    object_position = [0, 0, object_size[2] / 2]
    object_id = create_custom_object(object_size)
    
    # Initialize gripper in an open state
    gripper = Gripper([0, 0, object_size[2] + 0.2], p.getQuaternionFromEuler([0, 0, 0]))
    gripper.open_gripper()  # Make sure the gripper is open by default
    total_steps = 100  # Increased to generate more data points per run
    iterations = 10  # Number of iterations to further increase the number of poses

    # Save data to the original file
    output_file = 'grasp_data.csv'
    file_exists = os.path.isfile(output_file)

    # Open the CSV file in append mode to keep track of all data
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header only if the file does not exist (to avoid duplicate headers)
        if not file_exists:
            writer.writerow([
                "Step", "Position X", "Position Y", "Position Z",
                "Orientation Roll", "Orientation Pitch", "Orientation Yaw",
                "Initial Z", "Final Z", "Delta Z", "Success"
            ])

        # Run the simulation multiple times to generate more data points
        for iteration in range(iterations):
            for step in range(total_steps):
                # Reset the object position
                p.resetBasePositionAndOrientation(object_id, object_position, [0, 0, 0, 1])
                
                # Generate pose for the gripper
                position, orientation = generate_pose(object_position, object_size, 0, "horizontal", step, total_steps, 0.25)
                orientation_euler = p.getEulerFromQuaternion(orientation)
                initial_position, _ = p.getBasePositionAndOrientation(object_id)

                # Set the gripper to the desired position and orientation
                gripper.set_position(position, orientation)
                time.sleep(0.5)

                # Close the gripper after positioning
                gripper.close_gripper()
                time.sleep(0.5)

                # Move the gripper up to evaluate the grasp
                gripper.move_up_smoothly(target_z=position[2] + 0.3)
                time.sleep(0.5)

                # Evaluate grasp success
                success, delta_z, final_position = evaluate_grasp(object_id, initial_position)

                # Write the data to the CSV file
                writer.writerow([
                    iteration * total_steps + step + 1, position[0], position[1], position[2],
                    orientation_euler[0], orientation_euler[1], orientation_euler[2],
                    initial_position[2], final_position[2], delta_z, success
                ])

                # Reset for the next iteration
                p.removeBody(object_id)
                object_id = create_custom_object(object_size)
                gripper.open_gripper()  # Ensure the gripper is open for the next iteration
                time.sleep(0.5)

    # Disconnect simulation
    p.disconnect()

if __name__ == "__main__":
    main()