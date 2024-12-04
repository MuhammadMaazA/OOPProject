import pybullet as p
import time
import pybullet_data
import math
import csv
import os
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

def generate_pose(object_position, object_size, height, orientation_type, step, total_steps):
    obj_width, obj_depth, obj_height = object_size
    midplane_line = [object_position[0], object_position[1], object_position[2] + height]
    radius = 0.25
    angle = (2 * math.pi) * (step / total_steps)
    gripper_x = midplane_line[0] + radius * math.cos(angle)
    gripper_y = midplane_line[1] + radius * math.sin(angle)
    gripper_z = midplane_line[2]
    gripper_position = [gripper_x, gripper_y, gripper_z]
    direction_vector = [
        midplane_line[0] - gripper_position[0],
        midplane_line[1] - gripper_position[1],
        midplane_line[2] - gripper_position[2],
    ]
    magnitude = math.sqrt(sum([i ** 2 for i in direction_vector]))
    direction_vector = [i / magnitude for i in direction_vector]
    yaw = math.atan2(direction_vector[1], direction_vector[0])
    pitch = math.asin(-direction_vector[2])
    roll = 0

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
    
    gripper = Gripper([0, 0, object_size[2] + 0.2], p.getQuaternionFromEuler([0, 0, 0]))
    total_steps = 20

    # Check if CSV file exists
    file_exists = os.path.isfile('grasp_data.csv')

    # Open the CSV file in append mode to keep track of all data
    with open('grasp_data.csv', mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write the header only if the file does not exist (to avoid duplicate headers)
        if not file_exists:
            writer.writerow([
                "Step", "Position X", "Position Y", "Position Z",
                "Orientation Roll", "Orientation Pitch", "Orientation Yaw",
                "Initial Z", "Final Z", "Delta Z", "Success"
            ])

        # Horizontal orientation tests
        for step in range(total_steps):
            p.resetBasePositionAndOrientation(object_id, object_position, [0, 0, 0, 1])
            position, orientation = generate_pose(object_position, object_size, 0, "horizontal", step, total_steps)
            orientation_euler = p.getEulerFromQuaternion(orientation)
            initial_position, _ = p.getBasePositionAndOrientation(object_id)

            gripper.open_gripper()
            gripper.set_position(position, orientation)
            time.sleep(1)
            gripper.close_gripper()
            time.sleep(0.5)
            gripper.move_up_smoothly(target_z=position[2] + 0.3)
            time.sleep(0.5)

            success, delta_z, final_position = evaluate_grasp(object_id, initial_position)

            # Write the data to the CSV file
            writer.writerow([
                step + 1, position[0], position[1], position[2],
                orientation_euler[0], orientation_euler[1], orientation_euler[2],
                initial_position[2], final_position[2], delta_z, success
            ])

            p.removeBody(object_id)
            object_id = create_custom_object(object_size)
            time.sleep(1)

    # Disconnect simulation
    p.disconnect()

if __name__ == "__main__":
    main()
