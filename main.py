import pybullet as p
import time
import pybullet_data
import math
from gripper import Gripper
from datacollectionandlabel import collect_data, label_data, view_collected_data
import random
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
    """
    Generate pose for the gripper incrementally around the cuboid, ensuring it points toward the midplane.
    """
    obj_width, obj_depth, obj_height = object_size

    # Midplane Line: Passing through the cuboid's center at the specified height
    midplane_line = [object_position[0], object_position[1], object_position[2] + height]

    # Define the radius at which the gripper will be placed
    radius = 0.3  # Adjust as needed to ensure proper distance from the cuboid

    # Compute the angle for the current step (from 0 to 2*pi)
    angle = (2 * math.pi) * (step / total_steps)

    # Calculate gripper position in x and y using the angle
    gripper_x = midplane_line[0] + radius * math.cos(angle)
    gripper_y = midplane_line[1] + radius * math.sin(angle)
    gripper_z = midplane_line[2]  # Keep z constant

    gripper_position = [gripper_x, gripper_y, gripper_z]

    # Compute direction vector from gripper to the midplane (pointing toward the cuboid)
    direction_vector = [
    midplane_line[0] - gripper_position[0],
    midplane_line[1] - gripper_position[1],
    midplane_line[2] - gripper_position[2],
    ]

    # Normalize the direction vector
    magnitude = math.sqrt(sum([i ** 2 for i in direction_vector]))
    direction_vector = [i / magnitude for i in direction_vector]

    # Convert direction vector to Euler angles
    yaw = math.atan2(direction_vector[1], direction_vector[0])  # Rotation around z-axis
    pitch = math.asin(-direction_vector[2])                     # Rotation around y-axis
    roll = 0  # No roll for this case

    # Set orientation based on the specified type
    if orientation_type == "horizontal":
         orientation = p.getQuaternionFromEuler([roll, pitch, yaw])
    elif orientation_type == "vertical":
         orientation = p.getQuaternionFromEuler([roll, pitch + math.pi / 2, yaw])
    else:
        raise ValueError("Invalid orientation type")
    
    return gripper_position, orientation
'''
def generate_pose(object_position, object_size, height, orientation_type, step, total_steps):
    """
    Generate pose for the gripper incrementally around the cuboid, ensuring it rotates about the x-axis
    and points toward the midplane, with dynamic constraints based on the angle.
    """
    obj_width, obj_depth, obj_height = object_size

    # Midplane Line: Passing through the cuboid's center at the specified height
    midplane_line = [object_position[0], object_position[1], object_position[2] + (obj_height / 2)]

    # Compute the angle for the current step
    angle = (2 * math.pi) * (step / total_steps)  # From 0 to 2*pi (full rotation)

    # Adjust x constraint dynamically based on the angle
    base_x_offset = 0.3  # Base distance from the cuboid
    x_offset = base_x_offset * abs(math.cos(angle))  # Scale by cos(angle) for proximity
    gripper_x = midplane_line[0] - x_offset  # X-position of the gripper

    # Gripper z-position: dynamic adjustment to stay within cuboid height range
    min_z = object_position[2] + 0.1  # Slightly above the bottom of the cuboid
    max_z = object_position[2] + obj_height - 0.1  # Slightly below the top of the cuboid
    gripper_z = min_z + (max_z - min_z) * (step / total_steps)  # Linearly vary z within range

    # Gripper y-position remains fixed (aligned with the midplane)
    gripper_y = midplane_line[1]

    gripper_position = [gripper_x, gripper_y, gripper_z]
     # Compute direction vector from gripper to the midplane (pointing toward the cuboid)
    direction_vector = [
        midplane_line[0] - gripper_position[0],
        midplane_line[1] - gripper_position[1],
        midplane_line[2] - gripper_position[2],
    ]

    # Normalize the direction vector
    magnitude = math.sqrt(sum([i ** 2 for i in direction_vector]))
    direction_vector = [i / magnitude for i in direction_vector]

    # Convert direction vector to Euler angles
    yaw = math.atan2(direction_vector[1], direction_vector[0])  # Rotation around z-axis
    pitch = math.asin(-direction_vector[2])  # Rotation around y-axis
    roll = 0  # No roll for this case

    # Set orientation based on the specified type
    if orientation_type == "horizontal":
        orientation = p.getQuaternionFromEuler([roll, pitch, yaw])
    elif orientation_type == "vertical":
        orientation = p.getQuaternionFromEuler([roll, pitch + math.pi / 2, yaw])
    else:
        raise ValueError("Invalid orientation type")

    # Debugging output to verify the position and orientation
    print(f"Step {step}/{total_steps} -> Position: {gripper_position}, Orientation (Euler): [roll={roll}, pitch={pitch}, yaw={yaw}]")

    return gripper_position, orientation
'''

def check_grasp_success(object_id, initial_height):
    """Check if the object has been lifted."""
    time.sleep(0.5)
    object_pos, _ = p.getBasePositionAndOrientation(object_id)
    return object_pos[2] > initial_height + 0.1

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
    
    # Initialize the gripper
    gripper = Gripper([0, 0, object_size[2] + 0.2], p.getQuaternionFromEuler([0, 0, 0]))
    
    # Total steps for iterations
    total_steps = 20

    # Horizontal orientation tests
    for step in range(total_steps):
        # Reset object position and orientation before each test
        p.resetBasePositionAndOrientation(object_id, object_position, [0, 0, 0, 1])

        # Generate pose for the gripper
        position, orientation = generate_pose(object_position, object_size, 0, "horizontal", step, total_steps)

        # Convert orientation (quaternion) to Euler angles for logging
        orientation_euler = p.getEulerFromQuaternion(orientation)

        # Log the 6DOF
        print(f"Step {step+1}/{total_steps}: Position: {position}, Orientation (Euler): {orientation_euler}")

        # Record the initial position of the object
        initial_position, _ = p.getBasePositionAndOrientation(object_id)

        # Perform actions with the gripper
        gripper.open_gripper()
        gripper.set_position(position, orientation)
        time.sleep(1)
        gripper.close_gripper()
        time.sleep(0.5)
        gripper.move_up_smoothly(target_z=position[2] + 0.3)
        time.sleep(0.5)

        # Evaluate the grasp and log results
        success, delta_z, final_position = evaluate_grasp(object_id, initial_position)
        print(f"Step {step+1}/{total_steps}: Success: {success}, Δz: {delta_z:.3f}, Final Position: {final_position}")

        # Reset for the next step
        p.removeBody(object_id)
        object_id = create_custom_object(object_size)
        time.sleep(1)

    # Disconnect simulation
    p.disconnect()

if __name__ == "__main__":
    main()
