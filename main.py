import pybullet as p
import time
import pybullet_data
import math
from gripper import Gripper
from datacollectionandlabel import collect_data, label_data, view_collected_data

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

def generate_pose(object_position, object_size, height, orientation_type, offset=0):
    """Generate pose for the gripper based on specified parameters."""
    obj_width, obj_depth, obj_height = object_size
    
    if orientation_type == "horizontal":
        # Horizontal orientation
        position = [
            object_position[0] - 0.2 - 0.105,  # Adjust to ensure gripper fingers make contact
            object_position[1],
            object_position[2] + height
        ]
        orientation = p.getQuaternionFromEuler([0, 0, 0])
    elif orientation_type == "vertical":
        # Vertical orientation
        position = [
            object_position[0],
            object_position[1],
            object_position[2] + obj_height + height + offset
        ]
        orientation = p.getQuaternionFromEuler([0, math.pi/2, 0])
    else:
        raise ValueError("Invalid orientation type")
    
    return position, orientation

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
    
    collected_data_points = []
    
    # Horizontal orientation tests
    for height in [h / 10 for h in range(0, int(object_size[2] * 10), 1)]:
        # Reset object position and orientation before each test
        p.resetBasePositionAndOrientation(object_id, object_position, [0, 0, 0, 1])
        
        position, orientation = generate_pose(object_position, object_size, height, "horizontal")
        
        gripper.open_gripper()
        gripper.set_position(position, orientation)
        
        for _ in range(100):
            p.stepSimulation()
        time.sleep(0.5)
        
        gripper.close_gripper()
        for _ in range(100):
            p.stepSimulation()
        time.sleep(0.5)
        
        # Use the move_up_smoothly function to lift the object
        lift_height = position[2] + 0.4
        gripper.move_up_smoothly(lift_height)
        
        for _ in range(100):
            p.stepSimulation()
        time.sleep(0.5)
        
        # Check if the object was successfully lifted
        initial_pos, _ = p.getBasePositionAndOrientation(object_id)
        success = check_grasp_success(object_id, initial_pos[2])
        print(f"Horizontal grasp at height {height:.2f}: {'Success' if success else 'Failure'}")
        
        # Collect data
        data = collect_data(gripper, object_id)
        data['success'] = success
        collected_data_points.append(data)
        
        time.sleep(1)
    
    # Vertical orientation tests
    gripper_height = 0.2  # Approximate height of the gripper
    for offset in [o / 50 for o in range(0, 10)]:  # Offsets from 0 to 0.18 in steps of 0.02
        # Reset object position and orientation before each test
        p.resetBasePositionAndOrientation(object_id, object_position, [0, 0, 0, 1])
        
        position, orientation = generate_pose(object_position, object_size, gripper_height, "vertical", offset)
        
        gripper.open_gripper()
        gripper.set_position(position, orientation)
        
        for _ in range(100):
            p.stepSimulation()
        time.sleep(0.5)
        
        gripper.close_gripper()
        for _ in range(100):
            p.stepSimulation()
        time.sleep(0.5)
        
        # Use the move_up_smoothly function to lift the object
        lift_height = position[2] + 0.4
        gripper.move_up_smoothly(lift_height)
        
        for _ in range(100):
            p.stepSimulation()
        time.sleep(0.5)
        
        # Check if the object was successfully lifted
        initial_pos, _ = p.getBasePositionAndOrientation(object_id)
        success = check_grasp_success(object_id, initial_pos[2])
        print(f"Vertical grasp with offset {offset:.2f}: {'Success' if success else 'Failure'}")
        
        # Collect data
        data = collect_data(gripper, object_id)
        data['success'] = success
        collected_data_points.append(data)
        
        time.sleep(1)
    
    # Label and view the collected data
    label_data(collected_data_points)
    view_collected_data()
    
    p.disconnect()

if __name__ == "__main__":
    main()
