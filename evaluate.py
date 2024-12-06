import pybullet as p
import time

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
    # Determine outcome
    if delta_z > 0.1:
        success = 1  # Success
    elif 0.05 <= delta_z <= 0.1:
        success = 2  # Almost
    else:
        success = 0  # Failure

    return success, delta_z,final_position