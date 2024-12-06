import pybullet as p
import csv
import os
import math
import time
import pybullet_data
from gripper import Gripper
from evaluate import evaluate_grasp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load or initialize machine learning model
MODEL_FILENAME = "grasp_model.pkl"
if os.path.exists(MODEL_FILENAME):
    model = joblib.load(MODEL_FILENAME)
else:
    model = RandomForestClassifier()

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
"""
def generate_pose(object_position, object_size, height, orientation_type, step, total_steps):

    Generate pose for the gripper incrementally around the cuboid,
    ensuring it points toward the midplane and the fingers are properly aligned.

    obj_width, obj_depth, obj_height = object_size

    # Midplane Line: Passing through the cuboid's center at the specified height
    midplane_line = [
        object_position[0],
        object_position[1],
        object_position[2] + height
    ]

    # Adjust radius to bring gripper closer to the object
    radius = obj_width / 2 + 0.02  # Ensure gripper is slightly away from the object surface

    # Compute the angle for the current step (from 0 to 2*pi)
    angle = (2 * math.pi) * (step / total_steps)

    # Calculate gripper position in x and y using the angle
    gripper_x = midplane_line[0] + radius * math.cos(angle)
    gripper_y = midplane_line[1] + radius * math.sin(angle)
    gripper_z = midplane_line[2]  # Align with the midplane height

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
    pitch = math.asin(direction_vector[2])  # Rotation around y-axis

    # Set orientation based on the specified type
    if orientation_type == "horizontal":
        # For horizontal, roll is 0
        roll = 0
        orientation = p.getQuaternionFromEuler([roll, pitch, yaw])
    elif orientation_type == "vertical":
        # For vertical, adjust roll to rotate gripper fingers vertically
        roll = math.pi / 2
        orientation = p.getQuaternionFromEuler([roll, pitch, yaw])
    else:
        raise ValueError("Invalid orientation type")

    return gripper_position, orientation

def write_to_csv(filename, data, headers):

    Write the given data to a CSV file.

    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)
"""
def train_model(filename):
    """
    Train a machine learning model to predict grasp success.
    """
    data = pd.read_csv(filename)
    X = data[['Gripper X', 'Gripper Y', 'Gripper Z', 'Gripper Roll', 'Gripper Pitch', 'Gripper Yaw']]
    y = data['Success']
    y = y.astype(int)  # Ensure the success labels are integers
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model training completed. Accuracy: {accuracy * 100:.2f}%")
    
    # Save the trained model
    joblib.dump(model, MODEL_FILENAME)

def predict_success(position, orientation_euler):
    """
    Predict the success of a grasp using the trained model.
    """
    features = np.array([
        [
            position[0], position[1], position[2],
            math.degrees(orientation_euler[0]),
            math.degrees(orientation_euler[1]),
            math.degrees(orientation_euler[2])
        ]
    ])
    if hasattr(model, "predict"):
        return model.predict(features)[0]
    else:
        return False

def main():
    # Connect to PyBullet
    p.connect(p.DIRECT)  # Use DIRECT to reduce resource usage for overnight training
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)

    plane_id = p.loadURDF("plane.urdf")
    object_size = [0.05, 0.05, 0.8]
    object_position = [0, 0, object_size[2] / 2]
    object_id = create_custom_object(object_size)

    # Initialize the gripper
    gripper = Gripper(
        [0, 0, object_size[2] + 0.2],
        p.getQuaternionFromEuler([0, 0, 0])
    )

    # Total steps for iterations
    total_steps = 10
    heights = [
        object_size[2] / 2,  # Middle of the object
    ]
    orientation_types = ["horizontal", "vertical"]

    headers = [
        'Gripper X', 'Gripper Y', 'Gripper Z',
        'Gripper Roll', 'Gripper Pitch', 'Gripper Yaw',
        'Object X', 'Object Y', 'Object Z',
        'Success'
    ]
    filename = 'updated_grasp_data.csv'

    # Train the model if there's existing data
    if os.path.exists(filename):
        train_model(filename)

    for height in heights:
        for orientation_type in orientation_types:
            for step in range(total_steps):
                # Reset object position and orientation before each test
                p.resetBasePositionAndOrientation(
                    object_id, object_position, [0, 0, 0, 1]
                )

                # Generate pose for the gripper
                position, orientation = generate_pose(
                    object_position, object_size, height, orientation_type, step, total_steps
                )

                # Convert orientation (quaternion) to Euler angles for logging
                orientation_euler = p.getEulerFromQuaternion(orientation)

                # Predict success using the current model
                predicted_success = predict_success(position, orientation_euler)

                # Log the 6DOF
                print(f"Orientation: {orientation_type}, "
                      f"Step {step+1}/{total_steps}: Position: {position}, "
                      f"Orientation (Euler): {orientation_euler}, "
                      f"Predicted Success: {predicted_success}")

                # Record the initial position of the object
                initial_position, _ = p.getBasePositionAndOrientation(object_id)

                # Perform actions with the gripper
                gripper.open_gripper()
                gripper.set_position(position, orientation)
                time.sleep(0.1)
                gripper.close_gripper()
                time.sleep(0.1)
                gripper.move_up_smoothly(target_z=position[2] + 0.3)
                time.sleep(0.1)

                # Evaluate the grasp and log results
                success, delta_z, final_position = evaluate_grasp(
                    object_id, initial_position
                )
                print(f"Success: {success}, Δz: {delta_z:.3f}, Final Position: {final_position}")

                # Collect data and write to CSV
                data = {
                    'Gripper X': position[0],
                    'Gripper Y': position[1],
                    'Gripper Z': position[2],
                    'Gripper Roll': math.degrees(orientation_euler[0]),
                    'Gripper Pitch': math.degrees(orientation_euler[1]),
                    'Gripper Yaw': math.degrees(orientation_euler[2]),
                    'Object X': initial_position[0],
                    'Object Y': initial_position[1],
                    'Object Z': initial_position[2],
                    'Success': success
                }
                write_to_csv(filename, data, headers)
                time.sleep(0.1)

                # Retrain model periodically
                if (step + 1) % 5 == 0:
                    train_model(filename)

    # Disconnect simulation
    p.disconnect()

if __name__ == "__main__":
    main()
