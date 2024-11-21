import pybullet as p

# Data structure to store collected data for testing
collected_data = []

def collect_data(gripper, cuboid_id):
    # Get initial and final positions of the object
    initial_pos, _ = p.getBasePositionAndOrientation(cuboid_id)
    final_pos, _ = p.getBasePositionAndOrientation(cuboid_id)

    deviation_x = abs(final_pos[0] - initial_pos[0])
    deviation_y = abs(final_pos[1] - initial_pos[1])

    # Calculate average torque across all active joints
    torque = sum(p.getJointState(gripper.gripper, joint_index)[3] for joint_index in gripper.active_joints) / len(gripper.active_joints)

    # Check for slippage
    slippage_detected = deviation_x > 0.1 or deviation_y > 0.1

    # Collect the features in a dictionary
    data_point = {
        'deviation_x': deviation_x,
        'deviation_y': deviation_y,
        'torque': torque,
        'slippage_detected': slippage_detected
    }

    # Append to the global collected_data list
    collected_data.append(data_point)

    # Return the data point for immediate inspection
    return data_point

def label_data():
    labeled_data = []

    # Define thresholds for good vs. bad pose
    position_threshold = 0.05  # Example threshold for position deviation
    torque_threshold = 0.5     # Example torque threshold

    for data_point in collected_data:
        label = 0  # Default to "bad pose"

        # Check conditions for a good pose
        if abs(data_point['deviation_x']) < position_threshold and abs(data_point['deviation_y']) < position_threshold:
            if data_point['torque'] >= torque_threshold and not data_point['slippage_detected']:
                label = 1  # Good pose

        # Add the label to the data point
        data_point['is_good_pose'] = label
        labeled_data.append(data_point)

    # Return the labeled data
    return labeled_data

# Function to view the collected data
def view_collected_data():
    for index, data_point in enumerate(collected_data):
        print(f"Data Point {index + 1}: {data_point}")

# Function to save the data to a CSV later, if needed
def save_data_to_csv(filename="grip_data.csv"):
    import pandas as pd
    # Convert the collected data to a DataFrame
    df = pd.DataFrame(collected_data)
    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
