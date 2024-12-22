import pybullet as p
import time
import os
import csv

class GripperEvaluator:
    def __init__(self, csv_filename="gripper_data_old_format.csv"):
        """
        Initialize the GripperEvaluator for the 'old format' columns:
        Position X, Position Y, Position Z,
        Orientation Roll, Orientation Pitch, Orientation Yaw,
        Initial Z, Final Z, Delta Z, Success
        """
        self.csv_filename = csv_filename
        self.headers = [
            "Position X", "Position Y", "Position Z",
            "Orientation Roll", "Orientation Pitch", "Orientation Yaw",
            "Initial Z", "Final Z", "Delta Z",
            "Success"
        ]

        if not os.path.isfile(self.csv_filename):
            with open(self.csv_filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(self.headers)

    def evaluate_grasp(self, object_id, initial_position):
        """
        Evaluate the success of a grasp based on delta_z logic.
        Returns (success_code, delta_z, final_position).
          success_code = 1 if delta_z > 0.1
                         2 if 0.05 <= delta_z <= 0.1
                         0 otherwise
        """
        time.sleep(0.5)
        if not p.isConnected():
            return 0, 0.0, initial_position

        final_position, _ = p.getBasePositionAndOrientation(object_id)

        initial_z = initial_position[2]
        final_z = final_position[2]
        delta_z = final_z - initial_z

        if delta_z > 0.1:
            success = 1   # Fully successful
        elif 0.05 <= delta_z <= 0.1:
            success = 2   # Almost
        else:
            success = 0   # Failure

        return success, delta_z, final_position

    def save_to_csv(self, data):
        """
        Append a single row to CSV if unique.
        data should be in the format:
        [Position X, Position Y, Position Z,
         Orientation Roll, Orientation Pitch, Orientation Yaw,
         Initial Z, Final Z, Delta Z, Success]
        """
        existing_data = set()
        if os.path.isfile(self.csv_filename):
            with open(self.csv_filename, mode="r") as file:
                reader = csv.reader(file)
                next(reader, None)  # Skip header
                for row in reader:
                    existing_data.add(tuple(row))

        data_tuple = tuple(map(str, data))
        if data_tuple not in existing_data:
            with open(self.csv_filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(data)
        else:
            print("Duplicate data row detected. Skipping.")
