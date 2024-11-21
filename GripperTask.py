import numpy as np
import pybullet as p

class GripperTask:
    def __init__(self, robot):
        self.robot = robot

    def generate_random_pose(self):
        # Generate a random pose within a reasonable range around the object
        random_x = np.random.uniform(-0.2, 0.2) + self.robot.start_pos[0]
        random_y = np.random.uniform(-0.2, 0.2) + self.robot.start_pos[1]
        random_z = np.random.uniform(0.1, 0.3) + self.robot.start_pos[2]
        random_orientation = p.getQuaternionFromEuler([0, 0, np.random.uniform(0, 2 * np.pi)])
        
        # Move the gripper to the random pose
        p.resetBasePositionAndOrientation(self.robot.robot_model, [random_x, random_y, random_z], random_orientation)
        print(f"[INFO] Gripper moved to random pose: {[random_x, random_y, random_z]}")

    def close_gripper(self):
        # Close the gripper
        self.robot.gripper.grip()
        print("[INFO] Gripper closed")

    def lift_gripper(self):
        # Lift the gripper in a random upward direction
        current_pos, current_orientation = p.getBasePositionAndOrientation(self.robot.robot_model)
        random_lift_z = np.random.uniform(0.1, 0.3)
        new_pos = [current_pos[0], current_pos[1], current_pos[2] + random_lift_z]
        
        # Move the gripper to the new lifted position
        p.resetBasePositionAndOrientation(self.robot.robot_model, new_pos, current_orientation)
        print(f"[INFO] Gripper lifted to position: {new_pos}")

    def execute(self):
        print("[INFO] Executing Gripper Task...")
        self.generate_random_pose()
        self.close_gripper()
        self.lift_gripper()
