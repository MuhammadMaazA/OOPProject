import pybullet as p
import time


class Gripper:
    # Initialize the gripper with its position and orientation
    def __init__(self, position, orientation):
        # Load the gripper URDF model at the given position and orientation
        self.gripper = p.loadURDF(
            "pr2_gripper.urdf",
            basePosition=position,
            baseOrientation=orientation,
            useFixedBase=False
        )
        # Make the gripper lightweight by setting its mass to 0
        p.changeDynamics(self.gripper, -1, mass=0)
        # Retrieve and store active and fixed joint information
        self.active_joints, self.fixed_joints = self.get_joint_info(self.gripper)

    # Get the joint info: active joints for movement, fixed joints for structure
    def get_joint_info(self, gripper):
        num_joints = p.getNumJoints(gripper)
        active_joints, fixed_joints = [], []
        for i in range(num_joints):
            # Info includes joint type, name, and other properties
            info = p.getJointInfo(gripper, i)
            if info[2] != p.JOINT_FIXED:  # If the joint can move
                active_joints.append(i)
            else:  # If the joint is static
                fixed_joints.append(i)
        return active_joints, fixed_joints

    # Simulate the physics engine for a given number of steps
    def sim_step(self, steps=500, delay=0.001):
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(delay)  # Slow down to make simulation visible

    # Open the gripper to its maximum position
    def open_gripper(self):
        open_positions = [0.548, 0.548]  # Define target positions for each joint
        for target_position, joint_index in zip(open_positions, self.active_joints):
            # Set motor control to position mode for each joint
            p.setJointMotorControl2(
                bodyIndex=self.gripper,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_position,
                force=100  # Moderate force for opening
            )
        self.sim_step()

    # Close the gripper to grasp an object securely
    def close_gripper(self):
        close_positions = [0.0, 0.0]  # Target closed positions
        for target_position, joint_index in zip(close_positions, self.active_joints):
            p.setJointMotorControl2(
                bodyIndex=self.gripper,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_position,
                force=500  # High force for a firm grip
            )
        # Enhance friction for better gripping
        for joint_index in self.active_joints:
            p.changeDynamics(self.gripper, joint_index, lateralFriction=1.0)
        self.sim_step()

    # Instantly set the gripper to a new position and orientation
    def set_position(self, position, orientation):
        p.resetBasePositionAndOrientation(self.gripper, position, orientation)

    # Smoothly move the gripper upwards to a specified height
    def move_up_smoothly(self, target_z, steps=200, delay=0.01):
        current_position, _ = p.getBasePositionAndOrientation(self.gripper)
        start_z = current_position[2]
        delta_z = (target_z - start_z) / steps
        velocity = delta_z / delay

        for _ in range(steps):
            # Apply upward velocity to the base of the gripper
            p.resetBaseVelocity(self.gripper, linearVelocity=[0, 0, velocity])
            p.stepSimulation()
            time.sleep(delay)
        # Stop the gripper after reaching the target
        p.resetBaseVelocity(self.gripper, linearVelocity=[0, 0, 0])
