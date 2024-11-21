import pybullet as p
import time

class Gripper:
    def __init__(self, position, orientation):
        self.gripper = p.loadURDF(
            "pr2_gripper.urdf",
            basePosition=position,
            baseOrientation=orientation,
            useFixedBase=False
        )
        p.changeDynamics(self.gripper, -1, mass=0)
        self.active_joints, self.fixed_joints = self.get_joint_info(self.gripper)
    
    def get_joint_info(self, gripper):
        num_joints = p.getNumJoints(gripper)
        active_joints, fixed_joints = [], []
        for i in range(num_joints):
            info = p.getJointInfo(gripper, i)
            if info[2] != p.JOINT_FIXED:
                active_joints.append(i)
            else:
                fixed_joints.append(i)
        return (active_joints, fixed_joints)
    
    def sim_step(self, steps=500, delay=0.001):
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(delay)
    
    def open_gripper(self):
        open_positions = [0.548, 0.548]
        for target_position, joint_index in zip(open_positions, self.active_joints):
            p.setJointMotorControl2(
                bodyIndex=self.gripper,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_position,
                force=100
            )
        self.sim_step()
    
    def close_gripper(self):
        close_positions = [0.0, 0.0]
        for target_position, joint_index in zip(close_positions, self.active_joints):
            p.setJointMotorControl2(
                bodyIndex=self.gripper,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_position,
                force=500  # Increased force
            )
        # Set friction for better grip
        for joint_index in self.active_joints:
            p.changeDynamics(self.gripper, joint_index, lateralFriction=1.0)
        
        self.sim_step()

    def set_position(self, position, orientation):
        p.resetBasePositionAndOrientation(self.gripper, position, orientation)

    def move_up_smoothly(self, target_z, steps=200, delay=0.01):
        current_position, current_orientation = p.getBasePositionAndOrientation(self.gripper)
        start_z = current_position[2]
        delta_z = (target_z - start_z) / steps
        velocity = delta_z / delay

        for _ in range(steps):
            p.resetBaseVelocity(self.gripper, linearVelocity=[0, 0, velocity])
            p.stepSimulation()
            time.sleep(delay)
        p.resetBaseVelocity(self.gripper, linearVelocity=[0, 0, 0])
