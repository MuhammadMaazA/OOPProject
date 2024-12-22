import pybullet as p
import pybullet_data
import os
import numpy as np
import sys
import pandas as pd
sys.path.insert(1, 'utils/')
from simulation import sim
from Robot import Robot
from thing import Thing
from pointCloud import getPointCloud
from IPython import embed 
from Gripper import gripper

class robotSimulator:
    def __init__(self):
        initial_position = [0, 0, 0.7]
        initial_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.gripper = gripper(initial_position, initial_orientation)
        
        # Example usage
        self.gripper.open_gripper()
        self.gripper.move_up_smoothly(1.0)
        self.gripper.close_gripper()

    def loop(self):
        while(True):
            self.readResetButton()
            self.readToTargetButton()
            self.readGripButton()
            self.readOpenGripperButton()
            self.readTableButton()
            self.readObjectButton()
            self.readPlanPathButton()
            self.readFollowPathButton()
            self.readCameraButton()
            self.readSavePoseButton()
            self.readFollowedSavedPathButton()
            self.readPointCloudButton()
            self.readJointControlButton()
            self.readmultiGraspButton()
            self.readRandomPoseButton()  # New random pose button
            self.readCloseGripperButton()  # New close gripper button
            self.readMoveUpButton()  # New move up button
            if self.sim.robotParams["camera"]:
                self.robot.simStep()

    def initButtonVals(self):
        self.startButtonVal = 2.0
        self.gripButtonVal = 2.0
        self.openGripperButtonVal = 2.0
        self.resetButtonVal = 2.0
        self.tableButtonVal = 2.0
        self.planButtonVal = 2.0
        self.pathButtonVal = 2.0
        self.cameraButtonVal = 2.0
        self.objButtonVal = 2.0
        self.savePoseButtonVal = 2.0
        self.followSavedPathButtonVal = 2.0
        self.pointCloudButtonVal=2.0
        self.jointControlButtonVal = 2.0
        self.prevVal = 0
        self.multiGraspButtonVal = 2.0

        self.randomPoseButtonVal = 2.0
        self.closeGripperButtonVal = 2.0
        self.moveUpButtonVal = 2.0
    def readResetButton(self):
        if p.readUserDebugParameter(self.sim.resetButton) >= self.resetButtonVal:
            self.robot.setJointPosition(np.zeros((self.robot.numArmJoints+self.robot.numGripperJoints)))
            if self.tableButtonVal >2.0:
                p.resetBasePositionAndOrientation(self.robot.robot_model,[0,0,self.table.height], self.robot.start_orientation)
            else:
                p.resetBasePositionAndOrientation(self.robot.robot_model, self.robot.start_pos, self.robot.start_orientation)
            self.resetButtonVal = p.readUserDebugParameter(self.sim.resetButton) + 1.0

    def readToTargetButton(self):
        if p.readUserDebugParameter(self.sim.startButton) >= self.startButtonVal:
            #Read data
            data = self.sim.motionParams["pose_target"]

            data = self.robot.createDQObject(data)
        
            self.robot.moveArmToEETarget(data,0.6)
            self.startButtonVal = p.readUserDebugParameter(self.sim.startButton) + 1.0
            
    def readGripButton(self):
        if p.readUserDebugParameter(self.sim.gripButton) >= self.gripButtonVal:
            self.robot.gripper.grip()
            self.gripButtonVal = p.readUserDebugParameter(self.sim.gripButton) + 1.0

    def readOpenGripperButton(self): 
        if p.readUserDebugParameter(self.sim.openGripperButton) >= self.openGripperButtonVal:
            self.robot.gripper.openGripper()
            self.openGripperButtonVal = p.readUserDebugParameter(self.sim.openGripperButton) + 1.0

    def readTableButton(self):
        if p.readUserDebugParameter(self.sim.tableButton) >= self.tableButtonVal:
            self.table = Thing("table")
            p.resetBasePositionAndOrientation(self.robot.robot_model, [0,0,self.table.height], self.robot.start_orientation)
            self.tableButtonVal = p.readUserDebugParameter(self.sim.tableButton) + 1.0

    def readObjectButton(self):
        if p.readUserDebugParameter(self.sim.objButton) >= self.objButtonVal:
            if self.tableButtonVal > 2.0:
                self.container = Thing(self.sim.simParams["object_name"],onObject=self.table)
            else: 
                self.container = Thing(self.sim.simParams["object_name"],self.sim.simParams["object_position"])
            self.objButtonVal = p.readUserDebugParameter(self.sim.objButton) + 1.0

    def readPlanPathButton(self):
        if p.readUserDebugParameter(self.sim.planButton) >= self.planButtonVal:
            #TODO: add path planning
            print("[INFO]: not yet implemented")
            self.planButtonVal = p.readUserDebugParameter(self.sim.planButton) + 1.0

    def readFollowPathButton(self):
        if p.readUserDebugParameter(self.sim.pathButton) >= self.pathButtonVal:
            #TODO: add follow path
            print("[INFO]: not yet implemented")
            self.pathButtonVal = p.readUserDebugParameter(self.sim.pathButton) + 1.0

    def readCameraButton(self):
        if p.readUserDebugParameter(self.sim.cameraButton) >= self.cameraButtonVal:
            if hasattr(self,"container"):
                self.sim.getPicture(self.container.getPos())
            else:
                self.sim.getPicture()

            self.cameraButtonVal = p.readUserDebugParameter(self.sim.cameraButton) + 1.0

    def readSavePoseButton(self):
        if p.readUserDebugParameter(self.sim.savePoseButton) >= self.savePoseButtonVal:
            currPose = self.robot.getPose()
            self.pose.append(currPose)
            print("Saved pose: ",currPose)
            self.savePoseButtonVal = p.readUserDebugParameter(self.sim.savePoseButton) + 1.0
    
    def readFollowedSavedPathButton(self):
        if p.readUserDebugParameter(self.sim.followSavedPathButton) >= self.followSavedPathButtonVal:
            self.robot.followPath(self.pose)
            self.followSavedPathButtonVal = p.readUserDebugParameter(self.sim.followSavedPathButton) + 1.0

    def readPointCloudButton(self):
        if p.readUserDebugParameter(self.sim.pointCloudButton) >= self.pointCloudButtonVal:
            if hasattr(self,"container"):
                self.sim.getPointCloud(self.container.getPos(), self.robot.getPandO())
            else:
                self.sim.getPointCloud()
            self.pointCloudButtonVal = p.readUserDebugParameter(self.sim.pointCloudButton) + 1.0

    def readJointControlButton(self):
        if p.readUserDebugParameter(self.sim.jointControlButton) == self.jointControlButtonVal:
            self.sim.addjointControl(self.robot.numArmJoints,self.robot.getJointPosition())
            self.jointControlButtonVal = 1
            self.prevVal = self.sim.getJointControlVal()
        
        if self.jointControlButtonVal == 1:
            targetPos = self.sim.getJointControlVal()
            if self.prevVal != targetPos:
                self.robot.setJointPosition(targetPos)
                self.prevVal = self.sim.getJointControlVal()
    
    def readmultiGraspButton(self):
        if p.readUserDebugParameter(self.sim.multiGraspButton) >= self.multiGraspButtonVal:
            self.robot.multiGraspSimulation()
            self.multiGraspButtonVal = p.readUserDebugParameter(self.sim.multiGraspButton) + 1.0

    def readCloseGripperButton(self):
        if p.readUserDebugParameter(self.sim.closeGripperButton) >= self.closeGripperButtonVal:
            # Use the close_gripper() method from your Gripper class
            self.gripper.close_gripper()
            print("Gripper closed.")
            self.closeGripperButtonVal = p.readUserDebugParameter(self.sim.closeGripperButton) + 1.0

    def readRandomPoseButton(self):
        # Implement the functionality for the random pose button
        if p.readUserDebugParameter(self.sim.randomPoseButton) >= self.randomPoseButtonVal:
            # Example: Set the robot joints to a random position
            random_pose = np.random.uniform(low=-np.pi, high=np.pi, size=self.robot.numGripperJoints)
            self.robot.setJointPosition(random_pose)
            print("Random pose set:", random_pose)
            self.randomPoseButtonVal = p.readUserDebugParameter(self.sim.randomPoseButton) + 1.0

    def readOpenGripperButton(self):
        if p.readUserDebugParameter(self.sim.openGripperButton) >= self.openGripperButtonVal:
            # Use the open_gripper() method from your Gripper class
            self.gripper.open_gripper()
            print("Gripper opened.")
            self.openGripperButtonVal = p.readUserDebugParameter(self.sim.openGripperButton) + 1.0

    def readMoveUpButton(self):
        if p.readUserDebugParameter(self.sim.moveUpButton) >= self.moveUpButtonVal:
            # Use the move_up_smoothly() method from your Gripper class
            self.gripper.move_up_smoothly(1.0)  # Example target_z value
            print("Moved gripper up.")
            self.moveUpButtonVal = p.readUserDebugParameter(self.sim.moveUpButton) + 1.0




if __name__ == "__main__":
    main = robotSimulator()

    