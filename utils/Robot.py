import numpy as np
import scipy.linalg as LA
import time
from time import sleep
import pybullet as p
import dqrobotics
from dqrobotics import robot_modeling
from dqrobotics.robot_modeling import DQ_Kinematics
from dqrobotics import DQ

import math
from math import pi
import os
import pandas as pd

class Robot:
    def __init__(self, params, gripPath):
        """Initializes constants and sets up only the gripper"""
        self.gripperName = params["gripper"]
        self.start_pos = params["robot_start_pos"]
        self.start_orientation = p.getQuaternionFromEuler(params["robot_start_orientation_euler"])
        self.gripPath = gripPath

        # Load the chosen gripper's URDF model
        if self.gripperName == "customGripper":
            self.robot_model = p.loadURDF("pr2_gripper.urdf", self.start_pos, self.start_orientation, useFixedBase=True)
        elif self.gripperName == "threeFingers":
            self.robot_model = p.loadURDF("Robots/grippers/threeFingers/sdh/sdh.urdf", self.start_pos, self.start_orientation, useFixedBase=True)
        elif self.gripperName == "RG6":
            self.robot_model = p.loadURDF("Robots/grippers/RG6/robotiq_arg85_description.URDF", self.start_pos, self.start_orientation, useFixedBase=True)
        elif self.gripperName == "shadowHand":
            self.robot_model = p.loadURDF("Robots/grippers/sr_grasp_description/urdf/shadowHand.urdf", self.start_pos, self.start_orientation, useFixedBase=True)
        else:
            raise ValueError(f"[ERROR]: Gripper '{self.gripperName}' not supported.")

        # Set the number of gripper joints
        self.numGripperJoints = p.getNumJoints(self.robot_model)

        # Initialize the gripper
        if self.gripperName == "RG6":
            self.gripper = RG6(self)
        elif self.gripperName == "threeFingers":
            self.gripper = threeFingers(self)
        elif self.gripperName == "shadowHand":
            self.gripper = shadowHand(self)

        # Constants
        self.K = params.get("K", 1.0)
        self.enableCamera = params.get("camera", False)
        if self.enableCamera:
            self.cameraPos = params.get("camera_pos", [0, 0, 0])
            self.camera()

    def setJointPosition(self, targetJointPos):
        """ Set robot to a fixed position """
        for j in range(self.numGripperJoints):
            p.resetJointState(self.robot_model, j, targetJointPos[j])

    def getJointPosition(self):
        """Return joint position for each joint as a list"""
        return [p.getJointState(self.robot_model, j)[0] for j in range(self.numGripperJoints)]

    def getPose(self):
        """ return pose"""
        # Ensure self.serialManipulator is properly initialized if used
        theta = self.getJointPosition()
        pose = self.serialManipulator.fkm(theta) if hasattr(self, 'serialManipulator') else None
        return pose

    def getJointVelocity(self):
        """Return joint velocity for each joint as a list"""
        return [p.getJointState(self.robot_model, j)[1] for j in range(self.numGripperJoints)]

    def moveArmToEETarget(self, EE_target, epsilon):
        theta = np.zeros(self.numGripperJoints)
        error_pos = epsilon + 1
        iteration = 0
        while LA.norm(error_pos) > epsilon:
            theta = self.getJointPosition()
            EE_pos = self.serialManipulator.fkm(theta)
            J = self.serialManipulator.pose_jacobian(theta)
            if EE_pos.q[0] < 0:
                EE_pos = -1 * EE_pos
            EE_pos = EE_pos * (EE_pos.norm().inv())
            error = dqrobotics.DQ.vec8(EE_target - EE_pos)
            thetaout = theta + np.dot(np.dot(np.transpose(J), 0.5 * self.K), error)
            self.setJointPosition(thetaout)
            self.simStep()
            iteration += 1
            if iteration > 300:
                break

    def camera(self):
        """ Show camera image at given position """
        fov, aspect, nearplane, farplane = 60, 1.0, 0.01, 100
        projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)
        linkPos = self.numGripperJoints - 1
        com_p, com_o, _, _, _, _ = p.getLinkState(self.robot_model, linkPos, computeForwardKinematics=True)
        rot_matrix = np.array(p.getMatrixFromQuaternion(com_o)).reshape(3, 3)
        com_o_Eul = p.getEulerFromQuaternion(com_o)
        normtemp = np.linalg.norm(com_o_Eul)
        if normtemp != 0:
            com_p1 = com_p[0] - (0.2 / normtemp) * com_o_Eul[0]
            com_p2 = com_p[1] - (0.2 / normtemp) * com_o_Eul[1]
            com_p3 = com_p[2] - (0.2 / normtemp) * com_o_Eul[2]
        else:
            com_p1, com_p2, com_p3 = com_p
        com_p = (com_p1, com_p2, com_p3)
        init_camera_vector = (0, 0, 1)
        init_up_vector = (0, 1, 0)
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        view_matrix = p.computeViewMatrix(com_p, com_p + 0.5 * camera_vector, up_vector)
        img = p.getCameraImage(200, 200, view_matrix, projection_matrix)
        return img

    def simStep(self):
        """ Simulation step """
        if self.enableCamera:
            self.camera()
        else:
            sleep(1. / 20.)