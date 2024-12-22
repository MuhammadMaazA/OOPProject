import pybullet as p
import time

def move_down_smoothly(gripper, target_z, steps=200, delay=0.01):
    current_position, current_orientation = p.getBasePositionAndOrientation(gripper.gripper)
    start_z = current_position[2]
    delta_z = (target_z - start_z) / steps
    velocity = delta_z / delay

    for _ in range(steps):
        p.resetBaseVelocity(gripper.gripper, linearVelocity=[0, 0, velocity])
        p.stepSimulation()
        time.sleep(delay)
    p.resetBaseVelocity(gripper.gripper, linearVelocity=[0, 0, 0])