#!/usr/bin/env python
from time import sleep
import pybullet as p
import numpy as np

import random
from mamad_util import JointInfo

def setUpWorld(initialSimSteps=100):
    """
    Reset the simulation to the beginning and reload all models.

    Parameters
    ----------
    initialSimSteps : int

    Returns
    -------
    baxterId : int
    endEffectorId : int 
    """
    p.resetSimulation()

    
    

    sleep(0.1)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
    # Load Baxter
    Robot = p.loadSDF("./model.sdf")
    RobotId = Robot[0]
    euler_angle = [0,0,3.14]
    quaternion_angle = p.getQuaternionFromEuler(euler_angle)

    p.resetBasePositionAndOrientation(RobotId, [0, 0, 0],quaternion_angle)
    #p.resetBasePositionAndOrientation(RobotId, [0.5, -0.8, 0.0],[0,0,0,1])
    #p.resetBasePositionAndOrientation(RobotId, [0, 0, 0], )

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

    # Grab relevant joint IDs
    endEffectorId = 48 # (left gripper left finger)

    # Set gravity
    p.setGravity(0., 0., -10.)

    # Let the world run for a bit
    for _ in range(initialSimSteps):
        p.stepSimulation()

    return Robot,RobotId, endEffectorId

def setup_GUI_slidbars(Robot,RobotId):
    ids =[]
    #Getting robot info
    jointInfo = JointInfo()
    jointInfo.get_infoForAll_joints(Robot)
    #getting number of active joints
    active_joints_info  = jointInfo.getActiveJointsInfo()
    num_active_joints = jointInfo.getNumberOfActiveJoints()

    #setting up gui slider for each active joint
    for i in range(num_active_joints):
        jointName = active_joints_info[i]["jointName"]
        jointIndex = active_joints_info[i]["jointIndex"]
        jointll = active_joints_info[i]["jointLowerLimit"]
        jointul = active_joints_info[i]["jointUpperLimit"]
        jointPositionState = p.getJointState(RobotId,jointIndex)[0]
        sliderID = p.addUserDebugParameter(jointName,jointll,jointul,jointPositionState)
        ids.append(sliderID)
    return ids

def read_GUI_slidbars(sliders_id):
    motor_commands = []
    for id in sliders_id:
        motor_command = p.readUserDebugParameter(id)
        motor_commands.append(motor_command)
    return motor_commands


def check_collision(Robot,RobotId):
	
    
    #Getting robot info
    jointInfo = JointInfo()
    jointInfo.get_infoForAll_joints(Robot)
    #getting number of active joints
    active_joints_info  = jointInfo.getActiveJointsInfo()
    num_active_joints = jointInfo.getNumberOfActiveJoints()
    parent_list = []
    for i in range(num_active_joints):
        jointIndex = active_joints_info[i]["jointIndex"]
        parent_list.append([jointIndex,jointInfo.searchBy("jointIndex",jointIndex)[0]["parentIndex"]])
    
    collision_set=[]
    index_of_active_joints = [active_joints_info[i]["jointIndex"] for i in range(num_active_joints)]
    for i in index_of_active_joints:
    	for j in index_of_active_joints:
    		if i == j:
    			continue
    		contact = p.getClosestPoints(RobotId,RobotId,-0.001,i,j)
    
    		if len(contact)!=0:
    			collision_set.append([contact[0][3],contact[0][4]])

    check_flip=[]
    for i in range(len(collision_set)):
    	index_1=collision_set[i][0]
    	index_2=collision_set[i][1]
    	for j in range(i,len(collision_set)):
    		if i == j:
    			continue
    		if index_1 == collision_set[j][1] and index_2 ==  collision_set[j][0]:
    			check_flip.append(j)

    new_check=[]
    sort=np.argsort(check_flip)
    for i in range(len(check_flip)):
    	new_check.append(check_flip[sort[i]])
    for i in range(len(check_flip)):
    	del collision_set[new_check[i]-i]

    check_parent=[]
    for i in range(len(parent_list)):
    	index_parent_1=parent_list[i][0]
    	index_parent_2=parent_list[i][1]
    	for j in range(len(collision_set)):
    		if index_parent_1 == collision_set[j][0] and index_parent_2 ==  collision_set[j][1]:
    			check_parent.append(j)
    		if index_parent_1 == collision_set[j][1] and index_parent_2 ==  collision_set[j][0]:
    			check_parent.append(j)
    new_check_parent=[]
    sort_parent=np.argsort(check_parent)
    for i in range(len(check_parent)):
    	new_check_parent.append(check_parent[sort_parent[i]])
    for i in range(len(check_parent)):
    	del collision_set[new_check_parent[i]-i]    
    collision_result=[]
    for i in range (len(collision_set)):
    	index_collision_set_1=collision_set[i][0]
    	index_collision_set_2=collision_set[i][1]
    	for j in range(num_active_joints):
    		if index_collision_set_1 == active_joints_info[j]["jointIndex"]:
    			index_collision_set_1_result = j
    		if index_collision_set_2 == active_joints_info[j]["jointIndex"]:
    			index_collision_set_2_result = j	    
    	collision_result.append([active_joints_info[index_collision_set_1_result]["linkName"],active_joints_info[index_collision_set_2_result]["linkName"]])
    return collision_result


def setMotors(bodyId, jointPoses):
    """
    Parameters
    ----------
    bodyId : int
    jointPoses : [float] * numDofs
    """
    numJoints = p.getNumJoints(bodyId)

    for i in range(numJoints):
        jointInfo = p.getJointInfo(bodyId, i)
        #print(jointInfo)
        qIndex = jointInfo[3]
        if qIndex > -1:
            p.setJointMotorControl2(bodyIndex=bodyId, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[qIndex-7])



if __name__ == "__main__":
    guiClient = p.connect(p.GUI)
    p.resetDebugVisualizerCamera(1, 180, 0., [0., 0.2, 0.3])

    Robot,RobotId, endEffectorId = setUpWorld()

    # sliders_id = setup_GUI_slidbars(Robot,RobotId)
    # 
    # 
# 
    # maxIters = 100000
# 
    # sleep(1.)
# 
    # p.getCameraImage(320,200, renderer=p.ER_BULLET_HARDWARE_OPENGL )
    # for _ in range(maxIters):
    #   p.stepSimulation()
    #   jointPoses = read_GUI_slidbars(sliders_id)      
    #   setMotors(RobotId, jointPoses)
    #   collision = check_collision(Robot,RobotId)
    #   print(collision)
      #sleep(0.1)

