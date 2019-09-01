#!/usr/bin/env python
import random
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import random
import pybullet_data
from modelInfo_util import ModelInfo
from pkg_resources import parse_version
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)
from mamad_util import JointInfo

modelInfo = ModelInfo(path="./model_info.yml")
p.connect(p.GUI)
finger = p.loadSDF("./model.sdf")
fingerID = finger[0]
p.resetBasePositionAndOrientation(fingerID,[0.00000,0.200000,0.65000],[0.000000,0.000000,0.000000,1.000000])

jointInfo = JointInfo()
jointInfo.get_infoForAll_joints(finger)
active_joints_info  = jointInfo.getActiveJointsInfo()
num_active_joints = jointInfo.getNumberOfActiveJoints()
xpos = 0.05
ypos = 0.04
zpos = 1
ang = 3.14*0.5
orn = p.getQuaternionFromEuler([0,0,ang])
table = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"table/table.urdf"), 0.5000000,0.60000,0.0000,0.000000,0.000000,0.0,1.0)
texUid = p.loadTexture("./cube_new/aaa.png")
cube_objects = p.loadSDF("./cube_new/model.sdf")
cubeId = cube_objects[0]
p.changeVisualShape(cube_objects[0], -1,rgbaColor =[1,1,1,1])
p.changeVisualShape(cube_objects[0], -1, textureUniqueId = texUid)
p.resetBasePositionAndOrientation(cube_objects[0],(0.5000000,0.60000,0.6700),(0.717,0.0,0.0,0.717))
# print("active_joints_info::",active_joints_info)
# print("finger::",finger)
num_joints = p.getNumJoints(fingerID)
# print("`num of joints:::",num_joints)
blockUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"block.urdf"), xpos,ypos,zpos,orn[0],orn[1],orn[2],orn[3])
p.loadURDF(os.path.join(pybullet_data.getDataPath(),"plane.urdf"),[0,0,0])
for i in range(num_joints-1):
	j_info = p.getJointInfo(fingerID,i)
	# print("joint_info::",j_info)
p.setTimeStep(1.0/1000.0)
p.setRealTimeSimulation(0)
link_name ="lbr_iiwa_link_7"
link_name_encoded = link_name.encode(encoding='UTF-8',errors='strict')
kuka_ee_link_jointInfo = jointInfo.searchBy(key="linkName",value =link_name_encoded )[0]
#print(kuka_ee_link_jointInfo)
kuka_ee_link_Index = 6#kuka_ee_link_jointInfo["jointIndex"]
# print("kuka_ee_link_Index",kuka_ee_link_Index)
blockPos,blockOrn = p.getBasePositionAndOrientation(blockUid)
#print(blockPos,blockOrn)
new_pos = [xpos,ypos,zpos]
new_orn = [orn[0],orn[1],orn[2],orn[3]]
#print("test1",new_pos)
#print("test2",new_orn)
jointPoses  = p.calculateInverseKinematics(fingerID,kuka_ee_link_Index,new_pos,orn)

joint_state =[]
hand_links = modelInfo.get_hand_links()
hand_links_info = []

for link_name in hand_links:
	link_info = modelInfo.searchBy(key="link",value=link_name)
	hand_links_info.append(link_info)
hand_links_info_with_active_joints = []
for i in hand_links_info:
	if i["joint"]["j_type"] != "fixed":
		hand_links_info_with_active_joints.append(i)
#print("test2::::::",hand_links_info_with_active_joints)
hand_indexOf_active_joints =[]
for Link in hand_links_info_with_active_joints:
    link = jointInfo.searchBy("linkName",Link["link_name"])[0]
    hand_indexOf_active_joints.append(link["jointIndex"])
#print("test3::::::",hand_indexOf_active_joints)
#print(hand_links_info_with_active_joints)
for i in hand_indexOf_active_joints:
	joint_pos = p.getJointState(fingerID,i)[0]
	# print("test2::",p.getJointState(fingerID,i))
	joint_state.append(joint_pos)
#new_jointPoses = jointPoses +joint_state
# print("test::",joint_state)
# print("jointPoses::",jointPoses)	
# print("len(jointPoses)::",len(jointPoses))
# print("num_active_joints::",num_active_joints)
jointPoses_joint_name = []
for i in range(num_active_joints):
	jointPoses_joint_name.append(active_joints_info[i]["jointName"])

"""
['J0', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6',
 'lh_FFJ4', 'lh_FFJ3', 'lh_FFJ2', 'lh_FFJ1',
 'lh_MFJ4', 'lh_MFJ3', 'lh_MFJ2', 'lh_MFJ1',
 'lh_RFJ4', 'lh_RFJ3', 'lh_RFJ2', 'lh_RFJ1', 
 'lh_THJ5', 'lh_THJ4', 'lh_THJ2', 'lh_THJ1']

lh_FFJ3 ll:0.0  ul:1.5708
lh_MFJ3 ll:0.0  ul:1.5708
lh_RFJ3 ll:0.0  ul:1.5708

lh_FFJ4 ll:-0.349066  ul:0.349066
lh_MFJ4 ll:-0.349066  ul:0.349066
lh_RFJ4 ll:-0.349066  ul:0.349066

"""
jointPoses = (2.0, 1.0232196872785049, 0.15139632616115659, 2.640651350178653, -1.4544423913682578, 2.8769639265642334, 0.0,
 			 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
jointPoses = list(jointPoses)		  
#cuzing collsion between fingers			  
# changing lh_FFJ3
jointPoses[8] = jointPoses[8]
# changing lh_MFJ4 index = 13
jointPoses[11] = jointPoses[12]+0.3
# changing lh_RFJ3 index=17
jointPoses[16] = jointPoses[16]
contact_set = []
child_check = []


while(1):

	for i in range(num_active_joints):
		jointIndex = active_joints_info[i]["jointIndex"]
		jointll = active_joints_info[i]["jointLowerLimit"]
		jointul = active_joints_info[i]["jointUpperLimit"]
		jointState = p.getJointState(fingerID,jointIndex)[0]
		#print("jointName::",active_joints_info[i]["jointName"])
		#print("jointIndex::",jointIndex)
		# print("jointll::",jointll)
		# print("jointul::",jointul)
		#print("jointState::",jointState)
		#print(joint)
		# print("test:::",jointState)
		motor_command = (0.5,0.5,0.5)
		#print(motor_command)
	 	p.setJointMotorControl2(fingerID,7,p.POSITION_CONTROL,motor_command, targetVelocity=0,force=200.0, maxVelocity=0.35,positionGain=0.3,velocityGain=1)
	
	
	index_of_actvie_joints = [active_joints_info[i]["jointIndex"] for i in range(num_active_joints)]
	#print("index_of_actvie_joints",index_of_actvie_joints)
	for index_1 in index_of_actvie_joints:
		for index_2 in index_of_actvie_joints:
			if index_1 == index_2:
				continue
				
			contact=p.getClosestPoints(fingerID,fingerID,0.0,index_1,index_2)
			#print(contact)
			# contact=p.getClosestPoints(fingerID,fingerID,0.0)
			if len(contact) !=0:
				# print("contact",contact)
				# print("\n")
				# print("contact[0][3]",contact[0][3])
				link_one_name = jointInfo.searchBy("jointIndex",contact[0][3])[0]["linkName"]
				link_two_name = jointInfo.searchBy("jointIndex",contact[0][4])[0]["linkName"]
				#print("link_one_name dododod",jointInfo.searchBy("jointIndex",contact[0][3])[0])
				# print("link_one_name",link_one_name)
				contact_set.append([contact[0][3],contact[0][4]])
				#child_check.append([contact[0][3],contact[0][4]])
	#print("contact_set::",contact_set)
	#new_contact = []
	#new_contact = list(set(contact_set))
	#print(new_contact)
	#print("::",len(contact_set))
	new_contact = []
	for i in contact_set:
		#set1 = contact_set[i][0]
		#set2 = contact_set[i][1]
		#for i in len(contact_set):
			#if (not contact[i][0] in set1) and (not contact[i][1] in set2):
				#new_contact = 
		if not i in new_contact:
			new_contact.append(i)
	#print("new_contact::",new_contact)
	parent_set = []
	#print(len(new_contact))
	for i in range(len(new_contact)):
		#print(new_contact[i][0])
		if new_contact[i][0] - new_contact[i][1] == 1:
			parent_set.append(new_contact[i])
	#print(parent_set)
	child_set = []
	for i in range(len(new_contact)):
		if new_contact[i][1] - new_contact[i][0] == 1:
			child_set.append(new_contact[i])
	#print(child_set)

	parent_check = []
	for i in new_contact:
		for j in parent_set:
			if i == j:
				break
		else:
			parent_check.append(i)
	#print(parent_check)
	child_check = []
	for i in parent_check:
		for j in child_set:
			if i == j:
				break
		else:
			child_check.append(i)
	check_flip=[]
	for i in range(len(child_check)):
		index_1=child_check[i][0]
		index_2=child_check[i][1]
		for j in range(i,len(child_check)):
			if i == j:
				continue
			if index_1 == child_check[j][1] and index_2 ==  child_check[j][0]:
				check_flip.append(j)
	new_check=[]
	sort=np.argsort(check_flip)
	for i in range(len(check_flip)):
		new_check.append(check_flip[sort[i]])
	for i in range(len(check_flip)):
		del child_check[new_check[i]-i]
	#print(child_check)
	collision_result = []
	for i in range (len(child_check)):
		index_collision_set_1=child_check[i][0]
		index_collision_set_2=child_check[i][1]
		for j in range(num_active_joints):
			if index_collision_set_1 == active_joints_info[j]["jointIndex"]:
				index_collision_set_1_result = j
			if index_collision_set_2 == active_joints_info[j]["jointIndex"]:
				index_collision_set_2_result = j	

		collision_result.append([active_joints_info[index_collision_set_1_result]["linkName"],\
		active_joints_info[index_collision_set_2_result]["linkName"]])
	print("right hand self coliision -------",child_check)
	print("right hand self coliision -------",collision_result)
	print("\n")
p.stepSimulation()
	# print("jointPoses_joint_name",jointPoses_joint_name)
	# print("joint values form ik",jointPoses)
	# print("len(joint values form ik)",len(jointPoses))
	# print("getPhysicsEngineParameters",p.getPhysicsEngineParameters())

'''
	#print("contact_set::",contact_set)
	#new_contact = []
	#new_contact = list(set(contact_set))
	#print(new_contact)
	#print("::",len(contact_set))
	new_contact = []
	for i in contact_set:
		#set1 = contact_set[i][0]
		#set2 = contact_set[i][1]
		#for i in len(contact_set):
			#if (not contact[i][0] in set1) and (not contact[i][1] in set2):
				#new_contact = 
		if not i in new_contact:
			new_contact.append(i)
	#print("new_contact::",new_contact)
	parent_set = []
	#print(len(new_contact))
	for i in range(len(new_contact)):
		#print(new_contact[i][0])
		if new_contact[i][0] - new_contact[i][1] == 1:
			parent_set.append(new_contact[i])
	#print(parent_set)
	child_set = []
	for i in range(len(new_contact)):
		if new_contact[i][1] - new_contact[i][0] == 1:
			child_set.append(new_contact[i])
	#print(child_set)
'''