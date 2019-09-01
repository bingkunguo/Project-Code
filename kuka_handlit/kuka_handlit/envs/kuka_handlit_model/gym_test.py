#!/usr/bin/env python

import pybullet as p

from mamad_util import JointInfo
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)
import pybullet_data
from pkg_resources import parse_version


kuka_handId = 3
xpos = 0.43
ypos = 0.635
zpos = 0.909
ang = 3.14*0.5
orn = p.getQuaternionFromEuler([0,0,ang])

p.connect(p.GUI)
finger = p.loadSDF("./model.sdf")
fingerID = finger[0]
print("fingerID:::",fingerID)
p.resetBasePositionAndOrientation(fingerID,[0.00000,0.200000,0.65000],[0.000000,0.000000,0.000000,1.000000])
table = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"table/table.urdf"), 0.5000000,0.60000,0.0000,0.000000,0.000000,0.0,1.0)
texUid = p.loadTexture("./cube_new/aaa.png")
cube_objects = p.loadSDF("./cube_new/model.sdf")
cubeId = cube_objects[0]
p.changeVisualShape(cube_objects[0], -1,rgbaColor =[1,1,1,1])
p.changeVisualShape(cube_objects[0], -1, textureUniqueId = texUid)
p.resetBasePositionAndOrientation(cube_objects[0],(0.5000000,0.60000,0.6700),(0.717,0.0,0.0,0.717))
#blockUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"block.urdf"), xpos,ypos,zpos,orn[0],orn[1],orn[2],orn[3])
blockUid = 2
p.loadURDF(os.path.join(pybullet_data.getDataPath(),"plane.urdf"),[0,0,0])

jointInfo = JointInfo()
jointInfo.get_infoForAll_joints(finger)
active_joints_info  = jointInfo.getActiveJointsInfo()
num_active_joints = jointInfo.getNumberOfActiveJoints()



#print("active_joints_info::",active_joints_info)
#print("finger::",finger)
num_joints = p.getNumJoints(fingerID)
#print("`num of joints:::",num_joints)

for i in range(num_joints-1):
	j_info = p.getJointInfo(fingerID,i)
	#print("joint_info::",j_info)

p.setRealTimeSimulation(0)


link_name ="lbr_iiwa_link_7"
link_name_encoded = link_name.encode(encoding='UTF-8',errors='strict')
kuka_ee_link_jointInfo = jointInfo.searchBy(key="linkName",value =link_name_encoded )[0]
#print(kuka_ee_link_jointInfo)
kuka_ee_link_Index = kuka_ee_link_jointInfo["jointIndex"]
blockPos,blockOrn = p.getBasePositionAndOrientation(blockUid)
#print(blockPos,blockOrn)
new_pos = [xpos,ypos,zpos]
#new_orn = blockOrn
new_orn = [2,-0.1,0.0,0.717]
#print(kuka_handId)
#print("test1",new_pos)
#print("test2",new_orn)


jointPoses  = p.calculateInverseKinematics(fingerID,kuka_ee_link_Index,new_pos,new_orn)
#print("kuka_ee_link_Index",kuka_ee_link_Index)
#print("jointPoses::",jointPoses)
#print("len(jointPoses)::",len(jointPoses))
while(1):
		
	for i in range(num_active_joints):
		jointIndex = active_joints_info[i]["jointIndex"]
		jointll = active_joints_info[i]["jointLowerLimit"]
		jointul = active_joints_info[i]["jointUpperLimit"]
		motor_command = jointPoses[i]
		p.setJointMotorControl2(fingerID,jointIndex,p.POSITION_CONTROL,motor_command, targetVelocity=0,force=200.0, maxVelocity=0.35,positionGain=0.3,velocityGain=1)
	p.stepSimulation()


