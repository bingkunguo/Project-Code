import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)
from pkg_resources import resource_string,resource_filename

import pybullet as p
import numpy as np
import copy
import math
import pybullet_data
import time


from mamad_util import JointInfo
from kuka_handlit_model.modelInfo_util import ModelInfo


class Kuka_Handlit:

  def __init__(self,ee_xyz,kuka_loc =None, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01 ):
    self.kuka_loc =kuka_loc
    self.urdfRootPath = urdfRootPath
    self.timeStep = timeStep
    self.maxVelocity = .35
    self.maxForce = 200.
    self.fingerAForce = 2 
    self.fingerBForce = 2.5
    self.fingerTipForce = 2
    self.useSimulation = 1
    self.useNullSpace =21
    self.useOrientation = 1
    self.kuka_handId =3
    self.kukaEndEffectorIndex = 6
    self.kukaGripperIndex = 7
    self.ee_xyz = ee_xyz
    self.jointInfo = JointInfo()
    model_info_path = resource_filename(__name__,"/kuka_handlit_model/model_info.yml")
    self.modelInfo = ModelInfo(model_info_path)
    self.fingerTip_link_name = ["distal_FF","distal_MF","distal_RF","thdistal"] #this are joints between final link and one before it
    #lower limits for null space
    self.ll=[-.967,-2 ,-2.96,0.19,-2.96,-2.09,-3.05]
    #upper limits for null space
    self.ul=[.967,2 ,2.96,2.29,2.96,2.09,3.05]
    #joint ranges for null space
    self.jr=[5.8,4,5.8,4,5.8,4,6]
    #restposes for null space
    self.rp=[0,0,0,0.5*math.pi,0,-math.pi*0.5*0.66,0]
    #joint damping coefficents
    self.jd=[0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001]
    self.reset()




  def cameraSetup(self):
    #https://github.com/bulletphysics/bullet3/issues/1616
    width = 128
    height = 128
    
    fov = 60 # field of view
    aspect = width/height
    near = 0.02
    far =1
    endEffector_info = p.getLinkState(self.kuka_handId,self.kukaEndEffectorIndex,computeForwardKinematics=True)
    # print("endEffector_info",endEffector_info)
    
    endEffector_pos  = endEffector_info[4]
    endEffector_ori  = endEffector_info[5]

    # print("endEffector_pos",endEffector_pos)
    # print("endEffector_ori",endEffector_ori)
    endEffector_pos_Xoffset =0.
    endEffector_pos_Zoffset =-0.05

    endEffector_pos_InWorldPosition = endEffector_pos
    cameraEyePosition = endEffector_pos_InWorldPosition

    cameraEyePosition_ = [endEffector_pos[0]+endEffector_pos_Xoffset,endEffector_pos[1],endEffector_pos[2]+endEffector_pos_Zoffset]
    rot_matrix = p.getMatrixFromQuaternion(endEffector_ori)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    # Initial vectors
    init_camera_vector = (0, 0, 1) # modelInfo = ModelInfo(path="./kuka_handlit/model_info.yml")axis
    init_up_vector = (0, 1, 0) # y-axis
    # Rotated vectors
    camera_vector = rot_matrix.dot(init_camera_vector)
    up_vector = rot_matrix.dot(init_up_vector)
    
    cameraEyePosition = endEffector_pos_InWorldPosition
    dist_cameraTargetPosition = -0.02

    view_matrix =  p.computeViewMatrix(cameraEyePosition_, cameraEyePosition + 0.1 * camera_vector, up_vector)

    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    # Get depth values using the OpenGL renderer
   
    images = p.getCameraImage(width, height, view_matrix, projection_matrix, shadow=True,renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb_opengl= np.reshape(images[2], (height, width, 4))*1./255.
    depth_buffer_opengl = np.reshape(images[3], [width, height])
    depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)
    seg_opengl = np.reshape(images[4], [width, height])*1./255.
    time.sleep(1)
  
  def reset(self):

    
    robot_path = resource_filename(__name__,"/kuka_handlit_model/model.sdf")
    self._kuka_hand = p.loadSDF(robot_path)
    self.kuka_handId = self._kuka_hand[0]


    if self.kuka_loc !=None:
      p.resetBasePositionAndOrientation(self.kuka_handId,self.kuka_loc[0:3],self.kuka_loc[3:])
    self.jointInfo.get_infoForAll_joints(self._kuka_hand)
    self.numJoints = p.getNumJoints(self.kuka_handId)
    self.num_Active_joint = self.jointInfo.getNumberOfActiveJoints()
    self.indexOf_activeJoints  = self.jointInfo.getIndexOfActiveJoints()

    resetJoints= p.calculateInverseKinematics(self.kuka_handId,self.kukaEndEffectorIndex,self.ee_xyz)
    
    self.motorNames = []
    self.motorIndices = []
    
    for i in range (self.numJoints):
      jointInfo = p.getJointInfo(self.kuka_handId,i)
      qIndex = jointInfo[3]
      if qIndex > -1:

        self.motorNames.append(str(jointInfo[1]))
        self.motorIndices.append(i)

    for i in range (self.num_Active_joint):
      p.resetJointState(self.kuka_handId,self.indexOf_activeJoints[i],resetJoints[i])

  def getActionDimension(self):
    numOf_activeJoints = self.jointInfo.getNumberOfActiveJoints()
    return numOf_activeJoints #position x,y,z and roll/pitch/yaw euler angles of end effector

  def getObservationDimension(self):
    return len(self.getObservation())

  def getObservation_joint(self,format="list"):
    
    indexOfActiveJoints = self.jointInfo.getIndexOfActiveJoints()
    jointsInfo = self.jointInfo.getActiveJointsInfo()

    jointsStates = []
    joints_state = {} #key:joint ,value = joint state 
    
    for i in range(len(indexOfActiveJoints)):
      jointName  = jointsInfo[i]["jointName"]
      jointIndex = indexOfActiveJoints[i]
      jointState = p.getJointState(self._kuka_hand[0],jointIndex)
      joints_state[jointName] = jointState[0]
      jointsStates.append(jointState[0])

    if format == "dictinary":
      return joints_state
    else:
      return jointsStates

  def getObservation_palm(self):
    Index = self.get_palm_linkIndex()
    state = p.getLinkState(self.kuka_handId,Index)
    return state[0]+state[1]
  def getObservation(self,UseEuler=False):
    def finger_obs(UseEuler):
      state_dict = {}
      observation = []

      fingerTipIndexs = self.get_fingerTips_linkIndex()
      
      counter = 0 
      #getting fingers tip position and orientation
      for index in fingerTipIndexs:
        state = p.getLinkState(self.kuka_handId,index)#mamad:returns endeffector info position orientation
        pos = state[0] #mamad: linkWorldPosition
        orn = state[1] #mamad: linkWorldOrientation
        state_dict[self.fingerTip_link_name[counter]] = {"pos":pos,"orn":orn}                                          
        counter +=1
    
      #print("Debug::state_dict",state_dict)

      for finger in self.fingerTip_link_name:
        euler = p.getEulerFromQuaternion(state_dict[finger]["orn"])
        pos   = state_dict[finger]["pos"]  
        observation.extend(list(pos))
        if UseEuler ==True:
          observation.extend(list(euler))
        else:
          observation.extend(list(state_dict[finger]["orn"]))
      
      return observation

    def kuka_obs(UseEuler):
      observation = []
      kuka_ee_link_Index = self.get_kuka_ee_linkIndex()

      state = p.getLinkState(self.kuka_handId,kuka_ee_link_Index)#mamad:returns endeffector info position orientation
      pos = state[0] #mamad: linkWorldPosition
      orn = state[1] #mamad: linkWorldOrientation
      euler = p.getEulerFromQuaternion(orn)
          
      observation.extend(list(pos))
      if UseEuler ==True:
        observation.extend(list(euler))
      else:
        observation.extend(list(orn))
    
      return observation


    #self.cameraSetup()
    observation = kuka_obs(UseEuler)+finger_obs(UseEuler)
    #print("Debug::observation",observation)
    #print("Debug::len(observation)",len(observation))
    return observation

  def applyAction_2(self,motorCommands):
    
    counter = 0
    num_active_joints = self.jointInfo.getNumberOfActiveJoints()
    active_joints_info = self.jointInfo.getActiveJointsInfo()
    for i in range(num_active_joints):
      jointIndex = active_joints_info[i]["jointIndex"]
      motor_command = motorCommands[counter]
      #Applying command for kuka

      position = p.calculateInverseKinematics(self.kuka_handId,7,[0.5,0.6,1.2])
   
      if i <= 6:
        p.setJointMotorControl2(self.kuka_handId,jointIndex,p.POSITION_CONTROL,motor_command, targetVelocity=0,force=200.0, maxVelocity=0.35,positionGain=0.3,velocityGain=1)
     #Appying command for hand
      else:
    
        p.setJointMotorControl2(bodyIndex=self.kuka_handId, jointIndex=jointIndex, controlMode=p.POSITION_CONTROL,targetPosition=motor_command)

      counter = counter+1
  def applyAction(self, motorCommands):

    def apply_A_hand(motorCommands):
      """
      the first 16 command will be for the hand. the hand has 16 active joints
      The actions are going to Active joint values.
      gettting current state of Avtive joints before applying actions This is different that the state we get in getObservation
      motorCommands = motorCommands[0:16]
      """

      joint_state = [] #current joint postion
      new_joint_pos = [0]*len(self.get_hand_active_joints_index()) # new joint position
      for jointIndex in self.get_hand_active_joints_index():
        joint_pos = p.getJointState(self.kuka_handId,jointIndex)[0]
        joint_state.append(joint_pos)
    
      #making sure the joint values suggested by agent does not exceed joint limit
      #design question: should i give negative reward to agent for suggesting a joint value outside joint limit
      counter = 0 
     
      for jointIndex in self.get_hand_active_joints_index():
        jointinfo = self.jointInfo.searchBy("jointIndex",jointIndex)[0]
        joint_ll = jointinfo["jointLowerLimit"]
        joint_ul = jointinfo["jointUpperLimit"]
        if motorCommands[counter]<joint_ul and motorCommands[counter]>joint_ll:
          new_joint_pos[counter] = joint_state[counter]+motorCommands[counter]
        counter +=1
      
      return new_joint_pos
    
    def apply_A_kuka(motorCommands):
      kuka_ee_link_Index = self.get_kuka_ee_linkIndex()
      motorCommands = motorCommands[16:]# the first three command will be xyz and the second three command will be rpy
      kuka_ee_index = self.get_kuka_ee_linkIndex()
      kuka_ee_state  = p.getLinkState(self.kuka_handId,kuka_ee_link_Index)#mamad:returns endeffector info position orientation
      pos_state = kuka_ee_state[0] #mamad: linkWorldPosition
      orn_state = kuka_ee_state[1] #mamad: linkWorldOrientation
      pos_command = motorCommands[0:3]
      orn_command = motorCommands[3:]
      new_pos  = pos_command
      new_orn  = p.getQuaternionFromEuler(orn_command)
 

      #getting joint values using inverse kinematic
      jointPoses   = p.calculateInverseKinematics(self.kuka_handId,kuka_ee_index,new_pos,new_orn)
      jointPoses = (-0.36350324105018117, 0.08434877972795868, 1.3253607910308352, -1.3023695529206833, 1.1981002550715345, -2.4402425653234032, -0.3733762283635729, 0.011011669082923765, 0.01099649407428263, 0.010663992122582346, 0.01090501619723987, -0.405810978009454, 0.7964372388949594, 2.128544355877948e-06, 8.43294195425921e-08, 4.4055107276974525e-07, 0.4617504554232501, 1.3463347345016152e-07, 2.0510552526533784e-06, -1.1743303952431874, -0.21315269585784075, -0.6981321138974356, 5.54231275569411e-06)
     
      kuka_activeJ =self.get_kuka_Active_joints()
      #Applying new_joint_pos to kuka
      counter = 0
      return list(jointPoses[0:8])


    hand_joints_command=apply_A_hand(motorCommands)
    kuka_joints_command=apply_A_kuka(motorCommands)
 
    motorCommands = kuka_joints_command+hand_joints_command
    
    counter = 0
    num_active_joints = self.jointInfo.getNumberOfActiveJoints()
    active_joints_info = self.jointInfo.getActiveJointsInfo()
    for i in range(num_active_joints):
      jointIndex = active_joints_info[i]["jointIndex"]
      motor_command = motorCommands[counter]
    
      p.setJointMotorControl2(self.kuka_handId,jointIndex,p.POSITION_CONTROL,motor_command, force=1.0)
      counter = counter+1
  
  
  #utility functions

  def get_fingerTips_linkIndex(self):
    fingerTips_linkIndex = []
    fingerTips_jointInfo = []
    fingerTip_joint_name_bytes = [i.encode(encoding='UTF-8',errors='strict') for i in self.fingerTip_link_name]
  
    # getting joints for the final link
    for i in fingerTip_joint_name_bytes:
      fingerTips_jointInfo.append(self.jointInfo.searchBy(key="linkName",value = i)[0])
     
    #extracting joint index which equivalent to link index
   
    for i in fingerTips_jointInfo:

   
      fingerTips_linkIndex.append(i["jointIndex"])  
  
    return fingerTips_linkIndex


  def get_palm_linkIndex(self):
    link_name ="palm_fake"

    link_name_encoded = link_name.encode(encoding='UTF-8',errors='strict')
    palm_link_jointInfo = self.jointInfo.searchBy(key="linkName",value =link_name_encoded )[0]

    palm_link_Index  = palm_link_jointInfo["jointIndex"]
   
    return palm_link_Index

  def get_kuka_ee_linkIndex(self):
    link_name ="lbr_iiwa_link_7"

    link_name_encoded = link_name.encode(encoding='UTF-8',errors='strict')
    kuka_ee_link_jointInfo = self.jointInfo.searchBy(key="linkName",value =link_name_encoded )[0]

    kuka_ee_link_Index     = kuka_ee_link_jointInfo["jointIndex"]
   
    return kuka_ee_link_Index
  def get_kuka_Active_joints(self):
    link_regex = r'(lbr_iiwa_link_)+[0-7]'
    robot_active_joints = self.jointInfo.getActiveJointsInfo()
    kuka_active_joints  = self.jointInfo.searchBy_regex("jointName",link_regex,robot_active_joints)
    return kuka_active_joints

  def get_hand_active_joints_index(self):
          #getting links with active joints
        hand_links = self.modelInfo.get_hand_links()
        hand_links_info = []

        for link_name in hand_links:
          link_info = self.modelInfo.searchBy(key="link",value=link_name)
          hand_links_info.append(link_info)

        hand_links_info_with_active_joints = []
        for hand_link_info in hand_links_info:
          if hand_link_info["joint"]["j_type"] != "fixed":
            hand_links_info_with_active_joints.append(hand_link_info)
        hand_indexOf_active_joints =[]
        for Link in hand_links_info_with_active_joints:
          link = self.jointInfo.searchBy("linkName",Link["link_name"])[0]
          hand_indexOf_active_joints.append(link["jointIndex"])
        
        return hand_indexOf_active_joints