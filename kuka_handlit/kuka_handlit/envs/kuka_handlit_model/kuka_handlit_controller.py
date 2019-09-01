import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import pybullet as p
import numpy as np
import copy
import math
import pybullet_data
import time


from mamad_util import JointInfo
from modelInfo_util import ModelInfo


class Kuka_Handlit:

  def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01):
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
    self.jointInfo = JointInfo()
    self.modelInfo = ModelInfo(path="./model_info.yml")
    self.fingerTip_link_name = ["lh_ffdistal","lh_mfdistal","lh_rfdistal","lh_thdistal"] #this are joints between final link and one before it
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

    
    self._kuka_hand = p.loadSDF("model.sdf")
    self.kuka_handId = self._kuka_hand[0]
    print("self._kuka_hand",self._kuka_hand )
    print("self.kuka_handId",self.kuka_handId )
  
    #self.cameraSetup()
    # reset joints to the middle valu 
    
    p.resetBasePositionAndOrientation(self.kuka_handId,[-0.100000,0.000000,0.070000],[0.000000,0.000000,0.000000,1.000000])
    self.jointInfo.get_infoForAll_joints(self._kuka_hand)
    self.numJoints = p.getNumJoints(self.kuka_handId)
    self.num_Active_joint = self.jointInfo.getNumberOfActiveJoints()
    self.indexOf_activeJoints  = self.jointInfo.getIndexOfActiveJoints()
  
  
    
  
    
    
    self.motorNames = []
    self.motorIndices = []
    
    for i in range (self.numJoints):
      jointInfo = p.getJointInfo(self.kuka_handId,i)
      qIndex = jointInfo[3]
      if qIndex > -1:

        self.motorNames.append(str(jointInfo[1]))
        self.motorIndices.append(i)
 
  def getActionDimension(self):
    numOf_activeJoints = self.jointInfo.getNumberOfActiveJoints()
    return numOf_activeJoints #position x,y,z and roll/pitch/yaw euler angles of end effector

  def getObservationDimension(self):
    return len(self.getObservation())

  def getObservation(self):
    def finger_obs():
      state_dict = {}
      observation = []

      fingerTipIndexs = self.get_fingerTips_linkIndex()
      #print("Debug::fingerTipIndexs",fingerTipIndexs)
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
        observation.extend(list(euler))
      
      return observation

    def kuka_obs():
      observation = []
      kuka_ee_link_Index = self.get_kuka_ee_linkIndex()

      state = p.getLinkState(self.kuka_handId,kuka_ee_link_Index)#mamad:returns endeffector info position orientation
      pos = state[0] #mamad: linkWorldPosition
      orn = state[1] #mamad: linkWorldOrientation
      euler = p.getEulerFromQuaternion(orn)
          
      observation.extend(list(pos))
      observation.extend(list(euler))
    
      return observation


    #self.cameraSetup()
    observation = kuka_obs()+finger_obs()
    #print("Debug::observation",observation)
    #print("Debug::len(observation)",len(observation))
    return observation

  def applyAction(self, motorCommands):
    #print("controller Motorcommand",motorCommands)
    #print("controller Motorcommand len",len(motorCommands))
    def apply_A_hand(motorCommands):
      # the first 16 command will be for the hand. the hand has 16 active joints
      #The actions are going to Active joint values.
      #gettting current state of Avtive joints before applying actions This is different that the state we get in getObservation
      motorCommands = motorCommands[0:16]
      print("hand_motor_commands",len(motorCommands))
      joint_state = [] #current joint postion
      new_joint_pos = [0]*len(self.get_hand_active_joints_index()) # new joint position
      for jointIndex in self.get_hand_active_joints_index():
        joint_pos = p.getJointState(self.kuka_handId,jointIndex)[0]
        joint_state.append(joint_pos)
      #print("Debug::joint_state",joint_state)
      #making sure the joint values suggested by agent does not exceed joint limit
      #design question: should i give negative reward to agent for suggesting a joint value outside joint limit
      counter = 0 
      print("get_hand_active_joints_index()",self.get_hand_active_joints_index)
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
      new_orn  = new_pos

      #getting joint values using inverse kinematic
      jointPoses   = p.calculateInverseKinematics(self.kuka_handId,kuka_ee_index,new_pos,new_orn)
      kuka_activeJ =self.get_kuka_Active_joints()
      #Applying new_joint_pos to kuka
      counter = 0
      return jointPoses


    hand_joints_command=apply_A_hand(motorCommands)
    kuka_joints_command=apply_A_kuka(motorCommands)
    motorCommands = hand_joints_command+kuka_joints_command
    
    counter = 0
    num_active_joints = jointInfo.getNumberOfActiveJoints()
    for i in range(num_active_joints):
      jointIndex = active_joints_info[i]["jointIndex"]
      motor_command = motorCommands[counter]
      # print("motor_command",motor_command)
      p.setJointMotorControl2(self.kuka_handId,jointIndex,p.POSITION_CONTROL,motor_command, force=1.0)
      counter = counter+1
  #utility functions

  def get_fingerTips_linkIndex(self):
    fingerTips_linkIndex = []
    fingerTips_linkInfo = []
    fingerTip_link_name_bytes = [i.encode(encoding='UTF-8',errors='strict') for i in self.fingerTip_link_name]
    # getting joints for the final link
    for i in fingerTip_link_name_bytes:
      fingerTips_linkInfo.append(self.jointInfo.searchBy(key="linkName",value = i))
      
    #print("Debug8:: ",fingerTips_linkInfo)
    #extracting joint index which equivalent to link index
    #print("Debug13::len(fingerTips_linkInfo)",len(fingerTips_linkInfo))
    for i in fingerTips_linkInfo:
      i=i[0]
      #print("Debug11 i",i)
      #print("Debug12::i['jointIndex']",i["jointIndex"])
      fingerTips_linkIndex.append(i["jointIndex"])  
    #print("Debug14::fingerTips_linkIndex",fingerTips_linkIndex)
    return fingerTips_linkIndex


  def get_kuka_ee_linkIndex(self):
    link_name         ="lbr_iiwa_link_7"
    link_name_encoded = link_name.encode(encoding='UTF-8',errors='strict')
    kuka_ee_link_jointInfo = self.jointInfo.searchBy(key="linkName",value =link_name_encoded )[0]
    #kuka_ee_liml_jointLLimit = self.jointInfo.getJointLimits(key="linkName",value =link_name_encoded )
    print("Debug::kuka_ee_link_jointInfo",self.jointInfo)
    #print("Debug::kuka_ee_liml_jointLLimit",kuka_ee_liml_jointLLimit)
    
    #print("Debug2::AllJoint",self.jointInfo.get_stored_joints())
    #print("Debug1::kuka_ee_link_jointInfo",kuka_ee_link_jointInfo)
    kuka_ee_link_Index     = kuka_ee_link_jointInfo["jointIndex"]
    #print("Debug2::kuka_ee_link_Index",kuka_ee_link_Index)
    return kuka_ee_link_Index
  def get_kuka_Active_joints(self):
    link_regex = r'(lbr_iiwa_link_)+[0-7]'
    robot_active_joints = self.jointInfo.getActiveJointsInfo()
    kuka_active_joints  = self.jointInfo.searchBy_regex("jointIndex",link_regex,robot_active_joints)
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
        print ("hand_indexOf_active_joints",hand_indexOf_active_joints)
        return hand_indexOf_active_joints