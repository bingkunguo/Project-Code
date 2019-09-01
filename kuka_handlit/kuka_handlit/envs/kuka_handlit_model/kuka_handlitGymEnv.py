import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)

import random
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
from kuka_handlit_controller import Kuka_Handlit
import random
import pybullet_data
from pkg_resources import parse_version

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class Kuka_HandlitGymEnv(gym.Env):
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second' : 50
  }

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=False,
               maxSteps = 1000):
    #print("KukaGymEnv __init__")
    self._isDiscrete = isDiscrete
    self._timeStep = 1./240.
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self._maxSteps = maxSteps
    self.terminated = 0
    self._cam_dist = 1.3
    self._cam_yaw = 180
    self._cam_pitch = -40

    self._p = p
    #timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")
    self._seed()
    self.reset()
    observationDim = len(self.getExtendedObservation())
    #print("observationDim")
    #print(observationDim)

    observation_high = np.array([largeValObservation] * observationDim)
    if (self._isDiscrete):
      self.action_space = spaces.Discrete(7)
    else:
       action_dim = self._kuka_hand.num_Active_joint
       self._action_bound = 1
       action_high = np.array([self._action_bound] * action_dim)
       self.action_space = spaces.Box(-action_high, action_high)
    self.observation_space = spaces.Box(-observation_high, observation_high)
    self.viewer = None

  def _reset(self):
    #print("KukaGymEnv _reset")
    self.terminated = 0
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    texUid = p.loadTexture("./cube_new/aaa.png")
    cube_objects = p.loadSDF("./cube_new/model.sdf")
    self.cubeId = cube_objects[0]
    p.changeVisualShape(cube_objects[0], -1,rgbaColor =[1,1,1,1])
    p.changeVisualShape(cube_objects[0], -1, textureUniqueId = texUid)
    p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])
    self.tray = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"tray/tray.urdf"), 0.640000,0.075000,-0.190000,0.000000,0.000000,1.000000,0.000000)
    self.table = p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-0.820000,0.000000,0.000000,0.0,1.0)
    #print("table:::",self.table)
    #print("tray:::",self.tray)
 
    #This is where the object spawn at random i should adjust this so it spawn at random position with specific area bound if reward increased
       
    xpos = 0.58
    ypos = 0.04
    ang = 3.14*0.5
    orn = p.getQuaternionFromEuler([0,0,ang])
    self.blockUid =p.loadURDF(os.path.join(self._urdfRoot,"block.urdf"), xpos,ypos,-0.15,orn[0],orn[1],orn[2],orn[3])


    #This where robot is reset i should modefy this to adjust height of end-effector
    p.setGravity(0,0,-10)
    self._kuka_hand = Kuka_Handlit(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self._envStepCounter = 0
    p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array([self._observation])

  def __del__(self):
    p.disconnect()

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getExtendedObservation(self): # mamad : adds  {2d position and angle of object relative to gripper} to observation obtained from kuka.py file
    self._observation = self._kuka_hand.getObservation()
    hand_el = self._kuka_hand.modelInfo.get_hand_links()
    self.tips_eeIndex = []
    for linkName in hand_el:
      if linkName == "lh_ffdistal":
        linkinfo = self._kuka_hand.jointInfo.searchBy(key="linkName",value = linkName)[0]
        self.tips_eeIndex.append(linkinfo["jointIndex"])
      elif linkName == "lh_mfdistal":
        linkinfo = self._kuka_hand.jointInfo.searchBy(key="linkName",value = linkName)[0]
        self.tips_eeIndex.append(linkinfo["jointIndex"])
      elif linkName == "lh_rfdistal":
        linkinfo = self._kuka_hand.jointInfo.searchBy(key="linkName",value = linkName)[0]
        self.tips_eeIndex.append(linkinfo["jointIndex"])
      elif linkName == "lh_thdistal":
        linkinfo = self._kuka_hand.jointInfo.searchBy(key="linkName",value = linkName)[0]
        self.tips_eeIndex.append(linkinfo["jointIndex"])
    kuka_EE_State  = p.getLinkState(self._kuka_hand.kuka_handId,self._kuka_hand.kukaEndEffectorIndex)
    fingertip_state1 = p.getLinkState(self._kuka_hand.kuka_handId, self.tips_eeIndex[0])
    print(self.tips_eeIndex[0])
    fingertip_state2 = p.getLinkState(self._kuka_hand.kuka_handId, self.tips_eeIndex[1])
    fingertip_state3 = p.getLinkState(self._kuka_hand.kuka_handId, self.tips_eeIndex[2])
    fingertip_state4 = p.getLinkState(self._kuka_hand.kuka_handId, self.tips_eeIndex[3])
    kuka_EE_Pos = kuka_EE_State[0]
    kuka_EE_Orn = kuka_EE_State[1]
    finger_EE_pos1 = fingertip_state1[0]
    finger_EE_pos2 = fingertip_state2[0]
    finger_EE_pos3 = fingertip_state3[0]
    finger_EE_pos4 = fingertip_state4[0]
    finger_EE_orn1 = fingertip_state1[1]
    finger_EE_orn2 = fingertip_state2[1]
    finger_EE_orn3 = fingertip_state3[1]
    finger_EE_orn4 = fingertip_state4[1]
    self.blockPos,blockOrn = p.getBasePositionAndOrientation(self.blockUid)
    invGripperPos,invGripperOrn = p.invertTransform(kuka_EE_Pos,kuka_EE_Orn)
    invfingertipPos1, invfingertipOrn1 = p.invertTransform(finger_EE_pos1,finger_EE_orn1)
    invfingertipPos2, invfingertipOrn2 = p.invertTransform(finger_EE_pos2,finger_EE_orn2)
    invfingertipPos3, invfingertipOrn3 = p.invertTransform(finger_EE_pos3,finger_EE_orn3)
    invfingertipPos4, invfingertipOrn4 = p.invertTransform(finger_EE_pos4,finger_EE_orn4)
    gripperMat = p.getMatrixFromQuaternion(kuka_EE_Orn)
    fingertipMat1 = p.getMatrixFromQuaternion(finger_EE_orn1)
    fingertipMat2 = p.getMatrixFromQuaternion(finger_EE_orn2)
    fingertipMat3 = p.getMatrixFromQuaternion(finger_EE_orn3)
    fingertipMat4 = p.getMatrixFromQuaternion(finger_EE_orn4)
    dir0 = [gripperMat[0],gripperMat[3],gripperMat[6]] #mamad: used for debugging 
    dir1 = [gripperMat[1],gripperMat[4],gripperMat[7]] #mamad: used for debugging 
    dir2 = [gripperMat[2],gripperMat[5],gripperMat[8]] #mamad: used for debugging 

    gripperEul =  p.getEulerFromQuaternion(kuka_EE_Orn)
    #print("gripperEul")
    #print(gripperEul)
    blockPosInkuka_EE,blockOrnInkuka_EE = p.multiplyTransforms(invGripperPos,invGripperOrn,self.blockPos,blockOrn)
    blockPosInfingertip_EE1, blockOrnInfingertip_EE1 = p.multiplyTransforms(invfingertipPos1,invfingertipOrn1,self.blockPos,blockOrn)
    blockPosInfingertip_EE2, blockOrnInfingertip_EE2 = p.multiplyTransforms(invfingertipPos2,invfingertipOrn2,self.blockPos,blockOrn)
    blockPosInfingertip_EE3, blockOrnInfingertip_EE3 = p.multiplyTransforms(invfingertipPos3,invfingertipOrn3,self.blockPos,blockOrn)
    blockPosInfingertip_EE4, blockOrnInfingertip_EE4 = p.multiplyTransforms(invfingertipPos4,invfingertipOrn4,self.blockPos,blockOrn)
    #print("blockPosInkuka_EE::::",blockPosInkuka_EE)
    #print("blockOrnInkuka_EE::::",blockOrnInkuka_EE)
    projectedBlockPos2D =[blockPosInkuka_EE[0],blockPosInkuka_EE[1]]
    blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInkuka_EE)
    #print("projectedBlockPos2D")
    #print(projectedBlockPos2D)
    #print("blockEulerInGripper")
    #print(blockEulerInGripper)
    self.dist = math.sqrt(blockPosInkuka_EE[0]**2+blockPosInkuka_EE[1]**2+blockPosInkuka_EE[2]**2)
    self.dist2 = math.sqrt(blockPosInfingertip_EE1[0]**2+blockPosInfingertip_EE1[1]**2+blockPosInfingertip_EE1[2]**2)
    self.dist3 = math.sqrt(blockPosInfingertip_EE2[0]**2+blockPosInfingertip_EE2[1]**2+blockPosInfingertip_EE2[2]**2)
    self.dist4 = math.sqrt(blockPosInfingertip_EE3[0]**2+blockPosInfingertip_EE3[1]**2+blockPosInfingertip_EE3[2]**2)
    self.dist5 = math.sqrt(blockPosInfingertip_EE4[0]**2+blockPosInfingertip_EE4[1]**2+blockPosInfingertip_EE4[2]**2)
    #print(self.dist)
    #print(self.dist)
    #print(self.dist2)
    #print(self.dist3)
    #print(self.dist4)
    #print(self.dist5)
    #we return the relative x,y position and euler angle of block in gripper space
    blockInGripperPosXYEulZ =[blockPosInkuka_EE[0],blockPosInkuka_EE[1],blockEulerInGripper[2]]

    #p.addUserDebugLine(kuka_EE_Pos,[kuka_EE_Pos[0]+dir0[0],kuka_EE_Pos[1]+dir0[1],kuka_EE_Pos[2]+dir0[2]],[1,0,0],lifeTime=1)
    #p.addUserDebugLine(kuka_EE_Pos,[kuka_EE_Pos[0]+dir1[0],kuka_EE_Pos[1]+dir1[1],kuka_EE_Pos[2]+dir1[2]],[0,1,0],lifeTime=1)
    #p.addUserDebugLine(kuka_EE_Pos,[kuka_EE_Pos[0]+dir2[0],kuka_EE_Pos[1]+dir2[1],kuka_EE_Pos[2]+dir2[2]],[0,0,1],lifeTime=1)

    self._observation.extend(list(blockInGripperPosXYEulZ))
    
    return self._observation

  def _step(self, action):
    action = action[0]
    print("GYM Motorcommand",action)
    #print("Debug::action",action)
    #print("Debug::len(action)",len(action))
    #print("Debug::num_Active_joint",self._kuka_hand.num_Active_joint)

  
    #print("kukaGymEnv action=",action)
    dv = 0.005 # this value scales the action comming from the network. You can scale up or down depending on your application and netowrk upper and lower limit
    

   
    
    realAction = action
    return self.step2( realAction)

  def step2(self, action):
    for i in range(self._actionRepeat):
      self._kuka_hand.applyAction(action)
      p.stepSimulation()
      if self._termination():
        break
      self._envStepCounter += 1
    if self._renders:
      time.sleep(self._timeStep)
    self._observation = self.getExtendedObservation()

    #print("self._envStepCounter")
    #print(self._envStepCounter)

    done = self._termination()

    #print("actionCost")
    #print(actionCost)
    reward = self._reward()
    #print("reward")
    #print(reward)

    #print("len=%r" % len(self._observation))

    return np.array([self._observation]), reward, done, {}

  def _render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])
    
    base_pos,orn = self._p.getBasePositionAndOrientation(self._kuka_hand.kuka_handId)
    view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=self._cam_dist,
        yaw=self._cam_yaw,
        pitch=self._cam_pitch,
        roll=0,
        upAxisIndex=2)
    proj_matrix = self._p.computeProjectionMatrixFOV(
        fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
        nearVal=0.1, farVal=100.0)
    (_, _, px, _, _) = self._p.getCameraImage(
        width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
        projectionMatrix=proj_matrix, renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        #renderer=self._p.ER_TINY_RENDERER)

        
    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))
		
    rgb_array = rgb_array[:, :, :3]
    return rgb_array


  def _termination(self):
    #print (self._kuka_hand.endEffectorPos[2])
    state = p.getLinkState(self._kuka_hand.kuka_handId,self._kuka_hand.kukaEndEffectorIndex)
    actualEndEffectorPos = state[0]

    #print("self._envStepCounter")
    #print(s
    # self._envStepCounter)
    if (self.terminated or self._envStepCounter>self._maxSteps):
      self._observation = self.getExtendedObservation()
      return [True]
    maxDist = 0.005
    closestPoints = p.getClosestPoints(self.tray, self._kuka_hand.kuka_handId,maxDist)

    if (len(closestPoints)):#(actualEndEffectorPos[2] <= -0.43):
      self.terminated = 1

      #print("terminating, closing gripper, attempting grasp")
      #start grasp and terminate
      fingerAngle = 0.3
      for i in range (100):
        graspAction = [0,0,0.0001,0,fingerAngle]
        self._kuka_hand.applyAction(graspAction)
        p.stepSimulation()
        fingerAngle = fingerAngle-(0.3/100.)
        if (fingerAngle<0):
          fingerAngle=0

      for i in range (1000):
        graspAction = [0,0,0.001,0,fingerAngle]
        self._kuka_hand.applyAction(graspAction)
        p.stepSimulation()
        self.blockPos,blockOrn=p.getBasePositionAndOrientation(self.blockUid)
        if (self.blockPos[2] > 0.23):
          #print("BLOCKPOS!")
          #print(self.blockPos[2])
          break
        state = p.getLinkState(self._kuka_hand.kuka_handId,self._kuka_hand.kukaEndEffectorIndex)
        actualEndEffectorPos = state[0]
        if (actualEndEffectorPos[2]>0.5):
          break


      self._observation = self.getExtendedObservation()
      return [True]
    return [False]

  def _reward(self):
    """
    positive reward:
    reward the agent if it gets closer to target endeffector
    reward agent if picked up object
    reward agent for getting each finger as close as possible to object surface
    negative reward:
    punish agent for self collision
    punish aget for collsion wit table 
    """
    reward = -1000
    if (self.dist < 0.5) and (self.dist2 < 0.3) and (self.dist3 <0.3) and (self.dist4 <0.3) and (self.dist5 <0.3):
      reward += 100000
    elif (self.dist <1) and (self.dist2 < 0.6) and (self.dist3 <0.6) and (self.dist4 <0.6) and (self.dist5 <0.6):
      reward += 8000
    elif (self.dist < 0.5): 
      reward += 5000
    else:
      reward = -(self.dist + self.dist2 +self.dist3 + self.dist4 +self.dist5) * 10
    contactPoints = self.check_if_body_collision_has_happend()
    contact = self.check_if_collsion_with_table_has_happend()
    if contactPoints:
      reward += -2000
    if contact:
      reward += -3000
    #print(reward)
    #print(self.dist)
    return reward
    #print(reward)

    '''
    self.check_if_body_collision_has_happend()
    self. check_if_collsion_with_table_has_happend()
    #rewards is height of target object
    self.blockPos,blockOrn=p.getBasePositionAndOrientation(self.blockUid)
    closestPoints = p.getClosestPoints(self.blockUid,self._kuka_hand.kuka_handId,1000, -1, self._kuka_hand.kukaEndEffectorIndex)
    #print(closestPoints)
    reward = -1000

    numPt = len(closestPoints)
    #print(numPt)
    if (numPt>0):
      #print("reward:")
      #print(closestPoints[0][8])
      reward = -closestPoints[0][8]*10
    #print(reward)
    if (self.blockPos[2] >0.2):
      reward = reward+10000
      #print("successfully grasped a block!!!")
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #print("reward")
      #print(reward)
     #print("reward")
    #print(reward)
    return [reward]
    '''
  def render_sim(self):
    cid = p.connect(p.SHARED_MEMORY)
    if (cid<0):
      cid = p.connect(p.GUI)
  if parse_version(gym.__version__)>=parse_version('0.9.6'):
    render = _render
    reset  = _reset
    seed   = _seed
    step   = _step


  #utility function
  def generate_random_action(self):
    def generate_action(limits):
      return random.uniform(limits[0],limits[1])
    def generate_random_action_hand():
      hand_commands = []
      #getting links with active joints
      hand_links = self._kuka_hand.modelInfo.get_hand_links()
      #print("links for the hand::::",hand_links)

      hand_links_info = []
    
      for link_name in hand_links:
        link_info = self._kuka_hand.modelInfo.searchBy(key="link",value=link_name)
        #print(link_info)
        hand_links_info.append(link_info)
      #print(hand_links_info)
      hand_links_info_with_active_joints = []
      for hand_link_info in hand_links_info:
        if hand_link_info["joint"]["j_type"] != "fixed":
          hand_links_info_with_active_joints.append(hand_link_info)
      hand_indexOf_active_joints =[]
      for Link in hand_links_info_with_active_joints:
        link = self._kuka_hand.jointInfo.searchBy("linkName",Link["link_name"])[0]
        hand_indexOf_active_joints.append(link["jointIndex"])
      #print(hand_indexOf_active_joints)
      #print("hand_active_joints",len(hand_indexOf_active_joints))
      #getting joint limit for 
      for jointIndex in hand_indexOf_active_joints:
        jointinfo = self._kuka_hand.jointInfo.searchBy("jointIndex",jointIndex)[0]
        joint_ll = jointinfo["jointLowerLimit"]
        joint_ul = jointinfo["jointUpperLimit"]
        joint_limits = [joint_ll,joint_ul]
        #print("1111111",joint_limits)
        hand_commands.append(generate_action(joint_limits))
      #print(hand_commands)
      return hand_commands
    def generate_random_action_kuka():
      #todo:
      #-generate a point close to the block
      random_list = []
      kuka_close_point = []
      kuka_limits = []
      limitation = [2, 2, 2]
      for i in range(3):
        random_list.append(random.uniform(0,1))
      for j in range(3):
        kuka_limits.append(self.blockPos[j] + limitation[j]+random_list[j])
      kuka_euler = [0, -3.1415, 0]
      kuka_commands = kuka_limits + kuka_euler
      return kuka_commands
    hand_command = generate_random_action_hand()
    print("111111",hand_command)
    print("gym hand_commands len ",len(hand_command))
    kuka_command = generate_random_action_kuka()
    print("gym kuka_commands len ",len(kuka_command))
    return hand_command + kuka_command
  def check_if_body_collision_has_happend(self):
    hand_el = self._kuka_hand.modelInfo.get_hand_links()
    self.tips_eeIndex = []
    for linkName in hand_el:
      if linkName == "distal_FF":
        linkinfo = self._kuka_hand.jointInfo.searchBy(key="linkName",value = linkName)[0]
        self.tips_eeIndex.append(linkinfo["jointIndex"])
      elif linkName == "distal_MF":
        linkinfo = self._kuka_hand.jointInfo.searchBy(key="linkName",value = linkName)[0]
        self.tips_eeIndex.append(linkinfo["jointIndex"])
      elif linkName == "distal_RF":
        linkinfo = self._kuka_hand.jointInfo.searchBy(key="linkName",value = linkName)[0]
        self.tips_eeIndex.append(linkinfo["jointIndex"])
      elif linkName == "thdistal":
        linkinfo = self._kuka_hand.jointInfo.searchBy(key="linkName",value = linkName)[0]
        self.tips_eeIndex.append(linkinfo["jointIndex"])
    #print(self.tips_eeIndex)
    kuka_link = self._kuka_hand.modelInfo.get_kuka_links()
    #robot_info = self._kuka_hand.jointInfo.show_infoForAll_joints()
    #print(kuka_link)
    jointIndexs = []
    qIndexs = []
    uIndexs = []
    for link_name in kuka_link:
      if link_name == "lbr_iiwa_link_0":
        continue
      linkInfo = self._kuka_hand.jointInfo.searchBy(key="linkName",value = link_name)[0]
      jointIndexs.append(linkInfo["jointIndex"])
      qIndexs.append(linkInfo["qIndex"])
      uIndexs.append(linkInfo["uIndex"])
    #print(jointIndexs)
    #print(qIndexs)
    #print(uIndexs)
    #print("kuka links::",kuka_link)
    contact = []
    contactPoints=[]
    for i in range(len(jointIndexs)):
      contact = p.getContactPoints(self._kuka_hand.kuka_handId,jointIndexs[i])
      contactPoints.append(contact)
    #print(contactPoints)
    return contactPoints
  def check_if_collsion_with_table_has_happend(self):
    contact = p.getContactPoints(self._kuka_hand.kuka_handId,self.table)
    if contact:
      return True
    else:
      return False
    #print(contact)


class Unit_test_Kuka_HandlitGymEnv():
  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=False,
               maxSteps = 1000):
    self.env = Kuka_HandlitGymEnv(renders=True, isDiscrete=False)
    self.blockUid = self.env.blockUid
    self.table = self.env.table
  def check_collison_table(self):
    blockPos,blockOrn = p.getBasePositionAndOrientation(self.blockUid)
    tablePos,tableOrn = p.getBasePositionAndOrientation(self.table)
    #print(self.blockPos)
    #print(tablePos)
    col_target = [0,0,0]
    col_target [0] = blockPos[0]
    col_target [1] = blockPos[1]
    col_target [2] = tablePos[2] -0.5
    #print(col_target)
    kuka_commad = ()




    #collison_action = []
    #performing command that causes collsion
    #self.env.step(collison_action)

    #Checking for collsion1.55718838229

    test_result = self.env.check_if_collsion_with_table_has_happend()

    #expecting true for collision
    if test_result == True:
      return True
    else:
      while True:
        print("1") 
      #return False



'''
Todo:
[]Finish collision detection
Deadline:18Jun19
[]Finish the Reward function
Deadline:19Jun19
[]check if collision detection works
[]
'''