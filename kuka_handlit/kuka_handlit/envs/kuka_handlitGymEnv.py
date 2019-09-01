import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)
from pkg_resources import resource_string,resource_filename

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
from mamad_util import JointInfo
from AW import AdaptiveWorkspace
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
               kuka_inverse_kinematic = False,
               renders=False,
               isDiscrete=False,
               maxSteps = 1000,
               observation_frame = "palm_world_xyz"
               ):
 
    self._kuka_inverse_kinematic = kuka_inverse_kinematic
    self._isDiscrete = isDiscrete
    self._timeStep = 1./500.
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
    self._observation_frames = ["kuka_world","kuka_world_xyz","kuka","kuka_xyz","palm_world","palm_world_xyz","palm","palm_xyz"]
    self._observation_frame = observation_frame
    self._p = p
    self._AW = AdaptiveWorkspace([0.5000000,0.60000,1.500],0.1,[0.2,0.2,0.2],0.5,1)
    if self._renders:
      cid = p.connect(p.SHARED_MEMORY)
      if (cid<0):
         cid = p.connect(p.GUI)
      p.resetDebugVisualizerCamera(1.3,180,-41,[0.52,-0.2,-0.33])
    else:
      p.connect(p.DIRECT)
   
    self._seed()
    self.reset()
    self.observationDim = len(self.getExtendedObservation())


    self.observation_high = np.array([1, 1, 1.4, 6.28, 6.28, 6.28, 1, 1, 1.4, 1.5708, 1, 1, 1.4, 1.5708, 1, 1, 1.4, 1.5708, 1, 1, 1.4, 1.5708])
    self.observation_low = np.array([-1, -1, -0.3 ,-6.28, -6.28, 6.28, -1, -1, -0.3, 0, -1, -1, -0.3, 0, -1, -1, -0.3, 0, -1, -1, -0.3, 0])
    
    action_dim = self._kuka_hand.num_Active_joint
    self._action_bound = 1
    self.action_high = np.array([2.96706, 2.0944, 2.96706, 2.0944, 2.96706, 2.0944, 3.05433, 1.5708, 1.5708, 1.5708, 0.349066, 1.5708, 1.5708, 1.5708, 0.349066, 1.5708, 1.5708, 1.5708, 0.349066, 1.0472, 1.22173, 0.698132, 1.5708] )
    self.action_low = np.array([-2.96706, -2.0944, -2.96706, -2.0944, -2.96706, -2.0944, -3.05433, 0, 0, 0, -0.349066, 0, 0, 0, -0.349066, 0, 0, 0, -0.349066, -1.0472, 0, -0.698132, 0])
    self.action_space = spaces.Box(self.action_low, self.action_high)
    self.observation_space = spaces.Box(self.observation_low, self.observation_high)
    self.viewer = None

  def reset(self):

    self.terminated = 0
    p.resetSimulation()
    # p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.setRealTimeSimulation(0)
    cube_texture_path = resource_filename(__name__,"cube_new/aaa.png")
    cube_path = resource_filename(__name__,"cube_new/model.sdf")
    texUid = p.loadTexture(cube_texture_path)
    cube_objects = p.loadSDF(cube_path)
    self.cubeId = cube_objects[0]
    p.changeVisualShape(cube_objects[0], -1,rgbaColor =[1,1,1,1])
    p.changeVisualShape(cube_objects[0], -1, textureUniqueId = texUid)
    p.resetBasePositionAndOrientation(cube_objects[0],(0.5000000,0.60000,0.64000),(0.717,0.0,0.0,0.717))
    p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,0])
  
    self.table = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"table/table.urdf"), 0.5000000,0.60000,0.0000,0.000000,0.000000,0.0,1.0)



       

    xpos = 0.5
    ypos = 0.8
    zpos = 1
    ang = 3.14*0.5
    orn = p.getQuaternionFromEuler([0,0,ang])
    self.blockUid = 2

    p.setGravity(0,0,-10)
    ee_xyz = self._AW.pick_random_ee_position_in_AW()
    self._kuka_hand = Kuka_Handlit(ee_xyz,kuka_loc =[0.00000,0.200000,0.65000,0.000000,0.000000,0.000000,1.000000] ,urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
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
    def obs_distAndOrn_relativeToKukaAndFingers(worldFrame=True):  
      """
      if worldFrame=True
        all the outputs are relative to world frame
        outout:[kuka_ee_dsit,
                FF_dist,
                MF_dist,
                RF_dist,
                TH_dist,
                block_orn,
                kuka_ee_orn
                ]
      else:
        outout:[kuka_ee_dsit_kuka_eeFrame,
                FF_dist_FFFrame,
                MF_dist_MFFrame,
                RF_dist_RFFrame,
                TH_dist_THFrame,
                block_orn_kuka_eeFrame,
                ]
      """
      robot_state = self._kuka_hand.getObservation()
      #block state
      blockPos,blockOrn = p.getBasePositionAndOrientation(self.cubeId)

      fingers_dist_worldFrame =[]
      fingers_dist_fingerFrame =[]

      kuka_EE_State  = robot_state[:7]#xyz and quatrenium 
      kuka_EE_Pos = kuka_EE_State[:3]
      kuka_EE_Orn = kuka_EE_State[3:]

      #Distance of kuka_ee to obj in world frame
      kuka_dist_x = blockPos[0] - kuka_EE_Pos[0]
      kuka_dist_y = blockPos[1] - kuka_EE_Pos[1]
      kuka_dist_z = blockPos[2] - kuka_EE_Pos[2]

      # size of vector for kuka_ee to obj in world frame
      kuka_dist_worldFrame = math.sqrt(kuka_dist_x**2+ kuka_dist_y**2+ kuka_dist_z**2)

      invKuka_eePos,invKuka_eeOrn = p.invertTransform(kuka_EE_Pos,kuka_EE_Orn)
      blockPosInkuka_EE,blockOrnInkuka_EE = p.multiplyTransforms(invKuka_eePos,invKuka_eeOrn,blockPos,blockOrn)
      # size of vector for kuka_ee to obj in world frame
      kuka_dist_kukaFrame = math.sqrt(blockPosInkuka_EE[0]**2+ blockPosInkuka_EE[1]**2+blockPosInkuka_EE[2]**2)

      state_counter = 1
      for linkName in self._kuka_hand.get_fingerTips_linkIndex():
        state_counter +=1
        fingerState = robot_state[7*(state_counter-1):7*state_counter]#xyz and quatrenium 
        finger_EE_pos = fingerState[:3]
        finger_EE_orn = fingerState[3:]

        #Distance of finger_ee to obj in world frame
        finger_dist_x = blockPos[0] - finger_EE_pos[0]
        finger_dist_y = blockPos[1] - finger_EE_pos[1]
        finger_dist_z = blockPos[2] - finger_EE_pos[2]

        # size of vector for finger_ee to obj in world frame
        finger_dist_worldFrame = math.sqrt(finger_dist_x**2+ finger_dist_y**2+ finger_dist_z**2)

        invfingertipPos, invfingertipOrn = p.invertTransform(finger_EE_pos,finger_EE_orn)
        blockPosInfingertip_EE, blockOrnInfingertip_EE = p.multiplyTransforms(invfingertipPos,invfingertipOrn,blockPos,blockOrn)

        # size of vector for finger_ee to obj in world frame
        finger_dist_fingerFrame = math.sqrt(blockPosInfingertip_EE[0]**2+ blockPosInfingertip_EE[1]**2+blockPosInfingertip_EE[2]**2)

        fingers_dist_worldFrame.append(finger_dist_worldFrame)
        fingers_dist_fingerFrame.append(finger_dist_fingerFrame)

      robot_dist_worldFrame = [kuka_dist_worldFrame]+fingers_dist_worldFrame
      robot_dist_EEsFrame   = [kuka_dist_kukaFrame ]+fingers_dist_fingerFrame

      #kuka_ee orn
      kuka_EE_Orn = p.getEulerFromQuaternion(kuka_EE_Orn)
 
      
      #dists cube orn and palm orn 
      block_orn = p.getEulerFromQuaternion(blockOrn)
      obs_worldFrame = robot_dist_worldFrame+list(block_orn)+list(kuka_EE_Orn)

      obs_kukaFrame = robot_dist_EEsFrame + list(blockOrnInkuka_EE)
      if worldFrame ==True:
        return obs_worldFrame
      else:
        return obs_kukaFrame  


    def obs_distAndOrn_relativeToKukaAndFingers_xyz(worldFrame=True):  
      """
      if worldFrame=True
        all the outputs are relative to world frame
        outout:[kuka_ee_dsit_x,
                kuka_ee_dsit_y,
                kuka_ee_dsit_z,
                FF_dist_x,
                FF_dist_y,
                FF_dist_z,
                MF_dist_x,
                MF_dist_y,
                MF_dist_z,
                RF_dist_x,
                RF_dist_y,
                RF_dist_z,
                TH_dist_x,
                TH_dist_y,
                TH_dist_z,
                block_orn,
                kuka_ee_orn
                ]
      else:
        outout:[kuka_ee_dsit_x_kuka_eeFrame,
                kuka_ee_dsit_y_kuka_eeFrame,
                kuka_ee_dsit_z_kuka_eeFrame,
                FF_dist_x_FFFrame,
                FF_dist_y_FFFrame,
                FF_dist_z_FFFrame,
                MF_dist_x_MFFrame,
                MF_dist_y_MFFrame,
                MF_dist_z_MFFrame,
                RF_dist_x_RFFrame,
                RF_dist_y_RFFrame,
                RF_dist_z_RFFrame,
                TH_dist_x_THFrame,
                TH_dist_y_THFrame,
                TH_dist_z_THFrame,
                block_orn_kuka_eeFrame
                ]
      """
      robot_state = self._kuka_hand.getObservation()
      #block state
      blockPos,blockOrn = p.getBasePositionAndOrientation(self.cubeId)

      fingers_dist_worldFrame =[]
      fingers_dist_fingerFrame =[]

      kuka_EE_State  = robot_state[:7]#xyz and quatrenium 
      kuka_EE_Pos = kuka_EE_State[:3]
      kuka_EE_Orn = kuka_EE_State[3:]

      #Distance of kuka_ee to obj in world frame
      kuka_dist_x = blockPos[0] - kuka_EE_Pos[0]
      kuka_dist_y = blockPos[1] - kuka_EE_Pos[1]
      kuka_dist_z = blockPos[2] - kuka_EE_Pos[2]

      # size of vector for kuka_ee to obj in world frame
      kuka_dist_worldFrame = [kuka_dist_x, kuka_dist_y, kuka_dist_z]
      kuka_dist_worldFrame = [abs(i) for i in kuka_dist_worldFrame]

      invKuka_eePos,invKuka_eeOrn = p.invertTransform(kuka_EE_Pos,kuka_EE_Orn)
      blockPosInkuka_EE,blockOrnInkuka_EE = p.multiplyTransforms(invKuka_eePos,invKuka_eeOrn,blockPos,blockOrn)
      # size of vector for kuka_ee to obj in kuka frame
      kuka_dist_kukaFrame = [blockPosInkuka_EE[0],blockPosInkuka_EE[1],blockPosInkuka_EE[2]]
      kuka_dist_kukaFrame = [abs(i) for i in kuka_dist_kukaFrame]

      state_counter = 1
      for linkName in self._kuka_hand.get_fingerTips_linkIndex():
        state_counter +=1
        fingerState = robot_state[7*(state_counter-1):7*state_counter]#xyz and quatrenium 
        finger_EE_pos = fingerState[:3]
        finger_EE_orn = fingerState[3:]

        #Distance of finger_ee to obj in world frame
        finger_dist_x = blockPos[0] - finger_EE_Pos[0]
        finger_dist_y = blockPos[1] - finger_EE_Pos[1]
        finger_dist_z = blockPos[2] - finger_EE_Pos[2]

        # size of vector for finger_ee to obj in world frame
        finger_dist_worldFrame = [finger_dist_x, finger_dist_y, finger_dist_z]
        finger_dist_worldFrame = [abs(i) for i in finger_dist_worldFrame]

        invfingertipPos, invfingertipOrn = p.invertTransform(finger_EE_pos,finger_EE_orn)
        blockPosInfingertip_EE, blockOrnInfingertip_EE = p.multiplyTransforms(invfingertipPos,invfingertipOrn,blockPos,blockOrn)

        # size of vector for finger_ee to obj in world frame
        finger_dist_fingerFrame = [blockPosInfingertip_EE[0], blockPosInfingertip_EE[1],blockPosInfingertip_EE[2]]
        finger_dist_fingerFrame = [abs(i) for i in finger_dist_fingerFrame]

        fingers_dist_worldFrame.extend(finger_dist_worldFrame)
        fingers_dist_fingerFrame.extend(finger_dist_fingerFrame)

      robot_dist_worldFrame =kuka_dist_worldFrame+fingers_dist_worldFrame
      robot_dist_EEsFrame =kuka_dist_kukaFrame+fingers_dist_fingerFrame

      #kuka_ee orn
      kuka_EE_Orn = p.getEulerFromQuaternion(kuka_EE_Orn)
 
      
      #dists cube orn and palm orn 
      block_orn = p.getEulerFromQuaternion(block_orn)
      obs_worldFrame = robot_dist_worldFrame+blockOrn+kuka_EE_Orn

      obs_kukaFrame = robot_dist_EEsFrame + blockOrnInkuka_EE
      if worldFrame ==True:
        return obs_worldFrame
      else:
        return obs_kukaFrame  


    def obs_distAndOrn_relativeToPalmAndFingers(worldFrame=True):  
      """
      if worldFrame=True
        all the outputs are relative to world frame
        outout:[palm_dsit,
                FF_dist,
                MF_dist,
                RF_dist,
                TH_dist,
                block_orn,
                palm_orn
                ]
      else:
        outout:[palm_dsit_palmFrame,
                FF_dist_FFFrame,
                MF_dist_MFFrame,
                RF_dist_RFFrame,
                TH_dist_THFrame,
                block_orn_palmFrame,
                ]
      """
      robot_state = self._kuka_hand.getObservation()
      #block state
      blockPos,blockOrn = p.getBasePositionAndOrientation(self.cubeId)

      fingers_dist_worldFrame =[]
      fingers_dist_fingerFrame =[]

      palm_EE_State  = self._kuka_hand.getObservation_palm()#xyz and quatrenium 
      palm_EE_Pos = palm_EE_State[:3]
      palm_EE_Orn = palm_EE_State[3:]

      #Distance of palm_ee to obj in world frame
      palm_dist_x = blockPos[0] - palm_EE_Pos[0]
      palm_dist_y = blockPos[1] - palm_EE_Pos[1]
      palm_dist_z = blockPos[2] - palm_EE_Pos[2]

      # size of vector for palm_ee to obj in world frame
      palm_dist_worldFrame = math.sqrt(palm_dist_x**2+ palm_dist_y**2+ palm_dist_z**2)

      invpalm_eePos,invpalm_eeOrn = p.invertTransform(palm_EE_Pos,palm_EE_Orn)
      blockPosInpalm_EE,blockOrnInpalm_EE = p.multiplyTransforms(invpalm_eePos,invpalm_eeOrn,blockPos,blockOrn)
      # size of vector for palm_ee to obj in world frame
      palm_dist_palmFrame = math.sqrt(blockPosInpalm_EE[0]**2+ blockPosInpalm_EE[1]**2+blockPosInpalm_EE[2]**2)

      state_counter = 1
      for linkName in self._kuka_hand.get_fingerTips_linkIndex():
        state_counter +=1
        fingerState = robot_state[7*(state_counter-1):7*state_counter]#xyz and quatrenium 
       
        finger_EE_pos = fingerState[:3]
        finger_EE_orn = fingerState[3:]
   
        #Distance of finger_ee to obj in world frame
        finger_dist_x = blockPos[0] - finger_EE_pos[0]
        finger_dist_y = blockPos[1] - finger_EE_pos[1]
        finger_dist_z = blockPos[2] - finger_EE_pos[2]

        # size of vector for finger_ee to obj in world frame
        finger_dist_worldFrame = math.sqrt(finger_dist_x**2+ finger_dist_y**2+ finger_dist_z**2)

        invfingertipPos, invfingertipOrn = p.invertTransform(finger_EE_pos,finger_EE_orn)
        blockPosInfingertip_EE, blockOrnInfingertip_EE = p.multiplyTransforms(invfingertipPos,invfingertipOrn,blockPos,blockOrn)

        # size of vector for finger_ee to obj in world frame
        finger_dist_fingerFrame = math.sqrt(blockPosInfingertip_EE[0]**2+ blockPosInfingertip_EE[1]**2+blockPosInfingertip_EE[2]**2)

        fingers_dist_worldFrame.append(finger_dist_worldFrame)
        fingers_dist_fingerFrame.append(finger_dist_fingerFrame)

      robot_dist_worldFrame =[palm_dist_worldFrame]+fingers_dist_worldFrame
      robot_dist_EEsFrame =[palm_dist_palmFrame]+fingers_dist_fingerFrame

      #palm orn
     
      palm_EE_Orn = p.getEulerFromQuaternion(palm_EE_Orn)
      
      #dists cube orn and palm orn 
      blockOrn = p.getEulerFromQuaternion(blockOrn)
      obs_worldFrame = robot_dist_worldFrame+list(blockOrn)+list(palm_EE_Orn)
  
      
      blockOrnInpalm_EE = p.getEulerFromQuaternion(blockOrnInpalm_EE)
      obs_palmFrame = robot_dist_EEsFrame + list(blockOrnInpalm_EE)
      if worldFrame ==True:
        return obs_worldFrame
      else:
        return obs_palmFrame
    
    def obs_distAndOrn_relativeToPalmAndFingers_xyz(worldFrame=True):  
      """
      if worldFrame=True
        all the outputs are relative to world frame
        outout:[palm_dsit_x,
                palm_dsit_y,
                palm_dsit_z,
                FF_dist_x,
                FF_dist_y,
                FF_dist_z,
                MF_dist_x,
                MF_dist_y,
                MF_dist_z,
                RF_dist_x,
                RF_dist_y,
                RF_dist_z,
                TH_dist_x,
                TH_dist_y,
                TH_dist_z,
                block_orn,
                palm_orn
                ]
      else:
        outout:[palm_dsit_x_palmFrame,
                palm_dsit_y_palmFrame,
                palm_dsit_z_palmFrame,
                FF_dist_x_FFFrame,
                FF_dist_y_FFFrame,
                FF_dist_z_FFFrame,
                MF_dist_x_MFFrame,
                MF_dist_y_MFFrame,
                MF_dist_z_MFFrame,
                RF_dist_x_RFFrame,
                RF_dist_y_RFFrame,
                RF_dist_z_RFFrame,
                TH_dist_x_THFrame,
                TH_dist_y_THFrame,
                TH_dist_z_THFrame,
                block_orn_palmFrame
                ]
      """
      robot_state = self._kuka_hand.getObservation()
      #block state
      blockPos,blockOrn = p.getBasePositionAndOrientation(self.cubeId)

      fingers_dist_worldFrame =[]
      fingers_dist_fingerFrame =[]

      palm_EE_State  = self._kuka_hand.getObservation_palm()#xyz and quatrenium 
      palm_EE_Pos = palm_EE_State[:3]
      palm_EE_Orn = palm_EE_State[3:]

      #Distance of palm_ee to obj in world frame
      palm_dist_x = blockPos[0] - palm_EE_Pos[0]
      palm_dist_y = blockPos[1] - palm_EE_Pos[1]
      palm_dist_z = blockPos[2] - palm_EE_Pos[2]

      # size of vector for palm_ee to obj in world frame
      palm_dist_worldFrame = [palm_dist_x,palm_dist_y,palm_dist_z]
      palm_dist_worldFrame = [abs(i) for i in palm_dist_worldFrame]

      invpalm_eePos,invpalm_eeOrn = p.invertTransform(palm_EE_Pos,palm_EE_Orn)
      blockPosInpalm_EE,blockOrnInpalm_EE = p.multiplyTransforms(invpalm_eePos,invpalm_eeOrn,blockPos,blockOrn)
      # size of vector for palm_ee to obj in world frame
      palm_dist_palmFrame = [blockPosInpalm_EE[0],blockPosInpalm_EE[1],blockPosInpalm_EE[2]]
      palm_dist_palmFrame = [abs(i) for i in palm_dist_palmFrame]

      state_counter = 1
      for linkName in self._kuka_hand.get_fingerTips_linkIndex():
        state_counter +=1
        fingerState = robot_state[7*(state_counter-1):7*state_counter]#xyz and quatrenium 
       
        finger_EE_pos = fingerState[:3]
        finger_EE_orn = fingerState[3:]
   
        #Distance of finger_ee to obj in world frame
        finger_dist_x = blockPos[0] - finger_EE_pos[0]
        finger_dist_y = blockPos[1] - finger_EE_pos[1]
        finger_dist_z = blockPos[2] - finger_EE_pos[2]

        # size of vector for finger_ee to obj in world frame
        finger_dist_worldFrame = [finger_dist_x, finger_dist_y, finger_dist_z]
        finger_dist_worldFrame = [abs(i) for i in finger_dist_worldFrame]

        invfingertipPos, invfingertipOrn = p.invertTransform(finger_EE_pos,finger_EE_orn)
        blockPosInfingertip_EE, blockOrnInfingertip_EE = p.multiplyTransforms(invfingertipPos,invfingertipOrn,blockPos,blockOrn)

        # size of vector for finger_ee to obj in world frame
        finger_dist_fingerFrame = [blockPosInfingertip_EE[0],blockPosInfingertip_EE[1],blockPosInfingertip_EE[2]]
        finger_dist_fingerFrame = [abs(i) for i in finger_dist_fingerFrame]

        fingers_dist_worldFrame.extend(finger_dist_worldFrame)
        fingers_dist_fingerFrame.extend(finger_dist_fingerFrame)

      robot_dist_worldFrame =palm_dist_worldFrame+fingers_dist_worldFrame
      robot_dist_EEsFrame =palm_dist_palmFrame+fingers_dist_fingerFrame
      # print("palm_dist_worldFrame::",palm_dist_worldFrame)
      # print("fingers_dist_worldFrame::",fingers_dist_worldFrame)

      #palm orn
     
      palm_EE_Orn = p.getEulerFromQuaternion(palm_EE_Orn)
      
      #dists cube orn and palm orn 
      blockOrn = p.getEulerFromQuaternion(blockOrn)
      obs_worldFrame = robot_dist_worldFrame+list(blockOrn)+list(palm_EE_Orn)
      # print("robot_dist_worldFrame::",robot_dist_worldFrame)
      # print("list(blockOrn)::",list(blockOrn))
      # print("list(palm_EE_Orn)::",list(palm_EE_Orn))
      
      blockOrnInpalm_EE = p.getEulerFromQuaternion(blockOrnInpalm_EE)
      obs_palmFrame = robot_dist_EEsFrame + list(blockOrnInpalm_EE)
      if worldFrame ==True:
        return obs_worldFrame
      else:
        return obs_palmFrame
    
    if self._observation_frame == self._observation_frames[0]:
      self._observation = obs_distAndOrn_relativeToKukaAndFingers(worldFrame=True)
    if self._observation_frame == self._observation_frames[1]:
      self._observation = obs_distAndOrn_relativeToKukaAndFingers_xyz(worldFrame=True)
    if self._observation_frame == self._observation_frames[2]:
      self._observation = obs_distAndOrn_relativeToKukaAndFingers(worldFrame=False)
    if self._observation_frame == self._observation_frames[3]:
      self._observation = obs_distAndOrn_relativeToKukaAndFingers_xyz(worldFrame=False)
    if self._observation_frame == self._observation_frames[4]:
      self._observation = obs_distAndOrn_relativeToPalmAndFingers(worldFrame=True)
    if self._observation_frame == self._observation_frames[5]:
      self._observation = obs_distAndOrn_relativeToPalmAndFingers_xyz(worldFrame=True)
    if self._observation_frame == self._observation_frames[6]:
      self._observation = obs_distAndOrn_relativeToPalmAndFingers(worldFrame=False)
    if self._observation_frame == self._observation_frames[7]:
      self._observation = obs_distAndOrn_relativeToPalmAndFingers_xyz(worldFrame=False)
    
    return self._observation

  def step(self, action):
    action = action[0]

    

   
    
    realAction = action
    return self.step2( realAction)

  def step2(self, action):
   
    for i in range(self._actionRepeat):
      if self._kuka_inverse_kinematic == True:
        self._kuka_hand.applyAction(action)
      else:
      
        self._kuka_hand.applyAction_2(action)
      p.stepSimulation()
      if self._termination():
        break
      self._envStepCounter += 1
    if self._renders:
      time.sleep(self._timeStep)
    self._observation = self.getExtendedObservation()

  
    done = self._termination()

    reward = self._reward()
 



    return np.array([self._observation]), reward, done

  def render(self, mode="rgb_array", close=False):
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
       

        
    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

    rgb_array = rgb_array[:, :, :3]
    return rgb_array


  def _termination(self):

    state = p.getLinkState(self._kuka_hand.kuka_handId,self._kuka_hand.kukaEndEffectorIndex)
    actualEndEffectorPos = state[0]


    if (self.terminated or self._envStepCounter>self._maxSteps or 
        self.check_if_collsion_with_table_has_happend() or self._AW.check_if_ee_outside_AW(state)
        ):
      
      return True
    maxDist = 0.005
    closestPoints = p.getClosestPoints(self.table, self._kuka_hand.kuka_handId,maxDist)
    return False
  
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
    worldFrame = [self._observation_frames[0],self._observation_frames[1],self._observation_frames[4],self._observation_frames[5]]
    isWorldFrame =  self._observation_frame in worldFrame

    if isWorldFrame:
      weights = {"kuka_OR_palm":1,"fingers":1}

      kukaEE_OR_palm_dist = self._observation[0]
      FF_dist = self._observation[1]
      MF_dist = self._observation[2]
      RF_dist = self._observation[3]
      TH_dist = self._observation[4]

      reward = -1*(weights["kuka_OR_palm"]*kukaEE_OR_palm_dist+weights["fingers"]*FF_dist+weights["fingers"]*MF_dist+weights["fingers"]*RF_dist+weights["fingers"]*TH_dist)

    if not isWorldFrame:
      weights = {"kuka_OR_palm_x":1,"kuka_OR_palm_y":1,"kuka_OR_palm_z":1,"fingers_x":1,"fingers_y":1,"fingers_z":1}

      kukaEE_OR_palm_dist_x,kukaEE_OR_palm_dist_y,kukaEE_OR_palm_dist_z = self._observation[0:3]
      FF_dist_x,FF_dist_y,FF_dist_z = self._observation[3:6]
      MF_dist_x,MF_dist_y,MF_dist_z = self._observation[6:9]
      RF_dist_x,RF_dist_y,RF_dist_z = self._observation[9:12]
      TH_dist_x,TH_dist_y,TH_dist_z = self._observation[12:15]

      kuka_reward = -1*(weights["kuka_OR_palm_x"]*kukaEE_OR_palm_dist_x+weights["kuka_OR_palm_y"]*kukaEE_OR_palm_dist_y+weights["kuka_OR_palm_z"]*kukaEE_OR_palm_dist_z)
      
      reward_fingers_x = -1*weights["fingers_x"]*(FF_dist_x+MF_dist_x+RF_dist_x+TH_dist_x)
      reward_fingers_y = -1*weights["fingers_y"]*(FF_dist_y+MF_dist_y+RF_dist_y+TH_dist_y)
      reward_fingers_z = -1*weights["fingers_z"]*(FF_dist_z+MF_dist_z+RF_dist_z+TH_dist_z)

      reward = kuka_reward+reward_fingers_x+reward_fingers_y+reward_fingers_z
    
    
    contactPoints = self.check_if_self_collision_has_happend()
    contact = self.check_if_collsion_with_table_has_happend() 
    """
    if contactPoints == True:
      reward += -2000
    if contact ==True:
      reward += -3000
    """
   
    return reward
 

  def render_sim(self):
    cid = p.connect(p.SHARED_MEMORY)
    if (cid<0):
      cid = p.connect(p.GUI)
  if parse_version(gym.__version__)>=parse_version('0.9.6'):
    render = render
    reset  = reset
    seed   = _seed
    step   = step


  #utility function
  def change_targetLocation(self,delta_xyz=[0,0,0]):
    block_pos,block_orn = p.getBasePositionAndOrientation(self.blockUid)
    block_pos,block_orn = list(block_pos),list(block_orn)
    #changing target location
    for i in range(len(delta_xyz)):
      block_pos[i] += delta_xyz[i]
  
    p.resetBasePositionAndOrientation(self.blockUid,block_pos,block_orn)

  def generate_random_action(self):
    def generate_action(limits):
      return random.uniform(limits[0],limits[1])
    def generate_random_action_hand():
      hand_commands = []
      #getting links with active joints
      hand_links = self._kuka_hand.modelInfo.get_hand_links()
    

      hand_links_info = []
    
      for link_name in hand_links:
        link_info = self._kuka_hand.modelInfo.searchBy(key="link",value=link_name)
   
        hand_links_info.append(link_info)
     
      hand_links_info_with_active_joints = []
      for hand_link_info in hand_links_info:
        if hand_link_info["joint"]["j_type"] != "fixed":
          hand_links_info_with_active_joints.append(hand_link_info)
      hand_indexOf_active_joints =[]
      for Link in hand_links_info_with_active_joints:
        link = self._kuka_hand.jointInfo.searchBy("linkName",Link["link_name"])[0]
        hand_indexOf_active_joints.append(link["jointIndex"])
   
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
  
    kuka_command = generate_random_action_kuka()
  
    return hand_command + kuka_command

  def check_if_collsion_with_table_has_happend(self):
    contact = p.getContactPoints(self._kuka_hand.kuka_handId,self.table)
    # print("contact:::",contact)
    if len(contact)>0:
      return True
    else:
      return False

    #print(contact)
  def check_if_self_collision_has_happend(self):
    jointInfo = self._kuka_hand.jointInfo
  
    active_joints_info  = jointInfo.getActiveJointsInfo()
    num_active_joints = jointInfo.getNumberOfActiveJoints()
    index_of_actvie_joints = [active_joints_info[i]["jointIndex"] for i in range(num_active_joints)]
 
    contact_set = []
    child_check = []
    for index_1 in index_of_actvie_joints:
      for index_2 in index_of_actvie_joints:
        if index_1 == index_2:
          continue
          
        contact=p.getClosestPoints(self._kuka_hand.kuka_handId,self._kuka_hand.kuka_handId,0.0,index_1,index_2)

        if len(contact) !=0:

          link_one_name = jointInfo.searchBy("jointIndex",contact[0][3])[0]["linkName"]
          link_two_name = jointInfo.searchBy("jointIndex",contact[0][4])[0]["linkName"]

          contact_set.append([contact[0][3],contact[0][4]])

    new_contact = []
    for i in contact_set:

      if not i in new_contact:
        new_contact.append(i)
 
    parent_set = []
    
    for i in range(len(new_contact)):
      
      if new_contact[i][0] - new_contact[i][1] == 1:
        parent_set.append(new_contact[i])
   
    child_set = []
    for i in range(len(new_contact)):
      if new_contact[i][1] - new_contact[i][0] == 1:
        child_set.append(new_contact[i])
    

    parent_check = []
    for i in new_contact:
      for j in parent_set:
        if i == j:
          break
      else:
        parent_check.append(i)
    
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
    # print("right hand self coliision -------",child_check)
    # print("right hand self coliision -------",collision_result)
    # print("\n")
    if len(collision_result)>0:
      return True
    else:
      return False 

class Unit_test_Kuka_HandlitGymEnv():
  
  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=False,
               maxSteps = 10000):
    self.env = Kuka_HandlitGymEnv(renders=True, isDiscrete=False, maxSteps = maxSteps)
    self.blockUid = self.env.blockUid
    self.table = self.env.table
    #self.env._timeStep =1.0/10.0
    self.env._timeStep =1.0/500.0
  def check_collison_table(self,collide=True,use_kuka_IK = False):
    def move_kuka_end_ee(xpos=0.5,ypos=0.8,zpos=0.7,xAng=0,yAng=0,zAng=0):
        #kuka endeffector target-

        orn = p.getQuaternionFromEuler([xAng,yAng,zAng])    

        robotID = self.env._kuka_hand.kuka_handId
        modelInfo = self.env._kuka_hand.modelInfo
        jointInfo = self.env._kuka_hand.jointInfo

        active_joints_info  = jointInfo.getActiveJointsInfo()
        num_active_joints = jointInfo.getNumberOfActiveJoints()


        #getting joint index for kuka endeffector
        link_name ="lbr_iiwa_link_7"
        link_name_encoded = link_name.encode(encoding='UTF-8',errors='strict')
        kuka_ee_link_jointInfo = jointInfo.searchBy(key="linkName",value =link_name_encoded )[0]
        kuka_ee_link_Index = kuka_ee_link_jointInfo["jointIndex"]

        #setting target for kuka endeffector
        target_pos = [xpos,ypos,zpos]
        target_orn = [orn[0],orn[1],orn[2],orn[3]]

        #getting joint state for hand
        hand_joint_state =[]
        hand_links = modelInfo.get_hand_links()
        hand_links_info = []

        for link_name in hand_links:
          link_info = modelInfo.searchBy(key="link",value=link_name)
          hand_links_info.append(link_info)
        hand_links_info_with_active_joints = []
        for i in hand_links_info:
          if i["joint"]["j_type"] != "fixed":
            hand_links_info_with_active_joints.append(i)

        hand_indexOf_active_joints =[]

        for Link in hand_links_info_with_active_joints:
          link = jointInfo.searchBy("linkName",Link["link_name"])[0]
          hand_indexOf_active_joints.append(link["jointIndex"])

        for i in hand_indexOf_active_joints:
          joint_pos = p.getJointState(robotID,i)[0]
          hand_joint_state.append(joint_pos)


        #motor command
        motor_command = hand_joint_state+[xpos,ypos,zpos,xAng,yAng,zAng]
        stepCounter = 0
        collision = False
        while True:
          obs, done = self.env.reset(), False

          while not done:
            motorCommands = [-0.23364806507177005, 0.8363570016211279, 1.2411169105456517, 0.7149805793350532, 0.7430824669790088, 0.1774108157831189, 1.241051894486019, 0.03255280404701405, 0.9931661596163486, 0.6115282007509205, -0.21320959681646143, 0.7288280068336837, 0.015667746465842992, 0.9290643289604801, -0.18050325666907852, 1.1958384572255456, 3.5506053096348933, 2.878614142056888, 3.199938691793989, 0, -3.1415, 0]

            self.env.render()
            stepCounter +=1
            # print("stepCounter::",stepCounter)
            obs,reward,done=self.env.step([motor_command])
            self.env.change_targetLocation()
            # print("kuka IK joints::",jointPoses)
            # print("len kuka IK joints::",len(jointPoses))
            # print("hand_joint_state",hand_joint_state)
            # print("len(hand_joint_state)",len(hand_joint_state))
            # print("\n\n")
            collision = self.env.check_if_collsion_with_table_has_happend()
            print("collision",collision)
            # print("Euler",p.getEulerFromQuaternion([0,0,0,3.14*0.5]))
            if collision == True:
              return True
          
    def move_kuka_joint():
          
        kuka_command = (-0.36350324105018117, 0.08434877972795868, 1.3253607910308352, -1.3023695529206833, 1.1981002550715345, -2.4402425653234032, -0.3733762283635729)
        robotID = self.env._kuka_hand.kuka_handId
        modelInfo = self.env._kuka_hand.modelInfo
        jointInfo = self.env._kuka_hand.jointInfo

        active_joints_info  = jointInfo.getActiveJointsInfo()
        num_active_joints = jointInfo.getNumberOfActiveJoints()


        #getting joint index for kuka endeffector
        link_name ="lbr_iiwa_link_7"
        link_name_encoded = link_name.encode(encoding='UTF-8',errors='strict')
        kuka_ee_link_jointInfo = jointInfo.searchBy(key="linkName",value =link_name_encoded )[0]
        kuka_ee_link_Index = kuka_ee_link_jointInfo["jointIndex"]



        #getting joint state for hand
        hand_joint_state =[]
        hand_links = modelInfo.get_hand_links()
        hand_links_info = []

        for link_name in hand_links:
          link_info = modelInfo.searchBy(key="link",value=link_name)
          hand_links_info.append(link_info)
        hand_links_info_with_active_joints = []
        for i in hand_links_info:
          if i["joint"]["j_type"] != "fixed":
            hand_links_info_with_active_joints.append(i)

        hand_indexOf_active_joints =[]

        for Link in hand_links_info_with_active_joints:
          link = jointInfo.searchBy("linkName",Link["link_name"])[0]
          hand_indexOf_active_joints.append(link["jointIndex"])

        for i in hand_indexOf_active_joints:
          joint_pos = p.getJointState(robotID,i)[0]
          hand_joint_state.append(joint_pos)


        #motor command
        motor_command = list(kuka_command)+hand_joint_state
        motor_command=(0.8066802307167508, 1.0232196872785049, 0.15139632616115659, 2.640651350178653, -1.4544423913682578, -0.8769639265642334, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        stepCounter = 0
        collision = False
        while True:
          obs, done = self.env.reset(), False

          while not done:
            
            self.env.render()
            stepCounter +=1
            print("stepCounter::",stepCounter)
            obs,reward,done=self.env.step([motor_command])
            self.env.change_targetLocation()
            # print("kuka IK joints::",jointPoses)
            # print("len kuka IK joints::",len(jointPoses))
            # print("hand_joint_state",hand_joint_state)
            # print("len(hand_joint_state)",len(hand_joint_state))
            # print("\n\n")
            collision = self.env.check_if_collsion_with_table_has_happend()
            print("collision",collision)
            # print("Euler",p.getEulerFromQuaternion([0,0,0,3.14*0.5]))
            if collision == True:
              return True
          if done == True:
            return collision
    if collide ==True:
      if use_kuka_IK == True:
        return move_kuka_end_ee()
      else:
        return move_kuka_joint()
    else:
      if use_kuka_IK == True:
        # self.env.change_targetLocation([0,0,1])
        return move_kuka_end_ee(zpos =1.7,xAng =1.57)
      else:
        print(":::::::::::::::::::::::::::::::::::move_kuka_joint()")
        return move_kuka_joint()
  
  def check_self_collision(self):
    #motor_command=(0.8066802307167508, 1.0232196872785049, 0.15139632616115659, 2.640651350178653, -1.4544423913682578, -0.8769639265642334, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0)
    motor_command=(0.5865066887373056, 0.5150960353836044, 0.12028432167582896, -1.445094990000865, -0.1047064260103646, -0.08477672230318149, -0.004951987091249091,0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    stepCounter = 0
    collision = False
    while True:
      obs, done = self.env.reset(), False

      while not done:
        
        self.env.render()
        stepCounter +=1
        print("stepCounter::",stepCounter)
        obs,reward,done=self.env.step([motor_command])
        self.env.change_targetLocation()
        # print("kuka IK joints::",jointPoses)
        # print("len kuka IK joints::",len(jointPoses))
        # print("hand_joint_state",hand_joint_state)
        # print("len(hand_joint_state)",len(hand_joint_state))
        # print("\n\n")

        #collision = self.env.check_if_self_collision_has_happend()
        collision = self.env.check_if_collsion_with_table_has_happend
        print("collision",collision)
        # print("Euler",p.getEulerFromQuaternion([0,0,0,3.14*0.5]))
        if collision == True:
          return True
      if done == True:
        return collision


class Demo():

  def __init__(self):
    self.env = Kuka_HandlitGymEnv(renders=True, isDiscrete=False)
    self.env.reset()
    self.setp_counter = 0
    self.increment_counter = 0
   
  def get_commandIndexForpart(self,part):
    # stroing index of command in motor command 
    joint_names_KUKA  ={}
    joint_names_Base = {}
    joint_names_TH   = {}
    joint_names_FF   = {}
    joint_names_MF   = {}
    joint_names_RF   = {}
    joint_names_LF   = {}

    indexOfActiveJoints = self.env._kuka_hand.jointInfo.getIndexOfActiveJoints()
    jointsInfo = self.env._kuka_hand.jointInfo.getActiveJointsInfo()

    # print("indexOfActiveJoints::",indexOfActiveJoints)
    # getting position of specifc command in motorCommand array
    for jointInfo in jointsInfo:
      joint_name = jointInfo["jointName"]
      jointIndex = jointInfo["jointIndex"]
      # print("jointIndex::",jointIndex,"  jointName::",joint_name)
      jointll  = jointInfo["jointLowerLimit"]
      jointul  = jointInfo["jointUpperLimit"]
      if "KUKA" in joint_name:
        joint_names_KUKA[joint_name] = {"commandIndex":indexOfActiveJoints.index(jointIndex),
                                        "jointIndex":jointIndex,
                                        "jointll":jointll,
                                        "jointul":jointul
                                        }

      if "WR" in joint_name:
        joint_names_Base[joint_name] = {"commandIndex":indexOfActiveJoints.index(jointIndex),
                                        "jointIndex":jointIndex,
                                        "jointll":jointll,
                                        "jointul":jointul
                                        }
      if "TH" in joint_name:
        joint_names_TH[joint_name]   = {"commandIndex":indexOfActiveJoints.index(jointIndex),
                                        "jointIndex":jointIndex,
                                        "jointll":jointll,
                                        "jointul":jointul
                                        }
      if "FF" in joint_name:
        joint_names_FF[joint_name]   = {"commandIndex":indexOfActiveJoints.index(jointIndex),
                                        "jointIndex":jointIndex,
                                        "jointll":jointll,
                                        "jointul":jointul
                                        }
      if "MF" in joint_name:
        joint_names_MF[joint_name]   = {"commandIndex":indexOfActiveJoints.index(jointIndex),
                                        "jointIndex":jointIndex,
                                        "jointll":jointll,
                                        "jointul":jointul
                                        }
      if "RF" in joint_name:
        joint_names_RF[joint_name]   = {"commandIndex":indexOfActiveJoints.index(jointIndex),
                                        "jointIndex":jointIndex,
                                        "jointll":jointll,
                                        "jointul":jointul
                                        }
      if "LF" in joint_name:
        joint_names_LF[joint_name]   = {"commandIndex":indexOfActiveJoints.index(jointIndex),
                                        "jointIndex":jointIndex,
                                        "jointll":jointll,
                                        "jointul":jointul
                                        }
      

    if part =="FF":
      return joint_names_FF
    elif part =="MF":
      return joint_names_MF
    elif part =="RF":
      return joint_names_RF
    elif part =="LF":
      return joint_names_LF
    elif part=="Base":
      return joint_names_Base
    elif part=="KUKA":
      return joint_names_KUKA      

  def get_active_joints_name(self):
    jointsInfo = self.env._kuka_hand.jointInfo.getActiveJointsInfo()
    jointsNames = [jointInfo["jointName"] for jointInfo in jointsInfo]
    return jointsNames
  
  
  def move_robot(self,partName="FF",joint_name ="J4_FF",movement_direction = "positive",increment=2):
    print("Demo::move_robot")
  
    jointsStates = self.env._kuka_hand.getObservation_joint()
    motorCommand = jointsStates

    index_of_commandsIn_motorCommand = self.get_commandIndexForpart(partName)

    if self.setp_counter%10 == 0:
      for jointName in index_of_commandsIn_motorCommand:
        
        if jointName == joint_name:
          print("jointName::",jointName)
          jointll = index_of_commandsIn_motorCommand[jointName]["jointll"]
          jointul = index_of_commandsIn_motorCommand[jointName]["jointul"]
          if(abs(jointll) ==abs(jointul)):
            increment = abs(jointul)/10
          else:
            increment = (abs(jointll)-abs(jointul))/10
            if increment <0 and movement_direction =="positive":
              increment = abs(increment)
            elif increment>0 and  movement_direction =="negative":
              increment = -1*increment
          
          commandIndex = index_of_commandsIn_motorCommand[jointName]["commandIndex"] 
          if self.increment_counter <=increment:
            print("adding increment to joint")
            motorCommand[commandIndex] = jointsStates[commandIndex]+increment
            self.increment_counter +=1
          print("motorCommand[commandIndex]::",motorCommand[commandIndex])
          print("jointll::",jointll)
          print("jointul::",jointul)
          print(" motorCommand[commandIndex]<jointul and  motorCommand[commandIndex]>jointll", motorCommand[commandIndex]<jointul and  motorCommand[commandIndex]>jointll)
          if motorCommand[commandIndex]<jointul and  motorCommand[commandIndex]>jointll:
            print("increment added")
            motorCommand[commandIndex] = motorCommand[commandIndex]
          else:
            print("reseting to lower limit")
            motorCommand[commandIndex] = jointll

    self.setp_counter +=1
    print("\n\n")
    self.env.step([motorCommand])





'''
Todo:
[]Finish collision detection
Deadline:18Jun19
[]Finish the Reward function
Deadline:19Jun19
[]check if collision detection works
[]
'''
