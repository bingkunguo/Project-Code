import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)
import math
import numpy as np
import pybullet as p
from mamad_util import JointInfo
import pybullet_data
from kuka_handlit_controller import Kuka_Handlit
import random

class AdaptiveWorkspace():
  
  def __init__(self,
               target_xyz,
               initial_r,
               target_dim,hand_length,grownth_increment):
    self.target_dim = target_dim
    self.initial_r = initial_r
    self.hand_length = hand_length
    self.target_xyz = target_xyz
    self.radius = self.set_initial_r()
    self.grownth_increment = grownth_increment
    self.jointInfo = JointInfo()
    self.kuka_handId = 3
  def check_if_ee_outside_AW(self,ee_xyz):
    kuka_EE_State  = ee_xyz
    kuka_EE_Pos = kuka_EE_State[0]
    kuka_EE_Orn = kuka_EE_State[1]
    dist_x = kuka_EE_Pos[0] - self.target_xyz[0]
    dist_y = kuka_EE_Pos[1] - self.target_xyz[1]
    dist_z = kuka_EE_Pos[2] - self.target_xyz[2]
    ee_to_origin = math.sqrt(dist_x**2 + dist_y**2 + dist_z**2)
    if (ee_to_origin >= self.radius):
        return True
    else:
        return False
  def grow_AW(self):
    kuka_work_space_r = 10
    if ((self.radius + self.grownth_increment) <= kuka_work_space_r):
      self.radius = self.radius + self.grownth_increment
    else:
      self.radius = kuka_work_space_r - self.grownth_increment
  

  def pick_random_ee_position_in_AW(self):
    z_limit = self.target_dim[2]
    # if z_limit >= 0.5:
    #   x_y_limit = (self.initial_r - math.sqrt(self.initial_r**2 - 4*(z_limit**2)))/2
    # else:
    #   x_y_limit = self.initial_r - 0.2
    ee_xyz = [0,0,0]
    # ee_xyz[0] = random.uniform(self.target_xyz[0] - x_y_limit ,self.target_xyz[0] + x_y_limit)
    # ee_xyz[1] = random.uniform(self.target_xyz[1] - x_y_limit,self.target_xyz[1] + x_y_limit)
    ee_xyz[0] = random.uniform(self.target_xyz[0] - self.initial_r ,self.target_xyz[0] + self.initial_r)
    ee_xyz[1] = random.uniform(self.target_xyz[1] - self.initial_r,self.target_xyz[1] + self.initial_r)
    ee_xyz[2] = random.uniform(z_limit + self.target_xyz[2], self.initial_r + self.target_xyz[2])
    print("test:::",ee_xyz)
    
    return ee_xyz
  
  #utility

  def set_initial_r(self):
    target_length = self.target_dim[0]
    target_width = self.target_dim[1]
    target_height = self.target_dim[2]
    max = target_length
    if (target_width > max):
      max = target_width
    if (target_height > max):
      max = target_height
    initial_r = random.uniform(math.ceil(max + self.hand_length),1.5)
    return initial_r