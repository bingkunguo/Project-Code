
# -*- coding: utf-8 -*-
from collections import OrderedDict
import re
import pybullet as p
class JointInfo():

    def __init__(self):
        #dictornaries do not have an order this is a work around 
        self.dict_order = [ "jointIndex"      ,
                            "jointName"       ,
                            "jointType"       ,
                            "qIndex"          ,
                            "uIndex"          ,
                            "flags"           ,
                            "jointDamping"    ,
                            "jointFriction"   ,
                            "jointLowerLimit" ,
                            "jointUpperLimit" ,
                            "jointMaxForce"   ,
                            "jointMaxVelocity",
                            "linkName"        ,
                            "jointAxis"       ,
                            "parentFramePos"  ,
                            "parentFrameOrn"  ,
                            "parentIndex"     
                    ]
        self.alreadyCleaned = False
        self.joints_orderedDict = []
        self.searchKey = ["linkName","jointName","jointType","jointIndex"]
        self.jointType = {"revolute":0,"prismatic":1,"spherical":2,"planar":3,"fixed":4}
        self.active_joints = ["revolute","prismatic","spherical","planar"]
        self.jointsInfoCollected = False
    def getJointinfo_dict_func(self,getJointinfo):
        
        getJointinfo_dict = {
                            "jointIndex"      : None,
                            "jointName"       : None,
                            "jointType"       : None,
                            "qIndex"          : None,
                            "uIndex"          : None,
                            "flags"           : None,
                            "jointDamping"    : None,
                            "jointFriction"   : None,
                            "jointLowerLimit" : None,
                            "jointUpperLimit" : None,
                            "jointMaxForce"   : None,
                            "jointMaxVelocity": None,
                            "linkName"        : None,
                            "jointAxis"       : None,
                            "parentFramePos"  : None,
                            "parentFrameOrn"  : None,
                            "parentIndex"     : None
                            }
        dict_order = self.dict_order
        index = 0
        for key in dict_order:
            
            getJointinfo_dict [key] = getJointinfo[index] 
            index +=1

        return getJointinfo_dict

    def showInOrder(self,getJointinfo):
        getJointinfo_dict = self.getJointinfo_dict_func(getJointinfo)
        
        dict_order = self.dict_order
        index = 0
        print("@@@@@@@@@@@@@@Link Name: " + str(getJointinfo_dict["linkName"])+" @@@@@@@@@@@@@@@@@@@")
        for key in dict_order:
            
            print(dict_order[index] +"   "+str(getJointinfo_dict [key])) 
            index +=1


    def get_stored_joints(self):
        if len(self.joints_orderedDict) !=0:
            return self.joints_orderedDict
        else:
            print("Nothing is stored yet")
            return False
    def get_keys(self):
        return self.searchKey

    def searchBy(self,key,value):
        quary =[]
        #print("len(self.joints_orderedDict)",len(self.joints_orderedDict))
        #print("key:::",key)
        if key in self.searchKey:
            for item in self.joints_orderedDict:
                #print("item",item)
                for dic_key in item:
                    #print("dic_key::",dic_key)
                    if dic_key==key:
                        #print("Debug9::item[key]",item[key])
                        #print("Debug10::type(item[key])",type(item[key]))
                        if item[key] == value:
                            quary.append(item)
                            #print("£££££found one££££")
                            break
            if(len(quary)!=0):
                #print("len(quary)",len(quary))
                #print(quary)           
                return quary
                
        elif(len(quary)==0):   
            #print("No match found")
            return []             
        else:
            print("invalid search key")
            return False
            
        print(quary)

    def searchBy_regex(self,key,regex,quary=[]):
        quary =quary
        if key in self.searchKey:
            for item in self.joints_orderedDict:
                    #print("item",item)
                    for dic_key in item:
                        #print("dic_key::",dic_key)
                        if dic_key==key:
                            #print("Debug9::item[key]",item[key])
                            #print("Debug10::type(item[key])",type(item[key]))
                            sentence = item[key] 
                            #decoding binary to string
                            sentence = sentence.decode(encoding='UTF-8',errors='strict')
                            match = re.findall(regex,sentence)
                            if len(match) > 0: #matching regex
                                quary.append(item)
                                #print("£££££found one££££")
                                break
            if(len(quary)!=0):
                #print("len(quary)",len(quary))
                #print(quary)           
                return quary
        elif(len(quary)==0):   
            #print("No match found")
            return []             
        else:
            print("invalid search key")
            return False
        

    def getNumberOfActiveJoints(self):
        activeJT = self.jointType
        num_revolute  = (lambda:len(self.searchBy('jointType',activeJT["revolute"]))  if(self.searchBy('jointType',activeJT["revolute"])  != None) else 0)()
        num_prismatic = (lambda:len(self.searchBy('jointType',activeJT["prismatic"])) if(self.searchBy('jointType',activeJT["prismatic"]) != None) else 0)()
        num_spherical = (lambda:len(self.searchBy('jointType',activeJT["spherical"])) if(self.searchBy('jointType',activeJT["spherical"]) != None) else 0)()
        num_planar    = (lambda:len(self.searchBy('jointType',activeJT["planar"]))    if(self.searchBy('jointType',activeJT["planar"])    != None) else 0)()
        num_active    = num_revolute+num_prismatic+num_spherical+num_planar
        #print("Debug 5",num_active)
        return num_active
    
        #print("Debug 4::",self.searchBy('jointType',activeJT["prismatic"]))
    def getIndexOfActiveJoints(self):
        activeJT = self.jointType
        #print("debug3:::self.searchBy('jointType',activeJT['prismatic']",self.searchBy('jointType',activeJT["prismatic"]))
        revolute_j  = (lambda:self.searchBy('jointType',activeJT["revolute"])  if(self.searchBy('jointType',activeJT["revolute"])  != None) else [])()
        prismatic_j = (lambda:self.searchBy('jointType',activeJT["prismatic"])  if(self.searchBy('jointType',activeJT["prismatic"])  != None) else [])()
        spherical_j = (lambda:self.searchBy('jointType',activeJT["spherical"])  if(self.searchBy('jointType',activeJT["spherical"])  != None) else [])()
        planar_j    = (lambda:self.searchBy('jointType',activeJT["planar"])  if(self.searchBy('jointType',activeJT["planar"])  != None) else [])()
        active_joints = revolute_j+prismatic_j+spherical_j+planar_j
        active_joints_index =[]
        for item in active_joints:
            active_joints_index.append(item["jointIndex"])
        #print(active_joints_index)
        return active_joints_index
    def getActiveJointsInfo(self):
        activeJ_info = []
        indexOf_activeJoints = self.getIndexOfActiveJoints()
        for jointIndex in indexOf_activeJoints:
            jointinfo = self.searchBy("jointIndex",jointIndex)[0]
            activeJ_info.append(jointinfo)
        
        return activeJ_info
    def getJointLimits(self,key,value):
        quary = self.searchBy(key,value)[0]
        jointLL = quary["jointLowerLimit"]
        jointUL = quary["jointUpperLimit"]
        jointLimts = [jointLL,jointUL]

        return jointLimts

    def getJInfo_dict(self,getJointinfo):
        getJointinfo_dict =  self.getJointinfo_dict_func(getJointinfo)
        return getJointinfo_dict

    def getCleanJInfo_dict(self,getJointinfo):
        if self.alreadyCleaned !=True:
            getJointinfo_dict = self.getJointinfo_dict_func(getJointinfo)
            getJointinfo_dict["jointName" ] = str(getJointinfo_dict["jointName" ])[2:-1]
            getJointinfo_dict["linkName"]   = str(getJointinfo_dict["linkName"])[2:-1]
            self.alreadyCleaned =True
            
        return getJointinfo_dict
    def get_infoForAll_joints(self,robotModel):
        """
        This function should be called if you want to use this class
        it should be called inside reset function ans adter you load the robot
    
        input:an instance of robot that is loaded in pybullet
        """
        robotID = robotModel[0]
        self.robotID = robotID
        jointNames = []
        robot_info_dict = {}
        noJointsInSDF = p.getNumJoints(robotID)
        for i in range (noJointsInSDF):
          pybullet_jointInfo = p.getJointInfo(robotID,i)
          self.saveInorder(pybullet_jointInfo)

        self.jointsInfoCollected = True 

    def show_infoForAll_joints(self):
        print("\n\n\n")
        print(self.joints_orderedDict)
        print("\n\n\n")
    #utility function
    def saveInorder(self,getJointinfo):
        getJointinfo_dict = self.getJointinfo_dict_func(getJointinfo)
        
        jointOrderDict = OrderedDict()
        dict_order = self.dict_order
        index = 0
        for key in dict_order:
            jointOrderDict[dict_order[index]] = getJointinfo_dict [key]
            index +=1
        #print(jointOrderDict)
        self.joints_orderedDict.append(jointOrderDict)   

'''
test_list = [1, b'J1', 0, 8, 7, 1, 0.5, 0.0, -2.0944, 2.0944, 300.0, 10.0, b'lbr_iiwa_link_2', (0.0, 0.0, 1.0), (0.0, 0.029999999329447746, 0.08250001817941666), (-8.963236692910115e-07, 0.7071080610445867, 0.7071055013250558, 8.963204246140962e-07), 0]
test_list = list(test_list)
test_dict = JointInfo(test_list)
test_dict.showInOrder()
print(test_dict.getJInfo_dict())
print(test_dict.getCleanJInfo_dict())

'''
'''
test = str( b'lbr_iiwa_link_2')
test = test[2:-1]
print (test)
'''


