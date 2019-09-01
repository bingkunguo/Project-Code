

import yaml

class ModelInfo():
    def __init__(self,path):
        self.modelInfo_dic ={}
        pathToModelInfo = path
        with open(pathToModelInfo, 'r') as stream:
            try:
                self.modelInfo_dic = yaml.safe_load(stream)
                #print self.modelInfo_dic
            except yaml.YAMLError as exc:
                print(exc) 
        self.searchKey = ["robot","part","link","joint name","joint type"]
    def getHandInfo(self):
        return  self.modelInfo_dic["Hand"]
        
    def getKukaInfo(self):
        return  self.modelInfo_dic["KUKA"]

    def searchBy(self,key,value=None):
        quary_links =[]
        robots =  self.modelInfo_dic["Robots"]
        for Robot in robots:
            robot = robots[Robot]
            if (key =="robot" and value==Robot):
                return robot
            elif key =="robot":
                continue
            
            for Part in robot:
                part = robot[Part]
                if (key =="part" and value==Part):
                    return part
                elif key =="part":
                    continue

                links = part["Links"]
                for Link in links:
                    link = links[Link]
                    if (key =="link" and value==Link):
                        return link
                    elif key == "joint name" and value == links[Link]["joint"]["j_name"]:
                         return link
                    elif key == "joint type" and value == links[Link]["joint"]["j_type"]:
                      
                         quary_links.append(link)
                    elif key =="link" or key =="joint name" or key=="joint type":
                        continue
                    
        if key == "joint type":
            return quary_links


    def getModelActiveJoints(self):
        revolute = []
        revolute = self.searchBy(key="joint type",value="revolute")

        return revolute
               
    def get_hand_links(self):
        hand_links = []
    
        BASE = self.searchBy(key="part",value ="BASE")
        FF = self.searchBy(key="part",value ="FF")
        MF = self.searchBy(key="part",value ="MF")
        RF = self.searchBy(key="part",value ="RF")
        TH = self.searchBy(key="part",value ="TH")
        part_list = [BASE,FF,MF,RF,TH]
        for part in part_list:
            links = part["Links"]
            #print (links)
            for link in links:
                hand_links.append(link)

        return hand_links

    def get_kuka_links(self):
        kuka_links = []
        Body = self.searchBy(key="part",value ="Body")
       
        part_list = [Body]
        for part in part_list:
            links = part["Links"]
            for link in links:
                kuka_links.append(link)
        return kuka_links

            

"""
modelInfo = ModelInfo(path="model_info.yml")
#test = modelInfo.searchBy(key="joint type",value="revolute")
modelInfo.get_hand_links()
print(test[0])
""" 




