#!/usr/bin/env python

import pybullet as p



if __name__ == '__main__':
	p.connect(p.GUI)	
	objects = p.loadSDF("./model.sdf")
	texUid = p.loadTexture("./aaa.png")
	# tex = p.loadTexture("./side.png")

	p.changeVisualShape(objects[0], -1, rgbaColor=[1, 1, 1, 1])
	p.changeVisualShape(objects[0], -1, textureUniqueId=texUid)	

	while(1):
		cube = objects
		p.setRealTimeSimulation(1)
		