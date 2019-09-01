import gym
import kuka_handlit



def random_agent(episodes=100):
	env = gym.make("kuka_handlit-v0")
	demo = kuka_handlit

	env.reset()
	env.render()
	stepCounter =0
	motor_command=(0.5865066887373056, 0.5150960353836044, 0.12028432167582896, -1.445094990000865, -0.1047064260103646, -0.08477672230318149, -0.004951987091249091,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
	while True:
		obs, done = env.reset(), False

        
		#print("===================================")        
		#print("obs")
		#print(obs)
		episode_rew = 0
		while not done:
			env.render()
		
			stepCounter +=1
			#print("stepCounter",stepCounter)
			#print("Demo Motorcommand",action)
			#print("Demo Motorcommand len",len(action))

			env.step([motor_command])

if __name__ == "__main__":
    random_agent()
