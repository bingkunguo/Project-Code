import gym
from PPO_continuous import PPO, Memory
from PIL import Image
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_structureOf_hiddenLayers():
    x = input("please enter stucture of hidden layers(e.g:64 32)")
    str_layer = x.split()
    print(str_layer)
    layers = [int(layer) for layer in str_layer]
    print("layers::",layers)
    return layers


def load_training():
    import os
    import re
    files=os.listdir("./trail1")
  
    saved_models_file_name = []
   
    regex = r"(PPO_continuous_)+(.*)+(.pth)"
    for f in files:
        match = re.search(regex,f)
        if match:
            saved_models_file_name.append(f)
        
    if len(saved_models_file_name) !=0:
        print("Privious trains are::")
        for i in range(0,len(saved_models_file_name)):
            print(str(i)+". "+saved_models_file_name[i])
        prompt = input("would you like to start from an old training?(y/n)")
        if prompt =="y":
            x=input("please enter which model would you like to load(number): ")

            filename = saved_models_file_name[int(x)]
            directory = "./trail1/"
            path = directory+filename
            return path
        else:
            return False
    else:
        return False

def test():
    ############## Hyperparameters ##############
    env_name = "kuka_handlit-v0"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    n_episodes = 200000          # num of episodes to run
    max_timesteps = 1500    # max timesteps in one episode
    render = True           # render the environment
    save_gif = False        # png images are saved in gif folder
    
    # filename and directory to load model from
    path = load_training()

    hiddenLayer_neurons = get_structureOf_hiddenLayers()

    action_std = 0.5        # constant std for action distribution (Multivariate Normal)
    K_epochs = 80           # update policy for K epochs
    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor
    
    lr = 0.0003             # parameters for Adam optimizer
    betas = (0.9, 0.999)
    #############################################
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip,hiddenLayer_neurons)
    ppo.policy_old.load_state_dict(torch.load(path))
    
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_timesteps):
            action = ppo.select_action(state, memory)
            state, reward, done = env.step([action])
            ep_reward += reward
            if render:
                env.render()
            if save_gif:
                 img = env.render(mode = 'rgb_array')
                 img = Image.fromarray(img)
                 img.save('./gif/{}.jpg'.format(t))  
            if done:
                break
            
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
        ep_reward = 0
        env.close()
    
if __name__ == '__main__':
    test()
    
    
