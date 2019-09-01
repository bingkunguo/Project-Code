import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import gym
import numpy as np
import kuka_handlit
import os
import csv
import datetime


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim,action_dim, action_std,hiddenLayer_neurons=[64,32]):
        super(ActorCritic, self).__init__()

        self.hiddenLayer_neurons =hiddenLayer_neurons
       
        self.actor  = nn.Sequential()
        self.critic = nn.Sequential()

        #actor
        self.actor.add_module("nn_0",nn.Linear(state_dim,self.hiddenLayer_neurons[0]))
        self.actor.add_module("tan_0",nn.Tanh())

        for i in range(len(self.hiddenLayer_neurons)-1):
            self.actor.add_module("nn_"+str(i+1),nn.Linear(self.hiddenLayer_neurons[i],self.hiddenLayer_neurons[i+1]))
            self.actor.add_module("tan_"+str(i+1),nn.Tanh())
        
        self.actor.add_module("nn_-1",nn.Linear(self.hiddenLayer_neurons[-1],action_dim))
        self.actor.add_module("tan_-1",nn.Tanh())

        #critic
        self.critic.add_module("nn_0",nn.Linear(state_dim,self.hiddenLayer_neurons[0]))
        self.critic.add_module("tan_0",nn.Tanh())

        for i in range(len(self.hiddenLayer_neurons)-1):
            self.critic.add_module("nn_"+str(i+1),nn.Linear(self.hiddenLayer_neurons[i],self.hiddenLayer_neurons[i+1]))
            self.critic.add_module("tan_"+str(i+1),nn.Tanh())
        
        self.critic.add_module("nn_-1",nn.Linear(self.hiddenLayer_neurons[-1],1))
        


        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
     
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, state, action):   
        action_mean = torch.squeeze(self.actor(state))
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip,hiddenLayer_neurons=[64,32]):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, action_std,hiddenLayer_neurons=hiddenLayer_neurons).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, action_std,hiddenLayer_neurons=hiddenLayer_neurons).to(device)
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
def load_training():
    import os
    import re
    files=os.listdir("./")
    saved_models_file_name = []
   
    regex = r"(PPO_continuous_kuka_handlit-v0)+(.*)+(.pth)"
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
            directory = "./"
            path = directory+filename
            return path
        else:
            return False
    else:
        return False
    
   
        
    



def main():
    env_name = "kuka_handlit-v0"
    render = False
    ############## Hyperparameters ##############
  
    solved_reward = 1         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 10000        # max training episodes
    max_timesteps = 1500        # max timesteps in one episode

    hiddenLayer_neurons=[64,32,32]
    update_timestep = 4000      # update policy every n timesteps
    action_std = 0.5            # constant std for action distribution (Multivariate Normal)
    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    
    lr = 0.003                 # parameters for Adam optimizer
    betas = (0.9, 0.999)
    
    random_seed = None
    #############################################
    
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observationDim
    action_dim = env.action_space.shape[0]

    #House Keeping
    rewardFunctionVersion =1
    nnVersion =3

  


    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip,hiddenLayer_neurons)
   

    trainingToLoad = load_training()
    if trainingToLoad:
        print("loading "+trainingToLoad+" ...")
        ppo.policy_old.load_state_dict(torch.load(trainingToLoad))
  
        
    # logging variables
    running_reward = 0
    old_running_reward = -10000000000000000000000000000000000000000000
    avg_length = 0
    time_step = 0
    current_episode = 0
    episode = 0

    #getting latest episode
    # with open("reward_each_episode_kuka_handlit-v0_1_1.csv") as csvfile:
    #     #print("im inside:::")
    #     mLines = csvfile.readlines()
    #     if len(mLines):
    #         targetline = mLines[-1]
    #         current_episode = targetline.split(',')[1]

    # training loop
    for i_episode in range(1, max_episodes+1):
       
        state = env.reset()
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done= env.step([action])
            # Saving reward:
            memory.rewards.append(reward)
            
            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
            if render:
                env.render()
            if done:
                break
       
       
        
        avg_length += t
        
        with open("reward_each_episode_{}_{}_{}.csv".format(env_name,rewardFunctionVersion,nnVersion), "a+", newline ="") as csvfile:
    
            writer = csv.writer(csvfile)
            episode = int(current_episode) + i_episode
            writer.writerow([running_reward,episode])


        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
            break
        
        # save every 500 episodes #TODO
        if running_reward > old_running_reward:
            time = datetime.datetime.now()
            t= time.strftime("%d%b%y-%H:%M")
            print(t)
            print("\n\n\n")
            print("Saving the model....")
            torch.save(ppo.policy.state_dict(), './PPO_continuous_{}_{}_{}_{}.pth'.format(env_name,rewardFunctionVersion,nnVersion,t))
            print("\n\n\n")
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
        old_running_reward = running_reward #TODO
            
if __name__ == '__main__':
    main()
    
