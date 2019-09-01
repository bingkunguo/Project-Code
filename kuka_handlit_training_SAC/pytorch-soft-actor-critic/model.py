import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim=[256]):
        super(ValueNetwork, self).__init__()

        self.v = nn.ModuleList()
        #input layers
        self.v.append( nn.Linear(num_inputs, hidden_dim[0]))
        #hidden layers
        if len(hidden_dim)>0:
            for i in range(len(hidden_dim)-1):
                self.v.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
        #output layers
        self.v_out = nn.Linear(hidden_dim[-1], 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = state
        for layer in self.v:
            x = F.relu(layer(x))

        x = self.v_out(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=[256]):
        super(QNetwork, self).__init__()
        

        self.q1 = nn.ModuleList()
        self.q2 = nn.ModuleList()

        # Q1 architecture
        self.q1.append(nn.Linear(num_inputs + num_actions, hidden_dim[0]))
        if len(hidden_dim)>0:
            for  i in range(len(hidden_dim)-1):
                self.q1.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
        self.q1_out = nn.Linear(hidden_dim[-1], 1)

        # Q2 architecture
        self.q2.append(nn.Linear(num_inputs + num_actions, hidden_dim[0]))
        if len(hidden_dim)>0:
            for  i in range(len(hidden_dim)-1):
                self.q2.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
        self.q2_out = nn.Linear(hidden_dim[-1], 1)

        self.apply(weights_init_)

    def forward(self, state, action):

        xu = torch.cat([state, action], 1)
        x1=x2=xu
        
        for layer in self.q1:
            x1 = F.relu(layer(x1))
        x1 = self.q1_out(x1)

        for layer in self.q2:
            x2 = F.relu(layer(x2))
        x2 = self.q2_out(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=[256], action_space=None):
        super(GaussianPolicy, self).__init__()

        self.gp = nn.ModuleList()
        
        self.gp.append(nn.Linear(num_inputs, hidden_dim[0]))
        if len(hidden_dim)>0:
            for i in range(len(hidden_dim)-1):
                self.gp.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))

        self.mean_linear = nn.Linear(hidden_dim[-1], num_actions)
        self.log_std_linear = nn.Linear(hidden_dim[-1], num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x=state
        for layer in self.gp:
            x = F.relu(layer(x))
       

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=[256], action_space=None):
        super(DeterministicPolicy, self).__init__()

        self.dp = nn.ModuleList()

        self.dp.append(nn.Linear(num_inputs, hidden_dim[0]))

        if len(hidden_dim)>0:
            for i in range(len(hidden_dim)-1):
                self.dp.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))

        self.mean = nn.Linear(hidden_dim[-1], num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x=state
        for layer in self.dp:
            x = F.relu(layer(x))

        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)