import torch
from torch import nn
import random
import numpy as np
import collections
from pyinstrument import Profiler


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 
        

    def size(self): 
        return len(self.buffer)


class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()
        
        self.seq1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.Mish(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.Mish()
        )

        self.seq2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.Mish(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.Mish(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Mish()
        )

        self.linear = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Mish(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.Mish(),
            nn.Linear(64, output_dim)
        )


    
    def forward(self, x):
        x = self.seq1(x)
        x = self.seq2(x)

        out = self.linear(x)

        return out



class QValueNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QValueNet, self).__init__()

        self.seq1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.Mish(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.Mish()
        )

        self.seq2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.Mish(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.Mish(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.Mish()
        )


        self.action_linear = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.LayerNorm(128),
            nn.Mish(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.Mish()
        )

        self.linear = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, 32),
            nn.LayerNorm(32),
            nn.Mish(),
            nn.Linear(32, 1)
        )


    def forward(self, x, a):
        x = self.seq1(x)
        x = self.seq2(x)
        a = self.action_linear(a)

        out = torch.cat([x, a], dim=1)

        out = self.linear(out)

        return out
    
profile = Profiler()
class DDPG:
    ''' DDPG算法 '''
    def __init__(self, input_dim, output_dim, random_rate, sigma, actor_lr, critic_lr, tau, gamma, device):
        self.actor = PolicyNet(input_dim, output_dim).to(device)
        self.critic = QValueNet(input_dim, output_dim).to(device)
        self.target_actor = PolicyNet(input_dim, output_dim).to(device)
        self.target_critic = QValueNet(input_dim, output_dim).to(device)

        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.random_rate = random_rate
        self.device = device

        self.critic_mse_loss = nn.MSELoss().to(device)

    def take_action(self, state, noise=True):

        if random.random() < self.random_rate and noise:
            action = torch.rand(self.output_dim)
        else:
            with torch.no_grad():
                state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device).unsqueeze(0)
                state = state.to(self.device)
                action = self.actor(state).cpu()[0]

        return action


    def soft_update(self, net, target_net):
        
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)



    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float32).view(-1, 1).to(self.device)

        # 计算权重
        weights = torch.where(rewards < 0, torch.tensor(1.0), torch.tensor(1.0)).to(self.device)

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)

        critic_loss = self.critic_mse_loss.forward(self.critic(states, actions), q_targets)
        critic_loss = (critic_loss * weights).mean()  # 使用权重调整critic_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic.forward(states, self.actor(states))
        actor_loss = (actor_loss * weights).mean()  # 使用权重调整actor_loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络


