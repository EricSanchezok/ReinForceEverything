import torch
from torch import nn
import random
import numpy as np
import collections


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
    def __init__(self, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.action_bound = action_bound
        # 输出图像大小 = (输入图像大小 - 卷积核大小 + 2 x 填充) / 步长 + 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),# 96 - 3 + 2 x 1 / 1 + 1 = 96
            nn.BatchNorm2d(16),
            nn.Mish(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),# 96 - 3 + 2 x 1 / 1 + 1 = 96
            nn.BatchNorm2d(32),
            nn.Mish(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, padding=1, stride=2),# 96 - 3 + 2 x 1 / 2 + 1 = 48
            nn.BatchNorm2d(128),
            nn.Mish(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),# 48 - 3 + 2 x 1 / 2 + 1 = 24
            nn.BatchNorm2d(256),
            nn.Mish()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),# 24 - 3 + 2 x 1 / 2 + 1 = 12
            nn.BatchNorm2d(512),
            nn.Mish(),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=2),# 12 - 3 + 2 x 1 / 2 + 1 = 6
            nn.BatchNorm2d(1024),
            nn.Mish()
        )

        self.linear = nn.Sequential(
            nn.Linear(1024 * 6 * 6, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, 64),
            nn.LayerNorm(64),
            nn.Mish(),
            nn.Linear(64, action_dim)
        )
        

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = torch.tanh(x) * self.action_bound

        return x



class QValueNet(torch.nn.Module):
    def __init__(self, action_dim):
        super(QValueNet, self).__init__()
        # 输出图像大小 = (输入图像大小 - 卷积核大小 + 2 x 填充) / 步长 + 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),# 96 - 3 + 2 x 1 / 1 + 1 = 96
            nn.BatchNorm2d(16),
            nn.Mish(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),# 96 - 3 + 2 x 1 / 1 + 1 = 96
            nn.BatchNorm2d(32),
            nn.Mish(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),# 96 - 3 + 2 x 1 / 2 + 1 = 48
            nn.BatchNorm2d(64),
            nn.Mish(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),# 48 - 3 + 2 x 1 / 2 + 1 = 24
            nn.BatchNorm2d(128),
            nn.Mish()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),# 24 - 3 + 2 x 1 / 2 + 1 = 12
            nn.BatchNorm2d(256),
            nn.Mish(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),# 12 - 3 + 2 x 1 / 2 + 1 = 6
            nn.BatchNorm2d(512),
            nn.Mish()
        )

        self.action_linear = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.LayerNorm(64),
            nn.Mish(),
            nn.Linear(64, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, 512)
        )

        self.linear = nn.Sequential(
            nn.Linear(512 * 6 * 6 + 512, 512),
            nn.LayerNorm(512),
            nn.Mish(),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.Mish(),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.Mish(),
            nn.Linear(32, 1)
        )


    def forward(self, x, a):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)


        a = self.action_linear(a)
        cat = torch.concat((x, a), dim=1)
        out = self.linear(cat)

        return out
    

    

class DDPG:
    ''' DDPG算法 '''
    def __init__(self, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device):
        self.actor = PolicyNet(action_dim, action_bound).to(device)
        self.critic = QValueNet(action_dim).to(device)
        self.target_actor = PolicyNet(action_dim, action_bound).to(device)
        self.target_critic = QValueNet(action_dim).to(device)

        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device

        self.critic_mse_loss = nn.MSELoss().to(device)

    def take_action(self, state, noise=True):
        state = torch.FloatTensor(state).unsqueeze(0)

        state = state.to(self.device)

        action = self.actor(state).detach().cpu().numpy()[0]
        # 给动作添加噪声，增加探索
        if noise:
            action = action + self.sigma * np.random.randn(self.action_dim)

        # 设置action[0]的范围为[-1,1]
        action[0] = max(min(action[0], 1), -1)
        # 设置action[1]的范围为[0,1]
        action[1] = max(min(action[1], 1), 0)
        # 设置action[2]的范围为[0,1]
        action[2] = max(min(action[2], 1), 0)

        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)

        # 计算权重
        weights = torch.where(rewards < 0, torch.tensor(3.0), torch.tensor(1.0)).to(self.device)

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