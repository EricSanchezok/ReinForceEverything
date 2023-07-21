import torch
from torch import nn
import random
import numpy as np
import collections
import time


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
    def __init__(self, action_dim):
        super(PolicyNet, self).__init__()
        # 输出图像大小 = (输入图像大小 - 卷积核大小 + 2 x 填充) / 步长 + 1
        # 输入尺寸为210x160x3
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),     # 210 - 3 + 2 x 1 / 1 + 1 = 210, 160 - 3 + 2 x 1 / 1 + 1 = 160
            nn.BatchNorm2d(16),
            nn.Mish(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),    # 210 - 3 + 2 x 1 / 1 + 1 = 210, 160 - 3 + 2 x 1 / 1 + 1 = 160
            nn.BatchNorm2d(32),
            nn.Mish(),
        )

        # 最大池化层
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 210 / 2 = 105, 160 / 2 = 80

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),  # 105 - 3 + 2 x 1 / 2 + 1 = 53, 80 - 3 + 2 x 1 / 2 + 1 = 40
            nn.BatchNorm2d(64),
            nn.Mish(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),  # 53 - 3 + 2 x 1 / 2 + 1 = 27, 40 - 3 + 2 x 1 / 2 + 1 = 20
            nn.BatchNorm2d(128),
            nn.Mish()
        )

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 27 / 2 = 13, 20 / 2 = 10

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, padding=1),    # 13 - 3 + 2 x 1 / 1 + 1 = 13, 10 - 3 + 2 x 1 / 1 + 1 = 10
            nn.BatchNorm2d(512),
            nn.Mish(),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),   # 13 - 3 + 2 x 1 / 1 + 1 = 13, 10 - 3 + 2 x 1 / 1 + 1 = 10
            nn.BatchNorm2d(1024),
            nn.Mish()
        )

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 13 / 2 = 6, 10 / 2 = 5

        self.linear = nn.Sequential(
            nn.Linear(1024 * 6 * 5, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, 32),
            nn.LayerNorm(32),
            nn.Mish(),
            nn.Linear(32, action_dim)
        )
        

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        # softmax
        x = torch.softmax(x, dim=1)
        return x



class QValueNet(torch.nn.Module):
    def __init__(self, action_dim):
        super(QValueNet, self).__init__()
        # 输出图像大小 = (输入图像大小 - 卷积核大小 + 2 x 填充) / 步长 + 1
        # 输入尺寸为210x160x3
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),     # 210 - 3 + 2 x 1 / 1 + 1 = 210, 160 - 3 + 2 x 1 / 1 + 1 = 160
            nn.BatchNorm2d(16),
            nn.Mish(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),    # 210 - 3 + 2 x 1 / 1 + 1 = 210, 160 - 3 + 2 x 1 / 1 + 1 = 160
            nn.BatchNorm2d(32),
            nn.Mish(),
        )

        # 最大池化层
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 210 / 2 = 105, 160 / 2 = 80

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),  # 105 - 3 + 2 x 1 / 2 + 1 = 53, 80 - 3 + 2 x 1 / 2 + 1 = 40
            nn.BatchNorm2d(64),
            nn.Mish(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),  # 53 - 3 + 2 x 1 / 2 + 1 = 27, 40 - 3 + 2 x 1 / 2 + 1 = 20
            nn.BatchNorm2d(128),
            nn.Mish()
        )

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 27 / 2 = 13, 20 / 2 = 10

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, padding=1),    # 13 - 3 + 2 x 1 / 1 + 1 = 13, 10 - 3 + 2 x 1 / 1 + 1 = 10
            nn.BatchNorm2d(512),
            nn.Mish(),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),   # 13 - 3 + 2 x 1 / 1 + 1 = 13, 10 - 3 + 2 x 1 / 1 + 1 = 10
            nn.BatchNorm2d(1024),
            nn.Mish()
        )

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 13 / 2 = 6, 10 / 2 = 5

        self.action_linear = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.LayerNorm(128),
            nn.Mish(),
            nn.Linear(128, 512)
        )

        self.linear = nn.Sequential(
            nn.Linear(1024 * 6 * 5 + 512, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, 32),
            nn.LayerNorm(32),
            nn.Mish(),
            nn.Linear(32, 1)
        )



    def forward(self, x, a):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = torch.flatten(x, start_dim=1)

        a = self.action_linear(a)
        cat = torch.concat((x, a), dim=1)
        out = self.linear(cat)

        return out
    

class DDPG:
    ''' DDPG算法 '''
    def __init__(self, action_dim, random_rate, sigma, actor_lr, critic_lr, tau, gamma, device):
        self.actor = PolicyNet(action_dim).to(device)
        self.critic = QValueNet(action_dim).to(device)
        self.target_actor = PolicyNet(action_dim).to(device)
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
        self.random_rate = random_rate
        self.device = device

        self.critic_mse_loss = nn.MSELoss().to(device)

    def take_action(self, state, noise=True):
        if random.random() < self.random_rate and noise:
            action = np.random.uniform(-1, 1, self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)

            state = state.to(self.device)

            action = self.actor(state).cpu().detach().numpy()[0]
        # 选择action中最大值的下标作为action

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
