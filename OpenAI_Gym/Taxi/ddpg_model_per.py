import torch
from torch import nn
import random
import numpy as np
import collections
from pyinstrument import Profiler


class ReplayBuffer:
    def __init__(self, capacity, device='cpu', IF_PER=False):
        self.buffer = collections.deque(maxlen=capacity) 
        self.device = device

        self.IF_PER = IF_PER
        self.max_td_error = None

    def add(self, state, action, reward, next_state, done, ddpg_model=None):

        state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device).unsqueeze(0)
        action = torch.tensor(np.array(action), dtype=torch.float32).to(self.device).unsqueeze(0)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32).to(self.device).unsqueeze(0)
        reward = torch.tensor(np.array(reward), dtype=torch.float32).to(self.device).unsqueeze(0)
        done = torch.tensor(np.array(done), dtype=torch.float32).to(self.device).unsqueeze(0)

        if self.IF_PER:
            is_new_experience = True

            if ddpg_model is None:  raise ValueError('model is None, but IF_PER is True')

            for transition in self.buffer:
                if torch.all(torch.eq(state, transition[0])) and torch.all(torch.eq(action, transition[1])) \
                    and torch.all(torch.eq(reward, transition[2])) and torch.all(torch.eq(next_state, transition[3])) \
                    and torch.all(torch.eq(done, transition[4])):
                    is_new_experience = False
                    break

            if is_new_experience and self.max_td_error is not None:
                td_error = self.max_td_error

            else:
                with torch.no_grad():
                    next_q_value = ddpg_model.target_critic(next_state, ddpg_model.target_actor(next_state))
                    q_target = reward + ddpg_model.gamma * next_q_value * (1 - done)
                    q_value = ddpg_model.critic(state, action)
                    td_error = torch.abs(q_target - q_value).cpu().numpy()[0][0]

            self.buffer.append((state, action, reward, next_state, done, td_error))
        else:
            self.buffer.append((state, action, reward, next_state, done))

        
    def sample(self, batch_size): 

        if self.IF_PER:
            # 对TD误差进行排序
            self.buffer = sorted(self.buffer, key=lambda x: x[-1], reverse=True)
            td_errors = [transition[-1] for transition in self.buffer]
            self.max_td_error = td_errors[0]

            # 计算排名的倒数
            ranks = np.arange(len(self.buffer))
            rank_inverse = 1 / (ranks + 1)

            # 根据rank_inverse设置采样概率
            probs = rank_inverse / np.sum(rank_inverse)

            # 做softmax
            probs = np.exp(probs) / np.sum(np.exp(probs))

            # 根据概率采样经验
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            transitions = [self.buffer[i] for i in indices]

            # 返回采样的批次
            state, action, reward, next_state, done, _ = zip(*transitions)

        else:
            transitions = random.sample(self.buffer, batch_size)
            state, action, reward, next_state, done = zip(*transitions)
        
        
        return state, action, reward, next_state, done 
        

    def size(self): 
        return len(self.buffer)


class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
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

        self.linear = nn.Sequential(
            nn.Linear(512, 64),
            nn.LayerNorm(64),
            nn.Mish(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        if x.shape[1] != self.input_dim:
            raise ValueError('The input dimension is not correct!')
        if x.dtype != torch.float32:
            raise ValueError('The input dimension is not correct!')
        x = self.seq1(x)

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


        self.action_linear = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.LayerNorm(128),
            nn.Mish(),
            nn.Linear(128, 512)
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
            action = np.random.uniform(0, 1, self.output_dim)
            action = torch.softmax(torch.tensor(action, dtype=torch.float32), dim=0).numpy()
        else:
            with torch.no_grad():
                state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device).unsqueeze(0)
                state = state.to(self.device)
                action = self.actor(state).cpu().numpy()[0]

        return action


    def soft_update(self, net, target_net):
        
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)



    def update(self, transition_dict):

        states = torch.cat(transition_dict['states'], dim=0)
        actions = torch.cat(transition_dict['actions'], dim=0)
        rewards = torch.cat(transition_dict['rewards'], dim=0).view(-1, 1)
        next_states = torch.cat(transition_dict['next_states'], dim=0)
        dones = torch.cat(transition_dict['dones'], dim=0).view(-1, 1)
        
        # 计算权重
        weights = torch.where(rewards < 0, torch.tensor(0.05), torch.tensor(100.0)).to(self.device)

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


