import torch
import gym
from ddpg_model import ReplayBuffer, DDPG
from tqdm import tqdm
import numpy as np
import time


if __name__ == '__main__':
    actor_lr = 3e-5
    critic_lr = 3e-4
    num_episodes = 10000
    max_step_per_epoch = 1790
    gamma = 0.98
    tau = 0.005
    buffer_size = 50000
    minimal_size = 1000
    batch_size = 16
    sigma = 0.01

    env_name = 'ALE/Boxing-v5'

    action_dim = 18
    random_rate = 0.1
    
    difficulty = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = gym.make(env_name, difficulty=difficulty)

    replay_buffer = ReplayBuffer(buffer_size)
    agent = DDPG(action_dim, random_rate, sigma, actor_lr, critic_lr, tau, gamma, device)

    episode_returns = train_off_policy_agent(env_name, replay_buffer, agent, difficulty, num_episodes, max_step_per_epoch, minimal_size, batch_size, device)

    

    