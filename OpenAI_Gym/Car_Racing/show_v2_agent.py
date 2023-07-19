import torch
import gym
import torch
from torch import nn

from ddpg_conv_model_v2 import DDPG

import process_image



if __name__ == '__main__':

    actor_lr = 0.0001
    critic_lr = 0.001
    gamma = 0.98
    tau = 0.005
    sigma = 0.01
    env_name = 'CarRacing-v2'
    action_dim = 3
    action_bound = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(env_name, render_mode='human')
    agent = DDPG(action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)

    agent.actor.load_state_dict(torch.load('agent_v2/conv_actor_v2.pth', map_location=torch.device(device)))
    agent.critic.load_state_dict(torch.load('agent_v2/conv_critic_v2.pth', map_location=torch.device(device)))

    agent.target_actor.load_state_dict(torch.load('agent_v2/conv_actor_v2.pth', map_location=torch.device(device)))
    agent.target_critic.load_state_dict(torch.load('agent_v2/conv_critic_v2.pth', map_location=torch.device(device)))


    for i in range(20):
        state, _ = env.reset()
        continue_negative_reward_time = 0
        for step in range(1000):

            action = agent.take_action(state, noise=False)
            state, reward, done, _, _ = env.step(action)

            if step > 30:
                on_road_reward = process_image.cal_on_road_reward(state)
                reward += on_road_reward

                if_on_road = process_image.if_on_road(state)
                if not if_on_road:
                    reward -= 3
            
            if reward < 0:
                continue_negative_reward_time += 1
            else:
                continue_negative_reward_time = 0

            if done or continue_negative_reward_time > 60:
                break

    env.close()

    