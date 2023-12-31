import torch
import gym
from ddpg_conv_model_v2 import ReplayBuffer, DDPG
import process_image
from tqdm import tqdm


def train_off_policy_agent(env_name, replay_buffer, actor_lr, critic_lr, num_episodes, max_step_per_epoch, gamma, tau, minimal_size, batch_size, sigma, action_dim, action_bound, device):
    env = gym.make(env_name)
    agent = DDPG(action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)

    # 检查有没有之前训练的模型，有的话就加载
    try:
        agent.actor.load_state_dict(torch.load('OpenAI_Gym/Car_Racing/agent_v2/conv_actor_v2.pth', map_location=torch.device(device)))
        agent.critic.load_state_dict(torch.load('OpenAI_Gym/Car_Racing/agent_v2/conv_critic_v2.pth', map_location=torch.device(device)))

        agent.target_actor.load_state_dict(torch.load('OpenAI_Gym/Car_Racing/agent_v2/conv_actor_v2.pth', map_location=torch.device(device)))
        agent.target_critic.load_state_dict(torch.load('OpenAI_Gym/Car_Racing/agent_v2/conv_critic_v2.pth', map_location=torch.device(device)))
    
        print('Model loaded!')
    except:
        print('No model found!Create new model!')

    
    episode_returns = []
    
    for i in range(num_episodes):
        episode_return = 0
        state, _ = env.reset()

        continue_negative_reward_time = 0

        epoch_list = range(max_step_per_epoch)

        # 设置进度条
        pbar = tqdm(total=len(epoch_list), desc='epoch', unit='step')
        
        for step in epoch_list:
            action = agent.take_action(state)
            next_state, reward, done, _, _ = env.step(action)

            # 最开始的时候图像会从大到小变化，这时候不进行图像识别
            if step > 30:
                on_road_reward = process_image.cal_on_road_reward(next_state)
                reward += on_road_reward

                # 如果小车不在路上，那么就给一个负的 reward
                if_on_road = process_image.if_on_road(next_state)
                if not if_on_road:
                    reward -= 3

            # 如果持续 60 个 step 都是负的 reward，那么就结束这个 episode
            if reward < 0:
                continue_negative_reward_time += 1
            else:
                continue_negative_reward_time = 0

            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward

            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                agent.update(transition_dict)

            # 更新进度条的描述，描述当前的 episode 和 return
            pbar.set_description('Episode={}, return={}'.format(i, round(episode_return,2)))
            # 更新进度条的当前值
            pbar.update(1)

            if done or continue_negative_reward_time > 60:
                break

        episode_returns.append(episode_return)

        # Save the model
        if i % 10 == 0:
            torch.save(agent.actor.state_dict(), 'OpenAI_Gym/Car_Racing/agent_v2/conv_actor_v2.pth')
            torch.save(agent.critic.state_dict(), 'OpenAI_Gym/Car_Racing/agent_v2/conv_critic_v2.pth')
            print('Model saved!')


    env.close()
    return episode_returns



if __name__ == '__main__':
    actor_lr = 3e-5
    critic_lr = 3e-4
    num_episodes = 10000
    max_step_per_epoch = 2000
    gamma = 0.98
    tau = 0.005
    buffer_size = 50000
    minimal_size = 1000
    batch_size = 64
    sigma = 0.01
    env_name = 'CarRacing-v2'
    action_dim = 3
    action_bound = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    replay_buffer = ReplayBuffer(buffer_size)

    episode_returns = train_off_policy_agent(env_name, replay_buffer, actor_lr, critic_lr, num_episodes, max_step_per_epoch, gamma, tau, minimal_size, batch_size, sigma, action_dim, action_bound, device)

    

    