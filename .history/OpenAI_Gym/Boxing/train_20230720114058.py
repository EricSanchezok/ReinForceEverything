import torch
import gym
from ddpg_model import ReplayBuffer, DDPG
from tqdm import tqdm


def train_off_policy_agent(env_name, replay_buffer, actor_lr, critic_lr, num_episodes, max_step_per_epoch, gamma, tau, minimal_size, batch_size, sigma, action_dim, action_bound, device):
    
    env = gym.make(env_name, render_mode='human', difficulty=difficulty)
    
    # 检查有没有之前训练的模型，有的话就加载
    try:
        agent.actor.load_state_dict(torch.load('agent/actor_v1.pth', map_location=torch.device(device)))
        agent.critic.load_state_dict(torch.load('agent/critic_v1.pth', map_location=torch.device(device)))

        agent.target_actor.load_state_dict(torch.load('agent/actor_v1.pth', map_location=torch.device(device)))
        agent.target_critic.load_state_dict(torch.load('agent/critic_v1.pth', map_location=torch.device(device)))
    
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
            next_state, reward, done, truncated, _ = env.step(action)


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

            if done or truncated:
                break

        episode_returns.append(episode_return)

        # Save the model
        if i % 10 == 0:
            torch.save(agent.actor.state_dict(), 'agent/actor_v1.pth')
            torch.save(agent.critic.state_dict(), 'agent/critic_v1.pth')
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
    action_dim = 18
    random_rate = 0.1
    
    difficulty = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    replay_buffer = ReplayBuffer(buffer_size)
    agent = DDPG(action_dim, random_rate, sigma, actor_lr, critic_lr, tau, gamma, device)

    episode_returns = train_off_policy_agent(env_name, replay_buffer, actor_lr, critic_lr, num_episodes, max_step_per_epoch, gamma, tau, minimal_size, batch_size, sigma, action_dim, action_bound, device)

    

    