import torch
import gym
from ddpg_model_per import ReplayBuffer, DDPG
from tqdm import tqdm
import numpy as np


def train_off_policy_agent(env_name, replay_buffer, agent, num_episodes, max_step_per_epoch, minimal_size, batch_size, device, model_path):
    
    env = gym.make(env_name)
    
    # 检查有没有之前训练的模型，有的话就加载
    try:
        agent.actor.load_state_dict(torch.load(model_path[0], map_location=torch.device(device)))
        agent.critic.load_state_dict(torch.load(model_path[1], map_location=torch.device(device)))

        agent.target_actor.load_state_dict(torch.load(model_path[0], map_location=torch.device(device)))
        agent.target_critic.load_state_dict(torch.load(model_path[1], map_location=torch.device(device)))
    
        print('Model loaded!')
    except:
        torch.save(agent.actor.state_dict(), model_path[0])
        torch.save(agent.critic.state_dict(), model_path[1])
        print('No model found!Create new model!')

    
    episode_returns = []
    
    for i in range(num_episodes):
        episode_return = 0
        state, info = env.reset()
        state = np.eye(500)[state]

        epoch_list = range(max_step_per_epoch)

        # 设置进度条
        pbar = tqdm(total=len(epoch_list), desc='epoch', unit='step')
        
        for step in epoch_list:
            
            action = agent.take_action(state, noise=True) * info['action_mask']

            next_state, reward, done, truncated, info = env.step(np.argmax(action))
            next_state = np.eye(500)[next_state]

            replay_buffer.add(state, action, reward, next_state, done, agent)
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
        if i % 100 == 0:
            torch.save(agent.actor.state_dict(), model_path[0])
            torch.save(agent.critic.state_dict(), model_path[1])


    env.close()
    return episode_returns



if __name__ == '__main__':
    actor_lr = 0.0001
    critic_lr = 0.001
    num_episodes = 20000
    max_step_per_epoch = 200
    gamma = 0.98
    tau = 0.005
    buffer_size = 5000
    minimal_size = 1000
    batch_size = 512
    sigma = 0.01

    env_name = 'Taxi-v3'

    input_dim = 500
    output_dim = 6

    random_rate = 0.1


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = ['OpenAI_Gym/Taxi/agent/actor_v2.pth', 'OpenAI_Gym/Taxi/agent/critic_v2.pth']

    replay_buffer = ReplayBuffer(buffer_size, device=device, IF_PER=True)
    agent = DDPG(input_dim, output_dim, random_rate, sigma, actor_lr, critic_lr, tau, gamma, device)

    episode_returns = train_off_policy_agent(env_name, replay_buffer, agent, num_episodes, max_step_per_epoch, minimal_size, batch_size, device, model_path)

    

    