import torch
import gym
from agent_model import DDPG
from replay_buffer import ReplayBuffer
from tqdm import tqdm
import numpy as np

def idx_to_coordinates(idx):
    if idx == 0:
        return (0, 0)
    elif idx == 1:
        return (0, 4)
    elif idx == 2:
        return (4, 0)
    elif idx == 3:
        return (4, 3)
    else:
        return 'On taxi'

def get_taxi_coordinates(env):
    taxi_row, taxi_col, passenger_idx, dest_idx = env.decode(env.s)
    taxi_coordinates = (int(taxi_row), int(taxi_col))
    
    passenger_coordinates = idx_to_coordinates(passenger_idx)
    dest_coordinates = idx_to_coordinates(dest_idx)
    return taxi_coordinates, passenger_coordinates, dest_coordinates

def process_state(state):
    state = np.eye(500)[state]
    state = np.exp(state)
    state = state / np.sum(state)
    return state

def process_coordinates(taxi_coordinates, passenger_coordinates, dest_coordinates):
    if passenger_coordinates == 'On taxi':
        passenger_coordinates = (1e-4, 1e-4)
    return np.concatenate((taxi_coordinates, passenger_coordinates, dest_coordinates), axis=0, dtype=np.float32)


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
        state = process_state(state)
        taxi_coordinates, passenger_coordinates, dest_coordinates = get_taxi_coordinates(env)
        coords = process_coordinates(taxi_coordinates, passenger_coordinates, dest_coordinates)
        state = np.concatenate((state, coords), axis=0, dtype=np.float32)

        epoch_list = range(max_step_per_epoch)

        # 设置进度条
        pbar = tqdm(total=len(epoch_list), desc='epoch', unit='step')

        is_get_on = False
        
        for step in epoch_list:
            
            action = agent.take_action(state, noise=True)
            action = torch.softmax(action, dim=0) * info['action_mask']
            action = action.numpy()
            # 软化action
            action = np.exp(action)
            action = action / np.sum(action)

            next_state, reward, done, truncated, info = env.step(np.argmax(action))

            # 稳定action
            action = np.eye(6)[np.argmax(action)]
            action = np.exp(action)
            action = action / np.sum(action)

            next_state = process_state(next_state)

            taxi_coordinates, passenger_coordinates, dest_coordinates = get_taxi_coordinates(env)
            coords = process_coordinates(taxi_coordinates, passenger_coordinates, dest_coordinates)

            next_state = np.concatenate((next_state, coords), axis=0, dtype=np.float32)

            if passenger_coordinates == 'On taxi':
                
                if is_get_on == False:
                    reward += 20
                    is_get_on = True

                manhattan_distance = abs(taxi_coordinates[0] - dest_coordinates[0]) + abs(taxi_coordinates[1] - dest_coordinates[1])

            else:
                manhattan_distance = abs(taxi_coordinates[0] - passenger_coordinates[0]) + abs(taxi_coordinates[1] - passenger_coordinates[1])

            # 距离越近，reward 越大
            reward += (1 / (manhattan_distance + 1)) * 2

            replay_buffer.add(state, action, reward, next_state, done, agent)
            state = next_state
            episode_return += reward

            if replay_buffer.size() > minimal_size:
                b_s, b_a, b_ns, b_r, b_d = replay_buffer.sample(batch_size)
                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                agent.update(transition_dict)

            # 更新进度条的描述，描述当前的 episode 和 return
            pbar.set_description('Episode={}, return={}, action={}, reward={}'.format(i, round(episode_return,3), np.argmax(action), round(reward, 3)))
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
    actor_lr = 0.00003
    critic_lr = 0.0003
    num_episodes = 50000
    max_step_per_epoch = 200
    gamma = 0.98
    tau = 0.005
    buffer_size = 50000
    minimal_size = 1000
    batch_size = 64
    sigma = 0.01

    env_name = 'Taxi-v3'

    input_dim = 500 + 6
    output_dim = 6

    random_rate = 0.15


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = ['OpenAI_Gym/Taxi/agent/actor_v1.pth', 'OpenAI_Gym/Taxi/agent/critic_v1.pth']

    replay_buffer = ReplayBuffer(buffer_size, device=device)
    agent = DDPG(input_dim, output_dim, random_rate, sigma, actor_lr, critic_lr, tau, gamma, device)

    episode_returns = train_off_policy_agent(env_name, replay_buffer, agent, num_episodes, max_step_per_epoch, minimal_size, batch_size, device, model_path)

    

    