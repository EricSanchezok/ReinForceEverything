import gym
import keyboard
import multiprocessing
import time
import numpy as np

from ddpg_model import ReplayBuffer
import os


def monitor_keyboard():
    vector= np.zeros(2)

    if keyboard.is_pressed('w') or keyboard.is_pressed('W'):
        vector += np.array([0, 1])
    if keyboard.is_pressed('s') or keyboard.is_pressed('S'):
        vector += np.array([0, -1])
    if keyboard.is_pressed('a') or keyboard.is_pressed('A'):
        vector += np.array([-1, 0])
    if keyboard.is_pressed('d') or keyboard.is_pressed('D'):
        vector += np.array([1, 0])
    
    punch = True if keyboard.is_pressed('j') or keyboard.is_pressed('J') else False# 打拳 按一下打一拳，长按打一击长拳
    

    if vector[0] > 0 and vector[1] > 0:
        key = 14 if punch else 6 # 右上方移动
    elif vector[0] < 0 and vector[1] > 0:
        key = 15 if punch else 7 # 左上方移动
    elif vector[0] > 0 and vector[1] < 0:
        key = 16 if punch else 8 # 右下方移动
    elif vector[0] < 0 and vector[1] < 0:
        key = 17 if punch else 9 # 左下方移动
    elif vector[0] > 0 and vector[1] == 0:
        key = 11 if punch else 3 # 向右移动
    elif vector[0] < 0 and vector[1] == 0:
        key = 12 if punch else 4 # 向左移动
    elif vector[0] == 0 and vector[1] > 0:
        key = 10 if punch else 2 # 向上移动
    elif vector[0] == 0 and vector[1] < 0:
        key = 13 if punch else 5 # 向下移动

    elif vector[0] == 0 and vector[1] == 0:
        key = 1 if punch else 0

    return key
        


if __name__ == '__main__':
    buffer_size = 1000000

    env_name, difficulty = 'ALE/Boxing-v5', 1
    env = gym.make(env_name, difficulty=difficulty, render_mode='human')
    state, _ = env.reset()

    replay_buffer = ReplayBuffer(buffer_size)

    epoch_return = 0
    path = 'OpenAI_Gym/Boxing/player_data/' 
    
    while True:
        action = monitor_keyboard()
        
        next_state, reward, done, truncated, _ = env.step(action)

        replay_buffer.add(state, action, reward, next_state, done)
        epoch_return += reward

        state = next_state

        if done or truncated:
             
            # 读取所有文件的名字
            file_list = os.listdir(path)
            if file_list == []:
                i = 0
            else:
                # 根据文件名字的第一个数字排序
                file_list.sort(key=lambda x: int(x.split('_')[0]))
                # 读取最后一个文件的第一个数字
                last_file_num = int(file_list[-1].split('_')[0])
                i = last_file_num + 1
            replay_buffer.save(path + str(i) + '_' + str(difficulty) + '_' + str(round(epoch_return, 2)) + '.npy')
            print('Save data to ' + path + str(i) + '_' + str(difficulty) + '_' + str(round(epoch_return, 2)) + '.npy')

            break
    env.close()

    