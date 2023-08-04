import torch
import gym
import numpy as np
import keyboard

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

    if vector[0] > 0 and vector[1] == 0:
        key = 2
    elif vector[0] < 0 and vector[1] == 0:
        key = 3
    elif vector[0] == 0 and vector[1] > 0:
        key = 1
    elif vector[0] == 0 and vector[1] < 0:
        key = 0

    else:
        key = None

    return key

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


def process_coordinates(taxi_coordinates, passenger_coordinates, dest_coordinates):
    if passenger_coordinates == 'On taxi':
        passenger_coordinates = (1e-4, 1e-4)
    return np.concatenate((taxi_coordinates, passenger_coordinates, dest_coordinates), axis=0, dtype=np.float32)


def main():

    env = gym.make('Taxi-v3', render_mode='human')
    env.reset()
    
    while True:
        action = monitor_keyboard()


        if action is not None:
            next_state, reward, done, truncated, info = env.step(action)
            print(info['action_mask'])
        else:
            done = False
            truncated = False

        if done or truncated:
            print("fuckyou")
            break


    env.close()



if __name__ == '__main__':

    main()

    