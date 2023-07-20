import gym
from ale_py import ALEInterface

# from ale_py.roms import Breakout



ale = ALEInterface()
# ale.loadROM(Breakout)


if __name__ == '__main__':

    env_name = 'ALE/Boxing-v5'
    env = gym.make(env_name, render_mode='human')

    for i in range(20):
        state = env.reset()
        for step in range(1000):
            action = env.action_space.sample()
            state, reward, done, _, _ = env.step(action)
            if done:
                break

    env.close()