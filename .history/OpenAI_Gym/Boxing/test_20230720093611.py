import gym


if __name__ == '__main__':

    env_name = 'ALE/Boxing-v5'
    env = gym.make(env_name, render_mode='human', difficulty=3)


    for i in range(20):
        state = env.reset()
        for step in range(1000):
            action = env.action_space.sample()
            print(action)
            state, reward, done, _, _ = env.step(action)
            if done:
                break

    env.close()