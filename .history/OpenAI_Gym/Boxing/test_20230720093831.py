import gym


if __name__ == '__main__':

    env_name = 'ALE/Boxing-v5'
    env = gym.make(env_name, render_mode='human', difficulty=1)


    for i in range(20):
        state = env.reset()
        for step in range(1000):
            action = env.action_space.sample()
            state, reward, done, _, _ = env.step(action)
            print(reward)
            if done:
                break

    env.close()