import gym



if __name__ == '__main__':
    env_name = 'ALE/Boxing-v5'

    
    difficulty = 0

    env = gym.make(env_name, difficulty=difficulty)
    state, _ = env.reset()
    
    while True:

        action = env.action_space.sample()
        next_state, reward, done, truncated, _ = env.step(action)

        if done or truncated:
            break

    env.close()

    