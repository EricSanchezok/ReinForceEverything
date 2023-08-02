import gym
import networkx as nx
import matplotlib.pyplot as plt
import cv2

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


def print_taxi_state(env):
    taxi_coords, passenger_coords, dest_coords = get_taxi_coordinates(env)
    print(f"Taxi coordinates: {taxi_coords}")
    print(f"Passenger coordinates: {passenger_coords}")
    print(f"Destination coordinates: {dest_coords}")


def mapping(i, j):
    return int(i - 1), int((j - 1) / 2)

def create_custom_taxi_environment_map():
    map_str = [
        "+---------+",
        "|R: | : :G|",
        "| : | : : |",
        "| : : : : |",
        "| | : | : |",
        "|Y| : |B: |",
        "+---------+",
    ]

    rows = len(map_str)
    cols = len(map_str[0])

    G = nx.Graph()

    for i in range(rows):
        for j in range(cols):
            if map_str[i][j] in ['R', 'G', 'Y', 'B', ' ']:
                # Add nodes for passable spaces (RGBY and empty spaces)
                m, n = mapping(i, j)
                G.add_node((m, n))
                # Check adjacent nodes to add edges between passable spaces
                if i - 1 >= 0 and map_str[i - 1][j] in ['R', 'G', 'Y', 'B', ' ']:
                    p, q = mapping(i - 1, j)
                    G.add_edge((m, n), (p, q))
                if i + 1 < rows and map_str[i + 1][j] in ['R', 'G', 'Y', 'B', ' ']:
                    p, q = mapping(i + 1, j)
                    G.add_edge((m, n), (p, q))
                if j - 2 >= 0 and map_str[i][j - 1] in [':'] and map_str[i][j - 2] in ['R', 'G', 'Y', 'B', ' ']:
                    p, q = mapping(i, j - 2)
                    G.add_edge((m, n), (p, q))
                if j + 2 < cols and map_str[i][j + 1] in [':'] and map_str[i][j + 2] in ['R', 'G', 'Y', 'B', ' ']:
                    p, q = mapping(i, j + 2)
                    G.add_edge((m, n), (p, q))

    return G

def draw_custom_taxi_environment_map(taxi_map, shortest_path, taxi):
    # Draw the graph with the shortest path
    pos = {(node[0], node[1]): (node[1], -node[0]) for node in taxi_map.nodes()}
    nx.draw(taxi_map, pos, node_size=500, node_color='lightblue', with_labels=False)

    # 设置起始点为绿色，路径为红色
    nx.draw_networkx_nodes(taxi_map, pos, nodelist=shortest_path, node_color='red', node_size=400)
    nx.draw_networkx_nodes(taxi_map, pos, nodelist=[shortest_path[0]], node_color='lightgreen', node_size=500)
    # 绘制小车位置，蓝色
    nx.draw_networkx_nodes(taxi_map, pos, nodelist=[taxi], node_color='blue', node_size=500)


def from_path_to_action(shortest_path, env):
    taxi, _, _ = get_taxi_coordinates(env)

    for i in range(0, len(shortest_path) - 1):
        if shortest_path[i] == taxi:
            if shortest_path[i + 1] == (taxi[0] + 1, taxi[1]):
                action = 0
            elif shortest_path[i + 1] == (taxi[0] - 1, taxi[1]):
                action = 1
            elif shortest_path[i + 1] == (taxi[0], taxi[1] + 1):
                action = 2
            elif shortest_path[i + 1] == (taxi[0], taxi[1] - 1):
                action = 3

            return taxi, action
        
    return taxi, 'finish'


def play_randomly(env):
    env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()  # Choose a random action
        next_state, reward, done, truncated, _  = env.step(action)
        total_reward += reward
        env.render()
        print_taxi_state(env)

    print(f"Total reward: {total_reward}")

def play_automatically(env, draw_path=False, cv_delay=1, plt_delay=1):

    env.reset()

    taxi_coordinates, passenger_coordinates, dest_coordinates = get_taxi_coordinates(env)
    

    taxi_map = create_custom_taxi_environment_map()

    task = 'pick up'
    # Use A* algorithm to find the shortest path
    shortest_path = nx.astar_path(taxi_map, taxi_coordinates, passenger_coordinates)

    # Print the shortest path
    print("Shortest Path:", shortest_path)

    while True:

        taxi, action = from_path_to_action(shortest_path, env)

        if draw_path:
            draw_custom_taxi_environment_map(taxi_map, shortest_path, taxi)

        img = env.render()
        cv2.imshow('Taxi', img)
        
        if draw_path:
            key = cv2.waitKey(1)
            plt.pause(plt_delay)

        else:
            key = cv2.waitKey(cv_delay)

        if action == 'finish':

            if task == 'pick up':
                print('Finish! Ready to pick up passenger!')
                action = 4
                task = 'drop off'
                taxi_coordinates, passenger_coordinates, dest_coordinates = get_taxi_coordinates(env)
                shortest_path = nx.astar_path(taxi_map, taxi_coordinates, dest_coordinates)


            elif task == 'drop off':
                print('Finish! Ready to drop off passenger!')
                action = 5

        next_state, reward, done, truncated, _  = env.step(action)


        if done or truncated or key == ord('q'):

            break


if __name__ == '__main__':


    env = gym.make('Taxi-v3', render_mode='rgb_array')

    plt.figure(figsize=(11, 8))
    plt.title('Taxi Environment', fontsize=40)
    cv2.namedWindow('Taxi')

    for i in range(1000):
        play_automatically(env, draw_path=True, cv_delay=30, plt_delay=1.0)




