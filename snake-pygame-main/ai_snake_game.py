from SnakeGame import SnakeGame
from agent import Agent
import numpy as np
import matplotlib.pyplot as plt
from SnakeGame import frame_size_x, frame_size_y

plt.switch_backend('TkAgg')
EPISODES = 1500
BATCH_SIZE = 512
EPSILON_DECAY = 0.997
EPSILON_MIN = 0.01

def plot_results(episode_numbers, avg_rewards, scores):
    """
    Plots the average rewards and scores over episodes and saves the plot as an image. Used for debugging progress and changes to agent
    """
    plt.figure(figsize=(12, 6))
    plt.clf()
    plt.title("Agent Training Progress")
    plt.xlabel("Episodes")
    plt.ylabel("Scores / Avg Rewards")
    plt.plot(episode_numbers, avg_rewards, label="Avg Rewards")
    plt.plot(episode_numbers, scores[:len(episode_numbers)], label="Scores")
    plt.legend()
    plt.grid()
    plt.savefig(f"plot_episode_{episode_numbers[-1]}.png")
    plt.show()
    plt.close()

def get_state(game):
    """
    Converts the game state into a format suitable for the agent. Includes direction, food position, and danger zones--state representation of the game
    """
    snake_x, snake_y = game.snake_pos
    food_x, food_y = game.food_pos

    # Direction
    direction_up = game.direction == 'UP'
    direction_down = game.direction == 'DOWN'
    direction_left = game.direction == 'LEFT'
    direction_right = game.direction == 'RIGHT'

    # Food relative position
    food_left = food_x < snake_x
    food_right = food_x > snake_x
    food_up = food_y < snake_y
    food_down = food_y > snake_y

    # Danger zones
    danger_up = (snake_y - 10 < 0) or ([snake_x, snake_y - 10] in game.snake_body)
    danger_down = (snake_y + 10 >= frame_size_y) or ([snake_x, snake_y + 10] in game.snake_body)
    danger_left = (snake_x - 10 < 0) or ([snake_x - 10, snake_y] in game.snake_body)
    danger_right = (snake_x + 10 >= frame_size_x) or ([snake_x + 10, snake_y] in game.snake_body)

    # Normalized distances
    dist_to_food_x = (food_x - snake_x) / frame_size_x
    dist_to_food_y = (food_y - snake_y) / frame_size_y

    return [
        int(direction_up), int(direction_down), int(direction_left), int(direction_right),
        int(food_left), int(food_right), int(food_up), int(food_down),
        int(danger_up), int(danger_down), int(danger_left), int(danger_right),
        dist_to_food_x, dist_to_food_y
    ]

def train_agent():
    """
    Trains the agent using the SnakeGame environment
    """
    game = SnakeGame()
    agent = Agent(input_size=14, hidden_size=128, output_size=4)
    epsilon = agent.epsilon
    total_rewards = []
    scores = []
    avg_rewards = []
    episode_numbers = []

    # Episode training loop
    for episode in range(1, EPISODES + 1):
        game.reset()
        state = get_state(game)
        total_reward = 0
        game_over = False

        while not game_over:
            action = agent.select_action(state, epsilon)
            direction_map = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            game.change_to = direction_map[action]

            game_over, score, reward = game.play()
            if game_over:
                scores.append(score)
                game.reset()
            next_state = get_state(game)

            if not game_over:
                # Reward if agent gets food
                if abs(next_state[12]) < abs(state[12]) or abs(next_state[13]) < abs(state[13]):
                    reward += 1
                # "Punishment" for no food but staying alive at least
                else:
                    reward -= 1

            # Punishment for hitting itself or the walls, resulting in the game ending
            if game_over:
                reward -= 10

            # Have the agent remember information
            agent.remember(state, action, reward, next_state, game_over)
            state = next_state
            total_reward += reward

            # Train agent from memory
            agent.train_from_memory(BATCH_SIZE)

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        total_rewards.append(total_reward) # Data used for graphing later on


        # Save the experiences/learned knowledge of the agent every episode (this way we can terminate the program whenever we want without having to wait to do so until like episode 50)
        if episode % 1 == 0:
            avg_score = np.mean(scores[-50:])
            agent.save_model(epsilon, episode, avg_score, score, filepath="snake_model.npy")

        # Print out progress every 50 episodes
        if episode % 50 == 0:
            avg_reward = np.mean(total_rewards[-50:])
            avg_rewards.append(avg_reward)
            avg_score = np.mean(scores[-50:])
            episode_numbers.append(episode)
            print(f"Episode {episode}/{EPISODES}: Avg Reward = {avg_reward}, Average Score = {avg_score}")
            #agent.save_model(epsilon, episode, avg_score, score, filepath="snake_model.npy")
            print(f"Model saved at episode {episode}")


    # Print final plot (for debugging) and finally save the agent's experiences
    agent.save_model(epsilon, episode, avg_score,score, filepath="snake_model.npy")
    plot_results(range(1, EPISODES + 1), total_rewards, scores)



if __name__ == "__main__":
    train_agent()