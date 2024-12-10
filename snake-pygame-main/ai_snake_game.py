import pygame
from SnakeGame import SnakeGame
from agent import Agent
import numpy as np
import matplotlib.pyplot as plt
from SnakeGame import frame_size_x, frame_size_y


plt.switch_backend('TkAgg')


EPISODES = 500
BATCH_SIZE = 64
EPSILON_DECAY = 0.997
EPSILON_MIN = 0.01


def plot_results(episode_numbers, avg_rewards, scores):
    # Plots the average rewards and scores over episodes and saves the plot as an image.
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
    plt.show()  # Add this to display the plot interactively
    plt.close()


def get_state(game):
    # Converts the game state into a format suitable for the agent. Includes direction, food position, and danger zones.
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
    # Trains the agent using the SnakeGame environment.
    game = SnakeGame()
    agent = Agent(input_size=14, hidden_size=128, output_size=4)
    epsilon = 1.0
    total_rewards = []
    scores = []
    avg_rewards = []
    episode_numbers = []

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
                if abs(next_state[12]) < abs(state[12]) or abs(next_state[13]) < abs(state[13]):
                    reward += 1
                else:
                    reward -= 1

            if game_over:
                reward -= 10

            agent.remember(state, action, reward, next_state, game_over)
            state = next_state
            total_reward += reward

            agent.train_from_memory(BATCH_SIZE)

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        total_rewards.append(total_reward)


        # Average reward calculation for the current episode window
        if episode % 50 == 0:
            avg_reward = np.mean(total_rewards[-50:])
            avg_rewards.append(avg_reward)
            avg_score = np.mean(scores[-50:])
            episode_numbers.append(episode)
            print(f"Episode {episode}/{EPISODES}: Avg Reward = {avg_reward}, Average Score = {avg_score}")

        # Plot results after every episode
        if episode % 50 == 0 or episode == EPISODES:  # Plot every 10 episodes
            plot_results(range(1, episode + 1), total_rewards, scores)

    # Final plot
    plot_results(range(1, EPISODES + 1), total_rewards, scores)




if __name__ == "__main__":
    train_agent()