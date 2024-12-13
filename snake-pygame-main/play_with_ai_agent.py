import numpy as np
from SnakeGame import SnakeGame
from model import Qnet
import os
from training_agent import get_state

class PretrainedAgent:
    def __init__(self, model_filepath="snake_model.npy"):
        """
        Initializes the pretrained agent by loading a saved model.
        """
        self.q_net = Qnet(input_size=14, hidden_size=128, output_size=4)
        if model_filepath and os.path.exists(model_filepath):
            model_params = np.load(model_filepath, allow_pickle=True).item()
            self.q_net.weight1 = model_params["weight1"]
            self.q_net.bias1 = model_params["bias1"]
            self.q_net.weight2 = model_params["weight2"]
            self.q_net.bias2 = model_params["bias2"]
            print(f"Model loaded from {model_filepath}")
        else:
            raise FileNotFoundError(f"Model file not found at {model_filepath}")

    def select_action(self, state):
        """
        Selects the best action based on the current state using the Q-network.
        """
        q_values = self.q_net.forward(np.array([state]))
        action = np.argmax(q_values)
        return action

    def play_game(self):
        """
        Plays the Snake game using the pretrained agent without modifying the model.
        """
        game = SnakeGame()
        game.reset()
        state = get_state(game)
        game_over = False

        while not game_over:
            action = self.select_action(state)

            direction_map = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            game.change_to = direction_map[action]

            game_over, score, reward = game.play()

            if game_over:
                print(f"Game Over! Final Score: {score}")
                break

            state = get_state(game)

# Example usage
if __name__ == "__main__":
    pretrained_agent = PretrainedAgent(model_filepath="snake_model.npy")
    pretrained_agent.play_game()
