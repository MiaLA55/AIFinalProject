import random
import numpy as np
from model import Qnet, Trainer
import os


class Agent:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.0007, gamma=0.95):
        """
        Initializes Q-network, trainer, replay memory, and other parameters; if the agent already has "experience" (e.g. weights & biases) just load that from the snake_model.npy to pick up the agent from where it left off last in its learning experience
        """
        self.q_net = Qnet(input_size, hidden_size, output_size)
        if os.path.exists("snake_model.npy"):
            model_params = np.load("snake_model.npy", allow_pickle=True).item()
            self.q_net.weight1 = model_params["weight1"]
            self.q_net.bias1 = model_params["bias1"]
            self.q_net.weight2 = model_params["weight2"]
            self.q_net.bias2 = model_params["bias2"]
            self.epsilon = model_params["epsilon"]
        else:
            self.epsilon = 1

        self.trainer = Trainer(self.q_net, learning_rate, gamma)
        self.memory = []
        self.max_memory_size = 100_000
        self.gamma = gamma

    def select_action(self, state, epsilon):
        """
        Select action for agent based on current state and exploration rate
        """
        if np.random.rand() < epsilon:
            action = random.randint(0, 3)
        else:
            q_values = self.q_net.forward(np.array([state]))
            action = np.argmax(q_values)
        return action

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory
        """
        if len(self.memory) >= self.max_memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def train_from_memory(self, batch_size):
        """
        Train Q-network using a random mini-batch from memory
        """
        if len(self.memory) < batch_size:
            return
        mini_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in mini_batch:
            self.train_step(state, action, reward, next_state, done)

    def train_step(self, state, action, reward, next_state, done):
        """
        Perform single training step to update Q-network (via Bellman equation) based on agent's experience
        """
        state = np.array([state])
        next_state = np.array([next_state])
        target = self.q_net.forward(state)

        if done:
            target[0][action] = reward
        else:
            q_next = np.max(self.q_net.forward(next_state))
            target[0][action] = reward + self.gamma * q_next

        self.trainer.train_model(state, target)

    def save_model(self, epsilon, episodes, average_score, score, filepath="snake_model.npy"):
        """
        Saves important variables for agent to remember from game to game and run to run; this information can later be reloaded when run again so agent can remember past training intervals to help improve its score
        """
        total_episodes = 0
        if os.path.exists("snake_model.npy"):
            model_params = np.load(filepath, allow_pickle=True).item()
            total_episodes = model_params["episodes"] + episodes # Used to help us debug and see how agent has improved from run to run instead of episode to episode
        model_params = {
            "weight1": self.q_net.weight1,
            "bias1": self.q_net.bias1,
            "weight2": self.q_net.weight2,
            "bias2": self.q_net.bias2,
            "epsilon": epsilon,
            "episodes": total_episodes,
        }
        self.save_episodes_scores("episodes_scores.npy", total_episodes, average_score, score) # Used to help us debug and see how agent has improved from run to run instead of episode to episode
        np.save(filepath, model_params)

    def load_model(self, filepath="snake_model.npy"):
        """
        Load model again but with its known experiences so agent can just pick up the experience it had when it left off
        """
        if os.path.exists(filepath):
            model_params = np.load(filepath, allow_pickle=True).item()
            self.q_net.weight1 = model_params["weight1"]
            self.q_net.bias1 = model_params["bias1"]
            self.q_net.weight2 = model_params["weight2"]
            self.q_net.bias2 = model_params["bias2"]
            self.epsilon = model_params["epsilon"]
            self.episodes = model_params["episodes"]
            print(f"Model loaded from {filepath}")
        else:
            print("No saved model found, starting from scratch.")

    def save_episodes_scores(self, file_name, total_episodes, average_score, score):
        """
        Primarily used for debugging, this allows us to see episodes and their correlations to scores, namely high scores; this method generates a file with that data for us to analyze
        """
        if os.path.exists(file_name):
            data = np.load(file_name, allow_pickle=True).item()
            highest_score = data.get("highest_score", 0)
        else:
            data = {"total_episodes": 0, "average_scores": 0, "highest_score": 0}
            highest_score = 0

        # Update the highest score if the current score is higher
        if score > highest_score:
            highest_score = score

        # Update the data dictionary with new values
        data["total_episodes"] = total_episodes
        data["average_scores"] = average_score
        data["highest_score"] = highest_score

        # Save the updated data back to the file
        np.save(file_name, data)

        # Write the data entry to the text file
        data_entry = f"Total episodes: {total_episodes}, Average score: {average_score}, Highest score: {highest_score}\n"
        with open("dataepisodes.txt", "a") as f:
            f.write(data_entry)




