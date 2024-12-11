import random
import numpy as np
from model import LinearQnet, Trainer
import os


class Agent:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.0007, gamma=0.95):
        # Initialize Q-network, trainer, replay memory, and parameters
        self.q_net = LinearQnet(input_size, hidden_size, output_size)
        if os.path.exists("snake_model.npy"):
            model_params = np.load("snake_model.npy", allow_pickle=True).item()
            self.q_net.weight1 = model_params["weight1"]
            self.q_net.bias1 = model_params["bias1"]
            self.q_net.weight2 = model_params["weight2"]
            self.q_net.bias2 = model_params["bias2"]
            self.epsilon = model_params["epsilon"]
        else:
            self.epsilon=1
        self.trainer = Trainer(self.q_net, learning_rate, gamma)
        self.memory = []
        self.max_memory_size = 10000
        self.gamma = gamma

    def select_action(self, state, epsilon):
        # Select action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = random.randint(0, 3)
        else:
            q_values = self.q_net.forward(np.array([state]))
            action = np.argmax(q_values)
        return action

    def remember(self, state, action, reward, next_state, done):
        # Store experience in replay memory
        if len(self.memory) >= self.max_memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def train_from_memory(self, batch_size):
        # Train Q-network using a random mini-batch from memory
        if len(self.memory) < batch_size:
            return
        mini_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in mini_batch:
            self.train_step(state, action, reward, next_state, done)

    def train_step(self, state, action, reward, next_state, done):
        # Update Q-network using Bellman equation
        state = np.array([state])
        next_state = np.array([next_state])
        target = self.q_net.forward(state)

        if done:
            target[0][action] = reward
        else:
            q_next = np.max(self.q_net.forward(next_state))
            target[0][action] = reward + self.gamma * q_next

        self.trainer.train_model(state, target)

    def save_model(self, epsilon, episodes, filepath="snake_model.npy"):
        # Save model parameters (weights and biases) as a dictionary
        total_episodes = 0
        if os.path.exists("snake_model.npy"):
            model_params = np.load(filepath, allow_pickle=True).item()
            total_episodes = model_params["episodes"] + episodes
        model_params = {
            "weight1": self.q_net.weight1,
            "bias1": self.q_net.bias1,
            "weight2": self.q_net.weight2,
            "bias2": self.q_net.bias2,
            "epsilon": epsilon,
            "episodes": total_episodes,
        }
        np.save(filepath, model_params)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath="snake_model.npy"):
        # Load model parameters from the dictionary
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

