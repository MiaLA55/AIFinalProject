import random
import numpy as np
from model import LinearQnet, Trainer


class Agent:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.0005, gamma=0.95):
        # Initialize Q-network, trainer, replay memory, and parameters
        self.q_net = LinearQnet(input_size, hidden_size, output_size)
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
