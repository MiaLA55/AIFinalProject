import numpy as np
import os

class LinearQnet:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases for a simple feedforward neural network
        self.weight1 = np.random.rand(input_size, hidden_size) - 0.5
        self.bias1 = np.random.rand(hidden_size) - 0.5
        self.weight2 = np.random.rand(hidden_size, output_size) - 0.5
        self.bias2 = np.random.rand(output_size) - 0.5

    def forward(self, x):
        # Forward pass through the network
        z1 = np.dot(x, self.weight1) + self.bias1
        a1 = np.maximum(z1, 0)
        z2 = np.dot(a1, self.weight2) + self.bias2
        return z2

class Trainer:
    def __init__(self, model, learning_rate, gamma):
        # Initialize trainer with model, learning rate, and discount factor
        self.model = model
        self.learning_rate = learning_rate
        self.gamma = gamma

    def train_model(self, state, target):
        # Forward pass
        z1 = np.dot(state, self.model.weight1) + self.model.bias1
        a1 = np.maximum(z1, 0)
        z2 = np.dot(a1, self.model.weight2) + self.model.bias2

        # Compute loss and gradients
        loss = target - z2
        grad_z2 = -2 * loss
        grad_weight2 = np.dot(a1.T, grad_z2)
        grad_bias2 = np.sum(grad_z2, axis=0)

        grad_a1 = np.dot(grad_z2, self.model.weight2.T)
        grad_z1 = grad_a1 * (z1 > 0)
        grad_weight1 = np.dot(state.T, grad_z1)
        grad_bias1 = np.sum(grad_z1, axis=0)

        # Update weights and biases
        self.model.weight2 -= self.learning_rate * grad_weight2
        self.model.bias2 -= self.learning_rate * grad_bias2
        self.model.weight1 -= self.learning_rate * grad_weight1
        self.model.bias1 -= self.learning_rate * grad_bias1
