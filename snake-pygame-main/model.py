import numpy as np
import os

from numpy import dtype


class LinearQnet:
    def __init__(self, input_size, hidden_size, output_size):
        self.weight1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weight2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias2 = np.zeros((1, output_size))

    def forward(self, input):
        """
        Forward pass through neural network to compute output
        """
        self.z1 = np.dot(input, self.weight1) + self.bias1
        self.z2 = np.dot(self.rectified_linear_unit(self.z1), self.weight2) + self.bias2

        return self.z2

    def rectified_linear_unit(self, input):
        """
        This is like the activation function
        """
        return np.maximum(0, input)

    def rectified_linear_unit_deriv(self, input):
        return (input > 0).astype(float)

    def save_model(self, file_name = 'model.npy'):
        """
        Used to save model's weights and biases to a file which is useful to save the trained model to reload later without needing to retrain
        """
        model_directory = './model'

        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        path = os.path.join(model_directory, file_name)
        np.save(path, [self.weight1, self.bias1, self.weight2, self.bias2])

    def load_model(self, file_name = 'model.npy'):
        """
        Allows us to load model's weights and biases saved from file
        """
        path = os.path.join('./model', file_name)
        file_data = np.load(path, allow_pickle=True)

        self.weight1 = file_data[0]
        self.bias1 = file_data[1]
        self.weight2 = file_data[2]
        self.bias2 = file_data[3]

class Trainer:
    def __init__(self, model, learning_rate, gamma):
        self.model = model
        self.learning_rate = learning_rate
        self.gamma = gamma

    def train_model(self, state, action, reward, next, finish):
        # Convert inputs to arrays for network training usage
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        reward = np.array(reward, dtype=np.float32)
        next = np.array(next, dtype=np.float32)

        # Forward pass for cur state to get predicted q values for cur state
        prediction = self.model.predict(state)
        # Clone predictions to adjust q values
        target = np.copy(prediction)

        # Run through states in a batch where finish tracks if an episode has ended. If finish[i] == True, agent is done interacting with that environment for that episode
        for i in range(len(finish)):
            q = reward[i]
            # If episode not done, calculate updated q value using Bellman eq
            if not finish[i]:
                next_prediction = self.model.predict(next[i])
                # Update q to include future reward
                q_updated = np.max(next_prediction)*(self.gamma + reward[i])

            # Update preidcted q for action taken--the model is being trained to predict q values & need to update prediction for the action taken based on updated (more accurate) q value
            target[i, np.argmax(action[i])] = q_updated

        # Backpropagation: use adjusted target & predict q values to train model
        self.backpropgate(state, target, prediction)

    def backpropgate(self, state, target, prediction):
        # MSE
        loss = np.mean((target - prediction) ** 2)

        # Compute gradients (chain rule use)
        derivative_prediction = ((prediction - target) / prediction.shape[0]) * 2

        # 2nd layer gradients
        derivative_weight2 = np.dot(self.model.weight2.T, derivative_prediction)
        derivative_bias2 = np.sum(derivative_prediction, axis=0, keepdims=True)

        # Hidden layer gradients
        derivative_a1 = np.dot(derivative_prediction, self.model.weight1.T)
        derivative_z1 = derivative_a1 * self.model.rectified_linear_unit_deriv(self.model.z1)
        derivative_weight1 = np.dot(state.T, derivative_z1)
        derivative_bias1 = np.sum(derivative_z1, axis=0, keepdims=True)

        # Gradient descent update
        self.model.weight1 -= self.learning_rate * derivative_weight1
        self.model.bias1 -= self.learning_rate * derivative_bias1
        self.model.weight2 -= self.learning_rate * derivative_weight2
        self.model.bias2 -= self.learning_rate * derivative_bias2