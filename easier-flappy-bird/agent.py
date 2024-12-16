import numpy as np

class QLearningAgent:
    def __init__(self, 
                 initial_learning_rate=0.6,
                 min_learning_rate=0.1,
                 learning_rate_decay=0.995,
                 discount_factor=0.4):
        # Q-table dimensions: [x_positions][y_positions][actions]
        self.Q = np.zeros((7, 21, 2), dtype=float)
        
        # Learning parameters
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.discount_factor = discount_factor
        
        # Training tracking
        self.episodes = 0

    def get_action(self, state):
        """Choose action based on Q-values"""
        x, y = state
        return self.Q[x][y][1] > self.Q[x][y][0]

    def update(self, state, action, reward, next_state):
        """Update Q-values using Q-learning update rule"""
        x, y = state
        next_x, next_y = next_state
        
        # Get maximum Q-value for next state
        next_max_q = max(self.Q[next_x][next_y][0], self.Q[next_x][next_y][1])
        
        # Update Q-value for current state-action pair
        action_idx = 1 if action else 0
        current_q = self.Q[x][y][action_idx]
        
        # Q-learning update formula
        self.Q[x][y][action_idx] = (1 - self.learning_rate) * current_q + \
                                  self.learning_rate * (reward + self.discount_factor * next_max_q)
        
        # Decay learning rate
        self.episodes += 1
        self.learning_rate = max(
            self.min_learning_rate,
            self.initial_learning_rate * (self.learning_rate_decay ** self.episodes)
        )
    
    def get_learning_rate(self):
        """Return current learning rate"""
        return self.learning_rate