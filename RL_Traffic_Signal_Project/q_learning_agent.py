"""Tabular epsilon-greedy Q-learning agent."""
import numpy as np

class QLearningAgent:
    def __init__(self, alpha=0.12, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, seed=123):
        self.q = np.zeros((6, 6, 2, 2), dtype=float)
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, epsilon_min, epsilon_decay
        self.rng = np.random.default_rng(seed)

    def act(self, state, training=True):
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(2))
        values = self.q[state]
        # Deterministic tie-breaking: keep current signal.
        return int(np.argmax(values))

    def update(self, state, action, reward, next_state, done):
        target = reward if done else reward + self.gamma * float(np.max(self.q[next_state]))
        self.q[state + (action,)] += self.alpha * (target - self.q[state + (action,)])

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
