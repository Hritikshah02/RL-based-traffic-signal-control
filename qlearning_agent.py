"""
qlearning_agent.py - Q-Learning Agent for the Traffic Light Controller.

Implements a tabular Q-learning agent with epsilon-greedy exploration.
The Q-table maps (state_tuple) -> numpy array of Q-values for each action.

Usage:
    from qlearning_agent import QLearningAgent, train_agent
    agent = QLearningAgent()           # fresh agent
    agent = QLearningAgent.load()      # load pre-trained Q-table
    train_agent(episodes=500)          # train and save to q_table.pkl
"""

import os
import pickle
import random
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np

from traffic_env import TrafficEnv
from utils import NUM_ACTIONS

# ── File path for persisting the trained Q-table ─────────────────────────────
Q_TABLE_PATH = os.path.join(os.path.dirname(__file__) or ".", "q_table.pkl")

# ── Hyperparameters ──────────────────────────────────────────────────────────
DEFAULT_ALPHA   = 0.10    # learning rate
DEFAULT_GAMMA   = 0.95    # discount factor
DEFAULT_EPSILON = 1.00    # initial exploration rate
MIN_EPSILON     = 0.01    # floor for epsilon
EPSILON_DECAY   = 0.995   # multiplicative decay per episode


class QLearningAgent:
    """
    Tabular Q-learning agent.

    The Q-table is a defaultdict so unseen states auto-initialize to zeros.
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        gamma: float = DEFAULT_GAMMA,
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Q-table: state (tuple) -> np.array of shape (NUM_ACTIONS,)
        self.q_table: defaultdict = defaultdict(lambda: np.zeros(NUM_ACTIONS))

    # ── Action selection ─────────────────────────────────────────────────────
    def choose_action(self, state: Tuple, explore: bool = True) -> int:
        """
        Epsilon-greedy action selection.

        Parameters
        ----------
        state   : tuple  – the discretized environment state
        explore : bool   – if False, always exploit (for evaluation)

        Returns
        -------
        int – action index (0-3)
        """
        if explore and random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)
        # Exploit: pick action with highest Q-value (break ties randomly)
        q_values = self.q_table[state]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return int(random.choice(best_actions))

    # ── Q-table update ───────────────────────────────────────────────────────
    def update(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple,
    ) -> None:
        """Standard Q-learning update rule."""
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error  = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    # ── Epsilon decay ────────────────────────────────────────────────────────
    def decay_epsilon(self) -> None:
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

    # ── Save / Load ──────────────────────────────────────────────────────────
    def save(self, path: str = Q_TABLE_PATH) -> None:
        """Save Q-table to a pickle file (convert defaultdict -> dict)."""
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    @classmethod
    def load(cls, path: str = Q_TABLE_PATH) -> "QLearningAgent":
        """Load a pre-trained Q-table from disk."""
        agent = cls(epsilon=MIN_EPSILON)  # no exploration after loading
        if os.path.exists(path):
            with open(path, "rb") as f:
                raw = pickle.load(f)
            agent.q_table = defaultdict(lambda: np.zeros(NUM_ACTIONS), raw)
        return agent


# ── Training loop ────────────────────────────────────────────────────────────
def train_agent(
    episodes: int = 500,
    steps_per_episode: int = 200,
    verbose: bool = True,
) -> QLearningAgent:
    """
    Train the Q-learning agent across mixed traffic intensities.

    Each episode uses a randomly chosen intensity so the agent generalizes
    across Low / Medium / Heavy traffic.
    """
    agent = QLearningAgent()
    env = TrafficEnv()
    intensities = ["Low", "Medium", "Heavy"]

    for ep in range(1, episodes + 1):
        state = env.reset()
        intensity = random.choice(intensities)
        total_reward = 0.0

        for _ in range(steps_per_episode):
            action = agent.choose_action(state, explore=True)
            next_state, reward, _info = env.step(action, intensity)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        if verbose and ep % 50 == 0:
            print(
                f"Episode {ep:4d}/{episodes}  "
                f"ε={agent.epsilon:.3f}  "
                f"reward={total_reward:+8.1f}  "
                f"intensity={intensity}"
            )

    agent.save()
    if verbose:
        print(f"\nQ-table saved to {Q_TABLE_PATH}")
        print(f"States visited: {len(agent.q_table)}")
    return agent


# Allow running directly: python qlearning_agent.py
if __name__ == "__main__":
    train_agent()
