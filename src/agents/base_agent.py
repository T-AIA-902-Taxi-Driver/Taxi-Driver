"""Abstract base class defining the contract shared by every agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from src.config import Config


class BaseAgent(ABC):
    """Common interface for all RL agents (Strategy pattern).

    The training loop contract:

    - ``select_action(state)`` is called on the training path;
      ``select_action(state, greedy=True)`` on the evaluation path.
    - ``learn(state, action, reward, next_state, terminated)`` receives
      ``terminated`` only — never ``terminated or truncated``. An episode cut
      by a time limit must still bootstrap from the successor state.
    - ``end_episode()`` is called by the Trainer after every episode
      (terminated or truncated) so agents can advance schedules and flush
      episode buffers.
    """

    name: str = "base"

    def __init__(self, n_states: int, n_actions: int, config: Config, rng: np.random.Generator):
        self.n_states = n_states
        self.n_actions = n_actions
        self.config = config
        self.rng = rng
        self.n_episodes_trained = 0

    @abstractmethod
    def select_action(self, state: int, greedy: bool = False) -> int:
        """Choose an action for ``state``.

        Args:
            state: Encoded environment state.
            greedy: When True, act greedily w.r.t. learned values (evaluation
                path); no exploration statistics may be updated.

        Returns:
            The chosen action index.
        """

    @abstractmethod
    def learn(
        self, state: int, action: int, reward: float, next_state: int, terminated: bool
    ) -> None:
        """Update knowledge from one transition.

        Args:
            state: State before the action.
            action: Action taken.
            reward: Reward received (possibly shaped).
            next_state: Successor state.
            terminated: True only when the MDP reached a terminal state —
                truncation (time limit) must be passed as False so the update
                still bootstraps.
        """

    def end_episode(self) -> None:
        """Hook called after every episode; advances schedules/counters."""
        self.n_episodes_trained += 1

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the agent's learned parameters and metadata to ``path``."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Restore the agent's learned parameters from ``path``."""

    def memory_bytes(self) -> int:
        """Approximate memory footprint of learned parameters, in bytes."""
        return 0

    @property
    def exploration_level(self) -> float:
        """Scalar exploration level for logging (epsilon, temperature, ...)."""
        return 0.0
