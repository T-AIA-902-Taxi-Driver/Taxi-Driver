"""Tabular SARSA (Rummery & Niranjan, 1994) — on-policy TD control."""

from __future__ import annotations

import numpy as np

from src.agents.tabular_agent import TabularAgent
from src.config import Config


class SARSAAgent(TabularAgent):
    """SARSA: ``Q(s,a) += α [r + γ Q(s',a') − Q(s,a)]`` with a' actually taken.

    On-policy correctness requires that the a' used in the update is the very
    action executed at the next step: ``learn()`` draws a' from the
    exploration strategy, uses it for the update, and caches it so the next
    ``select_action(s')`` call returns it instead of re-sampling.
    """

    name = "sarsa"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        config: Config,
        rng: np.random.Generator,
        n_episodes: int | None = None,
    ) -> None:
        super().__init__(n_states, n_actions, config, rng, n_episodes)
        self._next_action: int | None = None

    def select_action(self, state: int, greedy: bool = False) -> int:
        if greedy:
            return super().select_action(state, greedy=True)
        if self._next_action is not None:
            action, self._next_action = self._next_action, None
            return action
        return super().select_action(state)

    def learn(
        self, state: int, action: int, reward: float, next_state: int, terminated: bool
    ) -> None:
        target = reward
        if not terminated:
            next_action = self.strategy.select(self.q_values(next_state), next_state)
            self._next_action = next_action
            target += self.config.gamma * self._q[next_state, next_action]
        self._q[state, action] += self.config.alpha * (target - self._q[state, action])

    def end_episode(self) -> None:
        super().end_episode()
        self._next_action = None
