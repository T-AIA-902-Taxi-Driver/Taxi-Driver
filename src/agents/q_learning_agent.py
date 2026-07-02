"""Tabular Q-Learning (Watkins, 1989) — off-policy TD control."""

from __future__ import annotations

from src.agents.tabular_agent import TabularAgent


class QLearningAgent(TabularAgent):
    """Q-Learning: ``Q(s,a) += α [r + γ max_a' Q(s',a') − Q(s,a)]``.

    Off-policy: the update bootstraps from the best next action regardless of
    the action actually taken by the exploration policy.
    """

    name = "q_learning"

    def learn(
        self, state: int, action: int, reward: float, next_state: int, terminated: bool
    ) -> None:
        target = reward
        if not terminated:
            target += self.config.gamma * float(self._q[next_state].max())
        self._q[state, action] += self.config.alpha * (target - self._q[state, action])
