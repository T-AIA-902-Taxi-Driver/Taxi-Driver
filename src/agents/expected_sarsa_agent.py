"""Tabular Expected SARSA (van Seijen et al., 2009)."""

from __future__ import annotations

import numpy as np

from src.agents.tabular_agent import TabularAgent


class ExpectedSARSAAgent(TabularAgent):
    """Expected SARSA: bootstraps from the expectation over the policy.

    ``Q(s,a) += α [r + γ Σ_a' π(a'|s') Q(s',a') − Q(s,a)]`` — replaces
    SARSA's single sampled a' with its expectation, removing that source of
    update variance while staying on-policy.
    """

    name = "expected_sarsa"

    def learn(
        self, state: int, action: int, reward: float, next_state: int, terminated: bool
    ) -> None:
        target = reward
        if not terminated:
            q_row = self.q_values(next_state)
            probs = self.strategy.action_probs(q_row, next_state)
            target += self.config.gamma * float(np.dot(probs, q_row))
        self._q[state, action] += self.config.alpha * (target - self._q[state, action])
