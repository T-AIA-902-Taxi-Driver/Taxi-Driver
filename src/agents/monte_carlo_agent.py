"""Tabular first-visit Monte Carlo control."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.agents.tabular_agent import TabularAgent
from src.config import Config


class MonteCarloAgent(TabularAgent):
    """First-visit Monte Carlo control with incremental-mean updates.

    Transitions are buffered during the episode; ``end_episode()`` walks the
    trajectory backwards accumulating the discounted return G and updates
    ``Q(s,a)`` towards G on the *first* visit of each (s, a) pair:
    ``Q(s,a) += (G − Q(s,a)) / N(s,a)``.

    Updates happen on episode end regardless of termination type; truncated
    episodes contribute their (biased) truncated return, a known limitation
    of Monte Carlo methods on time-limited environments.
    """

    name = "monte_carlo"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        config: Config,
        rng: np.random.Generator,
        n_episodes: int | None = None,
    ) -> None:
        super().__init__(n_states, n_actions, config, rng, n_episodes)
        self._counts: npt.NDArray[np.int64] = np.zeros((n_states, n_actions), dtype=np.int64)
        self._episode: list[tuple[int, int, float]] = []

    def learn(
        self, state: int, action: int, reward: float, next_state: int, terminated: bool
    ) -> None:
        self._episode.append((state, action, reward))

    def end_episode(self) -> None:
        if self._episode:
            self._update_from_episode()
            self._episode.clear()
        super().end_episode()

    def _update_from_episode(self) -> None:
        first_visit: dict[tuple[int, int], int] = {}
        for t, (state, action, _) in enumerate(self._episode):
            first_visit.setdefault((state, action), t)

        g = 0.0
        for t in range(len(self._episode) - 1, -1, -1):
            state, action, reward = self._episode[t]
            g = self.config.gamma * g + reward
            if first_visit[(state, action)] == t:
                self._counts[state, action] += 1
                self._q[state, action] += (g - self._q[state, action]) / self._counts[state, action]

    def memory_bytes(self) -> int:
        return int(self._q.nbytes + self._counts.nbytes)
