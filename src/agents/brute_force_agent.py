"""Random baseline agent — the comparison point required by the subject."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np

from src.agents.base_agent import BaseAgent
from src.config import Config


class BruteForceAgent(BaseAgent):
    """Uniform-random policy with no learning.

    Serves as the naive brute-force baseline (~350 steps per episode on
    Taxi-v3 vs ~13 for a tuned agent). ``greedy`` has no effect: a random
    policy has no greedy counterpart.
    """

    name = "brute_force"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        config: Config,
        rng: np.random.Generator,
        n_episodes: int | None = None,
    ) -> None:
        super().__init__(n_states, n_actions, config, rng)

    def select_action(self, state: int, greedy: bool = False) -> int:
        return int(self.rng.integers(self.n_actions))

    def learn(
        self, state: int, action: int, reward: float, next_state: int, terminated: bool
    ) -> None:
        """No-op: the baseline does not learn."""

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "algorithm": self.name,
            "seed": self.config.seed,
            "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        np.savez_compressed(path, metadata=json.dumps(metadata))

    def load(self, path: str | Path) -> None:
        """No learned parameters to restore."""
