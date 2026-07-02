"""Shared plumbing for tabular agents: Q-table storage and exploration."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.agents.base_agent import BaseAgent
from src.agents.exploration import create_exploration
from src.config import Config

FloatArray = npt.NDArray[np.float64]


class TabularAgent(BaseAgent):
    """Base class for agents backed by an in-memory Q-table.

    Subclasses implement ``learn()`` only. Greedy action selection uses a
    deterministic argmax (smallest index on ties) so that evaluation results
    are strictly comparable across agents; the training path delegates to the
    configured exploration strategy (random tie-breaking).
    """

    name = "tabular"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        config: Config,
        rng: np.random.Generator,
        n_episodes: int | None = None,
    ) -> None:
        super().__init__(n_states, n_actions, config, rng)
        self._init_tables()
        self.strategy = create_exploration(config, n_states, n_actions, rng, n_episodes)

    # ------------------------------------------------------------------ storage

    def _init_tables(self) -> None:
        self._q: FloatArray = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

    @property
    def q_table(self) -> FloatArray:
        """Full Q-table view (used by save/heatmaps); may be derived."""
        return self._q

    def q_values(self, state: int) -> FloatArray:
        """Row of Q-values used for action selection in ``state``."""
        row: FloatArray = self._q[state]
        return row

    # ------------------------------------------------------------------ actions

    def select_action(self, state: int, greedy: bool = False) -> int:
        if greedy:
            return int(np.argmax(self.q_values(state)))
        return self.strategy.select(self.q_values(state), state)

    def end_episode(self) -> None:
        super().end_episode()
        self.strategy.decay()

    # ------------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "algorithm": self.name,
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "n_episodes_trained": self.n_episodes_trained,
            "config_hash": self.config.config_hash(),
            "config": self.config.to_dict(),
            "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        np.savez_compressed(path, q_table=self.q_table, metadata=json.dumps(metadata))

    def load(self, path: str | Path) -> None:
        data = np.load(Path(path), allow_pickle=False)
        q_table = data["q_table"]
        if q_table.shape != (self.n_states, self.n_actions):
            raise ValueError(
                f"Q-table shape {q_table.shape} does not match environment "
                f"({self.n_states}, {self.n_actions})"
            )
        self._q = q_table.astype(np.float64)
        metadata = json.loads(str(data["metadata"]))
        self.n_episodes_trained = int(metadata.get("n_episodes_trained", 0))

    def memory_bytes(self) -> int:
        return int(self._q.nbytes)

    @property
    def exploration_level(self) -> float:
        return self.strategy.epsilon_like
