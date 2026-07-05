"""Tabular Double Q-Learning (van Hasselt, 2010)."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.agents.tabular_agent import TabularAgent

FloatArray = npt.NDArray[np.float64]


class DoubleQLearningAgent(TabularAgent):
    """Double Q-Learning: two tables to decorrelate selection and evaluation.

    On each transition a fair coin picks the table to update; the argmax is
    taken in the updated table but *evaluated* in the other one, correcting
    Q-Learning's max-operator overestimation bias:

    ``Q_A(s,a) += α [r + γ Q_B(s', argmax_a' Q_A(s',a')) − Q_A(s,a)]``

    Action selection uses the sum ``Q_A + Q_B``.
    """

    name = "double_q_learning"

    def _init_tables(self) -> None:
        self._qa: FloatArray = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self._qb: FloatArray = np.zeros((self.n_states, self.n_actions), dtype=np.float64)

    @property
    def q_table(self) -> FloatArray:
        return self._qa + self._qb

    def q_values(self, state: int) -> FloatArray:
        row: FloatArray = self._qa[state] + self._qb[state]
        return row

    def learn(
        self, state: int, action: int, reward: float, next_state: int, terminated: bool
    ) -> None:
        if self.rng.random() < 0.5:
            primary, other = self._qa, self._qb
        else:
            primary, other = self._qb, self._qa
        target = reward
        if not terminated:
            best_next = int(np.argmax(primary[next_state]))
            target += self.config.gamma * other[next_state, best_next]
        primary[state, action] += self.config.alpha * (target - primary[state, action])

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
        np.savez_compressed(
            path, q_table_a=self._qa, q_table_b=self._qb, metadata=json.dumps(metadata)
        )

    def load(self, path: str | Path) -> None:
        data = np.load(Path(path), allow_pickle=False)
        qa, qb = data["q_table_a"], data["q_table_b"]
        expected = (self.n_states, self.n_actions)
        if qa.shape != expected or qb.shape != expected:
            raise ValueError(
                f"Q-table shapes {qa.shape}/{qb.shape} do not match environment {expected}"
            )
        self._qa = qa.astype(np.float64)
        self._qb = qb.astype(np.float64)
        metadata = json.loads(str(data["metadata"]))
        self.n_episodes_trained = int(metadata.get("n_episodes_trained", 0))

    def memory_bytes(self) -> int:
        return int(self._qa.nbytes + self._qb.nbytes)
