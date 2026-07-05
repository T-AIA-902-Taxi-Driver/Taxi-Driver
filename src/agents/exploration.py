"""Pluggable exploration strategies shared by all tabular agents (and DQN).

A strategy picks an action from one row of Q-values and owns its own schedule
state (epsilon, temperature) or statistics (UCB visit counts). Greedy
evaluation bypasses strategies entirely (see ``TabularAgent.select_action``),
so evaluation never pollutes UCB counts nor advances decay schedules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from src.config import Config

FloatArray = npt.NDArray[np.float64]


class ExplorationStrategy(ABC):
    """Selects actions from Q-values during training. Stateful."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    @abstractmethod
    def select(self, q_row: FloatArray, state: int) -> int:
        """Pick an action for ``state`` given its row of Q-values."""

    @abstractmethod
    def action_probs(self, q_row: FloatArray, state: int) -> FloatArray:
        """Return π(a|s) under this strategy (used by Expected SARSA)."""

    def decay(self) -> None:  # noqa: B027 — optional hook, deliberate no-op default
        """Advance the schedule by one episode. Default: no schedule."""

    @property
    def epsilon_like(self) -> float:
        """Scalar summarising the current exploration level, for logging."""
        return 0.0

    @staticmethod
    def _random_tie_argmax(q_row: FloatArray, rng: np.random.Generator) -> int:
        """Argmax with uniform random tie-breaking (training path only)."""
        best = np.flatnonzero(q_row == q_row.max())
        return int(best[rng.integers(len(best))])


class EpsilonGreedy(ExplorationStrategy):
    """ε-greedy with exponential or linear per-episode decay."""

    def __init__(
        self,
        rng: np.random.Generator,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        decay_rate: float = 0.9995,
        decay_type: str = "exp",
    ) -> None:
        super().__init__(rng)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        self.decay_type = decay_type

    def select(self, q_row: FloatArray, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(q_row)))
        return self._random_tie_argmax(q_row, self.rng)

    def action_probs(self, q_row: FloatArray, state: int) -> FloatArray:
        n_actions = len(q_row)
        probs = np.full(n_actions, self.epsilon / n_actions, dtype=np.float64)
        best = np.flatnonzero(q_row == q_row.max())
        probs[best] += (1.0 - self.epsilon) / len(best)
        return probs

    def decay(self) -> None:
        if self.decay_type == "exp":
            self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_rate)
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.decay_rate)

    @property
    def epsilon_like(self) -> float:
        return self.epsilon


class Boltzmann(ExplorationStrategy):
    """Softmax (Boltzmann) exploration with temperature decay."""

    def __init__(
        self,
        rng: np.random.Generator,
        temperature: float = 1.0,
        temperature_min: float = 0.05,
        decay_rate: float = 0.999,
    ) -> None:
        super().__init__(rng)
        self.temperature = temperature
        self.temperature_min = temperature_min
        self.decay_rate = decay_rate

    def select(self, q_row: FloatArray, state: int) -> int:
        return int(self.rng.choice(len(q_row), p=self.action_probs(q_row, state)))

    def action_probs(self, q_row: FloatArray, state: int) -> FloatArray:
        # Max-subtraction for numerical stability: softmax is shift-invariant.
        logits = (q_row - q_row.max()) / self.temperature
        exp = np.exp(logits)
        probs: FloatArray = exp / exp.sum()
        return probs

    def decay(self) -> None:
        self.temperature = max(self.temperature_min, self.temperature * self.decay_rate)

    @property
    def epsilon_like(self) -> float:
        return self.temperature


class UCB(ExplorationStrategy):
    """Upper Confidence Bound exploration over per-(state, action) counts.

    Untried actions in a state always take priority; afterwards actions are
    ranked by ``Q(s, a) + c * sqrt(ln t / N(s, a))`` where ``t`` is the total
    number of selections in ``s``. Counts live inside the strategy and are
    only updated on the training path.
    """

    def __init__(
        self, rng: np.random.Generator, n_states: int, n_actions: int, c: float = 2.0
    ) -> None:
        super().__init__(rng)
        self.c = c
        self.counts: npt.NDArray[np.int64] = np.zeros((n_states, n_actions), dtype=np.int64)

    def select(self, q_row: FloatArray, state: int) -> int:
        counts_row = self.counts[state]
        untried = np.flatnonzero(counts_row == 0)
        if len(untried) > 0:
            action = int(untried[self.rng.integers(len(untried))])
        else:
            t = float(counts_row.sum())
            scores = q_row + self.c * np.sqrt(np.log(t) / counts_row)
            action = self._random_tie_argmax(scores, self.rng)
        self.counts[state, action] += 1
        return action

    def action_probs(self, q_row: FloatArray, state: int) -> FloatArray:
        # UCB is deterministic given its counts: one-hot on the current choice
        # (without recording a visit).
        counts_row = self.counts[state]
        untried = np.flatnonzero(counts_row == 0)
        if len(untried) > 0:
            probs = np.zeros(len(q_row), dtype=np.float64)
            probs[untried] = 1.0 / len(untried)
            return probs
        t = float(counts_row.sum())
        scores = q_row + self.c * np.sqrt(np.log(t) / counts_row)
        probs = np.zeros(len(q_row), dtype=np.float64)
        probs[int(np.argmax(scores))] = 1.0
        return probs


EXPLORATION_REGISTRY: dict[str, type[ExplorationStrategy]] = {
    "epsilon_greedy": EpsilonGreedy,
    "boltzmann": Boltzmann,
    "ucb": UCB,
}


def create_exploration(
    config: Config,
    n_states: int,
    n_actions: int,
    rng: np.random.Generator,
    n_episodes: int | None = None,
) -> ExplorationStrategy:
    """Build the exploration strategy described by ``config``.

    Args:
        config: Project configuration.
        n_states: Environment state count (UCB counts shape).
        n_actions: Environment action count.
        rng: Random generator owned by the calling agent.
        n_episodes: Training horizon used to resolve ``config.decay_frac``;
            defaults to ``config.n_train_episodes``.

    Returns:
        A ready-to-use ExplorationStrategy.
    """
    horizon = n_episodes if n_episodes is not None else config.n_train_episodes
    if config.exploration == "epsilon_greedy":
        return EpsilonGreedy(
            rng,
            epsilon=config.epsilon,
            epsilon_min=config.epsilon_min,
            decay_rate=config.effective_decay(horizon),
            decay_type=config.decay_type,
        )
    if config.exploration == "boltzmann":
        return Boltzmann(
            rng,
            temperature=config.temperature,
            temperature_min=config.temperature_min,
            decay_rate=config.temperature_decay,
        )
    return UCB(rng, n_states=n_states, n_actions=n_actions, c=config.ucb_c)
