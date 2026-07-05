"""Observer-pattern callbacks for the training loop."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.base_agent import BaseAgent
    from src.training.trainer import EpisodeResult, TrainingHistory


class StopTraining(Exception):  # noqa: N818 — control-flow signal like StopIteration, not an error
    """Raised by a callback to stop training gracefully."""


class Callback:
    """Base callback: override any hook; defaults are no-ops."""

    def on_training_start(self, n_episodes: int) -> None:
        """Called once before the first episode."""

    def on_episode_end(self, episode: int, result: EpisodeResult) -> None:
        """Called after every episode; may raise StopTraining."""

    def on_training_end(self, history: TrainingHistory) -> None:
        """Called once after the last episode (including early stops)."""


class LoggingCallback(Callback):
    """Prints a rolling summary every ``every`` episodes."""

    def __init__(
        self, every: int = 1000, window: int = 100, log_fn: Callable[[str], None] = print
    ) -> None:
        self.every = every
        self.window = window
        self.log_fn = log_fn
        self._recent_rewards: list[float] = []
        self._recent_steps: list[int] = []

    def on_episode_end(self, episode: int, result: EpisodeResult) -> None:
        self._recent_rewards.append(result.raw_reward)
        self._recent_steps.append(result.steps)
        if len(self._recent_rewards) > self.window:
            self._recent_rewards.pop(0)
            self._recent_steps.pop(0)
        if (episode + 1) % self.every == 0:
            mean_r = sum(self._recent_rewards) / len(self._recent_rewards)
            mean_s = sum(self._recent_steps) / len(self._recent_steps)
            self.log_fn(
                f"episode {episode + 1}: reward(last {len(self._recent_rewards)})="
                f"{mean_r:.2f}, steps={mean_s:.1f}"
            )


class EarlyStoppingCallback(Callback):
    """Stops when the rolling mean native reward reaches a target.

    Opt-in only: keep disabled during benchmark comparisons (stopping some
    algorithms earlier than others biases every convergence metric).
    """

    def __init__(self, target_reward: float = 8.0, window: int = 100) -> None:
        self.target_reward = target_reward
        self.window = window
        self._recent: list[float] = []

    def on_episode_end(self, episode: int, result: EpisodeResult) -> None:
        self._recent.append(result.raw_reward)
        if len(self._recent) > self.window:
            self._recent.pop(0)
        if (
            len(self._recent) == self.window
            and sum(self._recent) / self.window >= self.target_reward
        ):
            raise StopTraining(
                f"early stop at episode {episode + 1}: rolling mean reward "
                f"{sum(self._recent) / self.window:.2f} >= {self.target_reward}"
            )


class CheckpointCallback(Callback):
    """Saves the agent every ``every`` episodes."""

    def __init__(self, agent: BaseAgent, path: str | Path, every: int = 5000) -> None:
        self.agent = agent
        self.path = Path(path)
        self.every = every

    def on_episode_end(self, episode: int, result: EpisodeResult) -> None:
        if (episode + 1) % self.every == 0:
            self.agent.save(self.path)


class TimeBudgetCallback(Callback):
    """Stops training when a wall-clock deadline is reached.

    The ``clock`` is injectable so tests can use a fake clock instead of
    sleeping. Episode-level granularity is sufficient: tabular Taxi episodes
    run sub-millisecond.
    """

    def __init__(self, deadline: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.deadline = deadline
        self.clock = clock

    def on_episode_end(self, episode: int, result: EpisodeResult) -> None:
        if self.clock() >= self.deadline:
            raise StopTraining(f"time budget reached at episode {episode + 1}")
