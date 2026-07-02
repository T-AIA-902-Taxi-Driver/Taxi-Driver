"""Evaluation metrics: aggregate statistics over test episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import stats


def mean_ci95(values: list[float]) -> tuple[float, float]:
    """95% confidence interval of the mean (Student t distribution).

    Args:
        values: Sample values (n >= 2 for a meaningful interval).

    Returns:
        (low, high) bounds; degenerate (mean, mean) when n < 2.
    """
    n = len(values)
    mean = float(np.mean(values)) if values else 0.0
    if n < 2:
        return (mean, mean)
    sem = float(np.std(values, ddof=1)) / float(np.sqrt(n))
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    return (mean - t_crit * sem, mean + t_crit * sem)


@dataclass
class EvalResults:
    """Aggregate metrics of a greedy evaluation over N episodes.

    Rewards are native environment rewards. ``mean_episode_seconds`` is the
    subject's required "mean time for finishing the game".
    """

    n_episodes: int
    mean_reward: float
    std_reward: float
    median_reward: float
    reward_ci95: tuple[float, float]
    reward_percentiles: dict[str, float]  # p25 / p50 / p75 / p95
    mean_steps: float
    std_steps: float
    median_steps: float
    success_rate: float
    mean_illegal_actions: float
    mean_episode_seconds: float
    rewards: list[float] = field(repr=False)
    steps: list[int] = field(repr=False)
    successes: list[bool] = field(repr=False)
    illegal_actions: list[int] = field(repr=False)
    durations: list[float] = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        """Human-readable multi-line summary (console output)."""
        low, high = self.reward_ci95
        return (
            f"episodes            : {self.n_episodes}\n"
            f"mean reward         : {self.mean_reward:.2f} ± {self.std_reward:.2f} "
            f"(95% CI [{low:.2f}, {high:.2f}])\n"
            f"median reward       : {self.median_reward:.2f}\n"
            f"mean steps          : {self.mean_steps:.2f} ± {self.std_steps:.2f} "
            f"(median {self.median_steps:.0f})\n"
            f"success rate        : {self.success_rate * 100:.1f}%\n"
            f"illegal actions/ep  : {self.mean_illegal_actions:.2f}\n"
            f"mean time per game  : {self.mean_episode_seconds * 1000:.2f} ms"
        )


def compute_eval_results(
    rewards: list[float],
    steps: list[int],
    successes: list[bool],
    illegal_actions: list[int],
    durations: list[float],
) -> EvalResults:
    """Aggregate per-episode evaluation data into EvalResults.

    Args:
        rewards: Native cumulative reward per episode.
        steps: Step count per episode.
        successes: Whether each episode terminated (correct dropoff).
        illegal_actions: Count of -10 penalties per episode.
        durations: Wall-clock seconds per episode.

    Returns:
        Aggregated EvalResults.

    Raises:
        ValueError: If the input lists are empty or of mismatched lengths.
    """
    n = len(rewards)
    if n == 0:
        raise ValueError("cannot aggregate an empty evaluation")
    if not (len(steps) == len(successes) == len(illegal_actions) == len(durations) == n):
        raise ValueError("mismatched metric list lengths")
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    steps_arr = np.asarray(steps, dtype=np.float64)
    return EvalResults(
        n_episodes=n,
        mean_reward=float(rewards_arr.mean()),
        std_reward=float(rewards_arr.std(ddof=1)) if n > 1 else 0.0,
        median_reward=float(np.median(rewards_arr)),
        reward_ci95=mean_ci95(rewards),
        reward_percentiles={
            "p25": float(np.percentile(rewards_arr, 25)),
            "p50": float(np.percentile(rewards_arr, 50)),
            "p75": float(np.percentile(rewards_arr, 75)),
            "p95": float(np.percentile(rewards_arr, 95)),
        },
        mean_steps=float(steps_arr.mean()),
        std_steps=float(steps_arr.std(ddof=1)) if n > 1 else 0.0,
        median_steps=float(np.median(steps_arr)),
        success_rate=float(np.mean([1.0 if s else 0.0 for s in successes])),
        mean_illegal_actions=float(np.mean(illegal_actions)),
        mean_episode_seconds=float(np.mean(durations)),
        rewards=list(rewards),
        steps=list(steps),
        successes=list(successes),
        illegal_actions=list(illegal_actions),
        durations=list(durations),
    )
