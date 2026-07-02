"""Environment wrappers: Taxi-v3, multi-passenger extension, TrackMania."""

from __future__ import annotations

from src.config import Config
from src.environments.taxi_wrapper import TAXI_LOCATIONS, RewardFn, TaxiEnvWrapper

__all__ = ["TAXI_LOCATIONS", "RewardFn", "TaxiEnvWrapper", "create_env"]


def create_env(config: Config, reward_fn: RewardFn | None = None) -> TaxiEnvWrapper:
    """Build the environment described by the configuration.

    Args:
        config: Project configuration (``env``, ``seed``,
            ``max_steps_per_episode``).
        reward_fn: Optional reward-shaping function passed through to the
            wrapper (wired by the benchmarking module).

    Returns:
        A ready-to-use environment wrapper.

    Raises:
        NotImplementedError: If ``config.env`` is ``"multi"`` (upcoming).
    """
    if config.env == "taxi":
        return TaxiEnvWrapper(
            reward_fn=reward_fn,
            seed=config.seed,
            max_steps=config.max_steps_per_episode,
        )
    raise NotImplementedError("coming in feature/multi-passenger")
