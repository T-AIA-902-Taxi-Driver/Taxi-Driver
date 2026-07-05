"""Environment wrappers: Taxi-v3, multi-passenger extension, TrackMania."""

from __future__ import annotations

from typing import Any, Protocol

from src.config import Config
from src.environments.multi_passenger_env import (
    MULTI_MAX_STEPS,
    MultiEnvWrapper,
    MultiPassengerTaxiEnv,
)
from src.environments.taxi_wrapper import TAXI_LOCATIONS, RewardFn, TaxiEnvWrapper

__all__ = [
    "MULTI_MAX_STEPS",
    "TAXI_LOCATIONS",
    "EnvWrapper",
    "MultiEnvWrapper",
    "MultiPassengerTaxiEnv",
    "RewardFn",
    "TaxiEnvWrapper",
    "create_env",
]


class EnvWrapper(Protocol):
    """Common surface of the project env wrappers (structural typing).

    Satisfied by :class:`TaxiEnvWrapper` and :class:`MultiEnvWrapper`; this is
    what the Trainer, Evaluator and agent factory consume: int states/actions,
    first-reset constructor seeding with explicit ``reset(seed=...)``
    passthrough, and ``info["raw_reward"]`` set on every step.
    """

    @property
    def n_states(self) -> int:
        """Number of discrete states."""
        ...

    @property
    def n_actions(self) -> int:
        """Number of discrete actions."""
        ...

    def reset(self, seed: int | None = None) -> tuple[int, dict[str, Any]]:
        """Reset the environment, honouring the project seeding convention."""
        ...

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Take one step; ``info["raw_reward"]`` holds the native reward."""
        ...

    def render(self) -> str:
        """Return the current ANSI rendering."""
        ...

    def close(self) -> None:
        """Release the underlying environment resources."""
        ...


def create_env(config: Config, reward_fn: RewardFn | None = None) -> EnvWrapper:
    """Build the environment described by the configuration.

    Args:
        config: Project configuration (``env``, ``seed``,
            ``max_steps_per_episode``).
        reward_fn: Optional reward-shaping function passed through to the
            wrapper (wired by the benchmarking module).

    Returns:
        A ready-to-use environment wrapper. For ``env == "multi"`` the step
        cap is the fixed ``MULTI_MAX_STEPS`` (500) TimeLimit rather than
        ``config.max_steps_per_episode``: two-passenger optimal routes are
        roughly twice Taxi-v3's, plus early-training wandering.
    """
    if config.env == "taxi":
        return TaxiEnvWrapper(
            reward_fn=reward_fn,
            seed=config.seed,
            max_steps=config.max_steps_per_episode,
        )
    return MultiEnvWrapper(reward_fn=reward_fn, seed=config.seed, max_steps=MULTI_MAX_STEPS)
