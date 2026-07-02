"""Reward-shaping functions for the Taxi environment.

Three variants are compared in the H5 experiment (block E4):

- ``potential``: potential-based shaping F = γ·Φ(s') − Φ(s) with
  Φ(s) = −(Manhattan distance to the current objective). Ng, Harada &
  Russell (1999) prove this family preserves the optimal policy.
- ``naive_distance``: a plain distance-improvement bonus, NOT potential-based
  — expected to distort the optimal policy (negative control).
- ``step_penalty``: doubles the per-step cost (reward − 1 on moves).

All shapers receive and return floats via the wrapper hook signature
``(state, action, raw_reward, next_state) -> float``; the native reward is
always preserved in ``info["raw_reward"]`` by the wrapper.
"""

from __future__ import annotations

from collections.abc import Callable

from src.environments.taxi_wrapper import TAXI_LOCATIONS

RewardFn = Callable[[int, int, float, int], float]

_IN_TAXI = 4


def _decode(state: int) -> tuple[int, int, int, int]:
    """Decode a Taxi-v3 state integer without an env instance."""
    destination = state % 4
    state //= 4
    passenger = state % 5
    state //= 5
    col = state % 5
    row = state // 5
    return row, col, passenger, destination


def _objective_distance(state: int) -> int:
    """Manhattan distance from the taxi to its current objective.

    Objective = passenger location while not picked up, destination after.
    """
    row, col, passenger, destination = _decode(state)
    target = TAXI_LOCATIONS[destination] if passenger == _IN_TAXI else TAXI_LOCATIONS[passenger]
    return abs(row - target[0]) + abs(col - target[1])


def potential(state: int) -> float:
    """Φ(s) = −distance to the current objective (higher is better)."""
    return -float(_objective_distance(state))


def make_potential_shaper(gamma: float) -> RewardFn:
    """Potential-based shaping (optimal-policy preserving, Ng et al. 1999)."""

    def shaper(state: int, action: int, raw_reward: float, next_state: int) -> float:
        return raw_reward + gamma * potential(next_state) - potential(state)

    return shaper


def make_naive_distance_shaper(bonus: float = 0.5) -> RewardFn:
    """Non-potential distance bonus (negative control for H5).

    Adds ``+bonus`` whenever the taxi gets closer to its objective. Because
    the bonus is not a potential difference, cycles can accumulate reward and
    the optimal policy is not guaranteed to be preserved.
    """

    def shaper(state: int, action: int, raw_reward: float, next_state: int) -> float:
        if _objective_distance(next_state) < _objective_distance(state):
            return raw_reward + bonus
        return raw_reward

    return shaper


def make_step_penalty_shaper(extra_penalty: float = 1.0) -> RewardFn:
    """Extra per-step penalty on movement steps (rewards −1 become −2)."""

    def shaper(state: int, action: int, raw_reward: float, next_state: int) -> float:
        if raw_reward == -1.0:
            return raw_reward - extra_penalty
        return raw_reward

    return shaper


def create_reward_shaper(name: str, gamma: float) -> RewardFn | None:
    """Build the shaper named by ``Config.reward_shaping`` (None = native)."""
    if name == "none":
        return None
    if name == "potential":
        return make_potential_shaper(gamma)
    if name == "naive_distance":
        return make_naive_distance_shaper()
    if name == "step_penalty":
        return make_step_penalty_shaper()
    raise ValueError(f"unknown reward shaping: {name!r}")
