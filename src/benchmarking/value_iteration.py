"""Optimal-policy reference for Taxi-v3 via value iteration.

Value iteration uses the exact transition model ``env.unwrapped.P`` exposed
by Gymnasium's toy-text environments. It serves ONLY as a measuring stick
(R*, the reward of an optimal policy on the evaluation seed set, defining the
0.9·R* convergence threshold) — the trained agents themselves remain strictly
model-free as required by the subject.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt

from src.environments.taxi_wrapper import TaxiEnvWrapper

FloatArray = npt.NDArray[np.float64]


def optimal_q_table(
    env: TaxiEnvWrapper, gamma: float = 0.99, tolerance: float = 1e-10
) -> FloatArray:
    """Compute Q* for the wrapped Taxi-v3 env by value iteration.

    Args:
        env: Wrapped Taxi-v3 environment (must expose ``unwrapped.P``).
        gamma: Discount factor of the reference policy.
        tolerance: Sup-norm convergence threshold.

    Returns:
        The optimal Q-table of shape (n_states, n_actions).
    """
    transitions = cast(dict[int, dict[int, list[Any]]], env.env.unwrapped.P)  # type: ignore[attr-defined]
    n_states, n_actions = env.n_states, env.n_actions
    values: FloatArray = np.zeros(n_states, dtype=np.float64)
    while True:
        q = _q_from_values(transitions, values, n_states, n_actions, gamma)
        new_values = q.max(axis=1)
        if float(np.abs(new_values - values).max()) < tolerance:
            return q
        values = new_values


def _q_from_values(
    transitions: dict[int, dict[int, list[Any]]],
    values: FloatArray,
    n_states: int,
    n_actions: int,
    gamma: float,
) -> FloatArray:
    q: FloatArray = np.zeros((n_states, n_actions), dtype=np.float64)
    for state, actions in transitions.items():
        for action, outcomes in actions.items():
            total = 0.0
            for probability, next_state, reward, terminated in outcomes:
                bootstrap = 0.0 if terminated else gamma * values[next_state]
                total += probability * (reward + bootstrap)
            q[state, action] = total
    return q


def optimal_reference_reward(env: TaxiEnvWrapper, seeds: list[int], gamma: float = 0.99) -> float:
    """R*: mean native reward of the optimal policy over the given episodes.

    Args:
        env: Wrapped Taxi-v3 environment.
        seeds: Evaluation seed list (one seed per episode) — use the same
            fixed set as agent evaluations for a comparable reference.
        gamma: Discount used for the value-iteration policy.

    Returns:
        Mean cumulative native reward of the greedy optimal policy.
    """
    q_star = optimal_q_table(env, gamma)
    rewards: list[float] = []
    for seed in seeds:
        state, _ = env.reset(seed=seed)
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = int(np.argmax(q_star[state]))
            state, _, terminated, truncated, info = env.step(action)
            total += float(info["raw_reward"])
        rewards.append(total)
    return float(np.mean(rewards))
