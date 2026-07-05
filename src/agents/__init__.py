"""RL agents (tabular and deep) implementing the BaseAgent interface."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from src.agents.base_agent import BaseAgent
from src.agents.brute_force_agent import BruteForceAgent
from src.agents.double_q_learning_agent import DoubleQLearningAgent
from src.agents.expected_sarsa_agent import ExpectedSARSAAgent
from src.agents.monte_carlo_agent import MonteCarloAgent
from src.agents.q_learning_agent import QLearningAgent
from src.agents.sarsa_agent import SARSAAgent
from src.utils.seeding import spawn_rngs

if TYPE_CHECKING:
    from src.config import Config


class _EnvLike(Protocol):
    """Anything exposing state/action counts (env wrappers, test stubs)."""

    @property
    def n_states(self) -> int: ...

    @property
    def n_actions(self) -> int: ...


AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "brute_force": BruteForceAgent,
    "q_learning": QLearningAgent,
    "sarsa": SARSAAgent,
    "expected_sarsa": ExpectedSARSAAgent,
    "double_q_learning": DoubleQLearningAgent,
    "monte_carlo": MonteCarloAgent,
    # "dqn" is resolved lazily in create_agent() to keep torch optional at import time.
}


def create_agent(
    config: Config,
    env: _EnvLike,
    rng: np.random.Generator | None = None,
    n_episodes: int | None = None,
) -> BaseAgent:
    """Instantiate the agent named by ``config.algorithm`` (Factory pattern).

    Args:
        config: Project configuration (validated: algorithm name is known).
        env: Environment exposing ``n_states``/``n_actions`` — agents never
            hardcode Taxi-v3's 500 states.
        rng: Agent random stream; derived from ``config.seed`` when omitted.
        n_episodes: Training horizon for decay resolution; defaults to
            ``config.n_train_episodes``.

    Returns:
        A ready-to-train agent.
    """
    if rng is None:
        # Stream 0 is reserved for the environment by convention.
        rng = spawn_rngs(config.seed, 2)[1]
    if config.algorithm == "dqn":
        # Lazy import so the tabular stack never pays the torch import cost.
        module = importlib.import_module("src.agents.dqn.dqn_agent")
        agent = module.DQNAgent(env.n_states, env.n_actions, config, rng, n_episodes)
        return cast(BaseAgent, agent)
    agent_class = AGENT_REGISTRY[config.algorithm]
    return agent_class(env.n_states, env.n_actions, config, rng, n_episodes)  # type: ignore[call-arg]


__all__ = [
    "AGENT_REGISTRY",
    "BaseAgent",
    "BruteForceAgent",
    "DoubleQLearningAgent",
    "ExpectedSARSAAgent",
    "MonteCarloAgent",
    "QLearningAgent",
    "SARSAAgent",
    "create_agent",
]
