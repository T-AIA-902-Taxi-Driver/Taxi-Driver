"""Greedy evaluation of trained agents on fixed episode sets."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from src.evaluation.metrics import EvalResults, compute_eval_results

if TYPE_CHECKING:
    from src.agents.base_agent import BaseAgent
    from src.environments.taxi_wrapper import TaxiEnvWrapper

ACTION_NAMES = ("South", "North", "East", "West", "Pickup", "Dropoff")


class Evaluator:
    """Runs greedy test episodes with per-episode explicit seeding.

    Every episode ``i`` resets the environment with ``seeds[i]``, so all
    agents are evaluated on exactly the same start states — the precondition
    for fair pairwise comparisons and the statistical protocol.
    """

    def __init__(self, env: TaxiEnvWrapper, seeds: list[int] | None = None) -> None:
        self.env = env
        self.seeds = seeds

    def _episode_seed(self, index: int) -> int | None:
        if self.seeds is None:
            return None
        return self.seeds[index % len(self.seeds)]

    def evaluate(self, agent: BaseAgent, n_episodes: int) -> EvalResults:
        """Evaluate ``agent`` greedily (no learning, no exploration).

        Args:
            agent: Trained agent; ``select_action(..., greedy=True)`` is used.
            n_episodes: Number of test episodes.

        Returns:
            Aggregated EvalResults over the ``n_episodes`` episodes.
        """
        rewards: list[float] = []
        steps: list[int] = []
        successes: list[bool] = []
        illegal: list[int] = []
        durations: list[float] = []
        for i in range(n_episodes):
            t0 = time.perf_counter()
            state, _ = self.env.reset(seed=self._episode_seed(i))
            total_raw = 0.0
            n_steps = 0
            n_illegal = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action = agent.select_action(state, greedy=True)
                state, _, terminated, truncated, info = self.env.step(action)
                raw = float(info["raw_reward"])
                total_raw += raw
                n_illegal += raw == -10.0
                n_steps += 1
            durations.append(time.perf_counter() - t0)
            rewards.append(total_raw)
            steps.append(n_steps)
            successes.append(terminated)
            illegal.append(n_illegal)
        return compute_eval_results(rewards, steps, successes, illegal, durations)

    def display_episodes(
        self,
        agent: BaseAgent,
        k: int = 3,
        rng: np.random.Generator | None = None,
        delay: float = 0.0,
        log_fn: Callable[[str], None] = print,
        max_steps: int = 60,
    ) -> None:
        """Render ``k`` random episodes step by step (subject requirement).

        Args:
            agent: Trained agent, played greedily.
            k: Number of episodes to display.
            rng: Source of the random episode seeds (default: fresh rng).
            delay: Optional pause between steps, in seconds.
            log_fn: Sink for the rendered text (injectable for tests).
            max_steps: Display safety cap per episode.
        """
        rng = rng or np.random.default_rng()
        for episode in range(k):
            seed = int(rng.integers(1_000_000))
            state, _ = self.env.reset(seed=seed)
            log_fn(f"\n=== Episode {episode + 1}/{k} (seed {seed}) ===")
            log_fn(self.env.render())
            total = 0.0
            for step in range(1, max_steps + 1):
                action = agent.select_action(state, greedy=True)
                state, _, terminated, truncated, info = self.env.step(action)
                raw = float(info["raw_reward"])
                total += raw
                log_fn(
                    f"step {step}: {ACTION_NAMES[action]} -> reward {raw:+.0f} "
                    f"(cumulative {total:+.0f})"
                )
                log_fn(self.env.render())
                if delay > 0:
                    time.sleep(delay)
                if terminated:
                    log_fn(f"passenger delivered in {step} steps, return {total:+.0f}")
                    break
                if truncated:
                    log_fn(f"episode truncated after {step} steps, return {total:+.0f}")
                    break
            else:
                log_fn(f"display cap reached ({max_steps} steps), return {total:+.0f}")
