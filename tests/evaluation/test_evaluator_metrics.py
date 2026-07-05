"""Evaluator and metrics tests: fixed-seed determinism, aggregates, display."""

import numpy as np
import pytest

from src.agents import create_agent
from src.agents.exploration import UCB
from src.config import Config
from src.environments import create_env
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import compute_eval_results, mean_ci95
from src.training.trainer import Trainer
from src.utils.seeding import eval_seeds


class TestMetrics:
    def test_hand_computed_aggregates(self) -> None:
        results = compute_eval_results(
            rewards=[10.0, 20.0, 30.0, 40.0],
            steps=[5, 10, 15, 20],
            successes=[True, True, False, True],
            illegal_actions=[0, 1, 2, 1],
            durations=[0.1, 0.1, 0.1, 0.1],
        )
        assert results.mean_reward == 25.0
        assert results.median_reward == 25.0
        assert results.mean_steps == 12.5
        assert results.success_rate == 0.75
        assert results.mean_illegal_actions == 1.0
        assert results.mean_episode_seconds == pytest.approx(0.1)
        assert results.reward_percentiles["p50"] == 25.0
        low, high = results.reward_ci95
        assert low < 25.0 < high

    def test_ci95_widens_with_variance(self) -> None:
        tight = mean_ci95([10.0, 10.1, 9.9, 10.0])
        wide = mean_ci95([5.0, 15.0, 0.0, 20.0])
        assert (wide[1] - wide[0]) > (tight[1] - tight[0])

    def test_ci95_degenerate_cases(self) -> None:
        assert mean_ci95([]) == (0.0, 0.0)
        assert mean_ci95([7.0]) == (7.0, 7.0)

    def test_empty_and_mismatched_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_eval_results([], [], [], [], [])
        with pytest.raises(ValueError, match="mismatched"):
            compute_eval_results([1.0], [1, 2], [True], [0], [0.1])

    def test_summary_contains_required_outputs(self) -> None:
        results = compute_eval_results([8.0], [13], [True], [0], [0.001])
        text = results.summary()
        assert "mean reward" in text
        assert "mean time per game" in text  # subject-required output


class TestEvaluator:
    def test_fixed_seeds_give_identical_results(self) -> None:
        config = Config(algorithm="q_learning", seed=5)
        env = create_env(config)
        agent = create_agent(config, env)
        Trainer(agent, env, config).train(300)
        evaluator = Evaluator(env, seeds=eval_seeds(config.seed, 20))
        first = evaluator.evaluate(agent, 20)
        second = evaluator.evaluate(agent, 20)
        assert first.rewards == second.rewards
        assert first.steps == second.steps

    def test_greedy_evaluation_does_not_touch_ucb_counts(self) -> None:
        config = Config(algorithm="q_learning", exploration="ucb", seed=2)
        env = create_env(config)
        agent = create_agent(config, env)
        strategy = agent.strategy  # type: ignore[attr-defined]
        assert isinstance(strategy, UCB)
        counts_before = strategy.counts.sum()
        Evaluator(env, seeds=eval_seeds(2, 5)).evaluate(agent, 5)
        assert strategy.counts.sum() == counts_before

    def test_display_episodes_renders_actions_and_grid(self) -> None:
        config = Config(algorithm="brute_force", seed=9)
        env = create_env(config)
        agent = create_agent(config, env)
        lines: list[str] = []
        Evaluator(env).display_episodes(
            agent, k=2, rng=np.random.default_rng(0), log_fn=lines.append, max_steps=5
        )
        text = "\n".join(lines)
        assert text.count("=== Episode") == 2
        assert "+---------+" in text  # taxi grid rendering
        assert any(name in text for name in ("South", "North", "East", "West"))
        assert "cumulative" in text
