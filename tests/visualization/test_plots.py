"""Visualization tests: every generator produces a PNG from synthetic data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.visualization.plots import (
    grid_search_heatmap,
    learning_curves,
    probe_curves,
    q_values_heatmap,
    reward_boxplots,
    rolling_mean,
    sample_efficiency_plot,
    sensitivity_curves,
    steps_barplot,
)

RNG = np.random.default_rng(0)


def _runs(n_runs: int = 3, length: int = 50) -> list[list[float]]:
    return [list(RNG.normal(i, 1.0, size=length)) for i in range(n_runs)]


class TestRollingMean:
    def test_hand_case(self) -> None:
        out = rolling_mean([1.0, 2.0, 3.0, 4.0], window=2)
        assert list(out) == [1.0, 1.5, 2.5, 3.5]

    def test_empty(self) -> None:
        assert len(rolling_mean([], window=5)) == 0


class TestFigureGenerators:
    def test_learning_curves(self, tmp_path: Path) -> None:
        path = learning_curves({"QL": _runs(), "SARSA": _runs()}, tmp_path / "f1.png")
        assert path.exists() and path.stat().st_size > 10_000

    def test_probe_curves_with_threshold(self, tmp_path: Path) -> None:
        episodes = list(range(100, 600, 100))
        series = {"QL": (episodes, [[1.0, 3.0, 6.0, 7.5, 7.9]] * 3)}
        path = probe_curves(series, tmp_path / "probes.png", threshold=7.1)
        assert path.exists()

    def test_reward_boxplots(self, tmp_path: Path) -> None:
        groups = {"QL": list(RNG.normal(8, 0.3, 10)), "MC": list(RNG.normal(5, 1.0, 10))}
        assert reward_boxplots(groups, tmp_path / "f3.png").exists()

    def test_steps_barplot_log(self, tmp_path: Path) -> None:
        path = steps_barplot(
            {"brute_force": 350.0, "q_learning": 13.0},
            {"brute_force": 12.0, "q_learning": 0.4},
            tmp_path / "f4.png",
            log_scale=True,
        )
        assert path.exists()

    def test_grid_search_heatmap(self, tmp_path: Path) -> None:
        frame = pd.DataFrame(
            {
                "alpha": np.repeat([0.05, 0.15], 4),
                "gamma": list(np.tile([0.9, 0.99], 2)) * 2,
                "eval_mean_reward": RNG.normal(7, 0.5, 8),
            }
        )
        assert grid_search_heatmap(frame, tmp_path / "f5.png").exists()

    def test_sensitivity_curves(self, tmp_path: Path) -> None:
        frame = pd.DataFrame(
            {
                "alpha": [0.05, 0.05, 0.15, 0.15] * 2,
                "algorithm": ["q_learning"] * 4 + ["sarsa"] * 4,
                "eval_mean_reward": RNG.normal(7, 0.3, 8),
            }
        )
        assert sensitivity_curves(frame, "alpha", tmp_path / "f6.png").exists()

    def test_q_values_heatmap(self, tmp_path: Path) -> None:
        q_table = RNG.normal(0, 1, size=(500, 6))
        assert q_values_heatmap(q_table, None, tmp_path / "f11.png").exists()

    def test_sample_efficiency(self, tmp_path: Path) -> None:
        x = [100.0, 1_000.0, 10_000.0, 100_000.0]
        series = {"QL": (x, [[-200.0, -50.0, 5.0, 8.0]] * 3)}
        assert sample_efficiency_plot(series, tmp_path / "f9.png").exists()


class TestEpisodeGif:
    def test_gif_written_for_trained_agent(self, tmp_path: Path) -> None:
        pytest.importorskip("pygame")
        from src.agents import create_agent
        from src.config import Config
        from src.environments import create_env
        from src.training.trainer import Trainer
        from src.visualization.episode_replay import save_episode_gif

        config = Config(algorithm="q_learning", seed=1)
        env = create_env(config)
        agent = create_agent(config, env)
        Trainer(agent, env, config).train(400)
        path = save_episode_gif(agent, tmp_path / "episode.gif", seed=7, max_steps=40)
        assert path.exists() and path.stat().st_size > 1_000
