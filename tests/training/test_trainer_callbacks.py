"""Trainer + callbacks tests: end-to-end loop, probes, stop conditions."""

import numpy as np
import pytest

from src.agents import create_agent
from src.config import Config
from src.environments import create_env
from src.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    StopTraining,
    TimeBudgetCallback,
)
from src.training.trainer import EpisodeResult, Trainer, TrainingHistory


def make_result(reward: float = 5.0, steps: int = 10) -> EpisodeResult:
    return EpisodeResult(
        reward=reward,
        raw_reward=reward,
        steps=steps,
        terminated=True,
        truncated=False,
        illegal_actions=0,
    )


class TestTrainerLoop:
    def test_brute_force_end_to_end(self) -> None:
        config = Config(algorithm="brute_force", seed=42)
        env = create_env(config)
        agent = create_agent(config, env)
        history = Trainer(agent, env, config).train(5)
        assert len(history.rewards) == 5
        assert len(history.steps) == 5
        assert len(history.terminated) == 5
        assert history.wall_time > 0
        assert history.stop_reason == "completed"
        assert all(s > 0 for s in history.steps)

    def test_q_learning_improves_reward(self) -> None:
        config = Config(algorithm="q_learning", alpha=0.3, epsilon_decay=0.995, seed=1)
        env = create_env(config)
        agent = create_agent(config, env)
        history = Trainer(agent, env, config).train(800)
        assert history.mean_reward(100) > np.mean(history.rewards[:100])

    def test_probe_wiring(self) -> None:
        config = Config(algorithm="q_learning", eval_interval=5, n_probe_episodes=3, seed=3)
        env = create_env(config)
        probe_env = create_env(config)
        agent = create_agent(config, env)
        history = Trainer(agent, env, config, probe_env=probe_env).train(12)
        assert history.probe_episodes == [5, 10]
        assert len(history.probe_rewards) == 2
        assert len(history.probe_steps) == 2

    def test_history_dataframe_and_dict(self) -> None:
        history = TrainingHistory(
            rewards=[1.0, 2.0],
            shaped_rewards=[1.0, 2.0],
            steps=[3, 4],
            epsilons=[0.9, 0.8],
            illegal_actions=[0, 1],
            terminated=[True, False],
        )
        df = history.to_dataframe()
        assert list(df.columns) == [
            "episode",
            "reward",
            "shaped_reward",
            "steps",
            "epsilon",
            "illegal_actions",
            "terminated",
        ]
        assert history.to_dict()["rewards"] == [1.0, 2.0]
        assert history.mean_reward(1) == 2.0


class TestCallbacks:
    def test_time_budget_with_fake_clock(self) -> None:
        ticks = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        callback = TimeBudgetCallback(deadline=2.5, clock=lambda: next(ticks))
        callback.on_episode_end(0, make_result())  # t=0.0
        callback.on_episode_end(1, make_result())  # t=1.0
        callback.on_episode_end(2, make_result())  # t=2.0
        with pytest.raises(StopTraining, match="time budget"):
            callback.on_episode_end(3, make_result())  # t=3.0

    def test_time_budget_stops_trainer_gracefully(self) -> None:
        config = Config(algorithm="brute_force", seed=0)
        env = create_env(config)
        agent = create_agent(config, env)
        trainer = Trainer(agent, env, config, callbacks=[TimeBudgetCallback(deadline=0.0)])
        history = trainer.train(100)
        assert len(history.rewards) == 1  # stopped after the first episode
        assert "time budget" in history.stop_reason

    def test_early_stopping_fires_on_window_mean(self) -> None:
        callback = EarlyStoppingCallback(target_reward=4.0, window=3)
        callback.on_episode_end(0, make_result(3.0))
        callback.on_episode_end(1, make_result(4.0))
        with pytest.raises(StopTraining, match="early stop"):
            callback.on_episode_end(2, make_result(5.0))

    def test_early_stopping_needs_full_window(self) -> None:
        callback = EarlyStoppingCallback(target_reward=1.0, window=5)
        for episode in range(4):
            callback.on_episode_end(episode, make_result(10.0))  # no raise yet

    def test_checkpoint_writes_file(self, tmp_path: object) -> None:
        import pathlib

        assert isinstance(tmp_path, pathlib.Path)
        config = Config(algorithm="q_learning")
        env = create_env(config)
        agent = create_agent(config, env)
        path = tmp_path / "ckpt.npz"
        callback = CheckpointCallback(agent, path, every=2)
        callback.on_episode_end(0, make_result())
        assert not path.exists()
        callback.on_episode_end(1, make_result())
        assert path.exists()

    def test_logging_callback_emits(self) -> None:
        lines: list[str] = []
        callback = LoggingCallback(every=2, window=10, log_fn=lines.append)
        callback.on_episode_end(0, make_result(1.0, steps=7))
        callback.on_episode_end(1, make_result(3.0, steps=9))
        assert len(lines) == 1
        assert "reward" in lines[0] and "episode 2" in lines[0]
