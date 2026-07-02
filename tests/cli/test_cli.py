"""CLI tests: parser round-trips, prompts, end-to-end train/eval commands."""

import argparse
from pathlib import Path

import pytest

from src.cli.commands import cmd_eval, cmd_train
from src.cli.parser import build_parser
from src.cli.prompts import prompt_choice, prompt_confirm, prompt_float, prompt_int
from src.main import main


class TestParser:
    def test_train_round_trip(self) -> None:
        args = build_parser().parse_args(
            [
                "train",
                "--agent",
                "sarsa",
                "--train-episodes",
                "500",
                "--test-episodes",
                "20",
                "--alpha",
                "0.2",
                "--mode",
                "user",
                "--non-interactive",
            ]
        )
        assert args.command == "train"
        assert args.algorithm == "sarsa"
        assert args.n_train_episodes == 500
        assert args.n_test_episodes == 20
        assert args.alpha == 0.2
        assert args.non_interactive is True

    def test_absent_options_are_none(self) -> None:
        args = build_parser().parse_args(["train", "--non-interactive"])
        assert args.algorithm is None
        assert args.n_train_episodes is None
        assert args.gamma is None

    def test_eval_requires_model(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["eval"])

    def test_play_and_time_limited_options(self) -> None:
        args = build_parser().parse_args(["play", "--model", "m.npz", "--delay", "0"])
        assert args.command == "play" and args.delay == 0.0
        args = build_parser().parse_args(
            ["train", "--mode", "time-limited", "--time", "30", "--non-interactive"]
        )
        assert args.mode == "time-limited" and args.time_budget == 30.0

    def test_invalid_agent_rejected(self) -> None:
        with pytest.raises(SystemExit):
            build_parser().parse_args(["train", "--agent", "ppo"])


class TestPrompts:
    def test_prompt_int_default_and_validation(self) -> None:
        answers = iter(["", "abc", "0", "7"])
        fn = lambda _label: next(answers)  # noqa: E731
        assert prompt_int("n", 42, 1, fn) == 42
        assert prompt_int("n", 42, 1, fn) == 7  # skips 'abc' then '0'

    def test_prompt_float_bounds(self) -> None:
        answers = iter(["5.0", "0.3"])
        fn = lambda _label: next(answers)  # noqa: E731
        assert prompt_float("alpha", 0.1, 0.0, 1.0, fn) == 0.3  # 5.0 rejected

    def test_prompt_choice_prefix_match(self) -> None:
        fn = lambda _label: "sar"  # noqa: E731
        assert prompt_choice("agent", ("sarsa", "q_learning"), "q_learning", fn) == "sarsa"

    def test_prompt_confirm(self) -> None:
        assert prompt_confirm("ok?", True, lambda _l: "") is True
        assert prompt_confirm("ok?", True, lambda _l: "n") is False
        assert prompt_confirm("ok?", False, lambda _l: "oui") is True


class TestCommandsEndToEnd:
    def _train_args(self, tmp_path: Path, **overrides: object) -> argparse.Namespace:
        base: dict[str, object] = {
            "command": "train",
            "mode": "user",
            "config": None,
            "time_budget": None,
            "save": str(tmp_path / "model.npz"),
            "show_episodes": 0,
            "non_interactive": True,
            "algorithm": "q_learning",
            "env": None,
            "seed": 3,
            "n_train_episodes": 300,
            "n_test_episodes": 10,
            "alpha": 0.3,
            "gamma": None,
            "exploration": None,
            "epsilon": None,
            "epsilon_min": None,
            "epsilon_decay": 0.99,
            "decay_type": None,
            "decay_frac": None,
            "temperature": None,
            "ucb_c": None,
            "reward_shaping": None,
            "lr": None,
            "device": None,
        }
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_train_then_eval_round_trip(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cmd_train(self._train_args(tmp_path)) == 0
        out = capsys.readouterr().out
        assert "mean reward" in out and "mean time per game" in out
        assert (tmp_path / "model.npz").exists()

        eval_args = argparse.Namespace(
            command="eval",
            model=str(tmp_path / "model.npz"),
            n_test_episodes=5,
            show_episodes=0,
            seed=None,
        )
        assert cmd_eval(eval_args) == 0
        out = capsys.readouterr().out
        assert "q_learning" in out and "mean reward" in out

    def test_time_limited_respects_budget(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import time

        args = self._train_args(
            tmp_path,
            mode="time_limited",
            time_budget=3.0,
            n_train_episodes=10_000_000,  # cap far beyond the budget
            algorithm="q_learning",
            epsilon_decay=None,
        )
        start = time.monotonic()
        assert cmd_train(args) == 0
        elapsed = time.monotonic() - start
        assert elapsed < 10.0  # 3s budget + eval + slack, far below cap
        out = capsys.readouterr().out
        assert "time budget" in out

    def test_main_dispatch_and_error_handling(self, tmp_path: Path) -> None:
        rc = main(["eval", "--model", str(tmp_path / "missing.npz")])
        assert rc == 1  # FileNotFoundError mapped to exit code 1


class TestDisplayEpisodes:
    def test_show_episodes_prints_grid(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = TestCommandsEndToEnd()._train_args(
            tmp_path, show_episodes=2, n_train_episodes=50, n_test_episodes=3
        )
        assert cmd_train(args) == 0
        out = capsys.readouterr().out
        assert out.count("=== Episode") == 2
        assert "+---------+" in out
