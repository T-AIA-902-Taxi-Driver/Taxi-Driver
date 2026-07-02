"""Tests for the Config dataclass: YAML loading, CLI merge, validation."""

import argparse
import math
from pathlib import Path

import pytest

from src.config import Config


class TestDefaults:
    def test_default_config_is_valid(self) -> None:
        config = Config()
        assert config.algorithm == "q_learning"
        assert config.seed == 42
        assert config.early_stopping is False

    def test_config_hash_is_stable_across_seeds(self) -> None:
        assert Config(seed=1).config_hash() == Config(seed=2).config_hash()
        assert Config(alpha=0.1).config_hash() != Config(alpha=0.2).config_hash()


class TestFromYaml:
    def test_loads_values(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("alpha: 0.3\ngamma: 0.95\nalgorithm: sarsa\n")
        config = Config.from_yaml(path)
        assert config.alpha == 0.3
        assert config.gamma == 0.95
        assert config.algorithm == "sarsa"
        # untouched fields keep their defaults
        assert config.epsilon == 1.0

    def test_empty_file_gives_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("")
        assert Config.from_yaml(path) == Config()

    def test_unknown_key_raises_with_key_name(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("learning_rate: 0.1\n")
        with pytest.raises(ValueError, match="learning_rate"):
            Config.from_yaml(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            Config.from_yaml(tmp_path / "nope.yaml")

    def test_default_yaml_matches_dataclass_defaults(self) -> None:
        config = Config.from_yaml(Path(__file__).parent.parent / "configs" / "default.yaml")
        assert config == Config()


class TestMergeCliArgs:
    def test_cli_overrides_yaml(self) -> None:
        base = Config(alpha=0.3)
        args = argparse.Namespace(alpha=0.5, gamma=None)
        merged = base.merge_cli_args(args)
        assert merged.alpha == 0.5
        assert merged.gamma == base.gamma  # None → untouched

    def test_merge_returns_new_validated_instance(self) -> None:
        base = Config()
        args = argparse.Namespace(alpha=2.0)
        with pytest.raises(ValueError, match="alpha"):
            base.merge_cli_args(args)
        assert base.alpha == 0.1  # original untouched


class TestValidation:
    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("algorithm", "ppo", "algorithm"),
            ("env", "cartpole", "env"),
            ("mode", "turbo", "mode"),
            ("exploration", "greedy", "exploration"),
            ("decay_type", "cosine", "decay_type"),
            ("reward_shaping", "magic", "reward_shaping"),
            ("device", "tpu", "device"),
            ("alpha", 0.0, "alpha"),
            ("alpha", 1.5, "alpha"),
            ("gamma", 0.0, "gamma"),
            ("epsilon", 1.5, "epsilon"),
            ("decay_frac", 0.0, "decay_frac"),
            ("decay_frac", 1.5, "decay_frac"),
            ("n_test_episodes", 0, "n_test_episodes"),
            ("time_budget", 0.0, "time_budget"),
            ("tau", 0.0, "tau"),
            ("buffer_capacity", 8, "buffer_capacity"),  # < batch_size
            ("seed", -1, "seed"),
        ],
    )
    def test_invalid_values_raise(self, field: str, value: object, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            Config(**{field: value})  # type: ignore[arg-type]


class TestEffectiveDecay:
    def test_without_decay_frac_returns_raw_decay(self) -> None:
        config = Config(epsilon_decay=0.999)
        assert config.effective_decay(10_000) == 0.999

    def test_exp_decay_frac_reaches_min_at_fraction(self) -> None:
        config = Config(decay_frac=0.6, decay_type="exp", epsilon=1.0, epsilon_min=0.01)
        decay = config.effective_decay(10_000)
        horizon = 0.6 * 10_000
        assert math.isclose(1.0 * decay**horizon, 0.01, rel_tol=1e-9)

    def test_linear_decay_frac_reaches_min_at_fraction(self) -> None:
        config = Config(decay_frac=0.5, decay_type="linear", epsilon=1.0, epsilon_min=0.01)
        step = config.effective_decay(10_000)
        assert math.isclose(1.0 - step * 5_000, 0.01, rel_tol=1e-9)
