"""Configuration management for the Taxi Driver project.

Configuration values are resolved with the following precedence (lowest to highest):
dataclass defaults < YAML file (``--config``) < CLI arguments.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

VALID_ALGORITHMS = (
    "brute_force",
    "q_learning",
    "sarsa",
    "expected_sarsa",
    "double_q_learning",
    "monte_carlo",
    "dqn",
)
VALID_ENVS = ("taxi", "multi")
VALID_MODES = ("user", "time_limited")
VALID_EXPLORATIONS = ("epsilon_greedy", "boltzmann", "ucb")
VALID_DECAY_TYPES = ("exp", "linear")
VALID_REWARD_SHAPING = ("none", "potential", "naive_distance", "step_penalty")
VALID_DEVICES = ("auto", "cpu", "cuda")


@dataclass
class Config:
    """Central configuration for agents, environments, training and evaluation.

    Attributes mirror ``configs/default.yaml``; see that file for a commented
    reference of every field.
    """

    # --- General -----------------------------------------------------------
    algorithm: str = "q_learning"
    env: str = "taxi"
    mode: str = "user"
    seed: int = 42
    n_train_episodes: int = 10_000
    n_test_episodes: int = 100
    max_steps_per_episode: int = 200
    time_budget: float = 60.0

    # --- Tabular hyperparameters -------------------------------------------
    alpha: float = 0.1
    gamma: float = 0.99

    # --- Exploration ---------------------------------------------------------
    exploration: str = "epsilon_greedy"
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.9995
    decay_type: str = "exp"
    # When set, ``epsilon_decay`` is recomputed so that epsilon reaches
    # ``epsilon_min`` at ``decay_frac * n_train_episodes`` (see effective_decay()).
    decay_frac: float | None = None
    temperature: float = 1.0
    temperature_min: float = 0.05
    temperature_decay: float = 0.999
    ucb_c: float = 2.0

    # --- Reward shaping ------------------------------------------------------
    reward_shaping: str = "none"

    # --- DQN -----------------------------------------------------------------
    lr: float = 1e-3
    batch_size: int = 64
    hidden_size: int = 128
    buffer_capacity: int = 50_000
    warmup_steps: int = 1_000
    train_every: int = 1
    tau: float = 0.005
    double_dqn: bool = True
    grad_clip: float = 10.0
    device: str = "auto"

    # --- Training control ----------------------------------------------------
    # Early stopping is opt-in: it must stay disabled during benchmark
    # comparisons (stopping some algorithms earlier than others would bias
    # every convergence metric); the time-limited mode enables it explicitly.
    early_stopping: bool = False
    patience: int = 500
    target_reward: float = 8.0
    eval_interval: int = 0
    n_probe_episodes: int = 30

    # --- IO --------------------------------------------------------------------
    model_path: str = "models"
    results_dir: str = "results"

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a configuration from a YAML file on top of the defaults.

        Args:
            path: Path to a YAML file whose keys are Config field names.

        Returns:
            A validated Config instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the YAML contains unknown keys or invalid values.
        """
        path = Path(path)
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"Unknown configuration keys in {path}: {', '.join(unknown)}")
        return cls(**raw)

    def merge_cli_args(self, args: argparse.Namespace) -> Config:
        """Return a new Config where non-None CLI arguments override fields.

        Args:
            args: Parsed argparse namespace; only attributes matching Config
                field names and not None are applied.

        Returns:
            A new, validated Config instance.
        """
        overrides = {
            f.name: getattr(args, f.name)
            for f in fields(self)
            if getattr(args, f.name, None) is not None
        }
        return dataclasses.replace(self, **overrides)

    # ------------------------------------------------------------------ helpers

    def effective_decay(self, n_episodes: int) -> float:
        """Per-episode epsilon decay parameter honouring ``decay_frac``.

        When ``decay_frac`` is set, the decay is recomputed so that epsilon
        reaches ``epsilon_min`` after ``decay_frac * n_episodes`` episodes,
        decoupling the decay *shape* from the training horizon:

        - exp: epsilon is multiplied per episode by
          ``(epsilon_min / epsilon) ** (1 / (decay_frac * n_episodes))``
        - linear: epsilon decreases per episode by
          ``(epsilon - epsilon_min) / (decay_frac * n_episodes)``

        Args:
            n_episodes: Training horizon used to resolve ``decay_frac``.

        Returns:
            Multiplicative factor (exp) or subtractive step (linear).
        """
        if self.decay_frac is None:
            return self.epsilon_decay
        horizon = max(1.0, self.decay_frac * n_episodes)
        if self.decay_type == "exp":
            if self.epsilon <= 0:
                return self.epsilon_decay
            return float((self.epsilon_min / self.epsilon) ** (1.0 / horizon))
        return float((self.epsilon - self.epsilon_min) / horizon)

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a plain dictionary."""
        return dataclasses.asdict(self)

    def config_hash(self, exclude: tuple[str, ...] = ("seed",)) -> str:
        """Deterministic 8-hex-char hash of the config, excluding given fields.

        Used to name benchmark runs: the same configuration keeps the same
        hash across seeds and timestamps.
        """
        import hashlib

        payload = {k: v for k, v in self.to_dict().items() if k not in exclude}
        canonical = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:8]

    # ---------------------------------------------------------------- validation

    def validate(self) -> None:
        """Validate all fields, raising ValueError with an explicit message."""
        checks: list[tuple[bool, str]] = [
            (self.algorithm in VALID_ALGORITHMS, f"algorithm must be one of {VALID_ALGORITHMS}"),
            (self.env in VALID_ENVS, f"env must be one of {VALID_ENVS}"),
            (self.mode in VALID_MODES, f"mode must be one of {VALID_MODES}"),
            (
                self.exploration in VALID_EXPLORATIONS,
                f"exploration must be one of {VALID_EXPLORATIONS}",
            ),
            (
                self.decay_type in VALID_DECAY_TYPES,
                f"decay_type must be one of {VALID_DECAY_TYPES}",
            ),
            (
                self.reward_shaping in VALID_REWARD_SHAPING,
                f"reward_shaping must be one of {VALID_REWARD_SHAPING}",
            ),
            (self.device in VALID_DEVICES, f"device must be one of {VALID_DEVICES}"),
            (0.0 < self.alpha <= 1.0, "alpha (learning rate) must be in (0, 1]"),
            (0.0 < self.gamma <= 1.0, "gamma (discount factor) must be in (0, 1]"),
            (0.0 <= self.epsilon <= 1.0, "epsilon must be in [0, 1]"),
            (
                0.0 <= self.epsilon_min <= self.epsilon or self.epsilon == 0.0,
                "epsilon_min must be in [0, epsilon]",
            ),
            (self.epsilon_decay > 0.0, "epsilon_decay must be > 0"),
            (
                self.decay_frac is None or 0.0 < self.decay_frac <= 1.0,
                "decay_frac must be in (0, 1] when set",
            ),
            (self.temperature > 0.0, "temperature must be > 0"),
            (self.temperature_min > 0.0, "temperature_min must be > 0"),
            (self.ucb_c >= 0.0, "ucb_c must be >= 0"),
            (self.n_train_episodes >= 0, "n_train_episodes must be >= 0"),
            (self.n_test_episodes > 0, "n_test_episodes must be > 0"),
            (self.max_steps_per_episode > 0, "max_steps_per_episode must be > 0"),
            (self.time_budget > 0.0, "time_budget must be > 0 seconds"),
            (self.lr > 0.0, "lr (DQN learning rate) must be > 0"),
            (self.batch_size > 0, "batch_size must be > 0"),
            (self.hidden_size > 0, "hidden_size must be > 0"),
            (self.buffer_capacity >= self.batch_size, "buffer_capacity must be >= batch_size"),
            (self.warmup_steps >= 0, "warmup_steps must be >= 0"),
            (self.train_every > 0, "train_every must be > 0"),
            (0.0 < self.tau <= 1.0, "tau (soft update) must be in (0, 1]"),
            (self.grad_clip > 0.0, "grad_clip must be > 0"),
            (self.patience > 0, "patience must be > 0"),
            (self.eval_interval >= 0, "eval_interval must be >= 0"),
            (self.n_probe_episodes > 0, "n_probe_episodes must be > 0"),
            (self.seed >= 0, "seed must be >= 0"),
        ]
        errors = [message for ok, message in checks if not ok]
        if errors:
            raise ValueError("Invalid configuration: " + "; ".join(errors))


# Fields exposed as CLI overrides by the parser (PR: feature/cli).
CLI_OVERRIDABLE_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Config))
