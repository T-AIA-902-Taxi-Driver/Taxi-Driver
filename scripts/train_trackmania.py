"""Train a Soft Actor-Critic (SAC) agent on TrackMania 2020 via ``tmrl``.

Why SAC rather than PPO?
    The tmrl environment is real-time and single-instance: the game runs at
    ~20 FPS wall-clock, cannot be vectorised, paused or fast-forwarded, so
    every collected transition is expensive. SAC is off-policy and replays
    each transition many times from its buffer, making it far more
    sample-efficient than on-policy PPO in this setting. It natively handles
    the continuous ``[gas, brake, steer]`` action space, and its entropy
    regularisation keeps exploration alive on long tracks. Finally, tmrl's
    own reference training pipeline is SAC-based, which keeps our results
    comparable to the upstream baseline.

Hyperparameters live in ``configs/trackmania.yaml`` (precedence: script
defaults < YAML < CLI flags) and are aligned with tmrl's reference SAC
configuration; the resolved values of every run are dumped to
``<checkpoint-dir>/run_config.json``.

This script must run on the Windows machine hosting TrackMania 2020,
OpenPlanet and ``tmrl`` (see ``docs/TRACKMANIA.md``). It is intentionally a
standalone script: it is not part of the ``src`` package and is never
collected by pytest, because it cannot run without the game.

Usage:
    python scripts/train_trackmania.py --config configs/trackmania.yaml
    python scripts/train_trackmania.py --timesteps 2000   # smoke run
    python scripts/train_trackmania.py --resume           # continue last run
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_RE = re.compile(r"sac_trackmania_(\d+)_steps\.zip$")


@dataclass
class TrainConfig:
    """Resolved SAC training configuration (defaults < YAML < CLI)."""

    timesteps: int = 200_000
    seed: int | None = None
    learning_rate: float = 3.0e-5
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.995
    tau: float = 0.005
    ent_coef: float = 0.01
    train_freq: int = 1
    gradient_steps: int = 1
    train_freq_episode: bool = False
    learning_starts: int = 5_000
    net_arch: list[int] = dataclasses.field(default_factory=lambda: [256, 256])
    checkpoint_every: int = 25_000
    buffer_save_every: int = 100_000
    device: str = "auto"
    run_name: str = "sac_trackmania"


def _require(module: str, hint: str) -> None:
    """Exit with an actionable message when a game-machine dependency is missing.

    Args:
        module: Importable module name to check for.
        hint: Installation instructions shown to the user.
    """
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f"Missing dependency '{module}'. {hint} See docs/TRACKMANIA.md.")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train a SAC agent on TrackMania 2020 through tmrl (game machine only)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML hyperparameter file (default: configs/trackmania.yaml if it exists).",
    )
    parser.add_argument("--timesteps", type=int, default=None, help="Environment steps to run.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for SAC.")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="auto | cpu | cuda")
    parser.add_argument("--run-name", type=str, default=None, help="TensorBoard run name.")
    parser.add_argument(
        "--train-freq-episode",
        action="store_true",
        help="Train in bursts at episode boundaries instead of one gradient step per env step "
        "(fallback when per-step updates miss the 50 ms real-time budget).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the most recent checkpoint in --checkpoint-dir "
        "(--timesteps then means ADDITIONAL steps).",
    )
    parser.add_argument(
        "--resume-from", type=Path, default=None, help="Explicit checkpoint .zip to resume from."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("models/trackmania"),
        help="Directory for periodic checkpoints and the final model.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("results/trackmania"),
        help="Directory for Monitor episode logs and TensorBoard events.",
    )
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> TrainConfig:
    """Merge defaults, the YAML file and CLI overrides into a TrainConfig.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The resolved training configuration.
    """
    import yaml

    config = TrainConfig()

    yaml_path = args.config
    if yaml_path is None:
        default_yaml = REPO_ROOT / "configs" / "trackmania.yaml"
        yaml_path = default_yaml if default_yaml.exists() else None
    if yaml_path is not None:
        with open(yaml_path, encoding="utf-8") as handle:
            loaded: dict[str, Any] = yaml.safe_load(handle) or {}
        known = {field.name for field in dataclasses.fields(TrainConfig)}
        unknown = set(loaded) - known
        if unknown:
            raise SystemExit(f"Unknown keys in {yaml_path}: {sorted(unknown)}")
        for key, value in loaded.items():
            setattr(config, key, value)

    cli_fields = {
        "timesteps": args.timesteps,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "buffer_size": args.buffer_size,
        "ent_coef": args.ent_coef,
        "learning_starts": args.learning_starts,
        "checkpoint_every": args.checkpoint_every,
        "device": args.device,
        "run_name": args.run_name,
    }
    for key, value in cli_fields.items():
        if value is not None:
            setattr(config, key, value)
    if args.train_freq_episode:
        config.train_freq_episode = True
    return config


def _find_resume_checkpoint(checkpoint_dir: Path, explicit: Path | None) -> Path:
    """Locate the checkpoint to resume from.

    Prefers the most recently modified among the interrupt save
    (``sac_trackmania_last.zip``), periodic checkpoints and the final model.

    Args:
        checkpoint_dir: Directory containing the checkpoints.
        explicit: Explicit path passed via ``--resume-from``, if any.

    Returns:
        Path of the checkpoint to load.

    Raises:
        SystemExit: If no checkpoint is found.
    """
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"--resume-from checkpoint not found: {explicit}")
        return explicit
    candidates = [
        path
        for path in checkpoint_dir.glob("sac_trackmania*.zip")
        if CHECKPOINT_RE.search(path.name)
        or path.stem in ("sac_trackmania_last", "sac_trackmania_final")
    ]
    if not candidates:
        raise SystemExit(f"--resume: no sac_trackmania*.zip checkpoint found in {checkpoint_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    """Build the wrapped TrackMania environment and train SAC on it."""
    args = _parse_args()
    config = _load_config(args)

    _require(
        "tmrl",
        "Install TrackMania 2020 + OpenPlanet, then `pip install tmrl` on the game machine.",
    )
    _require(
        "stable_baselines3",
        "Install the deep-RL extra with `poetry install -E trackmania`.",
    )

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.environments.trackmania_wrapper import make_trackmania_env

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    buffer_path = args.checkpoint_dir / "replay_buffer.pkl"

    resuming = args.resume or args.resume_from is not None
    env = Monitor(
        make_trackmania_env(),
        filename=str(args.log_dir / "monitor.csv"),
        override_existing=not resuming,
    )

    class ReplayBufferSnapshotCallback(BaseCallback):
        """Periodically overwrite a single on-disk replay-buffer snapshot.

        ``CheckpointCallback(save_replay_buffer=True)`` would write one
        full-size (~700 MB, pre-allocated) pickle per checkpoint; one rolling
        snapshot bounds disk usage while still allowing resume.
        """

        def __init__(self, every: int, path: Path):
            super().__init__()
            self._every = every
            self._path = path

        def _on_step(self) -> bool:
            if self.num_timesteps % self._every == 0:
                self.model.save_replay_buffer(str(self._path))  # type: ignore[attr-defined]
            return True

    if resuming:
        checkpoint = _find_resume_checkpoint(args.checkpoint_dir, args.resume_from)
        print(f"Resuming from {checkpoint}")
        model = SAC.load(str(checkpoint), env=env, device=config.device)
        if buffer_path.exists():
            model.load_replay_buffer(str(buffer_path))
            print(f"Replay buffer restored ({model.replay_buffer.size()} transitions)")
        else:
            print("No replay buffer snapshot found: continuing with an empty buffer")
    else:
        train_freq: int | tuple[int, str] = config.train_freq
        gradient_steps = config.gradient_steps
        if config.train_freq_episode:
            train_freq = (1, "episode")
            gradient_steps = -1
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=config.learning_rate,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts,
            batch_size=config.batch_size,
            tau=config.tau,
            gamma=config.gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            ent_coef=config.ent_coef,
            policy_kwargs={"net_arch": list(config.net_arch)},
            tensorboard_log=str(args.log_dir / "tb"),
            seed=config.seed,
            device=config.device,
            verbose=1,
        )

    run_config = {
        **dataclasses.asdict(config),
        "resumed_from": str(checkpoint) if resuming else None,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sb3_policy": "MlpPolicy",
    }
    with open(args.checkpoint_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    callbacks = [
        CheckpointCallback(
            save_freq=config.checkpoint_every,
            save_path=str(args.checkpoint_dir),
            name_prefix="sac_trackmania",
            save_replay_buffer=False,
        ),
        ReplayBufferSnapshotCallback(config.buffer_save_every, buffer_path),
    ]

    interrupted = False
    try:
        model.learn(
            total_timesteps=config.timesteps,
            callback=callbacks,
            log_interval=1,
            tb_log_name=config.run_name,
            reset_num_timesteps=not resuming,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted: saving state before exit...")
    finally:
        last_path = args.checkpoint_dir / "sac_trackmania_last"
        model.save(str(last_path))
        model.save_replay_buffer(str(buffer_path))
        print(f"State saved: {last_path}.zip + {buffer_path}")
        print("Resume with: python scripts/train_trackmania.py --resume --timesteps <additional>")
        env.unwrapped.wait()
        env.close()

    if not interrupted:
        final_path = args.checkpoint_dir / "sac_trackmania_final"
        model.save(str(final_path))
        print(f"Training done: final model saved to {final_path}.zip")


if __name__ == "__main__":
    main()
