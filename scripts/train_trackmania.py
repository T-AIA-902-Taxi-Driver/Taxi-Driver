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

This script must run on the Windows machine hosting TrackMania 2020,
OpenPlanet and ``tmrl`` (see ``docs/TRACKMANIA.md``). It is intentionally a
standalone script: it is not part of the ``src`` package and is never
collected by pytest, because it cannot run without the game.

Usage:
    python scripts/train_trackmania.py --timesteps 500000 --seed 42
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_EVERY_STEPS = 50_000


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
        "--timesteps", type=int, default=200_000, help="Total environment steps (default: 200000)."
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
        help="Directory for Monitor episode logs.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for SAC (optional).")
    return parser.parse_args()


def main() -> None:
    """Build the wrapped TrackMania environment and train SAC on it."""
    args = _parse_args()

    _require(
        "tmrl",
        "Install TrackMania 2020 + OpenPlanet, then `pip install tmrl` on the game machine.",
    )
    _require(
        "stable_baselines3",
        "Install the deep-RL extra with `poetry install -E trackmania`.",
    )

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.environments.trackmania_wrapper import make_trackmania_env

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(make_trackmania_env(), filename=str(args.log_dir / "monitor.csv"))

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.995,
        tau=0.005,
        train_freq=1,
        seed=args.seed,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_EVERY_STEPS,
        save_path=str(args.checkpoint_dir),
        name_prefix="sac_trackmania",
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback, progress_bar=True)

    final_path = args.checkpoint_dir / "sac_trackmania_final"
    model.save(final_path)
    print(f"Training done: final model saved to {final_path}.zip")


if __name__ == "__main__":
    main()
