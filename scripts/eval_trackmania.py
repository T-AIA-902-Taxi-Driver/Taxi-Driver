"""Evaluate a trained SAC checkpoint on TrackMania 2020 and detect lap completions.

Runs N deterministic (greedy) episodes against the real tmrl environment and
reports per-episode reward, length and duration, plus lap detection:

    lap completed  <=>  terminated (not truncated) AND final-step reward >= threshold

Rationale: tmrl's reward is progress along a demonstration trajectory; the
only ``terminated=True`` outcomes are reaching the end of the track (per-step
progress + the ``END_OF_TRACK`` bonus, +100 by default) or the
``FAILURE_COUNTDOWN`` no-progress cut (no bonus). Hitting ``ep_max_length``
yields ``truncated``, not ``terminated``. The +100 bonus dwarfs normal
per-step progress rewards, so a threshold of 50 separates the two reliably;
it is a flag in case the machine's REWARD_CONFIG differs.

Lap time = episode length x 0.05 s (the 20 Hz control period).

Usage (game running, tmrl map loaded):
    python scripts/eval_trackmania.py --episodes 5
    python scripts/eval_trackmania.py --model models/trackmania/sac_trackmania_last.zip
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TIME_STEP_S = 0.05  # tmrl rtgym control period (20 Hz)


def _require(module: str, hint: str) -> None:
    """Exit with an actionable message when a game-machine dependency is missing."""
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f"Missing dependency '{module}'. {hint} See docs/TRACKMANIA.md.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a SAC checkpoint on TrackMania 2020 (game machine only)."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/trackmania/sac_trackmania_final.zip"),
        help="SAC checkpoint (.zip) to evaluate.",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Evaluation episodes to run.")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample the policy instead of using the deterministic mean action.",
    )
    parser.add_argument(
        "--lap-bonus-threshold",
        type=float,
        default=50.0,
        help="Final-step reward above which a terminated episode counts as a lap "
        "(END_OF_TRACK bonus is 100 by default).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/trackmania/eval.json"),
        help="JSON report destination.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the checkpoint, run greedy episodes, report rewards and laps."""
    args = _parse_args()

    _require(
        "tmrl",
        "Install TrackMania 2020 + OpenPlanet, then `pip install tmrl` on the game machine.",
    )
    _require(
        "stable_baselines3",
        "Install the deep-RL extra with `poetry install -E trackmania`.",
    )
    if not args.model.exists():
        raise SystemExit(f"Checkpoint not found: {args.model}")

    from stable_baselines3 import SAC

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.environments.trackmania_wrapper import make_trackmania_env

    env = make_trackmania_env()
    model = SAC.load(str(args.model), device="auto")
    deterministic = not args.stochastic
    print(f"Evaluating {args.model} for {args.episodes} episodes (deterministic={deterministic})")

    episodes: list[dict[str, float | int | bool]] = []
    try:
        for index in range(args.episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            length = 0
            final_reward = 0.0
            terminated = truncated = False
            start = time.perf_counter()
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(action)
                final_reward = float(reward)
                total_reward += final_reward
                length += 1
            wall = time.perf_counter() - start
            lap = terminated and final_reward >= args.lap_bonus_threshold
            episodes.append(
                {
                    "episode": index,
                    "reward": total_reward,
                    "length": length,
                    "duration_s": length * TIME_STEP_S,
                    "wall_time_s": wall,
                    "terminated": terminated,
                    "truncated": truncated,
                    "final_step_reward": final_reward,
                    "lap": lap,
                }
            )
            status = "LAP COMPLETED" if lap else ("terminated" if terminated else "truncated")
            print(
                f"episode {index}: reward={total_reward:9.2f}  steps={length:4d}  "
                f"time={length * TIME_STEP_S:6.2f}s  [{status}]"
            )
    finally:
        env.wait()
        env.close()

    laps = [e for e in episodes if e["lap"]]
    lap_times = sorted(float(e["duration_s"]) for e in laps)
    summary = {
        "model": str(args.model),
        "deterministic": deterministic,
        "n_episodes": len(episodes),
        "n_laps": len(laps),
        "best_lap_time_s": lap_times[0] if lap_times else None,
        "mean_reward": (
            sum(float(e["reward"]) for e in episodes) / len(episodes) if episodes else 0.0
        ),
        "lap_bonus_threshold": args.lap_bonus_threshold,
        "episodes": episodes,
    }
    print(
        f"\n{summary['n_laps']}/{summary['n_episodes']} laps completed"
        + (f", best lap {summary['best_lap_time_s']:.2f}s" if lap_times else "")
        + f", mean reward {summary['mean_reward']:.2f}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
