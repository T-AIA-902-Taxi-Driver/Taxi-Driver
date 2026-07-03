"""Validate the TrackMania/tmrl setup with a random agent and timing diagnostics.

Runs a random policy against the real tmrl environment (game machine only) and
reports everything needed to declare the setup training-ready:

- observation/action space validation (exact TM20LIDAR layout),
- real-time health against the 20 Hz budget: rtgym's own timing-violation
  warnings are counted (the authoritative signal — rtgym self-corrects small
  caller-side overruns), with per-step wall-time stats reported as context,
- reward statistics (the random-policy baseline that training must beat),
- episode statistics and NaN/range checks on observations,
- the last ``info`` dict verbatim (to discover what tmrl exposes there).

Also manages the versioned tmrl machine-config template
(``configs/tmrl_config.json``): ``--diff-config`` compares it against the live
``%USERPROFILE%/TmrlData/config/config.json`` (ignoring machine-secret keys),
``--apply-config`` copies the template values onto the live config (preserving
those secrets), and ``--capture-config`` snapshots the live config into the
template with secrets redacted.

Usage (game running, tmrl map loaded, window 958x488 top-left):
    python scripts/check_trackmania_setup.py --steps 400 --json results/trackmania/check_setup.json
    python scripts/check_trackmania_setup.py --diff-config
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPO_ROOT / "configs" / "tmrl_config.json"

# Keys whose values are machine secrets or machine-specific: never versioned,
# never overwritten by --apply-config.
REDACTED_KEYS = {"PASSWORD", "WANDB_KEY", "WANDB_ENTITY", "WANDB_PROJECT", "PUBLIC_IP_SERVER"}
REDACTION_PLACEHOLDER = "<machine-specific>"

MISS_RATE_LIMIT = 0.10  # tolerated rate of rtgym timing violations


def _live_config_path() -> Path:
    return Path.home() / "TmrlData" / "config" / "config.json"


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        loaded: dict[str, Any] = json.load(handle)
    return loaded


def _flatten_keys(obj: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into dotted-path keys for comparison."""
    flat: dict[str, Any] = {}
    for key, value in obj.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_keys(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def _is_redacted(dotted: str) -> bool:
    return dotted.split(".")[-1] in REDACTED_KEYS


def capture_config() -> int:
    """Snapshot the live tmrl config into the versioned template, redacted."""
    live = _load_json(_live_config_path())

    def redact(obj: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key in REDACTED_KEYS:
                out[key] = REDACTION_PLACEHOLDER
            elif isinstance(value, dict):
                out[key] = redact(value)
            else:
                out[key] = value
        return out

    TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TEMPLATE_PATH, "w", encoding="utf-8") as handle:
        json.dump(redact(live), handle, indent=2)
        handle.write("\n")
    print(f"Live config captured (redacted) to {TEMPLATE_PATH}")
    return 0


def diff_config() -> int:
    """Compare the versioned template with the live config (secrets ignored)."""
    if not TEMPLATE_PATH.exists():
        print(f"No template at {TEMPLATE_PATH}; run --capture-config first.")
        return 1
    template = _flatten_keys(_load_json(TEMPLATE_PATH))
    live = _flatten_keys(_load_json(_live_config_path()))
    differences = []
    for key in sorted(set(template) | set(live)):
        if _is_redacted(key):
            continue
        t_val, l_val = template.get(key, "<absent>"), live.get(key, "<absent>")
        if t_val != l_val:
            differences.append(f"  {key}: template={t_val!r} live={l_val!r}")
    if differences:
        print(f"{len(differences)} difference(s) between template and live config:")
        print("\n".join(differences))
        return 1
    print("Live tmrl config matches the versioned template.")
    return 0


def apply_config() -> int:
    """Copy template values onto the live config, preserving machine secrets."""
    if not TEMPLATE_PATH.exists():
        print(f"No template at {TEMPLATE_PATH}; run --capture-config first.")
        return 1
    live_path = _live_config_path()
    live = _load_json(live_path)

    def merge(template: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
        out = dict(target)
        for key, value in template.items():
            if key in REDACTED_KEYS:
                continue  # keep the live machine value
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = merge(value, out[key])
            else:
                out[key] = value
        return out

    backup = live_path.with_suffix(".json.bak")
    backup.write_text(live_path.read_text(encoding="utf-8"), encoding="utf-8")
    merged = merge(_load_json(TEMPLATE_PATH), live)
    with open(live_path, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)
        handle.write("\n")
    print(f"Template applied to {live_path} (backup: {backup})")
    return 0


def run_check(steps: int, json_out: Path | None) -> int:
    """Drive the real environment with random actions and report diagnostics.

    Args:
        steps: Number of environment steps to run.
        json_out: Optional path for the JSON report.

    Returns:
        Process exit code (0 = setup is training-ready).
    """
    if importlib.util.find_spec("tmrl") is None:
        raise SystemExit(
            "tmrl is not installed; this check runs on the game machine only. "
            "See docs/TRACKMANIA.md."
        )
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import numpy as np

    from src.environments.trackmania_wrapper import make_trackmania_env

    print("Building the tmrl environment (validates the TM20LIDAR layout)...")
    env = make_trackmania_env()
    print(f"observation_space: {env.observation_space}")
    print(f"action_space:      {env.action_space}")

    rng = np.random.default_rng(42)
    step_times: list[float] = []
    rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_rewards: list[float] = []
    terminated_count = 0
    truncated_count = 0
    nan_seen = False
    out_of_range = False
    last_info: dict[str, Any] = {}

    print(f"Running {steps} random steps (game window must be visible and focused)...")
    rtgym_timeouts = 0
    obs, _ = env.reset()
    current_len = 0
    current_reward = 0.0
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(steps):
                action = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
                start = time.perf_counter()
                obs, reward, terminated, truncated, info = env.step(action)
                elapsed = time.perf_counter() - start
                # The terminal step legitimately includes respawn work; its
                # wall time says nothing about the steady-state 20 Hz budget.
                if not (terminated or truncated):
                    step_times.append(elapsed)

                rewards.append(float(reward))
                current_len += 1
                current_reward += float(reward)
                last_info = dict(info)
                if np.isnan(obs).any():
                    nan_seen = True
                if not env.observation_space.contains(obs):
                    out_of_range = True
                if terminated or truncated:
                    terminated_count += int(terminated)
                    truncated_count += int(truncated)
                    episode_lengths.append(current_len)
                    episode_rewards.append(current_reward)
                    current_len, current_reward = 0, 0.0
                    obs, _ = env.reset()
        # rtgym warns once per real timing violation of its internal clock —
        # the authoritative signal, unlike raw caller-side wall time, which
        # rtgym self-corrects against on the next step.
        rtgym_timeouts = sum(1 for w in caught if "timed out" in str(w.message).lower())
    finally:
        env.wait()
        env.close()

    times_ms = [t * 1000.0 for t in step_times]
    times_sorted = sorted(times_ms)
    misses = rtgym_timeouts
    miss_rate = misses / steps if steps else 1.0
    report = {
        "steps": steps,
        "step_time_ms": {
            "mean": statistics.mean(times_ms),
            "std": statistics.pstdev(times_ms),
            "p50": times_sorted[len(times_sorted) // 2],
            "p95": times_sorted[int(len(times_sorted) * 0.95)],
            "max": max(times_ms),
        },
        "rtgym_timeouts": {"count": misses, "rate": miss_rate},
        "reward": {
            "sum": sum(rewards),
            "mean_per_step": statistics.mean(rewards) if rewards else 0.0,
            "min": min(rewards) if rewards else 0.0,
            "max": max(rewards) if rewards else 0.0,
        },
        "episodes": {
            "completed": len(episode_lengths),
            "terminated": terminated_count,
            "truncated": truncated_count,
            "lengths": episode_lengths,
            "rewards": episode_rewards,
        },
        "observations": {"nan_seen": nan_seen, "out_of_range": out_of_range},
        "last_info": {k: repr(v) for k, v in last_info.items()},
    }

    print(json.dumps(report, indent=2))
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Report written to {json_out}")

    failures = []
    if nan_seen:
        failures.append("NaN values in observations")
    if out_of_range:
        failures.append("observations outside the declared Box(-1, 1)")
    if miss_rate > MISS_RATE_LIMIT:
        failures.append(
            f"rtgym timing-violation rate {miss_rate:.1%} > {MISS_RATE_LIMIT:.0%} "
            "(rtgym cannot hold 20 Hz on this machine/settings)"
        )
    if failures:
        print("SETUP NOT READY: " + "; ".join(failures))
        return 1
    print("Setup is training-ready.")
    return 0


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Validate the TrackMania/tmrl setup (random agent + timing diagnostics)."
    )
    parser.add_argument("--steps", type=int, default=400, help="Random steps to run (~20 s).")
    parser.add_argument("--json", type=Path, default=None, help="Write the JSON report here.")
    parser.add_argument(
        "--capture-config",
        action="store_true",
        help="Snapshot the live TmrlData config into configs/tmrl_config.json (redacted).",
    )
    parser.add_argument(
        "--diff-config",
        action="store_true",
        help="Diff configs/tmrl_config.json against the live TmrlData config.",
    )
    parser.add_argument(
        "--apply-config",
        action="store_true",
        help="Apply configs/tmrl_config.json onto the live TmrlData config (backs it up first).",
    )
    args = parser.parse_args()

    if args.capture_config:
        raise SystemExit(capture_config())
    if args.diff_config:
        raise SystemExit(diff_config())
    if args.apply_config:
        raise SystemExit(apply_config())
    raise SystemExit(run_check(args.steps, args.json))


if __name__ == "__main__":
    main()
