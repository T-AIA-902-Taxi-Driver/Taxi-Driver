"""Experiment runner: reproducible, idempotent, parallelizable benchmark runs.

Each run is identified by ``(exp_id, config_hash, seed)`` and materialized as
a directory under ``results/raw/<exp_id>/``:

    <algo>__<confighash8>__s<seed>__<timestamp>/
        config.json          resolved config + code version + library versions
        train_episodes.csv   one row per training episode
        probes.csv           one row per greedy probe checkpoint
        eval_episodes.csv    one row per (fixed-seed) evaluation episode
        summary.json         aggregates consumed by results/aggregated/

Runs are idempotent: an existing completed run directory for the same
identity is reused (crash-resume for free). Parallel execution across
processes is safe because runs never share files; time measurements intended
for the report must come from sequential blocks only (protocol E7).
"""

from __future__ import annotations

import csv
import dataclasses
import datetime
import json
import platform
import sys
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from src.agents import create_agent
from src.benchmarking.reward_shaping import create_reward_shaper
from src.config import Config
from src.environments import create_env
from src.evaluation.evaluator import Evaluator
from src.training.trainer import Trainer, TrainingHistory
from src.utils.seeding import eval_seeds

# All runs are EVALUATED on the same fixed episode set, regardless of their
# training seed — the precondition for paired, fair comparisons.
EVAL_BASE_SEED = 42


@dataclass(frozen=True)
class RunSpec:
    """One benchmark run: a configuration trained with one seed."""

    exp_id: str
    config: Config
    seed: int  # training seed (env + agent streams)

    @property
    def run_id(self) -> str:
        return f"{self.config.algorithm}__{self.config.config_hash()}__s{self.seed}"


def _library_versions() -> dict[str, str]:
    import gymnasium
    import scipy

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }
    if "torch" in sys.modules:
        versions["torch"] = sys.modules["torch"].__version__
    return versions


def find_completed_run(results_root: Path, spec: RunSpec) -> Path | None:
    """Return an existing completed run directory for this identity, if any."""
    exp_dir = results_root / "raw" / spec.exp_id
    if not exp_dir.exists():
        return None
    for candidate in sorted(exp_dir.glob(f"{spec.run_id}__*")):
        if (candidate / "summary.json").exists():
            return candidate
    return None


def run_single(spec: RunSpec, results_root: str | Path = "results") -> dict[str, Any]:
    """Execute one benchmark run (train → probe → evaluate → persist).

    Args:
        spec: Run specification; ``spec.seed`` overrides the config seed for
            training streams. Evaluation always uses the fixed
            ``EVAL_BASE_SEED`` episode set.
        results_root: Base results directory.

    Returns:
        The summary dictionary (also written to ``summary.json``).
    """
    results_root = Path(results_root)
    existing = find_completed_run(results_root, spec)
    if existing is not None:
        with open(existing / "summary.json") as handle:
            return dict(json.load(handle))

    config = dataclasses.replace(spec.config, seed=spec.seed)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = results_root / "raw" / spec.exp_id / f"{spec.run_id}__{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    reward_fn = create_reward_shaper(config.reward_shaping, config.gamma)
    env = create_env(config, reward_fn=reward_fn)
    probe_env = create_env(config) if config.eval_interval > 0 else None  # native rewards
    agent = create_agent(config, env, n_episodes=config.n_train_episodes)

    trainer = Trainer(agent, env, config, probe_env=probe_env)
    history = trainer.train(config.n_train_episodes)

    evaluator = Evaluator(env, seeds=eval_seeds(EVAL_BASE_SEED, config.n_test_episodes))
    results = evaluator.evaluate(agent, config.n_test_episodes)

    model_name = "model.pt" if config.algorithm == "dqn" else "model.npz"
    agent.save(run_dir / model_name)

    _write_config(run_dir, spec, config)
    _write_train_csv(run_dir, history)
    _write_probes_csv(run_dir, history)
    _write_eval_csv(run_dir, results, EVAL_BASE_SEED)
    summary = _build_summary(spec, config, history, results, agent.memory_bytes())
    with open(run_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _write_config(run_dir: Path, spec: RunSpec, config: Config) -> None:
    payload = {
        "exp_id": spec.exp_id,
        "run_id": spec.run_id,
        "config": config.to_dict(),
        "config_hash": config.config_hash(),
        "eval_base_seed": EVAL_BASE_SEED,
        "versions": _library_versions(),
    }
    with open(run_dir / "config.json", "w") as handle:
        json.dump(payload, handle, indent=2)


def _write_train_csv(run_dir: Path, history: TrainingHistory) -> None:
    with open(run_dir / "train_episodes.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["episode", "reward", "shaped_reward", "steps", "epsilon", "illegal", "terminated"]
        )
        for i in range(len(history.rewards)):
            writer.writerow(
                [
                    i,
                    history.rewards[i],
                    history.shaped_rewards[i],
                    history.steps[i],
                    f"{history.epsilons[i]:.6f}",
                    history.illegal_actions[i],
                    int(history.terminated[i]),
                ]
            )


def _write_probes_csv(run_dir: Path, history: TrainingHistory) -> None:
    with open(run_dir / "probes.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "probe_mean_reward", "probe_mean_steps", "probe_max_q"])
        max_qs = history.probe_max_q or [float("nan")] * len(history.probe_episodes)
        for episode, reward, steps, max_q in zip(
            history.probe_episodes, history.probe_rewards, history.probe_steps, max_qs, strict=True
        ):
            writer.writerow([episode, reward, steps, max_q])


def _write_eval_csv(run_dir: Path, results: Any, eval_base_seed: int) -> None:
    seeds = eval_seeds(eval_base_seed, results.n_episodes)
    with open(run_dir / "eval_episodes.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["episode", "eval_seed", "reward", "steps", "success", "illegal", "seconds"]
        )
        for i in range(results.n_episodes):
            writer.writerow(
                [
                    i,
                    seeds[i],
                    results.rewards[i],
                    results.steps[i],
                    int(results.successes[i]),
                    results.illegal_actions[i],
                    f"{results.durations[i]:.6f}",
                ]
            )


def _first_success_episode(history: TrainingHistory) -> int | None:
    for i, terminated in enumerate(history.terminated):
        if terminated:
            return i
    return None


def _build_summary(
    spec: RunSpec,
    config: Config,
    history: TrainingHistory,
    results: Any,
    memory_bytes: int,
) -> dict[str, Any]:
    last100 = history.rewards[-100:]
    return {
        "exp_id": spec.exp_id,
        "run_id": spec.run_id,
        "algorithm": config.algorithm,
        "config_hash": config.config_hash(),
        "seed": spec.seed,
        # training
        "n_train_episodes": len(history.rewards),
        "train_wall_time_s": history.wall_time,
        "stop_reason": history.stop_reason,
        "first_success_episode": _first_success_episode(history),
        "post_convergence_std": float(np.std(last100)) if last100 else None,
        "cum_env_steps": int(np.sum(history.steps)),
        # probes (raw series kept for threshold computation at aggregation)
        "probe_episodes": history.probe_episodes,
        "probe_rewards": history.probe_rewards,
        "probe_max_q": history.probe_max_q,
        # evaluation (fixed seed set)
        "eval_mean_reward": results.mean_reward,
        "eval_std_reward": results.std_reward,
        "eval_median_reward": results.median_reward,
        "eval_mean_steps": results.mean_steps,
        "eval_median_steps": results.median_steps,
        "eval_success_rate": results.success_rate,
        "eval_mean_illegal": results.mean_illegal_actions,
        "eval_mean_episode_seconds": results.mean_episode_seconds,
        # resources
        "memory_bytes": memory_bytes,
        # hyperparameters (flattened for aggregation convenience)
        "alpha": config.alpha,
        "gamma": config.gamma,
        "exploration": config.exploration,
        "decay_type": config.decay_type,
        "decay_frac": config.decay_frac,
        "epsilon_decay": config.epsilon_decay,
        "reward_shaping": config.reward_shaping,
    }


def _run_single_star(payload: tuple[RunSpec, str]) -> dict[str, Any]:
    spec, results_root = payload
    # Keep BLAS/torch from oversubscribing cores in parallel pools.
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
    return run_single(spec, results_root)


def run_block(
    specs: list[RunSpec],
    results_root: str | Path = "results",
    n_workers: int = 6,
    sequential: bool = False,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Run a block of specs, in parallel (metrics) or sequentially (timing).

    Args:
        specs: Run specifications.
        results_root: Base results directory.
        n_workers: Pool size for the parallel path.
        sequential: True for timing-sensitive blocks (protocol E7): no pool,
            no thread capping, machine assumed otherwise idle.
        progress: Print one line per completed run.

    Returns:
        List of run summaries, in completion order.
    """
    results_root = str(results_root)
    summaries: list[dict[str, Any]] = []
    if sequential:
        for i, spec in enumerate(specs):
            summaries.append(run_single(spec, results_root))
            if progress:
                print(f"[{i + 1}/{len(specs)}] {spec.exp_id}/{spec.run_id} done")
        return summaries
    payloads = [(spec, results_root) for spec in specs]
    with get_context("spawn").Pool(n_workers) as pool:
        for i, summary in enumerate(pool.imap_unordered(_run_single_star, payloads)):
            summaries.append(summary)
            if progress:
                print(f"[{i + 1}/{len(specs)}] {summary['exp_id']}/{summary['run_id']} done")
    return summaries


# ------------------------------------------------------------------ aggregation


def episodes_to_threshold(
    probe_episodes: list[int],
    probe_rewards: list[float],
    threshold: float,
    sustain: int = 3,
) -> int | None:
    """First probe episode from which the reward stays >= threshold.

    The threshold must hold for ``sustain`` consecutive checkpoints; returns
    None when never reached (censored run).
    """
    count = 0
    start: int | None = None
    for episode, reward in zip(probe_episodes, probe_rewards, strict=True):
        if reward >= threshold:
            if count == 0:
                start = episode
            count += 1
            if count >= sustain:
                return start
        else:
            count = 0
            start = None
    return None


def aggregate_runs(
    results_root: str | Path,
    exp_id: str,
    r_star: float | None = None,
    threshold_fraction: float = 0.9,
) -> Any:
    """Collect all run summaries of an experiment into a DataFrame.

    Args:
        results_root: Base results directory.
        exp_id: Experiment identifier (e.g. ``"e2"``).
        r_star: Optimal reference reward; when given, adds an
            ``episodes_to_threshold`` column (None = censored).
        threshold_fraction: Fraction of R* defining convergence.

    Returns:
        pandas DataFrame, one row per run, written to
        ``results/aggregated/<exp_id>__runs.csv``.
    """
    import pandas as pd

    results_root = Path(results_root)
    rows: list[dict[str, Any]] = []
    for summary_path in sorted((results_root / "raw" / exp_id).glob("*/summary.json")):
        with open(summary_path) as handle:
            summary = json.load(handle)
        if r_star is not None:
            summary["episodes_to_threshold"] = episodes_to_threshold(
                summary.get("probe_episodes", []),
                summary.get("probe_rewards", []),
                threshold_fraction * r_star,
            )
            summary["converged"] = summary["episodes_to_threshold"] is not None
        summary.pop("probe_episodes", None)
        summary.pop("probe_rewards", None)
        rows.append(summary)
    frame = pd.DataFrame(rows)
    out_dir = results_root / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / f"{exp_id}__runs.csv", index=False)
    return frame


def write_optimized_yaml(config: Config, path: str | Path, note: str) -> None:
    """Persist a tuned configuration as ``configs/optimized.yaml``."""
    import yaml

    payload = config.to_dict()
    payload["mode"] = "time_limited"
    header = "# Optimized configuration (time-limited mode).\n" f"# {note}\n"
    with open(path, "w") as handle:
        handle.write(header)
        yaml.safe_dump(payload, handle, sort_keys=False)
