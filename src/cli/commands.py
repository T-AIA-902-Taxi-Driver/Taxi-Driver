"""Subcommand implementations: train (user / time-limited), eval, play."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.agents import create_agent
from src.cli.prompts import prompt_choice, prompt_confirm, prompt_float, prompt_int
from src.config import VALID_ALGORITHMS, VALID_EXPLORATIONS, Config
from src.environments import create_env
from src.evaluation.evaluator import Evaluator
from src.training.callbacks import (
    Callback,
    EarlyStoppingCallback,
    LoggingCallback,
    TimeBudgetCallback,
)
from src.training.trainer import Trainer
from src.utils.seeding import eval_seeds

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
OPTIMIZED_CONFIG_PATH = Path("configs/optimized.yaml")
# Fraction of the time budget reserved for training; the remainder covers
# evaluation and reporting (a 100-episode greedy tabular eval runs < 1 s).
TRAIN_BUDGET_FRACTION = 0.90


def _interactive(args: argparse.Namespace) -> bool:
    return sys.stdin.isatty() and not getattr(args, "non_interactive", False)


def _load_base_config(args: argparse.Namespace) -> Config:
    """Resolve the base config file according to mode and --config."""
    if getattr(args, "config", None):
        return Config.from_yaml(args.config)
    mode = getattr(args, "mode", None) or "user"
    if mode.replace("-", "_") == "time_limited":
        if OPTIMIZED_CONFIG_PATH.exists():
            return Config.from_yaml(OPTIMIZED_CONFIG_PATH)
        print(
            f"warning: {OPTIMIZED_CONFIG_PATH} not found, falling back to defaults "
            "(run the benchmark grid search to generate it)"
        )
        return Config()
    if DEFAULT_CONFIG_PATH.exists():
        return Config.from_yaml(DEFAULT_CONFIG_PATH)
    return Config()


def _prompt_user_mode(config: Config, input_fn: Callable[[str], str]) -> Config:
    """User mode: let the user tune the algorithm and its hyperparameters."""
    algorithm = prompt_choice("Agent", VALID_ALGORITHMS, config.algorithm, input_fn)
    updates: dict[str, object] = {"algorithm": algorithm}
    if algorithm != "brute_force":
        updates["alpha"] = prompt_float("alpha (learning rate)", config.alpha, 0.001, 1.0, input_fn)
        updates["gamma"] = prompt_float("gamma (discount)", config.gamma, 0.01, 1.0, input_fn)
        exploration = prompt_choice("exploration", VALID_EXPLORATIONS, config.exploration, input_fn)
        updates["exploration"] = exploration
        if exploration == "epsilon_greedy":
            updates["epsilon"] = prompt_float("epsilon initial", config.epsilon, 0.0, 1.0, input_fn)
            updates["epsilon_min"] = prompt_float(
                "epsilon min", config.epsilon_min, 0.0, 1.0, input_fn
            )
            updates["epsilon_decay"] = prompt_float(
                "epsilon decay (per episode)", config.epsilon_decay, 0.0, None, input_fn
            )
        elif exploration == "boltzmann":
            updates["temperature"] = prompt_float(
                "temperature initial", config.temperature, 0.001, None, input_fn
            )
        else:  # ucb
            updates["ucb_c"] = prompt_float("UCB c", config.ucb_c, 0.0, None, input_fn)
    import dataclasses

    return dataclasses.replace(config, **updates)  # type: ignore[arg-type]


def _print_recap(config: Config, time_limited: bool) -> None:
    print("\n--- Configuration ---")
    print(f"agent            : {config.algorithm}")
    print(f"environment      : {config.env}")
    print(f"train episodes   : {config.n_train_episodes}")
    print(f"test episodes    : {config.n_test_episodes}")
    if config.algorithm != "brute_force":
        print(f"alpha / gamma    : {config.alpha} / {config.gamma}")
        print(f"exploration      : {config.exploration}")
        if config.exploration == "epsilon_greedy":
            decay = (
                f"decay_frac={config.decay_frac}"
                if config.decay_frac is not None
                else f"decay={config.epsilon_decay} ({config.decay_type})"
            )
            print(f"epsilon          : {config.epsilon} -> {config.epsilon_min} ({decay})")
    if time_limited:
        print(
            f"time budget      : {config.time_budget:.0f}s (training capped at "
            f"{TRAIN_BUDGET_FRACTION * config.time_budget:.0f}s)"
        )
    print(f"seed             : {config.seed}")
    print("---------------------\n")


def cmd_train(args: argparse.Namespace) -> int:
    """Train + evaluate an agent (subject modes: user / time-limited)."""
    config = _load_base_config(args)
    mode = (getattr(args, "mode", None) or config.mode).replace("-", "_")
    args.mode = mode
    config = config.merge_cli_args(args)
    interactive = _interactive(args)

    if interactive:
        input_fn: Callable[[str], str] = input
        if mode == "user" and args.algorithm is None:
            config = _prompt_user_mode(config, input_fn)
        # Subject requirement: train AND test episode counts entered at launch
        # (in both modes); CLI arguments take precedence over prompts.
        import dataclasses

        updates: dict[str, int] = {}
        if args.n_train_episodes is None:
            label = (
                "Max training episodes (time budget may stop earlier)"
                if mode == "time_limited"
                else "Training episodes"
            )
            updates["n_train_episodes"] = prompt_int(label, config.n_train_episodes, 1, input_fn)
        if args.n_test_episodes is None:
            updates["n_test_episodes"] = prompt_int(
                "Test episodes", config.n_test_episodes, 1, input_fn
            )
        if updates:
            config = dataclasses.replace(config, **updates)  # type: ignore[arg-type]

    time_limited = mode == "time_limited"
    _print_recap(config, time_limited)
    if interactive and not prompt_confirm("Proceed?", True):
        print("aborted")
        return 1

    env = create_env(config)
    probe_env = create_env(config) if config.eval_interval > 0 else None
    agent = create_agent(config, env, n_episodes=config.n_train_episodes)

    callbacks: list[Callback] = [LoggingCallback(every=max(1, config.n_train_episodes // 10))]
    n_episodes = config.n_train_episodes
    start = time.monotonic()
    if time_limited:
        callbacks.append(
            TimeBudgetCallback(deadline=start + TRAIN_BUDGET_FRACTION * config.time_budget)
        )
        callbacks.append(EarlyStoppingCallback(target_reward=config.target_reward, window=100))
    elif config.early_stopping:
        callbacks.append(
            EarlyStoppingCallback(target_reward=config.target_reward, window=config.patience)
        )

    print(f"training {config.algorithm} for up to {n_episodes} episodes...")
    history = Trainer(agent, env, config, callbacks=callbacks, probe_env=probe_env).train(
        n_episodes
    )
    train_time = time.monotonic() - start
    print(
        f"\ntraining done: {len(history.rewards)} episodes in {history.wall_time:.1f}s "
        f"({history.stop_reason})"
    )

    # Torch checkpoints for DQN, compressed numpy archives for tabular agents.
    suffix = ".pt" if config.algorithm == "dqn" else ".npz"
    model_path = Path(
        args.save
        if getattr(args, "save", None)
        else Path(config.model_path)
        / f"{config.algorithm}_{datetime.datetime.now():%Y%m%dT%H%M%S}{suffix}"
    )
    agent.save(model_path)
    print(f"model saved to {model_path}")

    print(f"\nevaluating on {config.n_test_episodes} episodes (greedy policy)...")
    evaluator = Evaluator(env, seeds=eval_seeds(config.seed, config.n_test_episodes))
    results = evaluator.evaluate(agent, config.n_test_episodes)
    print(results.summary())
    if time_limited:
        total = time.monotonic() - start
        print(
            f"\ntime budget      : {config.time_budget:.0f}s | training {train_time:.1f}s "
            f"| total {total:.1f}s"
        )

    show = args.show_episodes if getattr(args, "show_episodes", None) is not None else 3
    if show > 0:
        evaluator.display_episodes(agent, k=show, rng=np.random.default_rng(config.seed))
    return 0


def _load_model_metadata(path: Path) -> dict[str, object]:
    """Read the metadata dict embedded in a saved model (.npz or .pt)."""
    if path.suffix == ".pt":
        # DQN torch checkpoint. Lazy import: the tabular path never pays torch.
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        metadata = checkpoint.get("metadata") if isinstance(checkpoint, dict) else None
        if not isinstance(metadata, dict):
            raise ValueError(f"{path} does not contain model metadata")
        return dict(metadata)
    data = np.load(path, allow_pickle=False)
    if "metadata" not in data:
        raise ValueError(f"{path} does not contain model metadata")
    return dict(json.loads(str(data["metadata"])))


def _rebuild_agent_from_model(path: Path, seed_override: int | None):  # type: ignore[no-untyped-def]
    """Rebuild env + agent from a saved model's embedded metadata."""
    metadata = _load_model_metadata(path)
    config_dict = metadata.get("config")
    if not isinstance(config_dict, dict):
        raise ValueError(f"{path} metadata is missing the training config")
    config = Config(**config_dict)
    if seed_override is not None:
        import dataclasses

        config = dataclasses.replace(config, seed=seed_override)
    env = create_env(config)
    agent = create_agent(config, env)
    agent.load(path)
    return config, env, agent


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate a saved model (algorithm auto-detected from metadata)."""
    config, env, agent = _rebuild_agent_from_model(Path(args.model), args.seed)
    n_episodes = args.n_test_episodes or config.n_test_episodes
    print(f"evaluating {agent.name} on {n_episodes} episodes (greedy policy)...")
    evaluator = Evaluator(env, seeds=eval_seeds(config.seed, n_episodes))
    results = evaluator.evaluate(agent, n_episodes)
    print(results.summary())
    show = args.show_episodes if args.show_episodes is not None else 0
    if show > 0:
        evaluator.display_episodes(agent, k=show, rng=np.random.default_rng(config.seed))
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    """Step-by-step replay of a saved model on random episodes."""
    config, env, agent = _rebuild_agent_from_model(Path(args.model), args.seed)
    rng = np.random.default_rng(config.seed if args.seed is None else args.seed)
    Evaluator(env).display_episodes(agent, k=args.episodes, rng=rng, delay=args.delay)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Hyperparameter sweep defined by a YAML file (grid search)."""
    import dataclasses
    import itertools

    import yaml

    from src.benchmarking.benchmarker import (
        RunSpec,
        aggregate_runs,
        run_block,
        write_optimized_yaml,
    )

    with open(args.sweep_config) as handle:
        sweep_config = yaml.safe_load(handle)
    exp_id = sweep_config.get("exp_id", "sweep")
    base = Config(**sweep_config.get("base", {}))
    algorithms: list[str] = sweep_config.get("algorithms", ["q_learning"])
    grid: dict[str, list[object]] = sweep_config.get("sweep", {})
    n_seeds: int = int(sweep_config.get("n_seeds", 10))
    seed_base: int = int(sweep_config.get("seed_base", 42))

    specs: list[RunSpec] = []
    param_names = list(grid)
    combos = list(itertools.product(*(grid[name] for name in param_names))) or [()]
    for algorithm in algorithms:
        for combo in combos:
            overrides = dict(zip(param_names, combo, strict=True))
            config = dataclasses.replace(base, algorithm=algorithm, **overrides)  # type: ignore[arg-type]
            specs.extend(
                RunSpec(exp_id=exp_id, config=config, seed=seed_base + k) for k in range(n_seeds)
            )
    print(
        f"sweep {exp_id}: {len(algorithms)} algo(s) x {len(combos)} combo(s) x "
        f"{n_seeds} seed(s) = {len(specs)} runs on {args.workers} workers"
    )
    run_block(specs, results_root=args.out, n_workers=args.workers)

    frame = aggregate_runs(args.out, exp_id, r_star=args.r_star)
    group_cols = ["algorithm", "config_hash", *param_names]
    ranking = (
        frame.groupby(group_cols, dropna=False)["eval_mean_reward"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    print("\ntop 5 configurations (mean test reward across seeds):")
    print(ranking.head(5).to_string(index=False))

    if args.write_optimized:
        best = ranking.iloc[0]
        best_overrides = {name: best[name] for name in param_names if name in ranking.columns}
        best_overrides["algorithm"] = str(best["algorithm"])
        best_config = dataclasses.replace(base, **best_overrides)
        write_optimized_yaml(
            best_config,
            OPTIMIZED_CONFIG_PATH,
            note=(
                f"Grid-search winner of '{exp_id}' "
                f"(mean test reward {best['mean']:.3f} over {int(best['count'])} seeds)."
            ),
        )
        print(f"\noptimized configuration written to {OPTIMIZED_CONFIG_PATH}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Head-to-head agent comparison under identical conditions."""
    import dataclasses

    from src.benchmarking.benchmarker import RunSpec, aggregate_runs, run_block
    from src.benchmarking.stats import apply_holm, compare_groups

    algorithms = [name.strip() for name in args.agents.split(",") if name.strip()]
    base = Config()
    if args.n_train_episodes is not None:
        base = dataclasses.replace(base, n_train_episodes=args.n_train_episodes)
    if args.n_test_episodes is not None:
        base = dataclasses.replace(base, n_test_episodes=args.n_test_episodes)

    specs = [
        RunSpec(
            exp_id="compare",
            config=dataclasses.replace(base, algorithm=algorithm),
            seed=base.seed + k,
        )
        for algorithm in algorithms
        for k in range(args.n_seeds)
    ]
    print(f"compare: {len(algorithms)} agents x {args.n_seeds} seeds = {len(specs)} runs")
    run_block(specs, results_root=args.out, n_workers=args.workers)

    frame = aggregate_runs(args.out, "compare")
    frame = frame[frame["algorithm"].isin(algorithms)]
    table = (
        frame.groupby("algorithm")
        .agg(
            reward=("eval_mean_reward", "mean"),
            reward_std=("eval_mean_reward", "std"),
            steps=("eval_mean_steps", "mean"),
            success=("eval_success_rate", "mean"),
            train_s=("train_wall_time_s", "mean"),
            memory_kb=("memory_bytes", lambda b: b.mean() / 1024),
        )
        .sort_values("reward", ascending=False)
    )
    print("\ncomparison (means over seeds, identical eval episodes):")
    print(table.round(3).to_string())

    reference = "q_learning" if "q_learning" in algorithms else algorithms[0]
    reference_rewards = frame[frame["algorithm"] == reference]["eval_mean_reward"].tolist()
    comparisons = [
        compare_groups(
            frame[frame["algorithm"] == algorithm]["eval_mean_reward"].tolist(),
            reference_rewards,
            algorithm,
            reference,
        )
        for algorithm in algorithms
        if algorithm != reference
    ]
    if comparisons:
        apply_holm(comparisons)
        print(f"\npairwise tests vs {reference} (Holm-corrected):")
        for comparison in comparisons:
            print("  " + comparison.format_fr())
    return 0
