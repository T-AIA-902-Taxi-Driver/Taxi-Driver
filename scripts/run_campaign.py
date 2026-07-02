"""Benchmark campaign driver: experiment blocks E0-E7 (docs/PROTOCOLE.md).

Blocks E0-E6 run in parallel across processes (their metrics are unaffected
by parallelism); E7 re-runs the head-to-head configurations SEQUENTIALLY and
is the only legitimate source of wall-clock timing and memory columns in the
report. Runs are idempotent — re-launching the script resumes after a crash.

Usage:
    python scripts/run_campaign.py --blocks e0,e1a,e2 --workers 6
    python scripts/run_campaign.py            # everything, ~4h30-5h30
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmarking.benchmarker import (
    EVAL_BASE_SEED,
    RunSpec,
    aggregate_runs,
    run_block,
    write_optimized_yaml,
)
from src.config import Config

N_SEEDS = 10
SEEDS = list(range(42, 42 + N_SEEDS))
TABULAR_EPISODES = 15_000
DQN_EPISODES = 5_000
MULTI_EPISODES = 150_000

# Reference tabular configuration (protocol §0): probes every 100 episodes,
# epsilon reaching its floor at 60% of the horizon, early stopping disabled.
REFERENCE = Config(
    algorithm="q_learning",
    alpha=0.15,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    decay_type="exp",
    decay_frac=0.6,
    n_train_episodes=TABULAR_EPISODES,
    n_test_episodes=100,
    eval_interval=100,
    n_probe_episodes=30,
    early_stopping=False,
)

TABULAR_ALGOS = (
    "q_learning",
    "sarsa",
    "expected_sarsa",
    "double_q_learning",
    "monte_carlo",
)

DQN_REFERENCE = dataclasses.replace(
    REFERENCE,
    algorithm="dqn",
    n_train_episodes=DQN_EPISODES,
    lr=1e-3,
    hidden_size=128,
    device="auto",
)


def _ref(**overrides: object) -> Config:
    return dataclasses.replace(REFERENCE, **overrides)  # type: ignore[arg-type]


def specs_e0() -> list[RunSpec]:
    """Brute-force baseline; the 2000-step variant removes the 200-step
    truncation artifact behind the subject's ~350-step figure."""
    capped = _ref(algorithm="brute_force", n_train_episodes=0, n_test_episodes=1000)
    uncapped = dataclasses.replace(capped, max_steps_per_episode=2000)
    return [RunSpec("e0", config, seed) for config in (capped, uncapped) for seed in SEEDS]


def specs_e1a() -> list[RunSpec]:
    """Q-Learning grid: 5 alpha x 4 gamma x 10 seeds = 200 runs (H1, H3, F5)."""
    return [
        RunSpec("e1a", _ref(alpha=alpha, gamma=gamma), seed)
        for alpha in (0.05, 0.1, 0.15, 0.2, 0.3)
        for gamma in (0.9, 0.95, 0.99, 0.999)
        for seed in SEEDS
    ]


def specs_e1b() -> list[RunSpec]:
    """Cross-algorithm grid: 4 algos x 3 alpha x 3 gamma x 10 seeds (H3)."""
    return [
        RunSpec("e1b", _ref(algorithm=algorithm, alpha=alpha, gamma=gamma), seed)
        for algorithm in ("sarsa", "expected_sarsa", "double_q_learning", "monte_carlo")
        for alpha in (0.05, 0.15, 0.3)
        for gamma in (0.9, 0.99, 0.999)
        for seed in SEEDS
    ]


def specs_e2() -> list[RunSpec]:
    """Head-to-head at the reference configuration: 6 tabular + DQN (T1, H2/6/7/8)."""
    specs = [
        RunSpec("e2", _ref(algorithm=algorithm), seed)
        for algorithm in TABULAR_ALGOS
        for seed in SEEDS
    ]
    specs += [RunSpec("e2", DQN_REFERENCE, seed) for seed in SEEDS]
    return specs


def specs_e3() -> list[RunSpec]:
    """Exploration strategies on Q-Learning (H4)."""
    variants = [
        _ref(),  # epsilon-greedy, exponential decay (reference)
        _ref(decay_type="linear"),
        _ref(exploration="boltzmann"),
        _ref(exploration="ucb"),
    ]
    return [RunSpec("e3", config, seed) for config in variants for seed in SEEDS]


def specs_e4() -> list[RunSpec]:
    """Reward shaping variants on Q-Learning (H5)."""
    return [
        RunSpec("e4", _ref(reward_shaping=shaping), seed)
        for shaping in ("none", "potential", "naive_distance", "step_penalty")
        for seed in SEEDS
    ]


def specs_e5() -> list[RunSpec]:
    """DQN sensitivity: lr x hidden, 5 seeds (exploratory)."""
    return [
        RunSpec(
            "e5",
            dataclasses.replace(DQN_REFERENCE, lr=lr, hidden_size=hidden),
            seed,
        )
        for lr in (1e-4, 5e-4, 1e-3)
        for hidden in (64, 128)
        # protocol cut order applied: E5 is exploratory, 3 seeds (~12 min/run)
        for seed in SEEDS[:3]
    ]


def specs_e6() -> list[RunSpec]:
    """Multi-passenger scaling: QL and SARSA on the 14,400-state env (H9)."""
    return [
        RunSpec(
            "e6",
            _ref(
                algorithm=algorithm,
                env="multi",
                n_train_episodes=MULTI_EPISODES,
                eval_interval=1000,
                max_steps_per_episode=500,
            ),
            seed,
        )
        for algorithm in ("q_learning", "sarsa")
        for seed in SEEDS
    ]


def specs_e7() -> list[RunSpec]:
    """Timing block (STRICTLY SEQUENTIAL): tabular x 10 seeds, DQN cuda/cpu x 3."""
    specs = [
        RunSpec("e7", _ref(algorithm=algorithm), seed)
        for algorithm in ("brute_force", *TABULAR_ALGOS)
        for seed in SEEDS
    ]
    for device in ("cuda", "cpu"):
        config = dataclasses.replace(DQN_REFERENCE, device=device)
        # sequential DQN timing runs are expensive: 2 seeds per device
        # (time varies little across seeds; documented in the report)
        specs += [RunSpec("e7", config, seed) for seed in SEEDS[:2]]
    return specs


BLOCKS: dict[str, tuple[object, bool]] = {
    # name: (spec factory, sequential)
    "e0": (specs_e0, False),
    "e1a": (specs_e1a, False),
    "e1b": (specs_e1b, False),
    "e2": (specs_e2, False),
    "e3": (specs_e3, False),
    "e4": (specs_e4, False),
    "e5": (specs_e5, False),
    "e6": (specs_e6, False),
    "e7": (specs_e7, True),
}


def compute_r_star(results_root: Path) -> float:
    """Compute (or reload) R* on the PROBE seed set.

    Probe rewards are measured on the probe episodes, so the convergence
    threshold must reference the probe-set optimum — thresholding probes
    against the eval-set optimum (8.05 vs 7.77 here) censors runs whose
    optimal plateau sits between the two.
    """
    r_star_path = results_root / "r_star.json"
    if r_star_path.exists():
        with open(r_star_path) as handle:
            return float(json.load(handle)["r_star_probe"])
    from src.benchmarking.value_iteration import optimal_reference_reward
    from src.environments import create_env
    from src.utils.seeding import eval_seeds, probe_seeds

    env = create_env(Config())
    r_star_eval = optimal_reference_reward(env, eval_seeds(EVAL_BASE_SEED, 100))
    r_star_probe = optimal_reference_reward(env, probe_seeds(EVAL_BASE_SEED, 30))
    results_root.mkdir(parents=True, exist_ok=True)
    with open(r_star_path, "w") as handle:
        json.dump(
            {"r_star": r_star_eval, "r_star_probe": r_star_probe, "gamma": 0.99},
            handle,
            indent=2,
        )
    print(f"R* (eval set) = {r_star_eval:.3f} | R* (probe set) = {r_star_probe:.3f}")
    return r_star_probe


def maybe_write_optimized(results_root: Path, r_star: float) -> None:
    """Promote the E1a grid winner to configs/optimized.yaml.

    Ties on final reward (several configurations reach the optimal policy)
    are broken by convergence speed — the property the time-limited mode
    actually needs.
    """
    frame = aggregate_runs(results_root, "e1a", r_star=r_star)
    ranking = (
        frame.groupby(["alpha", "gamma"])
        .agg(
            mean=("eval_mean_reward", "mean"),
            count=("eval_mean_reward", "count"),
            ett=("episodes_to_threshold", "mean"),
        )
        .sort_values(["mean", "ett"], ascending=[False, True])
        .reset_index()
    )
    best = ranking.iloc[0]
    best_config = _ref(alpha=float(best["alpha"]), gamma=float(best["gamma"]))
    write_optimized_yaml(
        best_config,
        Path("configs/optimized.yaml"),
        note=(
            f"Grid-search winner of E1a: alpha={best['alpha']}, gamma={best['gamma']} "
            f"(mean test reward {best['mean']:.3f} over {int(best['count'])} seeds)."
        ),
    )
    print(
        f"optimized.yaml <- alpha={best['alpha']}, gamma={best['gamma']} "
        f"(reward {best['mean']:.3f})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", default=",".join(BLOCKS), help="comma-separated block ids")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--results", default="results")
    parser.add_argument("--dry-run", action="store_true", help="print run counts and exit")
    args = parser.parse_args()

    results_root = Path(args.results)
    selected = [block.strip() for block in args.blocks.split(",") if block.strip()]
    unknown = set(selected) - set(BLOCKS)
    if unknown:
        parser.error(f"unknown blocks: {sorted(unknown)}")

    all_specs = {name: BLOCKS[name][0]() for name in selected}  # type: ignore[operator]
    total = sum(len(specs) for specs in all_specs.values())
    for name, specs in all_specs.items():
        sequential = BLOCKS[name][1]
        print(f"  {name}: {len(specs)} runs{' (sequential)' if sequential else ''}")
    print(f"total: {total} runs")
    if args.dry_run:
        return 0

    r_star = compute_r_star(results_root)
    campaign_start = time.monotonic()
    for name in selected:
        specs = all_specs[name]
        sequential = BLOCKS[name][1]
        block_start = time.monotonic()
        print(f"\n=== block {name}: {len(specs)} runs ===")
        run_block(
            specs,
            results_root=results_root,
            n_workers=args.workers,
            sequential=sequential,
        )
        aggregate_runs(results_root, name, r_star=r_star)
        print(f"=== block {name} done in {(time.monotonic() - block_start) / 60:.1f} min ===")
        if name == "e1a":
            maybe_write_optimized(results_root, r_star)
    print(f"\ncampaign finished in {(time.monotonic() - campaign_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
