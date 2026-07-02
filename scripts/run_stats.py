"""Statistical tests for the protocol hypotheses (H1-H9) from results/.

Each hypothesis family reads the relevant aggregated CSVs, runs the protocol
decision tree (Shapiro → Welch/Mann-Whitney, Holm per family, effect sizes),
prints the French report lines, and writes one CSV per family to
``results/aggregated/stats_<family>.csv``. Families whose experiment blocks
are missing are skipped, so the script works on partial campaigns.

    python scripts/run_stats.py            # all available families
    python scripts/run_stats.py --families h1,h4,h5
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.benchmarking.stats import ComparisonResult, apply_holm, compare_groups

CENSOR_EPISODES = 15_000  # censored runs count as worst-rank at the horizon


def _load(results: Path, exp_id: str) -> pd.DataFrame | None:
    path = results / "aggregated" / f"{exp_id}__runs.csv"
    if not path.exists():
        print(f"  [skip] {path.name} missing")
        return None
    return pd.read_csv(path)


def _ett(frame: pd.DataFrame) -> list[float]:
    """episodes_to_threshold with censored runs mapped to the horizon."""
    return [
        float(v) if pd.notna(v) else float(CENSOR_EPISODES) for v in frame["episodes_to_threshold"]
    ]


def _rewards(frame: pd.DataFrame) -> list[float]:
    return [float(v) for v in frame["eval_mean_reward"]]


def _write(results: Path, family: str, comparisons: list[ComparisonResult]) -> None:
    if not comparisons:
        return
    apply_holm(comparisons)
    rows = [dataclasses.asdict(c) for c in comparisons]
    out = results / "aggregated" / f"stats_{family}.csv"
    pd.DataFrame(rows).drop(columns=["extras"]).to_csv(out, index=False)
    print(f"\n=== {family.upper()} ===")
    for comparison in comparisons:
        marker = "*" if comparison.significant() else " "
        print(f" {marker} [{comparison.metric}] {comparison.format_fr()}")


def family_h1(results: Path) -> None:
    """γ=0.99 vs γ=0.9 (α=0.15): slower convergence, better final policy?"""
    frame = _load(results, "e1a")
    if frame is None:
        return
    base = frame[np.isclose(frame["alpha"], 0.15)]
    high = base[np.isclose(base["gamma"], 0.99)]
    low = base[np.isclose(base["gamma"], 0.9)]
    comparisons = [
        compare_groups(_ett(high), _ett(low), "γ=0.99", "γ=0.9", metric="episodes_to_threshold"),
        compare_groups(_rewards(high), _rewards(low), "γ=0.99", "γ=0.9", metric="final_reward"),
    ]
    _write(results, "h1", comparisons)


def family_h2_h7_h8(results: Path) -> None:
    """Head-to-head families on E2: QL vs SARSA (H2), DoubleQL (H7), MC (H8)."""
    frame = _load(results, "e2")
    if frame is None:
        return
    by_algo = dict(iter(frame.groupby("algorithm")))

    def pick(name: str) -> pd.DataFrame | None:
        if name not in by_algo:
            print(f"  [skip] {name} absent from e2")
            return None
        return by_algo[name]

    ql, sarsa = pick("q_learning"), pick("sarsa")
    if ql is not None and sarsa is not None:
        comparisons = [
            compare_groups(
                _ett(ql), _ett(sarsa), "Q-Learning", "SARSA", metric="episodes_to_threshold"
            ),
            compare_groups(
                [float(v) for v in ql["post_convergence_std"]],
                [float(v) for v in sarsa["post_convergence_std"]],
                "Q-Learning",
                "SARSA",
                metric="post_convergence_std",
            ),
            compare_groups(
                _rewards(ql), _rewards(sarsa), "Q-Learning", "SARSA", metric="final_reward"
            ),
        ]
        _write(results, "h2", comparisons)

    dql = pick("double_q_learning")
    if ql is not None and dql is not None:
        comparisons = [
            compare_groups(
                _ett(ql), _ett(dql), "Q-Learning", "Double QL", metric="episodes_to_threshold"
            ),
            compare_groups(
                [float(v) for v in ql["post_convergence_std"]],
                [float(v) for v in dql["post_convergence_std"]],
                "Q-Learning",
                "Double QL",
                metric="post_convergence_std",
            ),
        ]
        _write(results, "h7", comparisons)

    mc = pick("monte_carlo")
    if ql is not None and mc is not None and sarsa is not None:
        censored = int(mc["episodes_to_threshold"].isna().sum())
        print(f"\n  MC censored runs: {censored}/{len(mc)}")
        comparisons = [
            compare_groups(
                _ett(mc), _ett(ql), "Monte Carlo", "Q-Learning", metric="episodes_to_threshold"
            ),
            compare_groups(
                _ett(mc), _ett(sarsa), "Monte Carlo", "SARSA", metric="episodes_to_threshold"
            ),
        ]
        _write(results, "h8", comparisons)


def family_h3(results: Path) -> None:
    """Descriptive: hyperparameter sensitivity vs algorithm choice."""
    e1a = _load(results, "e1a")
    e1b = _load(results, "e1b")
    if e1a is None or e1b is None:
        return
    frame = pd.concat([e1a.assign(algorithm="q_learning"), e1b])
    per_config = frame.groupby(["algorithm", "alpha", "gamma"])["eval_mean_reward"].mean()
    intra = per_config.groupby("algorithm").agg(["min", "max", "mean"])
    intra["range"] = intra["max"] - intra["min"]
    best_per_algo = per_config.groupby("algorithm").max()
    inter_range = float(best_per_algo.max() - best_per_algo.min())
    ett = frame.copy()
    ett["ett"] = [
        float(v) if pd.notna(v) else float(CENSOR_EPISODES) for v in ett["episodes_to_threshold"]
    ]
    ett_per_config = ett.groupby(["algorithm", "alpha", "gamma"])["ett"].mean()
    intra_ett = ett_per_config.groupby("algorithm").agg(["min", "max"])
    intra_ett["range"] = intra_ett["max"] - intra_ett["min"]
    summary = pd.DataFrame(
        {
            "intra_algo_reward_range": intra["range"],
            "intra_algo_ett_range": intra_ett["range"],
        }
    )
    summary.loc["INTER-ALGO (best configs)", "intra_algo_reward_range"] = inter_range
    out = results / "aggregated" / "stats_h3.csv"
    summary.to_csv(out)
    print("\n=== H3 (descriptif) ===")
    print(summary.round(3).to_string())
    print(
        f"  étendue inter-algos aux meilleures configs : {inter_range:.3f} ; "
        f"étendue intra-algo max : {intra['range'].max():.3f}"
    )


def family_h4(results: Path) -> None:
    """Exploration strategies vs the ε-greedy exponential reference."""
    frame = _load(results, "e3")
    if frame is None:
        return

    def label(row: pd.Series) -> str:
        if row["exploration"] == "epsilon_greedy":
            return f"ε-{row['decay_type']}"
        return str(row["exploration"])

    frame = frame.assign(variant=frame.apply(label, axis=1))
    reference = frame[frame["variant"] == "ε-exp"]
    comparisons = []
    for variant in ("ε-linear", "boltzmann", "ucb"):
        subset = frame[frame["variant"] == variant]
        if len(subset) == 0:
            continue
        comparisons.append(
            compare_groups(
                _ett(subset), _ett(reference), variant, "ε-exp", metric="episodes_to_threshold"
            )
        )
        comparisons.append(
            compare_groups(
                _rewards(subset), _rewards(reference), variant, "ε-exp", metric="final_reward"
            )
        )
        comparisons.append(
            compare_groups(
                [float(v) for v in subset["first_success_episode"]],
                [float(v) for v in reference["first_success_episode"]],
                variant,
                "ε-exp",
                metric="first_success_episode",
            )
        )
    _write(results, "h4", comparisons)


def family_h5(results: Path) -> None:
    """Reward shaping variants vs native (all metrics on NATIVE rewards)."""
    frame = _load(results, "e4")
    if frame is None:
        return
    by_shaping = dict(iter(frame.groupby("reward_shaping")))
    reference = by_shaping.get("none")
    if reference is None:
        return
    comparisons = []
    for variant in ("potential", "naive_distance", "step_penalty"):
        subset = by_shaping.get(variant)
        if subset is None:
            continue
        comparisons.append(
            compare_groups(
                _ett(subset), _ett(reference), variant, "native", metric="episodes_to_threshold"
            )
        )
        comparisons.append(
            compare_groups(
                _rewards(subset), _rewards(reference), variant, "native", metric="final_reward"
            )
        )
    _write(results, "h5", comparisons)


def family_h6(results: Path) -> None:
    """Tabular vs DQN: final reward + convergence; costs come from E7."""
    e2 = _load(results, "e2")
    if e2 is None or "dqn" not in set(e2["algorithm"]):
        print("  [skip] dqn absent from e2")
        return
    ql = e2[e2["algorithm"] == "q_learning"]
    dqn = e2[e2["algorithm"] == "dqn"]
    comparisons = [
        compare_groups(_rewards(ql), _rewards(dqn), "Q-Learning", "DQN", metric="final_reward"),
        compare_groups(
            [float(v) for v in ql["cum_env_steps"]],
            [float(v) for v in dqn["cum_env_steps"]],
            "Q-Learning",
            "DQN",
            metric="cum_env_steps",
        ),
    ]
    _write(results, "h6", comparisons)
    e7 = _load(results, "e7")
    if e7 is not None:
        table = (
            e7.groupby(["algorithm"])
            .agg(
                train_s_mean=("train_wall_time_s", "mean"),
                train_s_median=("train_wall_time_s", "median"),
                infer_ms=("eval_mean_episode_seconds", lambda s: 1000 * s.mean()),
                memory_kb=("memory_bytes", lambda b: b.mean() / 1024),
            )
            .round(3)
        )
        table.to_csv(results / "aggregated" / "stats_h6_costs.csv")
        print("\n  coûts (E7 séquentiel) :")
        print(table.to_string())


def family_h9(results: Path) -> None:
    """Multi-passenger scaling (exploratory)."""
    e6 = _load(results, "e6")
    e2 = _load(results, "e2")
    if e6 is None or e2 is None:
        return
    print("\n=== H9 (exploratoire) ===")
    for algo, subset in e6.groupby("algorithm"):
        rewards = subset["eval_mean_reward"]
        success = subset["eval_success_rate"]
        print(
            f"  {algo}: reward {rewards.mean():.2f} ± {rewards.std():.2f} ; "
            f"succès {100 * success.mean():.1f} % ; "
            f"steps {subset['eval_mean_steps'].mean():.1f} (n={len(subset)})"
        )
    summary = e6.groupby("algorithm")[
        ["eval_mean_reward", "eval_mean_steps", "eval_success_rate"]
    ].agg(["mean", "std"])
    summary.to_csv(results / "aggregated" / "stats_h9.csv")


FAMILIES = {
    "h1": family_h1,
    "h2": family_h2_h7_h8,  # also emits h7 and h8
    "h3": family_h3,
    "h4": family_h4,
    "h5": family_h5,
    "h6": family_h6,
    "h9": family_h9,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--families", default=",".join(FAMILIES))
    args = parser.parse_args()
    results = Path(args.results)
    for name in (f.strip() for f in args.families.split(",") if f.strip()):
        if name not in FAMILIES:
            parser.error(f"unknown family: {name}")
        FAMILIES[name](results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
