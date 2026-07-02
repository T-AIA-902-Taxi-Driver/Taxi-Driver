"""Generate the report figures (F1-F12) from campaign data in results/.

Reads ONLY results/raw + results/aggregated (+ r_star.json); never re-runs
experiments. Any figure can therefore be regenerated at will:

    python scripts/make_figures.py            # everything available
    python scripts/make_figures.py --figures f1,f5

Missing experiment blocks are skipped with a notice (so the script works on
partial campaigns).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.evaluation.metrics import mean_ci95
from src.visualization import plots

ALGO_LABELS = {
    "brute_force": "BruteForce",
    "q_learning": "Q-Learning",
    "sarsa": "SARSA",
    "expected_sarsa": "Expected SARSA",
    "double_q_learning": "Double Q-Learning",
    "monte_carlo": "Monte Carlo",
    "dqn": "DQN",
}


def _runs_frame(results: Path, exp_id: str) -> pd.DataFrame | None:
    path = results / "aggregated" / f"{exp_id}__runs.csv"
    if not path.exists():
        print(f"  [skip] {path} missing")
        return None
    return pd.read_csv(path)


def _run_dirs(results: Path, exp_id: str) -> list[Path]:
    exp = results / "raw" / exp_id
    return sorted(d for d in exp.glob("*") if (d / "summary.json").exists()) if exp.exists() else []


def _train_series(run_dir: Path, column: str) -> list[float]:
    with open(run_dir / "train_episodes.csv") as handle:
        return [float(row[column]) for row in csv.DictReader(handle)]


def _probe_series(run_dir: Path) -> tuple[list[int], list[float], list[float]]:
    episodes: list[int] = []
    rewards: list[float] = []
    max_qs: list[float] = []
    with open(run_dir / "probes.csv") as handle:
        for row in csv.DictReader(handle):
            episodes.append(int(row["episode"]))
            rewards.append(float(row["probe_mean_reward"]))
            max_qs.append(float(row.get("probe_max_q", "nan")))
    return episodes, rewards, max_qs


def _algo_of(run_dir: Path) -> str:
    return run_dir.name.split("__")[0]


def _grouped_train_series(
    results: Path, exp_id: str, column: str, algos: list[str] | None = None
) -> dict[str, list[list[float]]]:
    series: dict[str, list[list[float]]] = {}
    for run_dir in _run_dirs(results, exp_id):
        algo = _algo_of(run_dir)
        if algos and algo not in algos:
            continue
        series.setdefault(ALGO_LABELS.get(algo, algo), []).append(_train_series(run_dir, column))
    return series


def _r_star(results: Path) -> float | None:
    path = results / "r_star.json"
    if not path.exists():
        return None
    with open(path) as handle:
        return float(json.load(handle)["r_star_probe"])


def fig_f1_f2(results: Path, out: Path) -> None:
    series_r = _grouped_train_series(results, "e2", "reward")
    if not series_r:
        return
    plots.learning_curves(series_r, out / "F1_courbes_apprentissage.png")
    series_s = _grouped_train_series(results, "e2", "steps")
    plots.learning_curves(
        series_s,
        out / "F2_courbes_steps.png",
        ylabel="Pas par épisode",
        title="Nombre de pas par épisode",
    )


def fig_f3_f4(results: Path, out: Path) -> None:
    frame_e2 = _runs_frame(results, "e2")
    if frame_e2 is None:
        return
    groups = {
        ALGO_LABELS.get(algo, algo): subset["eval_mean_reward"].tolist()
        for algo, subset in frame_e2.groupby("algorithm")
    }
    plots.reward_boxplots(groups, out / "F3_boxplots_rewards.png")

    frames = [frame_e2]
    frame_e0 = _runs_frame(results, "e0")
    if frame_e0 is not None:
        frames.append(frame_e0[frame_e0["algorithm"] == "brute_force"])
    merged = pd.concat(frames)
    means: dict[str, float] = {}
    cis: dict[str, float] = {}
    for algo, subset in merged.groupby("algorithm"):
        values = subset["eval_mean_steps"].tolist()
        label = ALGO_LABELS.get(str(algo), str(algo))
        means[label] = float(np.mean(values))
        low, high = mean_ci95(values)
        cis[label] = (high - low) / 2
    plots.steps_barplot(means, cis, out / "F4_barplot_steps.png", log_scale=True)


def fig_f5_f6(results: Path, out: Path) -> None:
    frame = _runs_frame(results, "e1a")
    if frame is None:
        return
    plots.grid_search_heatmap(frame, out / "F5_heatmap_grid.png")
    plots.grid_search_heatmap(
        frame,
        out / "F5b_heatmap_convergence.png",
        value_column="episodes_to_threshold",
        title="Grid search Q-Learning : épisodes jusqu'au seuil",
    )
    frame_all = frame.assign(algorithm="q_learning")
    frame_e1b = _runs_frame(results, "e1b")
    if frame_e1b is not None:
        frame_all = pd.concat([frame_all, frame_e1b])
    plots.sensitivity_curves(
        frame_all[frame_all["gamma"] == 0.99],
        "alpha",
        out / "F6a_sensibilite_alpha.png",
        title="Sensibilité à α (γ=0.99)",
    )
    plots.sensitivity_curves(
        frame_all[frame_all["alpha"] == 0.15],
        "gamma",
        out / "F6b_sensibilite_gamma.png",
        title="Sensibilité à γ (α=0.15)",
    )


def _probe_group(
    results: Path, exp_id: str, label_fn: object
) -> dict[str, tuple[list[int], list[list[float]]]]:
    grouped: dict[str, tuple[list[int], list[list[float]]]] = {}
    for run_dir in _run_dirs(results, exp_id):
        with open(run_dir / "config.json") as handle:
            config = json.load(handle)["config"]
        label = label_fn(config)  # type: ignore[operator]
        episodes, rewards, _ = _probe_series(run_dir)
        if label not in grouped:
            grouped[label] = (episodes, [])
        if len(rewards) == len(grouped[label][0]):
            grouped[label][1].append(rewards)
    return grouped


def fig_f7_f8(results: Path, out: Path) -> None:
    threshold = _r_star(results)
    threshold_09 = 0.9 * threshold if threshold is not None else None

    def explo_label(config: dict) -> str:
        if config["exploration"] == "epsilon_greedy":
            return f"ε-greedy ({config['decay_type']})"
        return str(config["exploration"])

    grouped = _probe_group(results, "e3", explo_label)
    if grouped:
        plots.probe_curves(
            grouped,
            out / "F7_exploration.png",
            title="Stratégies d'exploration (sondes greedy)",
            threshold=threshold_09,
        )
    grouped = _probe_group(results, "e4", lambda c: str(c["reward_shaping"]))
    if grouped:
        plots.probe_curves(
            grouped,
            out / "F8_reward_shaping.png",
            title="Reward shaping — récompense NATIVE (sondes greedy)",
            threshold=threshold_09,
        )


def fig_f9(results: Path, out: Path) -> None:
    """Sample efficiency: greedy probe reward vs cumulative env steps."""
    series: dict[str, tuple[list[float], list[list[float]]]] = {}
    for run_dir in _run_dirs(results, "e2"):
        algo = _algo_of(run_dir)
        if algo not in ("q_learning", "dqn"):
            continue
        episodes, rewards, _ = _probe_series(run_dir)
        steps = _train_series(run_dir, "steps")
        cum = np.cumsum(steps)
        x = [float(cum[min(e, len(cum)) - 1]) for e in episodes]
        label = ALGO_LABELS[algo]
        if label not in series:
            series[label] = (x, [])
        if len(rewards) == len(series[label][0]):
            series[label][1].append(rewards)
    if series:
        plots.sample_efficiency_plot(series, out / "F9_efficacite_echantillon.png")


def fig_f10(results: Path, out: Path) -> None:
    """Overestimation: probe max-Q trajectories QL vs Double QL vs realized."""
    grouped: dict[str, tuple[list[int], list[list[float]]]] = {}
    realized: dict[str, tuple[list[int], list[list[float]]]] = {}
    for run_dir in _run_dirs(results, "e2"):
        algo = _algo_of(run_dir)
        if algo not in ("q_learning", "double_q_learning"):
            continue
        episodes, rewards, max_qs = _probe_series(run_dir)
        label = ALGO_LABELS[algo]
        grouped.setdefault(label + " (max Q)", (episodes, []))[1].append(max_qs)
        realized.setdefault(label + " (retour réalisé)", (episodes, []))[1].append(rewards)
    if grouped:
        plots.probe_curves(
            {**grouped, **realized},
            out / "F10_surestimation.png",
            ylabel="Valeur",
            title="Biais de surestimation : max Q estimé vs retour réalisé",
        )


def fig_f11(results: Path, out: Path) -> None:
    """Q-value heatmaps + GIF from the best e2 Q-Learning run."""
    candidates = [d for d in _run_dirs(results, "e2") if _algo_of(d) == "q_learning"]
    if not candidates:
        return

    def _eval_reward(run_dir: Path) -> float:
        with open(run_dir / "summary.json") as handle:
            return float(json.load(handle)["eval_mean_reward"])

    best = max(candidates, key=_eval_reward)
    model = np.load(best / "model.npz")
    plots.q_values_heatmap(model["q_table"], None, out / "F11_heatmap_qvalues.png")

    from src.agents import create_agent
    from src.config import Config
    from src.environments import create_env
    from src.visualization.episode_replay import save_episode_gif

    config = Config(algorithm="q_learning")
    agent = create_agent(config, create_env(config))
    agent.load(best / "model.npz")
    save_episode_gif(agent, out / "F11b_episode.gif", seed=10042)


def fig_f12(results: Path, out: Path) -> None:
    grouped = _probe_group(results, "e6", lambda c: ALGO_LABELS[str(c["algorithm"])])
    if grouped:
        plots.probe_curves(
            grouped,
            out / "F12_multi_passagers.png",
            title="Multi-passagers (14 400 états) : convergence",
            log_x=True,
        )


FIGURES = {
    "f1": fig_f1_f2,
    "f3": fig_f3_f4,
    "f5": fig_f5_f6,
    "f7": fig_f7_f8,
    "f9": fig_f9,
    "f10": fig_f10,
    "f11": fig_f11,
    "f12": fig_f12,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/figures")
    parser.add_argument("--figures", default=",".join(FIGURES))
    args = parser.parse_args()

    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in (f.strip() for f in args.figures.split(",") if f.strip()):
        if name not in FIGURES:
            parser.error(f"unknown figure group: {name} (choose from {sorted(FIGURES)})")
        print(f"generating {name}...")
        FIGURES[name](results, out)
    print(f"figures written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
