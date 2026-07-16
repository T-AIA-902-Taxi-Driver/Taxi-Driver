"""Figure generators for the benchmark report (all data-driven, PNG 300 dpi).

Every function takes already-loaded data plus an output path, so any figure
can be regenerated from ``results/raw`` CSVs without re-running experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless rendering, before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

from src.evaluation.evaluator import ACTION_NAMES

FloatArray = npt.NDArray[np.float64]
DPI = 300


def _save(fig: Figure, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def rolling_mean(values: list[float], window: int = 100) -> FloatArray:
    """Trailing rolling mean with a growing window at the start."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    cumsum = np.cumsum(arr)
    out = np.empty_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        out[i] = (cumsum[i] - (cumsum[lo - 1] if lo > 0 else 0.0)) / (i - lo + 1)
    return out


def learning_curves(
    series: dict[str, list[list[float]]],
    out_path: str | Path,
    window: int = 100,
    ylabel: str = "Récompense native par épisode",
    title: str = "Courbes d'apprentissage",
) -> Path:
    """Overlaid learning curves with inter-seed mean ± std band (F1/F2).

    Args:
        series: label -> list of per-seed episode series (equal lengths per
            label; lengths may differ across labels).
        out_path: PNG destination.
        window: Rolling-mean window.
        ylabel: Y-axis label.
        title: Figure title.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, runs in series.items():
        smoothed = np.stack([rolling_mean(run, window) for run in runs])
        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)
        episodes = np.arange(len(mean))
        (line,) = ax.plot(episodes, mean, label=label, linewidth=1.4)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.15, color=line.get_color())
    ax.set_xlabel("Épisode")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (moyenne glissante {window}, bande ± σ inter-seeds)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def probe_curves(
    series: dict[str, tuple[list[int], list[list[float]]]],
    out_path: str | Path,
    xlabel: str = "Épisode d'entraînement",
    ylabel: str = "Récompense greedy (sondes)",
    title: str = "Convergence mesurée par sondes greedy",
    threshold: float | None = None,
    log_x: bool = False,
) -> Path:
    """Greedy-probe curves (mean ± std across seeds), optional threshold line."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, (episodes, runs) in series.items():
        arr = np.stack([np.asarray(r, dtype=np.float64) for r in runs])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        (line,) = ax.plot(episodes, mean, label=label, linewidth=1.4)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.15, color=line.get_color())
    if threshold is not None:
        ax.axhline(threshold, linestyle="--", color="grey", linewidth=1)
        ax.annotate(
            f"seuil 0.9·R* = {threshold:.2f}",
            xy=(0.02, threshold),
            xycoords=("axes fraction", "data"),
            fontsize=8,
            color="grey",
            va="bottom",
        )
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def reward_boxplots(
    groups: dict[str, list[float]],
    out_path: str | Path,
    ylabel: str = "Récompense moyenne de test (par run)",
    title: str = "Distribution des performances finales",
) -> Path:
    """Boxplots with individual run points overlaid (F3)."""
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(groups)), 5))
    labels = list(groups)
    data = [groups[label] for label in labels]
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    rng = np.random.default_rng(0)
    for i, values in enumerate(data, start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, s=14, alpha=0.6, zorder=3)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, out_path)


def steps_barplot(
    means: dict[str, float],
    ci_halfwidths: dict[str, float],
    out_path: str | Path,
    title: str = "Nombre moyen de pas par épisode (IC 95 %)",
    log_scale: bool = False,
) -> Path:
    """Barplot of mean steps with CI error bars; log scale fits brute force (F4)."""
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(means)), 5))
    labels = list(means)
    values = [means[label] for label in labels]
    errors = [ci_halfwidths.get(label, 0.0) for label in labels]
    bars = ax.bar(labels, values, yerr=errors, capsize=4)
    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel("Pas par épisode" + (" (échelle log)" if log_scale else ""))
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, out_path)


def grid_search_heatmap(
    frame: Any,
    out_path: str | Path,
    value_column: str = "eval_mean_reward",
    row_param: str = "alpha",
    col_param: str = "gamma",
    title: str = "Grid search Q-Learning : récompense finale",
) -> Path:
    """α×γ heatmap of a grid-search aggregate (F5). ``frame``: runs DataFrame."""
    import seaborn as sns

    pivot = frame.pivot_table(
        index=row_param, columns=col_param, values=value_column, aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax)
    ax.set_title(f"{title} (moyenne sur les seeds)")
    ax.set_xlabel(f"γ ({col_param})")
    ax.set_ylabel(f"α ({row_param})")
    return _save(fig, out_path)


def sensitivity_curves(
    frame: Any,
    param: str,
    out_path: str | Path,
    value_column: str = "eval_mean_reward",
    group_column: str | None = "algorithm",
    title: str | None = None,
) -> Path:
    """Mean ± std of a metric versus one hyperparameter (F6)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = [None] if group_column is None else sorted(frame[group_column].unique())
    for group in groups:
        subset = frame if group is None else frame[frame[group_column] == group]
        agg = subset.groupby(param)[value_column].agg(["mean", "std"]).reset_index()
        label = str(group) if group is not None else value_column
        ax.errorbar(agg[param], agg["mean"], yerr=agg["std"].fillna(0.0), marker="o", label=label)
    ax.set_xlabel(param)
    ax.set_ylabel(value_column)
    ax.set_title(title or f"Sensibilité à {param}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def q_values_heatmap(
    q_table: FloatArray,
    decode_state: Any,
    out_path: str | Path,
    title: str = "Q-values maximales par case (passager à bord, destination B)",
    passenger_loc: int = 4,
    destination: int = 3,
) -> Path:
    """5×5 per-action heatmaps of Q-values for a fixed passenger/dest context (F11).

    Args:
        q_table: (500, 6) learned table.
        decode_state: Unused placeholder for API symmetry (encoding is done
            locally); kept to make the data source explicit at call sites.
        out_path: PNG destination.
        title: Figure title.
        passenger_loc: Passenger component of the visualised slice.
        destination: Destination component of the visualised slice.
    """
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for action in range(6):
        grid = np.zeros((5, 5))
        for row in range(5):
            for col in range(5):
                state = ((row * 5 + col) * 5 + passenger_loc) * 4 + destination
                grid[row, col] = q_table[state, action]
        ax = axes[action // 3][action % 3]
        image = ax.imshow(grid, cmap="coolwarm")
        ax.set_title(ACTION_NAMES[action], fontsize=10)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle(title)
    return _save(fig, out_path)


def sample_efficiency_plot(
    series: dict[str, tuple[list[float], list[list[float]]]],
    out_path: str | Path,
    xlabel: str = "Pas d'environnement cumulés",
    title: str = "Efficacité en échantillons (échelle log)",
) -> Path:
    """Greedy reward vs cumulative env steps, log x (F9, tabular vs DQN)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, (x_values, runs) in series.items():
        arr = np.stack([np.asarray(r, dtype=np.float64) for r in runs])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        (line,) = ax.plot(x_values, mean, label=label, linewidth=1.4)
        ax.fill_between(x_values, mean - std, mean + std, alpha=0.15, color=line.get_color())
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Récompense greedy (sondes)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    return _save(fig, out_path)


def trackmania_learning_curve(
    timesteps: list[float],
    values: list[float],
    out_path: str | Path,
    window: int = 20,
    ylabel: str = "Récompense par épisode",
    title: str = "TrackMania — apprentissage SAC",
) -> Path:
    """Single-run learning curve over cumulative env steps (F13/F14).

    The x axis is cumulative environment timesteps rather than the episode
    index: TrackMania episodes have highly variable lengths (crashes end them
    after a few steps, laps after hundreds), so the timestep axis is the one
    on which sample efficiency reads correctly.

    Args:
        timesteps: Cumulative env steps at the end of each episode.
        values: Per-episode metric (reward, length, ...), same length.
        out_path: PNG destination.
        window: Rolling-mean window (in episodes).
        ylabel: Y-axis label.
        title: Figure title.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.asarray(timesteps, dtype=np.float64)
    ax.scatter(x, values, s=6, alpha=0.25, color="tab:blue", label="épisodes")
    ax.plot(
        x,
        rolling_mean(values, window),
        color="tab:blue",
        linewidth=1.6,
        label=f"moyenne glissante ({window} épisodes)",
    )
    ax.set_xlabel("Pas d'environnement cumulés")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    return _save(fig, out_path)
