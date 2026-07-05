"""Statistical methodology for benchmark comparisons (protocol §5).

Unit of analysis: the *run* (one seed), never individual episodes —
comparing episode-level samples across configurations would be
pseudo-replication. Normality is screened with Shapiro-Wilk (low power at
n=10, hence both parametric and non-parametric results are reported when
they disagree); Welch's t-test is used for normal-looking samples,
Mann-Whitney U otherwise; families of hypotheses are corrected with
Holm-Bonferroni; effect sizes are Hedges g (bias-corrected Cohen's d) and
Cliff's delta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats as sps

from src.evaluation.metrics import mean_ci95


def hedges_g(a: list[float], b: list[float]) -> float:
    """Bias-corrected standardized mean difference (Hedges g).

    Positive values mean ``a`` is larger than ``b``.
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled == 0:
        return 0.0
    d = (float(np.mean(a)) - float(np.mean(b))) / pooled
    correction = 1.0 - 3.0 / (4.0 * (n_a + n_b) - 9.0)
    return float(d * correction)


def cliffs_delta(a: list[float], b: list[float]) -> float:
    """Cliff's delta in [-1, 1]: P(a > b) − P(a < b)."""
    if not a or not b:
        return float("nan")
    a_arr = np.asarray(a)[:, None]
    b_arr = np.asarray(b)[None, :]
    greater = float(np.sum(a_arr > b_arr))
    lesser = float(np.sum(a_arr < b_arr))
    return (greater - lesser) / (len(a) * len(b))


def effect_size_label(g: float) -> str:
    """Cohen's interpretation bands for |g|."""
    magnitude = abs(g)
    if magnitude < 0.2:
        return "négligeable"
    if magnitude < 0.5:
        return "faible"
    if magnitude < 0.8:
        return "moyen"
    return "fort"


def holm_correction(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (monotone, capped at 1)."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p_values[index])
        running_max = max(running_max, value)
        adjusted[index] = running_max
    return adjusted


@dataclass
class ComparisonResult:
    """One two-group comparison, ready for the report's T4 table."""

    label_a: str
    label_b: str
    metric: str
    mean_a: float
    std_a: float
    ci_a: tuple[float, float]
    n_a: int
    mean_b: float
    std_b: float
    ci_b: tuple[float, float]
    n_b: int
    normal_a: bool
    normal_b: bool
    test_name: str  # "welch" | "mann-whitney"
    statistic: float
    p_value: float
    p_holm: float | None
    hedges_g: float
    cliffs_delta: float
    secondary_test_name: str | None = None
    secondary_p_value: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def significant(self, alpha: float = 0.05) -> bool:
        return (self.p_holm if self.p_holm is not None else self.p_value) < alpha

    def format_fr(self) -> str:
        """Protocol §5.6 French report format."""
        holm_text = f", p_Holm={self.p_holm:.4f}" if self.p_holm is not None else ""
        return (
            f"{self.label_a} : {self.mean_a:.2f} ± {self.std_a:.2f} "
            f"(IC 95 % [{self.ci_a[0]:.2f} ; {self.ci_a[1]:.2f}], n={self.n_a}) vs "
            f"{self.label_b} : {self.mean_b:.2f} ± {self.std_b:.2f} "
            f"(IC 95 % [{self.ci_b[0]:.2f} ; {self.ci_b[1]:.2f}], n={self.n_b}) ; "
            f"{self.test_name}, stat={self.statistic:.3f}, p={self.p_value:.4f}{holm_text}, "
            f"g={self.hedges_g:.2f} (effet {effect_size_label(self.hedges_g)}), "
            f"δ={self.cliffs_delta:.2f}"
        )


def _is_normal(values: list[float], alpha: float = 0.05) -> bool:
    """Shapiro-Wilk screen; degenerate/constant samples are non-normal."""
    if len(values) < 3 or float(np.std(values)) == 0.0:
        return False
    return float(sps.shapiro(values).pvalue) >= alpha


def compare_groups(
    a: list[float],
    b: list[float],
    label_a: str,
    label_b: str,
    metric: str = "reward",
) -> ComparisonResult:
    """Compare two run-level samples per the protocol's decision tree.

    Both samples normal (Shapiro-Wilk) → Welch t-test, otherwise
    Mann-Whitney U. When the two tests disagree on significance at α=0.05,
    the other test's p-value is attached as ``secondary_*`` so the report can
    surface the ambiguity instead of hiding it.
    """
    normal_a, normal_b = _is_normal(a), _is_normal(b)
    welch = sps.ttest_ind(a, b, equal_var=False)
    mannwhitney = sps.mannwhitneyu(a, b, alternative="two-sided")
    if normal_a and normal_b:
        test_name, statistic, p_value = "welch", float(welch.statistic), float(welch.pvalue)
        other_name, other_p = "mann-whitney", float(mannwhitney.pvalue)
    else:
        test_name, statistic, p_value = (
            "mann-whitney",
            float(mannwhitney.statistic),
            float(mannwhitney.pvalue),
        )
        other_name, other_p = "welch", float(welch.pvalue)
    disagree = (p_value < 0.05) != (other_p < 0.05)
    return ComparisonResult(
        label_a=label_a,
        label_b=label_b,
        metric=metric,
        mean_a=float(np.mean(a)),
        std_a=float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        ci_a=mean_ci95(a),
        n_a=len(a),
        mean_b=float(np.mean(b)),
        std_b=float(np.std(b, ddof=1)) if len(b) > 1 else 0.0,
        ci_b=mean_ci95(b),
        n_b=len(b),
        normal_a=normal_a,
        normal_b=normal_b,
        test_name=test_name,
        statistic=statistic,
        p_value=p_value,
        p_holm=None,
        hedges_g=hedges_g(a, b),
        cliffs_delta=cliffs_delta(a, b),
        secondary_test_name=other_name if disagree else None,
        secondary_p_value=other_p if disagree else None,
    )


def apply_holm(results: list[ComparisonResult]) -> list[ComparisonResult]:
    """Attach Holm-adjusted p-values to a family of comparisons (in place)."""
    adjusted = holm_correction([r.p_value for r in results])
    for result, p_holm in zip(results, adjusted, strict=True):
        result.p_holm = p_holm
    return results
