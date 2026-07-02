"""Statistics module tests against hand-computed and scipy references."""

import numpy as np
import pytest
from scipy import stats as sps

from src.benchmarking.stats import (
    apply_holm,
    cliffs_delta,
    compare_groups,
    effect_size_label,
    hedges_g,
    holm_correction,
)

RNG = np.random.default_rng(7)
NORMAL_A = list(RNG.normal(8.0, 0.4, size=10))
NORMAL_B = list(RNG.normal(7.2, 0.5, size=10))


class TestEffectSizes:
    def test_hedges_g_sign_and_magnitude(self) -> None:
        g = hedges_g(NORMAL_A, NORMAL_B)
        assert g > 0.8  # clearly separated samples → large effect
        assert hedges_g(NORMAL_B, NORMAL_A) == pytest.approx(-g)

    def test_hedges_g_zero_for_identical(self) -> None:
        assert hedges_g([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_cliffs_delta_hand_case(self) -> None:
        # a always greater: delta = +1 ; interleaved gives intermediate value
        assert cliffs_delta([5.0, 6.0], [1.0, 2.0]) == 1.0
        assert cliffs_delta([1.0, 2.0], [5.0, 6.0]) == -1.0
        # pairs: (1,2)<, (1,4)<, (3,2)>, (3,4)< -> (1-3)/4 = -0.5
        assert cliffs_delta([1.0, 3.0], [2.0, 4.0]) == pytest.approx(-0.5)

    def test_labels(self) -> None:
        assert effect_size_label(0.1) == "négligeable"
        assert effect_size_label(-0.9) == "fort"


class TestHolm:
    def test_holm_hand_case(self) -> None:
        # p = [0.01, 0.04, 0.03] sorted: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04→max(0.06)
        adjusted = holm_correction([0.01, 0.04, 0.03])
        assert adjusted[0] == pytest.approx(0.03)
        assert adjusted[2] == pytest.approx(0.06)
        assert adjusted[1] == pytest.approx(0.06)  # monotone enforcement

    def test_holm_caps_at_one(self) -> None:
        assert max(holm_correction([0.9, 0.8, 0.7])) == 1.0

    def test_apply_holm_attaches(self) -> None:
        results = [
            compare_groups(NORMAL_A, NORMAL_B, "a", "b"),
            compare_groups(NORMAL_A, NORMAL_A, "a", "a"),
        ]
        apply_holm(results)
        assert all(r.p_holm is not None for r in results)
        assert results[0].p_holm >= results[0].p_value


class TestCompareGroups:
    def test_welch_chosen_for_normal_samples(self) -> None:
        result = compare_groups(NORMAL_A, NORMAL_B, "A", "B")
        assert result.test_name == "welch"
        reference = sps.ttest_ind(NORMAL_A, NORMAL_B, equal_var=False)
        assert result.p_value == pytest.approx(float(reference.pvalue))
        assert isinstance(result.significant(), bool)

    def test_mannwhitney_chosen_for_non_normal(self) -> None:
        skewed = [0.1] * 8 + [50.0, 80.0]  # heavily skewed
        result = compare_groups(skewed, NORMAL_B, "skewed", "B")
        assert result.test_name == "mann-whitney"

    def test_constant_sample_is_non_normal(self) -> None:
        result = compare_groups([5.0] * 10, NORMAL_B, "const", "B")
        assert result.normal_a is False
        assert result.test_name == "mann-whitney"

    def test_format_fr_contains_protocol_fields(self) -> None:
        result = compare_groups(NORMAL_A, NORMAL_B, "QL", "SARSA")
        result.p_holm = 0.02
        text = result.format_fr()
        for token in ("IC 95 %", "n=10", "p=", "p_Holm=", "g=", "effet"):
            assert token in text
