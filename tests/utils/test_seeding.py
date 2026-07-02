"""Tests for seeding utilities: determinism, independence, disjoint seed sets."""

import numpy as np
import pytest

from src.utils.seeding import eval_seeds, probe_seeds, spawn_rngs


class TestSpawnRngs:
    def test_same_seed_same_streams(self) -> None:
        a1, b1 = spawn_rngs(42, 2)
        a2, b2 = spawn_rngs(42, 2)
        assert np.array_equal(a1.random(10), a2.random(10))
        assert np.array_equal(b1.random(10), b2.random(10))

    def test_streams_are_distinct(self) -> None:
        a, b = spawn_rngs(42, 2)
        assert not np.array_equal(a.random(10), b.random(10))

    def test_different_seeds_differ(self) -> None:
        (a,) = spawn_rngs(42, 1)
        (b,) = spawn_rngs(43, 1)
        assert not np.array_equal(a.random(10), b.random(10))


class TestSeedLists:
    def test_eval_seeds_deterministic_and_unique(self) -> None:
        seeds = eval_seeds(42, 100)
        assert seeds == eval_seeds(42, 100)
        assert len(set(seeds)) == 100

    def test_probe_and_eval_are_disjoint(self) -> None:
        assert not set(eval_seeds(42, 100)) & set(probe_seeds(42, 30))

    def test_probe_seeds_overlap_guard(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            probe_seeds(42, 10_000)
