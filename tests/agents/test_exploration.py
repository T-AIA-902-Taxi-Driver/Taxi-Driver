"""Tests for exploration strategies: selection laws, decay, probabilities."""

import numpy as np
import pytest

from src.agents.exploration import UCB, Boltzmann, EpsilonGreedy, create_exploration
from src.config import Config

Q_ROW = np.array([0.0, 5.0, 1.0, 5.0], dtype=np.float64)  # ties on actions 1 and 3


class TestEpsilonGreedy:
    def test_epsilon_zero_is_pure_argmax(self, rng: np.random.Generator) -> None:
        strategy = EpsilonGreedy(rng, epsilon=0.0)
        actions = {strategy.select(Q_ROW, 0) for _ in range(50)}
        assert actions <= {1, 3}  # only maximal actions (random tie-breaking)

    def test_epsilon_one_covers_all_actions(self, rng: np.random.Generator) -> None:
        strategy = EpsilonGreedy(rng, epsilon=1.0)
        actions = {strategy.select(Q_ROW, 0) for _ in range(200)}
        assert actions == {0, 1, 2, 3}

    def test_exp_decay_floors_at_min(self, rng: np.random.Generator) -> None:
        strategy = EpsilonGreedy(rng, epsilon=1.0, epsilon_min=0.5, decay_rate=0.1)
        for _ in range(10):
            strategy.decay()
        assert strategy.epsilon == 0.5

    def test_linear_decay(self, rng: np.random.Generator) -> None:
        strategy = EpsilonGreedy(
            rng, epsilon=1.0, epsilon_min=0.0, decay_rate=0.25, decay_type="linear"
        )
        strategy.decay()
        assert strategy.epsilon == pytest.approx(0.75)

    def test_action_probs_sum_to_one_and_favour_ties(self, rng: np.random.Generator) -> None:
        strategy = EpsilonGreedy(rng, epsilon=0.4)
        probs = strategy.action_probs(Q_ROW, 0)
        assert probs.sum() == pytest.approx(1.0)
        assert probs[1] == pytest.approx(0.1 + 0.6 / 2)  # eps/4 + (1-eps)/2 ties
        assert probs[0] == pytest.approx(0.1)

    def test_epsilon_like_reports_epsilon(self, rng: np.random.Generator) -> None:
        assert EpsilonGreedy(rng, epsilon=0.7).epsilon_like == 0.7


class TestBoltzmann:
    def test_probs_monotone_in_q(self, rng: np.random.Generator) -> None:
        strategy = Boltzmann(rng, temperature=1.0)
        probs = strategy.action_probs(Q_ROW, 0)
        assert probs.sum() == pytest.approx(1.0)
        assert probs[1] > probs[2] > probs[0]
        assert probs[1] == pytest.approx(probs[3])

    def test_low_temperature_approaches_greedy(self, rng: np.random.Generator) -> None:
        strategy = Boltzmann(rng, temperature=0.01)
        probs = strategy.action_probs(Q_ROW, 0)
        assert probs[1] + probs[3] == pytest.approx(1.0, abs=1e-6)

    def test_temperature_decay_floors(self, rng: np.random.Generator) -> None:
        strategy = Boltzmann(rng, temperature=1.0, temperature_min=0.5, decay_rate=0.1)
        for _ in range(10):
            strategy.decay()
        assert strategy.temperature == 0.5

    def test_numerical_stability_with_large_q(self, rng: np.random.Generator) -> None:
        huge = np.array([1e8, 1e8 + 1, 0.0], dtype=np.float64)
        probs = Boltzmann(rng, temperature=1.0).action_probs(huge, 0)
        assert np.isfinite(probs).all()
        assert probs.sum() == pytest.approx(1.0)


class TestUCB:
    def test_tries_every_untried_action_first(self, rng: np.random.Generator) -> None:
        strategy = UCB(rng, n_states=3, n_actions=4, c=2.0)
        first_four = {strategy.select(Q_ROW, state=1) for _ in range(4)}
        assert first_four == {0, 1, 2, 3}
        assert strategy.counts[1].sum() == 4
        assert strategy.counts[0].sum() == 0  # other states untouched

    def test_counts_only_grow_via_select(self, rng: np.random.Generator) -> None:
        strategy = UCB(rng, n_states=2, n_actions=4)
        strategy.action_probs(Q_ROW, 0)
        assert strategy.counts.sum() == 0

    def test_prefers_high_q_when_counts_equal(self, rng: np.random.Generator) -> None:
        strategy = UCB(rng, n_states=1, n_actions=4, c=0.5)
        for _ in range(4):
            strategy.select(Q_ROW, 0)
        picks = [strategy.select(Q_ROW, 0) for _ in range(20)]
        assert set(picks) <= {1, 3}

    def test_action_probs_one_hot_after_warmup(self, rng: np.random.Generator) -> None:
        strategy = UCB(rng, n_states=1, n_actions=4)
        for _ in range(4):
            strategy.select(Q_ROW, 0)
        probs = strategy.action_probs(Q_ROW, 0)
        assert probs.sum() == pytest.approx(1.0)
        assert (probs == 1.0).sum() == 1


class TestFactory:
    def test_builds_each_kind(self, rng: np.random.Generator) -> None:
        for name, cls in [
            ("epsilon_greedy", EpsilonGreedy),
            ("boltzmann", Boltzmann),
            ("ucb", UCB),
        ]:
            config = Config(exploration=name)
            assert isinstance(create_exploration(config, 10, 4, rng), cls)

    def test_decay_frac_resolved_through_factory(self, rng: np.random.Generator) -> None:
        config = Config(decay_frac=0.5, epsilon=1.0, epsilon_min=0.01)
        strategy = create_exploration(config, 10, 4, rng, n_episodes=1000)
        assert isinstance(strategy, EpsilonGreedy)
        # after 500 decays epsilon should hit epsilon_min
        for _ in range(500):
            strategy.decay()
        assert strategy.epsilon == pytest.approx(0.01, rel=1e-6)
