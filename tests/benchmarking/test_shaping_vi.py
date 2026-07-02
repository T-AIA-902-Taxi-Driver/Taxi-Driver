"""Reward shaping and value-iteration reference tests."""

import itertools

import pytest

from src.benchmarking.reward_shaping import (
    _decode,
    create_reward_shaper,
    make_naive_distance_shaper,
    make_potential_shaper,
    make_step_penalty_shaper,
    potential,
)
from src.benchmarking.value_iteration import optimal_q_table, optimal_reference_reward
from src.config import Config
from src.environments import create_env
from src.utils.seeding import eval_seeds


class TestDecode:
    def test_matches_env_decode(self) -> None:
        env = create_env(Config())
        for state in (0, 137, 258, 399, 499):
            assert _decode(state) == env.decode_state(state)


class TestPotentialShaping:
    def test_telescoping_sum_along_trajectory(self) -> None:
        """Potential-based shaping adds γΦ(s_T)−Φ(s_0) over any trajectory."""
        gamma = 0.9
        shaper = make_potential_shaper(gamma)
        trajectory = [42, 142, 262, 258]  # arbitrary state chain
        rewards = [-1.0, -1.0, -1.0]
        shaped_sum = sum(
            shaper(s, 0, r, s2)
            for s, r, s2 in zip(trajectory, rewards, trajectory[1:], strict=False)
        )
        native_sum = sum(rewards)
        # telescoping with discount 0.9 leaves a weighted potential difference
        expected_extra = sum(
            gamma * potential(s2) - potential(s) for s, s2 in itertools.pairwise(trajectory)
        )
        assert shaped_sum == pytest.approx(native_sum + expected_extra)

    def test_moving_toward_objective_is_rewarded(self) -> None:
        shaper = make_potential_shaper(1.0)
        # state with passenger at R(0,0): taxi at (0,1) → (0,0) gets closer
        state_far = ((0 * 5 + 1) * 5 + 0) * 4 + 1  # taxi (0,1), passenger R, dest G
        state_near = ((0 * 5 + 0) * 5 + 0) * 4 + 1  # taxi (0,0)
        assert shaper(state_far, 3, -1.0, state_near) > -1.0
        assert shaper(state_near, 2, -1.0, state_far) < -1.0


class TestOtherShapers:
    def test_naive_bonus_only_on_progress(self) -> None:
        shaper = make_naive_distance_shaper(bonus=0.5)
        state_far = ((0 * 5 + 1) * 5 + 0) * 4 + 1
        state_near = ((0 * 5 + 0) * 5 + 0) * 4 + 1
        assert shaper(state_far, 3, -1.0, state_near) == -0.5
        assert shaper(state_near, 2, -1.0, state_far) == -1.0

    def test_step_penalty_targets_moves_only(self) -> None:
        shaper = make_step_penalty_shaper(1.0)
        assert shaper(0, 0, -1.0, 1) == -2.0
        assert shaper(0, 5, -10.0, 1) == -10.0
        assert shaper(0, 5, 20.0, 1) == 20.0

    def test_factory(self) -> None:
        assert create_reward_shaper("none", 0.99) is None
        assert create_reward_shaper("potential", 0.99) is not None
        with pytest.raises(ValueError, match="unknown"):
            create_reward_shaper("magic", 0.99)


class TestValueIteration:
    def test_r_star_matches_known_taxi_optimum(self) -> None:
        env = create_env(Config())
        r_star = optimal_reference_reward(env, eval_seeds(42, 50))
        assert 7.0 < r_star < 9.0  # literature: ~7.9 for Taxi-v3

    def test_optimal_policy_solves_everything_quickly(self) -> None:
        import numpy as np

        env = create_env(Config())
        q_star = optimal_q_table(env)
        for seed in eval_seeds(42, 20):
            state, _ = env.reset(seed=seed)
            for _ in range(30):
                state, _, terminated, truncated, _ = env.step(int(np.argmax(q_star[state])))
                assert not truncated
                if terminated:
                    break
            assert terminated
