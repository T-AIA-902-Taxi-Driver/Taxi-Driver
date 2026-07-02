"""Tests for the 2-passenger env: encoding, transitions, TimeLimit, training."""

import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.agents import create_agent
from src.config import Config
from src.environments import create_env
from src.environments.multi_passenger_env import (
    N_ACTIONS,
    N_STATES,
    STATUS_DELIVERED,
    STATUS_IN_TAXI,
    MultiEnvWrapper,
    MultiPassengerTaxiEnv,
    decode,
    encode,
)
from src.environments.route_analysis import (
    greedy_nearest_order,
    min_route_length,
    optimal_pickup_order,
)
from src.training.trainer import Trainer

# Landmark indices, for readable scenario set-ups.
R, G, Y, B = 0, 1, 2, 3


@pytest.fixture()
def raw() -> MultiPassengerTaxiEnv:
    return MultiPassengerTaxiEnv()


class TestEncoding:
    def test_sizes(self) -> None:
        assert N_STATES == 14_400
        assert N_ACTIONS == 6

    def test_bijection_over_all_states(self) -> None:
        for state in range(N_STATES):
            row, col, s0, s1, d0, d1 = decode(state)
            assert 0 <= row < 5 and 0 <= col < 5
            assert 0 <= s0 < 6 and 0 <= s1 < 6
            assert 0 <= d0 < 4 and 0 <= d1 < 4
            assert encode(row, col, s0, s1, d0, d1) == state

    def test_extreme_states(self) -> None:
        assert encode(0, 0, 0, 0, 0, 0) == 0
        assert encode(4, 4, 5, 5, 3, 3) == N_STATES - 1


class TestReset:
    def test_invariants_over_200_seeded_resets(self, raw: MultiPassengerTaxiEnv) -> None:
        for seed in range(200):
            state, info = raw.reset(seed=seed)
            assert type(state) is int
            row, col, s0, s1, d0, d1 = decode(state)
            assert info["decoded"] == (row, col, s0, s1, d0, d1)
            assert 0 <= row < 5 and 0 <= col < 5
            assert 0 <= s0 <= 3 and 0 <= s1 <= 3  # both waiting at a landmark
            assert s0 != s1  # distinct start locations
            assert d0 != s0 and d1 != s1  # dest differs from own start

    def test_same_seed_same_initial_state(self, raw: MultiPassengerTaxiEnv) -> None:
        s1, _ = raw.reset(seed=42)
        s2, _ = raw.reset(seed=42)
        assert s1 == s2
        assert MultiPassengerTaxiEnv().reset(seed=42)[0] == s1


class TestWrapperSeeding:
    def test_same_constructor_seed_same_first_reset(self) -> None:
        s1, _ = MultiEnvWrapper(seed=7).reset()
        s2, _ = MultiEnvWrapper(seed=7).reset()
        assert s1 == s2

    def test_explicit_seed_reproducible_mid_sequence(self) -> None:
        env = MultiEnvWrapper(seed=42)
        s1, _ = env.reset(seed=123)
        env.step(0)
        env.reset()  # advance the RNG stream mid-sequence
        s2, _ = env.reset(seed=123)
        assert s1 == s2

    def test_bare_resets_do_not_reseed(self) -> None:
        env = MultiEnvWrapper(seed=42)
        states = [env.reset()[0] for _ in range(5)]
        assert len(set(states)) > 1


class TestPickup:
    def test_lowest_index_wins_when_both_eligible(self, raw: MultiPassengerTaxiEnv) -> None:
        # Impossible at reset (distinct starts): forced via _set_state.
        raw._set_state(0, 0, R, R, G, Y)
        state, reward, terminated, truncated, _ = raw.step(4)
        _, _, s0, s1, _, _ = decode(state)
        assert (s0, s1) == (STATUS_IN_TAXI, R)
        assert reward == -1.0
        assert not terminated and not truncated

    def test_illegal_pickup_empty_cell(self, raw: MultiPassengerTaxiEnv) -> None:
        before = raw._set_state(2, 2, R, Y, G, B)
        state, reward, *_ = raw.step(4)
        assert reward == -10.0
        assert state == before  # nothing changed

    def test_second_pickup_puts_both_in_taxi(self, raw: MultiPassengerTaxiEnv) -> None:
        raw._set_state(0, 0, R, G, G, R)
        raw.step(4)  # picks P1 at R
        raw._set_state(0, 4, STATUS_IN_TAXI, G, G, R)  # drive to G
        state, reward, *_ = raw.step(4)
        _, _, s0, s1, _, _ = decode(state)
        assert (s0, s1) == (STATUS_IN_TAXI, STATUS_IN_TAXI)  # capacity 2
        assert reward == -1.0


class TestDropoff:
    def test_dropoff_at_destination_delivers(self, raw: MultiPassengerTaxiEnv) -> None:
        raw._set_state(0, 4, STATUS_IN_TAXI, Y, G, B)  # taxi at G, P1 destined G
        state, reward, terminated, *_ = raw.step(5)
        _, _, s0, s1, _, _ = decode(state)
        assert s0 == STATUS_DELIVERED
        assert s1 == Y
        assert reward == 20.0
        assert not terminated  # P2 still waiting

    def test_second_delivery_terminates(self, raw: MultiPassengerTaxiEnv) -> None:
        raw._set_state(4, 3, STATUS_DELIVERED, STATUS_IN_TAXI, G, B)  # taxi at B
        state, reward, terminated, truncated, _ = raw.step(5)
        _, _, s0, s1, _, _ = decode(state)
        assert (s0, s1) == (STATUS_DELIVERED, STATUS_DELIVERED)
        assert reward == 20.0
        assert terminated and not truncated

    def test_illegal_dropoff_off_landmark(self, raw: MultiPassengerTaxiEnv) -> None:
        before = raw._set_state(2, 2, STATUS_IN_TAXI, Y, G, B)
        state, reward, *_ = raw.step(5)
        assert reward == -10.0
        assert state == before

    def test_setdown_forbidden_at_wrong_landmark(self, raw: MultiPassengerTaxiEnv) -> None:
        # Taxi on Y, P1 in taxi destined G: Taxi-v3 would set the passenger
        # down for -1; here the set-down is forbidden (-10, passenger stays).
        before = raw._set_state(4, 0, STATUS_IN_TAXI, R, G, B)
        state, reward, *_ = raw.step(5)
        assert reward == -10.0
        assert state == before
        assert decode(state)[2] == STATUS_IN_TAXI

    def test_same_destination_needs_two_dropoffs(self, raw: MultiPassengerTaxiEnv) -> None:
        raw._set_state(0, 4, STATUS_IN_TAXI, STATUS_IN_TAXI, G, G)  # both destined G
        state, reward, terminated, *_ = raw.step(5)
        _, _, s0, s1, _, _ = decode(state)
        assert (s0, s1) == (STATUS_DELIVERED, STATUS_IN_TAXI)  # lowest index first
        assert reward == 20.0 and not terminated
        state, reward, terminated, *_ = raw.step(5)
        _, _, s0, s1, _, _ = decode(state)
        assert (s0, s1) == (STATUS_DELIVERED, STATUS_DELIVERED)
        assert reward == 20.0 and terminated


class TestMoves:
    def test_wall_blocks_east_from_0_1(self, raw: MultiPassengerTaxiEnv) -> None:
        # MAP row 0 has a "|" between columns 1 and 2 ("|R: | : :G|").
        raw._set_state(0, 1, R, Y, G, B)
        state, reward, terminated, *_ = raw.step(2)
        assert decode(state)[:2] == (0, 1)  # position unchanged
        assert reward == -1.0 and not terminated

    def test_border_clamps_north(self, raw: MultiPassengerTaxiEnv) -> None:
        raw._set_state(0, 2, R, Y, G, B)
        state, reward, *_ = raw.step(1)
        assert decode(state)[:2] == (0, 2)
        assert reward == -1.0

    def test_open_move_south(self, raw: MultiPassengerTaxiEnv) -> None:
        raw._set_state(0, 2, R, Y, G, B)
        state, reward, *_ = raw.step(0)
        assert decode(state)[:2] == (1, 2)
        assert reward == -1.0


class TestRender:
    def test_ansi_grid_and_legend(self, raw: MultiPassengerTaxiEnv) -> None:
        raw._set_state(2, 2, R, Y, G, B)
        frame = raw.render()
        assert isinstance(frame, str)
        assert "+---------+" in frame
        assert "T" in frame
        assert "1" in frame and "2" in frame  # waiting passengers on their cells
        assert "P1: at R -> G | P2: at Y -> B" in frame

    def test_in_taxi_legend(self, raw: MultiPassengerTaxiEnv) -> None:
        raw._set_state(2, 2, STATUS_IN_TAXI, Y, G, B)
        frame = raw.render()
        assert frame is not None
        assert "P1: in taxi -> G" in frame


class TestFactoryAndTimeLimit:
    def test_create_env_multi_returns_wrapper(self) -> None:
        env = create_env(Config(env="multi", seed=7))
        assert isinstance(env, MultiEnvWrapper)
        assert env.n_states == 14_400
        assert env.n_actions == 6
        assert type(env.n_states) is int

    def test_truncates_at_500_pure_moves(self) -> None:
        env = create_env(Config(env="multi", seed=7))
        env.reset()
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(0)  # moving never terminates
            steps += 1
        assert steps == 500
        assert truncated and not terminated

    def test_step_surface_matches_taxi_wrapper(self) -> None:
        env = create_env(Config(env="multi", seed=7))
        state, info = env.reset()
        assert type(state) is int and isinstance(info, dict)
        next_state, reward, terminated, truncated, info = env.step(0)
        assert type(next_state) is int and type(reward) is float
        assert type(terminated) is bool and type(truncated) is bool
        assert info["raw_reward"] == reward

    def test_reward_fn_passthrough(self) -> None:
        env = create_env(Config(env="multi", seed=7), reward_fn=lambda s, a, r, s2: r + 1.0)
        env.reset()
        _, reward, _, _, info = env.step(0)
        assert reward == info["raw_reward"] + 1.0

    def test_wrapper_decode_state(self) -> None:
        env = create_env(Config(env="multi", seed=7))
        assert isinstance(env, MultiEnvWrapper)
        state, _ = env.reset()
        assert env.decode_state(state) == decode(state)


class TestEnvChecker:
    def test_gymnasium_check_env_passes(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            check_env(MultiPassengerTaxiEnv())


class TestTrainerIntegration:
    def test_q_learning_trains_300_episodes(self) -> None:
        config = Config(env="multi", algorithm="q_learning", seed=3, n_train_episodes=300)
        env = create_env(config)
        agent = create_agent(config, env, n_episodes=config.n_train_episodes)
        history = Trainer(agent, env, config).train(config.n_train_episodes)
        assert len(history.rewards) == 300
        q_table = agent.q_table  # type: ignore[attr-defined]
        assert q_table.shape == (14_400, 6)
        assert np.count_nonzero(q_table) > 0


class TestRouteAnalysis:
    def test_hand_computed_symmetric_case(self) -> None:
        # Taxi (2,2); P1 R->G, P2 Y->B. Enumerated by hand (Manhattan, no walls):
        #   p1 d1 p2 d2: 4+4+8+3 = 19    p1 p2 d1 d2: 4+4+8+5 = 21
        #   p1 p2 d2 d1: 4+4+3+5 = 16 *  p2 d2 p1 d1: 4+3+7+4 = 18
        #   p2 p1 d2 d1: 4+4+7+5 = 20    p2 p1 d1 d2: 4+4+4+5 = 17
        state = encode(2, 2, R, Y, G, B)
        assert optimal_pickup_order(state) == (0, 1)
        assert min_route_length(state) == 16
        assert greedy_nearest_order(state) == (0, 1)  # tie (4 vs 4) -> lowest index

    def test_hand_computed_asymmetric_case(self) -> None:
        # Taxi (3,0); P1 R->G, P2 Y->B. Best: p2 p1 d1 d2 = 1+4+4+5 = 14.
        state = encode(3, 0, R, Y, G, B)
        assert optimal_pickup_order(state) == (1, 0)
        assert min_route_length(state) == 14
        assert greedy_nearest_order(state) == (1, 0)  # Y (dist 1) closer than R (3)

    def test_in_taxi_passenger_contributes_dropoff_only(self) -> None:
        state = encode(0, 0, STATUS_IN_TAXI, Y, G, B)
        assert optimal_pickup_order(state) == (1,)
        assert greedy_nearest_order(state) == (1,)

    def test_all_delivered_empty_orders(self) -> None:
        state = encode(0, 0, STATUS_DELIVERED, STATUS_DELIVERED, G, B)
        assert optimal_pickup_order(state) == ()
        assert greedy_nearest_order(state) == ()
        assert min_route_length(state) == 0
