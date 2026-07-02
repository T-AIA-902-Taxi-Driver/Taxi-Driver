"""Tests for the Taxi-v3 wrapper: types, seeding, shaping, decoding, TimeLimit."""

from typing import Any, cast

import pytest

from src.config import Config
from src.environments import create_env
from src.environments.taxi_wrapper import TAXI_LOCATIONS, TaxiEnvWrapper


@pytest.fixture()
def env() -> TaxiEnvWrapper:
    return TaxiEnvWrapper(seed=42)


class TestSpaces:
    def test_n_states(self, env: TaxiEnvWrapper) -> None:
        assert env.n_states == 500
        assert isinstance(env.n_states, int)

    def test_n_actions(self, env: TaxiEnvWrapper) -> None:
        assert env.n_actions == 6
        assert isinstance(env.n_actions, int)


class TestResetAndStep:
    def test_reset_returns_int_and_info(self, env: TaxiEnvWrapper) -> None:
        state, info = env.reset()
        assert type(state) is int
        assert 0 <= state < 500
        assert isinstance(info, dict)

    def test_step_types(self, env: TaxiEnvWrapper) -> None:
        env.reset()
        state, reward, terminated, truncated, info = env.step(0)
        assert type(state) is int
        assert type(reward) is float
        assert type(terminated) is bool
        assert type(truncated) is bool
        assert isinstance(info, dict)

    def test_raw_reward_always_present(self, env: TaxiEnvWrapper) -> None:
        env.reset()
        _, reward, _, _, info = env.step(1)
        assert "raw_reward" in info
        assert info["raw_reward"] == reward


class TestRewardShaping:
    def test_shaped_reward_and_native_in_info(self) -> None:
        env = TaxiEnvWrapper(reward_fn=lambda s, a, r, s2: r + 1.0, seed=42)
        env.reset()
        _, reward, _, _, info = env.step(0)
        assert reward == info["raw_reward"] + 1.0

    def test_shaping_fn_receives_transition(self) -> None:
        seen: list[tuple[int, int, float, int]] = []

        def spy(s: int, a: int, r: float, s2: int) -> float:
            seen.append((s, a, r, s2))
            return r

        env = TaxiEnvWrapper(reward_fn=spy, seed=42)
        start, _ = env.reset()
        next_state, _, _, _, info = env.step(3)
        assert seen == [(start, 3, info["raw_reward"], next_state)]


class TestSeeding:
    def test_same_constructor_seed_same_first_reset(self) -> None:
        s1, _ = TaxiEnvWrapper(seed=7).reset()
        s2, _ = TaxiEnvWrapper(seed=7).reset()
        assert s1 == s2

    def test_explicit_seed_reproducible_mid_sequence(self, env: TaxiEnvWrapper) -> None:
        s1, _ = env.reset(seed=123)
        env.step(0)
        env.reset()  # advance the RNG stream mid-sequence
        s2, _ = env.reset(seed=123)
        assert s1 == s2

    def test_bare_resets_do_not_reseed(self, env: TaxiEnvWrapper) -> None:
        states = [env.reset()[0] for _ in range(5)]
        assert len(set(states)) > 1


class TestDecodeState:
    def test_decode_ranges(self, env: TaxiEnvWrapper) -> None:
        env.reset()
        state, *_ = env.step(0)
        decoded = env.decode_state(state)
        assert len(decoded) == 4
        row, col, passenger, destination = decoded
        assert all(type(v) is int for v in decoded)
        assert 0 <= row <= 4
        assert 0 <= col <= 4
        assert 0 <= passenger <= 4
        assert 0 <= destination <= 3

    def test_encode_decode_round_trip(self, env: TaxiEnvWrapper) -> None:
        encoded = int(cast(Any, env.env.unwrapped).encode(3, 1, 2, 0))
        row, col, passenger, destination = env.decode_state(encoded)
        assert (row, col, passenger, destination) == (3, 1, 2, 0)
        # Landmark indices map to the module-level grid coordinates.
        assert TAXI_LOCATIONS[passenger] == (4, 0)
        assert TAXI_LOCATIONS[destination] == (0, 0)

    def test_taxi_locations_match_env(self, env: TaxiEnvWrapper) -> None:
        locs = cast(Any, env.env.unwrapped).locs
        assert {i: tuple(loc) for i, loc in enumerate(locs)} == TAXI_LOCATIONS


class TestRender:
    def test_render_returns_grid_string(self, env: TaxiEnvWrapper) -> None:
        env.reset()
        frame = env.render()
        assert isinstance(frame, str)
        assert frame
        assert "+---------+" in frame


class TestTimeLimit:
    def test_truncated_after_max_steps(self) -> None:
        env = TaxiEnvWrapper(seed=42, max_steps=10)
        env.reset()
        truncated = False
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(0)
            assert not terminated  # action 0 (south) never ends the episode
        assert truncated


class TestFactory:
    def test_create_env_taxi(self) -> None:
        env = create_env(Config(seed=7, max_steps_per_episode=10))
        assert isinstance(env, TaxiEnvWrapper)
        env.reset()
        truncated = False
        for _ in range(10):
            *_, truncated, _ = env.step(0)
        assert truncated

    def test_create_env_passes_reward_fn(self) -> None:
        env = create_env(Config(seed=7), reward_fn=lambda s, a, r, s2: 0.0)
        env.reset()
        _, reward, _, _, info = env.step(0)
        assert reward == 0.0
        assert info["raw_reward"] != 0.0
