"""Tests for the TrackMania wrapper, using a fake tmrl-like environment.

The real tmrl environment needs TrackMania 2020 running on Windows; these
tests exercise the wrapper's flattening, normalization, clipping and
pass-through logic against a deterministic in-memory fake that satisfies
``TMEnvProtocol``.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pytest
from gymnasium import spaces
from numpy.typing import NDArray

from src.environments.trackmania_wrapper import (
    OBS_DIM,
    TrackManiaEnvWrapper,
    make_trackmania_env,
    validate_tmrl_spaces,
)

TupleObs = tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]


def make_obs(
    speed: float = 500.0,
    lidar: NDArray[np.float32] | None = None,
    prev_action: tuple[float, float, float] = (0.1, 0.2, 0.3),
    prev_prev_action: tuple[float, float, float] = (-0.1, -0.2, -0.3),
) -> TupleObs:
    """Build a canned tmrl-style LIDAR tuple observation with known values."""
    if lidar is None:
        lidar = np.full((4, 19), 100.0, dtype=np.float32)
    return (
        np.array([speed], dtype=np.float32),
        lidar,
        np.array(prev_action, dtype=np.float32),
        np.array(prev_prev_action, dtype=np.float32),
    )


class FakeTMEnv:
    """Deterministic stand-in for ``tmrl.get_environment()`` (TMEnvProtocol).

    Records the last action and reset arguments it receives so tests can
    assert what actually reached the underlying environment.
    """

    def __init__(
        self,
        obs: TupleObs | None = None,
        reward: float = 1.0,
        terminated: bool = False,
        truncated: bool = False,
        info: dict[str, Any] | None = None,
    ) -> None:
        self.observation_space = spaces.Tuple(
            (
                spaces.Box(0.0, 1000.0, shape=(1,), dtype=np.float32),
                spaces.Box(0.0, np.inf, shape=(4, 19), dtype=np.float32),
                spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
                spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
            )
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self._obs = obs if obs is not None else make_obs()
        self._reward = reward
        self._terminated = terminated
        self._truncated = truncated
        self._info = info if info is not None else {}
        self.last_action: NDArray[np.float32] | None = None
        self.last_seed: int | None = None
        self.last_options: dict[str, Any] | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self.last_seed = seed
        self.last_options = options
        return self._obs, dict(self._info)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self.last_action = np.asarray(action, dtype=np.float32)
        return self._obs, self._reward, self._terminated, self._truncated, dict(self._info)


class TestObservationFlattening:
    def test_shape_and_dtype(self) -> None:
        wrapper = TrackManiaEnvWrapper(FakeTMEnv())
        obs, _ = wrapper.reset()
        assert obs.shape == (OBS_DIM,)
        assert obs.shape == (83,)
        assert obs.dtype == np.float32
        step_obs, _, _, _, _ = wrapper.step(np.zeros(3, dtype=np.float32))
        assert step_obs.shape == (83,)
        assert step_obs.dtype == np.float32

    def test_flattening_order(self) -> None:
        lidar = np.arange(76, dtype=np.float32).reshape(4, 19)
        fake = FakeTMEnv(obs=make_obs(speed=500.0, lidar=lidar))
        wrapper = TrackManiaEnvWrapper(fake, max_speed=1000.0, max_lidar=400.0)
        obs, _ = wrapper.reset()
        assert obs[0] == pytest.approx(0.5)  # speed 500 / 1000
        assert np.allclose(obs[1:77], np.arange(76, dtype=np.float32) / 400.0)
        assert np.allclose(obs[77:80], [0.1, 0.2, 0.3])
        assert np.allclose(obs[80:83], [-0.1, -0.2, -0.3])

    def test_speed_normalization_and_clipping(self) -> None:
        wrapper = TrackManiaEnvWrapper(FakeTMEnv(obs=make_obs(speed=500.0)), max_speed=1000.0)
        assert wrapper.reset()[0][0] == pytest.approx(0.5)
        wrapper = TrackManiaEnvWrapper(FakeTMEnv(obs=make_obs(speed=2000.0)), max_speed=1000.0)
        assert wrapper.reset()[0][0] == pytest.approx(1.0)
        wrapper = TrackManiaEnvWrapper(FakeTMEnv(obs=make_obs(speed=-50.0)), max_speed=1000.0)
        assert wrapper.reset()[0][0] == pytest.approx(0.0)

    def test_lidar_normalization_is_clipped_to_unit_range(self) -> None:
        lidar = np.full((4, 19), 900.0, dtype=np.float32)  # beyond max_lidar=400
        lidar[0, 0] = -10.0
        lidar[0, 1] = 200.0
        wrapper = TrackManiaEnvWrapper(FakeTMEnv(obs=make_obs(lidar=lidar)), max_lidar=400.0)
        obs, _ = wrapper.reset()
        assert obs[1] == pytest.approx(0.0)  # negative clipped to 0
        assert obs[2] == pytest.approx(0.5)  # 200 / 400
        assert np.allclose(obs[3:77], 1.0)  # 900 / 400 clipped to 1

    def test_observation_is_within_declared_space(self) -> None:
        wrapper = TrackManiaEnvWrapper(FakeTMEnv())
        obs, _ = wrapper.reset()
        assert wrapper.observation_space.contains(obs)


class TestActionHandling:
    def test_out_of_range_action_is_clipped_before_env(self) -> None:
        fake = FakeTMEnv()
        wrapper = TrackManiaEnvWrapper(fake)
        wrapper.step(np.array([2.0, -3.5, 0.25], dtype=np.float32))
        assert fake.last_action is not None
        assert np.allclose(fake.last_action, [1.0, -1.0, 0.25])

    def test_in_range_action_is_forwarded_untouched(self) -> None:
        fake = FakeTMEnv()
        wrapper = TrackManiaEnvWrapper(fake)
        wrapper.step(np.array([0.5, -0.5, 1.0], dtype=np.float32))
        assert fake.last_action is not None
        assert np.allclose(fake.last_action, [0.5, -0.5, 1.0])


class TestPassThrough:
    def test_reward_terminated_and_info_pass_through(self) -> None:
        fake = FakeTMEnv(reward=2.5, terminated=True, truncated=False, info={"lap": 1})
        wrapper = TrackManiaEnvWrapper(fake)
        _, reward, terminated, truncated, info = wrapper.step(np.zeros(3, dtype=np.float32))
        assert reward == 2.5
        assert terminated is True
        assert truncated is False
        assert info == {"lap": 1}

    def test_truncated_passes_through(self) -> None:
        fake = FakeTMEnv(truncated=True)
        wrapper = TrackManiaEnvWrapper(fake)
        _, _, terminated, truncated, _ = wrapper.step(np.zeros(3, dtype=np.float32))
        assert terminated is False
        assert truncated is True


class TestReset:
    def test_seed_and_options_are_forwarded(self) -> None:
        fake = FakeTMEnv()
        wrapper = TrackManiaEnvWrapper(fake)
        wrapper.reset(seed=123, options={"track": "test"})
        assert fake.last_seed == 123
        assert fake.last_options == {"track": "test"}

    def test_reset_without_seed(self) -> None:
        fake = FakeTMEnv()
        wrapper = TrackManiaEnvWrapper(fake)
        obs, info = wrapper.reset()
        assert fake.last_seed is None
        assert obs.shape == (83,)
        assert info == {}


class TestSpaceValidation:
    def _lidar_spaces(self) -> tuple[spaces.Tuple, spaces.Box]:
        fake = FakeTMEnv()
        return fake.observation_space, fake.action_space

    def test_valid_lidar_spaces_pass(self) -> None:
        obs_space, action_space = self._lidar_spaces()
        validate_tmrl_spaces(obs_space, action_space)  # must not raise

    def test_non_tuple_observation_space_raises(self) -> None:
        _, action_space = self._lidar_spaces()
        box = spaces.Box(0.0, 255.0, shape=(4, 64, 64), dtype=np.float32)
        with pytest.raises(ValueError, match="RTGYM_INTERFACE"):
            validate_tmrl_spaces(box, action_space)

    def test_wrong_tuple_arity_raises(self) -> None:
        obs_space, action_space = self._lidar_spaces()
        five = spaces.Tuple((*obs_space.spaces, spaces.Box(0.0, 1.0, (1,), dtype=np.float32)))
        with pytest.raises(ValueError, match="RTGYM_INTERFACE"):
            validate_tmrl_spaces(five, action_space)

    def test_wrong_lidar_shape_raises(self) -> None:
        obs_space, action_space = self._lidar_spaces()
        bad = spaces.Tuple(
            (
                obs_space.spaces[0],
                spaces.Box(0.0, np.inf, shape=(4, 20), dtype=np.float32),
                obs_space.spaces[2],
                obs_space.spaces[3],
            )
        )
        with pytest.raises(ValueError, match=r"\(4, 20\)"):
            validate_tmrl_spaces(bad, action_space)

    def test_wrong_action_shape_raises(self) -> None:
        obs_space, _ = self._lidar_spaces()
        with pytest.raises(ValueError, match="action space"):
            validate_tmrl_spaces(obs_space, spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32))

    def test_wrapper_construction_validates_spaces(self) -> None:
        fake = FakeTMEnv()
        fake.observation_space = spaces.Box(0.0, 255.0, shape=(4, 64, 64), dtype=np.float32)
        with pytest.raises(ValueError, match="RTGYM_INTERFACE"):
            TrackManiaEnvWrapper(fake)


class TestFlattenErrors:
    def test_wrong_component_count_at_reset_raises(self) -> None:
        speed, lidar, prev, _ = make_obs()
        fake = FakeTMEnv(obs=(speed, lidar, prev))  # type: ignore[arg-type]
        wrapper = TrackManiaEnvWrapper(fake)
        with pytest.raises(ValueError, match="4-component"):
            wrapper.reset()

    def test_wrong_lidar_payload_shape_raises(self) -> None:
        lidar = np.full((3, 19), 100.0, dtype=np.float32)  # 3 scans instead of 4
        fake = FakeTMEnv(obs=make_obs(lidar=lidar))
        wrapper = TrackManiaEnvWrapper(fake)
        with pytest.raises(ValueError, match="shape"):
            wrapper.reset()


class TestWaitAndClose:
    class _WaitableFake(FakeTMEnv):
        def __init__(self) -> None:
            super().__init__()
            self.wait_calls = 0
            self.close_calls = 0

        def wait(self) -> None:
            self.wait_calls += 1

        def close(self) -> None:
            self.close_calls += 1

    def test_wait_is_forwarded(self) -> None:
        fake = self._WaitableFake()
        wrapper = TrackManiaEnvWrapper(fake)
        wrapper.wait()
        assert fake.wait_calls == 1

    def test_close_is_forwarded(self) -> None:
        fake = self._WaitableFake()
        wrapper = TrackManiaEnvWrapper(fake)
        wrapper.close()
        assert fake.close_calls == 1

    def test_wait_without_support_is_a_noop(self) -> None:
        wrapper = TrackManiaEnvWrapper(FakeTMEnv())
        wrapper.wait()  # must not raise

    def test_close_without_support_is_a_noop(self) -> None:
        wrapper = TrackManiaEnvWrapper(FakeTMEnv())
        wrapper.close()  # must not raise


class TestVigemRetry:
    class _FlakyFake(FakeTMEnv):
        """Fails the first N resets with the ViGEmBus attach assertion."""

        def __init__(self, failures: int, message: str = "could not connect to ViGEmBus.") -> None:
            super().__init__()
            self.failures = failures
            self.message = message
            self.reset_calls = 0

        def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
        ) -> tuple[Any, dict[str, Any]]:
            self.reset_calls += 1
            if self.reset_calls <= self.failures:
                raise AssertionError(self.message)
            return super().reset(seed=seed, options=options)

    def test_transient_attach_failure_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.environments.trackmania_wrapper as tw

        monkeypatch.setattr(tw, "VIGEM_RETRY_DELAY_S", 0.0)
        fake = self._FlakyFake(failures=1)
        wrapper = TrackManiaEnvWrapper(fake)
        obs, _ = wrapper.reset()
        assert obs.shape == (OBS_DIM,)
        assert fake.reset_calls == 2

    def test_persistent_attach_failure_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.environments.trackmania_wrapper as tw

        monkeypatch.setattr(tw, "VIGEM_RETRY_DELAY_S", 0.0)
        fake = self._FlakyFake(failures=99)
        wrapper = TrackManiaEnvWrapper(fake)
        with pytest.raises(RuntimeError, match="ViGEmBus"):
            wrapper.reset()

    def test_unrelated_assertion_propagates_immediately(self) -> None:
        fake = self._FlakyFake(failures=99, message="something else broke")
        wrapper = TrackManiaEnvWrapper(fake)
        with pytest.raises(AssertionError, match="something else"):
            wrapper.reset()
        assert fake.reset_calls == 1


class TestMakeTrackmaniaEnv:
    def test_missing_tmrl_raises_actionable_import_error(self) -> None:
        if importlib.util.find_spec("tmrl") is not None:
            pytest.skip("tmrl is installed: cannot exercise the missing-dependency path")
        with pytest.raises(ImportError, match=r"TRACKMANIA\.md"):
            make_trackmania_env()
