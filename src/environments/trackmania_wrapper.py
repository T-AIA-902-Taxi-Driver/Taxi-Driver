"""Gymnasium wrapper around the ``tmrl`` TrackMania 2020 real-time environment.

``tmrl`` exposes TrackMania 2020 as a Real-Time Gym environment whose default
LIDAR observation is a tuple ``(speed, lidar, prev_action, prev_prev_action)``
with shapes ``((1,), (4, 19), (3,), (3,))``. This module flattens that tuple
into a single normalized ``Box`` vector so standard deep-RL libraries
(e.g. Stable-Baselines3) can consume it directly.

The game only runs on a Windows machine with TrackMania 2020 and OpenPlanet
installed (see ``docs/TRACKMANIA.md``), so ``tmrl`` is imported lazily inside
:func:`make_trackmania_env` only. The wrapper itself is written against the
structural :class:`TMEnvProtocol`, which makes it unit-testable with a fake
environment on any machine, without the game.
"""

from __future__ import annotations

from typing import Any, Final, Protocol, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

SPEED_DIM: Final = 1
LIDAR_SHAPE: Final = (4, 19)
ACTION_DIM: Final = 3
OBS_DIM: Final = SPEED_DIM + LIDAR_SHAPE[0] * LIDAR_SHAPE[1] + 2 * ACTION_DIM  # 83


class TMEnvProtocol(Protocol):
    """Structural interface of the ``tmrl`` real-time environment.

    Any object exposing Gymnasium-style ``observation_space``/``action_space``
    attributes and ``reset``/``step`` methods satisfies this protocol, which is
    what allows the wrapper to be tested with a fake environment while the real
    one (``tmrl.get_environment()``) only exists on the game machine.
    """

    observation_space: Any
    action_space: Any

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the environment and return ``(observation, info)``."""
        ...

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Apply ``action`` and return ``(obs, reward, terminated, truncated, info)``."""
        ...


def validate_tmrl_spaces(observation_space: Any, action_space: Any) -> None:
    """Fail fast when the underlying environment is not on the LIDAR preset.

    The wrapper hard-assumes tmrl's ``TM20LIDAR`` interface. If the machine's
    tmrl configuration selects another preset (``TM20IMAGES`` camera images,
    ``TM20LIDARPROGRESS`` 5-component tuple, ...), every downstream shape
    breaks; validating the declared spaces at construction time surfaces the
    misconfiguration immediately with the exact fix, instead of failing on the
    first observation.

    Args:
        observation_space: Declared observation space of the underlying env.
        action_space: Declared action space of the underlying env.

    Raises:
        ValueError: If the spaces do not match the ``TM20LIDAR`` layout, with
            the config change to apply.
    """
    expected_shapes = ((SPEED_DIM,), LIDAR_SHAPE, (ACTION_DIM,), (ACTION_DIM,))
    hint = (
        'Set "RTGYM_INTERFACE": "TM20LIDAR" in the "ENV" section of '
        "%USERPROFILE%/TmrlData/config/config.json (see docs/TRACKMANIA.md)."
    )
    if not isinstance(observation_space, spaces.Tuple):
        raise ValueError(
            f"Expected a Tuple observation space (LIDAR preset), got "
            f"{type(observation_space).__name__}. {hint}"
        )
    shapes = tuple(getattr(space, "shape", None) for space in observation_space.spaces)
    if shapes != expected_shapes:
        raise ValueError(f"Expected observation shapes {expected_shapes}, got {shapes}. {hint}")
    if not isinstance(action_space, spaces.Box) or action_space.shape != (ACTION_DIM,):
        raise ValueError(
            f"Expected a Box(({ACTION_DIM},)) action space, got {action_space}. {hint}"
        )


class TrackManiaEnvWrapper(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Flatten and normalize tmrl's tuple observations into a single ``Box``.

    The tuple observation ``(speed, lidar, prev_action, prev_prev_action)`` is
    concatenated into a ``(83,)`` float32 vector:

    - ``speed / max_speed`` clipped to ``[0, 1]`` (1 value),
    - ``lidar.ravel() / max_lidar`` clipped to ``[0, 1]`` (76 values),
    - previous action, unchanged (3 values in ``[-1, 1]``),
    - previous-previous action, unchanged (3 values in ``[-1, 1]``).

    Actions are ``[gas, brake, steer]`` in ``[-1, 1]``; incoming actions are
    clipped to that range before being forwarded to the underlying environment.
    Reward, termination flags and info dictionaries pass through untouched.

    Args:
        env: Underlying tmrl-like environment satisfying :class:`TMEnvProtocol`.
        max_speed: Speed normalization constant (game speed units).
        max_lidar: LIDAR distance normalization constant (pixels).
    """

    def __init__(self, env: TMEnvProtocol, max_speed: float = 1000.0, max_lidar: float = 400.0):
        super().__init__()
        validate_tmrl_spaces(env.observation_space, env.action_space)
        self._env = env
        self._max_speed = max_speed
        self._max_lidar = max_lidar
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset the underlying environment and flatten its observation.

        Args:
            seed: Optional seed forwarded to the underlying environment.
            options: Optional options dictionary forwarded as-is.

        Returns:
            Tuple ``(flattened observation, info)``.
        """
        super().reset(seed=seed)
        obs, info = self._env.reset(seed=seed, options=options)
        return self._flatten(obs), info

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:
        """Clip ``action`` to ``[-1, 1]``, forward it, and flatten the observation.

        Args:
            action: ``[gas, brake, steer]`` array; values outside ``[-1, 1]``
                are clipped before reaching the underlying environment.

        Returns:
            Tuple ``(obs, reward, terminated, truncated, info)`` where only the
            observation is transformed; everything else passes through.
        """
        clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        obs, reward, terminated, truncated, info = self._env.step(clipped)
        return self._flatten(obs), reward, terminated, truncated, info

    def wait(self) -> None:
        """Release real-time control of the underlying environment.

        rtgym keeps applying the last action in real time between ``step``
        calls; its environments expose ``wait()`` to signal that the caller is
        pausing (end of training, long computation) so the car is released
        instead of replaying the last action forever. Environments without a
        ``wait`` method (e.g. test fakes) make this a no-op.
        """
        waiter = getattr(self._env, "wait", None)
        if waiter is None:
            waiter = getattr(getattr(self._env, "unwrapped", self._env), "wait", None)
        if callable(waiter):
            waiter()

    def close(self) -> None:
        """Close the underlying environment if it supports closing."""
        closer = getattr(self._env, "close", None)
        if callable(closer):
            closer()

    def _flatten(self, obs: tuple[Any, ...]) -> NDArray[np.float32]:
        """Concatenate and normalize a tmrl LIDAR tuple observation.

        Args:
            obs: Tuple ``(speed, lidar, prev_action, prev_prev_action)`` with
                shapes ``((1,), (4, 19), (3,), (3,))``.

        Returns:
            Normalized float32 vector of shape ``(83,)``.

        Raises:
            ValueError: If the observation does not match the LIDAR layout
                (e.g. tmrl is configured with camera images instead of LIDAR).
        """
        if len(obs) != 4:
            raise ValueError(
                f"Expected a 4-component LIDAR observation, got {len(obs)} components. "
                "Check that tmrl uses the LIDAR interface (~/TmrlData/config/config.json)."
            )
        speed, lidar, prev_action, prev_prev_action = obs
        parts = [
            np.clip(np.asarray(speed, dtype=np.float32).ravel() / self._max_speed, 0.0, 1.0),
            np.clip(np.asarray(lidar, dtype=np.float32).ravel() / self._max_lidar, 0.0, 1.0),
            np.asarray(prev_action, dtype=np.float32).ravel(),
            np.asarray(prev_prev_action, dtype=np.float32).ravel(),
        ]
        flat: NDArray[np.float32] = np.concatenate(parts).astype(np.float32)
        if flat.shape != (OBS_DIM,):
            raise ValueError(
                f"Flattened observation has shape {flat.shape}, expected ({OBS_DIM},). "
                "Check that tmrl uses the default LIDAR interface with 4 stacked scans."
            )
        return flat


def make_trackmania_env() -> TrackManiaEnvWrapper:
    """Build the wrapped TrackMania environment from a live tmrl instance.

    ``tmrl`` is imported lazily so that this module stays importable (and
    testable) on machines without the game.

    Returns:
        The tmrl environment wrapped in :class:`TrackManiaEnvWrapper`.

    Raises:
        ImportError: If ``tmrl`` is not installed, with setup instructions.
    """
    try:
        import tmrl
    except ImportError as exc:
        raise ImportError(
            "tmrl is not installed. The TrackMania environment only runs on a Windows "
            "machine with TrackMania 2020, OpenPlanet and `pip install tmrl`. "
            "See docs/TRACKMANIA.md for the full setup guide."
        ) from exc
    return TrackManiaEnvWrapper(tmrl.get_environment())
