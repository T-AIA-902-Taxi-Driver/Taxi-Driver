"""Thin Gymnasium Taxi-v3 wrapper with seeding, decoding and reward-shaping hooks.

The wrapper normalises the Gymnasium API for the rest of the project:

- observations are plain built-in ``int`` and rewards plain ``float``;
- the constructor seed is applied on the *first* reset only, following
  Gymnasium 1.x episodic seeding semantics (subsequent bare resets continue
  the RNG stream), while an explicit ``reset(seed=...)`` always re-seeds
  (used by the evaluator to replay fixed test episodes);
- an optional ``reward_fn`` shapes rewards while the native environment
  reward is always preserved in ``info["raw_reward"]`` for uniform logging.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, SupportsInt, cast

import gymnasium as gym
from gymnasium.spaces import Discrete

# Signature: (state, action, raw_reward, next_state) -> shaped_reward.
RewardFn = Callable[[int, int, float, int], float]

# Grid coordinates of the four Taxi-v3 landmarks (R, G, Y, B), mirroring
# ``TaxiEnv.locs``. Used by distance-based reward shaping.
TAXI_LOCATIONS: dict[int, tuple[int, int]] = {0: (0, 0), 1: (0, 4), 2: (4, 0), 3: (4, 3)}


class TaxiEnvWrapper:
    """Project-facing wrapper around the Gymnasium ``Taxi-v3`` environment.

    Attributes:
        env: The underlying Gymnasium environment (TimeLimit-wrapped).
        reward_fn: Optional reward-shaping function applied on each step.
    """

    def __init__(
        self,
        reward_fn: RewardFn | None = None,
        seed: int | None = None,
        env_id: str = "Taxi-v3",
        render_mode: str = "ansi",
        max_steps: int | None = None,
    ) -> None:
        """Create the wrapped environment.

        Args:
            reward_fn: Optional shaping function ``(state, action, raw_reward,
                next_state) -> shaped_reward``. When None, rewards pass through.
            seed: Seed applied on the first :meth:`reset` only.
            env_id: Gymnasium environment id.
            render_mode: Gymnasium render mode (``"ansi"`` yields strings).
            max_steps: Episode step limit. When set, it overrides the spec
                default (200 for Taxi-v3) via ``gym.make(max_episode_steps=...)``,
                avoiding a second nested TimeLimit wrapper.
        """
        make_kwargs: dict[str, Any] = {"render_mode": render_mode}
        if max_steps is not None:
            make_kwargs["max_episode_steps"] = max_steps
        self.env: gym.Env[Any, Any] = gym.make(env_id, **make_kwargs)
        self.reward_fn = reward_fn
        self._seed = seed
        self._seeded = False
        self._state = 0

    # ------------------------------------------------------------------ properties

    @property
    def n_states(self) -> int:
        """Number of discrete states (500 for Taxi-v3)."""
        space = self.env.observation_space
        assert isinstance(space, Discrete)
        # cast: numpy stubs make mypy infer ``Discrete.n`` as ``Any | np.void``.
        return int(cast(SupportsInt, space.n))

    @property
    def n_actions(self) -> int:
        """Number of discrete actions (6 for Taxi-v3)."""
        space = self.env.action_space
        assert isinstance(space, Discrete)
        return int(cast(SupportsInt, space.n))

    # ------------------------------------------------------------------ core API

    def reset(self, seed: int | None = None) -> tuple[int, dict[str, Any]]:
        """Reset the environment and return the initial state.

        Args:
            seed: Explicit episode seed. When given it is passed through
                (fixed evaluation episodes). Otherwise the constructor seed is
                used on the first reset only; later resets are bare so the
                RNG stream continues (Gymnasium 1.x seeding semantics).

        Returns:
            Tuple ``(state, info)`` with ``state`` as a built-in int.
        """
        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        elif not self._seeded:
            obs, info = self.env.reset(seed=self._seed)
        else:
            obs, info = self.env.reset()
        self._seeded = True
        self._state = int(obs)
        return self._state, info

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Take one environment step, applying reward shaping when configured.

        Args:
            action: Discrete action index.

        Returns:
            Tuple ``(next_state, reward, terminated, truncated, info)``.
            ``reward`` is the shaped reward when ``reward_fn`` is set; the
            native environment reward is always stored in
            ``info["raw_reward"]``.
        """
        state_before = self._state
        obs, reward, terminated, truncated, info = self.env.step(action)
        next_state = int(obs)
        raw_reward = float(reward)
        info["raw_reward"] = raw_reward
        shaped = (
            raw_reward
            if self.reward_fn is None
            else self.reward_fn(state_before, action, raw_reward, next_state)
        )
        self._state = next_state
        return next_state, float(shaped), bool(terminated), bool(truncated), info

    def render(self) -> str:
        """Return the current ANSI rendering of the environment."""
        frame: Any = self.env.render()
        return frame if isinstance(frame, str) else ""

    def close(self) -> None:
        """Release the underlying environment resources."""
        self.env.close()

    # ------------------------------------------------------------------ helpers

    def decode_state(self, state: int) -> tuple[int, int, int, int]:
        """Decode an encoded Taxi-v3 state into its components.

        Args:
            state: Encoded state in ``[0, n_states)``.

        Returns:
            Tuple ``(taxi_row, taxi_col, passenger_loc, destination)`` where
            ``passenger_loc`` is 0-3 for the landmarks or 4 when in the taxi,
            and ``destination`` is a landmark index 0-3.
        """
        unwrapped = cast(Any, self.env.unwrapped)
        taxi_row, taxi_col, passenger_loc, destination = (int(v) for v in unwrapped.decode(state))
        return taxi_row, taxi_col, passenger_loc, destination
