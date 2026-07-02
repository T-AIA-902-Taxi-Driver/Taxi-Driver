"""Two-passenger extension of Taxi-v3 with route optimisation (bonus extension).

The subject's bonus asks for a taxi that "collects 2 people and obtain 4
locations for each of them, the goal being to optimize the route". This module
provides:

- :class:`MultiPassengerTaxiEnv`, a deterministic Gymnasium environment on the
  exact Taxi-v3 5x5 map with two independent passengers;
- module-level :func:`encode` / :func:`decode` for the mixed-radix state
  encoding (14,400 states);
- :class:`MultiEnvWrapper`, the project-facing wrapper mirroring
  :class:`~src.environments.taxi_wrapper.TaxiEnvWrapper`'s public surface so
  the Trainer and Evaluator work unchanged.

State-count note: docs/CADRAGE.md estimated 25 x 5^2 x 4^2 = 10,000 states
(5 statuses per passenger, as in Taxi-v3). That is incorrect here: delivering
the first passenger does *not* end the episode, so "delivered" must be a
distinct status from the four landmarks and "in taxi". Each passenger thus has
6 statuses and the state space is 25 x 6^2 x 4^2 = **14,400** states.
"""

from __future__ import annotations

from typing import Any, SupportsInt, cast

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Space
from gymnasium.wrappers import TimeLimit

from src.environments.taxi_wrapper import RewardFn

# Same ascii map as Taxi-v3 (gymnasium.envs.toy_text.taxi.MAP). Walls are
# parsed the same way: the "|" between two columns blocks East/West moves,
# and the grid borders clamp every move.
MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

#: Grid coordinates of the four landmarks, in index order R, G, Y, B
#: (mirrors ``TaxiEnv.locs`` and ``TAXI_LOCATIONS``).
LOCATIONS: tuple[tuple[int, int], ...] = ((0, 0), (0, 4), (4, 0), (4, 3))
LOCATION_NAMES: tuple[str, ...] = ("R", "G", "Y", "B")

N_ROWS = 5
N_COLS = 5
N_LOCATIONS = 4
N_PASSENGERS = 2

#: Passenger statuses: 0-3 = waiting at landmark R/G/Y/B, 4 = in taxi,
#: 5 = delivered (distinct from "in taxi": the episode continues after the
#: first delivery, unlike single-passenger Taxi-v3).
STATUS_IN_TAXI = 4
STATUS_DELIVERED = 5
N_STATUSES = 6

#: 25 taxi cells x 6^2 passenger statuses x 4^2 destinations = 14,400.
N_STATES = N_ROWS * N_COLS * N_STATUSES**2 * N_LOCATIONS**2
N_ACTIONS = 6

#: Episode step cap applied by ``create_env`` via ``TimeLimit`` (500, not
#: Taxi-v3's 200: optimal two-passenger routes are roughly twice as long and
#: early training wanders a lot). The env itself has no internal step cap.
MULTI_MAX_STEPS = 500

MOVE_REWARD = -1.0
ILLEGAL_REWARD = -10.0
DELIVER_REWARD = 20.0


def encode(
    taxi_row: int, taxi_col: int, status_0: int, status_1: int, dest_0: int, dest_1: int
) -> int:
    """Encode state components into a single integer (mixed radix).

    Most significant first: ``((((row*5 + col)*6 + s0)*6 + s1)*4 + d0)*4 + d1``.

    Args:
        taxi_row: Taxi row in ``[0, 5)``.
        taxi_col: Taxi column in ``[0, 5)``.
        status_0: Passenger 0 status in ``[0, 6)``.
        status_1: Passenger 1 status in ``[0, 6)``.
        dest_0: Passenger 0 destination landmark in ``[0, 4)``.
        dest_1: Passenger 1 destination landmark in ``[0, 4)``.

    Returns:
        Encoded state in ``[0, 14_400)``.
    """
    state = taxi_row * N_COLS + taxi_col
    state = state * N_STATUSES + status_0
    state = state * N_STATUSES + status_1
    state = state * N_LOCATIONS + dest_0
    state = state * N_LOCATIONS + dest_1
    return state


def decode(state: int) -> tuple[int, int, int, int, int, int]:
    """Decode an encoded state into its components (inverse of :func:`encode`).

    Args:
        state: Encoded state in ``[0, 14_400)``.

    Returns:
        Tuple ``(taxi_row, taxi_col, status_0, status_1, dest_0, dest_1)``.
    """
    state, dest_1 = divmod(state, N_LOCATIONS)
    state, dest_0 = divmod(state, N_LOCATIONS)
    state, status_1 = divmod(state, N_STATUSES)
    state, status_0 = divmod(state, N_STATUSES)
    taxi_row, taxi_col = divmod(state, N_COLS)
    return taxi_row, taxi_col, status_0, status_1, dest_0, dest_1


class MultiPassengerTaxiEnv(gym.Env[int, int]):
    """Deterministic two-passenger taxi on the Taxi-v3 5x5 map.

    Actions (same order as Taxi-v3): 0 South, 1 North, 2 East, 3 West,
    4 Pickup, 5 Dropoff. Moves cost -1; walls and borders block the move
    (position unchanged, still -1). Contextual actions are deterministic with
    a lowest-index-wins rule when both passengers are eligible:

    - Pickup: a waiting passenger on the taxi cell boards (status -> 4) for
      -1; with no eligible passenger the action is illegal (-10). Capacity is
      2: both passengers may ride simultaneously, which is required for
      genuine route optimisation.
    - Dropoff: on a landmark that is the destination of an in-taxi passenger,
      that passenger is delivered (status -> 5) for +20; anything else is
      illegal (-10). **Deliberate divergence from Taxi-v3**: mid-episode
      set-down (dropping an in-taxi passenger anywhere other than their
      destination) is forbidden and penalised -10 — the passenger stays in
      the taxi — whereas Taxi-v3 re-places the passenger on the landmark for
      a plain -1.

    The episode terminates when both passengers are delivered. The env has no
    internal step cap; ``create_env`` wraps it in ``TimeLimit(500)``.

    Initial states sample the taxi cell uniformly over the 25 cells, two
    *distinct* start landmarks, and per-passenger destinations different from
    that passenger's start (the two destinations may coincide).

    The state space is ``Discrete(14_400)`` = 25 taxi cells x 6 statuses per
    passenger x 4 destinations per passenger (see module docstring for why
    "delivered" is a sixth status, correcting docs/CADRAGE.md's 10,000).
    """

    # RUF012 suppressed: gymnasium's Env declares ``metadata`` as a plain
    # mutable class attribute, so a ClassVar annotation would clash with it.
    metadata = {"render_modes": ["ansi"], "render_fps": 4}  # noqa: RUF012

    def __init__(self, render_mode: str | None = "ansi") -> None:
        """Create the environment.

        Args:
            render_mode: Only ``"ansi"`` (string frames) is supported.
        """
        self.render_mode = render_mode
        self.desc = np.asarray(MAP, dtype="c")
        # cast: Discrete is Space[np.int64] in gymnasium's stubs while this
        # env is typed over built-in int observations/actions.
        self.observation_space = cast("Space[int]", Discrete(N_STATES))
        self.action_space = cast("Space[int]", Discrete(N_ACTIONS))
        # Deterministic placeholder state; reset() randomises it.
        self._taxi_row = 0
        self._taxi_col = 0
        self._statuses = [0, 1]
        self._dests = [1, 0]

    # ------------------------------------------------------------------ properties

    @property
    def n_states(self) -> int:
        """Number of discrete states (14,400)."""
        return N_STATES

    @property
    def n_actions(self) -> int:
        """Number of discrete actions (6)."""
        return N_ACTIONS

    # ------------------------------------------------------------------ core API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[int, dict[str, Any]]:
        """Sample a new initial state.

        Taxi cell: uniform over the 25 cells. Passenger starts: two distinct
        landmarks. Destinations: ``dest_i != start_i`` for each passenger
        (destinations may coincide with each other).

        Args:
            seed: Seed for ``self.np_random`` (Gymnasium seeding semantics).
            options: Unused (Gymnasium API compatibility).

        Returns:
            Tuple ``(state, info)``; ``info["decoded"]`` holds the decoded
            component tuple for debugging.
        """
        super().reset(seed=seed)
        self._taxi_row = int(self.np_random.integers(N_ROWS))
        self._taxi_col = int(self.np_random.integers(N_COLS))
        starts = self.np_random.choice(N_LOCATIONS, size=N_PASSENGERS, replace=False)
        self._statuses = [int(start) for start in starts]
        self._dests = [self._random_destination(start) for start in self._statuses]
        state = self._encoded()
        return state, {"decoded": decode(state)}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Apply one action (deterministic transition).

        Args:
            action: 0 South, 1 North, 2 East, 3 West, 4 Pickup, 5 Dropoff.

        Returns:
            Tuple ``(state, reward, terminated, truncated, info)`` with
            ``truncated`` always False (time limits are applied by wrappers)
            and ``info["raw_reward"]`` always set to the native reward.
        """
        if action == 0:  # South
            self._taxi_row = min(self._taxi_row + 1, N_ROWS - 1)
            reward = MOVE_REWARD
        elif action == 1:  # North
            self._taxi_row = max(self._taxi_row - 1, 0)
            reward = MOVE_REWARD
        elif action == 2:  # East
            if self.desc[1 + self._taxi_row, 2 * self._taxi_col + 2] == b":":
                self._taxi_col += 1
            reward = MOVE_REWARD
        elif action == 3:  # West
            if self.desc[1 + self._taxi_row, 2 * self._taxi_col] == b":":
                self._taxi_col -= 1
            reward = MOVE_REWARD
        elif action == 4:  # Pickup
            reward = self._pickup()
        elif action == 5:  # Dropoff
            reward = self._dropoff()
        else:
            raise ValueError(f"invalid action {action} (expected 0-5)")
        terminated = all(status == STATUS_DELIVERED for status in self._statuses)
        state = self._encoded()
        info: dict[str, Any] = {"decoded": decode(state), "raw_reward": reward}
        return state, reward, terminated, False, info

    # type ignore: gymnasium types ``Env.render`` with an unbound ``RenderFrame``
    # TypeVar, which a concrete ``str`` return cannot satisfy under mypy strict.
    def render(self) -> str | None:  # type: ignore[override]
        """Render the grid in Taxi-v3 style plus a passenger legend.

        The taxi is drawn as ``T``, waiting passengers as ``1``/``2`` on
        their cells; statuses and destinations are summarised in a final
        legend line (e.g. ``P1: in taxi -> G | P2: at Y -> B``).
        """
        if self.render_mode != "ansi":
            return None
        grid = [list(line) for line in MAP]
        for i, status in enumerate(self._statuses):
            if status < N_LOCATIONS:
                row, col = LOCATIONS[status]
                grid[1 + row][2 * col + 1] = str(i + 1)
        grid[1 + self._taxi_row][2 * self._taxi_col + 1] = "T"
        legend = " | ".join(
            f"P{i + 1}: {self._status_text(status)} -> {LOCATION_NAMES[dest]}"
            for i, (status, dest) in enumerate(zip(self._statuses, self._dests, strict=True))
        )
        return "\n".join("".join(row) for row in grid) + "\n" + legend + "\n"

    # ------------------------------------------------------------------ helpers

    def decode_state(self, state: int) -> tuple[int, int, int, int, int, int]:
        """Alias of module-level :func:`decode` (parity with TaxiEnvWrapper)."""
        return decode(state)

    def _encoded(self) -> int:
        return encode(
            self._taxi_row,
            self._taxi_col,
            self._statuses[0],
            self._statuses[1],
            self._dests[0],
            self._dests[1],
        )

    def _set_state(
        self, taxi_row: int, taxi_col: int, status_0: int, status_1: int, dest_0: int, dest_1: int
    ) -> int:
        """Force the internal state (test helper for scripted scenarios).

        Returns:
            The resulting encoded state.
        """
        self._taxi_row = taxi_row
        self._taxi_col = taxi_col
        self._statuses = [status_0, status_1]
        self._dests = [dest_0, dest_1]
        return self._encoded()

    def _random_destination(self, start: int) -> int:
        """Sample a destination uniformly over the three landmarks != start."""
        dest = int(self.np_random.integers(N_LOCATIONS - 1))
        return dest + 1 if dest >= start else dest

    def _pickup(self) -> float:
        """Board the lowest-index waiting passenger on the taxi cell, if any."""
        cell = (self._taxi_row, self._taxi_col)
        for i, status in enumerate(self._statuses):
            if status < N_LOCATIONS and LOCATIONS[status] == cell:
                self._statuses[i] = STATUS_IN_TAXI
                return MOVE_REWARD
        return ILLEGAL_REWARD

    def _dropoff(self) -> float:
        """Deliver the lowest-index in-taxi passenger destined here, if any."""
        cell = (self._taxi_row, self._taxi_col)
        if cell in LOCATIONS:
            landmark = LOCATIONS.index(cell)
            for i, status in enumerate(self._statuses):
                if status == STATUS_IN_TAXI and self._dests[i] == landmark:
                    self._statuses[i] = STATUS_DELIVERED
                    return DELIVER_REWARD
        return ILLEGAL_REWARD

    @staticmethod
    def _status_text(status: int) -> str:
        if status == STATUS_IN_TAXI:
            return "in taxi"
        if status == STATUS_DELIVERED:
            return "delivered"
        return f"at {LOCATION_NAMES[status]}"


class MultiEnvWrapper:
    """Project-facing wrapper around a TimeLimit-wrapped MultiPassengerTaxiEnv.

    Mirrors :class:`~src.environments.taxi_wrapper.TaxiEnvWrapper`'s public
    surface — ``reset(seed)``/``step``/``render``/``close``, ``n_states``/
    ``n_actions``/``decode_state`` and ``info["raw_reward"]`` — so the
    Trainer, Evaluator and agent factory work unchanged. A dedicated wrapper
    is required because gymnasium 1.x removed ``Wrapper.__getattr__``
    delegation: ``TimeLimit(env).n_states`` raises AttributeError (only
    ``observation_space``/``action_space``/``unwrapped`` pass through), so
    custom attributes must be surfaced explicitly.

    Seeding follows the same convention as TaxiEnvWrapper: the constructor
    seed is applied on the first reset only; an explicit ``reset(seed=...)``
    always re-seeds; bare resets continue the RNG stream.

    Attributes:
        env: The underlying Gymnasium environment (TimeLimit-wrapped).
        reward_fn: Optional reward-shaping function applied on each step.
    """

    def __init__(
        self,
        reward_fn: RewardFn | None = None,
        seed: int | None = None,
        render_mode: str = "ansi",
        max_steps: int = MULTI_MAX_STEPS,
    ) -> None:
        """Create the wrapped environment.

        Args:
            reward_fn: Optional shaping function ``(state, action, raw_reward,
                next_state) -> shaped_reward``. When None, rewards pass through.
            seed: Seed applied on the first :meth:`reset` only.
            render_mode: Gymnasium render mode (``"ansi"`` yields strings).
            max_steps: Episode step limit enforced via ``TimeLimit`` (500 by
                default: two-passenger routes are ~2x Taxi-v3's, plus
                early-training wandering).
        """
        self.env: gym.Env[int, int] = TimeLimit(
            MultiPassengerTaxiEnv(render_mode=render_mode), max_episode_steps=max_steps
        )
        self.reward_fn = reward_fn
        self._seed = seed
        self._seeded = False
        self._state = 0

    # ------------------------------------------------------------------ properties

    @property
    def n_states(self) -> int:
        """Number of discrete states (14,400)."""
        space = self.env.observation_space
        assert isinstance(space, Discrete)
        # cast: numpy stubs make mypy infer ``Discrete.n`` as ``Any | np.void``.
        return int(cast(SupportsInt, space.n))

    @property
    def n_actions(self) -> int:
        """Number of discrete actions (6)."""
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

    def decode_state(self, state: int) -> tuple[int, int, int, int, int, int]:
        """Decode an encoded state into its components.

        Args:
            state: Encoded state in ``[0, n_states)``.

        Returns:
            Tuple ``(taxi_row, taxi_col, status_0, status_1, dest_0, dest_1)``.
        """
        return decode(state)
