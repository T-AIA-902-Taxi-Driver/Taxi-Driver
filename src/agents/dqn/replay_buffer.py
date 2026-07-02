"""Ring-buffer experience replay backed by preallocated numpy arrays.

A structure-of-arrays layout (one preallocated array per field) beats a deque
of transition tuples on both memory (no per-tuple Python object overhead) and
sampling speed (one fancy-index gather per field instead of unpacking
``batch_size`` tuples).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Batch = tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float32],
    npt.NDArray[np.int64],
    npt.NDArray[np.bool_],
]


class ReplayBuffer:
    """Fixed-capacity FIFO transition store with uniform random sampling.

    Once full, the oldest transition is overwritten first (ring semantics).
    Sampling is uniform with replacement over the filled region only.
    """

    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        """Preallocate storage.

        Args:
            capacity: Maximum number of transitions retained.
            rng: Random generator used for sampling (owned by the agent).
        """
        self.capacity = capacity
        self.rng = rng
        self._states: npt.NDArray[np.int64] = np.zeros(capacity, dtype=np.int64)
        self._actions: npt.NDArray[np.int64] = np.zeros(capacity, dtype=np.int64)
        self._rewards: npt.NDArray[np.float32] = np.zeros(capacity, dtype=np.float32)
        self._next_states: npt.NDArray[np.int64] = np.zeros(capacity, dtype=np.int64)
        self._terminated: npt.NDArray[np.bool_] = np.zeros(capacity, dtype=np.bool_)
        self._pos = 0
        self._size = 0

    def push(
        self, state: int, action: int, reward: float, next_state: int, terminated: bool
    ) -> None:
        """Store one transition, overwriting the oldest one when full.

        Args:
            state: State before the action.
            action: Action taken.
            reward: Reward received (possibly shaped).
            next_state: Successor state.
            terminated: True only for true MDP termination (never truncation).
        """
        i = self._pos
        self._states[i] = state
        self._actions[i] = action
        self._rewards[i] = reward
        self._next_states[i] = next_state
        self._terminated[i] = terminated
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        """Sample ``batch_size`` transitions uniformly from the filled region.

        Args:
            batch_size: Number of transitions to draw (with replacement).

        Returns:
            Tuple ``(states, actions, rewards, next_states, terminated)`` of
            numpy arrays, each of length ``batch_size``.
        """
        indices = self.rng.integers(self._size, size=batch_size)
        return (
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._terminated[indices],
        )

    def __len__(self) -> int:
        """Number of transitions currently stored."""
        return self._size

    def nbytes(self) -> int:
        """Total size of the preallocated storage arrays, in bytes."""
        return int(
            self._states.nbytes
            + self._actions.nbytes
            + self._rewards.nbytes
            + self._next_states.nbytes
            + self._terminated.nbytes
        )
