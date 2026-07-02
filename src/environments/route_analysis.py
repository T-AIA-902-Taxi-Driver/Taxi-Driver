"""Route-order analysis for the multi-passenger environment (T-2.2.3).

Brute-force enumeration of pickup/dropoff interleavings to compute the
minimal-distance route for a state, and a greedy nearest-passenger baseline.
Distances are Manhattan distances that **ignore walls**, so route lengths are
a heuristic lower bound on the true step counts. With two passengers there
are at most six valid interleavings (pickups before their dropoffs; taxi
capacity 2 never binds), so enumeration is exact and instant.

The report uses these to analyse the learned policy's route optimisation:
does the trained agent pick the nearest passenger first, and does it match
the enumerated optimal visit order?
"""

from __future__ import annotations

from itertools import permutations
from operator import itemgetter

from src.environments.multi_passenger_env import LOCATIONS, N_LOCATIONS, STATUS_IN_TAXI, decode

Cell = tuple[int, int]
#: One route event: ``(passenger_index, phase, cell)``.
Event = tuple[int, int, Cell]

PICKUP = 0
DROPOFF = 1


def optimal_pickup_order(state: int) -> tuple[int, ...]:
    """Pickup order of a minimal-Manhattan-distance route for ``state``.

    All valid event interleavings (each pickup before its dropoff; capacity 2
    allows any interleaving) are enumerated and the cheapest route is kept.
    Ties favour the lowest passenger index first (deterministic enumeration
    order). Distances ignore walls (heuristic lower bound).

    Args:
        state: Encoded state; passengers already in the taxi contribute only
            their dropoff, delivered passengers contribute nothing.

    Returns:
        Passenger indices in optimal pickup order (may be empty or length 1).
    """
    _, sequence = _best_route(state)
    return tuple(passenger for passenger, phase, _ in sequence if phase == PICKUP)


def greedy_nearest_order(state: int) -> tuple[int, ...]:
    """Pickup order from repeatedly driving to the nearest waiting passenger.

    Starting from the taxi cell, the nearest waiting passenger (ties: lowest
    index) is visited, then the next nearest from there, and so on. This is
    the myopic baseline the report compares against the enumerated optimum.

    Args:
        state: Encoded state; only waiting passengers (status 0-3) count.

    Returns:
        Passenger indices in greedy pickup order (may be empty or length 1).
    """
    taxi_row, taxi_col, status_0, status_1, _, _ = decode(state)
    position: Cell = (taxi_row, taxi_col)
    waiting = {
        i: LOCATIONS[status]
        for i, status in enumerate((status_0, status_1))
        if status < N_LOCATIONS
    }
    order: list[int] = []
    while waiting:
        _, nearest = min((_manhattan(position, cell), i) for i, cell in waiting.items())
        order.append(nearest)
        position = waiting.pop(nearest)
    return tuple(order)


def min_route_length(state: int) -> int:
    """Minimal Manhattan route length for ``state`` (walls ignored).

    An optimistic lower bound on the steps an optimal policy needs from
    ``state`` (pickup/dropoff actions are not counted, only moves).
    """
    cost, _ = _best_route(state)
    return cost


# ---------------------------------------------------------------------- internals


def _manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _remaining_events(state: int) -> tuple[Cell, list[Event]]:
    """Taxi cell and the pickup/dropoff events still to perform in ``state``."""
    taxi_row, taxi_col, status_0, status_1, dest_0, dest_1 = decode(state)
    events: list[Event] = []
    for i, (status, dest) in enumerate(((status_0, dest_0), (status_1, dest_1))):
        if status < N_LOCATIONS:
            events.append((i, PICKUP, LOCATIONS[status]))
            events.append((i, DROPOFF, LOCATIONS[dest]))
        elif status == STATUS_IN_TAXI:
            events.append((i, DROPOFF, LOCATIONS[dest]))
    return (taxi_row, taxi_col), events


def _is_valid(sequence: tuple[Event, ...]) -> bool:
    """True when every passenger's pickup precedes their dropoff."""
    needs_pickup = {passenger for passenger, phase, _ in sequence if phase == PICKUP}
    picked: set[int] = set()
    for passenger, phase, _ in sequence:
        if phase == PICKUP:
            picked.add(passenger)
        elif passenger in needs_pickup and passenger not in picked:
            return False
    return True


def _route_length(start: Cell, sequence: tuple[Event, ...]) -> int:
    total = 0
    position = start
    for _, _, cell in sequence:
        total += _manhattan(position, cell)
        position = cell
    return total


def _best_route(state: int) -> tuple[int, tuple[Event, ...]]:
    """Cheapest valid event sequence for ``state`` (first minimum wins)."""
    start, events = _remaining_events(state)
    candidates = (
        (_route_length(start, sequence), sequence)
        for sequence in permutations(events)
        if _is_valid(sequence)
    )
    # min() is stable: among equal costs the earliest enumerated sequence wins,
    # which (events being listed passenger-0 first) favours the lowest index.
    return min(candidates, key=itemgetter(0))
