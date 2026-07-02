"""Seeding utilities guaranteeing reproducible, independent random streams.

The project uses a single master seed (``Config.seed``) from which independent
streams are derived:

- one ``np.random.Generator`` per component (environment, agent, ...) via
  ``SeedSequence.spawn`` so adding a consumer never perturbs the others;
- a dedicated, fixed list of evaluation seeds so that every agent is tested
  on exactly the same episode start states;
- a distinct list of probe seeds used to measure convergence during training
  without contaminating the final evaluation set.
"""

from __future__ import annotations

import numpy as np

EVAL_SEED_OFFSET = 10_000
PROBE_SEED_OFFSET = 20_000


def spawn_rngs(seed: int, n: int) -> list[np.random.Generator]:
    """Create ``n`` independent random generators from a master seed.

    Args:
        seed: Master seed.
        n: Number of independent generators to create.

    Returns:
        List of ``np.random.Generator`` with statistically independent streams.
    """
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n)]


def eval_seeds(seed: int, n_episodes: int) -> list[int]:
    """Fixed evaluation seed list: identical test episodes for every agent.

    Args:
        seed: Master seed.
        n_episodes: Number of evaluation episodes.

    Returns:
        List of ``n_episodes`` environment seeds.
    """
    return [seed + EVAL_SEED_OFFSET + i for i in range(n_episodes)]


def probe_seeds(seed: int, n_episodes: int) -> list[int]:
    """Probe seed list, disjoint from :func:`eval_seeds` by construction.

    Args:
        seed: Master seed.
        n_episodes: Number of probe episodes (must stay < 10_000 to guarantee
            disjointness with the evaluation set).

    Returns:
        List of ``n_episodes`` environment seeds.

    Raises:
        ValueError: If ``n_episodes`` would overlap the evaluation seed range.
    """
    if n_episodes >= PROBE_SEED_OFFSET - EVAL_SEED_OFFSET:
        raise ValueError("n_episodes too large: probe seeds would overlap eval seeds")
    return [seed + PROBE_SEED_OFFSET + i for i in range(n_episodes)]
