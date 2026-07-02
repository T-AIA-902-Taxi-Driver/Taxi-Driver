"""Shared pytest fixtures."""

from dataclasses import dataclass

import numpy as np
import pytest

from src.config import Config


@dataclass(frozen=True)
class FakeEnvSpec:
    """Minimal stand-in exposing state/action counts (no gymnasium needed)."""

    n_states: int = 10
    n_actions: int = 4


@pytest.fixture
def fake_env() -> FakeEnvSpec:
    return FakeEnvSpec()


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(123)
