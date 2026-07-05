"""Fully-connected Q-network mapping one-hot encoded states to action values."""

from __future__ import annotations

from torch import Tensor, nn


class QNetwork(nn.Module):
    """MLP approximating ``Q(s, ·)`` from a one-hot state encoding.

    Architecture: ``Linear(n_states, hidden) → ReLU → Linear(hidden, hidden)
    → ReLU → Linear(hidden, n_actions)``. Discrete Taxi states carry no
    metric structure, so the one-hot input keeps the network from inventing
    spurious distances between state indices.
    """

    def __init__(self, n_states: int, n_actions: int, hidden_size: int = 128) -> None:
        """Build the network.

        Args:
            n_states: Input width (one-hot state dimension).
            n_actions: Output width (one Q-value per action).
            hidden_size: Width of the two hidden layers.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute Q-values for a batch of one-hot states.

        Args:
            x: Float tensor of shape ``(batch, n_states)``.

        Returns:
            Float tensor of shape ``(batch, n_actions)``.
        """
        out: Tensor = self.net(x)
        return out
