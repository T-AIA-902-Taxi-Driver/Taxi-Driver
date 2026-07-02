"""Deep Q-Network agent (Mnih et al., 2015) with optional Double DQN targets.

Design choices for the tiny discrete Taxi state space:

- one-hot state encoding into a small MLP (see ``QNetwork``);
- soft target updates (Polyak averaging, ``tau``) every training step instead
  of periodic hard copies;
- Huber loss + gradient-norm clipping for robustness to the -10 penalty
  outliers;
- the exploration strategy is shared with the tabular agents: it only ever
  sees a numpy row of Q-values, so ε-greedy, Boltzmann and UCB all work
  unchanged on top of the network.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F  # noqa: N812 — torch's canonical alias
from torch import Tensor, nn

from src.agents.base_agent import BaseAgent
from src.agents.dqn.q_network import QNetwork
from src.agents.dqn.replay_buffer import ReplayBuffer
from src.agents.exploration import create_exploration
from src.config import Config

FloatArray = npt.NDArray[np.float64]


class DQNAgent(BaseAgent):
    """DQN with experience replay, target network and optional Double DQN.

    Standard target: ``r + γ · max_a' Q_target(s', a') · (1 − terminated)``.
    Double DQN decouples selection from evaluation to fight maximisation
    bias: ``r + γ · Q_target(s', argmax_a' Q_policy(s', a')) · (1 − term.)``.
    """

    name = "dqn"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        config: Config,
        rng: np.random.Generator,
        n_episodes: int | None = None,
    ) -> None:
        """Build networks, optimizer, replay buffer and exploration strategy.

        Args:
            n_states: Environment state count (one-hot input width).
            n_actions: Environment action count (network output width).
            config: Project configuration (DQN block + exploration block).
            rng: Numpy generator for exploration and replay sampling.
            n_episodes: Training horizon used to resolve ``config.decay_frac``.
        """
        super().__init__(n_states, n_actions, config, rng)
        torch.manual_seed(config.seed)
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)

        self.policy_net = QNetwork(n_states, n_actions, config.hidden_size).to(self.device)
        self.target_net = QNetwork(n_states, n_actions, config.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(config.buffer_capacity, rng)
        self.strategy = create_exploration(config, n_states, n_actions, rng, n_episodes)
        self._step = 0  # global environment-step counter
        self.last_loss: float | None = None

    # ------------------------------------------------------------------ actions

    def q_values(self, state: int) -> FloatArray:
        """Row of Q-values predicted by the policy network for ``state``."""
        with torch.no_grad():
            q = self.policy_net(self._one_hot([state])).squeeze(0)
        row: FloatArray = q.cpu().numpy().astype(np.float64)
        return row

    def select_action(self, state: int, greedy: bool = False) -> int:
        q_row = self.q_values(state)
        if greedy:
            return int(np.argmax(q_row))
        return self.strategy.select(q_row, state)

    # ------------------------------------------------------------------ learning

    def learn(
        self, state: int, action: int, reward: float, next_state: int, terminated: bool
    ) -> None:
        self.buffer.push(state, action, reward, next_state, terminated)
        self._step += 1
        if self._step % self.config.train_every != 0:
            return
        if len(self.buffer) < max(self.config.warmup_steps, self.config.batch_size):
            return
        self._train_step()

    def _train_step(self) -> None:
        """One gradient step on a replayed minibatch, then a soft target update."""
        states, actions, rewards, next_states, terminated = self.buffer.sample(
            self.config.batch_size
        )
        state_batch = self._one_hot(states)
        next_batch = self._one_hot(next_states)
        action_batch = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        reward_batch = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        # (1 − terminated): truncated transitions keep bootstrapping.
        bootstrap_mask = torch.as_tensor(~terminated, dtype=torch.float32, device=self.device)

        q_pred = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.config.double_dqn:
                best_next = self.policy_net(next_batch).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_batch).gather(1, best_next).squeeze(1)
            else:
                next_q = self.target_net(next_batch).max(dim=1).values
            target = reward_batch + self.config.gamma * next_q * bootstrap_mask

        loss = self.loss_fn(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self._soft_update()
        self.last_loss = float(loss.item())

    def _soft_update(self) -> None:
        """Polyak averaging: ``target ← tau · policy + (1 − tau) · target``."""
        tau = self.config.tau
        with torch.no_grad():
            for target_param, param in zip(
                self.target_net.parameters(), self.policy_net.parameters(), strict=True
            ):
                target_param.mul_(1.0 - tau).add_(param, alpha=tau)

    def end_episode(self) -> None:
        super().end_episode()
        self.strategy.decay()

    # ------------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        """Persist networks, optimizer and metadata as a ``.pt`` checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "algorithm": self.name,
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "n_episodes_trained": self.n_episodes_trained,
            "config_hash": self.config.config_hash(),
            "config": self.config.to_dict(),
            "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step": self._step,
                "metadata": metadata,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Restore networks, optimizer and counters from a ``.pt`` checkpoint."""
        checkpoint: dict[str, Any] = torch.load(
            Path(path), map_location=self.device, weights_only=False
        )
        metadata = checkpoint["metadata"]
        saved_shape = (int(metadata["n_states"]), int(metadata["n_actions"]))
        if saved_shape != (self.n_states, self.n_actions):
            raise ValueError(
                f"Model dimensions {saved_shape} do not match environment "
                f"({self.n_states}, {self.n_actions})"
            )
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._step = int(checkpoint["step"])
        self.n_episodes_trained = int(metadata.get("n_episodes_trained", 0))

    def memory_bytes(self) -> int:
        n_params = sum(p.numel() for p in self.policy_net.parameters()) + sum(
            p.numel() for p in self.target_net.parameters()
        )
        return n_params * 4 + self.buffer.nbytes()

    @property
    def exploration_level(self) -> float:
        return self.strategy.epsilon_like

    # ------------------------------------------------------------------ helpers

    def _one_hot(self, states: npt.NDArray[np.int64] | list[int]) -> Tensor:
        """One-hot encode a batch of state indices on the agent's device."""
        indices = torch.as_tensor(np.asarray(states), dtype=torch.int64, device=self.device)
        return F.one_hot(indices, num_classes=self.n_states).float()
