"""Generic episodic training loop (Template Method pattern)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from src.training.callbacks import Callback, StopTraining

if TYPE_CHECKING:
    import pandas as pd

    from src.agents.base_agent import BaseAgent
    from src.config import Config
    from src.environments.taxi_wrapper import TaxiEnvWrapper


@dataclass
class EpisodeResult:
    """Outcome of one training episode."""

    reward: float  # possibly shaped (what the agent optimised)
    raw_reward: float  # native environment reward (what we report)
    steps: int
    terminated: bool
    truncated: bool
    illegal_actions: int


@dataclass
class TrainingHistory:
    """Per-episode training metrics plus periodic greedy-probe measurements.

    ``rewards`` are native env rewards; ``shaped_rewards`` differ only when
    reward shaping is active. Probe entries exist when ``eval_interval > 0``.
    """

    rewards: list[float] = field(default_factory=list)
    shaped_rewards: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    illegal_actions: list[int] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    probe_episodes: list[int] = field(default_factory=list)
    probe_rewards: list[float] = field(default_factory=list)
    probe_steps: list[float] = field(default_factory=list)
    wall_time: float = 0.0
    stop_reason: str = "completed"

    def mean_reward(self, last_n: int = 100) -> float:
        """Mean native reward over the last ``last_n`` episodes."""
        if not self.rewards:
            return 0.0
        window = self.rewards[-last_n:]
        return sum(window) / len(window)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_dataframe(self) -> pd.DataFrame:
        """Per-episode metrics as a DataFrame (probes excluded)."""
        import pandas as pd

        return pd.DataFrame(
            {
                "episode": range(len(self.rewards)),
                "reward": self.rewards,
                "shaped_reward": self.shaped_rewards,
                "steps": self.steps,
                "epsilon": self.epsilons,
                "illegal_actions": self.illegal_actions,
                "terminated": self.terminated,
            }
        )


class Trainer:
    """Runs the episodic RL loop for any BaseAgent on any wrapped env.

    Contract highlights:

    - ``agent.learn`` receives ``terminated`` only — truncation bootstraps.
    - ``agent.end_episode()`` runs after every episode (schedules, buffers).
    - Native rewards are read from ``info["raw_reward"]`` so shaping never
      leaks into reported metrics.
    - When ``config.eval_interval > 0``, a greedy probe evaluation runs on
      ``probe_env`` (a separate env instance with dedicated seeds) every
      ``eval_interval`` episodes, measuring true policy quality independently
      of the exploration schedule.
    """

    def __init__(
        self,
        agent: BaseAgent,
        env: TaxiEnvWrapper,
        config: Config,
        callbacks: list[Callback] | None = None,
        probe_env: TaxiEnvWrapper | None = None,
    ) -> None:
        self.agent = agent
        self.env = env
        self.config = config
        self.callbacks = callbacks or []
        self.probe_env = probe_env

    def train(self, n_episodes: int) -> TrainingHistory:
        """Train for at most ``n_episodes`` episodes.

        Returns:
            The full TrainingHistory; ``stop_reason`` records early stops.
        """
        history = TrainingHistory()
        start = time.perf_counter()
        for callback in self.callbacks:
            callback.on_training_start(n_episodes)
        try:
            for episode in range(n_episodes):
                result = self._run_episode()
                history.rewards.append(result.raw_reward)
                history.shaped_rewards.append(result.reward)
                history.steps.append(result.steps)
                history.epsilons.append(self.agent.exploration_level)
                history.illegal_actions.append(result.illegal_actions)
                history.terminated.append(result.terminated)
                if (
                    self.config.eval_interval > 0
                    and self.probe_env is not None
                    and (episode + 1) % self.config.eval_interval == 0
                ):
                    self._run_probe(episode, history)
                for callback in self.callbacks:
                    callback.on_episode_end(episode, result)
        except StopTraining as stop:
            history.stop_reason = str(stop)
        history.wall_time = time.perf_counter() - start
        for callback in self.callbacks:
            callback.on_training_end(history)
        return history

    def _run_episode(self) -> EpisodeResult:
        state, _ = self.env.reset()
        total_reward = 0.0
        total_raw = 0.0
        illegal = 0
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = self.agent.select_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            raw_reward = float(info["raw_reward"])
            self.agent.learn(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward
            total_raw += raw_reward
            illegal += raw_reward == -10.0
            steps += 1
        self.agent.end_episode()
        return EpisodeResult(
            reward=total_reward,
            raw_reward=total_raw,
            steps=steps,
            terminated=terminated,
            truncated=truncated,
            illegal_actions=illegal,
        )

    def _run_probe(self, episode: int, history: TrainingHistory) -> None:
        from src.evaluation.evaluator import Evaluator
        from src.utils.seeding import probe_seeds

        assert self.probe_env is not None
        evaluator = Evaluator(
            self.probe_env,
            seeds=probe_seeds(self.config.seed, self.config.n_probe_episodes),
        )
        results = evaluator.evaluate(self.agent, self.config.n_probe_episodes)
        history.probe_episodes.append(episode + 1)
        history.probe_rewards.append(results.mean_reward)
        history.probe_steps.append(results.mean_steps)
