"""DQN tests: network shape, replay-buffer semantics, agent learning dynamics.

Every test forces ``device="cpu"`` so CI stays deterministic and GPU-free;
the end-to-end convergence test on the real Taxi-v3 is marked ``slow`` and
excluded by the default pytest addopts.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.agents import create_agent
from src.agents.dqn.dqn_agent import DQNAgent
from src.agents.dqn.q_network import QNetwork
from src.agents.dqn.replay_buffer import ReplayBuffer
from src.config import Config

N_STATES, N_ACTIONS = 6, 4


def tiny_config(**overrides: object) -> Config:
    """Small, CPU-forced DQN config that trains from the very first steps."""
    params: dict[str, object] = {
        "algorithm": "dqn",
        "device": "cpu",
        "hidden_size": 16,
        "warmup_steps": 4,
        "batch_size": 4,
        "buffer_capacity": 64,
        "train_every": 1,
        "seed": 3,
    }
    params.update(overrides)
    return Config(**params)  # type: ignore[arg-type]


def make_agent(config: Config | None = None, seed: int = 7) -> DQNAgent:
    return DQNAgent(N_STATES, N_ACTIONS, config or tiny_config(), np.random.default_rng(seed))


def params_equal(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    return all(torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters(), strict=True))


class TestQNetwork:
    def test_output_shape(self) -> None:
        net = QNetwork(N_STATES, N_ACTIONS, hidden_size=16)
        states = torch.tensor([0, 1, 2, 3, 4, 5, 0])
        x = torch.nn.functional.one_hot(states, num_classes=N_STATES).float()
        assert net(x).shape == (7, N_ACTIONS)


class TestReplayBuffer:
    def test_len_grows_then_caps_at_capacity(self) -> None:
        buffer = ReplayBuffer(8, np.random.default_rng(0))
        for i in range(5):
            buffer.push(i, 0, 0.0, i + 1, False)
        assert len(buffer) == 5
        for i in range(10):
            buffer.push(i, 0, 0.0, i + 1, False)
        assert len(buffer) == 8

    def test_fifo_overwrite_drops_oldest(self) -> None:
        buffer = ReplayBuffer(8, np.random.default_rng(0))
        for i in range(9):  # capacity + 1 distinct rewards
            buffer.push(i, 0, float(i), i + 1, False)
        stored = set(buffer._rewards.tolist())
        assert 0.0 not in stored  # the oldest transition was overwritten
        assert stored == {float(i) for i in range(1, 9)}

    def test_sample_shapes_and_dtypes(self) -> None:
        buffer = ReplayBuffer(16, np.random.default_rng(0))
        for i in range(10):
            buffer.push(i, i % 4, -1.5, i + 1, i % 2 == 0)
        states, actions, rewards, next_states, terminated = buffer.sample(4)
        for array in (states, actions, rewards, next_states, terminated):
            assert array.shape == (4,)
        assert states.dtype == np.int64
        assert actions.dtype == np.int64
        assert rewards.dtype == np.float32
        assert next_states.dtype == np.int64
        assert terminated.dtype == np.bool_

    def test_sample_only_over_filled_region(self) -> None:
        buffer = ReplayBuffer(64, np.random.default_rng(0))
        for state in (1, 2, 3):
            buffer.push(state, 0, 9.0, state, False)
        for _ in range(20):
            states, _, rewards, _, _ = buffer.sample(32)
            assert set(states.tolist()) <= {1, 2, 3}
            assert set(rewards.tolist()) == {9.0}


class TestDQNAgent:
    def test_learn_updates_policy_net_and_records_loss(self) -> None:
        agent = make_agent()
        before = [p.clone() for p in agent.policy_net.parameters()]
        assert agent.last_loss is None
        for i in range(10):
            agent.learn(i % N_STATES, i % N_ACTIONS, -1.0, (i + 1) % N_STATES, False)
        assert isinstance(agent.last_loss, float)
        assert any(
            not torch.equal(b, p)
            for b, p in zip(before, agent.policy_net.parameters(), strict=True)
        )

    def test_target_net_soft_updates_toward_policy(self) -> None:
        agent = make_agent()
        assert params_equal(agent.policy_net, agent.target_net)  # synced at init
        target_before = [p.clone() for p in agent.target_net.parameters()]
        for i in range(10):
            agent.learn(i % N_STATES, i % N_ACTIONS, -1.0, (i + 1) % N_STATES, False)
        # The target moved (Polyak averaging ran) ...
        assert any(
            not torch.equal(b, p)
            for b, p in zip(target_before, agent.target_net.parameters(), strict=True)
        )
        # ... but lags behind the policy (tau << 1).
        assert not params_equal(agent.policy_net, agent.target_net)

    def test_double_dqn_flag_changes_updates(self) -> None:
        agents = {
            flag: DQNAgent(
                N_STATES,
                N_ACTIONS,
                tiny_config(double_dqn=flag, lr=0.05),
                np.random.default_rng(0),
            )
            for flag in (True, False)
        }
        rng = np.random.default_rng(42)
        transitions = [
            (
                int(rng.integers(N_STATES)),
                int(rng.integers(N_ACTIONS)),
                float(rng.normal(0.0, 5.0)),
                int(rng.integers(N_STATES)),
                False,
            )
            for _ in range(40)
        ]
        for agent in agents.values():  # identical init, identical transitions
            for transition in transitions:
                agent.learn(*transition)
        assert not params_equal(agents[True].policy_net, agents[False].policy_net)

    def test_terminated_masks_bootstrap(self) -> None:
        # Q(3, ·) is trained high first; if the terminal mask were broken, the
        # target for (2, 1) would be 5 + γ·max Q(3) ≈ 24.8 instead of 5.
        config = tiny_config(lr=0.01, buffer_capacity=512, batch_size=8)
        agent = make_agent(config, seed=1)
        for _ in range(150):
            agent.learn(3, 0, 20.0, 4, True)
        for _ in range(300):
            agent.learn(2, 1, 5.0, 3, True)
        assert agent.q_values(3)[0] > 10.0  # the decoy value is in place
        assert abs(agent.q_values(2)[1] - 5.0) < 0.5  # targets equal rewards

    def test_greedy_selection_and_exploration_decay(self) -> None:
        agent = make_agent(tiny_config(epsilon=1.0, epsilon_min=0.0, epsilon_decay=0.5))
        action = agent.select_action(0, greedy=True)
        assert action == int(np.argmax(agent.q_values(0)))
        assert agent.exploration_level == 1.0
        agent.end_episode()
        assert agent.exploration_level == 0.5
        assert agent.n_episodes_trained == 1

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        agent = make_agent()
        for i in range(20):
            agent.learn(i % N_STATES, i % N_ACTIONS, -1.0, (i + 1) % N_STATES, False)
        agent.end_episode()
        path = tmp_path / "dqn.pt"
        agent.save(path)

        clone = make_agent(seed=99)
        assert not params_equal(agent.policy_net, clone.policy_net)
        clone.load(path)
        assert params_equal(agent.policy_net, clone.policy_net)
        assert params_equal(agent.target_net, clone.target_net)
        assert clone.n_episodes_trained == agent.n_episodes_trained
        assert clone._step == agent._step

    def test_load_dimension_mismatch_raises(self, tmp_path: Path) -> None:
        agent = make_agent()
        path = tmp_path / "dqn.pt"
        agent.save(path)
        other = DQNAgent(N_STATES + 1, N_ACTIONS, tiny_config(), np.random.default_rng(0))
        with pytest.raises(ValueError, match="dimensions"):
            other.load(path)

    def test_memory_bytes_counts_nets_and_buffer(self) -> None:
        agent = make_agent()
        n_params = sum(p.numel() for p in agent.policy_net.parameters())
        assert agent.memory_bytes() == 2 * n_params * 4 + agent.buffer.nbytes()

    def test_create_agent_builds_dqn(self, fake_env: object) -> None:
        agent = create_agent(Config(algorithm="dqn", device="cpu"), fake_env)
        assert isinstance(agent, DQNAgent)
        assert agent.policy_net.net[0].in_features == fake_env.n_states  # type: ignore[union-attr]


@pytest.mark.slow
class TestDQNConvergence:
    def test_learns_taxi_v3(self) -> None:
        from src.environments import create_env
        from src.evaluation.evaluator import Evaluator
        from src.training.trainer import Trainer
        from src.utils.seeding import eval_seeds

        config = Config(
            algorithm="dqn",
            device="cpu",
            n_train_episodes=1500,
            decay_frac=0.5,
        )
        env = create_env(config)
        agent = create_agent(config, env, n_episodes=config.n_train_episodes)
        Trainer(agent, env, config).train(config.n_train_episodes)
        results = Evaluator(env, seeds=eval_seeds(config.seed, 30)).evaluate(agent, 30)
        assert results.mean_reward > 0
