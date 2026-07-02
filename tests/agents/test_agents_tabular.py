"""Hand-computed update tests for every tabular agent + shared contracts."""

from pathlib import Path

import numpy as np
import pytest

from src.agents import AGENT_REGISTRY, create_agent
from src.agents.base_agent import BaseAgent
from src.agents.brute_force_agent import BruteForceAgent
from src.agents.double_q_learning_agent import DoubleQLearningAgent
from src.agents.expected_sarsa_agent import ExpectedSARSAAgent
from src.agents.monte_carlo_agent import MonteCarloAgent
from src.agents.q_learning_agent import QLearningAgent
from src.agents.sarsa_agent import SARSAAgent
from src.config import Config

N_STATES, N_ACTIONS = 10, 4
LEARN_CONFIG = Config(alpha=0.5, gamma=0.9, epsilon=0.0)  # greedy strategy for determinism


def make(cls: type, config: Config | None = None, seed: int = 7) -> BaseAgent:
    agent: BaseAgent = cls(N_STATES, N_ACTIONS, config or LEARN_CONFIG, np.random.default_rng(seed))
    return agent


class TestBaseContracts:
    def test_base_agent_is_abstract(self, config: Config, rng: np.random.Generator) -> None:
        with pytest.raises(TypeError):
            BaseAgent(N_STATES, N_ACTIONS, config, rng)  # type: ignore[abstract]

    def test_registry_contains_all_tabular_agents(self) -> None:
        assert set(AGENT_REGISTRY) == {
            "brute_force",
            "q_learning",
            "sarsa",
            "expected_sarsa",
            "double_q_learning",
            "monte_carlo",
        }

    def test_create_agent_uses_env_dimensions(self, fake_env: object) -> None:
        agent = create_agent(Config(algorithm="q_learning"), fake_env)
        assert isinstance(agent, QLearningAgent)
        assert agent.q_table.shape == (fake_env.n_states, fake_env.n_actions)

    def test_create_agent_resolves_dqn_lazily(self, fake_env: object) -> None:
        from src.agents.dqn.dqn_agent import DQNAgent

        agent = create_agent(Config(algorithm="dqn", device="cpu"), fake_env)
        assert isinstance(agent, DQNAgent)

    @pytest.mark.parametrize("name", sorted(AGENT_REGISTRY))
    def test_select_action_in_range(self, name: str, fake_env: object) -> None:
        agent = create_agent(Config(algorithm=name), fake_env)
        for greedy in (False, True):
            action = agent.select_action(0, greedy=greedy)
            assert 0 <= action < fake_env.n_actions

    @pytest.mark.parametrize("name", sorted(set(AGENT_REGISTRY) - {"brute_force"}))
    def test_save_load_round_trip(self, name: str, tmp_path: Path) -> None:
        config = Config(algorithm=name, epsilon=1.0)
        agent = make(AGENT_REGISTRY[name], config)
        # random-ish training to fill tables
        for _ in range(3):
            for s in range(N_STATES - 1):
                agent.learn(s, agent.select_action(s), -1.0, s + 1, s == N_STATES - 2)
            agent.end_episode()
        path = tmp_path / f"{name}.npz"
        agent.save(path)
        clone = make(AGENT_REGISTRY[name], config)
        clone.load(path)
        assert np.array_equal(agent.q_table, clone.q_table)  # type: ignore[attr-defined]
        assert clone.n_episodes_trained == agent.n_episodes_trained

    def test_load_shape_mismatch_raises(self, tmp_path: Path) -> None:
        agent = make(QLearningAgent)
        path = tmp_path / "q.npz"
        agent.save(path)
        small = QLearningAgent(5, N_ACTIONS, LEARN_CONFIG, np.random.default_rng(0))
        with pytest.raises(ValueError, match="shape"):
            small.load(path)

    def test_reproducibility_same_seed_same_tables(self, fake_env: object) -> None:
        config = Config(algorithm="q_learning", epsilon=1.0)
        tables = []
        for _ in range(2):
            agent = create_agent(config, fake_env)
            for s in range(fake_env.n_states - 1):
                a = agent.select_action(s)
                agent.learn(s, a, float(a), s + 1, False)
            agent.end_episode()
            tables.append(agent.q_table.copy())  # type: ignore[attr-defined]
        assert np.array_equal(tables[0], tables[1])


class TestBruteForce:
    def test_actions_cover_space_and_learn_is_noop(self) -> None:
        agent = make(BruteForceAgent)
        actions = {agent.select_action(0) for _ in range(200)}
        assert actions == set(range(N_ACTIONS))
        agent.learn(0, 0, 10.0, 1, False)  # must not raise nor store anything
        assert agent.memory_bytes() == 0


class TestQLearning:
    def test_hand_computed_update(self) -> None:
        agent = make(QLearningAgent)
        agent._q[3] = np.array([1.0, 7.0, 2.0, 0.0])
        agent.learn(state=0, action=1, reward=10.0, next_state=3, terminated=False)
        # target = 10 + 0.9 * 7 = 16.3 ; q = 0 + 0.5 * 16.3 = 8.15
        assert agent._q[0, 1] == pytest.approx(8.15)

    def test_terminal_update_ignores_next_state(self) -> None:
        agent = make(QLearningAgent)
        agent._q[3] = np.array([100.0, 100.0, 100.0, 100.0])
        agent.learn(0, 2, 20.0, 3, terminated=True)
        assert agent._q[0, 2] == pytest.approx(10.0)  # 0 + 0.5 * 20


class TestSARSA:
    def test_update_uses_and_caches_next_action(self) -> None:
        agent = make(SARSAAgent)
        agent._q[3] = np.array([1.0, 7.0, 2.0, 0.0])
        agent.learn(0, 1, 10.0, 3, terminated=False)
        # greedy strategy (eps=0) picks a'=1 → target = 10 + 0.9*7 = 16.3
        assert agent._q[0, 1] == pytest.approx(8.15)
        # the very next on-policy selection in s'=3 must return the cached a'
        assert agent.select_action(3) == 1
        # cache is one-shot
        agent._q[3] = np.array([9.0, 0.0, 0.0, 0.0])
        assert agent.select_action(3) == 0

    def test_end_episode_clears_cache(self) -> None:
        agent = make(SARSAAgent)
        agent._q[3] = np.array([0.0, 5.0, 0.0, 0.0])
        agent.learn(0, 0, 0.0, 3, terminated=False)
        agent.end_episode()
        agent._q[3] = np.array([9.0, 0.0, 0.0, 0.0])
        assert agent.select_action(3) == 0  # fresh draw, not stale cache


class TestExpectedSARSA:
    def test_hand_computed_expectation(self) -> None:
        config = Config(alpha=0.5, gamma=0.9, epsilon=0.4)
        agent = make(ExpectedSARSAAgent, config)
        agent._q[3] = np.array([0.0, 8.0, 4.0, 0.0])
        agent.learn(0, 1, 10.0, 3, terminated=False)
        # π(a|3): eps/4 = 0.1 each + 0.6 on argmax (a=1, unique)
        # E[Q] = 0.1*0 + 0.7*8 + 0.1*4 + 0.1*0 = 6.0 ; target = 10 + 0.9*6 = 15.4
        assert agent._q[0, 1] == pytest.approx(0.5 * 15.4)


class TestDoubleQLearning:
    def test_cross_table_update_matches_coin(self) -> None:
        agent = make(DoubleQLearningAgent, seed=11)
        peek = np.random.default_rng(11)
        agent._qa[3] = np.array([0.0, 6.0, 0.0, 0.0])  # argmax A in s'=3 → a'=1
        agent._qb[3] = np.array([0.0, 2.0, 0.0, 0.0])  # Q_B(3,1) = 2
        coin_updates_a = peek.random() < 0.5
        agent.learn(0, 0, 1.0, 3, terminated=False)
        if coin_updates_a:
            # target = 1 + 0.9 * Q_B(3, argmax Q_A(3)) = 1 + 0.9*2 = 2.8
            assert agent._qa[0, 0] == pytest.approx(0.5 * 2.8)
            assert agent._qb[0, 0] == 0.0
        else:
            # target = 1 + 0.9 * Q_A(3, argmax Q_B(3)) = 1 + 0.9*6 = 6.4
            assert agent._qb[0, 0] == pytest.approx(0.5 * 6.4)
            assert agent._qa[0, 0] == 0.0

    def test_selection_uses_sum_of_tables(self) -> None:
        agent = make(DoubleQLearningAgent)
        agent._qa[0] = np.array([1.0, 0.0, 0.0, 0.0])
        agent._qb[0] = np.array([0.0, 0.0, 2.0, 0.0])
        assert agent.select_action(0, greedy=True) == 2
        assert agent.memory_bytes() == 2 * agent._qa.nbytes


class TestMonteCarlo:
    def test_first_visit_returns_on_synthetic_episode(self) -> None:
        config = Config(alpha=0.5, gamma=0.5, epsilon=0.0)
        agent = make(MonteCarloAgent, config)
        agent.learn(0, 0, 0.0, 1, False)
        agent.learn(1, 1, 1.0, 2, True)
        assert agent._q[0, 0] == 0.0  # nothing updated before episode end
        agent.end_episode()
        # G(s1,a1) = 1 ; G(s0,a0) = 0 + 0.5*1 = 0.5 ; counts = 1 → exact means
        assert agent._q[1, 1] == pytest.approx(1.0)
        assert agent._q[0, 0] == pytest.approx(0.5)

    def test_first_visit_counts_repeat_pairs_once(self) -> None:
        config = Config(gamma=1.0, epsilon=0.0)
        agent = make(MonteCarloAgent, config)
        # (0,0) visited twice: returns from first visit = 3, from second = 1
        agent.learn(0, 0, 1.0, 1, False)
        agent.learn(1, 0, 1.0, 0, False)
        agent.learn(0, 0, 1.0, 2, True)
        agent.end_episode()
        assert agent._counts[0, 0] == 1
        assert agent._q[0, 0] == pytest.approx(3.0)  # first-visit return only

    def test_truncated_episode_flushed_by_end_episode(self) -> None:
        agent = make(MonteCarloAgent, Config(gamma=1.0, epsilon=0.0))
        agent.learn(0, 0, 5.0, 1, False)  # episode truncated: terminated stays False
        agent.end_episode()
        assert agent._q[0, 0] == pytest.approx(5.0)
        assert not agent._episode  # buffer cleared


class TestExplorationDecayWiring:
    def test_end_episode_decays_epsilon(self) -> None:
        config = Config(epsilon=1.0, epsilon_min=0.0, epsilon_decay=0.5)
        agent = make(QLearningAgent, config)
        assert agent.exploration_level == 1.0
        agent.end_episode()
        assert agent.exploration_level == 0.5
        assert agent.n_episodes_trained == 1
