# Taxi Driver — Reinforcement Learning for Taxi-v3

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Taxi--v3-green?logo=openaigym&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-DQN-red?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)

> A model-free reinforcement learning agent that solves the Gymnasium Taxi-v3 environment using Q-Learning, Deep Q-Network, and brute-force baselines. Epitech T-AIA-902 project.

## Table of Contents

- [About](#about)
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Algorithms](#algorithms)
- [Benchmarking & Visualization](#benchmarking--visualization)
- [Tech Stack](#tech-stack)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Team](#team)
- [License](#license)

## About

Taxi Driver solves the classic **Taxi-v3** discrete control problem from Gymnasium using optimized model-free episodic reinforcement learning algorithms. The taxi must navigate a 5x5 grid to pick up passengers at one of 4 locations and drop them off at the correct destination.

**Environment details:**
- **State space:** 500 discrete states (25 taxi positions x 5 passenger locations x 4 destinations)
- **Action space:** 6 actions (North, South, East, West, Pickup, Dropoff)
- **Rewards:** +20 for successful dropoff, -1 per step, -10 for illegal pickup/dropoff

The project implements multiple RL algorithms, compares them against a brute-force baseline, and produces comprehensive benchmark reports demonstrating that a fine-tuned RL agent solves the problem in ~13 steps vs ~350 for random exploration.

## Features

**Core Algorithms**
- Q-Learning agent with configurable hyperparameters (alpha, gamma, epsilon, decay)
- Deep Q-Network (DQN) agent with experience replay and target network
- Brute-force random baseline for comparison
- Monte Carlo first-visit agent (optional comparison)

**Execution Modes**
- **User mode** — interactively tune algorithm parameters and observe training
- **Time-limited mode** — train and solve using pre-optimized parameters

**Analysis & Benchmarking**
- Multi-algorithm comparison framework (BruteForce vs Q-Learning vs DQN)
- Hyperparameter sweep with grid search
- Reward shaping experiments
- Exportable results (CSV/JSON)

**Visualization**
- Learning curves (reward and steps per episode)
- Q-table heatmaps
- Episode replay (terminal rendering)
- Comparative charts across algorithms

**Bonus**
- Extended 2-passenger environment with route optimization

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
git clone https://github.com/T-AIA-902-Taxi-Driver/Taxi-Driver.git
cd Taxi-Driver
poetry install
poetry shell
```

### Quick Validation

```bash
python -m src.main train --algorithm qlearning --episodes 100 --test-episodes 10
```

## Usage

### User Mode

Interactively set algorithm parameters:

```bash
python -m src.main train --mode user \
    --algorithm qlearning \
    --alpha 0.1 \
    --gamma 0.99 \
    --epsilon 1.0 \
    --epsilon-decay 0.995 \
    --episodes 10000 \
    --test-episodes 100
```

### Time-Limited Mode

Use pre-optimized parameters:

```bash
python -m src.main train --mode timed --episodes 5000 --test-episodes 100
```

### CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--mode` | Execution mode (`user` / `timed`) | `user` |
| `--algorithm` | Algorithm (`qlearning` / `dqn` / `bruteforce` / `montecarlo`) | `qlearning` |
| `--episodes` | Number of training episodes | `10000` |
| `--test-episodes` | Number of evaluation episodes | `100` |
| `--alpha` | Learning rate | `0.1` |
| `--gamma` | Discount factor | `0.99` |
| `--epsilon` | Initial exploration rate | `1.0` |
| `--epsilon-decay` | Epsilon decay rate | `0.995` |
| `--render` | Display episode replays | `false` |
| `--save-model` | Path to save trained model | — |
| `--load-model` | Path to load pre-trained model | — |
| `--seed` | Random seed for reproducibility | `42` |

## Project Structure

```
Taxi-Driver/
├── src/
│   ├── main.py                  # CLI entry point
│   ├── config.py                # Configuration management
│   ├── agents/
│   │   ├── base_agent.py        # Abstract base agent (ABC)
│   │   ├── brute_force_agent.py # Random action baseline
│   │   ├── q_learning_agent.py  # Tabular Q-Learning
│   │   ├── dqn_agent.py         # Deep Q-Network
│   │   └── monte_carlo_agent.py # Monte Carlo first-visit
│   ├── environments/
│   │   ├── taxi_wrapper.py      # Gymnasium Taxi-v3 wrapper
│   │   └── multi_passenger_env.py # 2-passenger extension (bonus)
│   ├── training/
│   │   ├── trainer.py           # Generic training pipeline
│   │   └── callbacks.py         # Logging, early stopping
│   ├── evaluation/
│   │   ├── evaluator.py         # Agent evaluation
│   │   └── metrics.py           # Metrics computation
│   ├── benchmarking/
│   │   ├── benchmarker.py       # Parameter sweep & comparison
│   │   └── reward_shaping.py    # Custom reward functions
│   └── visualization/
│       ├── plots.py             # Learning curves, heatmaps
│       └── episode_replay.py    # Terminal episode replay
├── configs/
│   ├── default.yaml             # Default hyperparameters
│   ├── optimized.yaml           # Optimized preset (timed mode)
│   └── benchmark_sweep.yaml     # Grid search configuration
├── models/                      # Saved Q-tables & model weights
├── results/                     # Benchmark outputs & plots
├── tests/                       # Unit & integration tests
├── docs/
│   ├── project.pdf              # Epitech project specification
│   ├── CADRAGE.md               # Project framing document
│   ├── ARCHITECTURE.md          # Software architecture
│   └── BACKLOG.md               # Product backlog (WSJF)
├── .github/workflows/ci.yml     # CI pipeline
├── pyproject.toml               # Poetry config (deps, scripts, tools)
├── poetry.lock                  # Locked dependency versions
└── README.md
```

## Algorithms

### Q-Learning (Tabular)

Off-policy TD control using a state-action value table. Bellman update:

```
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
```

Ideal for Taxi-v3's small discrete state space (500 states x 6 actions). Converges in ~2,000-5,000 episodes to near-optimal performance (~13 steps/episode).

### Deep Q-Network (DQN)

Neural network approximation of Q-values with experience replay and target network stabilization. Included for comparative analysis — demonstrates how DQN handles discrete environments vs. tabular methods.

### Brute-Force Baseline

Uniform random action selection. Provides the lower-bound baseline (~350 steps/episode). Essential reference point for demonstrating RL improvement.

### Monte Carlo (First-Visit)

On-policy episodic method averaging returns from first visits to each state-action pair. Offers an alternative convergence profile compared to TD methods.

## Benchmarking & Visualization

### Metrics Tracked

| Metric | Target |
|--------|--------|
| Mean reward per episode | > 7.0 |
| Mean steps per episode | < 15 |
| Success rate | > 95% |
| Convergence episodes | < 5,000 |
| Improvement ratio vs brute-force | ~20x |

### Generated Plots

- **Learning curves** — reward and steps over training episodes
- **Algorithm comparison** — side-by-side bar charts and overlaid curves
- **Q-table heatmap** — visualize learned state-action values
- **Hyperparameter sensitivity** — performance vs. alpha, gamma, epsilon

Sample outputs are saved to `results/`.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Poetry | Dependency management & virtualenv |
| Gymnasium | RL environment (Taxi-v3) |
| NumPy | Numerical computation, Q-table storage |
| PyTorch | DQN neural network |
| Matplotlib / Seaborn | Visualization & plots |
| PyYAML | Configuration files |
| argparse | CLI interface |
| pytest | Testing framework |
| ruff / black | Linting & formatting |
| mypy | Static type checking |
| GitHub Actions | CI/CD pipeline |

## Documentation

| Document | Description |
|---|---|
| [CADRAGE.md](docs/CADRAGE.md) | Project framing — process, user stories, metrics, risks |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Software architecture with Mermaid diagrams |
| [BACKLOG.md](docs/BACKLOG.md) | Product backlog with WSJF prioritization |
| [project.pdf](docs/project.pdf) | Epitech project specification |

## Contributing

1. Branch from `dev` using the convention `feature/<name>` or `fix/<name>`
2. Follow [PEP 8](https://peps.python.org/pep-0008/) with type hints
3. Write tests for new functionality
4. Submit a PR to `dev` — squash merge after review
5. Use conventional commits: `type(scope): description`

## Team

| Name | GitHub | Role |
|------|--------|------|
| Romain Bernier | [@Romain-Ber](https://github.com/Romain-Ber) | — |
| Victor Vattier | [@VictorVattierEpitech](https://github.com/VictorVattierEpitech) | — |
| Duncan Carbonnier | [@DuncanCbr](https://github.com/DuncanCbr) | — |
| Emeric Legendre | [@Macfreeze](https://github.com/Macfreeze) | — |
| Marin Chevalier | — | — |

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

Built with reinforcement learning at **{Epitech}** — T-AIA-902
