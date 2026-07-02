"""Argument parser: subcommands and Config-overridable options."""

from __future__ import annotations

import argparse

from src.config import (
    VALID_ALGORITHMS,
    VALID_DECAY_TYPES,
    VALID_DEVICES,
    VALID_ENVS,
    VALID_EXPLORATIONS,
    VALID_REWARD_SHAPING,
)


def _add_config_overrides(parser: argparse.ArgumentParser) -> None:
    """Options mapped 1:1 onto Config fields (None = not provided)."""
    parser.add_argument("--agent", dest="algorithm", choices=VALID_ALGORITHMS, default=None)
    parser.add_argument("--env", choices=VALID_ENVS, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-episodes", dest="n_train_episodes", type=int, default=None)
    parser.add_argument("--test-episodes", dest="n_test_episodes", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None, help="learning rate")
    parser.add_argument("--gamma", type=float, default=None, help="discount factor")
    parser.add_argument("--exploration", choices=VALID_EXPLORATIONS, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--epsilon-min", dest="epsilon_min", type=float, default=None)
    parser.add_argument("--epsilon-decay", dest="epsilon_decay", type=float, default=None)
    parser.add_argument("--decay-type", dest="decay_type", choices=VALID_DECAY_TYPES, default=None)
    parser.add_argument(
        "--decay-frac",
        dest="decay_frac",
        type=float,
        default=None,
        help="epsilon reaches epsilon_min at this fraction of the training horizon",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Boltzmann temperature")
    parser.add_argument("--ucb-c", dest="ucb_c", type=float, default=None)
    parser.add_argument(
        "--reward-shaping", dest="reward_shaping", choices=VALID_REWARD_SHAPING, default=None
    )
    parser.add_argument("--lr", type=float, default=None, help="DQN learning rate")
    parser.add_argument("--device", choices=VALID_DEVICES, default=None)


def build_parser() -> argparse.ArgumentParser:
    """Build the taxi-driver CLI parser (train / eval / play / benchmark / compare)."""
    parser = argparse.ArgumentParser(
        prog="taxi-driver",
        description="Model-free RL agents solving Gymnasium Taxi-v3 (Epitech T-AIA-902)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train an agent then evaluate it")
    train.add_argument(
        "--mode",
        choices=("user", "time-limited"),
        default=None,
        help="user: tune hyperparameters; time-limited: optimized config within --time",
    )
    train.add_argument("--config", default=None, help="YAML configuration file")
    train.add_argument(
        "--time",
        dest="time_budget",
        type=float,
        default=None,
        help="wall-clock training budget in seconds (time-limited mode)",
    )
    train.add_argument("--save", default=None, help="model output path (.npz / .pt)")
    train.add_argument(
        "--show-episodes",
        type=int,
        default=None,
        help="number of random episodes displayed after evaluation (default 3)",
    )
    train.add_argument(
        "--non-interactive",
        action="store_true",
        help="never prompt (CI); missing values fall back to config defaults",
    )
    _add_config_overrides(train)

    evaluate = subparsers.add_parser("eval", help="evaluate a saved model")
    evaluate.add_argument("--model", required=True, help="path to a saved model (.npz)")
    evaluate.add_argument("--test-episodes", dest="n_test_episodes", type=int, default=None)
    evaluate.add_argument("--show-episodes", type=int, default=None)
    evaluate.add_argument("--seed", type=int, default=None)

    play = subparsers.add_parser("play", help="replay episodes of a saved model step by step")
    play.add_argument("--model", required=True)
    play.add_argument("--episodes", type=int, default=3)
    play.add_argument("--delay", type=float, default=0.2, help="pause between steps (s)")
    play.add_argument("--seed", type=int, default=None)

    benchmark = subparsers.add_parser(
        "benchmark", help="run the experiment campaign (feature/benchmarking-viz)"
    )
    benchmark.add_argument("--sweep-config", default="configs/benchmark_sweep.yaml")
    benchmark.add_argument("--out", default="results")

    compare = subparsers.add_parser(
        "compare", help="compare agents head-to-head (feature/benchmarking-viz)"
    )
    compare.add_argument("--agents", default="brute_force,q_learning,sarsa")
    compare.add_argument("--train-episodes", dest="n_train_episodes", type=int, default=None)
    compare.add_argument("--test-episodes", dest="n_test_episodes", type=int, default=None)
    compare.add_argument("--n-seeds", type=int, default=5)
    compare.add_argument("--out", default="results")

    return parser
