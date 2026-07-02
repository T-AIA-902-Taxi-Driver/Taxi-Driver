"""Benchmarker tests: run persistence, idempotency, aggregation, thresholds."""

import dataclasses
from pathlib import Path

from src.benchmarking.benchmarker import (
    RunSpec,
    aggregate_runs,
    episodes_to_threshold,
    find_completed_run,
    run_single,
    write_optimized_yaml,
)
from src.config import Config

TINY = Config(
    algorithm="q_learning",
    n_train_episodes=40,
    n_test_episodes=5,
    eval_interval=10,
    n_probe_episodes=2,
)


class TestRunSingle:
    def test_writes_all_artifacts(self, tmp_path: Path) -> None:
        spec = RunSpec(exp_id="tst", config=TINY, seed=42)
        summary = run_single(spec, results_root=tmp_path)
        run_dirs = list((tmp_path / "raw" / "tst").glob("q_learning__*__s42__*"))
        assert len(run_dirs) == 1
        for name in (
            "config.json",
            "train_episodes.csv",
            "probes.csv",
            "eval_episodes.csv",
            "summary.json",
        ):
            assert (run_dirs[0] / name).exists(), name
        assert summary["n_train_episodes"] == 40
        assert summary["algorithm"] == "q_learning"
        assert len(summary["probe_episodes"]) == 4  # every 10 of 40
        # train csv has header + 40 rows
        lines = (run_dirs[0] / "train_episodes.csv").read_text().strip().splitlines()
        assert len(lines) == 41
        # eval csv has the fixed eval seeds
        eval_lines = (run_dirs[0] / "eval_episodes.csv").read_text().strip().splitlines()
        assert len(eval_lines) == 6
        assert "10042" in eval_lines[1]  # EVAL_BASE_SEED + offset

    def test_idempotent_resume(self, tmp_path: Path) -> None:
        spec = RunSpec(exp_id="tst", config=TINY, seed=43)
        first = run_single(spec, results_root=tmp_path)
        assert find_completed_run(tmp_path, spec) is not None
        second = run_single(spec, results_root=tmp_path)  # must reuse, not re-train
        assert first == second
        assert len(list((tmp_path / "raw" / "tst").glob("*__s43__*"))) == 1

    def test_zero_training_episodes_brute_force(self, tmp_path: Path) -> None:
        config = dataclasses.replace(TINY, algorithm="brute_force", n_train_episodes=0)
        summary = run_single(RunSpec("tst", config, 42), results_root=tmp_path)
        assert summary["n_train_episodes"] == 0
        assert summary["first_success_episode"] is None
        assert summary["post_convergence_std"] is None
        assert summary["eval_mean_reward"] < 0  # random policy is terrible

    def test_shaped_run_reports_native_rewards(self, tmp_path: Path) -> None:
        config = dataclasses.replace(TINY, reward_shaping="potential")
        summary = run_single(RunSpec("tst", config, 44), results_root=tmp_path)
        # native eval rewards on Taxi are bounded by 20 - steps
        assert summary["eval_mean_reward"] <= 20


class TestAggregation:
    def test_aggregate_and_threshold(self, tmp_path: Path) -> None:
        for seed in (42, 43):
            run_single(RunSpec("agg", TINY, seed), results_root=tmp_path)
        frame = aggregate_runs(tmp_path, "agg", r_star=8.0)
        assert len(frame) == 2
        assert {
            "algorithm",
            "seed",
            "eval_mean_reward",
            "episodes_to_threshold",
            "converged",
        } <= set(frame.columns)
        assert (tmp_path / "aggregated" / "agg__runs.csv").exists()

    def test_episodes_to_threshold_sustain_logic(self) -> None:
        episodes = [100, 200, 300, 400, 500]
        # single spike does not count; sustained window returns its start
        assert episodes_to_threshold(episodes, [1, 9, 1, 1, 1], 8.0, sustain=3) is None
        assert episodes_to_threshold(episodes, [1, 9, 9, 9, 1], 8.0, sustain=3) == 200
        assert episodes_to_threshold(episodes, [9, 9, 1, 9, 9], 8.0, sustain=3) is None


class TestWriteOptimized:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "optimized.yaml"
        write_optimized_yaml(TINY, path, note="test note")
        loaded = Config.from_yaml(path)
        assert loaded.algorithm == "q_learning"
        assert loaded.mode == "time_limited"
        assert "test note" in path.read_text()
