"""Episode replay exports: animated GIF of a greedy episode."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np

if TYPE_CHECKING:
    from src.agents.base_agent import BaseAgent


def save_episode_gif(
    agent: BaseAgent,
    out_path: str | Path,
    seed: int = 42,
    env_id: str = "Taxi-v3",
    fps: int = 3,
    max_steps: int = 60,
) -> Path:
    """Render one greedy episode to an animated GIF (report/presentation).

    Uses a dedicated ``rgb_array`` environment instance (pygame-backed), so
    the agent's training/evaluation envs are untouched.

    Args:
        agent: Trained agent, played greedily.
        out_path: GIF destination.
        seed: Episode seed.
        env_id: Gymnasium environment id.
        fps: Animation speed.
        max_steps: Safety cap.

    Returns:
        The written path.
    """
    import imageio.v3 as iio

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = gym.make(env_id, render_mode="rgb_array")
    try:
        state, _ = env.reset(seed=seed)
        frames: list[Any] = [env.render()]
        for _ in range(max_steps):
            action = agent.select_action(int(state), greedy=True)
            state, _, terminated, truncated, _ = env.step(action)
            frames.append(env.render())
            if terminated or truncated:
                break
        iio.imwrite(out_path, [np.asarray(f) for f in frames], duration=1000 // fps, loop=0)
    finally:
        env.close()
    return out_path
