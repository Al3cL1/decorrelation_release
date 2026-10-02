"""2D start-to-goal task: straight-line trajectories from a random start to a random goal."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


STATE_DIM = 2
OBS_DIM = 4          # [start_x, start_y, goal_x, goal_y]
HORIZON = 24
BOX_LIM = 1.0        # workspace is [-BOX_LIM, BOX_LIM]^2
MIN_START_GOAL_DIST = 0.9


@dataclass
class ProblemBatch:
    starts: torch.Tensor
    goals: torch.Tensor
    traj: torch.Tensor

    @property
    def obs(self) -> torch.Tensor:
        return torch.cat([self.starts, self.goals], dim=-1)


def sample_problem_pairs(n: int, rng: np.random.Generator):
    """Sample n (start, goal) pairs uniformly in the box, with start-goal distance > MIN."""
    starts_all, goals_all = [], []
    while sum(x.shape[0] for x in starts_all) < n:
        starts = rng.uniform(-BOX_LIM, BOX_LIM, size=(max(4 * n, 1024), STATE_DIM)).astype(np.float32)
        goals = rng.uniform(-BOX_LIM, BOX_LIM, size=(max(4 * n, 1024), STATE_DIM)).astype(np.float32)
        keep = np.linalg.norm(goals - starts, axis=1) > MIN_START_GOAL_DIST
        starts_all.append(starts[keep])
        goals_all.append(goals[keep])
    starts = np.concatenate(starts_all, axis=0)[:n]
    goals = np.concatenate(goals_all, axis=0)[:n]
    return starts, goals


def linear_init(starts_t: torch.Tensor, goals_t: torch.Tensor):
    """Oracle trajectory: straight line from start to goal over HORIZON steps."""
    alphas = torch.linspace(
        1.0 / HORIZON, 1.0, HORIZON,
        device=starts_t.device, dtype=starts_t.dtype,
    )[None, :, None]
    return starts_t[:, None, :] + alphas * (goals_t - starts_t)[:, None, :]


def build_dataset(n_samples: int, seed: int, device: str = "cpu", plan_batch_size: int = 256):
    rng = np.random.default_rng(seed)
    starts_np, goals_np = sample_problem_pairs(n_samples, rng)
    starts_t = torch.from_numpy(starts_np)
    goals_t = torch.from_numpy(goals_np)

    traj_chunks = []
    device_t = torch.device(device)
    for lo in range(0, n_samples, plan_batch_size):
        hi = min(lo + plan_batch_size, n_samples)
        s = starts_t[lo:hi].to(device_t)
        g = goals_t[lo:hi].to(device_t)
        traj_chunks.append(linear_init(s, g).cpu())

    return ProblemBatch(
        starts=starts_t.float(),
        goals=goals_t.float(),
        traj=torch.cat(traj_chunks, dim=0).float(),
    )


class ProblemSampler:
    """Holds a fixed dataset and samples minibatches from it with replacement."""

    def __init__(self, dataset_size: int = 4096, seed: int = 0, oracle_device: str = "cpu"):
        self.dataset = build_dataset(dataset_size, seed=seed, device=oracle_device)
        self.rng = np.random.default_rng(seed + 1)

    def sample_batch(self, batch_size: int):
        ds = self.dataset
        idx = torch.from_numpy(self.rng.integers(0, ds.traj.shape[0], size=batch_size)).long()
        return ProblemBatch(starts=ds.starts[idx], goals=ds.goals[idx], traj=ds.traj[idx])


def trajectory_stats(traj: torch.Tensor, goals: torch.Tensor):
    final_err = torch.linalg.norm(traj[:, -1, :] - goals, dim=-1)
    path_len = torch.linalg.norm(traj[:, 1:, :] - traj[:, :-1, :], dim=-1).sum(dim=1)
    return {"goal_error": final_err, "path_length": path_len}
