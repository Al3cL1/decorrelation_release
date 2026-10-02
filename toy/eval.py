"""Evaluate composed sampling on 2D start-to-goal trajectory tasks."""

import argparse
import json
import pathlib

import numpy as np
import torch

from data import ProblemSampler, trajectory_stats
from diffusion import DDPMSchedule
from visualize import ddpm_sample, ddpm_sample_with_diagnostics, load_model


def summarize(traj, goals):
    stats = trajectory_stats(traj, goals)
    goal_error = stats["goal_error"].cpu().numpy()
    path_length = stats["path_length"].cpu().numpy()
    success = goal_error < 0.12
    return {
        "goal_error_mean": float(goal_error.mean()),
        "goal_error_median": float(np.median(goal_error)),
        "path_length_mean": float(path_length.mean()),
        "success_rate": float(success.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--lambdas", nargs="+", default=["0", "0.1", "1"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--num_diffusion_steps", type=int, default=100)
    ap.add_argument("--dataset_size", type=int, default=512)
    ap.add_argument("--oracle_device", type=str, default="cpu")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_path", default="outputs/eval.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    schedule = DDPMSchedule(num_timesteps=args.num_diffusion_steps, device=device)
    # Different seed than training so eval is a fresh draw from the same distribution.
    sampler = ProblemSampler(
        dataset_size=args.dataset_size,
        seed=args.seed + 1000,
        oracle_device=args.oracle_device,
    )

    ds = sampler.dataset
    results = {}
    diagnostics = {}
    print(f"{'lambda':>8}  {'goal':>8}  {'path':>8}  {'succ':>8}  {'er':>6}  {'|cos|':>6}  {'dc':>8}")
    print("-" * 64)
    for lam in args.lambdas:
        model = load_model(pathlib.Path(args.ckpt_dir) / f"lambda_{lam}" / "final.pt", device, K=args.K)
        obs = ds.obs.to(device)
        traj_dev, diag = ddpm_sample_with_diagnostics(model, schedule, obs, device)
        traj = traj_dev.cpu()
        metrics = summarize(traj, ds.goals)
        results[lam] = metrics
        diagnostics[lam] = diag
        print(
            f"{lam:>8}  {metrics['goal_error_mean']:>8.4f}  "
            f"{metrics['path_length_mean']:>8.4f}  "
            f"{metrics['success_rate']:>8.4f}  "
            f"{diag['effective_rank']:>6.3f}  "
            f"{diag['avg_abs_cos_offdiag']:>6.3f}  "
            f"{diag['double_counting']:>+8.3f}"
        )

    out_path = pathlib.Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out_path}")

    # Diagnostics: <out_path stem>_diagnostics.json sibling
    diag_path = out_path.with_name(out_path.stem + "_diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"saved {diag_path}")


if __name__ == "__main__":
    main()
