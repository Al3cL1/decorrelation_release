"""One factor alone on 1D problems (the other axis held at zero everywhere).

For every lambda_* checkpoint, factor 0 alone and factor 1 alone are each run on pure-x and
pure-y problems: four success rates per lambda.
"""

import argparse
import os
import json
import pathlib

import numpy as np
import torch

from data import BOX_LIM, HORIZON, MIN_START_GOAL_DIST, STATE_DIM, trajectory_stats
from diffusion import DDPMSchedule
from visualize import load_model, ddpm_sample_with_diagnostics


@torch.no_grad()
def ddpm_sample_masked(model, schedule, obs, device, keep, K, active_axis, seed=0):
    """DDIM rollout of factor `keep` alone with the other axis held at zero, weighted by the router
    (or 1/K without one).
    """
    n_steps = schedule.num_timesteps
    batch = obs.shape[0]
    off = 1 - active_axis
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x = torch.randn(batch, HORIZON, STATE_DIM, generator=gen, device=device)
    x[..., off] = 0.0
    if model.use_router:
        w_keep = model.router(obs)[:, keep].view(batch, 1, 1)
    else:
        w_keep = torch.full((batch, 1, 1), 1.0 / K, device=device)
    for t in reversed(range(n_steps)):
        t_idx = torch.full((batch,), t, device=device, dtype=torch.long)
        t_norm = (t_idx.float() / n_steps).unsqueeze(-1)
        _, eps_per = model(x, obs, t_norm)
        eps = w_keep * eps_per[:, keep]
        eps[..., off] = 0.0
        acp_t = schedule.alphas_cumprod[t]
        acp_prev = schedule.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=device)
        x0_hat = (x - torch.sqrt(1.0 - acp_t) * eps) / torch.sqrt(acp_t)
        x = torch.sqrt(acp_prev) * x0_hat + torch.sqrt(1.0 - acp_prev) * eps
        x[..., off] = 0.0
    return x


def make_1d_problems(n, axis, rng):
    """Pure-1D problems with the off-axis identically 0 (start and goal)."""
    starts, goals = [], []
    while len(starts) < n:
        s = np.zeros(STATE_DIM, dtype=np.float32)
        g = np.zeros(STATE_DIM, dtype=np.float32)
        s[axis] = rng.uniform(-BOX_LIM, BOX_LIM)
        g[axis] = rng.uniform(-BOX_LIM, BOX_LIM)
        if abs(g[axis] - s[axis]) > MIN_START_GOAL_DIST:
            starts.append(s)
            goals.append(g)
    return np.stack(starts), np.stack(goals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--num_diffusion_steps", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_path", default="outputs/eval_1d.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    schedule = DDPMSchedule(num_timesteps=args.num_diffusion_steps, device=device)

    rng = np.random.default_rng(0)
    probs = {0: make_1d_problems(args.n, axis=0, rng=rng),
             1: make_1d_problems(args.n, axis=1, rng=rng)}
    obs = {ax: torch.from_numpy(np.concatenate(p, axis=-1)).to(device) for ax, p in probs.items()}
    goals = {ax: torch.from_numpy(p[1]) for ax, p in probs.items()}
    axis_name = {0: "x", 1: "y"}

    ckpt_root = pathlib.Path(args.ckpt_dir)
    lam_dirs = sorted(ckpt_root.glob("lambda_*"), key=lambda d: float(d.name.split("_")[1]))

    results = {}
    diagnostics = {}
    print(f"{'lambda':>8}  {'F0only_x':>9}  {'F0only_y':>9}  {'F1only_x':>9}  {'F1only_y':>9}  {'er':>5}  {'|cos|':>6}")
    print("-" * 70)
    for d in lam_dirs:
        lam = d.name.split("_")[1]
        model = load_model(d / "final.pt", device, K=args.K)
        row = {}
        for keep in range(args.K):
            for ax in (0, 1):
                traj = ddpm_sample_masked(
                    model, schedule, obs[ax], device,
                    keep=keep, K=args.K, active_axis=ax, seed=0,
                ).cpu()
                err = trajectory_stats(traj, goals[ax])["goal_error"].numpy()
                row[f"F{keep}only_{axis_name[ax]}"] = float((err < 0.12).mean())
        results[lam] = row

        # Structural diagnostics: capture per-factor eps + router weights on the
        # 1D obs distributions (both axes), aggregated into one diag dict per lambda.
        per_axis_diag = {}
        for ax in (0, 1):
            _, diag_ax = ddpm_sample_with_diagnostics(model, schedule, obs[ax], device)
            per_axis_diag[f"axis_{axis_name[ax]}"] = diag_ax
        diagnostics[lam] = per_axis_diag

        avg_er = np.mean([per_axis_diag[k]['effective_rank'] for k in per_axis_diag])
        avg_ac = np.mean([per_axis_diag[k]['avg_abs_cos_offdiag'] for k in per_axis_diag])
        print(f"{lam:>8}  {row['F0only_x']:>9.4f}  {row['F0only_y']:>9.4f}  "
              f"{row['F1only_x']:>9.4f}  {row['F1only_y']:>9.4f}  "
              f"{avg_er:>5.3f}  {avg_ac:>6.3f}")

    out_path = pathlib.Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out_path}")

    diag_path = out_path.with_name(out_path.stem + "_diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"saved {diag_path}")


if __name__ == "__main__":
    main()
