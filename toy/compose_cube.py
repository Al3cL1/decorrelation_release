"""2D -> 3D composition with no 3D training.

A 3D straight line is three independent 1D lines, so one decorrelated factor, applied to each
of the x, y, z channels as a 1D primitive, makes a 3D denoiser.
"""

import argparse
import os
import json
import pathlib

import numpy as np
import torch

from data import BOX_LIM, HORIZON, MIN_START_GOAL_DIST
from diffusion import DDPMSchedule
from visualize import load_model, ddpm_sample_with_diagnostics


@torch.no_grad()
def cube_sample(model, schedule, start3d, goal3d, device, keep_factor=0, K=2, seed=0):
    """3D DDIM where each axis is denoised by factor `keep` acting as a 1D primitive, weighted by the
    router (or 1/K without one).
    """
    n_steps = schedule.num_timesteps
    B = start3d.shape[0]
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x = torch.randn(B, HORIZON, 3, generator=gen, device=device)
    zeros_h = torch.zeros(B, HORIZON, device=device)
    zeros_b = torch.zeros(B, device=device)

    # Per-axis obs and solo-factor weight (router-dependent but t-independent).
    obs_per_axis, w_per_axis = [], []
    for a in range(3):
        obs2d = torch.stack([start3d[:, a], zeros_b, goal3d[:, a], zeros_b], dim=-1)  # (B, 4)
        obs_per_axis.append(obs2d)
        if model.use_router:
            w_per_axis.append(model.router(obs2d)[:, keep_factor].view(B, 1))
        else:
            w_per_axis.append(torch.full((B, 1), 1.0 / K, device=device))

    for t in reversed(range(n_steps)):
        t_idx = torch.full((B,), t, device=device, dtype=torch.long)
        t_norm = (t_idx.float() / n_steps).unsqueeze(-1)

        eps3d = torch.zeros(B, HORIZON, 3, device=device)
        for a in range(3):
            traj2d = torch.stack([x[:, :, a], zeros_h], dim=-1)        # (B, H, 2)
            _, eps_per = model(traj2d, obs_per_axis[a], t_norm)        # (B, K, H, 2)
            eps3d[:, :, a] = w_per_axis[a] * eps_per[:, keep_factor, :, 0]

        acp_t = schedule.alphas_cumprod[t]
        acp_prev = schedule.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=device)
        x0_hat = (x - torch.sqrt(1.0 - acp_t) * eps3d) / torch.sqrt(acp_t)
        x = torch.sqrt(acp_prev) * x0_hat + torch.sqrt(1.0 - acp_prev) * eps3d
    return x


def make_cube_problems(n, rng):
    starts, goals = [], []
    while len(starts) < n:
        s = rng.uniform(-BOX_LIM, BOX_LIM, size=3).astype(np.float32)
        g = rng.uniform(-BOX_LIM, BOX_LIM, size=3).astype(np.float32)
        if np.linalg.norm(g - s) > MIN_START_GOAL_DIST:
            starts.append(s)
            goals.append(g)
    return np.stack(starts), np.stack(goals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--lambdas", nargs="+", default=["0", "0.1"])
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--num_diffusion_steps", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_path", default="outputs/compose_cube.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    schedule = DDPMSchedule(num_timesteps=args.num_diffusion_steps, device=device)

    rng = np.random.default_rng(0)
    starts_np, goals_np = make_cube_problems(args.n, rng)
    start3d = torch.from_numpy(starts_np).to(device)
    goal3d = torch.from_numpy(goals_np).to(device)

    results = {}
    diagnostics = {}
    # Per-axis 2D obs used for diagnostics: matches cube_sample's per-axis call.
    zeros_b = torch.zeros(start3d.shape[0], device=device)
    print(f"{'lambda':>8}  {'keep':>5}  {'goal_err':>9}  {'succ':>8}")
    print("-" * 38)
    for lam in args.lambdas:
        model = load_model(pathlib.Path(args.ckpt_dir) / f"lambda_{lam}" / "final.pt", device, K=args.K)
        row = {}
        for keep in range(args.K):
            traj = cube_sample(model, schedule, start3d, goal3d, device, keep_factor=keep, K=args.K)
            goal_err = torch.linalg.norm(traj[:, -1, :] - goal3d, dim=-1).cpu().numpy()
            succ = float((goal_err < 0.12).mean())
            row[f"F{keep}"] = {"goal_err_mean": float(goal_err.mean()), "success_rate": succ}
            print(f"{lam:>8}  {keep:>5}  {goal_err.mean():>9.4f}  {succ:>8.4f}")
        results[lam] = row

        # Structural diagnostics: capture per-factor eps on the per-axis 2D obs
        # that cube_sample feeds the model. Run on all 3 axes for completeness.
        per_axis_diag = {}
        for a in range(3):
            obs2d = torch.stack([start3d[:, a], zeros_b, goal3d[:, a], zeros_b], dim=-1)
            _, diag_ax = ddpm_sample_with_diagnostics(model, schedule, obs2d, device)
            per_axis_diag[f"axis_{a}"] = diag_ax
        diagnostics[lam] = per_axis_diag

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
