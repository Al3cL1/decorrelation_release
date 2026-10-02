"""Visualization utilities for the 2D start-to-goal toy."""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import BOX_LIM, HORIZON, ProblemSampler, STATE_DIM
from diffusion import DDPMSchedule
from model import CompositionalDiffusionPolicy


def load_model(ckpt_path, device, K=2):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = CompositionalDiffusionPolicy(
        K=K,
        hidden=args.get("hidden", 256),
        use_router=args.get("use_router", False),
        softmax_weights=args.get("softmax_weights", False),
        affine_weights=args.get("affine_weights", False),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def ddpm_sample(model, schedule, obs, device, seed=None):
    """DDIM deterministic sampling (η=0)."""
    n_steps = schedule.num_timesteps
    batch = obs.shape[0]
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        x = torch.randn(batch, HORIZON, STATE_DIM, generator=gen, device=device)
    else:
        x = torch.randn(batch, HORIZON, STATE_DIM, device=device)
    for t in reversed(range(n_steps)):
        t_idx = torch.full((batch,), t, device=device, dtype=torch.long)
        t_norm = (t_idx.float() / n_steps).unsqueeze(-1)
        eps, _ = model(x, obs, t_norm)
        acp_t = schedule.alphas_cumprod[t]
        acp_prev = (
            schedule.alphas_cumprod[t - 1] if t > 0
            else torch.tensor(1.0, device=device)
        )
        x0_hat = (x - torch.sqrt(1.0 - acp_t) * eps) / torch.sqrt(acp_t)
        x = torch.sqrt(acp_prev) * x0_hat + torch.sqrt(1.0 - acp_prev) * eps
    return x


@torch.no_grad()
def ddpm_sample_with_diagnostics(model, schedule, obs, device, seed=None):
    """ddpm_sample that also returns the structural diagnostics collected at every step."""
    from diagnostics_util import DiagnosticsAccumulator
    acc = DiagnosticsAccumulator()
    n_steps = schedule.num_timesteps
    batch = obs.shape[0]
    if seed is not None:
        gen = torch.Generator(device=device); gen.manual_seed(seed)
        x = torch.randn(batch, HORIZON, STATE_DIM, generator=gen, device=device)
    else:
        x = torch.randn(batch, HORIZON, STATE_DIM, device=device)
    for t in reversed(range(n_steps)):
        t_idx = torch.full((batch,), t, device=device, dtype=torch.long)
        t_norm = (t_idx.float() / n_steps).unsqueeze(-1)
        eps, eps_per = model(x, obs, t_norm)
        w = model.router(obs) if getattr(model, 'router', None) is not None else None
        acc.update(eps_per, w)
        acp_t = schedule.alphas_cumprod[t]
        acp_prev = (
            schedule.alphas_cumprod[t - 1] if t > 0
            else torch.tensor(1.0, device=device)
        )
        x0_hat = (x - torch.sqrt(1.0 - acp_t) * eps) / torch.sqrt(acp_t)
        x = torch.sqrt(acp_prev) * x0_hat + torch.sqrt(1.0 - acp_prev) * eps
    return x, acc.finalize()


@torch.no_grad()
def ddpm_sample_weighted(model, schedule, obs, device, weights, K, seed=None):
    """DDIM sampling with eps = sum_k weights[k] F_k / K (all-ones weights is the usual composition)."""
    n_steps = schedule.num_timesteps
    batch = obs.shape[0]
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        x = torch.randn(batch, HORIZON, STATE_DIM, generator=gen, device=device)
    else:
        x = torch.randn(batch, HORIZON, STATE_DIM, device=device)
    w = torch.tensor(weights, device=device, dtype=torch.float32).view(1, len(weights), 1, 1)
    for t in reversed(range(n_steps)):
        t_idx = torch.full((batch,), t, device=device, dtype=torch.long)
        t_norm = (t_idx.float() / n_steps).unsqueeze(-1)
        _, eps_per = model(x, obs, t_norm)
        eps = (w * eps_per).sum(dim=1) / K
        acp_t = schedule.alphas_cumprod[t]
        acp_prev = (
            schedule.alphas_cumprod[t - 1] if t > 0
            else torch.tensor(1.0, device=device)
        )
        x0_hat = (x - torch.sqrt(1.0 - acp_t) * eps) / torch.sqrt(acp_t)
        x = torch.sqrt(acp_prev) * x0_hat + torch.sqrt(1.0 - acp_prev) * eps
    return x


def draw_problem(ax, start, goal):
    ax.scatter(start[0], start[1], c="black", s=35, label="start")
    ax.scatter(goal[0], goal[1], c="tab:green", s=45, marker="*", label="goal")
    ax.set_xlim(-BOX_LIM - 0.1, BOX_LIM + 0.1)
    ax.set_ylim(-BOX_LIM - 0.1, BOX_LIM + 0.1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)


def plot_rollouts(args):
    device = torch.device(args.device)
    schedule = DDPMSchedule(num_timesteps=args.num_diffusion_steps, device=device)
    sampler = ProblemSampler(
        dataset_size=args.dataset_size,
        seed=args.seed + 1000,
        oracle_device=args.oracle_device,
    )
    ds = sampler.dataset

    fig, axes = plt.subplots(
        len(args.lambdas), args.n_examples,
        figsize=(4.5 * args.n_examples, 4.0 * len(args.lambdas)),
        squeeze=False,
    )
    example_ids = np.arange(min(args.n_examples, ds.traj.shape[0]))
    for row, lam in enumerate(args.lambdas):
        model = load_model(pathlib.Path(args.ckpt_dir) / f"lambda_{lam}" / "final.pt", device, K=args.K)
        for col, idx in enumerate(example_ids):
            ax = axes[row, col]
            start = ds.starts[idx].numpy()
            goal = ds.goals[idx].numpy()
            obs = ds.obs[idx : idx + 1].to(device)
            pred = ddpm_sample(model, schedule, obs, device)[0].cpu().numpy()
            oracle = ds.traj[idx].numpy()
            draw_problem(ax, start, goal)
            full_oracle = np.concatenate([start[None, :], oracle], axis=0)
            full_pred = np.concatenate([start[None, :], pred], axis=0)
            ax.plot(full_oracle[:, 0], full_oracle[:, 1], color="tab:blue", linewidth=2.0, label="oracle")
            ax.plot(full_pred[:, 0], full_pred[:, 1], color="tab:orange", linewidth=1.8, linestyle="--", label="sample")
            ax.set_title(f"λ={lam}  ex {idx}")
            if row == 0 and col == 0:
                ax.legend(loc="upper left", framealpha=0.9)
    out_path = pathlib.Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


def _save_quiver_panel(out_path, starts, field, *, title, goal, quiver_scale):
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.set_xlim(-BOX_LIM - 0.1, BOX_LIM + 0.1)
    ax.set_ylim(-BOX_LIM - 0.1, BOX_LIM + 0.1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    mags = np.linalg.norm(field, axis=1)
    ax.quiver(
        starts[:, 0], starts[:, 1], field[:, 0], field[:, 1], mags,
        cmap="viridis", angles="xy", scale_units="xy", scale=quiver_scale,
    )
    ax.scatter(goal[0], goal[1], c="red", s=140, marker="*",
               edgecolors="black", linewidths=0.8, zorder=5)
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _save_quiver_combined(out_path, starts, composed_field, factor_fields,
                          *, lam, alpha, goal,
                          composed_quiver_scale, factor_quiver_scale):
    """3-panel side-by-side: composed | factor 0 dominant | factor 1 dominant."""
    K = len(factor_fields)
    fig, axes = plt.subplots(1, 1 + K, figsize=(5.0 * (1 + K), 5.0))
    panels = [("composed", composed_field, composed_quiver_scale)]
    for k in range(K):
        panels.append((f"factor {k} dominant (α={alpha})", factor_fields[k],
                       factor_quiver_scale))
    for ax, (title, field, qs) in zip(axes, panels):
        ax.set_xlim(-BOX_LIM - 0.1, BOX_LIM + 0.1)
        ax.set_ylim(-BOX_LIM - 0.1, BOX_LIM + 0.1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)
        mags = np.linalg.norm(field, axis=1)
        ax.quiver(
            starts[:, 0], starts[:, 1], field[:, 0], field[:, 1], mags,
            cmap="viridis", angles="xy", scale_units="xy", scale=qs,
        )
        ax.scatter(goal[0], goal[1], c="red", s=140, marker="*",
                   edgecolors="black", linewidths=0.8, zorder=5)
        ax.set_title(title)
    fig.suptitle(f"λ={lam}", y=1.02, fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_fields(args):
    """Quiver plots of the first-action displacement per factor and lambda, with factor k dominant:
    eps = (F_k + alpha * sum_{j != k} F_j) / K.
    """
    device = torch.device(args.device)
    schedule = DDPMSchedule(num_timesteps=args.num_diffusion_steps, device=device)
    starts_1d = np.linspace(-0.9, 0.9, args.grid_n)
    gx, gy = np.meshgrid(starts_1d, starts_1d, indexing="xy")
    starts = np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)
    goal = np.array(args.goal, dtype=np.float32)[:STATE_DIM]
    goals = np.repeat(goal[None, :], starts.shape[0], axis=0)
    obs = torch.from_numpy(np.concatenate([starts, goals], axis=-1)).to(device)

    out_root = pathlib.Path(args.out_dir)
    B = obs.shape[0]
    alpha = float(args.alpha)

    for lam in args.lambdas:
        model = load_model(
            pathlib.Path(args.ckpt_dir) / f"lambda_{lam}" / "final.pt", device, K=args.K,
        )

        composed_acc = torch.zeros(B, STATE_DIM, device=device)
        factor_acc = [torch.zeros(B, STATE_DIM, device=device) for _ in range(args.K)]
        start_t = obs[:, :STATE_DIM]

        for s_idx in range(args.n_avg):
            seed = args.seed + s_idx
            traj_full = ddpm_sample(model, schedule, obs, device, seed=seed)
            composed_acc += traj_full[:, 0, :] - start_t
            for k in range(args.K):
                weights = [alpha] * args.K
                weights[k] = 1.0
                traj_dom = ddpm_sample_weighted(
                    model, schedule, obs, device,
                    weights=weights, K=args.K, seed=seed,
                )
                factor_acc[k] += traj_dom[:, 0, :] - start_t

        composed_field = (composed_acc / args.n_avg).cpu().numpy()
        factor_fields = [(factor_acc[k] / args.n_avg).cpu().numpy() for k in range(args.K)]

        lam_padded = f"{float(lam):.3f}"
        _save_quiver_panel(
            out_root / "composed" / f"lambda_{lam_padded}.png",
            starts, composed_field,
            title=f"composed   λ={lam}",
            goal=goal, quiver_scale=args.composed_quiver_scale,
        )
        for k in range(args.K):
            _save_quiver_panel(
                out_root / f"factor_{k}" / f"lambda_{lam_padded}.png",
                starts, factor_fields[k],
                title=f"factor {k} dominant (α={alpha})   λ={lam}",
                goal=goal, quiver_scale=args.factor_quiver_scale,
            )
        _save_quiver_combined(
            out_root / "combined" / f"lambda_{lam_padded}.png",
            starts, composed_field, factor_fields,
            lam=lam, alpha=alpha, goal=goal,
            composed_quiver_scale=args.composed_quiver_scale,
            factor_quiver_scale=args.factor_quiver_scale,
        )
        print(f"λ={lam}: saved composed + {args.K} dominant-factor + combined plots under {out_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rollouts", "fields"], required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--lambdas", nargs="+", default=["0", "0.1", "1"])
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--num_diffusion_steps", type=int, default=100)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--oracle_device", type=str, default="cpu")
    ap.add_argument("--dataset_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_examples", type=int, default=4)
    ap.add_argument("--grid_n", type=int, default=13)
    ap.add_argument("--goal", type=float, nargs="+", default=[0.0, 0.0])
    ap.add_argument("--n_avg", type=int, default=6,
                    help="number of noise seeds to average per rollout (fields mode)")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="scale applied to non-dominant factors (fields mode)")
    ap.add_argument("--composed_quiver_scale", type=float, default=1.2)
    ap.add_argument("--factor_quiver_scale", type=float, default=1.2)
    ap.add_argument("--out_path", default="outputs/viz.png",
                    help="output PNG path (rollouts mode)")
    ap.add_argument("--out_dir", default="outputs/2D_traj",
                    help="output root directory (fields mode)")
    args = ap.parse_args()

    if args.mode == "rollouts":
        plot_rollouts(args)
    else:
        plot_fields(args)


if __name__ == "__main__":
    main()
