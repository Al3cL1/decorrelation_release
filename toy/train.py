"""Train the compositional score model on 2D start-to-target trajectories."""

import argparse
import json
import logging
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import HORIZON, ProblemSampler, STATE_DIM
from diffusion import DDPMSchedule
from model import (
    CompositionalDiffusionPolicy,
    abs_cos_diagnostic,
    decorrelation_cos2_per_example,
    decorrelation_loss_flat_abs,
    decorrelation_loss_flat_abs_envelope,
    decorrelation_loss_flat_sum_cos2,
    decorrelation_loss_flat_sum_relu_cos2,
    decorrelation_loss_flat_sum_abs,
    decorrelation_loss_fc_cos2,
    decorrelation_loss_flat_cos2_envelope,
)


def setup_logging(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
        force=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lambda_decorr", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_steps", type=int, default=6000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num_diffusion_steps", type=int, default=100)
    p.add_argument("--dataset_size", type=int, default=4096)
    p.add_argument("--oracle_device", type=str, default="cpu")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Router options (FDP-style). Default off -> uniform-mean composition.
    p.add_argument("--use_router", action="store_true",
                   help="Compose factors with a learned router instead of a uniform mean.")
    p.add_argument("--softmax_weights", action="store_true",
                   help="Router weights pass through softmax (sum to 1, non-negative). "
                        "Mutex with --affine_weights. Off (and --affine_weights off) = unconstrained.")
    p.add_argument("--affine_weights", action="store_true",
                   help="Router weights mean-centered + 1/K shift (sum to 1, signed allowed). "
                        "Mutex with --softmax_weights.")
    p.add_argument("--weighted_decorr", action="store_true",
                   help="Weight each pair's cos^2 by sg(w_i*w_j) (requires --use_router).")
    p.add_argument("--loss_mode", choices=["per_step_cos2", "flat_abs", "flat_abs_envelope",
                                            "flat_sum_cos2", "flat_sum_relu_cos2",
                                            "flat_sum_abs", "fc_cos2",
                                            "flat_cos2_envelope"],
                   default="per_step_cos2",
                   help="Decorrelation loss formulation. flat_abs = "
                        "flat |cos|, mean over pairs. flat_abs_envelope = flat |cos| on "
                        "the elementwise-|.| envelope, removes per-timestep sign cancellation. "
                        "flat_sum_relu_cos2 = one-sided cos²: max(0, cos)², allowing "
                        "anti-aligned (cos < 0) factors with zero penalty. "
                        "fc_cos2 = functional-connectivity orthogonality on BATCH-MEAN factor "
                        "predictions (weak per-example, only avg direction must differ).")
    p.add_argument("--bg_mode", choices=["none", "marginal"], default="none",
                   help="Subtract a shared marginal background from eps_per before decorrelation. "
                        "marginal = MC estimate of E_c'[mean_i F_i(x_t, c', t)] via --bg_samples "
                        "extra forwards with arbitrary obs drawn from the training distribution; "
                        "decorrelation then operates on r_i = F_i(x_t, c, t) - bg_shared(x_t, t).")
    p.add_argument("--bg_samples", type=int, default=5,
                   help="Number of arbitrary-task forwards per batch element to average for the "
                        "marginal background (only used when --bg_mode marginal).")
    args = p.parse_args()
    if args.loss_mode in ("flat_abs", "flat_abs_envelope",
                           "flat_sum_cos2", "flat_sum_relu_cos2",
                           "flat_sum_abs", "fc_cos2",
                           "flat_cos2_envelope") and args.weighted_decorr:
        raise SystemExit(f"--weighted_decorr is not supported with --loss_mode {args.loss_mode} "
                         "(unweighted recipe).")
    if args.softmax_weights and args.affine_weights:
        raise SystemExit("--softmax_weights and --affine_weights are mutually exclusive.")
    if args.bg_mode != "none" and args.bg_samples < 1:
        raise SystemExit("--bg_samples must be >= 1 when --bg_mode is not none.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tag = f"lambda_{args.lambda_decorr:g}"
    out_dir = pathlib.Path(args.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "train.log")
    logging.info(f"args: {vars(args)}")

    sampler = ProblemSampler(
        dataset_size=args.dataset_size,
        seed=args.seed,
        oracle_device=args.oracle_device,
    )
    device = torch.device(args.device)
    model = CompositionalDiffusionPolicy(
        K=args.K, hidden=args.hidden,
        use_router=args.use_router,
        softmax_weights=args.softmax_weights,
        affine_weights=args.affine_weights,
    ).to(device)
    schedule = DDPMSchedule(num_timesteps=args.num_diffusion_steps, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    t0 = time.time()
    for step in range(args.num_steps):
        batch = sampler.sample_batch(args.batch_size)
        obs = batch.obs.to(device)
        traj = batch.traj.to(device)

        t = torch.randint(0, args.num_diffusion_steps, (args.batch_size,), device=device)
        noise = torch.randn_like(traj)
        traj_noisy = schedule.add_noise(traj, noise, t)
        t_norm = (t.float() / args.num_diffusion_steps).unsqueeze(-1)

        eps, eps_per = model(traj_noisy, obs, t_norm)
        mse = F.mse_loss(eps, noise)

        if args.bg_mode == "marginal":
            # background score: mean factor prediction over M other observations
            M = args.bg_samples
            bg_batch = sampler.sample_batch(args.batch_size * M)
            obs_bg = bg_batch.obs.to(device)                        # (B*M, OBS_DIM)
            traj_bg = traj_noisy.repeat_interleave(M, dim=0)        # (B*M, H, D)
            t_bg = t_norm.repeat_interleave(M, dim=0)               # (B*M, 1)
            _, eps_per_bg_all = model(traj_bg, obs_bg, t_bg)        # (B*M, K, H, D)
            eps_per_bg = eps_per_bg_all.view(
                args.batch_size, M, args.K, HORIZON, STATE_DIM
            ).mean(dim=1)                                            # (B, K, H, D) avg over M
            bg_shared = eps_per_bg.mean(dim=1, keepdim=True)        # (B, 1, H, D) avg over K
            decorr_input = eps_per - bg_shared                      # broadcast -> (B, K, H, D)
        else:
            decorr_input = eps_per

        if args.loss_mode == "flat_abs":
            decorr = decorrelation_loss_flat_abs(decorr_input)
        elif args.loss_mode == "flat_abs_envelope":
            decorr = decorrelation_loss_flat_abs_envelope(decorr_input)
        elif args.loss_mode == "flat_sum_cos2":
            decorr = decorrelation_loss_flat_sum_cos2(decorr_input)
        elif args.loss_mode == "flat_sum_relu_cos2":
            decorr = decorrelation_loss_flat_sum_relu_cos2(decorr_input)
        elif args.loss_mode == "flat_sum_abs":
            decorr = decorrelation_loss_flat_sum_abs(decorr_input)
        elif args.loss_mode == "fc_cos2":
            decorr = decorrelation_loss_fc_cos2(decorr_input)
        elif args.loss_mode == "flat_cos2_envelope":
            decorr = decorrelation_loss_flat_cos2_envelope(decorr_input)
        else:
            decorr_pe = decorrelation_cos2_per_example(decorr_input)  # (B,)
            if args.weighted_decorr:
                # L = lambda * E[ sg(w_0 w_1) cos^2 ];  stop-grad so decorr never moves the router.
                w = model.router(obs)
                w_prod = (w[:, 0] * w[:, 1]).detach()
                decorr = (w_prod * decorr_pe).mean()
            else:
                decorr = decorr_pe.mean()
        loss = mse + args.lambda_decorr * decorr

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % args.log_every == 0:
            abs_cos = abs_cos_diagnostic(eps_per).item()
            elapsed = time.time() - t0
            rec = {
                "step": step + 1,
                "mse": mse.item(),
                "decorr_cos2": decorr.item(),
                "abs_cos": abs_cos,
                "total": loss.item(),
                "elapsed": elapsed,
            }
            if args.use_router:
                with torch.no_grad():
                    w_mean = model.router(obs).mean(dim=0)
                rec["w_mean"] = [round(x, 4) for x in w_mean.tolist()]
            extra = ""
            if args.bg_mode == "marginal":
                with torch.no_grad():
                    bg_norm = bg_shared.flatten(start_dim=1).norm(dim=-1).mean().item()
                    r_norm = decorr_input.flatten(start_dim=2).norm(dim=-1).mean().item()
                rec["bg_norm"] = bg_norm
                rec["r_norm"] = r_norm
                extra = f" | ||bg||={bg_norm:.3f} ||r||={r_norm:.3f}"
            logging.info(
                f"step {step+1:6d} | mse {mse.item():.4f} | "
                f"decorr_cos2 {decorr.item():.4f} | abs_cos {abs_cos:.4f} | "
                f"total {loss.item():.4f} | elapsed {elapsed:.1f}s"
                + (f" | w_mean {rec.get('w_mean')}" if args.use_router else "")
                + extra
            )
            history.append(rec)

    ckpt = {"state_dict": model.state_dict(), "args": vars(args), "history": history}
    torch.save(ckpt, out_dir / "final.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    logging.info(f"saved checkpoint to {out_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
