"""Structural diagnostics of a trained FDP checkpoint on the held-out validation split.

For every validation window and inference timestep, takes the K per-factor eps predictions r_i
and the router weights w_i, and reports:
    signed / abs cosine matrices   E[cos(r_i, r_j)], E[|cos(r_i, r_j)|]
    double_counting                E[2 sum_{i<j} w_i w_j <r_i, r_j>]
    d_rel / i_rel                  that cross term (signed / absolute) over sum_i w_i^2 ||r_i||^2
    effective_rank                 exp(entropy of the normalized eigenvalues of E[<r_i, r_j>])

Usage:
    python diagnostics.py -c path/to/ckpt -o out_dir/
"""
import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import json
import click
import dill
import hydra
import omegaconf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# Match train.py: the saved cfg may carry ${eval:...} interpolations.
OmegaConf.register_new_resolver("eval", eval, replace=True)

EPS = 1e-8


def _to_device(x, device):
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x


def _fix_crop_randomizers(policy):
    """Deterministic center-crop at eval time (mirrors eval.py)."""
    types = []
    try:
        import robomimic.models.base_nets as rmbn
        types.append(rmbn.CropRandomizer)
    except Exception:
        pass
    try:
        from fdp.perception.crop_randomizer import CropRandomizer as FdpCrop
        types.append(FdpCrop)
    except Exception:
        pass
    if not types:
        return
    for _, m in policy.named_modules():
        if isinstance(m, tuple(types)):
            m.eval()


def _update(acc, eps, w):
    """Accumulate metric sums from one (batch, tau) slice.
    eps: (B,K,H,D) raw per-expert preds;  w: (B,K) router weights."""
    eps = eps.float()
    w = w.float()
    B, K, H, D = eps.shape

    # --- Section 1: per-(b,h) pairwise cosine ---
    norm = F.normalize(eps, dim=-1).permute(0, 2, 1, 3)   # (B,H,K,D)
    cos = norm @ norm.transpose(-1, -2)                   # (B,H,K,K)
    acc["signed_cos"] += cos.sum(dim=(0, 1))              # (K,K)
    acc["abs_cos"] += cos.abs().sum(dim=(0, 1))           # (K,K)

    # --- Section 2: per-(b,h) weighted interference ---
    ep = eps.permute(0, 2, 1, 3)                          # (B,H,K,D)
    g = ep @ ep.transpose(-1, -2)                         # (B,H,K,K)  <r_i,r_j>
    ww = (w[:, :, None] * w[:, None, :]).unsqueeze(1)     # (B,1,K,K)  w_i w_j
    gw = ww * g                                           # (B,H,K,K)
    energy = gw.diagonal(dim1=-2, dim2=-1).sum(-1)        # (B,H)  sum_i w_i^2 ||r_i||^2
    dc = gw.sum(dim=(-1, -2)) - energy                    # (B,H)  2 sum_{i<j} w_i w_j <r_i,r_j>
    i_numer = (ww * g.abs()).sum(dim=(-1, -2)) - energy   # (B,H)  2 sum_{i<j} |w_i w_j <r_i,r_j>|
    denom = energy + EPS
    acc["dc_sum"] += dc.sum().item()
    acc["drel_sum"] += (dc / denom).sum().item()
    acc["irel_sum"] += (i_numer / denom).sum().item()
    acc["count_bh"] += B * H

    # --- Section 3: effective-rank covariance, per-(b,tau) ---
    flat = eps.reshape(B, K, H * D)                       # (B,K,H*D)
    acc["sigma"] += (flat @ flat.transpose(-1, -2)).sum(0)  # (K,K)
    acc["count_b"] += B

    # --- mean router activation (the true softmax simplex), per-(b,tau) ---
    acc["w_sum"] += w.sum(0)                              # (K,)


def _finalize(acc, K):
    signed = (acc["signed_cos"] / acc["count_bh"])
    abscos = (acc["abs_cos"] / acc["count_bh"])
    sigma = acc["sigma"] / acc["count_b"]
    evals = torch.linalg.eigvalsh(sigma).clamp(min=0)
    p = evals / (evals.sum() + EPS)
    entropy = -(p * torch.log(p + EPS)).sum()
    return {
        "signed_cosine_matrix": signed.tolist(),
        "abs_cosine_matrix": abscos.tolist(),
        "double_counting": acc["dc_sum"] / acc["count_bh"],
        "d_rel": acc["drel_sum"] / acc["count_bh"],
        "i_rel": acc["irel_sum"] / acc["count_bh"],
        "effective_rank": torch.exp(entropy).item(),
        "covariance_matrix": sigma.tolist(),
        "covariance_eigenvalues": evals.tolist(),
        "mean_router_weight": (acc["w_sum"] / acc["count_b"]).tolist(),
    }


@click.command()
@click.option("-c", "--checkpoint", required=True, help="path to a .ckpt file")
@click.option("-o", "--output_dir", required=True, help="directory for <name>_diagnostics.json")
@click.option("-d", "--device", default="cuda:0", help="device to run on")
@click.option("--n_diffusion_steps", default=None, type=int,
              help="override #denoising steps swept (default: policy's num_inference_steps)")
@click.option("--max_batches", default=None, type=int,
              help="cap #val batches (default: all)")
def main(checkpoint, output_dir, device, n_diffusion_steps, max_batches):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    name = os.path.basename(os.path.dirname(os.path.dirname(checkpoint)))
    device = torch.device(device)

    # --- load checkpoint -> workspace -> policy (prefer EMA, mirrors eval.py) ---
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir, lazy_instantiation=False)


    workspace.load_payload(payload, exclude_keys=["optimizer"], include_keys=None)

    if workspace.ema_model is not None \
            and "ema_model" not in payload.get("state_dicts", {}):
        workspace.ema_model.load_state_dict(workspace.model.state_dict())
    policy = workspace.model
    try:
        if cfg.training.use_ema and workspace.ema_model is not None:
            policy = workspace.ema_model
            print("    using EMA weights")
    except omegaconf.errors.ConfigAttributeError:
        pass
    _fix_crop_randomizers(policy)
    policy.to(device)
    policy.eval()

    if not hasattr(policy, "diagnostic_components"):
        raise click.ClickException(
            f"policy {type(policy).__name__} has no diagnostic_components()")

    # --- held-out val split of the training zarr ---
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    val_dataset = dataset.get_validation_dataset()
    batch_size = int(cfg.val_dataloader.batch_size)
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"    val windows: {len(val_dataset)}  (batch_size={batch_size})")

    # --- diffusion-step schedule ---
    n_steps = n_diffusion_steps or policy.num_inference_steps
    policy.noise_scheduler.set_timesteps(n_steps)
    timesteps = [int(t) for t in policy.noise_scheduler.timesteps]
    print(f"    sweeping {len(timesteps)} denoising steps: {timesteps}")

    acc = {"signed_cos": 0.0, "abs_cos": 0.0, "sigma": 0.0, "w_sum": 0.0,
           "dc_sum": 0.0, "drel_sum": 0.0, "irel_sum": 0.0,
           "count_bh": 0, "count_b": 0}
    K = None
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            batch = _to_device(batch, device)
            B = batch["action"].shape[0]
            for t in timesteps:
                ts = torch.full((B,), t, device=device, dtype=torch.long)
                eps, w = policy.diagnostic_components(batch, ts)
                if K is None:
                    K = eps.shape[1]
                _update(acc, eps, w)
            print(f"    batch {bi + 1}: done", flush=True)

    if K is None:
        raise click.ClickException("no val data — nothing to diagnose")

    metrics = _finalize(acc, K)
    out = {
        "name": name,
        "checkpoint": checkpoint,
        "data": "val_split",
        "K": K,
        "n_val_windows": len(val_dataset),
        "n_diffusion_steps": len(timesteps),
        "diffusion_timesteps": timesteps,
        **metrics,
    }
    out_path = os.path.join(output_dir, f"{name}_diagnostics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"    effective_rank = {metrics['effective_rank']:.3f}  (K={K})")
    print(f"    d_rel = {metrics['d_rel']:.4f}   i_rel = {metrics['i_rel']:.4f}")
    print(f"    mean_router_weight = {[round(x, 3) for x in metrics['mean_router_weight']]}")
    print(f"    wrote {out_path}")


if __name__ == "__main__":
    main()
