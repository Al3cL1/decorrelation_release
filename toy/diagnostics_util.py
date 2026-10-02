"""Structural diagnostics for the toy, the same metrics as diagnostics.py at the repo root:
cosine matrices, double counting, d_rel / i_rel, effective rank and the mean router weight,
accumulated over sampling steps.
"""
from __future__ import annotations

from typing import Iterable, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-8


class DiagnosticsAccumulator:
    """Accumulates the diagnostics across diffusion steps.

        acc = DiagnosticsAccumulator()
        for t in reversed(range(n_steps)):
            _, eps_per = model(x, obs, t_norm)
            acc.update(eps_per, model.router(obs) if model.router is not None else None)
            ...
        metrics = acc.finalize()
    """

    def __init__(self):
        self.signed_cos = None     # (K, K)
        self.abs_cos = None        # (K, K)
        self.sigma = None          # (K, K)  E[<r_i, r_j>] over (b, h, t)
        self.w_sum = None          # (K,)
        self.dc_sum = 0.0
        self.drel_sum = 0.0
        self.irel_sum = 0.0
        self.count_bh = 0
        self.count_b = 0

    @torch.no_grad()
    def update(self, eps_per: torch.Tensor, weights: torch.Tensor | None):
        """eps_per: (B, K, H, D);  weights: (B, K) or None (uniform 1/K)."""
        eps = eps_per.detach().float()
        B, K, H, D = eps.shape
        if weights is None:
            w = torch.full((B, K), 1.0 / K, device=eps.device, dtype=torch.float32)
        else:
            w = weights.detach().float()

        # init lazy K-sized accumulators
        if self.signed_cos is None:
            self.signed_cos = torch.zeros(K, K, device=eps.device, dtype=torch.float32)
            self.abs_cos = torch.zeros(K, K, device=eps.device, dtype=torch.float32)
            self.sigma = torch.zeros(K, K, device=eps.device, dtype=torch.float32)
            self.w_sum = torch.zeros(K, device=eps.device, dtype=torch.float32)

        # per-(b, h) pairwise cosine on the D-vector
        norm = F.normalize(eps, dim=-1).permute(0, 2, 1, 3)         # (B, H, K, D)
        cos = norm @ norm.transpose(-1, -2)                          # (B, H, K, K)
        self.signed_cos = self.signed_cos + cos.sum(dim=(0, 1))
        self.abs_cos = self.abs_cos + cos.abs().sum(dim=(0, 1))

        # per-(b, h) weighted interference / double-counting
        ep = eps.permute(0, 2, 1, 3)                                 # (B, H, K, D)
        g = ep @ ep.transpose(-1, -2)                                # (B, H, K, K) <r_i, r_j>
        ww = (w[:, :, None] * w[:, None, :]).unsqueeze(1)            # (B, 1, K, K)
        gw = ww * g                                                  # (B, H, K, K)
        energy = gw.diagonal(dim1=-2, dim2=-1).sum(-1)               # (B, H)
        dc = gw.sum(dim=(-1, -2)) - energy                           # (B, H)
        i_num = (ww * g.abs()).sum(dim=(-1, -2)) - energy            # (B, H)
        denom = energy + EPS
        self.dc_sum += dc.sum().item()
        self.drel_sum += (dc / denom).sum().item()
        self.irel_sum += (i_num / denom).sum().item()
        self.count_bh += B * H

        # effective-rank covariance: per-(b) flatten (H, D) -> (K, H*D), <r, r>
        flat = eps.reshape(B, K, H * D)
        self.sigma = self.sigma + (flat @ flat.transpose(-1, -2)).sum(0)
        self.count_b += B

        # router activation
        self.w_sum = self.w_sum + w.sum(0) / H if False else self.w_sum + w.sum(0)

    def finalize(self) -> Dict:
        if self.count_bh == 0:
            return {}
        K = self.signed_cos.shape[0]
        signed = (self.signed_cos / self.count_bh).cpu().numpy()
        absc = (self.abs_cos / self.count_bh).cpu().numpy()
        sigma = (self.sigma / self.count_b).cpu().numpy()
        evals = np.clip(np.linalg.eigvalsh(sigma), 0, None)
        p = evals / (evals.sum() + EPS)
        entropy = -np.sum(p * np.log(p + EPS))
        # off-diag |cos| convenience scalar
        mask = ~np.eye(K, dtype=bool)
        avg_abs_cos = float(absc[mask].mean()) if K > 1 else float("nan")
        return {
            "K": K,
            "signed_cosine_matrix": signed.tolist(),
            "abs_cosine_matrix": absc.tolist(),
            "avg_abs_cos_offdiag": avg_abs_cos,
            "double_counting": self.dc_sum / self.count_bh,
            "d_rel": self.drel_sum / self.count_bh,
            "i_rel": self.irel_sum / self.count_bh,
            "effective_rank": float(np.exp(entropy)),
            "covariance_matrix": sigma.tolist(),
            "covariance_eigenvalues": evals.tolist(),
            "mean_router_weight": (self.w_sum / self.count_b).cpu().numpy().tolist(),
            "n_samples_bh": self.count_bh,
        }
