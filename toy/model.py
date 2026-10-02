"""K factor MLPs composed as eps = sum_k w_k F_k, with w from a router or w = 1/K."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import HORIZON, OBS_DIM, STATE_DIM


class FactorComponent(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        in_dim = HORIZON * STATE_DIM + OBS_DIM + 1
        out_dim = HORIZON * STATE_DIM
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, traj_flat, obs, t_norm):
        return self.net(torch.cat([traj_flat, obs, t_norm], dim=-1))


class Router(nn.Module):
    """FDP's router: obs -> K weights (Linear, Mish, Linear, over a learned temperature).

    affine   w = logits - mean(logits) + 1/K   (sums to 1, signed)
    softmax  w = softmax(logits)               (sums to 1, non-negative)
    neither  w = logits
    Starts near uniform (1/K each).
    """

    def __init__(self, obs_dim, K, hidden=64, softmax=True, affine=False):
        super().__init__()
        self.softmax = softmax
        self.affine = affine
        self.K = K
        self.backbone = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Mish())
        self.head = nn.Linear(hidden, K)
        self.temperature = nn.Parameter(torch.ones(1))
        nn.init.zeros_(self.head.weight)  # obs-independent at init
        if affine or softmax:
            # affine: any constant bias is mean-centered out -> w = 1/K regardless.
            # softmax: logits 0 -> uniform.
            nn.init.zeros_(self.head.bias)
        else:
            # raw weights ~1/K each at init (logits/temp ~ 1/K, temp~softplus(1)+1e-2)
            nn.init.constant_(self.head.bias, (1.0 / K) * (F.softplus(torch.ones(1)).item() + 1e-2))

    def forward(self, obs):
        logits = self.head(self.backbone(obs))
        logits = logits / (F.softplus(self.temperature) + 1e-2)
        if self.affine:
            return logits - logits.mean(dim=-1, keepdim=True) + 1.0 / self.K
        return torch.softmax(logits, dim=-1) if self.softmax else logits


class CompositionalDiffusionPolicy(nn.Module):
    """Composed noise prediction = router/mean of K factor noise predictions."""

    def __init__(self, K=2, hidden=256, use_router=False, softmax_weights=True,
                 affine_weights=False, router_hidden=64):
        super().__init__()
        self.K = K
        self.use_router = use_router
        self.components = nn.ModuleList([FactorComponent(hidden) for _ in range(K)])
        self.router = Router(OBS_DIM, K, hidden=router_hidden,
                             softmax=softmax_weights, affine=affine_weights) if use_router else None

    def forward(self, traj_noisy, obs, t_norm):
        batch = traj_noisy.shape[0]
        traj_flat = traj_noisy.reshape(batch, -1)
        eps_per = torch.stack(
            [c(traj_flat, obs, t_norm) for c in self.components], dim=1
        )  # (B, K, HORIZON*STATE_DIM)
        eps_per = eps_per.reshape(batch, self.K, HORIZON, STATE_DIM)
        if self.use_router:
            w = self.router(obs)                          # (B, K)
            eps = (w[:, :, None, None] * eps_per).sum(dim=1)
        else:
            eps = eps_per.mean(dim=1)
        return eps, eps_per


def _pairwise_cos(eps_norm):
    return torch.matmul(eps_norm, eps_norm.transpose(-1, -2))


def decorrelation_cos2_per_example(eps_per):
    """Per-example cos^2 between factors at each horizon step, averaged over steps and pairs. (B,)"""
    _, K, _, _ = eps_per.shape
    mask = torch.triu(torch.ones(K, K, device=eps_per.device, dtype=torch.bool), diagonal=1)
    norm = F.normalize(eps_per, dim=-1).permute(0, 2, 1, 3).contiguous()
    cos = _pairwise_cos(norm)
    return (cos[..., mask] ** 2).mean(dim=(1, 2))


def decorrelation_loss_cos2(eps_per):
    """Mean per-timestep cos^2 between distinct factors' noise predictions (scalar)."""
    return decorrelation_cos2_per_example(eps_per).mean()


def abs_cos_diagnostic(eps_per):
    """Mean |cos| between distinct factors' noise predictions, per timestep."""
    _, K, _, _ = eps_per.shape
    mask = torch.triu(torch.ones(K, K, device=eps_per.device, dtype=torch.bool), diagonal=1)
    norm = F.normalize(eps_per, dim=-1).permute(0, 2, 1, 3).contiguous()
    cos = _pairwise_cos(norm)
    return cos[..., mask].abs().mean()


def decorrelation_loss_flat_abs(eps_per):
    """|cos| between the flattened (H*D) factor predictions, mean over pairs and batch."""
    B, K = eps_per.shape[0], eps_per.shape[1]
    mask = torch.triu(torch.ones(K, K, device=eps_per.device, dtype=torch.bool), diagonal=1)
    flat = eps_per.reshape(B, K, -1)
    norm = F.normalize(flat, dim=-1)
    cos = torch.matmul(norm, norm.transpose(-1, -2))
    return cos[..., mask].abs().mean()


def decorrelation_loss_flat_abs_envelope(eps_per):
    """|cos| between the flattened elementwise-|.| envelopes, so opposite signs cannot cancel."""
    B, K = eps_per.shape[0], eps_per.shape[1]
    mask = torch.triu(torch.ones(K, K, device=eps_per.device, dtype=torch.bool), diagonal=1)
    env = eps_per.reshape(B, K, -1).abs()
    norm = F.normalize(env, dim=-1)
    cos = torch.matmul(norm, norm.transpose(-1, -2))
    return cos[..., mask].mean()


def decorrelation_loss_flat_cos2_envelope(eps_per):
    """The paper's loss: cos^2 between the flattened elementwise-|.| envelopes, mean over pairs."""
    B, K = eps_per.shape[0], eps_per.shape[1]
    mask = torch.triu(torch.ones(K, K, device=eps_per.device, dtype=torch.bool), diagonal=1)
    env = eps_per.reshape(B, K, -1).abs()
    norm = F.normalize(env, dim=-1)
    cos = torch.matmul(norm, norm.transpose(-1, -2))
    return (cos[..., mask] ** 2).mean()


def decorrelation_loss_flat_sum_cos2(eps_per):
    """cos^2 between the flattened predictions, summed over ordered pairs, mean over batch."""
    B, K = eps_per.shape[0], eps_per.shape[1]
    off_diag = ~torch.eye(K, dtype=torch.bool, device=eps_per.device)
    flat = eps_per.reshape(B, K, -1)
    norm = F.normalize(flat, dim=-1)
    cos = torch.matmul(norm, norm.transpose(-1, -2))           # (B, K, K)
    return (cos[..., off_diag] ** 2).sum(dim=-1).mean()


def decorrelation_loss_flat_sum_relu_cos2(eps_per):
    """As flat_sum_cos2 but penalizing only positive cosines (anti-aligned factors are free)."""
    B, K = eps_per.shape[0], eps_per.shape[1]
    off_diag = ~torch.eye(K, dtype=torch.bool, device=eps_per.device)
    flat = eps_per.reshape(B, K, -1)
    norm = F.normalize(flat, dim=-1)
    cos = torch.matmul(norm, norm.transpose(-1, -2))           # (B, K, K)
    return (cos.clamp(min=0)[..., off_diag] ** 2).sum(dim=-1).mean()


def decorrelation_loss_flat_sum_abs(eps_per):
    """|cos| between the flattened predictions, summed over ordered pairs, mean over batch."""
    B, K = eps_per.shape[0], eps_per.shape[1]
    off_diag = ~torch.eye(K, dtype=torch.bool, device=eps_per.device)
    flat = eps_per.reshape(B, K, -1)
    norm = F.normalize(flat, dim=-1)
    cos = torch.matmul(norm, norm.transpose(-1, -2))           # (B, K, K)
    return cos[..., off_diag].abs().sum(dim=-1).mean()


def decorrelation_loss_fc_cos2(eps_per, normalize=True):
    """cos^2 between the batch-mean predictions of each factor (only the average directions must differ)."""
    B, K = eps_per.shape[0], eps_per.shape[1]
    mu = eps_per.reshape(B, K, -1).mean(dim=0)                  # (K, H*D)
    if normalize:
        mu = mu / (mu.norm(dim=-1, keepdim=True) + 1e-8)
    gram = mu @ mu.transpose(-1, -2)                            # (K, K)
    off_diag = ~torch.eye(K, dtype=torch.bool, device=eps_per.device)
    return (gram[off_diag] ** 2).sum() / (K * (K - 1))
