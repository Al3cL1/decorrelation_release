"""DDPM linear-beta schedule and add_noise."""
import torch


class DDPMSchedule:
    def __init__(self, num_timesteps=100, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x0, noise, t):
        a = self.sqrt_alphas_cumprod[t]
        b = self.sqrt_one_minus_alphas_cumprod[t]
        while a.dim() < x0.dim():
            a = a.unsqueeze(-1)
            b = b.unsqueeze(-1)
        return a * x0 + b * noise
