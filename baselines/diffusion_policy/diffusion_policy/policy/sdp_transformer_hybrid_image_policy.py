"""SDP (Sparse Diffusion Policy) baseline on SDP's own MoE code (github.com/AnthonyHuo/SDP).

Uses SDP's experts and MoE transformer verbatim with its capacity setting (8 experts of width
dim_feedforward // 2, top-2), so active FFN parameters match dense DP. One change: a single
observation-conditioned gate (task_num=1) instead of per-task gates, since batches here mix
tasks; the task id still reaches the gate through the observation.
"""
import torch

from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import (
    DiffusionTransformerHybridImagePolicy,
)
from diffusion_policy.model.diffusion.transformer_for_diffusion_moe import (
    TransformerForDiffusion as MoETransformerForDiffusion,
)


class SDPTransformerHybridImagePolicy(DiffusionTransformerHybridImagePolicy):
    def __init__(self, *args, n_tasks: int = 1, w_MI: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)

        dense = self.model
        # Recover the dims the dense model was built with, so the MoE backbone is
        # identical except for the FFN.
        input_dim = dense.input_emb.in_features
        output_dim = dense.head.out_features
        cond_dim = dense.cond_obs_emb.in_features
        n_emb = dense.input_emb.out_features

        self.model = MoETransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=kwargs["horizon"],
            n_obs_steps=kwargs["n_obs_steps"],
            cond_dim=cond_dim,
            n_tasks=n_tasks,
            n_layer=kwargs.get("n_layer", 8),
            n_head=kwargs.get("n_head", 4),
            n_emb=n_emb,
            p_drop_emb=kwargs.get("p_drop_emb", 0.0),
            p_drop_attn=kwargs.get("p_drop_attn", 0.3),
            causal_attn=kwargs.get("causal_attn", True),
            time_as_cond=kwargs.get("time_as_cond", True),
            obs_as_cond=kwargs.get("obs_as_cond", True),
            n_cond_layers=kwargs.get("n_cond_layers", 0),
        )
        self.n_tasks = n_tasks
        for m in self.model.modules():
            if hasattr(m, "w_MI"):
                m.w_MI = w_MI

        n_dense = sum(p.numel() for p in dense.parameters())
        n_total = sum(p.numel() for p in self.model.parameters())
        moes = [m for m in self.model.modules() if hasattr(m, "output_experts")]
        if moes:
            # count both expert projections (in->hidden and hidden->in)
            ex = sum(p.numel() for p in moes[0].experts.parameters()) \
               + sum(p.numel() for p in moes[0].output_experts.parameters())
            per_expert = ex / moes[0].num_experts
            n_inactive = len(moes) * (moes[0].num_experts - moes[0].k) * per_expert
            print(f"    [SDP] {moes[0].num_experts} experts, top-{moes[0].k}, "
                  f"{len(moes)} layers, task_num={n_tasks}, w_MI={w_MI}")
            print(f"    [SDP] trunk params: active={(n_total - n_inactive)/1e6:.3f}M  "
                  f"total={n_total/1e6:.3f}M   (dense DP trunk={n_dense/1e6:.3f}M)")

    # the MoE backbone also returns SDP's auxiliary routing loss, added at train time

    def conditional_sample(self, condition_data, condition_mask,
                           cond=None, generator=None, **kwargs):
        model = self.model
        scheduler = self.noise_scheduler
        trajectory = torch.randn(
            size=condition_data.shape, dtype=condition_data.dtype,
            device=condition_data.device, generator=generator)
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output, _aux, _probs = model(trajectory, t, cond, task_id=0)
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator, **kwargs).prev_sample
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def compute_loss(self, batch):
        # Swap in a shim so the parent's compute_loss, which expects a single
        # tensor from self.model, keeps working unchanged; capture the aux loss.
        real = self.model
        captured = {}

        class _Shim(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, sample, timestep, cond=None, **kw):
                out, aux, _probs = self.inner(sample, timestep, cond, task_id=0)
                captured["aux"] = aux
                return out

        object.__setattr__(self, "model", _Shim(real))
        try:
            loss = super().compute_loss(batch)
        finally:
            object.__setattr__(self, "model", real)

        aux = captured.get("aux", 0.0)
        if torch.is_tensor(aux):
            loss = loss + aux
        return loss
