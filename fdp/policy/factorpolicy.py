import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from fdp.policy.base_policy import BasePolicy
from fdp.perception.base_obs_encoder import BaseObservationEncoder
from fdp.model.common.normalizer import LinearNormalizer
from fdp.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from fdp.model.diffusion.conditional_unet1d import ConditionalUnet1D
from fdp.model.common.positional_embedding import SinusoidalPosEmb

from typing import List, Dict, Optional, Tuple, Union

"""
formulation: A ~ Π Pi(A | O) via energy-based composition
"""


def apply_weight_mode(logits: torch.Tensor,
                       softmax_weights: bool,
                       affine_weights: bool,
                       dim: int = 0) -> torch.Tensor:
    """Map per-factor router logits to composition weights; `dim` is the factor axis.

    affine_weights   w = logits - mean(logits) + 1/K   (sums to 1, signed)
    softmax_weights  w = softmax(logits)               (sums to 1, non-negative)
    neither          w = logits                        (FDP's original router)
    """
    if affine_weights:
        K = logits.shape[dim]
        return logits - logits.mean(dim=dim, keepdim=True) + (1.0 / K)
    if softmax_weights:
        return F.softmax(logits, dim=dim)
    return logits


class WeightPredictor(nn.Module):
    def __init__(self,
        action_dim, horizon, cond_dim, num_experts,
        time_embed_dim=32, hidden_dim=256,
        use_state_dependent_weighting=False,
        use_time_dependent_weighting=False,
        use_temperature=True,
    ):
        super().__init__()
        self.use_state_dependent_weighting = use_state_dependent_weighting
        self.use_time_dependent_weighting = use_time_dependent_weighting
        self.use_temperature = use_temperature
        
        if use_time_dependent_weighting:
            self.time_emb = SinusoidalPosEmb(time_embed_dim)
        else:
            self.time_emb = None

        self.traj_flatten_dim = action_dim * horizon
        
        input_dim = cond_dim
        if use_time_dependent_weighting:
            input_dim += time_embed_dim
        if use_state_dependent_weighting:
            input_dim += self.traj_flatten_dim
        
        self.input_dim = input_dim

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_experts)
        ])
        if use_temperature:
            self.temperature = nn.Parameter(torch.ones(1))
        else:
            self.register_parameter("temperature", None)
        
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Tolerate legacy affine ckpts that still carry a `temperature` param.
        key = prefix + "temperature"
        if not self.use_temperature and key in state_dict:
            del state_dict[key]
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                             missing_keys, unexpected_keys, error_msgs)

    def forward(self, trajectory, timestep, cond):
        # trajectory [B, T, D], timestep [B], cond [B, C]
        
        inputs = [cond]
        
        if self.use_time_dependent_weighting:
            t_emb = self.time_emb(timestep) # [B, time_embed_dim]
            inputs.append(t_emb)
        
        if self.use_state_dependent_weighting:
            traj_flat = trajectory.reshape(trajectory.shape[0], -1) # [B, T*D]
            inputs.append(traj_flat)
            
        x = torch.cat(inputs, dim=-1)
        
        feat = self.backbone(x)
        logits_list = [head(feat) for head in self.heads]
        logits = torch.cat(logits_list, dim=-1)

        if not self.use_temperature:
            return logits
        # Ensure temperature is positive and non-zero
        temp = F.softplus(self.temperature) + 1e-2
        return logits / temp


class FactorizedDiffusionTransformerPolicy(BasePolicy):
    def __init__(
        self,
        shape_meta: Dict,
        noise_scheduler: Union[DDIMScheduler, DDPMScheduler],
        obs_encoder: BaseObservationEncoder,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        # policy model params
        num_experts: int = 4,
        embed_dim: int = 768,
        n_layers: int = 8,
        n_heads: int = 12,
        dropout: float = 0.1,
        use_state_dependent_weighting: bool = False,
        use_time_dependent_weighting: bool = False,
        unitize_experts: bool = False,
        softmax_weights: bool = True,
        affine_weights: bool = False,
        entropy_coef: float = 0.0,
        expert_dropout: float = 0.0,
        ortho_coef: float = 0.0,
        ortho_mode: str = "per_step_cos2",
        # inference params
        num_inference_steps: Optional[int] = None,
    ):
        super().__init__()

        modalities = obs_encoder.modalities()
        obs_feature_dim = obs_encoder.output_feature_dim()
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_key_shapes = dict()
        obs_ports = []
        for key, attr in shape_meta['obs'].items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            obs_type = attr['type']
            if obs_type in modalities:
                obs_ports.append(key)

        # create diffusion transformer
        models = nn.ModuleList([
            TransformerForDiffusion(
                input_dim=action_dim,
                output_dim=action_dim,
                horizon=horizon,
                n_obs_steps=n_obs_steps,
                cond_dim=obs_feature_dim,
                n_layer=n_layers,
                n_head=n_heads,
                n_emb=embed_dim,
                p_drop_emb=dropout,
                p_drop_attn=dropout,
                causal_attn=True,
                time_as_cond=True,
                obs_as_cond=True,
            ) for _ in range(num_experts)
        ])

        # create weight prediction network, output magnitude
        global_cond_dim = obs_feature_dim * n_obs_steps
        weight_predictor = WeightPredictor(
            action_dim=action_dim,
            horizon=horizon,
            cond_dim=global_cond_dim,
            num_experts=len(models),
            use_state_dependent_weighting=use_state_dependent_weighting,
            use_time_dependent_weighting=use_time_dependent_weighting,
            use_temperature=not affine_weights,
        )

        self.modalities = modalities
        self.obs_key_shapes = obs_key_shapes
        self.obs_ports = obs_ports
        self.obs_encoder = obs_encoder
        self.models = models
        self.inference_time_mask = [True] * len(models)
        self.weight_predictor = weight_predictor
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.use_state_dependent_weighting = use_state_dependent_weighting
        self.use_time_dependent_weighting = use_time_dependent_weighting
        self.softmax_weights = softmax_weights
        self.affine_weights = affine_weights
        if softmax_weights and affine_weights:
            raise ValueError("softmax_weights and affine_weights are mutually exclusive")
        self.entropy_coef = entropy_coef
        self.expert_dropout = expert_dropout
        self.unitize_experts = unitize_experts
        self.ortho_coef = float(ortho_coef)
        self.ortho_mode = str(ortho_mode)

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # report
        num_obs_params = sum(p.numel() for p in obs_encoder.parameters())
        num_trainable_obs_params = sum(p.numel() for p in obs_encoder.parameters() if p.requires_grad)
        obs_trainable_ratio = num_trainable_obs_params / num_obs_params
        num_model_params = sum(p.numel() for p in models.parameters())
        num_trainable_model_params = sum(p.numel() for p in models.parameters() if p.requires_grad)
        model_trainable_ratio = num_trainable_model_params / num_model_params
        print(
            f"{self.get_policy_name()} initialized with\n"
            f"  obs enc: {num_obs_params/1e6:.1f}M ({obs_trainable_ratio:.5%} trainable)\n"
            f"  policy : {num_model_params/1e6:.1f}M ({model_trainable_ratio:.5%} trainable)\n"
        )

    def get_observation_encoder(self):
        return self.obs_encoder

    def get_observation_modalities(self):
        return self.modalities
    
    def get_observation_ports(self):
        return self.obs_ports
    
    def get_policy_name(self):
        base_name = f'fdp_{len(self.models)}trans_'
        for modality in self.modalities:
            if modality != 'state':
                base_name += modality + '|'
        return base_name[:-1]

    def create_dummy_observation(self,
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        return super().create_dummy_observation(
            batch_size=batch_size,
            horizon=self.n_obs_steps,
            obs_key_shapes=self.obs_key_shapes,
            device=device
        )
    
    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.obs_encoder.set_normalizer(normalizer)
        
    def get_optimizer(
        self, 
        policy_lr: float,
        obs_enc_lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        optim_groups = []
        for model in self.models:
            optim_groups.extend(
                model.get_optim_groups(
                    lr=policy_lr,
                    weight_decay=weight_decay
                )
            )
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "lr": obs_enc_lr,
            "weight_decay": weight_decay
        })
        optim_groups.append({
            "params": self.weight_predictor.parameters(),
            "lr": policy_lr,
            "weight_decay": weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, betas=betas
        )
        return optimizer
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert any(self.inference_time_mask), "At least one expert must be enabled in inference_time_mask."

        # encode observation
        features = self.obs_encoder(obs_dict)   # [B, To, d]

        # per-inference weight prediction
        if not (self.use_state_dependent_weighting or self.use_time_dependent_weighting):
            logits = self.weight_predictor(
                None, None,
                features.reshape(features.shape[0], -1)
            ).transpose(0, 1)

            # apply inference time mask
            for i, mask in enumerate(self.inference_time_mask):
                if not mask:
                    if self.softmax_weights:
                        logits[i] = -float('inf')
                    else:
                        logits[i] = 0.0

            # uniform_router: zero logits give w = 1/K
            if getattr(self, "uniform_router", False):
                logits = torch.zeros_like(logits)

            weights = apply_weight_mode(logits, self.softmax_weights, self.affine_weights, dim=0)

        # diffusion sampling
        scheduler = self.noise_scheduler
        models = self.models
        trajectory = torch.randn(
            size=(len(features), self.horizon, self.action_dim),
            dtype=features.dtype,
            device=features.device,
        )
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:

            # per-timestep weight prediction
            if self.use_state_dependent_weighting or self.use_time_dependent_weighting:
                timesteps = torch.full((len(features),), t, device=features.device, dtype=torch.long)
                logits = self.weight_predictor(
                    trajectory, timesteps, 
                    features.reshape(features.shape[0], -1)
                ).transpose(0, 1)

                # apply inference time mask
                for i, mask in enumerate(self.inference_time_mask):
                    if not mask:
                        if self.softmax_weights:
                            logits[i] = -float('inf')
                        else:
                            logits[i] = 0.0
                
                weights = apply_weight_mode(logits, self.softmax_weights, self.affine_weights, dim=0)

            scores = []
            for w, model, mask in zip(weights, models, self.inference_time_mask):
                if mask:
                    out = model(trajectory, t, features)
                    if self.unitize_experts:
                        out = F.normalize(out.view(out.shape[0], -1), dim=1).view(out.shape)
                    scores.append(w[:,None,None] * out)

            trajectory = scheduler.step(
                sum(scores),
                t, trajectory
            ).prev_sample
        
        # unnormalize prediction
        action_pred = self.normalizer['action'].unnormalize(trajectory[...,:self.action_dim])

        # receding horizon
        action = action_pred[:,:self.n_action_steps]

        # router dump (eval_libero.py --dump_router): one (B, K) record per call
        log = getattr(self, "_router_log", None)
        if log is not None:
            log.append(weights.detach().float().transpose(0, 1).cpu().numpy())

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    def forward(self, batch):
        # normalize action
        trajectory = self.normalizer['action'].normalize(batch['action'])
        batch_size = trajectory.shape[0]

        # encode observation
        features = self.obs_encoder(batch['obs'])   # [B, To, d]
        assert features.shape[:2] == (batch_size, self.n_obs_steps)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )

        # predict noise residual
        logits = self.weight_predictor(
            noisy_trajectory, timesteps, 
            features.reshape(features.shape[0], -1).detach()
        ).transpose(0, 1)

        # router_var uses the logits before expert dropout (cloned: dropout writes in place)
        logits_clean = logits.clone() if self.ortho_mode == "router_var" else logits

        # expert dropout
        if self.training and self.expert_dropout > 0:
            mask = torch.rand_like(logits) < self.expert_dropout
            # Ensure at least one expert is active per sample
            all_masked = mask.all(dim=0) # [B]
            if all_masked.any():
                bad_indices = torch.nonzero(all_masked).squeeze(1)
                random_experts = torch.randint(0, logits.shape[0], (bad_indices.size(0),), device=logits.device)
                mask[random_experts, bad_indices] = False
            if self.softmax_weights:
                logits[mask] = -float('inf')
            else:
                logits[mask] = 0.0
            
        weights = apply_weight_mode(logits, self.softmax_weights, self.affine_weights, dim=0)

        if self.ortho_coef > 0 and self.ortho_mode == "act_decov":
            self._register_feat_hooks()

        scores = []
        factor_preds = []
        for w, model in zip(weights, self.models):
            out = model(noisy_trajectory, timesteps, features)
            factor_preds.append(out)  # raw per-factor eps, pre-unitize, pre-weight
            if self.unitize_experts:
                out = F.normalize(out.view(out.shape[0], -1), dim=1).view(out.shape)
            scores.append(w[:,None,None] * out)

        pred = sum(scores)
        loss = F.mse_loss(pred, noise)

        # decorrelation loss: LOCUS_MODES act on the router / features / weights
        # (_locus_loss), every other mode on the per-factor eps predictions (_ortho_loss)
        if self.ortho_coef > 0:
            if self.ortho_mode in self.LOCUS_MODES:
                loss = loss + self.ortho_coef * self._locus_loss(self.ortho_mode, logits_clean)
            else:
                loss = loss + self.ortho_coef * self._ortho_loss(factor_preds, self.ortho_mode)

        # entropy regularization
        if self.entropy_coef > 0:
            probs = weights.transpose(0, 1) # [B, num_experts]
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
            loss = loss - self.entropy_coef * entropy

        return loss

    @staticmethod
    def _ortho_loss(factor_preds: List[torch.Tensor], mode: str = "per_step_cos2") -> torch.Tensor:
        """Decorrelation penalty between the K per-factor eps predictions, each (B, H, D).

        The paper's loss is "flat_cos2_envelope": flatten each prediction over (H, D), take its
        elementwise absolute value (the envelope, so opposite signs cannot cancel), and average
        cos^2 over factor pairs. The other modes are the Appendix B variants and two score-level
        baselines:
            per_step_cos2        cos^2 of the D-dim vectors at each horizon step
            flat_abs             |cos| of the flattened predictions
            flat_abs_envelope    |cos| of the flattened envelopes
            flat_sum_cos2        cos^2 of the flattened predictions, summed over pairs
            flat_sum_relu_cos2   flat_sum_cos2 penalizing only positive cosines
            ncl                  negative correlation learning (Liu & Yao, 1999)
            decov                batch cross-correlation between factors (Cogswell et al., 2016)
        """
        eps = torch.stack(factor_preds, dim=1)              # (B, K, H, D)
        K = eps.shape[1]
        mask = torch.triu(torch.ones(K, K, device=eps.device, dtype=torch.bool), diagonal=1)
        if mode == "flat_abs":
            flat = eps.flatten(start_dim=2)                 # (B, K, H*D)
            norm = F.normalize(flat, dim=-1)
            cos = torch.matmul(norm, norm.transpose(-1, -2))    # (B, K, K)
            return cos[..., mask].abs().mean()
        elif mode == "flat_abs_envelope":
            env = eps.flatten(start_dim=2).abs()            # (B, K, H*D), >=0
            norm = F.normalize(env, dim=-1)
            cos = torch.matmul(norm, norm.transpose(-1, -2))    # (B, K, K), in [0,1]
            return cos[..., mask].mean()
        elif mode == "flat_sum_cos2":
            flat = eps.flatten(start_dim=2)                 # (B, K, H*D)
            norm = F.normalize(flat, dim=-1)
            cos = torch.matmul(norm, norm.transpose(-1, -2))    # (B, K, K)
            return (cos[..., mask] ** 2).sum(dim=-1).mean()     # sum over pairs, mean over B
        elif mode == "flat_sum_relu_cos2":
            flat = eps.flatten(start_dim=2)                 # (B, K, H*D)
            norm = F.normalize(flat, dim=-1)
            cos = torch.matmul(norm, norm.transpose(-1, -2))    # (B, K, K)
            return (cos.clamp(min=0)[..., mask] ** 2).sum(dim=-1).mean()
        elif mode == "flat_cos2_envelope":
            env = eps.flatten(start_dim=2).abs()            # (B, K, H*D), >=0
            norm = F.normalize(env, dim=-1)
            cos = torch.matmul(norm, norm.transpose(-1, -2))    # (B, K, K), in [0,1]
            return (cos[..., mask] ** 2).mean()
        elif mode == "ncl":
            # Negative Correlation Learning (Liu & Yao, 1999) around the uniform mean of
            # the factors, normalised by their total energy so it stays bounded.
            f = eps.flatten(2)                                   # (B,K,HD)
            d = f - f.mean(dim=1, keepdim=True)
            num = d.pow(2).sum(dim=(1, 2))                       # (B,)
            den = f.pow(2).sum(dim=(1, 2)) + 1e-8
            return -(num / den).mean()
        elif mode == "decov":
            # DeCov at the score: standardise each factor's output across the batch and
            # penalise the squared cross-correlation between factor pairs.
            f = eps.flatten(2)                                   # (B,K,HD)
            z = (f - f.mean(dim=0, keepdim=True)) / (f.std(dim=0, keepdim=True) + 1e-6)
            K_ = z.shape[1]
            tot, n = 0.0, 0
            for i in range(K_):
                for j in range(i + 1, K_):
                    rho = (z[:, i] * z[:, j]).mean(dim=0)        # (HD,)
                    tot = tot + rho.pow(2).mean()
                    n += 1
            return tot / max(n, 1)
        elif mode == "per_step_cos2":
            norm = F.normalize(eps, dim=-1).permute(0, 2, 1, 3) # (B, H, K, D)
            cos = torch.matmul(norm, norm.transpose(-1, -2))    # (B, H, K, K)
            return (cos[..., mask] ** 2).mean()
        else:
            raise ValueError(f"unknown ortho_mode: {mode!r}")

    # ---- Other-locus baselines (Table 7): router, features, output weights ----
    LOCUS_MODES = ("router_var", "act_decov", "param_srip", "param_so")
    # router_var hinge target: Var_n[w_i] = 0.01, i.e. std 0.1 (uniform weight at K=8 is 0.125)
    ROUTER_VAR_TARGET = 0.01

    def _register_feat_hooks(self):
        """Record the input to each factor's output head (its last hidden feature) for act_decov."""
        if getattr(self, "_feat_hooks", None) is not None:
            return
        self._feat_buf = {}
        self._feat_hooks = []
        for i, m in enumerate(self.models):
            def _mk(idx):
                def _hook(_mod, inp, _out):
                    self._feat_buf[idx] = inp[0]
                return _hook
            self._feat_hooks.append(m.head.register_forward_hook(_mk(i)))

    def _locus_loss(self, mode: str, logits: torch.Tensor) -> torch.Tensor:
        """Baselines that decorrelate somewhere other than the score.

        router_var acts on the router weights, act_decov on the factors' last hidden features,
        param_srip / param_so on their output weights. `logits` are the (K, B) router logits
        before expert dropout.
        """
        if mode == "router_var":
            # routing variance (Guo et al., 2025) on the affine weights, as a bounded hinge
            w = apply_weight_mode(logits, self.softmax_weights,
                                  self.affine_weights, dim=0)     # (K, B)
            v = w.var(dim=1, unbiased=False).mean()
            v0 = self.ROUTER_VAR_TARGET
            return torch.clamp(v0 - v, min=0.0) / v0

        if mode == "act_decov":
            # DeCov (Cogswell et al., 2016) on each factor's last hidden feature: squared
            # cross-correlation between factor pairs, standardised across the batch.
            n_f = len(self.models)
            missing = [i for i in range(n_f) if i not in getattr(self, "_feat_buf", {})]
            if missing:
                raise RuntimeError(
                    f"act_decov: no captured features for factors {missing}; "
                    "_register_feat_hooks() must run before the factor forward pass")
            # (B, T, d) -> (B, d), mean-pooled over the horizon
            H = torch.stack([self._feat_buf[i].mean(dim=1) for i in range(n_f)], dim=1)
            B = H.shape[0]
            Z = (H - H.mean(dim=0, keepdim=True)) / (H.std(dim=0, keepdim=True) + 1e-6)
            tot, n = 0.0, 0
            for i in range(n_f):
                for j in range(i + 1, n_f):
                    rho = (Z[:, i].transpose(0, 1) @ Z[:, j]) / B   # (d, d)
                    tot = tot + rho.pow(2).mean()                   # the 1/d^2
                    n += 1
            return tot / max(n, 1)

        if mode in ("param_srip", "param_so"):
            # SRIP / SO (Bansal et al., 2018) on the factors' normalized output-projection weights
            U = torch.stack([F.normalize(m.head.weight.reshape(-1), dim=0)
                             for m in self.models])                # (K, P)
            E = U @ U.t() - torch.eye(U.shape[0], device=U.device, dtype=U.dtype)
            if mode == "param_so":
                return E.pow(2).sum()
            # SRIP = spectral norm of (UU^T - I). E is K x K and symmetric, so
            # that is max|eigenvalue| -- exact via eigvalsh, no power iteration.
            return torch.linalg.eigvalsh(E).abs().max()

        raise ValueError(f"unknown locus mode: {mode!r}")

    @torch.no_grad()
    def locus_diagnostics(self, batch, timestep=None):
        """Metrics for the other-locus baselines on one batch at a fixed timestep.

        Router weight variance and entropy, cross-factor feature correlation, and the output-weight
        Gram matrix, so each baseline can be checked to move its own target.
        """
        trajectory = self.normalizer['action'].normalize(batch['action'])
        B = trajectory.shape[0]
        features = self.obs_encoder(batch['obs'])
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        t_val = (self.noise_scheduler.config.num_train_timesteps // 2
                 if timestep is None else int(timestep))
        ts = torch.full((B,), t_val, device=trajectory.device, dtype=torch.long)
        noisy = self.noise_scheduler.add_noise(trajectory, noise, ts)

        logits = self.weight_predictor(
            noisy, ts, features.reshape(features.shape[0], -1).detach()).transpose(0, 1)
        p = F.softmax(logits, dim=0)                                  # (K, B)
        ent = -(p * (p + 1e-8).log()).sum(dim=0).mean()
        # the composition weights the policy actually uses
        w = apply_weight_mode(logits, self.softmax_weights, self.affine_weights, dim=0)

        self._register_feat_hooks()
        for model in self.models:
            model(noisy, ts, features)
        n_f = len(self.models)
        H = torch.stack([self._feat_buf[i].mean(dim=1) for i in range(n_f)], dim=1)
        Z = (H - H.mean(dim=0, keepdim=True)) / (H.std(dim=0, keepdim=True) + 1e-6)
        dg = sq = 0.0
        n = 0
        for i in range(n_f):
            for j in range(i + 1, n_f):
                rho = (Z[:, i].transpose(0, 1) @ Z[:, j]) / B
                dg += rho.diagonal().abs().mean().item()
                sq += rho.pow(2).mean().item()
                n += 1

        U = torch.stack([F.normalize(m.head.weight.reshape(-1), dim=0) for m in self.models])
        G = U @ U.t()
        eye = torch.eye(n_f, device=G.device, dtype=G.dtype)
        off = ~torch.eye(n_f, dtype=torch.bool, device=G.device)
        return {
            "router_score_var": p.var(dim=1, unbiased=False).mean().item(),
            "router_entropy":   ent.item(),
            "router_w_var":     w.var(dim=1, unbiased=False).mean().item(),
            "router_w_std":     w.std(dim=1).mean().item(),
            "feat_corr":        dg / max(n, 1),
            "feat_decov":       sq / max(n, 1),
            "param_gram_off":   G[off].abs().mean().item(),
            "param_srip":       torch.linalg.eigvalsh(G - eye).abs().max().item(),
        }

    def diagnostic_components(self, batch, timesteps):
        """Per-factor eps predictions and router weights at the given timesteps, for diagnostics.py.

        Same computation as forward(), without expert dropout or the loss. Returns eps (B, K, H, D)
        and weights (B, K).
        """
        trajectory = self.normalizer['action'].normalize(batch['action'])
        features = self.obs_encoder(batch['obs'])

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        logits = self.weight_predictor(
            noisy_trajectory, timesteps,
            features.reshape(features.shape[0], -1).detach()
        ).transpose(0, 1)
        weights = apply_weight_mode(logits, self.softmax_weights, self.affine_weights, dim=0)

        factor_preds = [model(noisy_trajectory, timesteps, features) for model in self.models]
        eps = torch.stack(factor_preds, dim=1)       # (B, K, H, D)
        return eps, weights.transpose(0, 1)          # (B, K, H, D), (B, K)


class FactorizedDiffusionUnetPolicy(BasePolicy):
    def __init__(self, 
        shape_meta: dict,
        noise_scheduler: Union[DDIMScheduler, DDPMScheduler],
        obs_encoder: BaseObservationEncoder,
        horizon, 
        n_action_steps, 
        n_obs_steps,
        # policy model params
        num_experts: int,
        diffusion_step_embed_dim: int = 128,
        down_dims: Tuple[int] = (128,256,512),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        use_state_dependent_weighting: bool = False,
        use_time_dependent_weighting: bool = False,
        unitize_experts: bool = False,
        softmax_weights: bool = False,
        affine_weights: bool = False,
        entropy_coef: float = 0.0,
        expert_dropout: float = 0.0,
        ortho_coef: float = 0.0,
        ortho_mode: str = "per_step_cos2",
        # inference params
        num_inference_steps: Optional[int] = None,
    ):
        super().__init__()

        modalities = obs_encoder.modalities()
        obs_feature_dim = obs_encoder.output_feature_dim()
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_key_shapes = dict()
        obs_ports = []
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            type = attr['type']
            if type in modalities:
                obs_ports.append(key)

        # create diffusion unet
        global_cond_dim = obs_feature_dim * n_obs_steps
        models = nn.ModuleList([
            ConditionalUnet1D(
                input_dim=action_dim,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ) for _ in range(num_experts)
        ])

        # create weight prediction network, output magnitude
        weight_predictor = WeightPredictor(
            action_dim=action_dim,
            horizon=horizon,
            cond_dim=global_cond_dim,
            num_experts=len(models),
            use_state_dependent_weighting=use_state_dependent_weighting,
            use_time_dependent_weighting=use_time_dependent_weighting,
            use_temperature=not affine_weights,
        )

        self.modalities = modalities
        self.obs_key_shapes = obs_key_shapes
        self.obs_ports = obs_ports
        self.obs_encoder = obs_encoder
        self.models = models
        self.inference_time_mask = [True] * len(models)
        self.weight_predictor = weight_predictor
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.use_state_dependent_weighting = use_state_dependent_weighting
        self.use_time_dependent_weighting = use_time_dependent_weighting
        self.softmax_weights = softmax_weights
        self.affine_weights = affine_weights
        if softmax_weights and affine_weights:
            raise ValueError("softmax_weights and affine_weights are mutually exclusive")
        self.entropy_coef = entropy_coef
        self.expert_dropout = expert_dropout
        self.unitize_experts = unitize_experts
        self.ortho_coef = float(ortho_coef)
        self.ortho_mode = str(ortho_mode)

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # report
        num_obs_params = sum(p.numel() for p in obs_encoder.parameters())
        num_trainable_obs_params = sum(p.numel() for p in obs_encoder.parameters() if p.requires_grad)
        obs_trainable_ratio = num_trainable_obs_params / num_obs_params
        num_model_params = sum(p.numel() for p in models.parameters())
        num_trainable_model_params = sum(p.numel() for p in models.parameters() if p.requires_grad)
        model_trainable_ratio = num_trainable_model_params / num_model_params
        print(
            f"{self.get_policy_name()} initialized with\n"
            f"  obs enc: {num_obs_params/1e6:.1f}M ({obs_trainable_ratio:.5%} trainable)\n"
            f"  policy : {num_model_params/1e6:.1f}M ({model_trainable_ratio:.5%} trainable)\n"
        )

    def get_observation_encoder(self) -> BaseObservationEncoder:
        return self.obs_encoder
    
    def get_observation_modalities(self) -> List[str]:
        return self.modalities
    
    def get_observation_ports(self) -> List[str]:
        return self.obs_ports

    def get_policy_name(self) -> str:
        base_name = f'fdp_{len(self.models)}unet_'
        for modality in self.modalities:
            if modality != 'state':
                base_name += modality + '|'
        return base_name[:-1]

    def create_dummy_observation(self, 
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        return super().create_dummy_observation(
            batch_size=batch_size,
            horizon=self.n_obs_steps,
            obs_key_shapes=self.obs_key_shapes,
            device=device
        )
    
    def get_optimizer(
        self, 
        policy_lr: float,
        obs_enc_lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        optim_groups = [
            {
                "params": self.models.parameters(),
                "lr": policy_lr,
                "weight_decay": weight_decay
            },
            {
                "params": self.weight_predictor.parameters(),
                "lr": policy_lr,
                "weight_decay": weight_decay
            },
            {
                "params": self.obs_encoder.parameters(),
                "lr": obs_enc_lr,
                "weight_decay": weight_decay
            }
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, betas=betas
        )
        return optimizer
    
    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.obs_encoder.set_normalizer(normalizer)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert any(self.inference_time_mask), "At least one expert must be enabled in inference_time_mask."

        # encode observation
        features = self.obs_encoder(obs_dict)   # (B, To, d)
        features = features.reshape(features.shape[0], -1)  # (B, To*d)

        # per-inference weight prediction
        if not (self.use_state_dependent_weighting or self.use_time_dependent_weighting):
            logits = self.weight_predictor(
                None, None, 
                features.reshape(features.shape[0], -1)
            ).transpose(0, 1)

            # apply inference time mask
            for i, mask in enumerate(self.inference_time_mask):
                if not mask:
                    if self.softmax_weights:
                        logits[i] = -float('inf')
                    else:
                        logits[i] = 0.0
            
            weights = apply_weight_mode(logits, self.softmax_weights, self.affine_weights, dim=0)

        # diffusion sampling
        scheduler = self.noise_scheduler
        models = self.models
        trajectory = torch.randn(
            size=(len(features), self.horizon, self.action_dim),
            dtype=features.dtype,
            device=features.device,
        )
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:

            # per-timestep weight prediction
            if self.use_state_dependent_weighting or self.use_time_dependent_weighting:
                timesteps = torch.full((len(features),), t, device=features.device, dtype=torch.long)
                logits = self.weight_predictor(
                    trajectory, timesteps, 
                    features.reshape(features.shape[0], -1)
                ).transpose(0, 1)

                # apply inference time mask
                for i, mask in enumerate(self.inference_time_mask):
                    if not mask:
                        if self.softmax_weights:
                            logits[i] = -float('inf')
                        else:
                            logits[i] = 0.0
                
                weights = apply_weight_mode(logits, self.softmax_weights, self.affine_weights, dim=0)

            scores = []
            for w, model, mask in zip(weights, models, self.inference_time_mask):
                if mask:
                    out = model(trajectory, t, global_cond=features)
                    if self.unitize_experts:
                        out = F.normalize(out.view(out.shape[0], -1), dim=1).view(out.shape)
                    scores.append(w[:,None,None] * out)

            trajectory = scheduler.step(
                sum(scores),
                t, trajectory
            ).prev_sample
        
        # unnormalize prediction
        action_pred = self.normalizer['action'].unnormalize(trajectory[...,:self.action_dim])

        # receding horizon
        action = action_pred[:,:self.n_action_steps]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    def forward(self, batch):
        # normalize action
        trajectory = self.normalizer['action'].normalize(batch['action'])
        batch_size = trajectory.shape[0]

        # encode observation
        features = self.obs_encoder(batch['obs'])  # [B, To, d]
        assert features.shape[:2] == (batch_size, self.n_obs_steps)
        features = features.reshape(batch_size, -1)  # [B, To*d]

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )
        
        # predict noise residual
        logits = self.weight_predictor(noisy_trajectory, timesteps, features.detach()).transpose(0, 1)
        
        # expert dropout
        if self.training and self.expert_dropout > 0:
            mask = torch.rand_like(logits) < self.expert_dropout
            # Ensure at least one expert is active per sample
            all_masked = mask.all(dim=0) # [B]
            if all_masked.any():
                bad_indices = torch.nonzero(all_masked).squeeze(1)
                random_experts = torch.randint(0, logits.shape[0], (bad_indices.size(0),), device=logits.device)
                mask[random_experts, bad_indices] = False
            if self.softmax_weights:
                logits[mask] = -float('inf')
            else:
                logits[mask] = 0.0
            
        weights = apply_weight_mode(logits, self.softmax_weights, self.affine_weights, dim=0)

        scores = []
        factor_preds = []
        for w, model in zip(weights, self.models):
            out = model(noisy_trajectory, timesteps, global_cond=features)
            factor_preds.append(out)  # raw per-factor eps, pre-unitize, pre-weight
            if self.unitize_experts:
                out = F.normalize(out.view(out.shape[0], -1), dim=1).view(out.shape)
            scores.append(w[:,None,None] * out)

        pred = sum(scores)
        loss = F.mse_loss(pred, noise)

        if self.ortho_coef > 0:
            loss = loss + self.ortho_coef * FactorizedDiffusionTransformerPolicy._ortho_loss(
                factor_preds, self.ortho_mode)

        # entropy regularization
        if self.entropy_coef > 0:
            probs = weights.transpose(0, 1) # [B, num_experts]
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
            loss = loss - self.entropy_coef * entropy

        return loss
