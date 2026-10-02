"""Evaluate a Diffusion Policy checkpoint on LIBERO with FDP's LiberoRunner, so DP and FDP are
scored by the same rollout code.

Usage:
    python baselines/eval_dp_libero.py -c <dp_ckpt> -o <out_dir> --test_start_seed 7
"""
from __future__ import annotations

import os
import sys
import pathlib
import json

# Make sure both FDP and DP packages are importable.
_HERE = pathlib.Path(__file__).resolve().parents[1]      # repo root (the `fdp` package)
sys.path.insert(0, str(_HERE))
# `diffusion_policy` comes from the overlaid upstream clone on PYTHONPATH (baselines/README.md)

import click
import dill
import hydra
import torch
import numpy as np
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

DEFAULT_TASK_YAML = str(_HERE / "fdp/config/task/libero/atomics25.yaml")


class DPPolicyAdapter:
    """Expose a DP policy through the interface FDP's runners call (device, dtype,
    get_policy_name, get_observation_ports, reset, predict_action), feeding it CHW images.
    """
    def __init__(self, dp_policy, obs_ports):
        self._policy = dp_policy
        self._obs_ports = list(obs_ports)

    @property
    def device(self):
        return next(self._policy.parameters()).device

    @property
    def dtype(self):
        return next(self._policy.parameters()).dtype

    def get_policy_name(self) -> str:
        return "dp_transformer"

    def get_observation_ports(self):
        return self._obs_ports

    def reset(self):
        # DP policies are stateless across calls.
        pass

    def predict_action(self, obs_dict, **kwargs):
        # HWC images -> CHW float in [0, 1] (4D+ tensors with 3 channels, not state vectors)
        fixed = {}
        for k, v in obs_dict.items():
            if torch.is_tensor(v) and v.ndim >= 4 and v.shape[-1] == 3 \
                    and (v.dtype == torch.uint8 or v.is_floating_point()):
                # HWC → CHW permute (moves last dim to position v.ndim-3)
                v = v.permute(*range(v.ndim - 3), v.ndim - 1, v.ndim - 3, v.ndim - 2)
                # Dtype: uint8 → float [0,1]; float values >1.5 → assume [0,255]
                if v.dtype == torch.uint8:
                    v = v.float() / 255.0
                elif v.is_floating_point() and v.abs().max() > 1.5:
                    v = v / 255.0
            fixed[k] = v
        return self._policy.predict_action(fixed)

    def eval(self):
        self._policy.eval()
        return self


def _load_dp_workspace(ckpt_path: str, output_dir: str):
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=["optimizer"], include_keys=None)
    return workspace, payload, cfg


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-n", "--n_test", default=None, type=int)
@click.option("--test_start_seed", default=None, type=int)
@click.option("--n_parallel_envs", default=None, type=int)
@click.option("-d", "--device", default="cuda:0")
@click.option("--fdp_task_yaml", default=None,
              help=f"FDP task yaml whose env_runner to use. Default: {DEFAULT_TASK_YAML}")
@click.option("--ddim_steps", default=None, type=int,
              help="swap the ckpt's DDPM sampler for FDP's DDIM config at this "
                   "step count, so DP and FDP are compared under one sampler. "
                   "Weights are unaffected: both trained on identical betas.")
def main(checkpoint, output_dir, n_test, test_start_seed, n_parallel_envs, device,
         fdp_task_yaml, ddim_steps):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(device)

    print(f"\n=== eval_dp_libero: {checkpoint}\n    out: {output_dir}", flush=True)
    workspace, payload, cfg = _load_dp_workspace(checkpoint, output_dir)

    # prefer the EMA weights, as DP's own eval does
    policy = workspace.model
    ema = getattr(workspace, "ema_model", None)
    if ema is not None:
        policy = ema
        print("    using EMA weights")
    policy.to(device).eval()

    if ddim_steps is not None:
        # --ddim_steps: sample with FDP's DDIM scheduler (same training betas)
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        old = policy.noise_scheduler
        policy.noise_scheduler = DDIMScheduler(
            num_train_timesteps=old.config.num_train_timesteps,
            beta_start=old.config.beta_start,
            beta_end=old.config.beta_end,
            beta_schedule=old.config.beta_schedule,
            clip_sample=old.config.clip_sample,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type=old.config.prediction_type,
        )
        policy.num_inference_steps = ddim_steps
        print(f"    [sampler] {type(old).__name__} -> DDIMScheduler, "
              f"num_inference_steps={ddim_steps} (was {old.config.num_train_timesteps})")

    n_params = sum(p.numel() for p in policy.parameters())
    n_enc = sum(p.numel() for p in policy.obs_encoder.parameters())
    print(f"    params: total={n_params/1e6:.3f}M  obs_encoder={n_enc/1e6:.3f}M  "
          f"trunk={(n_params-n_enc)/1e6:.3f}M  (n_layer={cfg.policy.n_layer})")

    # DP cfg.shape_meta.obs.keys() = the policy's expected obs ports.
    obs_ports = list(cfg.shape_meta.obs.keys())
    print(f"    DP obs_ports: {obs_ports}")

    adapter = DPPolicyAdapter(policy, obs_ports)

    task_cfg = OmegaConf.load(fdp_task_yaml or DEFAULT_TASK_YAML)
    runner_cfg = task_cfg.env_runner
    if n_test is not None:           runner_cfg.n_test = n_test
    if n_parallel_envs is not None:  runner_cfg.n_parallel_envs = n_parallel_envs
    # The FDP task yaml references ${n_obs_steps} / ${n_action_steps} from
    # train_factorpolicy.yaml, which isn't loaded here. Resolve from DP's cfg.
    runner_cfg.n_obs_steps    = int(cfg.n_obs_steps)
    runner_cfg.n_action_steps = int(cfg.n_action_steps)

    # n_test_vis=0 disables per-episode mp4 recording, matching eval_libero.py.
    runner_kwargs = {"output_dir": output_dir, "n_test_vis": 0}
    if test_start_seed is not None:
        runner_kwargs["test_start_seed"] = test_start_seed
    print(f"    instantiating LiberoRunner: task_name={runner_cfg.task_name}, "
          f"n_test={runner_cfg.n_test}, n_parallel_envs={runner_cfg.n_parallel_envs}")
    runner = hydra.utils.instantiate(runner_cfg, **runner_kwargs)

    print(f"    running rollouts...", flush=True)
    runner_log = runner.run(adapter)
    runner.close()

    if "mean_success_rate" in runner_log:
        print(f"    mean_success_rate = {runner_log['mean_success_rate']:.3f}")
    for k, v in sorted(runner_log.items()):
        if k.endswith("/mean_success_rate") and k != "mean_success_rate":
            try:
                print(f"    {k} = {float(v):.3f}")
            except Exception:
                pass

    def _make_safe(d):
        out = {}
        for k, v in d.items():
            try:
                json.dumps(v)
                out[k] = v
            except Exception:
                out[k] = str(v)
        return out

    out_path = os.path.join(output_dir, "eval_log.json")
    with open(out_path, "w") as f:
        json.dump({"checkpoint": checkpoint, "n_test_override": n_test, **_make_safe(runner_log)},
                  f, indent=2, sort_keys=True)
    print(f"    wrote {out_path}")


if __name__ == "__main__":
    main()
