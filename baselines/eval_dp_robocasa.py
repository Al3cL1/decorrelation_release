"""Evaluate a Diffusion Policy checkpoint on RoboCasa with FDP's RobocasaRunner.

Applies the same RobocasaEnv patches as eval_robocasa.py (HWC uint8 images, raw-to-env action
order). Render headless with MUJOCO_GL=egl PYOPENGL_PLATFORM=egl.

Usage:
    python baselines/eval_dp_robocasa.py -c <dp_ckpt> -o <out_dir> --test_start_seed 7
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

# ---- RobocasaEnv monkey-patches (mirrors eval_robocasa.py) --------------------
import gymnasium as _gym
from fdp.env.robocasa.env import RobocasaEnv as _RobocasaEnv

_orig_init = _RobocasaEnv.__init__
def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    for cam in self.camera_names:
        key = f"{cam}_rgb"
        sp = self.observation_space.spaces.get(key)
        if sp is not None and len(sp.shape) == 3 and sp.shape[0] == 3:
            self.observation_space.spaces[key] = _gym.spaces.Box(
                low=0, high=255,
                shape=(sp.shape[1], sp.shape[2], sp.shape[0]),
                dtype=np.uint8,
            )
_RobocasaEnv.__init__ = _patched_init

_orig_extract_obs = _RobocasaEnv._extract_obs
def _patched_extract_obs(self, raw_obs=None):
    obs = _orig_extract_obs(self, raw_obs)
    for cam in self.camera_names:
        key = f"{cam}_rgb"
        if key in obs and obs[key].ndim == 3 and obs[key].shape[0] == 3:
            arr = obs[key].transpose(1, 2, 0)
            obs[key] = np.ascontiguousarray((arr * 255.0).clip(0, 255).astype(np.uint8))
    return obs
_RobocasaEnv._extract_obs = _patched_extract_obs

_orig_step = _RobocasaEnv.step
def _patched_step(self, action):
    if isinstance(action, np.ndarray) and action.ndim >= 1 and action.shape[-1] == 12:
        action = np.concatenate([action[..., 5:12], action[..., 0:5]], axis=-1)
    return _orig_step(self, action)
_RobocasaEnv.step = _patched_step

print("[eval_dp_robocasa] patched RobocasaEnv: HWC uint8 + raw->env action permute", flush=True)
# --------------------------------------------------------------------------


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
              help="path to FDP task yaml whose env_runner to use. Defaults to "
                   "fdp/config/task/robocasa/atomics_H16_12D_raw_tid.yaml.")
def main(checkpoint, output_dir, n_test, test_start_seed, n_parallel_envs, device,
         fdp_task_yaml):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(device)

    print(f"\n=== eval_dp_robocasa: {checkpoint}\n    out: {output_dir}", flush=True)
    workspace, payload, cfg = _load_dp_workspace(checkpoint, output_dir)

    # Prefer EMA if present (DP workspace stores ema_model attr).
    policy = workspace.model
    try:
        ema = getattr(workspace, "ema_model", None)
        if ema is not None:
            policy = ema
            print("    using EMA weights")
    except Exception:
        pass
    policy.to(device).eval()

    # DP cfg.shape_meta.obs.keys() = the policy's expected obs ports.
    obs_ports = list(cfg.shape_meta.obs.keys())
    print(f"    DP obs_ports: {obs_ports}")

    adapter = DPPolicyAdapter(policy, obs_ports)

    # Load FDP env_runner config and instantiate it.
    if fdp_task_yaml is None:
        fdp_task_yaml = str(_HERE / "fdp/config/task/robocasa/atomics_H16_12D_raw_tid.yaml")
    task_cfg = OmegaConf.load(fdp_task_yaml)
    runner_cfg = task_cfg.env_runner
    if n_test is not None:           runner_cfg.n_test = n_test
    if n_parallel_envs is not None:  runner_cfg.n_parallel_envs = n_parallel_envs
    # Resolve the n_obs_steps / n_action_steps references; the FDP task yaml
    # references ${n_obs_steps} from train_factorpolicy.yaml. DP's analogues:
    runner_cfg.n_obs_steps    = int(cfg.n_obs_steps)
    runner_cfg.n_action_steps = int(cfg.n_action_steps)
    if test_start_seed is not None:
        runner_kwargs = {"output_dir": output_dir, "test_start_seed": test_start_seed}
    else:
        runner_kwargs = {"output_dir": output_dir}
    print(f"    instantiating RobocasaRunner: task_group={runner_cfg.task_group}, "
          f"n_test={runner_cfg.n_test}, n_parallel_envs={runner_cfg.n_parallel_envs}")
    runner = hydra.utils.instantiate(runner_cfg, **runner_kwargs)

    print(f"    running rollouts...", flush=True)
    runner_log = runner.run(adapter)
    runner.close()

    # Console summary
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
