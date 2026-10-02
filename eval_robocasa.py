"""Evaluate FDP checkpoints on RoboCasa.

Same as eval.py, with RobocasaEnv patched to match the training data: images as HWC uint8
(the env emits CHW float) and actions mapped from the zarr's raw layout to the env's.
"""
import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import json
import re
from typing import List

import click
import dill
import hydra
import numpy as np
import omegaconf
import torch
import wandb

from fdp.workspace.base_workspace import BaseWorkspace
from fdp.workspace.train_policy import TrainPolicyWorkspace
from fdp.policy.base_policy import BasePolicy
from fdp.env_runner.base_runner import BaseRunner

# --- RobocasaEnv patches to match the training data -------------------------
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
            # CHW float [0,1] -> HWC uint8 [0,255] (matches zarr training data)
            arr = obs[key].transpose(1, 2, 0)
            obs[key] = np.ascontiguousarray((arr * 255.0).clip(0, 255).astype(np.uint8))
    return obs
_RobocasaEnv._extract_obs = _patched_extract_obs

# step(): raw action layout -> env layout (env[0:7] = raw[5:12], env[7:12] = raw[0:5])
_orig_step = _RobocasaEnv.step
def _patched_step(self, action):
    if isinstance(action, np.ndarray) and action.ndim >= 1 and action.shape[-1] == 12:
        action = np.concatenate([action[..., 5:12], action[..., 0:5]], axis=-1)
    return _orig_step(self, action)
_RobocasaEnv.step = _patched_step

print("[eval_robocasa] patched RobocasaEnv: rgb -> HWC uint8 [0,255] + action "
      "raw->env permutation (env[0:7]=raw[5:12], env[7:12]=raw[0:5])",
      flush=True)
# ---------------------------------------------------------------------------


def _make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, wandb.sdk.data_types.video.Video):
        return getattr(obj, "_path", str(obj))
    return obj


def _fix_crop_randomizers(policy: torch.nn.Module) -> int:
    target_types = []
    try:
        import robomimic.models.base_nets as rmbn
        target_types.append(rmbn.CropRandomizer)
    except Exception:
        pass
    try:
        from fdp.perception.crop_randomizer import CropRandomizer as FdpCrop
        target_types.append(FdpCrop)
    except Exception:
        pass

    if not target_types:
        return 0
    target_types = tuple(target_types)

    fixed = 0
    for _, m in policy.named_modules():
        if isinstance(m, target_types):
            m.eval()
            fixed += 1
    return fixed


def _resolve_ckpts(ckpt_arg: str) -> List[str]:
    if os.path.isdir(ckpt_arg):
        return [
            os.path.join(ckpt_arg, f)
            for f in sorted(os.listdir(ckpt_arg))
            if f.endswith(".ckpt") and f != "latest.ckpt"
        ]
    return [ckpt_arg]


def _output_dir_for_ckpt(base_dir: str, ckpt_path: str, single: bool) -> str:
    if single:
        return base_dir
    name = os.path.basename(ckpt_path)
    m = re.match(r"^ep-(\d{4})_sr-(\d\.\d{3})\.ckpt$", name)
    sub = f"ep-{m.group(1)}" if m else name.replace(".ckpt", "")
    out = os.path.join(base_dir, sub)
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    return out


def _load_workspace(ckpt_path: str, output_dir: str):
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=output_dir, lazy_instantiation=False)


    # skip optimizer state (also lets older checkpoints load)
    workspace.load_payload(payload, exclude_keys=["optimizer"], include_keys=None)

    # Checkpoints saved without an 'ema_model' state_dict: mirror the trained
    # weights so code that prefers EMA still evaluates the trained policy.
    if workspace.ema_model is not None \
            and "ema_model" not in payload.get("state_dicts", {}):
        print("    [mirror] ema_model has no saved state — copying from model")
        workspace.ema_model.load_state_dict(workspace.model.state_dict())

    return workspace, payload, cfg


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-n", "--n_test", default=None, type=int)
@click.option("--test_start_seed", default=None, type=int)
@click.option("--n_parallel_envs", default=None, type=int)
@click.option("-d", "--device", default="cuda:0")
@click.option("-u", "--update", is_flag=True)
@click.option("--task_yaml", default=None,
              help="path to a task yaml whose `env_runner` should override the "
                   "one baked into the ckpt's cfg. Use this to evaluate a ckpt "
                   "with the current task config rather than the one it was "
                   "trained with; without it, eval instantiates whatever "
                   "env_runner the saved cfg encodes.")
def eval_policy(checkpoint, output_dir, n_test, test_start_seed, n_parallel_envs, device, update,
                task_yaml):
    if os.path.exists(output_dir) and os.path.isfile(output_dir):
        raise click.UsageError(f"--output_dir {output_dir} is a file")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    ckpts = _resolve_ckpts(checkpoint)
    if not ckpts:
        raise click.UsageError(f"no .ckpt files found under {checkpoint}")

    device = torch.device(device)
    single = len(ckpts) == 1

    for ckpt in ckpts:
        out = _output_dir_for_ckpt(output_dir, ckpt, single=single)
        print(f"\n=== eval: {ckpt}\n    out:   {out}", flush=True)

        workspace, _payload, cfg = _load_workspace(ckpt, out)

        # --task_yaml replaces the checkpoint's task config
        if task_yaml is not None:
            import omegaconf as _oc
            new_task = _oc.OmegaConf.load(task_yaml)
            print(f"    [task override] cfg.task ← {task_yaml}\n"
                  f"      env_runner.task_name = {new_task.env_runner.get('task_name', '?')}, "
                  f"task_group = {new_task.env_runner.get('task_group', '?')}")
            cfg.task = new_task

        policy: BasePolicy = workspace.model
        try:
            if cfg.training.use_ema and workspace.ema_model is not None:
                policy = workspace.ema_model
                print("    using EMA weights")
        except omegaconf.errors.ConfigAttributeError:
            pass

        n_fixed = _fix_crop_randomizers(policy)
        print(f"    crop randomizers set to eval mode: {n_fixed}")

        policy.to(device)
        policy.eval()

        # n_test_vis=0 disables per-episode mp4 recording (RobocasaRunner only —
        # rlbench runner ignores it). Skips wandb_video uploads too.
        runner_kwargs = {"output_dir": out, "n_test_vis": 0}
        if n_test is not None:
            runner_kwargs["n_test"] = n_test
        if test_start_seed is not None:
            runner_kwargs["test_start_seed"] = test_start_seed
        if n_parallel_envs is not None:
            runner_kwargs["n_parallel_envs"] = n_parallel_envs
        env_runner: BaseRunner = hydra.utils.instantiate(
            cfg.task.env_runner, **runner_kwargs
        )

        runner_log = env_runner.run(policy)
        env_runner.close()

        if "mean_success_rate" in runner_log:
            print(f"    mean_success_rate = {runner_log['mean_success_rate']:.3f}")
        for k, v in sorted(runner_log.items()):
            if k.endswith("/mean_success_rate") and k != "mean_success_rate":
                print(f"    {k} = {float(v):.3f}")

        json_log = {
            "checkpoint": ckpt,
            "n_test_override": n_test,
            **runner_log,
        }
        with open(os.path.join(out, "eval_log.json"), "w") as f:
            json.dump(_make_json_safe(json_log), f, indent=2, sort_keys=True)
        print(f"    wrote {os.path.join(out, 'eval_log.json')}")

        if update and "mean_success_rate" in runner_log:
            sr = float(runner_log["mean_success_rate"])
            new_name = re.sub(
                r"(\d\.\d{3})\.ckpt$",
                f"{sr:.3f}.ckpt",
                os.path.basename(ckpt),
            )
            if new_name != os.path.basename(ckpt):
                new_path = os.path.join(os.path.dirname(ckpt), new_name)
                os.rename(ckpt, new_path)
                print(f"    renamed: {ckpt} -> {new_path}")


if __name__ == "__main__":
    eval_policy()
