"""Evaluate FDP checkpoints in simulation with the task's env_runner (RLBench by default).

Uses the EMA weights when present and puts the crop randomizers in eval mode (center crop).
rlbench/mt7 runs 175 episodes (25 per task); override with -n.

Usage:
    python eval.py -c path/to/latest.ckpt -o eval_out/
    python eval.py -c path/to/ckpts/ -o eval_out/      # every .ckpt in the directory
"""
import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import json
import re
from typing import List, Optional

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
    """Put every CropRandomizer in eval mode (deterministic center crop); returns how many."""
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
    """Build the checkpoint's workspace (eagerly, so load_payload has modules to fill) and load it."""
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    # FDP's workspace takes (cfg, output_dir, lazy_instantiation); set
    # lazy=False so self.model / self.ema_model exist before load_payload.
    workspace: BaseWorkspace = cls(cfg, output_dir=output_dir, lazy_instantiation=False)
    # skip optimizer state (also lets older checkpoints load)
    workspace.load_payload(payload, exclude_keys=["optimizer"], include_keys=None)
    return workspace, payload, cfg


@click.command()
@click.option("-c", "--checkpoint", required=True,
              help="path to a .ckpt file or a directory containing .ckpt files")
@click.option("-o", "--output_dir", required=True,
              help="directory for eval JSON dumps + rollout videos")
@click.option("-n", "--n_test", default=None, type=int,
              help="override cfg.task.env_runner.n_test (paper default: 175 for mt7 = 25/task)")
@click.option("--test_start_seed", default=None, type=int,
              help="override cfg.task.env_runner.test_start_seed (env seed of episode 0; "
                   "episode i gets test_start_seed+i)")
@click.option("--n_parallel_envs", default=None, type=int,
              help="override cfg.task.env_runner.n_parallel_envs (must be <= n_test; "
                   "lower it for a small-n_test smoke run)")
@click.option("-d", "--device", default="cuda:0", help="device to run on")
@click.option("-u", "--update", is_flag=True,
              help="rename ckpt with the measured success rate (e.g. ep-1000_sr-0.700.ckpt)")
def eval_policy(checkpoint, output_dir, n_test, test_start_seed, n_parallel_envs, device, update):
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

        # Prefer EMA weights if the run trained with EMA, else use the live
        # model (matches PC; matches FDP's own training-time rollout policy).
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

        # -n overrides the task config's n_test
        runner_kwargs = {"output_dir": out}
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

        # Console summary (per-task + aggregate) — matches what PC prints.
        if "mean_success_rate" in runner_log:
            print(f"    mean_success_rate = {runner_log['mean_success_rate']:.3f}")
        for k, v in sorted(runner_log.items()):
            if k.endswith("/mean_success_rate") and k != "mean_success_rate":
                print(f"    {k} = {float(v):.3f}")

        # Persist JSON (same shape as PC's eval_log.json).
        json_log = {
            "checkpoint": ckpt,
            "n_test_override": n_test,
            **runner_log,
        }
        with open(os.path.join(out, "eval_log.json"), "w") as f:
            json.dump(_make_json_safe(json_log), f, indent=2, sort_keys=True)
        print(f"    wrote {os.path.join(out, 'eval_log.json')}")

        # Optionally rename the ckpt with the measured SR.
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
