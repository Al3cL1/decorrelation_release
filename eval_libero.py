"""Evaluate FDP checkpoints on LIBERO.

Same loading as eval.py. --task_yaml rolls a checkpoint out on another task set (e.g.
decomp12.yaml for a composites6-trained policy); --lesion, --keep_only, --uniform_router and
--dump_router are factor analyses.

Usage:
    python eval_libero.py -c path/to/ckpt -o eval_out/ --test_start_seed 7
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
@click.option("--keep_only", default=None, type=int,
              help="enable only factor i (inverse of --lesion)")
@click.option("--lesion", default=None, type=int,
              help="disable factor i at inference")
@click.option("--task_yaml", default=None,
              help="task yaml whose env_runner replaces the checkpoint's (e.g. decomp12.yaml)")
@click.option("--uniform_router", is_flag=True,
              help="use w = 1/K instead of the learned router")
@click.option("--dump_router", default=None,
              help="write the router weights of every policy call to this NPZ")
def eval_policy(checkpoint, output_dir, n_test, test_start_seed, n_parallel_envs, device, update,
                task_yaml, lesion, keep_only, uniform_router, dump_router):
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

        if keep_only is not None:
            m = getattr(policy, "inference_time_mask", None)
            if m is None or keep_only >= len(m):
                raise click.UsageError(f"--keep_only {keep_only} invalid; K={len(m) if m else '?'}")
            for i in range(len(m)):
                policy.inference_time_mask[i] = (i == keep_only)
            print(f"    [keep_only] factor {keep_only} -> mask {policy.inference_time_mask}")

        if lesion is not None:
            m = getattr(policy, "inference_time_mask", None)
            if m is None or lesion >= len(m):
                raise click.UsageError(f"--lesion {lesion} invalid; policy has K={len(m) if m else '?'}")
            policy.inference_time_mask[lesion] = False
            print(f"    [lesion] factor {lesion} disabled -> mask {policy.inference_time_mask}")

        if uniform_router:
            # the flag is only applied on the obs-only router branch
            if getattr(policy, "use_state_dependent_weighting", False) \
               or getattr(policy, "use_time_dependent_weighting", False):
                raise click.UsageError(
                    "--uniform_router is only wired for the obs-only router branch; "
                    "this ckpt uses state/time-dependent weighting.")
            policy.uniform_router = True
            print(f"    [uniform_router] w = 1/K (K={len(policy.inference_time_mask)}), learned router bypassed")

        if dump_router is not None:
            policy._router_log = []
            print(f"    [dump_router] recording router weights -> {dump_router}")

        n_fixed = _fix_crop_randomizers(policy)
        print(f"    crop randomizers set to eval mode: {n_fixed}")

        policy.to(device)
        policy.eval()

        # n_test_vis=0 disables per-episode mp4 recording (LiberoRunner honors
        # it). Skips wandb_video uploads too.
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

        if dump_router is not None:
            w = np.concatenate(policy._router_log, axis=0)   # (n_calls*B, K)
            np.savez_compressed(dump_router, weights=w,
                                checkpoint=np.array(ckpt),
                                n_calls=np.array(len(policy._router_log)))
            print(f"    [dump_router] {w.shape[0]} records x K={w.shape[1]} -> {dump_router}")
            print(f"      per-factor mean {np.round(w.mean(0), 3).tolist()}")
            print(f"      min {w.min():.3f}  max {w.max():.3f}  "
                  f"frac|w|>1: {(np.abs(w) > 1).mean():.4f}  sum-to-one err {np.abs(w.sum(1) - 1).max():.2e}")
            policy._router_log = None

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
