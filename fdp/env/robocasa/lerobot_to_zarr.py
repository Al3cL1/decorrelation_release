"""Bake the RoboCasa365 zarr that robocasa/atomics_H16_12D_raw[_tid] trains on.

Reads the mimicgen pretrain split of each task in env._ATOMICS_TASKS (whose order sets
task_id), and keeps the full 16D state, the raw 12D action
[base(4), ctrl(1), eef_pos(3), eef_rot(3), grip(1)] and 128x128 images.

Usage:
    python fdp/env/robocasa/lerobot_to_zarr.py --n-eps 100 --out $DATA_ROOT/robocasa/zarr/atomics_v365_a12_raw_N1000.zarr
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from robocasa.utils.dataset_registry_utils import get_ds_path

from fdp.env.robocasa.env import _ATOMICS_TASKS


class _PatchedLeRobotDataset(LeRobotDataset):
    """LeRobotDataset (lerobot 0.3.3) with two fixes.

    1. _query_hf_dataset skips video keys, which are not parquet columns.
    2. __getitem__ maps the stored episode_index to its position in `episodes`, as lerobot's
       index lookup expects (otherwise a subset of episodes can index out of range).

    __getitem__ is adapted from LeRobotDataset.__getitem__ (Apache-2.0).
    """

    def _query_hf_dataset(self, query_indices):
        cols = set(self.hf_dataset.column_names)
        result = {}
        for key, q_idx in query_indices.items():
            if key not in cols:
                continue  # video keys handled separately by _query_videos
            result[key] = torch.stack(self.hf_dataset.select(q_idx)[key])
        return result

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        ep_idx_orig = item["episode_index"].item()

        if self.episodes is not None:
            # build remap once and cache on the instance
            remap = getattr(self, "_ep_idx_remap", None)
            if remap is None:
                remap = {orig: dense for dense, orig in enumerate(self.episodes)}
                self._ep_idx_remap = remap
            ep_idx_dense = remap[ep_idx_orig]
        else:
            ep_idx_dense = ep_idx_orig

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx_dense)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx_orig)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            for cam in self.meta.camera_keys:
                item[cam] = self.image_transforms(item[cam])

        # Add task as a string
        item["task"] = self.meta.tasks[item["task_index"].item()]
        return item


# --- spec ---

@dataclass
class BakeSpec:
    """Everything needed to bake one zarr from one (sim, task_group)."""
    sim: str
    task_group: str
    tasks: List[str]                             # task names (defines task_id 0..K-1)
    resolve_root: Callable[[str], str]           # task_name → lerobot dataset root path
    rgb_keys: List[Tuple[str, str]]              # (our_key, lerobot_image_key)
    state_slices: Dict[str, Tuple[int, int]]     # our_key → slice into observation.state
    action_slices: List[Tuple[int, int]]         # raw_action[s:e] pieces, concat in order
    action_layout: List[str]                     # per-output-dim names (len == action_dim)
    target_res: int = 128                        # output RGB resolution (square)
    fps: int = 20
    source_version: str = "v365"
    video_backend: str = "torchcodec"


# --- workers (module-level so a spawn Pool can pickle them) ---

_WORKER_DS = None
_WORKER_RGB_KEYS: List[Tuple[str, str]] = []
_WORKER_STATE_SLICES: Dict[str, Tuple[int, int]] = {}
_WORKER_TARGET_RES = 128


def _init_worker(repo_id: str, root_str: str, n_eps: int,
                 rgb_keys: List[Tuple[str, str]],
                 state_slices: Dict[str, Tuple[int, int]],
                 target_res: int, video_backend: str) -> None:
    global _WORKER_DS, _WORKER_RGB_KEYS, _WORKER_STATE_SLICES, _WORKER_TARGET_RES
    _WORKER_DS = _PatchedLeRobotDataset(
        repo_id=repo_id,
        root=Path(root_str),
        episodes=list(range(n_eps)),
        download_videos=False,
        video_backend=video_backend,
    )
    _WORKER_RGB_KEYS = rgb_keys
    _WORKER_STATE_SLICES = state_slices
    _WORKER_TARGET_RES = target_res


def _decode_frame(idx: int) -> dict:
    """Return a per-frame dict of numpy arrays + episode index."""
    item = _WORKER_DS[idx]

    out = {}
    for our_key, lr_key in _WORKER_RGB_KEYS:
        t = item[lr_key]                                      # (3, H, W) float32 [0,1]
        if t.shape[-1] != _WORKER_TARGET_RES:
            t = F.interpolate(t.unsqueeze(0), size=_WORKER_TARGET_RES,
                              mode="area").squeeze(0)
        img = (t.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        out[our_key] = img

    state = item["observation.state"].numpy().astype(np.float32)
    for our_key, (s, e) in _WORKER_STATE_SLICES.items():
        out[our_key] = state[s:e]

    out["_action_raw"] = item["action"].numpy().astype(np.float32)
    out["_episode_index"] = int(item["episode_index"].item())
    return out


# --- bake ---

def _slice_action(raw: np.ndarray, slices: List[Tuple[int, int]]) -> np.ndarray:
    return np.concatenate([raw[:, s:e] for s, e in slices], axis=-1)


def bake(spec: BakeSpec, n_eps_per_task: int, out_path: Path, workers: int) -> None:
    action_dim = sum(e - s for s, e in spec.action_slices)
    assert action_dim == len(spec.action_layout), (
        f"action_layout has {len(spec.action_layout)} dims but action_slices "
        f"sum to {action_dim}"
    )

    # Resolve roots and effective episode counts upfront so we can fail fast.
    print(f"=== {spec.sim} / {spec.task_group} ===", flush=True)
    task_specs = []
    for task_id, task_name in enumerate(spec.tasks):
        root = spec.resolve_root(task_name)
        if not Path(root).exists():
            raise FileNotFoundError(f"missing dataset at {root}")
        meta = LeRobotDatasetMetadata(repo_id=f"local/{task_name}", root=Path(root))
        n_eps_eff = min(n_eps_per_task, meta.total_episodes)
        task_specs.append({"task_id": task_id, "task_name": task_name,
                           "root": root, "n_eps": n_eps_eff})
        print(f"  {task_name:35s} task_id={task_id:2d}  using {n_eps_eff}/{meta.total_episodes} eps",
              flush=True)

    # Allocate zarr (resizable; we extend per task).
    out_path.parent.mkdir(parents=True, exist_ok=True)
    z = zarr.open_group(store=str(out_path), mode="w")
    data = z.create_group("data")
    meta_grp = z.create_group("meta")

    H = spec.target_res
    rgb_chunks = (64, H, H, 3)              # ~3 MB / chunk
    state_chunk_n = 1024                    # frames per state chunk

    rgb_arrs = {
        k: data.create_array(name=k, shape=(0, H, H, 3),
                             chunks=rgb_chunks, dtype="uint8")
        for k, _ in spec.rgb_keys
    }
    state_arrs = {
        k: data.create_array(name=k, shape=(0, e - s),
                             chunks=(state_chunk_n, e - s), dtype="float32")
        for k, (s, e) in spec.state_slices.items()
    }
    task_id_arr = data.create_array("task_id", shape=(0, 1),
                                    chunks=(state_chunk_n, 1), dtype="int32")
    action_arr = data.create_array("action", shape=(0, action_dim),
                                   chunks=(state_chunk_n, action_dim), dtype="float32")

    episode_ends_global: List[int] = []
    cumulative = 0
    flush_every = 512
    ctx = get_context("spawn")     # clean torchcodec init in workers

    def _flush(buf: list, write_offset: int, task_id: int) -> int:
        if not buf:
            return write_offset
        n_b = len(buf)
        for k, _ in spec.rgb_keys:
            arr = rgb_arrs[k]
            arr.resize((write_offset + n_b,) + arr.shape[1:])
            arr[write_offset:write_offset + n_b] = np.stack([f[k] for f in buf])
        for k in spec.state_slices:
            arr = state_arrs[k]
            arr.resize((write_offset + n_b,) + arr.shape[1:])
            arr[write_offset:write_offset + n_b] = np.stack([f[k] for f in buf])
        actions_b = _slice_action(np.stack([f["_action_raw"] for f in buf]),
                                  spec.action_slices)
        action_arr.resize((write_offset + n_b, action_dim))
        action_arr[write_offset:write_offset + n_b] = actions_b
        task_id_arr.resize((write_offset + n_b, 1))
        task_id_arr[write_offset:write_offset + n_b] = np.full((n_b, 1), task_id,
                                                                dtype=np.int32)
        return write_offset + n_b

    for ts in task_specs:
        task_id   = ts["task_id"]
        task_name = ts["task_name"]
        root      = ts["root"]
        n_eps     = ts["n_eps"]
        repo_id   = f"local/{task_name}"

        # Probe to count frames (cheap; just metadata).
        probe = _PatchedLeRobotDataset(
            repo_id=repo_id, root=Path(root),
            episodes=list(range(n_eps)),
            download_videos=False, video_backend=spec.video_backend,
        )
        n_frames = len(probe)
        del probe

        t_task = time.time()
        print(f"\n[{task_id+1}/{len(task_specs)}] {task_name}: decoding {n_frames} frames "
              f"({n_eps} eps) with {workers} workers", flush=True)

        ep_indices_task: List[int] = []
        write_offset = cumulative
        buf: List[dict] = []

        with ctx.Pool(workers,
                      initializer=_init_worker,
                      initargs=(repo_id, root, n_eps,
                                spec.rgb_keys, spec.state_slices,
                                spec.target_res, spec.video_backend)) as pool:
            for i, frame in enumerate(pool.imap(_decode_frame, range(n_frames),
                                                chunksize=8)):
                ep_indices_task.append(frame["_episode_index"])
                buf.append(frame)
                if len(buf) >= flush_every:
                    write_offset = _flush(buf, write_offset, task_id)
                    buf = []
                if i and (i % 5000 == 0):
                    elapsed = time.time() - t_task
                    eta = elapsed * (n_frames - i) / i
                    print(f"    {i}/{n_frames}  ({i/elapsed:.0f} fps, eta {eta/60:.1f} min)",
                          flush=True)
            write_offset = _flush(buf, write_offset, task_id)

        # Episode boundaries: where episode_index changes, plus task end.
        ep = np.asarray(ep_indices_task)
        boundaries = list(np.where(np.diff(ep) != 0)[0] + 1)
        ep_ends_local = boundaries + [n_frames]
        episode_ends_global.extend(cumulative + b for b in ep_ends_local)

        cumulative = write_offset
        elapsed = time.time() - t_task
        print(f"  done in {elapsed/60:.1f} min ({n_frames/elapsed:.0f} fps avg)",
              flush=True)

    # Write episode_ends.
    ep_ends_arr = np.asarray(episode_ends_global, dtype=np.int64)
    meta_grp.create_array("episode_ends", shape=ep_ends_arr.shape,
                          chunks=(min(len(ep_ends_arr), 4096),), dtype="int64")
    meta_grp["episode_ends"][:] = ep_ends_arr

    # Schema header.
    z.attrs["schema_version"]   = 1
    z.attrs["sim"]              = spec.sim
    z.attrs["task_group"]       = spec.task_group
    z.attrs["source_version"]   = spec.source_version
    z.attrs["fps"]              = spec.fps
    z.attrs["image_resolution"] = spec.target_res
    z.attrs["action_layout"]    = spec.action_layout
    z.attrs["task_id_map"]      = {i: t for i, t in enumerate(spec.tasks)}

    print(f"\nDone: {len(episode_ends_global)} episodes, {cumulative} frames", flush=True)
    print(f"Wrote {out_path}", flush=True)


# --- the atomics spec ---

# observation.state: base_pos 0:3, base_rot 3:7, eef_pos 7:10, eef_rot 10:14, gripper 14:16
_ROBOCASA_RGB_KEYS = [
    ("robot0_agentview_left_rgb",  "observation.images.robot0_agentview_left"),
    ("robot0_agentview_right_rgb", "observation.images.robot0_agentview_right"),
    ("robot0_eye_in_hand_rgb",     "observation.images.robot0_eye_in_hand"),
]
_ROBOCASA_STATE_SLICES_FULL = {
    "robot0_base_pos":          (0, 3),
    "robot0_base_quat":         (3, 7),
    "robot0_base_to_eef_pos":   (7, 10),
    "robot0_base_to_eef_quat":  (10, 14),
    "robot0_gripper_qpos":      (14, 16),
}


def _robocasa_root(task: str, source: str, split: str) -> str:
    rel = get_ds_path(task=task, source=source, split=split)
    if rel is None:
        raise FileNotFoundError(f"no registered dataset for task={task} source={source} split={split}")
    return rel


ATOMICS_SPEC = BakeSpec(
    sim="robocasa",
    task_group="atomics",
    tasks=_ATOMICS_TASKS,
    resolve_root=lambda t: _robocasa_root(t, "mg", "pretrain"),
    rgb_keys=_ROBOCASA_RGB_KEYS,
    state_slices=_ROBOCASA_STATE_SLICES_FULL,
    action_slices=[(0, 12)],                            # NO reorder — raw layout
    action_layout=[
        "base_x", "base_y", "base_rot_z", "base_z",
        "control_mode",
        "eef_pos_x", "eef_pos_y", "eef_pos_z",
        "eef_rot_x", "eef_rot_y", "eef_rot_z",
        "grip",
    ],
)


# --- CLI ---

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-eps", type=int, required=True,
                   help="cap episodes per task (effective = min(n_eps, total_available))")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    bake(ATOMICS_SPEC, args.n_eps, args.out, args.workers)


if __name__ == "__main__":
    main()
