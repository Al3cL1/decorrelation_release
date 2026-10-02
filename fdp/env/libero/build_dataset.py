"""Build the LIBERO zarrs: atomics25_N1250.zarr (25 LIBERO-90 tasks) and composites6_N300.zarr
(6 LIBERO-10 tasks), 50 demos per task.

Each task's hdf5 is converted to a zarr, then the tasks of a group are interleaved in a
random (seeded) episode order and written as one zarr. Needs zarr<3.

Usage:
    python fdp/env/libero/build_dataset.py --download --hdf5_dir $DATA_ROOT/libero/hdf5 --out_dir $DATA_ROOT/libero/zarr
"""
import argparse
import glob
import os
import tempfile

import numcodecs
import numpy as np

from fdp.common.replay_buffer import ReplayBuffer
from fdp.env.libero.dataset_conversion import convert_libero_hdft_to_zarr
from fdp.env.libero.factory import MT_TASKS

HF_REPO = "yifengzhu-hf/LIBERO-datasets"
GROUPS = {"atomics25": "libero_90", "composites6": "libero_10"}   # group -> suite dir in HF_REPO
COMPRESSOR = numcodecs.Blosc(cname='zstd', clevel=5, shuffle=1)


def download(hdf5_dir):
    from huggingface_hub import snapshot_download
    patterns = [f"{suite}/{task}_demo.hdf5" for group, suite in GROUPS.items()
                for task in MT_TASKS[group]]
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=hdf5_dir,
                      allow_patterns=patterns, max_workers=8)


def build_group(group, hdf5_dir, out_dir):
    found = {os.path.basename(p): p
             for p in glob.glob(os.path.join(hdf5_dir, "**", "*.hdf5"), recursive=True)}
    missing = [t for t in MT_TASKS[group] if f"{t}_demo.hdf5" not in found]
    if missing:
        raise FileNotFoundError(f"{len(missing)} {group} hdf5s missing under {hdf5_dir}, "
                                f"e.g. {missing[0]}_demo.hdf5 (run with --download)")

    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        # per-task zarrs on disk, so the merge below streams episodes instead of
        # holding every task in memory twice
        buffers = []
        for task in MT_TASKS[group]:
            path = os.path.join(tmp, f"{task}.zarr")
            convert_libero_hdft_to_zarr(found[f"{task}_demo.hdf5"]).save_to_path(
                path, compressors=COMPRESSOR)
            buffers.append(ReplayBuffer.create_from_path(path))

        merged = ReplayBuffer.create_empty_zarr()
        next_ep = [0] * len(buffers)
        while buffers:
            i = np.random.randint(len(buffers))
            merged.add_episode(buffers[i].get_episode(next_ep[i], copy=True))
            next_ep[i] += 1
            if next_ep[i] == buffers[i].n_episodes:
                buffers.pop(i)
                next_ep.pop(i)

    out = os.path.join(out_dir, f"{group}_N{merged.n_episodes}.zarr")
    merged.save_to_path(out, compressors=COMPRESSOR)
    print(f"{out}:\n{merged}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5_dir", required=True, help="where the LIBERO hdf5s are (or go, with --download)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--download", action="store_true", help=f"fetch the 31 hdf5s from {HF_REPO} first")
    args = p.parse_args()

    np.random.seed(0)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.download:
        download(args.hdf5_dir)
    for group in GROUPS:
        build_group(group, args.hdf5_dir, args.out_dir)


if __name__ == "__main__":
    main()
