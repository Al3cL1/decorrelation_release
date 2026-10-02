"""Plot a grid of training examples for inspection of the training distribution."""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from data import ProblemSampler
from visualize import draw_problem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--dataset_size", type=int, default=4096)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--out_path", type=str, default="outputs/training_examples.png")
    args = ap.parse_args()

    sampler = ProblemSampler(dataset_size=args.dataset_size, seed=args.seed)
    ds = sampler.dataset
    rng = np.random.default_rng(args.seed + 1234)
    idx = rng.choice(ds.traj.shape[0], size=args.rows * args.cols, replace=False)

    fig, axes = plt.subplots(
        args.rows, args.cols,
        figsize=(3.5 * args.cols, 3.5 * args.rows),
        squeeze=False,
    )
    for ax, i in zip(axes.ravel(), idx):
        start = ds.starts[i].numpy()
        goal = ds.goals[i].numpy()
        traj = ds.traj[i].numpy()
        full = np.concatenate([start[None, :], traj], axis=0)
        draw_problem(ax, start, goal)
        ax.plot(full[:, 0], full[:, 1], color="tab:blue", linewidth=1.8)
        ax.set_title(f"ex {i}")

    out_path = pathlib.Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(f"training examples  seed={args.seed}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
