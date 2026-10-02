"""Stitch the composed / factor_0 / factor_1 quiver PNGs from visualize.py --mode fields side by
side, one combined PNG per lambda.

Usage:
    python utils.py combine --root outputs/2D_traj --panels composed factor_0 factor_1
"""
from __future__ import annotations

import argparse
import pathlib

from PIL import Image


def combine_lambda_panels(root: pathlib.Path, panels: list[str], out_subdir: str = "combined"):
    """For each filename in <root>/<panels[0]>/, paste the matching files from
    every panel side-by-side into <root>/<out_subdir>/<filename>."""
    panel_dirs = [root / p for p in panels]
    for p in panel_dirs:
        if not p.is_dir():
            raise FileNotFoundError(f"missing panel directory: {p}")
    out_dir = root / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    filenames = sorted(f.name for f in panel_dirs[0].iterdir() if f.suffix == ".png")
    if not filenames:
        print(f"no PNGs found in {panel_dirs[0]}; nothing to combine.")
        return

    for fname in filenames:
        srcs = [pdir / fname for pdir in panel_dirs]
        missing = [str(s) for s in srcs if not s.exists()]
        if missing:
            print(f"skipping {fname}: missing {missing}")
            continue
        imgs = [Image.open(s) for s in srcs]
        max_h = max(im.height for im in imgs)
        total_w = sum(im.width for im in imgs)
        canvas = Image.new("RGB", (total_w, max_h), "white")
        x = 0
        for im in imgs:
            y = (max_h - im.height) // 2
            canvas.paste(im, (x, y))
            x += im.width
        canvas.save(out_dir / fname)
    print(f"wrote {len(filenames)} combined PNG(s) to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    cp = sub.add_parser("combine", help="horizontally stitch per-panel PNGs into one per lambda")
    cp.add_argument("--root", default="outputs/2D_traj",
                    help="root containing one subdirectory per panel")
    cp.add_argument("--panels", nargs="+", default=["composed", "factor_0", "factor_1"],
                    help="panel subdirectory names, in left-to-right order")
    cp.add_argument("--out_subdir", default="combined",
                    help="subdirectory under root to write combined PNGs into")
    args = ap.parse_args()
    if args.cmd == "combine":
        combine_lambda_panels(pathlib.Path(args.root), args.panels, args.out_subdir)


if __name__ == "__main__":
    main()
