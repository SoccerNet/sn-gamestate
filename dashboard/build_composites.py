#!/usr/bin/env python3
"""Render resumable synchronized broadcast + minimap playback for every clip."""
import argparse
import glob
from pathlib import Path

from dashboard.video import build_composite_video


def sequences(split):
    paths = glob.glob(f"dashboard/data/{split}/*.parquet")
    return [int(Path(path).stem) for path in paths if not path.endswith("_forecast.parquet")]


def build(splits, overwrite=False):
    work = [(split, seq) for split in splits for seq in sorted(sequences(split))]
    completed = 0
    for index, (split, seq) in enumerate(work, start=1):
        output = Path(f"dashboard/videos/{split}/{seq:03d}_composite.mp4")
        if output.exists() and not overwrite:
            print(f"[{index}/{len(work)}] skip {split} {seq:03d}", flush=True)
            continue
        print(f"[{index}/{len(work)}] render {split} {seq:03d}", flush=True)
        build_composite_video(seq, split, output_path=output)
        completed += 1
    print(f"done: rendered {completed}, total available {len(work)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch render synchronized dashboard playback")
    parser.add_argument("splits", nargs="*", choices=("valid", "test"), default=["valid", "test"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build(args.splits, args.overwrite)
