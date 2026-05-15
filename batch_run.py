# -*- coding: utf-8 -*-
"""
Batch MaterialGAN pipeline.

Iterates over every subfolder of --data-root that contains a `raw/` directory
with input photos, and runs the full capture -> 256 latent -> 512 -> 1024
pipeline on each. Skips folders that already have completed outputs.

Designed for unattended runs on Colab where the session may disconnect — just
re-run and it picks up where it left off.

Usage:
    python batch_run.py --data-root /content/drive/MyDrive/reflective_dataset \\
        --size 17.0 --depth 0.1 --envmap --log /content/drive/MyDrive/materialgan_log.json
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from src.scripts import (
    gen_targets_from_capture,
    optim_ganlatent,
    optim_perpixel,
    render_envmap,
)


COMPLETION_MARKER = Path("optim_latent") / "1024" / "dif.png"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def has_raw_images(folder: Path) -> bool:
    raw = folder / "raw"
    if not raw.is_dir():
        return False
    return any(p.suffix.lower() in IMAGE_SUFFIXES for p in raw.iterdir())


def is_done(folder: Path) -> bool:
    return (folder / COMPLETION_MARKER).exists()


def process_one(folder: Path, size: float, depth: float, do_envmap: bool) -> None:
    gen_targets_from_capture(folder, size=size, depth=depth)
    optim_ganlatent(
        folder / "optim_latent_256.json", 256, 0.02, [1000, 10, 10], tex_init="auto"
    )
    optim_perpixel(
        folder / "optim_pixel_256_to_512.json", 512, 0.01, 20, tex_init="textures"
    )
    optim_perpixel(
        folder / "optim_pixel_512_to_1024.json", 1024, 0.01, 20, tex_init="textures"
    )
    if do_envmap:
        render_envmap(folder / "optim_latent" / "1024", 256)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="Folder containing per-object subfolders (each with raw/).")
    parser.add_argument("--size", type=float, default=17.0,
                        help="AprilTag print size in cm (default: 17.0).")
    parser.add_argument("--depth", type=float, default=0.1,
                        help="Distance between marker plane and material plane in cm (default: 0.1).")
    parser.add_argument("--envmap", action="store_true",
                        help="Also render the environment-map relighting GIF for each object.")
    parser.add_argument("--log", type=Path, default=None,
                        help="Optional path to write a JSON status log.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many folders (debugging).")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated folder names to process (debugging).")
    args = parser.parse_args()

    if not args.data_root.is_dir():
        print(f"ERROR: --data-root {args.data_root} is not a directory", file=sys.stderr)
        return 2

    only_set = set(s.strip() for s in args.only.split(",")) if args.only else None

    candidates = sorted(d for d in args.data_root.iterdir() if d.is_dir())
    folders = [d for d in candidates if has_raw_images(d)]
    if only_set is not None:
        folders = [d for d in folders if d.name in only_set]
    if args.limit is not None:
        folders = folders[: args.limit]

    skipped_no_raw = [d.name for d in candidates if d not in folders and (only_set is None or d.name in only_set)]
    if skipped_no_raw:
        print(f"Note: {len(skipped_no_raw)} folder(s) skipped (no raw/ images): {skipped_no_raw[:10]}{'...' if len(skipped_no_raw) > 10 else ''}")

    print(f"Found {len(folders)} folder(s) to consider under {args.data_root}")
    print(f"Params: size={args.size} cm, depth={args.depth} cm, envmap={args.envmap}")
    print()

    results = []
    t_start = time.time()

    for i, folder in enumerate(folders, 1):
        t0 = time.time()
        header = f"[{i}/{len(folders)}] {folder.name}"
        print("=" * len(header))
        print(header)
        print("=" * len(header))

        if is_done(folder):
            print("  -> already complete (found optim_latent/1024/dif.png), skipping")
            results.append({"folder": folder.name, "status": "skipped", "elapsed_s": 0.0})
            continue

        try:
            process_one(folder, args.size, args.depth, args.envmap)
            elapsed = time.time() - t0
            print(f"  -> OK ({elapsed:.1f}s)")
            results.append({"folder": folder.name, "status": "ok", "elapsed_s": round(elapsed, 1)})
        except Exception as e:
            elapsed = time.time() - t0
            tb = traceback.format_exc()
            print(f"  -> FAILED after {elapsed:.1f}s: {e}")
            print(tb)
            results.append({
                "folder": folder.name,
                "status": "failed",
                "elapsed_s": round(elapsed, 1),
                "error": str(e),
                "trace": tb,
            })

        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            args.log.write_text(json.dumps(results, indent=2))

    total = time.time() - t_start
    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "failed")

    print()
    print("=" * 60)
    print(f"SUMMARY: {ok} processed | {skipped} skipped | {failed} failed | total {total/60:.1f} min")
    print("=" * 60)
    if failed:
        print("Failed folders:")
        for r in results:
            if r["status"] == "failed":
                print(f"  - {r['folder']}: {r['error']}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
