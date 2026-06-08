"""python src/main_util.py

Command-line entry that uses src.util.util to process the dataset PNGs and print a summary.

Place this file in src/ and run with:
  python -m main_util
or
  python src/main_util.py

It expects the dataset folder at ../dataset/N-code(png) by default (when run from src/).
"""
from __future__ import annotations
import os
import argparse
# Robust import of the util module: ensure src/ is on sys.path and guarantee `util` is defined
import sys
src_dir = os.path.dirname(__file__)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
import importlib
util = None
try:
    util = importlib.import_module('util.util')
except Exception as e:
    # Fail fast: ensure util is always defined or program exits
    print(f"Failed to import util.util from {src_dir}: {e}", file=sys.stderr)
    sys.exit(1)



def main():
    parser = argparse.ArgumentParser(description="Run leather pixel counting for dataset PNGs")
    # Default dataset path should be relative to this file's location (src/),
    # not the current working directory. Compute it from __file__ so running from
    # project root works as expected.
    default_dataset = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "dataset", "N-code(png)"))
    parser.add_argument("dataset_dir", nargs="?", default=default_dataset,
                        help=f"Path to dataset N-code(png) folder (default: {default_dataset})")
    parser.add_argument("--json", help="Optional path to write per-file JSON results")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N files (0 = all)")
    parser.add_argument("--workers", type=int, default=0, help="Number of worker processes to use (0 = cpu_count())")
    parser.add_argument("--scale", type=int, default=1, help="Upscale factor to make pixel unit smaller (integer >=1)")
    args = parser.parse_args()

    # get PNG files
    try:
        files = util.find_png_files(args.dataset_dir)
    except Exception as e:
        print("Error while listing files:", e)
        return

    if not files:
        print("No PNG files found in", args.dataset_dir)
        return

    # optional limit for quick tests
    limit = args.limit
    if limit and limit < len(files):
        files = files[:limit]

    # Helper to trim file names like '1. N0696153.png' -> 'N0696153'
    import re
    def trim_filename(fname: str) -> str:
        name = re.sub(r"^\d+\.\s*", "", fname)  # remove leading numbering like '1. '
        name = re.sub(r"(?i)\.png$", "", name)   # remove .png or .PNG suffix
        return name

    # Print header
    header_name = "NAME"
    header_red = "RED_PX"
    header_green = "GREEN_PX"
    header_red_in = "RED_IN_PCS"
    header_green_in = "GREEN_IN_PCS"
    header_pieces = "PIECES"
    header_avg = "AVG_AREA"
    header_rej_s = "REJ_SMALL"
    header_rej_l = "REJ_LARGE"
    col_name_w = 20
    col_num_w = 12


    results = {}
    total_red = total_used = 0

    # Use multiprocessing Pool to process images in parallel
    import multiprocessing
    num_workers = args.workers if args.workers and args.workers > 0 else (multiprocessing.cpu_count() or 1)

    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            iterable = [(p, args.scale) for p in files]
            it = pool.imap_unordered(util.analyze_image_wrapper, iterable, chunksize=1)
            processed = 0
            for res in it:
                processed += 1
                basename, data = res
                display_name = trim_filename(basename)
                results[basename] = data
                if "error" in data:
                    print(f"[{processed}/{len(files)}] {display_name:<{col_name_w}} ERROR: {data['error']}")
                else:
                    red = data.get("red_pixels", 0)
                    used = data.get("used_red_pixels", 0)
                    util_pct = data.get("utilization", 0.0)
                    total_red += red
                    total_used += used
                    # progress
                    print(f"[{processed}/{len(files)}] Processed {display_name} (used={used}, util={util_pct:.2f}%)", end="\r")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Will print partial results.")

    # Print final minimal table
    header_name = "NAME"
    header_red = "RED_PX"
    header_used = "USED_RED_PX"
    header_util = "UTIL(%)"
    col_name_w = 25
    col_num_w = 12

    print(f"\n{header_name:<{col_name_w}}{header_red:>{col_num_w}}{header_used:>{col_num_w}}{header_util:>{col_num_w}}")
    print("-" * (col_name_w + col_num_w * 3))

    for basename, v in sorted(results.items()):
        display_name = trim_filename(basename)
        if "error" in v:
            print(f"{display_name:<{col_name_w}}{'ERROR':>{col_num_w}}  {v['error']}")
        else:
            red = v.get("red_pixels", 0)
            used = v.get("used_red_pixels", 0)
            util_pct = v.get("utilization", 0.0)
            print(f"{display_name:<{col_name_w}}{red:>{col_num_w}d}{used:>{col_num_w}d}{util_pct:>{col_num_w}.2f}")

    print("-" * (col_name_w + col_num_w * 3))
    total_util = (total_used / total_red * 100.0) if total_red else 0.0
    print(f"{'TOTAL':<{col_name_w}}{total_red:>{col_num_w}d}{total_used:>{col_num_w}d}{total_util:>{col_num_w}.2f}")

    if args.json:
        import json
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("Wrote JSON results to", args.json)


if __name__ == "__main__":
    main()

