"""Merge Reddit and Tumblr parquets into unified pipeline inputs.

Reads:
    data/user_timelines.parquet       (Reddit)
    data/user_labels.parquet          (Reddit)
    data/user_timelines_tumblr.parquet
    data/user_labels_tumblr.parquet

Writes:
    data/user_timelines_merged.parquet
    data/user_labels_merged.parquet

Then you can run the full pipeline on the merged files by editing
DATA_DIR paths, or by passing --timelines / --labels flags (if added).

Usage
-----
    python src/merge_sources.py
    python src/merge_sources.py --replace   # overwrite the original files
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replace", action="store_true",
                   help="Overwrite user_timelines.parquet and user_labels.parquet "
                        "with merged data (makes merged data the new default)")
    args = p.parse_args()

    # ── Load available sources ────────────────────────────────────────────
    sources_tl  = []
    sources_lbl = []

    for name, tl_fn, lbl_fn in [
        ("reddit",  "user_timelines.parquet",        "user_labels.parquet"),
        ("tumblr",  "user_timelines_tumblr.parquet", "user_labels_tumblr.parquet"),
    ]:
        tl_path  = DATA_DIR / tl_fn
        lbl_path = DATA_DIR / lbl_fn
        if tl_path.exists() and lbl_path.exists():
            tl  = pd.read_parquet(tl_path)
            lbl = pd.read_parquet(lbl_path)
            tl["source"]  = name
            lbl["source"] = name
            sources_tl.append(tl)
            sources_lbl.append(lbl)
            print(f"[merge] loaded {name}: {len(tl)} timeline rows, "
                  f"{len(lbl)} label rows")
        else:
            print(f"[merge] skipping {name} (files not found)")

    if not sources_tl:
        print("[merge] Nothing to merge.")
        return

    # ── Merge & deduplicate ───────────────────────────────────────────────
    tl_merged  = pd.concat(sources_tl,  ignore_index=True).drop_duplicates("id")
    lbl_merged = pd.concat(sources_lbl, ignore_index=True).drop_duplicates("author")

    # Ensure datetime dtype
    tl_merged["created_utc"] = pd.to_datetime(tl_merged["created_utc"], utc=True)

    print(f"\n[merge] merged: {len(tl_merged)} timeline rows, "
          f"{len(lbl_merged)} label rows")
    print(f"  Label breakdown:\n{lbl_merged['label'].value_counts().to_string()}")
    print(f"  Source breakdown:\n{lbl_merged['source'].value_counts().to_string()}")

    # ── Save ─────────────────────────────────────────────────────────────
    if args.replace:
        tl_out  = DATA_DIR / "user_timelines.parquet"
        lbl_out = DATA_DIR / "user_labels.parquet"
        # Back up originals first
        for fn in ["user_timelines.parquet", "user_labels.parquet"]:
            src = DATA_DIR / fn
            if src.exists():
                src.rename(DATA_DIR / fn.replace(".parquet", "_reddit_backup.parquet"))
                print(f"[merge] backed up {fn}")
    else:
        tl_out  = DATA_DIR / "user_timelines_merged.parquet"
        lbl_out = DATA_DIR / "user_labels_merged.parquet"

    tl_merged.to_parquet(tl_out,  index=False)
    lbl_merged.to_parquet(lbl_out, index=False)
    print(f"\n[merge] saved -> {tl_out}")
    print(f"[merge] saved -> {lbl_out}")

    if not args.replace:
        print("\n  To use merged data as the pipeline default, re-run with --replace")


if __name__ == "__main__":
    main()
