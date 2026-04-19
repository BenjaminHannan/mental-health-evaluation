"""Load the Reddit mental health dataset from HuggingFace.

Stage 1 of the pipeline. Downloads `solomonk/reddit_mental_health_posts`
(~151k posts from r/depression, r/ADHD, r/PTSD, r/OCD, r/aspergers with
author, title, body, created_utc, subreddit), prints basic statistics,
shows sample posts for a sanity check, and writes a parquet file to
``data/raw_posts.parquet`` that downstream stages read.

Run:
    python src/load_data.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET_NAME = "solomonk/reddit_mental_health_posts"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "raw_posts.parquet"


def load_raw() -> pd.DataFrame:
    """Download the HF dataset and return it as a pandas DataFrame."""
    print(f"[load_data] Downloading {DATASET_NAME} from HuggingFace...")
    ds = load_dataset(DATASET_NAME, split="train")
    df = ds.to_pandas()
    print(f"[load_data] Loaded {len(df):,} rows, columns: {list(df.columns)}")
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize timestamps and drop rows unusable for timeline analysis.

    - Parse ``created_utc`` to a tz-aware datetime.
    - Drop rows with missing author/body/timestamp.
    - Drop ``[deleted]`` / ``[removed]`` authors (no timeline possible).
    """
    df = df.copy()
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True, errors="coerce")
    before = len(df)
    df = df.dropna(subset=["author", "body", "created_utc"])
    df = df[~df["author"].isin(["[deleted]", "[removed]"])]
    df = df[df["body"].str.strip().astype(bool)]
    df = df[~df["body"].str.strip().isin(["[deleted]", "[removed]"])]
    print(f"[load_data] Kept {len(df):,}/{before:,} rows after cleaning")
    return df.reset_index(drop=True)


def print_stats(df: pd.DataFrame) -> None:
    """Print posts, unique users, date range, and per-subreddit counts."""
    n_posts = len(df)
    n_users = df["author"].nunique()
    date_min = df["created_utc"].min()
    date_max = df["created_utc"].max()
    print("\n===== Dataset statistics =====")
    print(f"Total posts       : {n_posts:,}")
    print(f"Unique authors    : {n_users:,}")
    print(f"Date range        : {date_min}  ->  {date_max}")
    print(f"Posts per author  : mean={n_posts / n_users:.1f}, "
          f"median={int(df.groupby('author').size().median())}, "
          f"max={int(df.groupby('author').size().max())}")
    print("\nPosts per subreddit:")
    print(df["subreddit"].value_counts().to_string())
    mp = df.groupby("author").size()
    print("\nAuthors with >=5 posts: "
          f"{(mp >= 5).sum():,} "
          f"(>= 10 posts: {(mp >= 10).sum():,})")


def print_samples(df: pd.DataFrame, n: int = 5) -> None:
    """Print ``n`` sample posts so we can eyeball the raw content."""
    print(f"\n===== {n} sample posts =====")
    sample = df.sample(n=n, random_state=42)
    for i, row in enumerate(sample.itertuples(index=False), 1):
        body = (row.body or "")[:400].replace("\n", " ")
        title = (row.title or "")[:120].replace("\n", " ")
        print(f"\n--- Sample {i} ---")
        print(f"subreddit : r/{row.subreddit}")
        print(f"author    : {row.author}")
        print(f"timestamp : {row.created_utc}")
        print(f"title     : {title}")
        print(f"body      : {body}{'...' if len(row.body or '') > 400 else ''}")


def main() -> None:
    """Run stage 1: load, clean, report, and save to parquet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = normalize(load_raw())
    print_stats(df)
    print_samples(df, n=5)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\n[load_data] Wrote cleaned posts to {OUT_PATH}")


if __name__ == "__main__":
    main()
