"""Extract temporal posting-behaviour features per user and window.

Stage 3b of the pipeline. Reads ``data/user_timelines.parquet`` and
``data/user_labels.parquet`` and computes 6 temporal features across
the same four windows used by ``extract_features.py``:

    baseline : all posts older than (TP - 4 weeks)
    pre_4w   : posts in [TP - 4w, TP)
    pre_2w   : posts in [TP - 2w, TP)
    pre_1w   : posts in [TP - 1w, TP)

Temporal features
-----------------
1. hour_entropy       : normalised Shannon entropy of the hour-of-day
                        distribution (0 = same hour, 1 = uniform)
2. late_night_rate    : fraction of posts with hour in [0, 4)
                        (midnight - 4 a.m., local UTC)
3. interval_mean_hr   : mean time between consecutive posts (hours)
4. interval_std_hr    : std of inter-post intervals (hours)
5. max_gap_hr         : largest gap between consecutive posts (hours)
6. weekend_rate       : fraction of posts on Saturday or Sunday

The output is a wide-format parquet with columns
``<feature>_<window>`` plus metadata columns. Intended to be merged
with the raw and z-norm feature parquets in ``train_model.py``.

Run:
    python src/extract_temporal.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR       = Path(__file__).resolve().parent.parent / "data"
LABELS_IN      = DATA_DIR / "user_labels.parquet"
TIMELINES_IN   = DATA_DIR / "user_timelines.parquet"
FEATURES_OUT   = DATA_DIR / "features_temporal.parquet"

WINDOW_WEEKS = 4

WINDOWS: list[tuple[str, Optional[float]]] = [
    ("baseline", None),
    ("pre_4w",   4.0),
    ("pre_2w",   2.0),
    ("pre_1w",   1.0),
]

TEMPORAL_FEATURES = [
    "hour_entropy", "late_night_rate",
    "interval_mean_hr", "interval_std_hr", "max_gap_hr",
    "weekend_rate",
]


# ── Feature helpers ────────────────────────────────────────────────────────

def _hour_entropy(hours: np.ndarray) -> float:
    """Normalised Shannon entropy of the hour-of-day distribution.

    Returns 0 when all posts share a single hour and 1 when posts are
    distributed uniformly across the 24 hours. NaN if no posts.
    """
    if len(hours) == 0:
        return float("nan")
    counts = np.bincount(hours, minlength=24).astype(float)
    p = counts / counts.sum()
    # Shannon entropy with log2, ignoring zero bins
    nonzero = p[p > 0]
    H = -np.sum(nonzero * np.log2(nonzero))
    # Normalise by log2(24) = max entropy
    return float(H / np.log2(24))


def _late_night_rate(hours: np.ndarray) -> float:
    """Fraction of posts with hour in [0, 4). NaN if no posts."""
    if len(hours) == 0:
        return float("nan")
    return float(((hours >= 0) & (hours < 4)).mean())


def _weekend_rate(weekdays: np.ndarray) -> float:
    """Fraction of posts on Saturday (5) or Sunday (6). NaN if no posts."""
    if len(weekdays) == 0:
        return float("nan")
    return float((weekdays >= 5).mean())


def _interval_stats(timestamps: pd.Series) -> tuple[float, float, float]:
    """Return (mean, std, max) of inter-post intervals in hours.

    Requires >=2 posts; otherwise NaNs.
    """
    if len(timestamps) < 2:
        return (float("nan"), float("nan"), float("nan"))
    ts = timestamps.sort_values().values
    # numpy timedelta -> seconds -> hours
    diffs = np.diff(ts).astype("timedelta64[s]").astype(float) / 3600.0
    return (
        float(np.mean(diffs)),
        float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float("nan"),
        float(np.max(diffs)),
    )


def extract_temporal_window(posts: pd.DataFrame) -> dict[str, float]:
    """Compute all 6 temporal features for posts in a single window."""
    if posts.empty:
        return {f: float("nan") for f in TEMPORAL_FEATURES}

    ts        = posts["created_utc"]
    hours     = ts.dt.hour.to_numpy()
    weekdays  = ts.dt.weekday.to_numpy()

    mean_int, std_int, max_gap = _interval_stats(ts)

    return {
        "hour_entropy":     _hour_entropy(hours),
        "late_night_rate":  _late_night_rate(hours),
        "interval_mean_hr": mean_int,
        "interval_std_hr":  std_int,
        "max_gap_hr":       max_gap,
        "weekend_rate":     _weekend_rate(weekdays),
    }


# ── Window slicing (mirrors extract_features.py) ──────────────────────────

def _window_posts(
    user_posts: pd.DataFrame,
    tp_date: pd.Timestamp,
    window: str,
) -> pd.DataFrame:
    """Return the subset of posts falling inside the named window."""
    ts = user_posts["created_utc"]
    cutoff_4w = tp_date - pd.Timedelta(weeks=4)
    cutoff_2w = tp_date - pd.Timedelta(weeks=2)
    cutoff_1w = tp_date - pd.Timedelta(weeks=1)

    if window == "baseline":
        mask = ts < cutoff_4w
    elif window == "pre_4w":
        mask = (ts >= cutoff_4w) & (ts < tp_date)
    elif window == "pre_2w":
        mask = (ts >= cutoff_2w) & (ts < tp_date)
    elif window == "pre_1w":
        mask = (ts >= cutoff_1w) & (ts < tp_date)
    else:
        raise ValueError(f"Unknown window: {window!r}")
    return user_posts.loc[mask]


def build_temporal_row(
    author: str,
    user_posts: pd.DataFrame,
    label_row: pd.Series,
) -> dict:
    """Build one wide-format temporal-feature row for a single user."""
    row: dict = {
        "author":          author,
        "label":           label_row["label"],
        "low_confidence":  label_row["low_confidence"],
        "n_posts":         len(user_posts),
        "tp_date":         label_row["tp_date"],
    }

    tp_date = label_row["tp_date"]
    if pd.isnull(tp_date):
        tp_date = user_posts["created_utc"].max()

    for win_name, _ in WINDOWS:
        win_posts = _window_posts(user_posts, tp_date, win_name)
        feats     = extract_temporal_window(win_posts)
        for feat, val in feats.items():
            row[f"{feat}_{win_name}"] = val

    # Delta: pre_window - baseline (same convention as linguistic features)
    for win_name in ("pre_4w", "pre_2w", "pre_1w"):
        for feat in TEMPORAL_FEATURES:
            baseline_val = row.get(f"{feat}_baseline", float("nan"))
            window_val   = row.get(f"{feat}_{win_name}", float("nan"))
            row[f"{feat}_delta_{win_name}"] = window_val - baseline_val

    return row


# ── Summary ────────────────────────────────────────────────────────────────

def print_summary(features: pd.DataFrame) -> None:
    """Print mean temporal features per label × window."""
    windows = ["baseline", "pre_4w", "pre_2w", "pre_1w"]
    labels  = ["crisis", "recovery", "neither"]
    print("\n===== Mean temporal features by label x window =====\n")
    for feat in TEMPORAL_FEATURES:
        cols = [f"{feat}_{w}" for w in windows]
        print(f"  {feat}")
        header = f"    {'label':<12}" + "".join(f"{w:>14}" for w in windows)
        print(header)
        print("    " + "-" * (12 + 14 * len(windows)))
        for lbl in labels:
            subset = features[features["label"] == lbl][cols]
            means  = subset.mean()
            vals   = "".join(f"{means[c]:>14.4f}" for c in cols)
            print(f"    {lbl:<12}{vals}")
        print()


def main() -> None:
    print("[extract_temporal] Loading labels and timelines...")
    user_labels    = pd.read_parquet(LABELS_IN)
    user_timelines = pd.read_parquet(TIMELINES_IN)

    if "low_confidence" not in user_labels.columns:
        user_labels["low_confidence"] = False

    labels_idx = user_labels.set_index("author")
    grouped    = user_timelines.groupby("author", sort=False)
    total      = len(grouped)

    print(f"[extract_temporal] Extracting temporal features for {total} authors...")
    rows: list[dict] = []
    for i, (author, user_posts) in enumerate(grouped, 1):
        if i % 100 == 0:
            print(f"[extract_temporal]   {i}/{total}...", flush=True)
        if author not in labels_idx.index:
            continue
        label_row = labels_idx.loc[author]
        rows.append(build_temporal_row(author, user_posts, label_row))

    features = pd.DataFrame(rows)
    features.to_parquet(FEATURES_OUT, index=False)
    print(f"[extract_temporal] Saved {len(features)} rows to {FEATURES_OUT}")
    print_summary(features)


if __name__ == "__main__":
    main()
