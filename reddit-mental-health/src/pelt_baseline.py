"""Offline change-point detection baseline using PELT.

Stage 6 of the pipeline. For each user we build a weekly sentiment
time series from ``data/user_timelines.parquet`` and run PELT
(van den Burg & Williams 2020) via the ``ruptures`` library. For the
132 crisis/recovery users with a known turning point we report the
hit rate of the nearest detected change-point at +/- 1 and +/- 2 weeks
against the ground-truth turning-point date.

This provides a non-learned baseline that competes with the
classifier on the Moments-of-Change task as formulated by
Tsakalidis et al. (ACL 2022), without relying on any supervised
label signal.

Comparison
----------
For each labelled user we compute:
    - nearest CP distance to the true TP in weeks
    - hit@1w: nearest distance <= 1 week
    - hit@2w: nearest distance <= 2 weeks
We also compare to a null model: uniform-random CP selection across
the user's active timeline yields an expected hit@1w rate of
(2 weeks) / (active weeks).

Run:
    python src/pelt_baseline.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DATA_DIR     = Path(__file__).resolve().parent.parent / "data"
LABELS_IN    = DATA_DIR / "user_labels.parquet"
TIMELINES_IN = DATA_DIR / "user_timelines.parquet"
RESULTS_OUT  = DATA_DIR / "pelt_baseline.json"

# PELT hyperparameters
PELT_MODEL   = "l2"    # change in mean of a Gaussian signal
PELT_PENALTY = 1.5     # lower = more CPs, higher = fewer. Tuned to 1-2 CPs/user
MIN_WEEKS    = 6       # don't run PELT on users shorter than this


# ── Per-user time series ──────────────────────────────────────────────────

VADER = SentimentIntensityAnalyzer()


def _post_sentiment(row: pd.Series) -> float:
    text = (str(row.get("title") or "") + " " +
            str(row.get("body")  or "")).strip()
    if not text:
        return float("nan")
    return VADER.polarity_scores(text)["compound"]


def build_weekly_series(
    user_posts: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a per-ISO-week mean-sentiment time series for one user.

    Returns
    -------
    week_starts : array of pd.Timestamp for each bucket (monotonically increasing)
    values      : array of mean VADER compound per week (NaN for empty weeks,
                  filled by carry-forward from previous week)
    """
    posts = user_posts.copy()
    posts["sent"] = posts.apply(_post_sentiment, axis=1)
    posts = posts.dropna(subset=["sent", "created_utc"])
    if posts.empty:
        return np.array([]), np.array([])

    # Bucket by ISO week (Monday start)
    posts["week"] = posts["created_utc"].dt.tz_localize(None).dt.to_period("W-MON").dt.start_time

    weekly = posts.groupby("week")["sent"].mean().sort_index()
    # Re-index to fill missing weeks with NaN, then forward-fill
    full_range = pd.date_range(weekly.index.min(), weekly.index.max(), freq="W-TUE")
    weekly = weekly.reindex(full_range)
    weekly = weekly.ffill().bfill()    # handle leading NaNs too

    return np.array(weekly.index), weekly.values.astype(float)


# ── PELT detection ────────────────────────────────────────────────────────

def detect_changepoints(values: np.ndarray) -> list[int]:
    """Run PELT on a 1-D series and return change-point indices.

    ruptures returns 1-based end-of-segment indices (including the final
    n). We convert to 0-based CP positions inside the series (exclusive
    of the trailing n) so they can be used to slice ``week_starts``.
    """
    if len(values) < MIN_WEEKS:
        return []
    algo = rpt.Pelt(model=PELT_MODEL).fit(values.reshape(-1, 1))
    cps  = algo.predict(pen=PELT_PENALTY)   # includes len(values)
    return [c for c in cps if c < len(values)]


def nearest_cp_distance_weeks(
    cp_indices: list[int],
    week_starts: np.ndarray,
    tp_date: pd.Timestamp,
) -> float:
    """Distance (weeks, absolute) from nearest CP to the true TP date."""
    if not cp_indices:
        return float("inf")
    if pd.isnull(tp_date):
        return float("nan")
    tp = pd.Timestamp(tp_date).tz_localize(None) if tp_date.tzinfo else pd.Timestamp(tp_date)
    cp_dates = week_starts[np.array(cp_indices)]
    diffs = np.abs((cp_dates - np.datetime64(tp)).astype("timedelta64[D]")
                    .astype(float) / 7.0)
    return float(diffs.min())


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    import json
    print("[pelt_baseline] loading data...")
    labels    = pd.read_parquet(LABELS_IN)
    timelines = pd.read_parquet(TIMELINES_IN)
    if "low_confidence" not in labels.columns:
        labels["low_confidence"] = False

    labelled = labels[labels["label"].isin(["crisis", "recovery"])].copy()
    print(f"[pelt_baseline] {len(labelled)} labelled users (crisis+recovery)")

    labels_idx = labels.set_index("author")
    results: list[dict] = []
    no_cp = 0
    for author, grp in timelines.groupby("author", sort=False):
        if author not in labels_idx.index:
            continue
        lbl = labels_idx.loc[author]
        if lbl["label"] not in ("crisis", "recovery"):
            continue
        tp_date = lbl["tp_date"]

        week_starts, vals = build_weekly_series(grp)
        if len(vals) < MIN_WEEKS:
            continue
        cps = detect_changepoints(vals)
        dist = nearest_cp_distance_weeks(cps, week_starts, tp_date)

        if not cps:
            no_cp += 1

        results.append({
            "author":     author,
            "label":      lbl["label"],
            "n_weeks":    int(len(vals)),
            "n_cps":      int(len(cps)),
            "nearest_cp_dist_weeks": dist,
            "tp_date":    str(tp_date),
        })

    df = pd.DataFrame(results)
    n = len(df)
    print(f"[pelt_baseline] evaluated {n} users "
          f"(skipped {len(labelled) - n} with <{MIN_WEEKS} active weeks)")
    print(f"[pelt_baseline] users with >=1 detected CP: {n - no_cp}/{n}")
    print(f"[pelt_baseline] mean CPs per user: {df['n_cps'].mean():.2f}")

    # Restrict hit-rate stats to users that had at least one CP
    with_cp = df[df["nearest_cp_dist_weeks"] != float("inf")]
    hit_1 = (with_cp["nearest_cp_dist_weeks"] <= 1.0).mean()
    hit_2 = (with_cp["nearest_cp_dist_weeks"] <= 2.0).mean()
    hit_4 = (with_cp["nearest_cp_dist_weeks"] <= 4.0).mean()

    # Overall (users with 0 CPs count as misses)
    hit_1_all = (df["nearest_cp_dist_weeks"] <= 1.0).mean()
    hit_2_all = (df["nearest_cp_dist_weeks"] <= 2.0).mean()
    hit_4_all = (df["nearest_cp_dist_weeks"] <= 4.0).mean()

    # Random-CP null baseline: E[hit@+-k weeks] = 2k / n_weeks per user
    mean_active_weeks = df["n_weeks"].mean()
    null_hit_1 = 2.0 / mean_active_weeks
    null_hit_2 = 4.0 / mean_active_weeks
    null_hit_4 = 8.0 / mean_active_weeks

    print("\n===== PELT change-point hit rate vs ground-truth TP =====")
    print(f"  mean active weeks / user: {mean_active_weeks:.1f}")
    print(f"  tolerance      PELT (with-CP)   PELT (all)   random null")
    print(f"  +- 1 week       {hit_1:.3f}          {hit_1_all:.3f}       {null_hit_1:.3f}")
    print(f"  +- 2 weeks      {hit_2:.3f}          {hit_2_all:.3f}       {null_hit_2:.3f}")
    print(f"  +- 4 weeks      {hit_4:.3f}          {hit_4_all:.3f}       {null_hit_4:.3f}")

    # Per-class breakdown
    print("\n===== PELT hit rate by label =====")
    for lbl in ["crisis", "recovery"]:
        sub = df[df["label"] == lbl]
        if sub.empty:
            continue
        h1 = (sub["nearest_cp_dist_weeks"] <= 1.0).mean()
        h2 = (sub["nearest_cp_dist_weeks"] <= 2.0).mean()
        h4 = (sub["nearest_cp_dist_weeks"] <= 4.0).mean()
        n_cp = (sub["n_cps"] > 0).sum()
        print(f"  {lbl:<10}  n={len(sub)}  with_cp={n_cp}  "
              f"hit@1w={h1:.3f}  hit@2w={h2:.3f}  hit@4w={h4:.3f}")

    payload = {
        "n_users":             n,
        "mean_active_weeks":   float(mean_active_weeks),
        "users_with_cp":       int(n - no_cp),
        "mean_cps_per_user":   float(df["n_cps"].mean()),
        "hit_rate_with_cp":    {"1w": float(hit_1), "2w": float(hit_2), "4w": float(hit_4)},
        "hit_rate_all":        {"1w": float(hit_1_all), "2w": float(hit_2_all), "4w": float(hit_4_all)},
        "random_null":         {"1w": float(null_hit_1), "2w": float(null_hit_2), "4w": float(null_hit_4)},
        "pelt_model":          PELT_MODEL,
        "pelt_penalty":        PELT_PENALTY,
        "min_weeks":           MIN_WEEKS,
    }
    with open(RESULTS_OUT, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[pelt_baseline] saved to {RESULTS_OUT}")


if __name__ == "__main__":
    main()
