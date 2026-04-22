"""Extract additional hand-engineered features (readability, punctuation, pronouns).

Stage 3d of the pipeline. Adds 8 features per window on top of the 7
already-present linguistic features, computed from the same four
sliding windows:

    flesch_reading_ease   : higher = easier to read (<= 100)
    flesch_kincaid_grade  : US reading-grade level
    exclaim_rate          : '!' per word
    question_rate         : '?' per word
    ellipsis_rate         : '...' occurrences per post
    caps_word_rate        : ALL-CAPS-word rate (length >= 2)
    i_vs_we_ratio         : log((#I)+1 / (#we)+1)  → isolation signal
    i_vs_you_ratio        : log((#I)+1 / (#you)+1) → self-focus signal

Why these?
    - Readability drops in acute crisis (shorter sentences, simpler words).
    - Exclamation and ellipsis spikes are common pre-crisis affect markers.
    - I/we and I/you log-ratios capture isolation/self-focus that a raw
      first-person rate cannot.
    - ALL-CAPS rate is a common informal intensity marker on Reddit.

Output
------
    data/features_bonus.parquet

One row per user with columns like ``flesch_reading_ease_baseline``,
``flesch_reading_ease_delta_pre_1w``, and so on for all 8 bonus
features across 4 windows + 3 delta windows = 56 columns plus
metadata.

Run
---
    python src/extract_bonus_features.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import textstat

DATA_DIR      = Path(__file__).resolve().parent.parent / "data"
LABELS_IN     = DATA_DIR / "user_labels.parquet"
TIMELINES_IN  = DATA_DIR / "user_timelines.parquet"
FEATURES_OUT  = DATA_DIR / "features_bonus.parquet"

BONUS_FEATURES = [
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "exclaim_rate",
    "question_rate",
    "ellipsis_rate",
    "caps_word_rate",
    "i_vs_we_ratio",
    "i_vs_you_ratio",
]

WINDOWS = [
    ("baseline", None),
    ("pre_4w",   4.0),
    ("pre_2w",   2.0),
    ("pre_1w",   1.0),
]
DELTA_WINDOWS = ["pre_4w", "pre_2w", "pre_1w"]


# ── Primitive helpers ─────────────────────────────────────────────────────

_WORD_RE   = re.compile(r"\b[A-Za-z']+\b")          # case-preserving
_LOWER_RE  = re.compile(r"\b[a-z']+\b")             # lower-only
_ELLIPSIS_RE = re.compile(r"\.{3,}")


def _safe_flesch_ease(text: str) -> float:
    try:
        if len(text.split()) < 3:
            return float("nan")
        return float(textstat.flesch_reading_ease(text))
    except Exception:
        return float("nan")


def _safe_flesch_grade(text: str) -> float:
    try:
        if len(text.split()) < 3:
            return float("nan")
        return float(textstat.flesch_kincaid_grade(text))
    except Exception:
        return float("nan")


def _punctuation_rates(text: str, n_tokens: int) -> tuple[float, float, float]:
    if n_tokens == 0:
        return float("nan"), float("nan"), float("nan")
    excl  = text.count("!") / n_tokens
    ques  = text.count("?") / n_tokens
    ellip = len(_ELLIPSIS_RE.findall(text)) / n_tokens
    return excl, ques, ellip


def _caps_word_rate(text: str) -> float:
    words = _WORD_RE.findall(text)
    if not words:
        return float("nan")
    caps = sum(1 for w in words if len(w) >= 2 and w.isupper())
    return caps / len(words)


def _pronoun_ratios(text: str) -> tuple[float, float]:
    """Return log-ratios I/we and I/you (+1 smoothed)."""
    toks = _LOWER_RE.findall(text.lower())
    if not toks:
        return float("nan"), float("nan")
    n_i   = sum(1 for t in toks if t in {"i", "me", "my", "mine", "myself"})
    n_we  = sum(1 for t in toks if t in {"we", "us", "our", "ours", "ourselves"})
    n_you = sum(1 for t in toks if t in {"you", "your", "yours", "yourself", "yourselves"})
    # log((I+1) / (we+1)) — positive = more self-focused than collective
    iw_ratio  = float(np.log((n_i + 1.0) / (n_we  + 1.0)))
    iy_ratio  = float(np.log((n_i + 1.0) / (n_you + 1.0)))
    return iw_ratio, iy_ratio


# ── Window feature extractor ──────────────────────────────────────────────

def extract_bonus_window_features(posts: pd.DataFrame) -> dict[str, float]:
    """Compute 8 bonus features over a set of posts within one time window."""
    if posts.empty:
        return {k: float("nan") for k in BONUS_FEATURES}

    texts = (posts["body"].fillna("") + " " + posts["title"].fillna("")).tolist()
    combined = " ".join(texts).strip()
    if not combined:
        return {k: float("nan") for k in BONUS_FEATURES}

    # Tokens (lowercase) for rate denominators
    all_tokens = _LOWER_RE.findall(combined.lower())
    n_tok      = len(all_tokens)

    excl, ques, ellip = _punctuation_rates(combined, n_tok)
    iw, iy            = _pronoun_ratios(combined)

    return {
        "flesch_reading_ease":  _safe_flesch_ease(combined),
        "flesch_kincaid_grade": _safe_flesch_grade(combined),
        "exclaim_rate":         excl,
        "question_rate":        ques,
        "ellipsis_rate":        ellip,
        "caps_word_rate":       _caps_word_rate(combined),
        "i_vs_we_ratio":        iw,
        "i_vs_you_ratio":       iy,
    }


# ── Window slicing (same as extract_features) ─────────────────────────────

def _window_posts(user_posts: pd.DataFrame,
                  tp_date: pd.Timestamp,
                  window: str) -> pd.DataFrame:
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
        raise ValueError(window)
    return user_posts.loc[mask]


def build_bonus_row(author: str,
                    user_posts: pd.DataFrame,
                    label_row: pd.Series) -> dict:
    row: dict = {
        "author":         author,
        "label":          label_row["label"],
        "low_confidence": label_row.get("low_confidence", False),
        "n_posts":        len(user_posts),
        "tp_date":        label_row["tp_date"],
    }
    tp_date = label_row["tp_date"]
    if pd.isnull(tp_date):
        tp_date = user_posts["created_utc"].max()

    for win_name, _ in WINDOWS:
        win_posts = _window_posts(user_posts, tp_date, win_name)
        feats     = extract_bonus_window_features(win_posts)
        for k, v in feats.items():
            row[f"{k}_{win_name}"] = v

    for win in DELTA_WINDOWS:
        for feat in BONUS_FEATURES:
            b = row.get(f"{feat}_baseline", float("nan"))
            w = row.get(f"{feat}_{win}",    float("nan"))
            row[f"{feat}_delta_{win}"] = w - b
    return row


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("[extract_bonus] loading timelines and labels...")
    labels    = pd.read_parquet(LABELS_IN)
    timelines = pd.read_parquet(TIMELINES_IN)
    if "low_confidence" not in labels.columns:
        labels["low_confidence"] = False

    labels_idx = labels.set_index("author")
    grouped    = timelines.groupby("author", sort=False)
    total      = len(grouped)
    print(f"[extract_bonus] extracting 8 bonus features for {total} authors...")

    rows: list[dict] = []
    for i, (author, posts) in enumerate(grouped, 1):
        if i % 100 == 0:
            print(f"[extract_bonus]   {i}/{total}", flush=True)
        if author not in labels_idx.index:
            continue
        lbl = labels_idx.loc[author]
        rows.append(build_bonus_row(author, posts, lbl))

    df = pd.DataFrame(rows)
    df.to_parquet(FEATURES_OUT, index=False)
    print(f"[extract_bonus] saved {len(df)} rows to {FEATURES_OUT}")

    # Quick sanity summary (mean per label for the pre_1w delta)
    print("\n===== Bonus-feature pre_1w deltas (mean by label) =====")
    for feat in BONUS_FEATURES:
        col = f"{feat}_delta_pre_1w"
        if col not in df.columns:
            continue
        print(f"  {feat}")
        for lbl in ["crisis", "recovery", "neither"]:
            sub = df[df["label"] == lbl][col]
            if sub.empty:
                continue
            print(f"    {lbl:<10} mean={sub.mean():+.4f}  std={sub.std():.4f}  n={sub.notna().sum()}")


if __name__ == "__main__":
    main()
