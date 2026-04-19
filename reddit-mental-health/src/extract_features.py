"""Extract sliding-window linguistic features for each user's timeline.

Stage 3 of the pipeline. Reads ``data/user_timelines.parquet`` and
``data/user_labels.parquet`` and computes 7 features across 4 time
windows relative to each user's turning point (TP):

    baseline : all posts before (TP - 4 weeks)   — long-run behaviour
    pre_4w   : posts in the 4-week window before TP
    pre_2w   : posts in the 2-week window before TP
    pre_1w   : posts in the 1-week window before TP

Features
--------
1. sentiment_mean    : mean VADER compound score  (-1 worst → +1 best)
2. ttr              : type-token ratio (unique tokens / total tokens)
3. avg_sent_len     : mean sentence length in words
4. post_freq        : posts per week within the window
5. fp_pronoun_rate  : first-person singular pronoun rate (I/me/my/mine/myself)
6. neg_affect_rate  : negative-affect word rate (LIWC-style wordlist)
7. avg_post_len     : mean post length in words

The output is a wide-format feature matrix — one row per user — with
columns named ``<feature>_<window>`` plus label and low_confidence flag.
Delta columns (``<feature>_delta_<window>`` = window − baseline) are
appended for each window and are the primary inputs for the classifier.

Also updates ``data/user_labels.parquet`` to add the ``low_confidence``
flag for users whose turning-point was matched by the ambiguous phrase
"made it".

Run:
    python src/extract_features.py
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import nltk
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DATA_DIR       = Path(__file__).resolve().parent.parent / "data"
LABELS_IN      = DATA_DIR / "user_labels.parquet"
TIMELINES_IN   = DATA_DIR / "user_timelines.parquet"
FEATURES_OUT   = DATA_DIR / "features.parquet"

# Window definitions: (label, weeks_before_tp)
# "baseline" is open-ended (all posts older than WINDOW_WEEKS before TP)
WINDOW_WEEKS = 4

# ── NLP resources ─────────────────────────────────────────────────────────

VADER = SentimentIntensityAnalyzer()

FP_PRONOUNS: frozenset[str] = frozenset({
    "i", "me", "my", "mine", "myself",
})

# Compact LIWC-style negative-affect wordlist (Pennebaker et al. categories:
# negative emotion, anger, anxiety, sadness). Kept to ~120 unambiguous items.
NEG_AFFECT_WORDS: frozenset[str] = frozenset({
    # sadness
    "sad", "sadness", "unhappy", "miserable", "depressed", "depression",
    "hopeless", "hopelessness", "despair", "despairing", "grief", "grieve",
    "grieving", "cry", "crying", "tears", "weep", "weeping", "heartbroken",
    "devastated", "devastation", "empty", "numb", "alone", "lonely",
    "loneliness", "isolated", "isolation", "worthless", "worthlessness",
    # anxiety
    "anxious", "anxiety", "worried", "worry", "worrying", "fear", "scared",
    "terrified", "terror", "panic", "panicking", "dread", "dreading",
    "nervous", "overwhelmed", "overwhelming", "stress", "stressed",
    "stressful",
    # anger
    "angry", "anger", "furious", "fury", "rage", "raging", "hate",
    "hatred", "hating", "resentment", "resentful", "bitter", "bitterness",
    "frustrated", "frustration",
    # general negative emotion
    "horrible", "terrible", "awful", "dreadful", "disgusting", "disgusted",
    "ashamed", "shame", "guilt", "guilty", "regret", "regretful",
    "broken", "shattered", "suffering", "suffer", "pain", "hurt",
    "hurting", "misery", "agony", "agonizing", "exhausted", "exhaustion",
    "burned out", "burnout", "lost", "confused", "confusion",
    "nauseous", "sick", "dying",
})

_WORD_RE   = re.compile(r"\b[a-z']+\b")
_SENT_SPLIT = re.compile(r"[.!?]+")


# ── Text feature helpers ───────────────────────────────────────────────────

def _tokens(text: str) -> list[str]:
    """Lowercase word tokens from a post body."""
    return _WORD_RE.findall(text.lower())


def _sentiment(text: str) -> float:
    """VADER compound score for a single text."""
    return VADER.polarity_scores(text)["compound"]


def _ttr(tokens: list[str]) -> float:
    """Type-token ratio; NaN if fewer than 2 tokens."""
    if len(tokens) < 2:
        return float("nan")
    return len(set(tokens)) / len(tokens)


def _avg_sentence_length(text: str) -> float:
    """Mean word count per sentence; NaN if no sentences."""
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sentences:
        return float("nan")
    lengths = [len(_WORD_RE.findall(s)) for s in sentences]
    return float(np.mean(lengths))


def _fp_pronoun_rate(tokens: list[str]) -> float:
    """First-person singular pronoun count / total tokens."""
    if not tokens:
        return float("nan")
    fp = sum(1 for t in tokens if t in FP_PRONOUNS)
    return fp / len(tokens)


def _neg_affect_rate(tokens: list[str]) -> float:
    """Negative-affect word count / total tokens."""
    if not tokens:
        return float("nan")
    na = sum(1 for t in tokens if t in NEG_AFFECT_WORDS)
    return na / len(tokens)


def _avg_post_len(tokens_list: list[list[str]]) -> float:
    """Mean post length in words across a list of token lists."""
    lengths = [len(t) for t in tokens_list]
    return float(np.mean(lengths)) if lengths else float("nan")


# ── Window extraction ──────────────────────────────────────────────────────

def _window_posts(
    user_posts: pd.DataFrame,
    tp_date: pd.Timestamp,
    window: str,
) -> pd.DataFrame:
    """Return the subset of posts falling inside the named window.

    baseline : posted before (tp_date - 4 weeks)
    pre_4w   : posted in [tp_date - 4w,  tp_date)
    pre_2w   : posted in [tp_date - 2w,  tp_date)
    pre_1w   : posted in [tp_date - 1w,  tp_date)
    """
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


def extract_window_features(
    posts: pd.DataFrame,
    window_weeks: Optional[float],
) -> dict[str, float]:
    """Compute all 7 features for a set of posts within one time window.

    ``window_weeks`` is used to compute post_freq; pass None for baseline
    (window length is variable — we use the actual date span instead).
    """
    if posts.empty:
        return {
            "sentiment_mean": float("nan"),
            "ttr":            float("nan"),
            "avg_sent_len":   float("nan"),
            "post_freq":      0.0,
            "fp_pronoun_rate": float("nan"),
            "neg_affect_rate": float("nan"),
            "avg_post_len":   float("nan"),
        }

    texts       = (posts["body"].fillna("") + " " + posts["title"].fillna("")).tolist()
    tokens_list = [_tokens(t) for t in texts]
    all_tokens  = [tok for sublist in tokens_list for tok in sublist]
    combined    = " ".join(texts)

    # Post frequency: posts per week
    if window_weeks is not None:
        freq = len(posts) / window_weeks if window_weeks > 0 else float("nan")
    else:
        # Baseline: compute week span from first to last post
        span_days = (posts["created_utc"].max() - posts["created_utc"].min()).days
        span_weeks = max(span_days / 7, 1)
        freq = len(posts) / span_weeks

    return {
        "sentiment_mean":  float(np.mean([_sentiment(t) for t in texts])),
        "ttr":             _ttr(all_tokens),
        "avg_sent_len":    float(np.nanmean([_avg_sentence_length(t) for t in texts])),
        "post_freq":       freq,
        "fp_pronoun_rate": _fp_pronoun_rate(all_tokens),
        "neg_affect_rate": _neg_affect_rate(all_tokens),
        "avg_post_len":    _avg_post_len(tokens_list),
    }


WINDOWS: list[tuple[str, Optional[float]]] = [
    ("baseline", None),
    ("pre_4w",   4.0),
    ("pre_2w",   2.0),
    ("pre_1w",   1.0),
]

FEATURE_NAMES = [
    "sentiment_mean", "ttr", "avg_sent_len",
    "post_freq", "fp_pronoun_rate", "neg_affect_rate", "avg_post_len",
]


# ── Per-user feature extraction ────────────────────────────────────────────

def build_feature_row(
    author: str,
    user_posts: pd.DataFrame,
    label_row: pd.Series,
) -> dict:
    """Build one wide-format feature row for a single user."""
    row: dict = {
        "author":          author,
        "label":           label_row["label"],
        "low_confidence":  label_row["low_confidence"],
        "n_posts":         len(user_posts),
        "tp_date":         label_row["tp_date"],
    }

    tp_date = label_row["tp_date"]
    # For "neither" users use the last post as a pseudo turning point
    if pd.isnull(tp_date):
        tp_date = user_posts["created_utc"].max()

    for win_name, win_weeks in WINDOWS:
        win_posts = _window_posts(user_posts, tp_date, win_name)
        feats     = extract_window_features(win_posts, win_weeks)
        for feat, val in feats.items():
            row[f"{feat}_{win_name}"] = val

    # Delta columns: pre_window − baseline (positive = increase toward TP)
    for win_name in ("pre_4w", "pre_2w", "pre_1w"):
        for feat in FEATURE_NAMES:
            baseline_val = row.get(f"{feat}_baseline", float("nan"))
            window_val   = row.get(f"{feat}_{win_name}", float("nan"))
            row[f"{feat}_delta_{win_name}"] = window_val - baseline_val

    return row


# ── Main ──────────────────────────────────────────────────────────────────

def flag_low_confidence(user_labels: pd.DataFrame) -> pd.DataFrame:
    """Add low_confidence=True for users matched by the ambiguous 'made it' phrase."""
    user_labels = user_labels.copy()
    ambiguous_phrases = {"made it"}
    user_labels["low_confidence"] = (
        user_labels["tp_matched_phrase"]
        .str.lower()
        .isin(ambiguous_phrases)
        .fillna(False)
    )
    n = user_labels["low_confidence"].sum()
    print(f"[extract_features] Flagged {n} users as low_confidence "
          f"(turning point matched 'made it')")
    return user_labels


def print_summary(features: pd.DataFrame) -> None:
    """Print mean feature values per label × window in a readable table."""
    windows    = ["baseline", "pre_4w", "pre_2w", "pre_1w"]
    labels     = ["crisis", "recovery", "neither"]

    print("\n===== Mean feature values by label × window =====\n")
    for feat in FEATURE_NAMES:
        cols = [f"{feat}_{w}" for w in windows]
        print(f"  {feat}")
        header = f"    {'label':<12}" + "".join(f"{w:>12}" for w in windows)
        print(header)
        print("    " + "-" * (12 + 12 * len(windows)))
        for lbl in labels:
            subset = features[features["label"] == lbl][cols]
            means  = subset.mean()
            vals   = "".join(f"{means[c]:>12.4f}" for c in cols)
            print(f"    {lbl:<12}{vals}")
        print()

    # Delta summary (pre_1w delta only, most informative)
    print("  === pre_1w delta (pre_1w - baseline) ===")
    delta_cols = [f"{f}_delta_pre_1w" for f in FEATURE_NAMES]
    header = f"    {'label':<12}" + "".join(f"{f.replace('_delta_pre_1w',''):>18}" for f in delta_cols)
    print(header)
    print("    " + "-" * (12 + 18 * len(delta_cols)))
    for lbl in labels:
        subset = features[features["label"] == lbl][delta_cols]
        means  = subset.mean()
        vals   = "".join(f"{means[c]:>18.4f}" for c in delta_cols)
        print(f"    {lbl:<12}{vals}")
    print()

    n_labelled = len(features[features["label"] != "neither"])
    n_low_conf = len(features[features["low_confidence"] == True])
    print(f"  Total users: {len(features)} "
          f"(labelled: {n_labelled}, low_confidence: {n_low_conf})")


def main() -> None:
    """Run stage 3: extract features and save the feature matrix."""
    print("[extract_features] Loading labels and timelines...")
    user_labels    = pd.read_parquet(LABELS_IN)
    user_timelines = pd.read_parquet(TIMELINES_IN)

    # Add low_confidence flag and save back to user_labels.parquet
    user_labels = flag_low_confidence(user_labels)
    user_labels.to_parquet(LABELS_IN, index=False)
    print(f"[extract_features] Updated {LABELS_IN} with low_confidence flag")

    labels_idx = user_labels.set_index("author")
    grouped    = user_timelines.groupby("author", sort=False)
    total      = len(grouped)

    print(f"[extract_features] Extracting features for {total} authors...")
    rows: list[dict] = []
    for i, (author, user_posts) in enumerate(grouped, 1):
        if i % 100 == 0:
            print(f"[extract_features]   {i}/{total}...", flush=True)
        label_row = labels_idx.loc[author]
        rows.append(build_feature_row(author, user_posts, label_row))

    features = pd.DataFrame(rows)
    features.to_parquet(FEATURES_OUT, index=False)
    print(f"[extract_features] Saved {len(features)} rows to {FEATURES_OUT}")

    print_summary(features)


if __name__ == "__main__":
    main()
