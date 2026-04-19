"""Identify turning-point posts and label user timelines.

Stage 2 of the pipeline. Reads ``data/raw_posts.parquet``, groups posts
by author, and searches for the earliest post whose text matches a
crisis or recovery keyword phrase. That post becomes the user's
"turning point". Users with no matching post are labelled ``neither``.

Inclusion criteria (configurable via constants below):
  - Author must have > MIN_POSTS total posts.
  - Author must have >= 1 post in the baseline window, defined as any
    post that predates (turning_point_date - WINDOW_WEEKS weeks).

Outputs:
  data/user_labels.parquet  — one row per included author with label and
                              turning-point metadata.
  data/user_timelines.parquet — all posts for included authors, augmented
                                with days_to_tp (days until turning point).

Run:
    python src/label_users.py
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IN_PATH  = DATA_DIR / "raw_posts.parquet"
LABELS_OUT    = DATA_DIR / "user_labels.parquet"
TIMELINES_OUT = DATA_DIR / "user_timelines.parquet"

# ── Cohort filters ──────────────────────────────────────────────────────────
MIN_POSTS    = 10   # minimum total posts per author
WINDOW_WEEKS = 4    # how far before the turning-point to look for pre-window posts

# ── Keyword lists ───────────────────────────────────────────────────────────
CRISIS_PHRASES: list[str] = [
    "want to die",
    "wanting to die",
    "ending it",
    "end it all",
    "can't do this",
    "cant do this",
    "goodbye",
    "no point",
    "kill myself",
    "killing myself",
    "take my own life",
    "not worth living",
]

RECOVERY_PHRASES: list[str] = [
    "got help",
    "getting help",
    "starting therapy",
    "started therapy",
    "feeling better",
    "things are improving",
    "made it",
    "on the road to recovery",
    "doing better",
    "finally better",
]

Label = Literal["crisis", "recovery", "neither"]


def _compile_pattern(phrases: list[str]) -> re.Pattern[str]:
    """Build a single case-insensitive regex from a list of phrase strings."""
    escaped = [re.escape(p) for p in phrases]
    return re.compile(r"(?:" + "|".join(escaped) + r")", re.IGNORECASE)


CRISIS_RE   = _compile_pattern(CRISIS_PHRASES)
RECOVERY_RE = _compile_pattern(RECOVERY_PHRASES)


def _text(row: pd.Series) -> str:
    """Concatenate title + body for a post row."""
    return f"{row.get('title', '') or ''} {row.get('body', '') or ''}"


def find_turning_point(
    user_posts: pd.DataFrame,
) -> tuple[Label, pd.Series | None]:
    """Find the earliest crisis or recovery turning-point post for a user.

    Returns the label and the turning-point row, or ("neither", None) if
    no keyword match is found.
    """
    # Sort chronologically so we take the first matching post
    posts = user_posts.sort_values("created_utc").reset_index(drop=True)

    tp_row: pd.Series | None = None
    tp_label: Label = "neither"
    tp_date  = pd.Timestamp.max.tz_localize("UTC")

    for _, row in posts.iterrows():
        text = _text(row)
        is_crisis   = bool(CRISIS_RE.search(text))
        is_recovery = bool(RECOVERY_RE.search(text))
        if not (is_crisis or is_recovery):
            continue
        # If both match in one post, crisis takes precedence
        label: Label = "crisis" if is_crisis else "recovery"
        if row["created_utc"] < tp_date:
            tp_date  = row["created_utc"]
            tp_row   = row
            tp_label = label

    return tp_label, tp_row


def build_user_record(
    author: str,
    user_posts: pd.DataFrame,
    label: Label,
    tp_row: pd.Series | None,
) -> dict:
    """Build a single-row dict for the user_labels DataFrame."""
    base: dict = {
        "author":    author,
        "label":     label,
        "n_posts":   len(user_posts),
        "tp_post_id":    None,
        "tp_date":       pd.NaT,
        "tp_subreddit":  None,
        "tp_matched_phrase": None,
    }
    if tp_row is not None:
        text = _text(tp_row)
        # Record which specific phrase triggered the match
        phrase_match = (CRISIS_RE.search(text) if label == "crisis"
                        else RECOVERY_RE.search(text))
        base.update({
            "tp_post_id":       tp_row.get("id"),
            "tp_date":          tp_row["created_utc"],
            "tp_subreddit":     tp_row.get("subreddit"),
            "tp_matched_phrase": phrase_match.group(0) if phrase_match else None,
        })
    return base


def label_users(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full labelling pass over all authors.

    Parameters
    ----------
    df : cleaned posts DataFrame from load_data.py

    Returns
    -------
    user_labels :  one row per included author with label + turning-point metadata
    user_timelines : all posts for included authors plus ``days_to_tp`` column
    """
    # Step 1: filter to authors with enough posts
    post_counts = df["author"].value_counts()
    active_authors = post_counts[post_counts > MIN_POSTS].index
    df_active = df[df["author"].isin(active_authors)].copy()
    print(f"[label_users] Authors with >{MIN_POSTS} posts: {len(active_authors):,} "
          f"({len(df_active):,} posts)")

    label_rows: list[dict] = []
    timeline_rows: list[pd.DataFrame] = []
    skipped_no_baseline = 0

    grouped = df_active.groupby("author", sort=False)
    total = len(grouped)
    for i, (author, user_posts) in enumerate(grouped, 1):
        if i % 200 == 0:
            print(f"[label_users]   Processing author {i}/{total}...", flush=True)

        label, tp_row = find_turning_point(user_posts)

        # For labelled users, enforce the baseline requirement:
        # ≥1 post more than WINDOW_WEEKS before the turning point
        if label != "neither" and tp_row is not None:
            baseline_cutoff = tp_row["created_utc"] - pd.Timedelta(weeks=WINDOW_WEEKS)
            baseline_posts = user_posts[user_posts["created_utc"] < baseline_cutoff]
            if len(baseline_posts) < 1:
                skipped_no_baseline += 1
                label = "neither"
                tp_row = None

        # Annotate posts with days_to_tp
        user_df = user_posts.copy()
        if tp_row is not None:
            tp_ts = tp_row["created_utc"]
            user_df["days_to_tp"] = (
                (user_df["created_utc"] - tp_ts)
                .dt.total_seconds() / 86400
            )
        else:
            user_df["days_to_tp"] = float("nan")

        label_rows.append(build_user_record(author, user_posts, label, tp_row))
        timeline_rows.append(user_df)

    print(f"[label_users] Skipped {skipped_no_baseline} labelled users "
          f"with no baseline posts before the {WINDOW_WEEKS}-week window")

    user_labels     = pd.DataFrame(label_rows)
    user_timelines  = pd.concat(timeline_rows, ignore_index=True)
    return user_labels, user_timelines


def print_label_breakdown(user_labels: pd.DataFrame) -> None:
    """Print a summary table of crisis / recovery / neither counts."""
    counts = user_labels["label"].value_counts()
    total  = len(user_labels)
    print("\n===== User label breakdown =====")
    for lbl in ["crisis", "recovery", "neither"]:
        n = counts.get(lbl, 0)
        print(f"  {lbl:<12}: {n:>5,}  ({100 * n / total:.1f}%)")
    print(f"  {'TOTAL':<12}: {total:>5,}")

    labelled = user_labels[user_labels["label"] != "neither"]
    if not labelled.empty:
        print("\nTop matched phrases (labelled users):")
        print(labelled["tp_matched_phrase"].value_counts().head(15).to_string())

        print("\nSubreddit breakdown (labelled users):")
        sub_lbl = (labelled.groupby(["tp_subreddit", "label"])
                   .size().unstack(fill_value=0))
        print(sub_lbl.to_string())

        print("\nPost count distribution for labelled users:")
        print(labelled["n_posts"].describe().round(1).to_string())


def main() -> None:
    """Run stage 2: label users and save outputs."""
    print(f"[label_users] Loading {IN_PATH}...")
    df = pd.read_parquet(IN_PATH)
    print(f"[label_users] {len(df):,} posts, {df['author'].nunique():,} unique authors")

    user_labels, user_timelines = label_users(df)
    print_label_breakdown(user_labels)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    user_labels.to_parquet(LABELS_OUT, index=False)
    user_timelines.to_parquet(TIMELINES_OUT, index=False)
    print(f"\n[label_users] Saved {LABELS_OUT}")
    print(f"[label_users] Saved {TIMELINES_OUT}")


if __name__ == "__main__":
    main()
