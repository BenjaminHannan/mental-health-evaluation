"""Collect Tumblr post histories for mental health research.

Pipeline
--------
1. Search mental health tags to discover usernames + auto-assign labels
2. Collect each user's full text-post history across all tags
3. Save into the same schema as user_timelines.parquet + user_labels.parquet

Label assignment (tag-based)
-----------------------------
    crisis   : posted with #suicidewatch / #suicidal / #selfharm / #sh
    recovery : posted with #depression / #anxiety / #mentalhealth / #bipolar
    neither  : posted with neutral tags (#photography #books #art etc.)

Usage
-----
    # First: get a free Tumblr API key at https://www.tumblr.com/oauth/apps
    # Then set it as an env variable OR pass via --api-key:

    python src/collect_tumblr.py --api-key YOUR_KEY

    # Collect more users per tag (default 200):
    python src/collect_tumblr.py --api-key YOUR_KEY --users-per-tag 500

    # Append to existing timelines:
    python src/collect_tumblr.py --api-key YOUR_KEY --append

    # Skip timeline collection, only discover usernames:
    python src/collect_tumblr.py --api-key YOUR_KEY --discover-only

Output
------
    data/user_timelines_tumblr.parquet
    data/user_labels_tumblr.parquet

Merge with Reddit data
----------------------
    python src/merge_sources.py   (merges Reddit + Tumblr parquets)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Force line-buffered output so progress is visible when running in background
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import requests

DATA_DIR       = Path(__file__).resolve().parent.parent / "data"
TIMELINES_OUT  = DATA_DIR / "user_timelines_tumblr.parquet"
LABELS_OUT     = DATA_DIR / "user_labels_tumblr.parquet"

BASE_URL   = "https://api.tumblr.com/v2"
SLEEP_SEC  = 0.4   # Tumblr rate limit: ~1000 req/hr
PAGE_SIZE  = 20    # max for /tagged; 50 for blog posts

# ── Label tag groups ──────────────────────────────────────────────────────

CRISIS_TAGS = [
    # Explicit crisis / active ideation only
    "suicidewatch", "suicidal", "suicidal thoughts",
    "want to die", "active self harm",
]
RECOVERY_TAGS = [
    # Explicit recovery / stability framing only
    "mentalillnessrecovery", "mental health recovery",
    "depression recovery", "anxiety recovery",
    "selfharm recovery", "self harm recovery",
    "healing journey", "in recovery",
]
CONTROL_TAGS = [
    # Clearly off-topic
    "photography", "books", "art", "travel", "cooking",
    "gaming", "movies", "music", "nature",
]

TAG_TO_LABEL: dict[str, str] = (
    {t: "crisis"   for t in CRISIS_TAGS}
    | {t: "recovery" for t in RECOVERY_TAGS}
    | {t: "neither"  for t in CONTROL_TAGS}
)

# Priority order if a user appears in multiple tag groups
LABEL_PRIORITY = {"crisis": 0, "recovery": 1, "neither": 2}

REQUIRED_COLS = [
    "author", "body", "created_utc", "id",
    "num_comments", "score", "subreddit",
    "title", "upvote_ratio", "url", "days_to_tp",
]


# ── API helpers ────────────────────────────────────────────────────────────

def _get(endpoint: str, api_key: str, params: dict | None = None,
         retries: int = 3) -> dict | None:
    """GET wrapper with retries."""
    url = f"{BASE_URL}{endpoint}"
    p   = {"api_key": api_key, **(params or {})}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=p, timeout=20)
            if r.status_code == 429:
                print("  [tumblr] rate-limited, sleeping 1 hour...")
                time.sleep(3600)
                continue
            if r.status_code != 200:
                return None
            return r.json().get("response")
        except requests.RequestException as e:
            if attempt == retries - 1:
                print(f"  [tumblr] request error: {e}")
                return None
            time.sleep(2)
    return None


# ── Step 1: discover usernames via tag search ─────────────────────────────

def discover_users_for_tag(tag: str, api_key: str,
                            limit: int = 200) -> list[str]:
    """Return up to `limit` unique blog names that posted with this tag."""
    authors: set[str] = set()
    before: int | None = None

    while len(authors) < limit:
        params: dict = {"tag": tag, "limit": PAGE_SIZE, "filter": "text"}
        if before:
            params["before"] = before

        resp = _get("/tagged", api_key, params)
        if not resp:
            break

        posts = resp if isinstance(resp, list) else []
        if not posts:
            break

        for post in posts:
            name = post.get("blog_name", "").strip()
            if name:
                authors.add(name)

        before = posts[-1].get("timestamp")
        if len(posts) < PAGE_SIZE:
            break
        time.sleep(SLEEP_SEC)

    return list(authors)[:limit]


def discover_all_users(api_key: str,
                       users_per_tag: int = 200) -> dict[str, str]:
    """Return {username: label} dict discovered across all tag groups."""
    all_users: dict[str, str] = {}

    all_tags = (
        [(t, "crisis")   for t in CRISIS_TAGS]
        + [(t, "recovery") for t in RECOVERY_TAGS]
        + [(t, "neither")  for t in CONTROL_TAGS]
    )

    for tag, label in all_tags:
        print(f"[tumblr] discovering users for tag: #{tag} ({label})")
        names = discover_users_for_tag(tag, api_key, limit=users_per_tag)
        print(f"  found {len(names)} users")
        for name in names:
            # Keep higher-priority label if user appears in multiple tags
            if name not in all_users or (
                LABEL_PRIORITY[label] < LABEL_PRIORITY[all_users[name]]
            ):
                all_users[name] = label
        time.sleep(SLEEP_SEC)

    return all_users


# ── Step 2: collect a user's full post history ─────────────────────────────

def _post_to_row(post: dict, author: str) -> dict | None:
    """Convert one Tumblr post dict to the pipeline schema. Returns None to skip."""
    ptype = post.get("type", "")

    # Only collect text-bearing post types
    if ptype == "text":
        body  = post.get("body", "") or ""
        title = post.get("title", "") or ""
    elif ptype == "quote":
        body  = (post.get("text", "") or "") + " " + (post.get("source", "") or "")
        title = ""
    elif ptype == "answer":
        body  = post.get("answer", "") or ""
        title = post.get("question", "") or ""
    elif ptype == "chat":
        lines = post.get("dialogue", []) or []
        body  = " ".join(l.get("phrase", "") for l in lines)
        title = post.get("title", "") or ""
    else:
        return None   # skip photo/video/audio/link

    body  = body.strip()
    title = title.strip()
    if not body and not title:
        return None

    ts = post.get("timestamp", 0)

    return {
        "author":       author,
        "body":         body,
        "created_utc":  pd.to_datetime(ts, unit="s", utc=True),
        "id":           str(post.get("id", "")),
        "num_comments": int(post.get("reply_count", 0) or 0),
        "score":        int(post.get("note_count", 0) or 0),
        "subreddit":    "tumblr:" + ",".join(post.get("tags", [])[:3]),
        "title":        title,
        "upvote_ratio": float("nan"),
        "url":          post.get("post_url", "") or "",
        "days_to_tp":   float("nan"),
    }


def fetch_user_posts(blog_name: str, api_key: str,
                     max_posts: int | None = None,
                     verbose: bool = False) -> list[dict]:
    """Fetch all text posts from a Tumblr blog."""
    rows: list[dict] = []
    offset = 0

    while True:
        params = {
            "limit":  50,
            "offset": offset,
            "filter": "text",
        }
        resp = _get(f"/blog/{blog_name}.tumblr.com/posts", api_key, params)
        if not resp:
            break

        posts = resp.get("posts", []) if isinstance(resp, dict) else []
        if not posts:
            break

        for post in posts:
            row = _post_to_row(post, blog_name)
            if row:
                rows.append(row)
                if max_posts and len(rows) >= max_posts:
                    break

        if verbose:
            print(f"    fetched {len(rows)} posts so far (offset {offset})")

        if max_posts and len(rows) >= max_posts:
            break

        total = resp.get("total_posts", 0) if isinstance(resp, dict) else 0
        offset += len(posts)
        if offset >= total or len(posts) < 50:
            break

        time.sleep(SLEEP_SEC)

    return rows


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-key", default=os.environ.get("TUMBLR_API_KEY"),
                   help="Tumblr consumer/API key. Or set env var TUMBLR_API_KEY.")
    p.add_argument("--users-per-tag", type=int, default=200,
                   help="Max users to discover per tag (default 200)")
    p.add_argument("--max-posts", type=int, default=None,
                   help="Max posts per user (omit for no limit)")
    p.add_argument("--append", action="store_true",
                   help="Append to existing tumblr parquets")
    p.add_argument("--discover-only", action="store_true",
                   help="Only discover usernames, skip timeline collection")
    args = p.parse_args()

    if not args.api_key:
        print(
            "ERROR: No API key provided.\n"
            "  Get a free key at: https://www.tumblr.com/oauth/apps\n"
            "  Then run:  python src/collect_tumblr.py --api-key YOUR_KEY\n"
            "  Or set:    $env:TUMBLR_API_KEY = 'YOUR_KEY'  (PowerShell)"
        )
        return

    # ── Discover users ────────────────────────────────────────────────────
    print(f"[tumblr] discovering users ({args.users_per_tag} per tag)...")
    user_labels = discover_all_users(args.api_key, args.users_per_tag)
    print(f"[tumblr] total unique users discovered: {len(user_labels)}")

    from collections import Counter
    counts = Counter(user_labels.values())
    print(f"  crisis={counts['crisis']}  recovery={counts['recovery']}  "
          f"neither={counts['neither']}")

    if args.discover_only:
        # Save just the labels file
        label_df = pd.DataFrame([
            {"author": u, "label": l, "low_confidence": False, "tp_date": pd.NaT}
            for u, l in user_labels.items()
        ])
        label_df.to_parquet(LABELS_OUT, index=False)
        print(f"[tumblr] labels saved to {LABELS_OUT}")
        return

    # ── Collect timelines ─────────────────────────────────────────────────
    total      = len(user_labels)
    all_rows:  list[dict] = []
    label_rows: list[dict] = []

    for i, (username, label) in enumerate(user_labels.items(), 1):
        print(f"[tumblr] ({i}/{total}) {username}  [{label}]")
        posts = fetch_user_posts(username, args.api_key,
                                 max_posts=args.max_posts, verbose=False)
        print(f"  -> {len(posts)} posts")

        if not posts:
            continue

        all_rows.extend(posts)
        label_rows.append({
            "author":         username,
            "label":          label,
            "low_confidence": False,
            "tp_date":        pd.NaT,   # set manually if known
        })
        time.sleep(SLEEP_SEC)

    if not all_rows:
        print("[tumblr] No posts collected.")
        return

    # ── Save ──────────────────────────────────────────────────────────────
    tl_df  = pd.DataFrame(all_rows,  columns=REQUIRED_COLS)
    lbl_df = pd.DataFrame(label_rows)
    tl_df["created_utc"] = pd.to_datetime(tl_df["created_utc"], utc=True)

    if args.append and TIMELINES_OUT.exists():
        existing_tl  = pd.read_parquet(TIMELINES_OUT)
        existing_lbl = pd.read_parquet(LABELS_OUT) if LABELS_OUT.exists() else pd.DataFrame()
        tl_df  = pd.concat([existing_tl,  tl_df],  ignore_index=True).drop_duplicates("id")
        lbl_df = pd.concat([existing_lbl, lbl_df], ignore_index=True).drop_duplicates("author")

    tl_df.to_parquet(TIMELINES_OUT,  index=False)
    lbl_df.to_parquet(LABELS_OUT,    index=False)

    print(f"\n[tumblr] saved {len(tl_df)} timeline rows  -> {TIMELINES_OUT}")
    print(f"[tumblr] saved {len(lbl_df)} label rows     -> {LABELS_OUT}")
    print(f"\n  Label breakdown:")
    print(lbl_df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
