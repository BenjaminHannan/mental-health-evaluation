"""Collect Reddit post histories via the Arctic Shift API.

Takes a list of Reddit usernames and fetches all available posts,
saving them into the same schema used by user_timelines.parquet.

Usage
-----
    # Collect for a list of usernames in a text file (one per line):
    python src/collect_data.py --users data/usernames.txt

    # Or pass usernames directly:
    python src/collect_data.py --users user1 user2 user3

    # Append to existing timelines instead of overwriting:
    python src/collect_data.py --users data/usernames.txt --append

    # Limit posts per user (useful for testing):
    python src/collect_data.py --users data/usernames.txt --max-posts 500

Output
------
    data/user_timelines.parquet  (created or appended to)

Arctic Shift API docs: https://arctic-shift.photon-reddit.com/api
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR     = Path(__file__).resolve().parent.parent / "data"
TIMELINES_OUT = DATA_DIR / "user_timelines.parquet"

BASE_URL  = "https://arctic-shift.photon-reddit.com/api"
PAGE_SIZE = 100   # max per request
SLEEP_SEC = 0.5   # be polite; arctic shift has no hard rate limit but don't hammer it


# ── Schema helpers ────────────────────────────────────────────────────────

# These are the columns the rest of the pipeline expects
REQUIRED_COLS = [
    "author", "body", "created_utc", "id",
    "num_comments", "score", "subreddit",
    "title", "upvote_ratio", "url", "days_to_tp",
]


def _item_to_row(item: dict) -> dict:
    """Normalise one Arctic Shift post/comment dict to the pipeline schema."""
    # Arctic Shift returns Unix timestamps as integers
    created = pd.to_datetime(item.get("created_utc", 0), unit="s", utc=True)

    # Submissions have 'selftext'; comments have 'body'
    body = item.get("selftext") or item.get("body") or ""
    if body in ("[deleted]", "[removed]"):
        body = ""

    title = item.get("title", "")   # empty for comments

    return {
        "author":       item.get("author", ""),
        "body":         body,
        "created_utc":  created,
        "id":           item.get("id", ""),
        "num_comments": int(item.get("num_comments", 0) or 0),
        "score":        int(item.get("score", 0) or 0),
        "subreddit":    item.get("subreddit", ""),
        "title":        title,
        "upvote_ratio": float(item.get("upvote_ratio", float("nan")) or float("nan")),
        "url":          item.get("url") or item.get("permalink") or "",
        "days_to_tp":   float("nan"),   # filled in later via label assignment
    }


# ── Arctic Shift fetcher ──────────────────────────────────────────────────

def fetch_user_posts(username: str,
                     max_posts: int | None = None,
                     verbose: bool = True) -> list[dict]:
    """Fetch all submissions + comments for a user via Arctic Shift."""
    rows: list[dict] = []

    for kind in ("posts", "comments"):
        endpoint = f"{BASE_URL}/{kind}/search"
        after    = None
        fetched  = 0

        while True:
            params: dict = {
                "author": username,
                "limit":  PAGE_SIZE,
                "sort":   "asc",
            }
            if after:
                params["after"] = after

            try:
                resp = requests.get(endpoint, params=params, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as e:
                print(f"  [collect] WARNING: {kind} request failed for {username}: {e}")
                break

            data  = resp.json()
            items = data.get("data", [])
            if not items:
                break

            for item in items:
                rows.append(_item_to_row(item))
                fetched += 1
                if max_posts and fetched >= max_posts:
                    break

            if verbose:
                print(f"  [collect]   {kind}: {fetched} posts fetched so far", flush=True)

            if max_posts and fetched >= max_posts:
                break

            # Arctic Shift pagination: use the last item's created_utc as cursor
            after = items[-1].get("created_utc")
            if len(items) < PAGE_SIZE:
                break   # last page

            time.sleep(SLEEP_SEC)

    return rows


# ── Main ─────────────────────────────────────────────────────────────────

def load_usernames(arg: list[str]) -> list[str]:
    """Parse --users: either a .txt file path or a list of names."""
    if len(arg) == 1 and Path(arg[0]).exists():
        lines = Path(arg[0]).read_text().splitlines()
        return [l.strip() for l in lines if l.strip() and not l.startswith("#")]
    return arg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--users", nargs="+", required=True,
                   help="Usernames or path to a .txt file with one username per line")
    p.add_argument("--append", action="store_true",
                   help="Append to existing user_timelines.parquet instead of overwriting")
    p.add_argument("--max-posts", type=int, default=None,
                   help="Max posts per user (per type). Omit for no limit.")
    p.add_argument("--out", type=Path, default=TIMELINES_OUT,
                   help="Output parquet path")
    args = p.parse_args()

    usernames = load_usernames(args.users)
    print(f"[collect] {len(usernames)} user(s) to collect")

    all_rows: list[dict] = []
    for i, user in enumerate(usernames, 1):
        print(f"[collect] ({i}/{len(usernames)}) fetching: {user}")
        rows = fetch_user_posts(user, max_posts=args.max_posts)
        print(f"[collect]   -> {len(rows)} total posts")
        all_rows.extend(rows)
        time.sleep(SLEEP_SEC)

    if not all_rows:
        print("[collect] No posts collected. Exiting.")
        return

    new_df = pd.DataFrame(all_rows, columns=REQUIRED_COLS)
    # Ensure correct dtypes
    new_df["created_utc"] = pd.to_datetime(new_df["created_utc"], utc=True)
    new_df["days_to_tp"]  = new_df["days_to_tp"].astype(float)

    if args.append and args.out.exists():
        existing = pd.read_parquet(args.out)
        combined = pd.concat([existing, new_df], ignore_index=True)
        # Drop duplicates by post id
        combined = combined.drop_duplicates(subset=["id"])
        combined.to_parquet(args.out, index=False)
        print(f"[collect] Appended -> {args.out}  "
              f"(was {len(existing)}, now {len(combined)} rows)")
    else:
        new_df.to_parquet(args.out, index=False)
        print(f"[collect] Saved {len(new_df)} rows -> {args.out}")

    # Quick summary
    by_user = new_df.groupby("author").size().sort_values(ascending=False)
    print("\n  Posts per user (top 10):")
    print(by_user.head(10).to_string())


if __name__ == "__main__":
    main()
