# Mental Health Reddit Scraper (Chrome Extension)

Chrome extension that collects Reddit posts + comments from mental-health
subreddits and writes them as JSON into `~/Downloads/mental-health-scraper/`.
The output schema is drop-in compatible with the pipeline in
`../reddit-mental-health/`.

## What it does

1. **Discover users** — walks the `new` / `hot` / `top` listings of the
   selected subreddits (default: `depression, ADHD, PTSD, OCD, aspergers`).
2. **Harvest history** — for every user discovered, fetches their recent
   submissions and comments. This produces the time-series needed by
   `reddit-mental-health/src/label_users.py` for baseline-window analysis.
3. **Save** — writes three files under `~/Downloads/<outputFolder>/`:
   - `posts.json` — every post/comment as a canonical record
   - `users.json` — per-user aggregate (post count, date range, subs)
   - `run_log.json` — metadata from each run

Records are deduplicated across runs, so re-running merges rather than
duplicates.

## Canonical record schema

Matches `reddit-mental-health/src/collect_data.py::_item_to_row` exactly:

| field | type | notes |
|---|---|---|
| `author` | string | deleted/removed authors are dropped |
| `body` | string | `selftext` for posts, `body` for comments |
| `created_utc` | string (ISO-8601, UTC) | |
| `id` | string | used for dedupe |
| `num_comments` | int | 0 for comments |
| `score` | int | |
| `subreddit` | string (lowercase) | |
| `title` | string | empty for comments |
| `upvote_ratio` | float or null | posts only |
| `url` | string | permalink |
| `days_to_tp` | null | filled in later by the pipeline |

## Install (unpacked)

1. Open `chrome://extensions`.
2. Toggle **Developer mode** on.
3. Click **Load unpacked** and select this `scraper-extension/` directory.
4. Pin the extension to the toolbar.

## Use

1. Click the toolbar icon.
2. Adjust subreddits (chips) and listing/page caps. Start small:
   `listingPageCap = 1`, `userHistoryPageCap = 1` for a smoke test.
3. Click **Run scrape**. The popup shows live progress (phase, current
   subreddit / user, totals).
4. When finished, check `~/Downloads/mental-health-scraper/posts.json`.
5. **Export JSON** re-emits the current dataset without re-scraping.
6. **Clear dataset** wipes the extension's internal store (does not
   delete already-downloaded files).

Defaults (output folder, rate limit, `MIN_POSTS` filter for "qualified"
users) are editable on the options page.

## Feeding the pipeline

The pipeline expects Parquet with the same column names. Convert:

```bash
python -c "
import json, pandas as pd, pathlib
src = pathlib.Path.home() / 'Downloads/mental-health-scraper/posts.json'
df = pd.DataFrame(json.loads(src.read_text()))
df['created_utc'] = pd.to_datetime(df['created_utc'], utc=True)
df.to_parquet('reddit-mental-health/data/user_timelines.parquet', index=False)
print(df.shape, df.columns.tolist())
"
```

Then run the normal pipeline (from `reddit-mental-health/`):

```bash
python src/label_users.py
python src/extract_features.py
python src/train_model.py
```

## Politeness / terms of use

- Only public `.json` endpoints are used. No login, no modmail, no
  private data.
- Default rate limit: 1.1 s between requests (well under Reddit's
  60 req/min public quota). Exponential backoff on 429 / 503.
- This tool is intended for **research use** of publicly available
  content. Respect Reddit's Terms of Service and don't redistribute
  raw per-user data without the appropriate ethics review.

## Architecture

```
manifest.json          MV3; host perms for *.reddit.com
background.js          service worker: two-phase scrape, dedupe, downloads
popup.{html,css,js}    toolbar UI
options.{html,js}      defaults / rate-limit / output folder
lib/
  schema.js            normalize Reddit Listing children → canonical record
  reddit.js            paginated iterators over subreddit + user endpoints
  rate_limit.js        RateLimiter + politeFetch (backoff on 429/503)
  dedupe.js            chrome.storage.local persistence + id dedupe
```

Persistence uses `chrome.storage.local` with the `unlimitedStorage`
permission. The full dataset is re-emitted on every save so
`posts.json` is always complete, never a fragment.
