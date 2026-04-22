// Thin wrappers over Reddit's public .json endpoints.
// Auth-free; paginates via the Listing `after` cursor.

import { politeFetch } from "./rate_limit.js";
import { normalize } from "./schema.js";

const BASE = "https://www.reddit.com";
// Chrome injects its own browser UA; Reddit's public JSON works fine with that.
// We still set Accept so caches behave well.
const HEADERS = { Accept: "application/json" };

function listingUrl(path, { limit = 100, after = null, t = null } = {}) {
  const params = new URLSearchParams({ limit: String(limit), raw_json: "1" });
  if (after) params.set("after", after);
  if (t) params.set("t", t);
  return `${BASE}${path}?${params.toString()}`;
}

// Walk a subreddit listing (new|hot|top), yielding normalized records.
export async function* iterateSubreddit({
  subreddit,
  listing = "new",
  pageCap = 2,
  timeWindow = null, // only meaningful for listing === "top"
  limiter,
}) {
  let after = null;
  const path = `/r/${encodeURIComponent(subreddit)}/${listing}.json`;
  for (let page = 0; page < pageCap; page++) {
    const url = listingUrl(path, { after, t: timeWindow });
    let data;
    try {
      data = await politeFetch(url, { limiter, headers: HEADERS });
    } catch (err) {
      console.warn(`[reddit] listing ${subreddit}/${listing} page ${page} failed:`, err);
      return;
    }
    const children = data?.data?.children || [];
    for (const child of children) {
      const rec = normalize(child);
      if (rec) yield rec;
    }
    after = data?.data?.after;
    if (!after) return;
  }
}

// Walk a user's submitted + comments history, yielding normalized records.
export async function* iterateUserHistory({
  username,
  pageCap = 2,
  limiter,
}) {
  for (const endpoint of ["submitted", "comments"]) {
    let after = null;
    const path = `/user/${encodeURIComponent(username)}/${endpoint}.json`;
    for (let page = 0; page < pageCap; page++) {
      const url = listingUrl(path, { after });
      let data;
      try {
        data = await politeFetch(url, { limiter, headers: HEADERS });
      } catch (err) {
        console.warn(`[reddit] user ${username} ${endpoint} page ${page} failed:`, err);
        break;
      }
      const children = data?.data?.children || [];
      for (const child of children) {
        const rec = normalize(child);
        if (rec) yield rec;
      }
      after = data?.data?.after;
      if (!after) break;
    }
  }
}
