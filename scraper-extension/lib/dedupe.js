// Persistent dedupe + user-stat aggregation via chrome.storage.local.
// The extension declares "unlimitedStorage" so chrome.storage.local is unquotaed.

const POSTS_KEY = "posts";
const SEEN_KEY = "seen_ids";
const USER_STATS_KEY = "user_stats";

export async function loadState() {
  const res = await chrome.storage.local.get([
    POSTS_KEY,
    SEEN_KEY,
    USER_STATS_KEY,
  ]);
  return {
    posts: res[POSTS_KEY] || [],
    seen: new Set(res[SEEN_KEY] || []),
    userStats: res[USER_STATS_KEY] || {},
  };
}

export async function saveState(state) {
  await chrome.storage.local.set({
    [POSTS_KEY]: state.posts,
    [SEEN_KEY]: Array.from(state.seen),
    [USER_STATS_KEY]: state.userStats,
  });
}

export async function clearState() {
  await chrome.storage.local.remove([POSTS_KEY, SEEN_KEY, USER_STATS_KEY]);
}

// Merge new records into state, deduped by id. Returns the count actually added.
export function mergeRecords(state, records) {
  let added = 0;
  for (const rec of records) {
    if (!rec || !rec.id) continue;
    if (state.seen.has(rec.id)) continue;
    state.seen.add(rec.id);
    state.posts.push(rec);
    added++;

    const existing = state.userStats[rec.author];
    if (!existing) {
      state.userStats[rec.author] = {
        author: rec.author,
        post_count: 1,
        first_seen: rec.created_utc,
        last_seen: rec.created_utc,
        subreddits: [rec.subreddit],
      };
    } else {
      existing.post_count++;
      if (rec.created_utc < existing.first_seen)
        existing.first_seen = rec.created_utc;
      if (rec.created_utc > existing.last_seen)
        existing.last_seen = rec.created_utc;
      if (!existing.subreddits.includes(rec.subreddit))
        existing.subreddits.push(rec.subreddit);
    }
  }
  return added;
}
