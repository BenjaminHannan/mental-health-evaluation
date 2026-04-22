// Canonical record schema used by the reddit-mental-health pipeline.
// Mirrors reddit-mental-health/src/collect_data.py::_item_to_row.

export const REQUIRED_COLS = [
  "author",
  "body",
  "created_utc",
  "id",
  "num_comments",
  "score",
  "subreddit",
  "title",
  "upvote_ratio",
  "url",
  "days_to_tp",
];

const DELETED_BODIES = new Set(["[deleted]", "[removed]"]);
const DELETED_AUTHORS = new Set(["", "[deleted]", "[removed]"]);

// Normalize a Reddit listing child (either `t3` submission or `t1` comment) to
// the pipeline's canonical schema. Returns null for records that should be dropped.
export function normalize(child) {
  if (!child || typeof child !== "object") return null;
  const kind = child.kind;
  const item = child.data || {};

  const author = (item.author || "").trim();
  if (DELETED_AUTHORS.has(author)) return null;

  const createdSec = Number(item.created_utc);
  if (!Number.isFinite(createdSec) || createdSec <= 0) return null;

  let body;
  if (kind === "t3") body = item.selftext || "";
  else body = item.body || "";
  if (DELETED_BODIES.has(body)) body = "";

  const permalink = item.permalink
    ? `https://www.reddit.com${item.permalink}`
    : "";

  const upvoteRatio =
    kind === "t3" && Number.isFinite(+item.upvote_ratio)
      ? +item.upvote_ratio
      : null;

  return {
    author,
    body,
    created_utc: new Date(createdSec * 1000).toISOString(),
    id: item.id || "",
    num_comments: Number.isFinite(+item.num_comments) ? +item.num_comments : 0,
    score: Number.isFinite(+item.score) ? +item.score : 0,
    subreddit: (item.subreddit || "").toLowerCase(),
    title: kind === "t3" ? item.title || "" : "",
    upvote_ratio: upvoteRatio,
    url: item.url || permalink || "",
    days_to_tp: null,
  };
}
