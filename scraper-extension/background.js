// Service worker: orchestrates the two-phase scrape and writes JSON outputs
// into ~/Downloads/<outputFolder>/ via chrome.downloads.

import { iterateSubreddit, iterateUserHistory } from "./lib/reddit.js";
import { RateLimiter, sleep } from "./lib/rate_limit.js";
import {
  loadState,
  saveState,
  clearState,
  mergeRecords,
} from "./lib/dedupe.js";

const DEFAULT_OPTS = {
  subreddits: ["depression", "ADHD", "PTSD", "OCD", "aspergers"],
  listing: "new", // new | hot | top
  timeWindow: "month", // only used for listing === "top": hour|day|week|month|year|all
  listingPageCap: 2, // pages × 100 posts per subreddit
  userHistoryPageCap: 2, // pages × 100 per endpoint (submitted + comments)
  rateLimitMs: 1100,
  minPosts: 10, // mirrors MIN_POSTS in label_users.py
  outputFolder: "mental-health-scraper",
};

// Module-scope job handle. MV3 service workers can suspend, so we also
// persist progress to storage for UI resilience.
let scrapeJob = null;

async function getOptions() {
  const stored = await chrome.storage.local.get("options");
  return { ...DEFAULT_OPTS, ...(stored.options || {}) };
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  (async () => {
    try {
      switch (msg?.cmd) {
        case "status":
          sendResponse({ ok: true, status: await statusSnapshot() });
          break;
        case "start":
          sendResponse({ ok: true, started: await startScrape(msg.config || {}) });
          break;
        case "stop":
          if (scrapeJob) scrapeJob.cancel = true;
          sendResponse({ ok: true });
          break;
        case "export":
          sendResponse({ ok: true, ...(await exportJSON()) });
          break;
        case "clear":
          await clearState();
          sendResponse({ ok: true });
          break;
        case "getOptions":
          sendResponse({ ok: true, options: await getOptions() });
          break;
        case "setOptions":
          await chrome.storage.local.set({
            options: { ...(await getOptions()), ...(msg.options || {}) },
          });
          sendResponse({ ok: true });
          break;
        default:
          sendResponse({ ok: false, error: `unknown cmd: ${msg?.cmd}` });
      }
    } catch (err) {
      console.error("[background] handler failed:", err);
      sendResponse({ ok: false, error: String((err && err.message) || err) });
    }
  })();
  return true; // async response
});

async function statusSnapshot() {
  const state = await loadState();
  const minPosts = (await getOptions()).minPosts;
  return {
    running: !!scrapeJob && !scrapeJob.done,
    error: scrapeJob?.error || null,
    progress: scrapeJob?.progress || null,
    totals: {
      posts: state.posts.length,
      users: Object.keys(state.userStats).length,
      qualified: Object.values(state.userStats).filter(
        (u) => u.post_count > minPosts
      ).length,
    },
  };
}

async function startScrape(overrides) {
  if (scrapeJob && !scrapeJob.done) {
    throw new Error("a scrape is already running");
  }
  const opts = { ...(await getOptions()), ...overrides };
  scrapeJob = {
    progress: { phase: "starting", sub: null, usersQueued: 0, usersDone: 0 },
    cancel: false,
    done: false,
    error: null,
    errors: 0,
  };
  // Fire-and-forget — popup polls status
  runScrape(opts)
    .catch((e) => {
      console.error("[scrape] run failed:", e);
      scrapeJob.error = String((e && e.message) || e);
    })
    .finally(() => {
      scrapeJob.done = true;
    });
  return true;
}

async function runScrape(opts) {
  const limiter = new RateLimiter(opts.rateLimitMs);
  const state = await loadState();
  const discoveredAuthors = new Set();

  // Phase 1: discover users via subreddit listings
  for (const sub of opts.subreddits) {
    if (scrapeJob.cancel) break;
    scrapeJob.progress = {
      phase: "discover",
      sub,
      usersQueued: discoveredAuthors.size,
      usersDone: 0,
    };
    try {
      for await (const rec of iterateSubreddit({
        subreddit: sub,
        listing: opts.listing,
        pageCap: opts.listingPageCap,
        timeWindow: opts.timeWindow,
        limiter,
      })) {
        if (scrapeJob.cancel) break;
        mergeRecords(state, [rec]);
        discoveredAuthors.add(rec.author);
      }
    } catch (err) {
      console.warn(`[scrape] sub ${sub} failed:`, err);
      scrapeJob.errors++;
    }
  }
  await saveState(state);

  // Phase 2: harvest each user's history
  const authors = Array.from(discoveredAuthors);
  let done = 0;
  for (const author of authors) {
    if (scrapeJob.cancel) break;
    scrapeJob.progress = {
      phase: "history",
      sub: null,
      currentAuthor: author,
      usersQueued: authors.length,
      usersDone: done,
    };
    try {
      for await (const rec of iterateUserHistory({
        username: author,
        pageCap: opts.userHistoryPageCap,
        limiter,
      })) {
        if (scrapeJob.cancel) break;
        mergeRecords(state, [rec]);
      }
    } catch (err) {
      console.warn(`[scrape] history ${author} failed:`, err);
      scrapeJob.errors++;
    }
    done++;
    if (done % 5 === 0) await saveState(state);
  }

  await saveState(state);
  await writeOutputs(state, opts);
  await writeRunLog(opts, scrapeJob, state);
  scrapeJob.progress = {
    phase: scrapeJob.cancel ? "canceled" : "complete",
    sub: null,
    usersQueued: authors.length,
    usersDone: done,
  };
}

async function exportJSON() {
  const opts = await getOptions();
  const state = await loadState();
  await writeOutputs(state, opts);
  return {
    posts: state.posts.length,
    users: Object.keys(state.userStats).length,
  };
}

async function writeOutputs(state, opts) {
  const folder = sanitizeFolder(opts.outputFolder);
  await downloadJSON(`${folder}/posts.json`, state.posts);
  const users = Object.values(state.userStats).map((u) => ({
    ...u,
    qualifies_min_posts: u.post_count > (opts.minPosts ?? 10),
  }));
  await downloadJSON(`${folder}/users.json`, users);
}

async function writeRunLog(opts, job, state) {
  const folder = sanitizeFolder(opts.outputFolder);
  const { runs = [] } = await chrome.storage.local.get("runs");
  runs.push({
    timestamp: new Date().toISOString(),
    subreddits: opts.subreddits,
    listing: opts.listing,
    timeWindow: opts.timeWindow,
    listingPageCap: opts.listingPageCap,
    userHistoryPageCap: opts.userHistoryPageCap,
    rateLimitMs: opts.rateLimitMs,
    posts_total: state.posts.length,
    users_total: Object.keys(state.userStats).length,
    errors: job.errors,
    canceled: !!job.cancel,
  });
  await chrome.storage.local.set({ runs });
  await downloadJSON(`${folder}/run_log.json`, runs);
}

function sanitizeFolder(name) {
  const cleaned = String(name || "mental-health-scraper")
    .replace(/[^a-zA-Z0-9_\-\/]/g, "-")
    .replace(/^\/+|\/+$/g, "");
  return cleaned || "mental-health-scraper";
}

// Write JSON to ~/Downloads/<filename>. Tries blob URL first (fastest,
// works in recent Chrome service workers); falls back to data: URL.
async function downloadJSON(relpath, data) {
  const json = JSON.stringify(data, null, 2);
  let url;
  let revoke = null;
  try {
    const blob = new Blob([json], { type: "application/json" });
    url = URL.createObjectURL(blob);
    revoke = () => URL.revokeObjectURL(url);
  } catch {
    url = `data:application/json;charset=utf-8,${encodeURIComponent(json)}`;
  }
  try {
    await chrome.downloads.download({
      url,
      filename: relpath,
      conflictAction: "overwrite",
      saveAs: false,
    });
    // Chrome needs a beat to read the blob before we revoke it.
    await sleep(1000);
  } finally {
    if (revoke) revoke();
  }
}
