const els = {
  outputFolder: document.getElementById("outputFolder"),
  rateLimitMs: document.getElementById("rateLimitMs"),
  minPosts: document.getElementById("minPosts"),
  subreddits: document.getElementById("subreddits"),
  save: document.getElementById("save"),
  savedMsg: document.getElementById("saved-msg"),
};

function send(cmd, extra = {}) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ cmd, ...extra }, (resp) => resolve(resp));
  });
}

async function load() {
  const { options } = await send("getOptions");
  els.outputFolder.value = options.outputFolder;
  els.rateLimitMs.value = options.rateLimitMs;
  els.minPosts.value = options.minPosts;
  els.subreddits.value = (options.subreddits || []).join(", ");
}

els.save.addEventListener("click", async () => {
  const subs = els.subreddits.value
    .split(",")
    .map((s) => s.trim().replace(/^r\//, ""))
    .filter(Boolean);
  const rate = Math.max(500, parseInt(els.rateLimitMs.value, 10) || 1100);
  const minPosts = Math.max(1, parseInt(els.minPosts.value, 10) || 10);
  await send("setOptions", {
    options: {
      outputFolder: els.outputFolder.value.trim() || "mental-health-scraper",
      rateLimitMs: rate,
      minPosts,
      subreddits: subs,
    },
  });
  els.savedMsg.textContent = "saved";
  setTimeout(() => (els.savedMsg.textContent = ""), 1500);
});

load();
