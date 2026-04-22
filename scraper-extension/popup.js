const DEFAULT_SUBS = ["depression", "ADHD", "PTSD", "OCD", "aspergers"];

const els = {
  chips: document.getElementById("sub-chips"),
  addSub: document.getElementById("add-sub"),
  addSubBtn: document.getElementById("add-sub-btn"),
  listing: document.getElementById("listing"),
  timeWindow: document.getElementById("timeWindow"),
  listingPageCap: document.getElementById("listingPageCap"),
  userHistoryPageCap: document.getElementById("userHistoryPageCap"),
  tPosts: document.getElementById("t-posts"),
  tUsers: document.getElementById("t-users"),
  tQual: document.getElementById("t-qual"),
  phase: document.getElementById("phase"),
  runBtn: document.getElementById("run-btn"),
  stopBtn: document.getElementById("stop-btn"),
  exportBtn: document.getElementById("export-btn"),
  clearBtn: document.getElementById("clear-btn"),
  optionsLink: document.getElementById("options-link"),
};

let subs = [];

function send(cmd, extra = {}) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ cmd, ...extra }, (resp) => resolve(resp));
  });
}

function renderChips() {
  els.chips.innerHTML = "";
  for (const s of subs) {
    const chip = document.createElement("span");
    chip.className = "chip active";
    chip.innerHTML = `<span>r/${s}</span><span class="x" title="remove">&times;</span>`;
    chip.querySelector(".x").addEventListener("click", (e) => {
      e.stopPropagation();
      subs = subs.filter((x) => x !== s);
      persistSubs();
      renderChips();
    });
    els.chips.appendChild(chip);
  }
}

async function persistSubs() {
  await send("setOptions", { options: { subreddits: subs } });
}

async function loadUI() {
  const { options } = await send("getOptions");
  subs = (options.subreddits && options.subreddits.length) ? options.subreddits : DEFAULT_SUBS.slice();
  els.listing.value = options.listing || "new";
  els.timeWindow.value = options.timeWindow || "month";
  els.listingPageCap.value = options.listingPageCap ?? 2;
  els.userHistoryPageCap.value = options.userHistoryPageCap ?? 2;
  renderChips();
  await refreshStatus();
}

async function refreshStatus() {
  const { status } = await send("status");
  if (!status) return;
  els.tPosts.textContent = status.totals.posts;
  els.tUsers.textContent = status.totals.users;
  els.tQual.textContent = status.totals.qualified;
  els.runBtn.disabled = status.running;
  els.stopBtn.disabled = !status.running;
  if (status.error) {
    els.phase.textContent = `error: ${status.error}`;
  } else if (status.progress) {
    const p = status.progress;
    const detail = p.phase === "discover"
      ? `discover r/${p.sub || "?"}`
      : p.phase === "history"
      ? `history ${p.usersDone}/${p.usersQueued} (${p.currentAuthor || ""})`
      : p.phase;
    els.phase.textContent = detail;
  } else {
    els.phase.textContent = status.running ? "running…" : "idle";
  }
}

els.addSubBtn.addEventListener("click", () => {
  const v = els.addSub.value.trim().replace(/^r\//, "");
  if (!v) return;
  if (!subs.includes(v)) subs.push(v);
  els.addSub.value = "";
  persistSubs();
  renderChips();
});

els.addSub.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    els.addSubBtn.click();
  }
});

els.runBtn.addEventListener("click", async () => {
  els.runBtn.disabled = true;
  const config = {
    subreddits: subs,
    listing: els.listing.value,
    timeWindow: els.timeWindow.value,
    listingPageCap: parseInt(els.listingPageCap.value, 10) || 1,
    userHistoryPageCap: parseInt(els.userHistoryPageCap.value, 10) || 0,
  };
  await send("setOptions", { options: config });
  const resp = await send("start", { config });
  if (!resp.ok) {
    els.phase.textContent = `error: ${resp.error}`;
    els.runBtn.disabled = false;
  }
});

els.stopBtn.addEventListener("click", async () => {
  await send("stop");
});

els.exportBtn.addEventListener("click", async () => {
  const resp = await send("export");
  if (resp.ok) {
    els.phase.textContent = `exported ${resp.posts} posts / ${resp.users} users`;
  } else {
    els.phase.textContent = `export failed: ${resp.error}`;
  }
});

els.clearBtn.addEventListener("click", async () => {
  if (!confirm("Clear the entire collected dataset from extension storage?")) return;
  await send("clear");
  await refreshStatus();
});

els.optionsLink.addEventListener("click", (e) => {
  e.preventDefault();
  chrome.runtime.openOptionsPage();
});

loadUI();
setInterval(refreshStatus, 1500);
