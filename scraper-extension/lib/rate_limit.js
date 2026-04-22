// Politeness utilities: fixed inter-request spacing plus exponential backoff
// on 429 / 503 responses.

export function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

export class RateLimiter {
  constructor(minIntervalMs = 1100) {
    this.minInterval = Math.max(500, Number(minIntervalMs) || 1100);
    this._last = 0;
  }
  async wait() {
    const now = Date.now();
    const elapsed = now - this._last;
    if (elapsed < this.minInterval) {
      await sleep(this.minInterval - elapsed);
    }
    this._last = Date.now();
  }
}

// Fetch JSON with rate limiting and exponential backoff on transient errors.
export async function politeFetch(
  url,
  { limiter, headers = {}, maxRetries = 4 } = {}
) {
  let delay = 2000;
  let lastErr = null;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    if (limiter) await limiter.wait();
    let resp;
    try {
      resp = await fetch(url, { headers, credentials: "omit" });
    } catch (err) {
      lastErr = err;
      if (attempt === maxRetries) throw err;
      await sleep(delay);
      delay *= 2;
      continue;
    }
    if (resp.status === 429 || resp.status === 503) {
      lastErr = new Error(`HTTP ${resp.status}`);
      if (attempt === maxRetries) throw lastErr;
      await sleep(delay);
      delay *= 2;
      continue;
    }
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status} ${resp.statusText} for ${url}`);
    }
    return resp.json();
  }
  throw lastErr || new Error("politeFetch: exhausted retries");
}
