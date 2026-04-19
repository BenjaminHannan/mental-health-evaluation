"""Compute MentalBERT embeddings and derive per-user semantic-shift features.

Stage 3c of the pipeline. Runs ``mental/mental-bert-base-uncased`` over
every post in ``data/user_timelines.parquet`` and caches the [CLS]
embeddings to ``data/mentalbert_embeddings.npz``. A second pass
aggregates the per-post embeddings into per-user-window mean vectors
and derives 6 scalar features capturing semantic shift relative to
the user's own baseline:

    cos_sim_pre_4w   cosine similarity(baseline mean, pre_4w mean)
    cos_sim_pre_2w
    cos_sim_pre_1w
    l2_dist_pre_4w   L2 distance between the same two centroids
    l2_dist_pre_2w
    l2_dist_pre_1w

Why only 6 features?
-------------------
The full 4 x 768 mean-pooled vectors would overwhelm a 581-user
classifier. Representing MentalBERT as a semantic-shift magnitude is
interpretable and directly motivated by the Moments-of-Change framing.

Run:
    python src/extract_mentalbert.py           # full run (all posts)
    python src/extract_mentalbert.py --labelled-only   # only labelled users
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

DATA_DIR      = Path(__file__).resolve().parent.parent / "data"
LABELS_IN     = DATA_DIR / "user_labels.parquet"
TIMELINES_IN  = DATA_DIR / "user_timelines.parquet"
EMB_CACHE     = DATA_DIR / "mentalbert_embeddings.npz"
FEATURES_OUT  = DATA_DIR / "features_mentalbert.parquet"

# Tried in order; first one that loads wins. MentalBERT is gated on HF;
# the mental-roberta mirror is sometimes open. If both fail, a
# domain-general sentence encoder is used as a documented fallback.
MODEL_CANDIDATES = [
    "AIMH/mental-roberta-base",
    "mental/mental-roberta-base",
    "mental/mental-bert-base-uncased",
    "sentence-transformers/all-mpnet-base-v2",
]
MAX_LEN       = 256
BATCH_SIZE    = 32

WINDOWS = ["baseline", "pre_4w", "pre_2w", "pre_1w"]
PRE_WINDOWS = ["pre_4w", "pre_2w", "pre_1w"]
WINDOW_WEEKS = 4


# ── Embedding cache ───────────────────────────────────────────────────────

def _compose_text(row: pd.Series) -> str:
    title = row.get("title") or ""
    body  = row.get("body")  or ""
    return (str(title) + " " + str(body)).strip()


def _load_model_with_fallback() -> tuple[object, object, str]:
    """Try MODEL_CANDIDATES in order; return (tokenizer, model, name)."""
    last_err: Exception | None = None
    for name in MODEL_CANDIDATES:
        try:
            print(f"[extract_mentalbert] trying model: {name}")
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModel.from_pretrained(name)
            print(f"[extract_mentalbert] loaded: {name}")
            return tok, mdl, name
        except Exception as e:
            print(f"[extract_mentalbert]   -> unavailable ({type(e).__name__})")
            last_err = e
            continue
    raise RuntimeError(
        f"All model candidates failed. Last error: {last_err}"
    )


def compute_post_embeddings(timelines: pd.DataFrame) -> tuple[dict[str, np.ndarray], str]:
    """Run MentalBERT over every post and return ({post_id: [CLS]}, model_name)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract_mentalbert] device: {device}")
    tokenizer, model, model_name = _load_model_with_fallback()
    model = model.to(device).eval()

    ids     = timelines["id"].tolist()
    texts   = [_compose_text(r) for _, r in timelines.iterrows()]
    n_total = len(ids)
    print(f"[extract_mentalbert] embedding {n_total} posts "
          f"(batch={BATCH_SIZE}, max_len={MAX_LEN})...")

    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for start in range(0, n_total, BATCH_SIZE):
            batch_texts = texts[start:start + BATCH_SIZE]
            batch_ids   = ids[start:start + BATCH_SIZE]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(device)
            outputs = model(**enc)
            # [CLS] token embedding: last_hidden_state[:, 0, :]
            cls = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            for pid, vec in zip(batch_ids, cls):
                out[pid] = vec.astype(np.float32)
            if (start // BATCH_SIZE) % 10 == 0:
                done = start + len(batch_texts)
                print(f"[extract_mentalbert]   {done}/{n_total}", flush=True)
    return out


def save_embedding_cache(emb: dict[str, np.ndarray], path: Path) -> None:
    """Save post-id -> 768-dim vector map as an npz archive."""
    ids  = np.array(list(emb.keys()))
    vecs = np.stack([emb[pid] for pid in ids]).astype(np.float32)
    np.savez_compressed(path, ids=ids, vecs=vecs)
    print(f"[extract_mentalbert] saved {len(ids)} embeddings to {path} "
          f"({vecs.nbytes/1e6:.1f} MB uncompressed)")


def load_embedding_cache(path: Path) -> dict[str, np.ndarray]:
    """Load npz archive back into {post_id: vec}."""
    arr = np.load(path, allow_pickle=False)
    ids, vecs = arr["ids"], arr["vecs"]
    return {str(pid): vec for pid, vec in zip(ids, vecs)}


# ── Window slicing ────────────────────────────────────────────────────────

def _window_ids(
    user_posts: pd.DataFrame,
    tp_date: pd.Timestamp,
    window: str,
) -> list[str]:
    ts = user_posts["created_utc"]
    cutoff_4w = tp_date - pd.Timedelta(weeks=4)
    cutoff_2w = tp_date - pd.Timedelta(weeks=2)
    cutoff_1w = tp_date - pd.Timedelta(weeks=1)
    if window == "baseline":
        mask = ts < cutoff_4w
    elif window == "pre_4w":
        mask = (ts >= cutoff_4w) & (ts < tp_date)
    elif window == "pre_2w":
        mask = (ts >= cutoff_2w) & (ts < tp_date)
    elif window == "pre_1w":
        mask = (ts >= cutoff_1w) & (ts < tp_date)
    else:
        raise ValueError(window)
    return user_posts.loc[mask, "id"].tolist()


def _mean_vec(
    post_ids: list[str],
    emb: dict[str, np.ndarray],
) -> np.ndarray | None:
    vecs = [emb[pid] for pid in post_ids if pid in emb]
    if not vecs:
        return None
    return np.mean(np.stack(vecs), axis=0)


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def build_user_row(
    author: str,
    user_posts: pd.DataFrame,
    label_row: pd.Series,
    emb: dict[str, np.ndarray],
) -> dict:
    """Derive 6 semantic-shift features plus metadata for one user."""
    row: dict = {
        "author":         author,
        "label":          label_row["label"],
        "low_confidence": label_row["low_confidence"],
        "n_posts":        len(user_posts),
        "tp_date":        label_row["tp_date"],
    }
    tp_date = label_row["tp_date"]
    if pd.isnull(tp_date):
        tp_date = user_posts["created_utc"].max()

    win_means: dict[str, np.ndarray | None] = {}
    for win in WINDOWS:
        pids = _window_ids(user_posts, tp_date, win)
        win_means[win] = _mean_vec(pids, emb)

    base = win_means["baseline"]
    for win in PRE_WINDOWS:
        v = win_means[win]
        if base is None or v is None:
            cos = float("nan")
            l2  = float("nan")
        else:
            cos = _cos_sim(base, v)
            l2  = float(np.linalg.norm(base - v))
        row[f"cos_sim_{win}"] = cos
        row[f"l2_dist_{win}"] = l2

    return row


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labelled-only", action="store_true",
        help="Only embed posts from the 132 labelled crisis/recovery users",
    )
    parser.add_argument(
        "--use-cache", action="store_true",
        help="Skip embedding; read cached post embeddings and derive features",
    )
    args = parser.parse_args()

    print("[extract_mentalbert] loading timelines and labels...")
    timelines = pd.read_parquet(TIMELINES_IN)
    labels    = pd.read_parquet(LABELS_IN)
    if "low_confidence" not in labels.columns:
        labels["low_confidence"] = False

    if args.labelled_only:
        keep = labels[labels["label"].isin(["crisis", "recovery"])]["author"]
        timelines = timelines[timelines["author"].isin(keep)].copy()
        print(f"[extract_mentalbert] restricted to {len(timelines)} posts "
              f"from {timelines['author'].nunique()} labelled users")

    if args.use_cache and EMB_CACHE.exists():
        print(f"[extract_mentalbert] loading cached embeddings from {EMB_CACHE}")
        emb = load_embedding_cache(EMB_CACHE)
    else:
        emb = compute_post_embeddings(timelines)
        save_embedding_cache(emb, EMB_CACHE)

    print("[extract_mentalbert] building per-user semantic-shift features...")
    labels_idx = labels.set_index("author")
    grouped    = timelines.groupby("author", sort=False)
    rows: list[dict] = []
    for i, (author, user_posts) in enumerate(grouped, 1):
        if author not in labels_idx.index:
            continue
        if i % 100 == 0:
            print(f"[extract_mentalbert]   {i}/{len(grouped)}", flush=True)
        rows.append(build_user_row(author, user_posts, labels_idx.loc[author], emb))

    features = pd.DataFrame(rows)
    features.to_parquet(FEATURES_OUT, index=False)
    print(f"[extract_mentalbert] saved {len(features)} rows to {FEATURES_OUT}")

    # Brief summary
    print("\n===== MentalBERT semantic shift (mean by label) =====")
    for feat in ["cos_sim_pre_4w", "cos_sim_pre_2w", "cos_sim_pre_1w",
                 "l2_dist_pre_4w", "l2_dist_pre_2w", "l2_dist_pre_1w"]:
        print(f"  {feat}")
        for lbl in ["crisis", "recovery", "neither"]:
            vals = features[features["label"] == lbl][feat].dropna()
            if len(vals):
                print(f"    {lbl:<10} mean={vals.mean():.4f}  "
                      f"std={vals.std():.4f}  n={len(vals)}")


if __name__ == "__main__":
    main()
