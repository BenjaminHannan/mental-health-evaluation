"""Verify Tumblr tag-based labels using MentalBERT zero-shot classification.

Approach (label verification via content)
-----------------------------------------
Tag-based labels are noisy (someone posting #bpd could be in crisis, recovery,
or just educational). We use MentalBERT to sanity-check each user's label by
embedding their post content and comparing it to class prototype embeddings.

A user is kept in the final dataset only when the tag-label and the
content-label AGREE. Disagreements are either dropped or flagged as
low_confidence depending on --mode.

Pipeline
--------
1. Load data/user_timelines_tumblr.parquet + user_labels_tumblr.parquet
2. Average the MentalBERT embedding of each user's posts
3. Embed class-prototype prompts (e.g. "I want to kill myself") and average
4. Each user's content label = argmax(cosine similarity to each prototype)
5. Compute confidence = softmax margin between top 2 classes
6. Cross-check tag label vs content label; keep / drop / flag accordingly

Output
------
    data/user_labels_tumblr_verified.parquet   (the kept users)
    data/label_verification_report.json        (agreement stats)

Usage
-----
    python src/verify_labels_mentalbert.py                 # strict: drop disagreements
    python src/verify_labels_mentalbert.py --mode flag     # keep all, mark low-conf
    python src/verify_labels_mentalbert.py --min-posts 5   # require >=5 posts/user
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR        = Path(__file__).resolve().parent.parent / "data"
TIMELINES_IN    = DATA_DIR / "user_timelines_tumblr.parquet"
LABELS_IN       = DATA_DIR / "user_labels_tumblr.parquet"
LABELS_OUT      = DATA_DIR / "user_labels_tumblr_verified.parquet"
REPORT_OUT      = DATA_DIR / "label_verification_report.json"

MODEL_NAME = "mental/mental-bert-base-uncased"

# ── Class prototypes ──────────────────────────────────────────────────────
# Each class gets multiple prototype sentences; we average them.

CLASS_PROTOTYPES: dict[str, list[str]] = {
    "crisis": [
        "i want to kill myself tonight",
        "i cant do this anymore i want to die",
        "im going to end it all",
        "i have a plan to hurt myself",
        "im suicidal right now and cant stop",
        "everything hurts and i dont want to be alive",
    ],
    "recovery": [
        "im doing better than i was last year",
        "therapy has been helping me cope",
        "ive been clean from self harm for six months",
        "im learning to manage my depression",
        "today was hard but i made it through",
        "im proud of the progress ive made",
    ],
    "neither": [
        "i went to the park with my friends",
        "the new movie was really good",
        "i love this book so far",
        "heres a photo from my trip last week",
        "cant wait for the weekend",
        "my favorite song is on the radio",
    ],
}

LABELS = ["crisis", "recovery", "neither"]


# ── Embedding helpers ─────────────────────────────────────────────────────

def mean_pool(last_hidden: torch.Tensor,
              attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-6)
    return summed / counts


@torch.no_grad()
def embed_texts(texts: list[str],
                tokenizer: Any,
                model: Any,
                device: torch.device,
                batch_size: int = 32,
                max_len: int = 128) -> np.ndarray:
    if not texts:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)

    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state
        vec = mean_pool(h, enc["attention_mask"])
        vec = torch.nn.functional.normalize(vec, dim=-1)   # L2 normalise
        out.append(vec.cpu().numpy().astype(np.float32))
    return np.vstack(out)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of normalised vectors (rows of a vs rows of b)."""
    return a @ b.T


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["drop", "flag"], default="drop",
                   help="drop: remove users whose tag label disagrees with content. "
                        "flag: keep all users but set low_confidence=True on disagreements.")
    p.add_argument("--min-posts", type=int, default=5,
                   help="Skip users with fewer than this many usable posts.")
    p.add_argument("--max-posts-per-user", type=int, default=100,
                   help="Cap per-user posts used for embedding (speed).")
    p.add_argument("--min-margin", type=float, default=0.02,
                   help="Minimum softmax margin for the content label to count.")
    args = p.parse_args()

    # ── Load data ────────────────────────────────────────────────────────
    if not TIMELINES_IN.exists() or not LABELS_IN.exists():
        print("ERROR: Run src/collect_tumblr.py first.")
        return

    print("[verify] loading timelines and labels...")
    tl  = pd.read_parquet(TIMELINES_IN)
    lbl = pd.read_parquet(LABELS_IN)
    print(f"  {len(lbl)} users, {len(tl)} posts")

    # Build text per post
    tl["text"] = (tl["title"].fillna("") + " " + tl["body"].fillna("")).str.strip()
    tl = tl[tl["text"].str.len() > 10]

    # ── Load MentalBERT ──────────────────────────────────────────────────
    print(f"[verify] loading {MODEL_NAME}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    print(f"  device={device}")

    # ── Embed class prototypes ───────────────────────────────────────────
    print("[verify] embedding class prototypes...")
    proto_vecs = []
    for cls in LABELS:
        vecs = embed_texts(CLASS_PROTOTYPES[cls], tok, model, device)
        proto_vecs.append(vecs.mean(axis=0, keepdims=True))
    proto = np.vstack(proto_vecs)                     # (3, d)
    proto = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-8)

    # ── Score each user ──────────────────────────────────────────────────
    print("[verify] scoring users...")
    grouped = tl.groupby("author", sort=False)
    rows: list[dict] = []

    n_total = len(lbl)
    for i, (author, tag_label, tp_date, low_conf) in enumerate(
            lbl[["author", "label", "tp_date", "low_confidence"]].itertuples(
                index=False, name=None), 1):

        if author not in grouped.groups:
            continue
        user_posts = grouped.get_group(author)["text"].tolist()
        if len(user_posts) < args.min_posts:
            continue
        user_posts = user_posts[:args.max_posts_per_user]

        user_vecs = embed_texts(user_posts, tok, model, device)
        user_mean = user_vecs.mean(axis=0, keepdims=True)
        user_mean = user_mean / (np.linalg.norm(user_mean) + 1e-8)

        sims = cosine(user_mean, proto).flatten()     # (3,)
        # Softmax for margin / confidence
        sims_s = sims - sims.max()                    # numerical stability
        probs  = np.exp(sims_s * 10) / np.exp(sims_s * 10).sum()  # temperature=0.1
        top_i  = int(np.argmax(probs))
        top2_i = int(np.argsort(probs)[-2])
        content_label = LABELS[top_i]
        margin        = float(probs[top_i] - probs[top2_i])
        confidence    = float(probs[top_i])

        agrees = (content_label == tag_label) and (margin >= args.min_margin)

        rows.append({
            "author":         author,
            "label":          tag_label,
            "content_label":  content_label,
            "agrees":         agrees,
            "confidence":     confidence,
            "margin":         margin,
            "sim_crisis":     float(sims[0]),
            "sim_recovery":   float(sims[1]),
            "sim_neither":    float(sims[2]),
            "n_posts":        len(user_posts),
            "tp_date":        tp_date,
            "low_confidence": bool(low_conf) or not agrees,
        })

        if i % 50 == 0:
            print(f"  {i}/{n_total} users scored", flush=True)

    if not rows:
        print("[verify] no users passed filtering.")
        return

    df = pd.DataFrame(rows)

    # ── Agreement report ─────────────────────────────────────────────────
    print("\n===== Agreement report =====")
    agree_rate = df["agrees"].mean()
    print(f"  overall agreement: {agree_rate:.1%}  ({df['agrees'].sum()}/{len(df)})")

    per_label = {}
    for lab in LABELS:
        sub = df[df["label"] == lab]
        if sub.empty:
            continue
        rate = sub["agrees"].mean()
        print(f"  {lab:<10}  n={len(sub):<4}  agreement={rate:.1%}")
        per_label[lab] = {
            "n":         int(len(sub)),
            "agree":     int(sub["agrees"].sum()),
            "agree_pct": float(rate),
        }

    # Cross-tab of tag vs content
    print("\n  Tag label vs content label (counts):")
    xt = pd.crosstab(df["label"], df["content_label"])
    print(xt.to_string())

    # ── Filter / flag ────────────────────────────────────────────────────
    if args.mode == "drop":
        kept = df[df["agrees"]].copy()
    else:
        kept = df.copy()      # flag mode keeps everything
    print(f"\n[verify] kept {len(kept)}/{len(df)} users in mode={args.mode}")
    print(kept["label"].value_counts().to_string())

    # ── Save ─────────────────────────────────────────────────────────────
    kept_out = kept[["author", "label", "low_confidence", "tp_date",
                     "confidence", "content_label"]]
    kept_out.to_parquet(LABELS_OUT, index=False)
    print(f"[verify] saved -> {LABELS_OUT}")

    report = {
        "total_users":   int(len(df)),
        "kept_users":    int(len(kept)),
        "agreement_rate": float(agree_rate),
        "per_label":     per_label,
        "crosstab":      xt.to_dict(),
        "mode":          args.mode,
        "min_margin":    args.min_margin,
        "min_posts":     args.min_posts,
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2))
    print(f"[verify] report -> {REPORT_OUT}")


if __name__ == "__main__":
    main()
