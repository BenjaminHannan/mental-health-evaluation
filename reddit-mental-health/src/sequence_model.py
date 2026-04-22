"""LSTM temporal sequence model for crisis/recovery/neither classification.

Stage 6 alternative to the flat-feature classifiers in train_model.py.
Rather than aggregating pre-T windows into fixed-width features, this
script bucketises each user's baseline + pre-windows into **weekly
feature vectors** (7 linguistic features per week), pads to a common
length, and trains a small LSTM with an attention head to predict the
3-way label.

Motivation
----------
The flat-feature approach loses trajectory shape: a user whose sentiment
falls gradually over 8 weeks looks identical to one who crashes in the
final week, once you reduce both to pre_4w / pre_2w / pre_1w buckets.
An LSTM that consumes the whole weekly time series can (in principle)
learn the *shape* of pre-crisis dynamics directly.

This is the approach Tsakalidis et al. (ACL 2022) use for their
Moments-of-Change task.

Data representation
-------------------
For each labelled user we build a per-week matrix X_u of shape
(n_weeks, n_features). The last WEEKS_PRE_T weeks end at the
turning-point date T; everything before that is the baseline. Missing
weeks are filled with zeros and flagged by a presence mask.

We truncate or pad every user to MAX_WEEKS = 40 total weeks (the
median user in our cohort has ~36 active weeks).

Model
-----
    input  -> bidirectional LSTM (hidden=64) -> masked attention pool
           -> dropout -> dense(3) -> softmax

Training: AdamW, learning_rate=1e-3, 40 epochs, class-weighted
cross-entropy (to handle the imbalance that was hurting RF recall).
Evaluation: 5-fold stratified CV, pooled out-of-fold AUC to match
train_model.py's convention.

Run
---
    python src/sequence_model.py
    python src/sequence_model.py --epochs 80 --hidden 128
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as e:
    raise SystemExit(
        "[sequence_model] PyTorch required: pip install torch"
    ) from e

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR      = Path(__file__).resolve().parent.parent / "data"
LABELS_IN     = DATA_DIR / "user_labels.parquet"
TIMELINES_IN  = DATA_DIR / "user_timelines.parquet"
RESULTS_OUT   = DATA_DIR / "sequence_model_results.json"

# ── Hyperparameters ───────────────────────────────────────────────────────
MAX_WEEKS     = 40
N_FOLDS       = 5
RANDOM_STATE  = 42
EPOCHS        = 40
BATCH_SIZE    = 16
HIDDEN_DIM    = 64
DROPOUT       = 0.3
LR            = 1e-3
WEIGHT_DECAY  = 1e-4

LABEL_ORDER   = ["crisis", "recovery", "neither"]
LABEL_TO_IDX  = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}

VADER = SentimentIntensityAnalyzer()

# ── Per-week feature bucketing ─────────────────────────────────────────────

import re
_WORD_RE = re.compile(r"\b[a-z']+\b")

FP_PRONOUNS = frozenset({"i", "me", "my", "mine", "myself"})
NEG_AFFECT_WORDS = frozenset({
    "sad", "depressed", "hopeless", "empty", "alone", "lonely",
    "anxious", "worried", "scared", "panic", "stress", "hate",
    "angry", "awful", "terrible", "horrible", "broken", "pain",
    "hurt", "exhausted", "lost", "confused", "dying", "shame",
    "guilt", "tired", "crying", "tears",
})


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _week_features(posts_in_week: pd.DataFrame) -> np.ndarray:
    """Compute a 7-dim feature vector for posts in one week.

    Returns zeros if no posts.
    """
    if posts_in_week.empty:
        return np.zeros(7, dtype=np.float32)
    texts       = (posts_in_week["body"].fillna("") + " " +
                   posts_in_week["title"].fillna("")).tolist()
    tokens_list = [_tokens(t) for t in texts]
    all_tokens  = [tok for sub in tokens_list for tok in sub]
    n_tok       = len(all_tokens) or 1

    sentiment   = np.mean([VADER.polarity_scores(t)["compound"] for t in texts])
    ttr         = len(set(all_tokens)) / n_tok if n_tok > 1 else 0.0
    avg_sent_len = np.mean([len(_WORD_RE.findall(t)) for t in texts])
    post_freq   = float(len(posts_in_week))    # this week only
    fp_rate     = sum(1 for t in all_tokens if t in FP_PRONOUNS) / n_tok
    neg_rate    = sum(1 for t in all_tokens if t in NEG_AFFECT_WORDS) / n_tok
    avg_len     = np.mean([len(t) for t in tokens_list])

    return np.array([sentiment, ttr, avg_sent_len, post_freq,
                     fp_rate, neg_rate, avg_len], dtype=np.float32)


def build_user_sequence(
    user_posts: pd.DataFrame,
    tp_date: pd.Timestamp,
    max_weeks: int = MAX_WEEKS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, mask) for one user.

    X is shape (max_weeks, 7). The last week ends at tp_date; earlier
    weeks extend backwards.  Weeks with no posts are zero-filled and
    the mask is 0 there.
    """
    posts = user_posts.sort_values("created_utc").copy()
    # Strip timezone so all comparisons happen in naive UTC.
    posts["created_utc"] = posts["created_utc"].dt.tz_localize(None)
    if tp_date.tzinfo:
        tp_date = pd.Timestamp(tp_date).tz_localize(None)
    else:
        tp_date = pd.Timestamp(tp_date)

    X    = np.zeros((max_weeks, 7), dtype=np.float32)
    mask = np.zeros(max_weeks,        dtype=np.float32)

    for i in range(max_weeks):
        # Week i=0 is the most recent (ending at tp_date); i increases
        # into the past. We will place these in chronological order in X
        # with the most recent at the LAST index.
        end_i   = tp_date - pd.Timedelta(weeks=i)
        start_i = end_i   - pd.Timedelta(weeks=1)
        week_posts = posts[(posts["created_utc"] >= start_i)
                         & (posts["created_utc"] <  end_i)]
        slot = max_weeks - 1 - i    # chronological ordering
        X[slot] = _week_features(week_posts)
        mask[slot] = 1.0 if not week_posts.empty else 0.0
    return X, mask


# ── Dataset ──────────────────────────────────────────────────────────────

class UserSequenceDataset(Dataset):
    def __init__(self, X_list: list[np.ndarray], mask_list: list[np.ndarray],
                 y: np.ndarray):
        self.X    = np.stack(X_list)        # (N, T, F)
        self.mask = np.stack(mask_list)     # (N, T)
        self.y    = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (torch.tensor(self.X[i]),
                torch.tensor(self.mask[i]),
                torch.tensor(self.y[i], dtype=torch.long))


# ── Model ─────────────────────────────────────────────────────────────────

class BiLSTMAttn(nn.Module):
    def __init__(self, n_features: int = 7, hidden: int = HIDDEN_DIM,
                 n_classes: int = 3, dropout: float = DROPOUT):
        super().__init__()
        self.bilstm = nn.LSTM(n_features, hidden, batch_first=True,
                              bidirectional=True)
        self.attn   = nn.Linear(hidden * 2, 1)
        self.drop   = nn.Dropout(dropout)
        self.fc     = nn.Linear(hidden * 2, n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: (B, T, F)  mask: (B, T)
        h, _  = self.bilstm(x)              # (B, T, 2H)
        score = self.attn(h).squeeze(-1)    # (B, T)
        score = score.masked_fill(mask < 0.5, -1e9)
        w     = F.softmax(score, dim=1).unsqueeze(-1)    # (B, T, 1)
        pooled = (h * w).sum(dim=1)         # (B, 2H)
        return self.fc(self.drop(pooled))


# ── Training ─────────────────────────────────────────────────────────────

def _class_weights(y: np.ndarray, n_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes)
    # inverse frequency; normalise so mean=1
    w = len(y) / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def train_one_fold(
    X_train, mask_train, y_train,
    X_val,   mask_val,   y_val,
    epochs: int, device: torch.device, verbose: bool = True,
) -> tuple[np.ndarray, BiLSTMAttn]:
    """Train one fold, return (val_proba (N,3), trained_model)."""
    model   = BiLSTMAttn().to(device)
    opt     = torch.optim.AdamW(model.parameters(), lr=LR,
                                weight_decay=WEIGHT_DECAY)
    weights = _class_weights(y_train).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    ds_tr = UserSequenceDataset(X_train, mask_train, y_train)
    ds_va = UserSequenceDataset(X_val,   mask_val,   y_val)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)

    best_val_loss = float("inf")
    best_proba    = None
    for epoch in range(epochs):
        model.train()
        for x, m, y in dl_tr:
            x, m, y = x.to(device), m.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x, m)
            loss   = loss_fn(logits, y)
            loss.backward()
            opt.step()

        # Validation pass (full val set at once; small fold)
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(np.stack(X_val)).to(device)
            mv = torch.tensor(np.stack(mask_val)).to(device)
            yv = torch.tensor(y_val, dtype=torch.long).to(device)
            logits = model(xv, mv)
            val_loss = loss_fn(logits, yv).item()
            proba    = F.softmax(logits, dim=1).cpu().numpy()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_proba    = proba
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}/{epochs}  val_loss={val_loss:.4f}")
    return best_proba, model


def cross_validate(
    X_list: list[np.ndarray],
    mask_list: list[np.ndarray],
    y: np.ndarray,
    epochs: int,
    device: torch.device,
) -> dict:
    """5-fold stratified CV with pooled OOF predictions."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    n  = len(y)
    oof = np.zeros((n, 3), dtype=np.float32)

    fold_aucs: list[float] = []
    for fold, (tr, va) in enumerate(cv.split(np.zeros(n), y), 1):
        print(f"  fold {fold}/{N_FOLDS}  train={len(tr)} val={len(va)}")
        X_tr = [X_list[i]    for i in tr]
        m_tr = [mask_list[i] for i in tr]
        X_va = [X_list[i]    for i in va]
        m_va = [mask_list[i] for i in va]
        proba, _ = train_one_fold(
            X_tr, m_tr, y[tr],
            X_va, m_va, y[va],
            epochs=epochs, device=device, verbose=False,
        )
        oof[va] = proba
        try:
            fold_auc = roc_auc_score(y[va], proba, multi_class="ovr",
                                     average="macro",
                                     labels=np.array([0, 1, 2]))
        except ValueError:
            fold_auc = float("nan")
        print(f"    fold macro AUC={fold_auc:.4f}")
        fold_aucs.append(fold_auc)

    # Pooled metrics
    y_pred = oof.argmax(axis=1)
    per_class_auc = []
    for i in range(3):
        try:
            per_class_auc.append(roc_auc_score((y == i).astype(int), oof[:, i]))
        except ValueError:
            per_class_auc.append(float("nan"))
    macro_auc = float(np.nanmean(per_class_auc))

    prec, rec, f1, _ = precision_recall_fscore_support(
        y, y_pred, labels=[0, 1, 2], zero_division=0
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y, y_pred, average="macro", zero_division=0
    )

    return {
        "fold_aucs":     [float(x) for x in fold_aucs],
        "macro_auc":     float(macro_auc),
        "macro_f1":      float(macro_f1),
        "per_class_auc": dict(zip(LABEL_ORDER, [float(a) for a in per_class_auc])),
        "per_class_f1":  dict(zip(LABEL_ORDER, [float(f) for f in f1])),
        "per_class_precision": dict(zip(LABEL_ORDER, [float(p) for p in prec])),
        "per_class_recall":    dict(zip(LABEL_ORDER, [float(r) for r in rec])),
    }


# ── Data building ────────────────────────────────────────────────────────

def build_all_sequences(max_weeks: int) -> tuple[
    list[np.ndarray], list[np.ndarray], np.ndarray, list[str]
]:
    """Return parallel lists of per-user X, mask, y, and authors.

    For 'neither' users the last post date is used as a pseudo-TP so
    we always have an anchor for the weekly grid.
    """
    labels    = pd.read_parquet(LABELS_IN)
    timelines = pd.read_parquet(TIMELINES_IN)
    labels    = labels[labels["label"].isin(LABEL_ORDER)].copy()

    idx = labels.set_index("author")
    X_list: list[np.ndarray] = []
    m_list: list[np.ndarray] = []
    y_list: list[int] = []
    authors: list[str] = []

    n_total = len(idx)
    for i, (author, posts) in enumerate(timelines.groupby("author", sort=False)):
        if author not in idx.index:
            continue
        row    = idx.loc[author]
        tp     = row["tp_date"]
        if pd.isnull(tp):
            tp = posts["created_utc"].max()
        X, mask = build_user_sequence(posts, tp, max_weeks=max_weeks)
        X_list.append(X)
        m_list.append(mask)
        y_list.append(LABEL_TO_IDX[row["label"]])
        authors.append(author)
        if (i + 1) % 100 == 0:
            print(f"[sequence_model]   built sequences: {i+1}/{n_total}")

    return X_list, m_list, np.array(y_list, dtype=np.int64), authors


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    global HIDDEN_DIM
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs",    type=int, default=EPOCHS)
    p.add_argument("--hidden",    type=int, default=HIDDEN_DIM)
    p.add_argument("--max-weeks", type=int, default=MAX_WEEKS,
                   dest="max_weeks")
    args = p.parse_args()

    HIDDEN_DIM = args.hidden

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sequence_model] device: {device}")
    print(f"[sequence_model] max_weeks={args.max_weeks} hidden={args.hidden} "
          f"epochs={args.epochs}")

    print("[sequence_model] building per-user weekly sequences...")
    X_list, mask_list, y, authors = build_all_sequences(args.max_weeks)
    n = len(y)
    counts = np.bincount(y, minlength=3)
    print(f"[sequence_model] n={n}  "
          f"crisis={counts[0]} recovery={counts[1]} neither={counts[2]}")

    # Per-user feature z-score standardisation (important for LSTMs).
    # Compute mean/std on non-zero weeks only to avoid squashing signal
    # into zero masks.
    all_X = np.concatenate([x[m > 0.5] for x, m in zip(X_list, mask_list)],
                           axis=0)
    if len(all_X) == 0:
        raise SystemExit("[sequence_model] no active weeks found -- aborting")
    mu, sd = all_X.mean(axis=0), all_X.std(axis=0) + 1e-6
    X_list = [((x - mu) / sd) * m[:, None] for x, m in zip(X_list, mask_list)]

    print("[sequence_model] running 5-fold stratified CV...")
    results = cross_validate(X_list, mask_list, y, args.epochs, device)

    print("\n===== Sequence model (BiLSTM + attention) =====")
    print(f"  per-fold macro AUC: {results['fold_aucs']}")
    print(f"  mean fold AUC:      {np.mean(results['fold_aucs']):.4f}")
    print(f"  pooled macro AUC:   {results['macro_auc']:.4f}")
    print(f"  pooled macro F1:    {results['macro_f1']:.4f}")
    print(f"  per-class AUC:      {results['per_class_auc']}")
    print(f"  per-class F1:       {results['per_class_f1']}")

    payload = {
        "n_users":     int(n),
        "max_weeks":   args.max_weeks,
        "hidden_dim":  args.hidden,
        "epochs":      args.epochs,
        "results":     results,
    }
    RESULTS_OUT.write_text(json.dumps(payload, indent=2))
    print(f"\n[sequence_model] saved to {RESULTS_OUT}")


if __name__ == "__main__":
    main()
