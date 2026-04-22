"""Uncertainty quantification for the kitchen-sink Random Forest.

Adds two things the knowledge-base audit flagged as must-haves at n=132:
  1. Bootstrap 95% confidence intervals on macro ROC-AUC and macro F1
     (Varoquaux 2018, NeuroImage) — the spread across CV folds
     substantially underestimates true predictive uncertainty at small n.
  2. Label-permutation p-value on macro ROC-AUC (Ojala & Garriga 2010,
     JMLR) — non-parametric test of "is any of this above chance."

Strategy
--------
For speed and reproducibility we recompute pooled out-of-fold predictions
once (same 5-fold stratified CV as train_model.evaluate_cv), then:
  - Bootstrap: resample user indices with replacement B times, recompute
    metrics on each resample, report 2.5/97.5 percentiles.
  - Permutation: shuffle y, re-run the full 5-fold CV pipeline P times,
    report p = (1 + #{permuted_auc >= observed_auc}) / (1 + P).

Outputs
-------
  data/bootstrap_cis.json    — {model, features, point, ci_low, ci_high}
  data/permutation_test.json — {model, features, observed, null_mean, p}

Run
---
  python src/evaluate_uncertainty.py                 # bootstrap only (fast)
  python src/evaluate_uncertainty.py --permutation   # + permutation test (slow)
  python src/evaluate_uncertainty.py --n-boot 2000 --n-perm 500
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from train_model import (
    DATA_DIR,
    RANDOM_STATE,
    N_FOLDS,
    build_presence_flags,
    discover_feature_cols,
    load_all_features,
    make_rf,
    prepare_dataset,
)

warnings.filterwarnings("ignore", category=UserWarning)

BOOT_OUT = DATA_DIR / "bootstrap_cis.json"
PERM_OUT = DATA_DIR / "permutation_test.json"


# ── Metric helpers ────────────────────────────────────────────────────────

def _macro_ovr_auc(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> float:
    """Macro one-vs-rest ROC-AUC (skip classes with no positives in resample)."""
    aucs: list[float] = []
    for i in range(n_classes):
        pos = (y_true == i).astype(int)
        if pos.sum() == 0 or pos.sum() == len(pos):
            continue  # degenerate class in this resample
        aucs.append(roc_auc_score(pos, y_proba[:, i]))
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# ── OOF prediction pipeline ───────────────────────────────────────────────

def pooled_oof_predictions(
    X: pd.DataFrame, y: np.ndarray, random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_pred, y_proba) pooled across 5-fold stratified CV."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)
    pipeline = make_rf()
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, method="predict")
    y_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")
    return y_pred, y_proba


# ── Bootstrap CIs ─────────────────────────────────────────────────────────

def bootstrap_cis(
    y: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
    n_boot: int = 1000, seed: int = RANDOM_STATE,
) -> dict:
    """Percentile bootstrap over user indices.

    At each iteration we draw n user indices with replacement, compute
    macro OvR AUC and macro F1 on that resample, and record them.
    Returns point estimate, 95% CI (2.5/97.5 percentiles), and SE.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    n_classes = len(np.unique(y))

    aucs: list[float] = []
    f1s:  list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        p_b = y_pred[idx]
        pr_b = y_proba[idx]
        auc = _macro_ovr_auc(y_b, pr_b, n_classes)
        if not np.isnan(auc):
            aucs.append(auc)
        f1s.append(_macro_f1(y_b, p_b))

    auc_arr = np.asarray(aucs)
    f1_arr  = np.asarray(f1s)

    return {
        "n_boot":            n_boot,
        "macro_roc_auc": {
            "point":   float(_macro_ovr_auc(y, y_proba, n_classes)),
            "mean":    float(auc_arr.mean()),
            "se":      float(auc_arr.std(ddof=1)),
            "ci_low":  float(np.percentile(auc_arr, 2.5)),
            "ci_high": float(np.percentile(auc_arr, 97.5)),
        },
        "macro_f1": {
            "point":   float(_macro_f1(y, y_pred)),
            "mean":    float(f1_arr.mean()),
            "se":      float(f1_arr.std(ddof=1)),
            "ci_low":  float(np.percentile(f1_arr,  2.5)),
            "ci_high": float(np.percentile(f1_arr,  97.5)),
        },
    }


# ── Label-permutation test ────────────────────────────────────────────────

def permutation_test(
    X: pd.DataFrame, y: np.ndarray,
    observed_auc: float, n_perm: int = 1000, seed: int = RANDOM_STATE,
) -> dict:
    """Label-permutation p-value on macro OvR AUC.

    Each iteration shuffles y, reruns the full 5-fold CV, computes macro
    OvR AUC on the pooled OOF predictions. p = (1 + #{null >= obs}) / (1 + P).
    This is expensive: each permutation refits 5 RF models.
    """
    rng = np.random.default_rng(seed)
    n_classes = len(np.unique(y))
    null_aucs: list[float] = []
    t0 = time.time()
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        y_pred_p, y_proba_p = pooled_oof_predictions(X, y_perm, random_state=RANDOM_STATE)
        auc = _macro_ovr_auc(y_perm, y_proba_p, n_classes)
        if not np.isnan(auc):
            null_aucs.append(auc)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_perm - (i + 1))
            print(f"[perm] {i+1}/{n_perm}  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    null = np.asarray(null_aucs)
    p_val = (1 + int(np.sum(null >= observed_auc))) / (1 + len(null))
    return {
        "n_perm":       n_perm,
        "observed_auc": float(observed_auc),
        "null_mean":    float(null.mean()),
        "null_std":     float(null.std(ddof=1)),
        "null_ci_low":  float(np.percentile(null, 2.5)),
        "null_ci_high": float(np.percentile(null, 97.5)),
        "p_value":      float(p_val),
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot",      type=int, default=1000,
                        help="Number of bootstrap resamples (default 1000)")
    parser.add_argument("--n-perm",      type=int, default=1000,
                        help="Number of label permutations (default 1000)")
    parser.add_argument("--permutation", action="store_true",
                        help="Also run label-permutation test (slow)")
    parser.add_argument("--dataset",     type=str, default="full",
                        choices=["full", "high_conf"])
    args = parser.parse_args()

    print("[uncertainty] loading kitchen-sink features...")
    df = load_all_features()
    df = build_presence_flags(df)
    feature_cols = discover_feature_cols(df)
    X, y, le = prepare_dataset(
        df,
        high_confidence_only=(args.dataset == "high_conf"),
        feature_cols=feature_cols,
    )
    print(f"[uncertainty] n={len(y)} features={X.shape[1]} classes={list(le.classes_)}")

    print("[uncertainty] computing pooled OOF predictions (RF, 5-fold)...")
    t0 = time.time()
    y_pred, y_proba = pooled_oof_predictions(X, y)
    print(f"[uncertainty]   OOF predictions in {time.time()-t0:.1f}s")

    # Bootstrap
    print(f"[uncertainty] bootstrap (n_boot={args.n_boot})...")
    t0 = time.time()
    boot = bootstrap_cis(y, y_pred, y_proba, n_boot=args.n_boot)
    print(f"[uncertainty]   done in {time.time()-t0:.1f}s")

    auc_b = boot["macro_roc_auc"]
    f1_b  = boot["macro_f1"]
    print(f"\n  Macro ROC-AUC  = {auc_b['point']:.3f}   "
          f"95% CI [{auc_b['ci_low']:.3f}, {auc_b['ci_high']:.3f}]   "
          f"SE = {auc_b['se']:.3f}")
    print(f"  Macro F1       = {f1_b['point']:.3f}   "
          f"95% CI [{f1_b['ci_low']:.3f}, {f1_b['ci_high']:.3f}]   "
          f"SE = {f1_b['se']:.3f}\n")

    boot_payload = {
        "model":        "RandomForest",
        "features":     "kitchen_sink",
        "dataset":      args.dataset,
        "n":            int(len(y)),
        "n_features":   int(X.shape[1]),
        **boot,
    }
    with open(BOOT_OUT, "w") as fh:
        json.dump(boot_payload, fh, indent=2)
    print(f"[uncertainty] wrote {BOOT_OUT}")

    # Permutation (optional)
    if args.permutation:
        print(f"\n[uncertainty] permutation test (n_perm={args.n_perm}) — slow...")
        t0 = time.time()
        perm = permutation_test(X, y, boot["macro_roc_auc"]["point"], n_perm=args.n_perm)
        print(f"[uncertainty]   done in {time.time()-t0:.1f}s")
        print(f"\n  Observed macro AUC  = {perm['observed_auc']:.3f}")
        print(f"  Null mean (shuffled) = {perm['null_mean']:.3f}   "
              f"95% CI [{perm['null_ci_low']:.3f}, {perm['null_ci_high']:.3f}]")
        print(f"  p-value              = {perm['p_value']:.4f}\n")
        perm_payload = {
            "model":      "RandomForest",
            "features":   "kitchen_sink",
            "dataset":    args.dataset,
            "n":          int(len(y)),
            "n_features": int(X.shape[1]),
            **perm,
        }
        with open(PERM_OUT, "w") as fh:
            json.dump(perm_payload, fh, indent=2)
        print(f"[uncertainty] wrote {PERM_OUT}")


if __name__ == "__main__":
    main()
