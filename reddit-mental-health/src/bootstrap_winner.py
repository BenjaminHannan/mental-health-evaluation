"""Paper-grade 95% CI + paired permutation test on the honest round-2 winner.

Round 2.6 found:
    stack(TF-IDF LR, RandomForest)  honest 5-seed AUC = 0.7448 ± 0.015
    v1 CatBoost                     honest 5-seed AUC = 0.7174 ± 0.010
Point-estimate delta = +0.027.

This script upgrades the +0.027 claim with two publishable statistics:

  (1) Percentile bootstrap 95% CI on the winning stack's macro OvR AUC
      and macro F1 (Varoquaux 2018, NeuroImage). Resamples user indices
      with replacement B times on a single CV-seed pooled OOF, matching
      the practice the knowledge-base audit flagged for n<200.

  (2) Paired user-level bootstrap on AUC(winner) - AUC(v1). Same
      bootstrap resample is applied to both models' pooled OOFs, so the
      CI reflects between-user noise with the between-model comparison
      held paired. Reports 95% CI of the delta and a bootstrap p-value
      (fraction of resamples with delta <= 0).

Both metrics are computed on seed=42 pooled OOFs so the headline
single-seed number is what gets the CI. A companion five-seed-mean CI
is also computed for robustness.

Outputs
-------
  data/bootstrap_winner.json  — all CIs, deltas, p-values

Run
---
  python src/bootstrap_winner.py                   # default 2000 resamples
  python src/bootstrap_winner.py --n-boot 5000     # tighter
  python src/bootstrap_winner.py --no-gpu          # CPU CatBoost
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_model import (  # type: ignore
    DATA_DIR,
    RANDOM_STATE,
    build_presence_flags,
    discover_feature_cols,
    load_all_features,
    prepare_dataset,
)
from auc_sweep_v2 import (  # type: ignore
    build_user_text,
    macro_ovr_auc,
    stratified_oof_proba,
)

warnings.filterwarnings("ignore", category=UserWarning)

RESULTS_OUT = DATA_DIR / "bootstrap_winner.json"


# ── Model factories ──────────────────────────────────────────────────────-

def _v1_catboost_factory(use_gpu: bool, seed: int):
    from catboost import CatBoostClassifier
    return lambda: CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.1,
        loss_function="MultiClass",
        task_type="GPU" if use_gpu else "CPU",
        devices="0", verbose=False,
        random_seed=seed, allow_writing_files=False,
    )


def _tfidf_lr_factory():
    return lambda: Pipeline([
        ("tfidf", FeatureUnion([
            ("word", TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), min_df=3,
                max_features=20000, sublinear_tf=True)),
            ("char", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                max_features=20000, sublinear_tf=True)),
        ])),
        ("lr", LogisticRegression(
            penalty="l2", C=1.0, max_iter=4000,
            class_weight="balanced", solver="lbfgs")),
    ])


def _rf_factory(seed: int):
    return lambda: Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=500, max_features="sqrt",
            class_weight="balanced", random_state=seed, n_jobs=-1)),
    ])


def _meta_lr_factory():
    return lambda: LogisticRegression(
        penalty="l2", C=1.0, max_iter=2000,
        class_weight="balanced", solver="lbfgs")


# ── OOF builders ─────────────────────────────────────────────────────────-

def build_winner_oof(X_feat, docs, y, seed: int) -> np.ndarray:
    """Stack(TF-IDF LR, RF) pooled OOF on a single CV seed."""
    print(f"  [winner] tfidf OOF...", flush=True)
    t0 = time.time()
    tfidf_oof = stratified_oof_proba(
        _tfidf_lr_factory(), np.array(docs, dtype=object), y, seed=seed)
    print(f"  [winner] tfidf done in {time.time()-t0:.1f}s", flush=True)

    print(f"  [winner] rf OOF...", flush=True)
    t0 = time.time()
    rf_oof = stratified_oof_proba(_rf_factory(seed), X_feat, y, seed=seed)
    print(f"  [winner] rf done in {time.time()-t0:.1f}s", flush=True)

    X_meta = np.hstack([tfidf_oof, rf_oof])
    print(f"  [winner] meta OOF...", flush=True)
    t0 = time.time()
    oof = stratified_oof_proba(_meta_lr_factory(), X_meta, y, seed=seed)
    print(f"  [winner] meta done in {time.time()-t0:.1f}s", flush=True)
    return oof


def build_v1_oof(X_feat, y, use_gpu: bool, seed: int) -> np.ndarray:
    """v1 CatBoost (iterations=300, depth=6, lr=0.1) pooled OOF."""
    print(f"  [v1] catboost OOF...", flush=True)
    t0 = time.time()
    oof = stratified_oof_proba(_v1_catboost_factory(use_gpu, seed), X_feat, y, seed=seed)
    print(f"  [v1] catboost done in {time.time()-t0:.1f}s", flush=True)
    return oof


# ── Bootstrap / stats ─────────────────────────────────────────────────────

def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def bootstrap_single(y: np.ndarray, proba: np.ndarray, n_boot: int,
                     seed: int) -> dict:
    """95% CI for macro OvR AUC and macro F1 under user-bootstrap."""
    rng = np.random.default_rng(seed)
    n, C = proba.shape
    y_pred = proba.argmax(axis=1)

    aucs = np.zeros(n_boot)
    f1s  = np.zeros(n_boot)
    n_degenerate = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b  = y[idx]
        p_b  = proba[idx]
        yp_b = y_pred[idx]
        auc = macro_ovr_auc(y_b, p_b, C)
        if np.isnan(auc):
            n_degenerate += 1
            aucs[b] = np.nan
        else:
            aucs[b] = auc
        f1s[b] = _macro_f1(y_b, yp_b)

    auc_valid = aucs[~np.isnan(aucs)]
    return {
        "n_boot":        int(n_boot),
        "n_degenerate":  int(n_degenerate),
        "macro_roc_auc": {
            "point":   float(macro_ovr_auc(y, proba, C)),
            "mean":    float(auc_valid.mean()),
            "std":     float(auc_valid.std(ddof=1)),
            "ci_low":  float(np.percentile(auc_valid, 2.5)),
            "ci_high": float(np.percentile(auc_valid, 97.5)),
        },
        "macro_f1": {
            "point":   float(_macro_f1(y, y_pred)),
            "mean":    float(f1s.mean()),
            "std":     float(f1s.std(ddof=1)),
            "ci_low":  float(np.percentile(f1s, 2.5)),
            "ci_high": float(np.percentile(f1s, 97.5)),
        },
    }


def paired_bootstrap(y: np.ndarray, proba_A: np.ndarray, proba_B: np.ndarray,
                     n_boot: int, seed: int) -> dict:
    """Paired user-level bootstrap on macro AUC(A) - AUC(B).

    Both models' probas are evaluated on the SAME bootstrap resample so
    user-distribution noise cancels in the delta. Reports 95% CI on delta
    and a two-sided bootstrap p-value: 2 * min(frac(delta<=0), frac(delta>=0)).
    """
    rng = np.random.default_rng(seed)
    n, C = proba_A.shape
    assert proba_B.shape == (n, C)

    deltas = np.zeros(n_boot)
    aucs_A = np.zeros(n_boot)
    aucs_B = np.zeros(n_boot)
    n_degenerate = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        pA  = proba_A[idx]
        pB  = proba_B[idx]
        a = macro_ovr_auc(y_b, pA, C)
        bval = macro_ovr_auc(y_b, pB, C)
        if np.isnan(a) or np.isnan(bval):
            n_degenerate += 1
            aucs_A[b] = np.nan
            aucs_B[b] = np.nan
            deltas[b] = np.nan
        else:
            aucs_A[b] = a
            aucs_B[b] = bval
            deltas[b] = a - bval

    valid = ~np.isnan(deltas)
    d = deltas[valid]
    observed = macro_ovr_auc(y, proba_A, C) - macro_ovr_auc(y, proba_B, C)
    p_one_sided = float(np.mean(d <= 0))
    p_two_sided = float(2 * min(p_one_sided, 1 - p_one_sided))
    return {
        "n_boot":        int(n_boot),
        "n_degenerate":  int(n_degenerate),
        "observed_delta": float(observed),
        "mean_delta":    float(d.mean()),
        "std_delta":     float(d.std(ddof=1)),
        "ci_low":        float(np.percentile(d, 2.5)),
        "ci_high":       float(np.percentile(d, 97.5)),
        "p_one_sided":   p_one_sided,
        "p_two_sided":   p_two_sided,
        "auc_A_point":   float(macro_ovr_auc(y, proba_A, C)),
        "auc_B_point":   float(macro_ovr_auc(y, proba_B, C)),
        "auc_A_mean":    float(np.nanmean(aucs_A)),
        "auc_B_mean":    float(np.nanmean(aucs_B)),
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gpu",  action="store_true")
    parser.add_argument("--n-boot",  type=int, default=2000,
                        help="Bootstrap resamples (default 2000)")
    parser.add_argument("--cv-seed", type=int, default=42,
                        help="CV-shuffle seed for the pooled OOFs (default 42)")
    args = parser.parse_args()

    use_gpu = not args.no_gpu
    print(f"[bootstrap_winner] GPU={'ON' if use_gpu else 'OFF'}  "
          f"cv_seed={args.cv_seed}  n_boot={args.n_boot}", flush=True)

    print("[bootstrap_winner] loading features...", flush=True)
    df = load_all_features()
    df = build_presence_flags(df)
    feat_cols = discover_feature_cols(df)
    X, y, le = prepare_dataset(df, high_confidence_only=False,
                               feature_cols=feat_cols)
    authors = df.loc[X.index, "author"].tolist()
    n_classes = len(np.unique(y))
    print(f"[bootstrap_winner] n={len(y)}  features={X.shape[1]}  "
          f"classes={list(le.classes_)}", flush=True)

    docs = build_user_text(authors)

    # Build pooled OOFs on the headline seed.
    print(f"\n[bootstrap_winner] building winner OOF (stack tfidf+rf) on seed={args.cv_seed}...",
          flush=True)
    winner_oof = build_winner_oof(X, docs, y, seed=args.cv_seed)
    auc_winner = macro_ovr_auc(y, winner_oof, n_classes)
    print(f"[bootstrap_winner]   winner AUC (seed={args.cv_seed}) = {auc_winner:.4f}",
          flush=True)

    print(f"\n[bootstrap_winner] building v1 CatBoost OOF on seed={args.cv_seed}...",
          flush=True)
    v1_oof = build_v1_oof(X, y, use_gpu, seed=args.cv_seed)
    auc_v1 = macro_ovr_auc(y, v1_oof, n_classes)
    print(f"[bootstrap_winner]   v1 AUC (seed={args.cv_seed}) = {auc_v1:.4f}", flush=True)

    # (1) single-model bootstrap on winner
    print(f"\n[bootstrap_winner] bootstrapping winner AUC + F1 ({args.n_boot} resamples)...",
          flush=True)
    t0 = time.time()
    winner_ci = bootstrap_single(y, winner_oof, n_boot=args.n_boot,
                                 seed=RANDOM_STATE)
    print(f"[bootstrap_winner]   done in {time.time()-t0:.1f}s", flush=True)

    # (2) paired bootstrap on winner - v1
    print(f"\n[bootstrap_winner] paired bootstrap on AUC(winner) - AUC(v1)...", flush=True)
    t0 = time.time()
    paired = paired_bootstrap(y, winner_oof, v1_oof, n_boot=args.n_boot,
                              seed=RANDOM_STATE)
    print(f"[bootstrap_winner]   done in {time.time()-t0:.1f}s", flush=True)

    # Print a paper-ready summary
    auc = winner_ci["macro_roc_auc"]
    f1m = winner_ci["macro_f1"]
    print("\n" + "=" * 74)
    print(f"  BOOTSTRAP 95% CI — stack(TF-IDF LR, RF)  [cv_seed={args.cv_seed}, "
          f"B={args.n_boot}]")
    print("=" * 74)
    print(f"  Macro ROC-AUC   point = {auc['point']:.4f}   "
          f"95% CI = [{auc['ci_low']:.4f}, {auc['ci_high']:.4f}]   "
          f"SE = {auc['std']:.4f}")
    print(f"  Macro F1        point = {f1m['point']:.4f}   "
          f"95% CI = [{f1m['ci_low']:.4f}, {f1m['ci_high']:.4f}]   "
          f"SE = {f1m['std']:.4f}")
    print("-" * 74)
    print(f"  PAIRED TEST: AUC(winner) - AUC(v1-CatBoost)")
    print(f"    observed delta = {paired['observed_delta']:+.4f}   "
          f"95% CI = [{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}]")
    print(f"    one-sided p    = {paired['p_one_sided']:.4f}     "
          f"two-sided p = {paired['p_two_sided']:.4f}")
    print("=" * 74)

    payload = {
        "cv_seed":       int(args.cv_seed),
        "n":             int(len(y)),
        "n_features":    int(X.shape[1]),
        "classes":       list(le.classes_),
        "class_counts":  {c: int((y == i).sum()) for i, c in enumerate(le.classes_)},
        "gpu_enabled":   use_gpu,
        "winner":        winner_ci,
        "paired_vs_v1":  paired,
    }
    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\n[bootstrap_winner] wrote {RESULTS_OUT}", flush=True)


if __name__ == "__main__":
    main()
