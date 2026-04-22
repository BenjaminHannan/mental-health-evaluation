"""Honest 5-seed verification of the round-2 sweep's single-seed claims.

The v2 sweep (seed=42) reported:
    Phase H  Optuna CatBoost  = 0.7464
    Phase I  stacked LR       = 0.7594   (+0.024 over v1's 0.7355)
But the 5-seed run of the full stack (Phase K) collapsed to 0.7137 ± 0.008,
actually *below* v1's honest 5-seed CatBoost number of 0.7181 ± 0.008.

So Phase I's +0.024 lift is CV-split-lucky. This script asks the next
question: is Phase H's tuned CatBoost a real improvement under honest CV,
or is it also split-lucky? Same for intermediate stacks that drop the
weak MentalBERT-embedding component.

We run 5 seeds (matching v1 Phase E + v2 Phase K) over:
    (1) v1 default CatBoost  (iterations=300, depth=6, lr=0.1)
    (2) Optuna-tuned CatBoost (v2 Phase H best params)
    (3) Stack: CatBoost (Optuna) + TF-IDF LR          — drops the weak emb channel
    (4) Stack: CatBoost (Optuna) + TF-IDF LR + emb LR — the full v2 Phase I

so the table cleanly shows whether each added complexity buys honest AUC.

Output
------
  data/auc_verify_results.json   — mean, std, per-seed AUC for each config

Run
---
  python src/auc_verify.py                 # default: 5 seeds, GPU
  python src/auc_verify.py --n-seeds 10    # more seeds for tighter CI
  python src/auc_verify.py --no-gpu        # CPU CatBoost (slow)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_model import (  # type: ignore
    DATA_DIR,
    RANDOM_STATE,
    N_FOLDS,
    build_presence_flags,
    discover_feature_cols,
    load_all_features,
    prepare_dataset,
)
from auc_sweep_v2 import (  # type: ignore
    RESULTS_OUT as V2_RESULTS,
    build_user_embedding_matrix,
    build_user_text,
    macro_ovr_auc,
    stratified_oof_proba,
)

warnings.filterwarnings("ignore", category=UserWarning)

RESULTS_OUT = DATA_DIR / "auc_verify_results.json"


# ── Estimator factories ──────────────────────────────────────────────────-

def _catboost_factory(use_gpu: bool, params: dict, seed: int) -> Callable:
    from catboost import CatBoostClassifier
    full = {
        **params,
        "loss_function": "MultiClass",
        "task_type": "GPU" if use_gpu else "CPU",
        "devices": "0",
        "verbose": False,
        "random_seed": seed,
        "allow_writing_files": False,
    }
    return lambda: CatBoostClassifier(**full)


def _tfidf_factory() -> Callable:
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


def _emb_lr_factory() -> Callable:
    return lambda: Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="l2", C=1.0, max_iter=2000,
            class_weight="balanced", solver="lbfgs")),
    ])


def _meta_factory() -> Callable:
    return lambda: LogisticRegression(
        penalty="l2", C=1.0, max_iter=2000,
        class_weight="balanced", solver="lbfgs")


# ── Per-config scoring across seeds ──────────────────────────────────────-

def score_catboost(X, y, use_gpu: bool, params: dict,
                   seeds: list[int]) -> list[float]:
    aucs = []
    for s in seeds:
        make = _catboost_factory(use_gpu, params, s)
        oof = stratified_oof_proba(make, X, y, seed=s)
        auc = macro_ovr_auc(y, oof, len(np.unique(y)))
        aucs.append(auc)
        print(f"    seed={s}  AUC={auc:.4f}", flush=True)
    return aucs


def score_stack(X_feat, docs, X_emb, y, use_gpu: bool, cat_params: dict,
                seeds: list[int], use_emb: bool = True) -> list[float]:
    """5-seed stacked-LR score.  use_emb=False drops the MentalBERT channel."""
    aucs = []
    for s in seeds:
        cat_oof = stratified_oof_proba(
            _catboost_factory(use_gpu, cat_params, s), X_feat, y, seed=s)
        tfidf_oof = stratified_oof_proba(
            _tfidf_factory(), np.array(docs, dtype=object), y, seed=s)
        if use_emb:
            emb_oof = stratified_oof_proba(_emb_lr_factory(), X_emb, y, seed=s)
            X_meta = np.hstack([cat_oof, tfidf_oof, emb_oof])
        else:
            X_meta = np.hstack([cat_oof, tfidf_oof])
        oof_meta = stratified_oof_proba(_meta_factory(), X_meta, y, seed=s)
        auc = macro_ovr_auc(y, oof_meta, len(np.unique(y)))
        aucs.append(auc)
        tag = "cat+tfidf+emb" if use_emb else "cat+tfidf"
        print(f"    seed={s}  {tag} AUC={auc:.4f}", flush=True)
    return aucs


def _summarize(label: str, aucs: list[float]) -> dict:
    arr = np.asarray(aucs)
    s = {
        "aucs":  [float(a) for a in aucs],
        "mean":  float(arr.mean()),
        "std":   float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min":   float(arr.min()),
        "max":   float(arr.max()),
    }
    print(f"  {label:30s}  mean={s['mean']:.4f}  std={s['std']:.4f}  "
          f"[{s['min']:.4f}, {s['max']:.4f}]")
    return s


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gpu", action="store_true",
                        help="Use CPU CatBoost (slow)")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Seeds to average across (default 5)")
    args = parser.parse_args()

    use_gpu = not args.no_gpu
    seeds = list(range(1, args.n_seeds + 1))
    print(f"[auc_verify] GPU={'ON' if use_gpu else 'OFF'}  seeds={seeds}", flush=True)

    # Load data once.
    print("[auc_verify] loading features...", flush=True)
    df = load_all_features()
    df = build_presence_flags(df)
    feat_cols = discover_feature_cols(df)
    X, y, le = prepare_dataset(df, high_confidence_only=False,
                               feature_cols=feat_cols)
    authors = df.loc[X.index, "author"].tolist()
    print(f"[auc_verify] n={len(y)}  features={X.shape[1]}  "
          f"classes={list(le.classes_)}", flush=True)

    # Pull Optuna best params from the v2 results.
    if not V2_RESULTS.exists():
        raise SystemExit(f"Missing {V2_RESULTS}; run src/auc_sweep_v2.py first.")
    v2 = json.loads(V2_RESULTS.read_text())
    optuna_params = v2["phase_h"]["best_params"]
    v1_params     = {"iterations": 300, "depth": 6, "learning_rate": 0.1}
    print(f"[auc_verify] v1 CatBoost params = {v1_params}", flush=True)
    print(f"[auc_verify] Optuna CatBoost params = {optuna_params}", flush=True)

    # Build text + embedding matrices once (cached for stacks).
    docs  = build_user_text(authors)
    X_emb = build_user_embedding_matrix(authors, mode="mean")

    results: dict = {
        "n":            int(len(y)),
        "n_features":   int(X.shape[1]),
        "classes":      list(le.classes_),
        "class_counts": {c: int((y == i).sum()) for i, c in enumerate(le.classes_)},
        "seeds":        seeds,
        "gpu_enabled":  use_gpu,
        "v1_params":    v1_params,
        "optuna_params": optuna_params,
    }

    # (1) v1 default CatBoost, 5 seeds
    print("\n=== (1) v1 default CatBoost, 5 seeds ===", flush=True)
    t0 = time.time()
    a1 = score_catboost(X, y, use_gpu, v1_params, seeds)
    results["v1_catboost"] = _summarize("v1 CatBoost", a1)
    results["v1_catboost"]["elapsed_sec"] = round(time.time() - t0, 1)

    # (2) Optuna-tuned CatBoost, 5 seeds
    print("\n=== (2) Optuna-tuned CatBoost, 5 seeds ===", flush=True)
    t0 = time.time()
    a2 = score_catboost(X, y, use_gpu, optuna_params, seeds)
    results["optuna_catboost"] = _summarize("Optuna CatBoost", a2)
    results["optuna_catboost"]["elapsed_sec"] = round(time.time() - t0, 1)

    # (3) Stack: Optuna CatBoost + TF-IDF LR (no emb)
    print("\n=== (3) Stack [Optuna-CatBoost + TF-IDF LR], 5 seeds ===", flush=True)
    t0 = time.time()
    a3 = score_stack(X, docs, X_emb, y, use_gpu, optuna_params, seeds, use_emb=False)
    results["stack_cat_tfidf"] = _summarize("Stack cat+tfidf", a3)
    results["stack_cat_tfidf"]["elapsed_sec"] = round(time.time() - t0, 1)

    # (4) Stack: Optuna CatBoost + TF-IDF LR + emb LR (full v2 Phase I)
    print("\n=== (4) Stack [Optuna-CatBoost + TF-IDF LR + MentalBERT-emb LR], 5 seeds ===",
          flush=True)
    t0 = time.time()
    a4 = score_stack(X, docs, X_emb, y, use_gpu, optuna_params, seeds, use_emb=True)
    results["stack_cat_tfidf_emb"] = _summarize("Stack cat+tfidf+emb", a4)
    results["stack_cat_tfidf_emb"]["elapsed_sec"] = round(time.time() - t0, 1)

    # Summary table
    v1_mean_ref = 0.7181   # from v1 Phase E
    v1_std_ref  = 0.0080
    print("\n" + "=" * 66)
    print("  HONEST 5-SEED VERIFICATION SUMMARY (n=505, macro OvR AUC)")
    print("=" * 66)
    fmt = "  {label:32s}  {mean:.4f} ± {std:.4f}  [{mn:.4f}, {mx:.4f}]"
    for key, label in [
        ("v1_catboost",        "v1 CatBoost (defaults)"),
        ("optuna_catboost",    "Optuna CatBoost"),
        ("stack_cat_tfidf",    "Stack: cat + tfidf"),
        ("stack_cat_tfidf_emb","Stack: cat + tfidf + emb"),
    ]:
        s = results[key]
        print(fmt.format(label=label, mean=s["mean"], std=s["std"],
                         mn=s["min"], mx=s["max"]))
    print("-" * 66)
    print(f"  (v1 Phase E reference               {v1_mean_ref:.4f} ± {v1_std_ref:.4f})")
    print("=" * 66)

    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n[auc_verify] wrote {RESULTS_OUT}", flush=True)


if __name__ == "__main__":
    main()
