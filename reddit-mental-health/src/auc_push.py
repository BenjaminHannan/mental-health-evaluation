"""Focused post-verification push: try to lift the cat+tfidf stack further.

auc_verify.py established that the honest winner is
    Stack: Optuna-CatBoost + TF-IDF LR  →  0.7304 ± 0.0108  (5-seed)
vs. v1 CatBoost at 0.7174 ± 0.0099 (a real +0.013 lift).

This script asks whether two well-motivated, cheap additions lift it
further under honest CV:

  (A) Add RandomForest on the 121-feature tabular matrix as a 3rd base,
      and LR-stack cat + tfidf + rf.  RF and CatBoost have very different
      bias/variance profiles, so their OOF probas may be diversifying.
  (B) Tune the meta-learner's L2 strength (C ∈ {0.1, 0.3, 1, 3, 10}) on
      the winning 2-base stack via inner 5-fold CV *per outer fold*.
      Guards against a bad default dominating the result.

For each 5-seed repeat we write per-seed AUC, mean, std, and the min/max
so the paper can cite stable numbers. Output lands alongside the
previous verify results.

Run
---
    python src/auc_push.py                # default: 5 seeds, GPU
    python src/auc_push.py --n-seeds 10   # tighter CI
    python src/auc_push.py --no-gpu       # CPU CatBoost (slow)
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
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
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

RESULTS_OUT = DATA_DIR / "auc_push_results.json"


def _catboost_factory(use_gpu: bool, params: dict, seed: int):
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


def _tfidf_factory():
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
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )),
    ])


def _meta_lr_factory(C: float = 1.0):
    return lambda: LogisticRegression(
        penalty="l2", C=C, max_iter=2000,
        class_weight="balanced", solver="lbfgs")


def _meta_lrcv_factory():
    """LogisticRegressionCV picks C per outer fold via inner 5-fold CV."""
    return lambda: LogisticRegressionCV(
        Cs=[0.1, 0.3, 1.0, 3.0, 10.0],
        cv=5, penalty="l2", max_iter=2000,
        class_weight="balanced", solver="lbfgs",
        scoring="roc_auc_ovr_weighted",
    )


def _summarize(label: str, aucs: list[float]) -> dict:
    arr = np.asarray(aucs)
    s = {
        "aucs":  [float(a) for a in aucs],
        "mean":  float(arr.mean()),
        "std":   float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min":   float(arr.min()),
        "max":   float(arr.max()),
    }
    print(f"  {label:38s}  mean={s['mean']:.4f}  std={s['std']:.4f}  "
          f"[{s['min']:.4f}, {s['max']:.4f}]", flush=True)
    return s


def score_stack(bases_oof: list[np.ndarray], y: np.ndarray,
                meta_factory, seed: int) -> float:
    X_meta = np.hstack(bases_oof)
    oof = stratified_oof_proba(meta_factory, X_meta, y, seed=seed)
    return macro_ovr_auc(y, oof, len(np.unique(y)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()

    use_gpu = not args.no_gpu
    seeds = list(range(1, args.n_seeds + 1))
    print(f"[auc_push] GPU={'ON' if use_gpu else 'OFF'}  seeds={seeds}",
          flush=True)

    # Load
    print("[auc_push] loading features...", flush=True)
    df = load_all_features()
    df = build_presence_flags(df)
    feat_cols = discover_feature_cols(df)
    X, y, le = prepare_dataset(df, high_confidence_only=False,
                               feature_cols=feat_cols)
    authors = df.loc[X.index, "author"].tolist()
    v2 = json.loads(V2_RESULTS.read_text())
    optuna_params = v2["phase_h"]["best_params"]
    print(f"[auc_push] n={len(y)}  features={X.shape[1]}", flush=True)
    print(f"[auc_push] Optuna CatBoost params = {optuna_params}", flush=True)

    docs  = build_user_text(authors)

    # Per-seed base OOFs cached, so we can reuse them across meta configs.
    print("\n[auc_push] computing per-seed base OOF probas (cat, tfidf, rf)...",
          flush=True)

    cat_oofs:   dict[int, np.ndarray] = {}
    tfidf_oofs: dict[int, np.ndarray] = {}
    rf_oofs:    dict[int, np.ndarray] = {}
    for s in seeds:
        print(f"  seed={s}", flush=True)
        t0 = time.time()
        cat_oofs[s] = stratified_oof_proba(
            _catboost_factory(use_gpu, optuna_params, s), X, y, seed=s)
        t_cat = time.time() - t0
        t0 = time.time()
        tfidf_oofs[s] = stratified_oof_proba(
            _tfidf_factory(), np.array(docs, dtype=object), y, seed=s)
        t_tfidf = time.time() - t0
        t0 = time.time()
        rf_oofs[s] = stratified_oof_proba(
            _rf_factory(s), X, y, seed=s)
        t_rf = time.time() - t0
        print(f"    cat={t_cat:.1f}s  tfidf={t_tfidf:.1f}s  rf={t_rf:.1f}s",
              flush=True)

    # Now score many stacking configurations using the cached OOFs.
    n_classes = len(np.unique(y))
    results: dict = {
        "n":             int(len(y)),
        "n_features":    int(X.shape[1]),
        "classes":       list(le.classes_),
        "class_counts":  {c: int((y == i).sum()) for i, c in enumerate(le.classes_)},
        "seeds":         seeds,
        "gpu_enabled":   use_gpu,
        "optuna_params": optuna_params,
    }

    configs: list[tuple[str, list[str], object]] = [
        # (name, base_key_list, meta_factory)
        ("cat-alone",                 ["cat"],                _meta_lr_factory(1.0)),
        ("tfidf-alone",               ["tfidf"],              _meta_lr_factory(1.0)),
        ("rf-alone",                  ["rf"],                 _meta_lr_factory(1.0)),
        ("stack cat+tfidf (baseline)", ["cat", "tfidf"],      _meta_lr_factory(1.0)),
        ("stack cat+tfidf  (meta C=0.3)",  ["cat", "tfidf"],  _meta_lr_factory(0.3)),
        ("stack cat+tfidf  (meta C=3.0)",  ["cat", "tfidf"],  _meta_lr_factory(3.0)),
        ("stack cat+tfidf  (meta LRCV)",   ["cat", "tfidf"],  _meta_lrcv_factory()),
        ("stack cat+rf",              ["cat", "rf"],          _meta_lr_factory(1.0)),
        ("stack tfidf+rf",            ["tfidf", "rf"],        _meta_lr_factory(1.0)),
        ("stack cat+tfidf+rf",        ["cat", "tfidf", "rf"], _meta_lr_factory(1.0)),
        ("stack cat+tfidf+rf (LRCV)", ["cat", "tfidf", "rf"], _meta_lrcv_factory()),
    ]

    oof_lookup = {"cat": cat_oofs, "tfidf": tfidf_oofs, "rf": rf_oofs}

    print("\n[auc_push] scoring configs across all seeds...", flush=True)
    for name, keys, meta_factory in configs:
        aucs: list[float] = []
        for s in seeds:
            if len(keys) == 1:
                # For "alone" rows, the "stack" is trivial: just eval base OOF.
                oof = oof_lookup[keys[0]][s]
                aucs.append(macro_ovr_auc(y, oof, n_classes))
            else:
                bases = [oof_lookup[k][s] for k in keys]
                aucs.append(score_stack(bases, y, meta_factory, seed=s))
        results[name] = _summarize(name, aucs)

    # Summary table
    print("\n" + "=" * 76)
    print("  AUC PUSH — honest 5-seed scores (n=505, macro OvR AUC)")
    print("=" * 76)
    print("  v1 CatBoost reference             0.7174 ± 0.0099")
    print("  verify Stack cat+tfidf reference  0.7304 ± 0.0108")
    print("-" * 76)
    # Reprint each config in table form, sorted by mean desc
    ranked = sorted(
        [(name, results[name]) for name, _, _ in configs],
        key=lambda kv: kv[1]["mean"], reverse=True,
    )
    for name, s in ranked:
        print(f"  {name:40s}  {s['mean']:.4f} ± {s['std']:.4f}  "
              f"[{s['min']:.4f}, {s['max']:.4f}]")
    print("=" * 76)

    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n[auc_push] wrote {RESULTS_OUT}", flush=True)


if __name__ == "__main__":
    main()
