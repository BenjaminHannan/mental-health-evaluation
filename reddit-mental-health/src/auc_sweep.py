"""AUC-maximization sweep on the kitchen-sink feature matrix.

Drops a series of increasingly powerful tabular models onto the same
kitchen-sink features (raw + z-norm + temporal + mentalbert embeddings,
127+ columns, 581 users) and reports macro one-vs-rest ROC-AUC from
5-fold stratified CV with pooled out-of-fold predictions.

Phases
------
A. Model sweep          : RF, HistGBM, XGBoost (GPU), LightGBM (GPU), CatBoost (GPU).
B. Grid-tune the winner : small grid so we don't overfit at n=581.
C. Feature selection    : drop near-zero variance + high-correlation pairs.
D. Stacking ensemble    : logistic-regression meta over RF + winner + LR base.
E. Multi-seed verify    : run the final pipeline with 5 seeds; report mean +/- std.

All metrics are macro one-vs-rest ROC-AUC on the "full" dataset (n=581)
to stay directly comparable with data/model_results_kitchen_sink.json.

Output: data/auc_sweep_results.json  with every phase's numbers.

Run:
    python src/auc_sweep.py                 # full pipeline
    python src/auc_sweep.py --phase A       # just the model sweep
    python src/auc_sweep.py --no-gpu        # disable GPU (for debugging)
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_model import (
    DATA_DIR,
    RANDOM_STATE,
    N_FOLDS,
    build_presence_flags,
    discover_feature_cols,
    load_all_features,
    prepare_dataset,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SWEEP_OUT = DATA_DIR / "auc_sweep_results.json"


# ── GPU-capable GBMs (built lazily so missing deps don't kill the script) ──

def _xgb(use_gpu: bool, **params) -> "object":
    from xgboost import XGBClassifier
    base = dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        device="cuda" if use_gpu else "cpu",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
    )
    base.update(params)
    return XGBClassifier(**base)


def _lgbm(use_gpu: bool, **params) -> "object":
    from lightgbm import LGBMClassifier
    base = dict(
        n_estimators=500,
        max_depth=-1,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multiclass",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=-1,
    )
    if use_gpu:
        # LightGBM 4.x: 'cuda' requires a CUDA build; 'gpu' is OpenCL.
        # Try cuda first; caller may catch and fall back.
        base["device"] = "cuda"
    base.update(params)
    return LGBMClassifier(**base)


def _catboost(use_gpu: bool, **params) -> "object":
    from catboost import CatBoostClassifier
    base = dict(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        task_type="GPU" if use_gpu else "CPU",
        devices="0",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )
    base.update(params)
    return CatBoostClassifier(**base)


# ── Core CV evaluation ────────────────────────────────────────────────────

def _macro_ovr_auc(y: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    aucs = []
    for i in range(n_classes):
        pos = (y == i).astype(int)
        if pos.sum() in (0, len(pos)):
            continue
        aucs.append(roc_auc_score(pos, proba[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def cv_macro_auc(
    estimator, X: pd.DataFrame, y: np.ndarray,
    seed: int = RANDOM_STATE, need_impute: bool = True,
) -> tuple[float, float]:
    """Run 5-fold CV and return (macro OvR AUC, elapsed seconds).

    Wraps the estimator in a pipeline: impute (median) + scale for LR +
    estimator. XGBoost / LightGBM / CatBoost / HGB natively handle NaN,
    so need_impute=False skips the imputer for them to preserve native
    missing-value handling.
    """
    steps = []
    if need_impute:
        steps.append(("imputer", SimpleImputer(strategy="median")))
    steps.append(("clf", estimator))
    pipe = Pipeline(steps)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    t0 = time.time()
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=1)
    elapsed = time.time() - t0
    auc = _macro_ovr_auc(y, proba, n_classes=len(np.unique(y)))
    return auc, elapsed


# ── Phase A: model sweep ──────────────────────────────────────────────────

def phase_a_model_sweep(X: pd.DataFrame, y: np.ndarray, use_gpu: bool) -> dict:
    """Run each candidate once at default-ish hyperparameters."""
    print("\n" + "=" * 72)
    print("  Phase A: Model sweep  (5-fold CV, kitchen-sink features)")
    print("=" * 72)
    results: dict[str, dict] = {}

    def _run(name: str, estimator, need_impute: bool) -> None:
        try:
            auc, elapsed = cv_macro_auc(estimator, X, y, need_impute=need_impute)
            print(f"  {name:<18} AUC={auc:.4f}   ({elapsed:.1f}s)")
            results[name] = {"auc": auc, "elapsed_sec": round(elapsed, 2)}
        except Exception as e:
            print(f"  {name:<18} FAILED: {type(e).__name__}: {e}")
            results[name] = {"error": f"{type(e).__name__}: {e}"}

    # Baseline RF (same config as train_model.make_rf)
    _run("RandomForest",
         RandomForestClassifier(
             n_estimators=300, max_depth=6, class_weight="balanced",
             random_state=RANDOM_STATE, n_jobs=-1,
         ),
         need_impute=True)

    # HistGBM (CPU only, but very fast and often strong)
    _run("HistGBM",
         HistGradientBoostingClassifier(
             max_iter=400, max_depth=6, learning_rate=0.05,
             class_weight="balanced", random_state=RANDOM_STATE,
         ),
         need_impute=False)

    # XGBoost GPU
    _run(f"XGBoost{'_GPU' if use_gpu else ''}",
         _xgb(use_gpu=use_gpu),
         need_impute=False)

    # LightGBM GPU (cuda build required); fall back to CPU if unavailable.
    try:
        _run(f"LightGBM{'_GPU' if use_gpu else ''}",
             _lgbm(use_gpu=use_gpu),
             need_impute=False)
    except Exception as e:
        if use_gpu:
            print(f"  LightGBM GPU failed: {e}; retrying on CPU...")
            _run("LightGBM_CPU", _lgbm(use_gpu=False), need_impute=False)

    # CatBoost GPU
    _run(f"CatBoost{'_GPU' if use_gpu else ''}",
         _catboost(use_gpu=use_gpu),
         need_impute=False)

    # Pick best by AUC
    ranked = sorted(
        ((k, v["auc"]) for k, v in results.items() if "auc" in v),
        key=lambda kv: kv[1], reverse=True,
    )
    best_name = ranked[0][0] if ranked else None
    print(f"\n  winner: {best_name}   AUC={ranked[0][1]:.4f}")
    return {"results": results, "winner": best_name}


# ── Phase B: grid tune the winner ─────────────────────────────────────────

def phase_b_tune_winner(
    winner_name: str, X: pd.DataFrame, y: np.ndarray, use_gpu: bool,
) -> dict:
    """Hand-rolled small grid search (keeps complexity down at n=581)."""
    print("\n" + "=" * 72)
    print(f"  Phase B: Grid tune  (winner={winner_name})")
    print("=" * 72)

    # Small grids — at n=581 large grids just chase noise.
    grids: dict[str, list[dict]] = {
        "CatBoost_GPU": [
            {"iterations": it, "depth": d, "learning_rate": lr}
            for it in (300, 500, 800)
            for d  in (4, 6, 8)
            for lr in (0.03, 0.05, 0.1)
        ],
        "CatBoost": [
            {"iterations": it, "depth": d, "learning_rate": lr}
            for it in (300, 500, 800)
            for d  in (4, 6, 8)
            for lr in (0.03, 0.05, 0.1)
        ],
        "XGBoost_GPU": [
            {"n_estimators": n, "max_depth": d, "learning_rate": lr}
            for n  in (200, 400, 800)
            for d  in (3, 4, 6)
            for lr in (0.03, 0.05, 0.1)
        ],
        "XGBoost": [
            {"n_estimators": n, "max_depth": d, "learning_rate": lr}
            for n  in (200, 400, 800)
            for d  in (3, 4, 6)
            for lr in (0.03, 0.05, 0.1)
        ],
        "LightGBM_GPU": [
            {"n_estimators": n, "num_leaves": nl, "learning_rate": lr}
            for n  in (300, 500, 800)
            for nl in (15, 31, 63)
            for lr in (0.03, 0.05, 0.1)
        ],
        "LightGBM_CPU": [
            {"n_estimators": n, "num_leaves": nl, "learning_rate": lr}
            for n  in (300, 500, 800)
            for nl in (15, 31, 63)
            for lr in (0.03, 0.05, 0.1)
        ],
        "HistGBM": [
            {"max_iter": n, "max_depth": d, "learning_rate": lr}
            for n  in (200, 400, 800)
            for d  in (3, 4, 6)
            for lr in (0.03, 0.05, 0.1)
        ],
        "RandomForest": [
            {"n_estimators": n, "max_depth": d}
            for n in (200, 400, 800)
            for d in (4, 6, 8, None)
        ],
    }

    grid = grids.get(winner_name, [])
    if not grid:
        print(f"  No grid defined for {winner_name}; returning unchanged.")
        return {"winner": winner_name, "best_params": None, "best_auc": None}

    def _build(params: dict):
        if winner_name.startswith("CatBoost"):
            return _catboost(use_gpu=use_gpu, **params), False
        if winner_name.startswith("XGBoost"):
            return _xgb(use_gpu=use_gpu, **params), False
        if winner_name.startswith("LightGBM"):
            return _lgbm(use_gpu=use_gpu and winner_name.endswith("GPU"), **params), False
        if winner_name == "HistGBM":
            return (HistGradientBoostingClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                **params,
            ), False)
        if winner_name == "RandomForest":
            return (RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                **params,
            ), True)
        raise ValueError(winner_name)

    runs: list[dict] = []
    best = {"auc": -np.inf, "params": None}
    t0 = time.time()
    for i, params in enumerate(grid, 1):
        estimator, need_impute = _build(params)
        auc, elapsed = cv_macro_auc(estimator, X, y, need_impute=need_impute)
        runs.append({"params": params, "auc": auc, "elapsed_sec": round(elapsed, 2)})
        marker = " *" if auc > best["auc"] else ""
        print(f"  [{i:>2}/{len(grid):>2}] AUC={auc:.4f}  {params}{marker}", flush=True)
        if auc > best["auc"]:
            best = {"auc": auc, "params": params}

    print(f"\n  best_auc={best['auc']:.4f}  best_params={best['params']}  "
          f"(grid elapsed {time.time()-t0:.0f}s)")
    return {"winner": winner_name, "best_params": best["params"],
            "best_auc": best["auc"], "runs": runs}


# ── Phase C: feature selection ────────────────────────────────────────────

def phase_c_feature_selection(
    winner_name: str, best_params: dict, X: pd.DataFrame, y: np.ndarray,
    use_gpu: bool,
) -> dict:
    """Two-stage filter: near-zero variance, then pairwise correlation > 0.95.

    Evaluates the tuned winner on the full feature matrix vs the filtered
    set. Reports which one wins.
    """
    print("\n" + "=" * 72)
    print("  Phase C: Feature selection  (variance + correlation filters)")
    print("=" * 72)

    # Drop near-zero variance (threshold 0.0 → constant columns only)
    vt = VarianceThreshold(threshold=1e-6)
    X_imp = X.fillna(X.median(numeric_only=True))
    vt.fit(X_imp)
    kept = X.columns[vt.get_support()].tolist()
    dropped_var = [c for c in X.columns if c not in kept]
    print(f"  dropped {len(dropped_var)} near-zero-variance features "
          f"({len(kept)} remain)")

    # Drop high-correlation pairs (|r| > 0.95), keep the one with higher variance
    Xk = X[kept].copy()
    corr = Xk.fillna(Xk.median(numeric_only=True)).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop: set[str] = set()
    for col in upper.columns:
        high = upper.index[upper[col] > 0.95].tolist()
        for other in high:
            # Keep the higher-variance of the pair
            if Xk[col].var() >= Xk[other].var():
                to_drop.add(other)
            else:
                to_drop.add(col)
    kept_after_corr = [c for c in kept if c not in to_drop]
    print(f"  dropped {len(to_drop)} highly-correlated features "
          f"({len(kept_after_corr)} remain)")

    def _build():
        if winner_name.startswith("CatBoost"):
            return _catboost(use_gpu=use_gpu, **(best_params or {})), False
        if winner_name.startswith("XGBoost"):
            return _xgb(use_gpu=use_gpu, **(best_params or {})), False
        if winner_name.startswith("LightGBM"):
            return _lgbm(use_gpu=use_gpu and winner_name.endswith("GPU"),
                         **(best_params or {})), False
        if winner_name == "HistGBM":
            return HistGradientBoostingClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE, **(best_params or {})), False
        if winner_name == "RandomForest":
            return RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE,
                n_jobs=-1, **(best_params or {})), True
        raise ValueError(winner_name)

    est, need_impute = _build()
    auc_full, _ = cv_macro_auc(est, X, y, need_impute=need_impute)
    est, need_impute = _build()
    auc_filt, _ = cv_macro_auc(est, X[kept_after_corr], y, need_impute=need_impute)
    print(f"\n  AUC(full {X.shape[1]:>3} feats) = {auc_full:.4f}")
    print(f"  AUC(filt {len(kept_after_corr):>3} feats) = {auc_filt:.4f}")

    winner = "filtered" if auc_filt > auc_full else "full"
    print(f"  winner: {winner}")
    return {
        "dropped_variance": dropped_var,
        "dropped_correlation": list(to_drop),
        "kept_features": kept_after_corr,
        "auc_full": auc_full,
        "auc_filtered": auc_filt,
        "winner": winner,
    }


# ── Phase D: stacking ensemble ────────────────────────────────────────────

def phase_d_stacking(
    winner_name: str, best_params: dict,
    X: pd.DataFrame, y: np.ndarray, use_gpu: bool,
) -> dict:
    """Stack LR + RF + tuned GBM with a logistic-regression meta learner."""
    print("\n" + "=" * 72)
    print("  Phase D: Stacking ensemble  (LR + RF + tuned-GBM -> LR meta)")
    print("=" * 72)

    def _winner_estimator():
        if winner_name.startswith("CatBoost"):
            return _catboost(use_gpu=use_gpu, **(best_params or {}))
        if winner_name.startswith("XGBoost"):
            return _xgb(use_gpu=use_gpu, **(best_params or {}))
        if winner_name.startswith("LightGBM"):
            return _lgbm(use_gpu=use_gpu and winner_name.endswith("GPU"),
                         **(best_params or {}))
        if winner_name == "HistGBM":
            return HistGradientBoostingClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE, **(best_params or {}))
        if winner_name == "RandomForest":
            return RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE,
                n_jobs=-1, **(best_params or {}))
        raise ValueError(winner_name)

    # Use inner-pipelines so LR gets scaled + imputed properly.
    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(
            max_iter=1000, class_weight="balanced", C=0.5,
            random_state=RANDOM_STATE,
        )),
    ])
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf",     RandomForestClassifier(
            n_estimators=300, max_depth=6, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])
    gbm = _winner_estimator()

    stack = StackingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("gbm", gbm)],
        final_estimator=LogisticRegression(
            max_iter=1000, C=1.0, random_state=RANDOM_STATE,
        ),
        stack_method="predict_proba",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
    )

    auc, elapsed = cv_macro_auc(stack, X, y, need_impute=False)
    print(f"  Stacking AUC = {auc:.4f}   ({elapsed:.1f}s)")
    return {"auc": auc, "elapsed_sec": round(elapsed, 2)}


# ── Phase E: multi-seed verification ──────────────────────────────────────

def phase_e_multiseed(
    winner_name: str, best_params: dict,
    X: pd.DataFrame, y: np.ndarray,
    use_gpu: bool, kept_features: list[str] | None,
    n_seeds: int = 5,
) -> dict:
    """Run the full tuned winner with multiple CV-split seeds."""
    print("\n" + "=" * 72)
    print(f"  Phase E: Multi-seed verification  ({n_seeds} seeds)")
    print("=" * 72)

    X_use = X[kept_features] if kept_features else X

    def _build():
        if winner_name.startswith("CatBoost"):
            return _catboost(use_gpu=use_gpu, **(best_params or {})), False
        if winner_name.startswith("XGBoost"):
            return _xgb(use_gpu=use_gpu, **(best_params or {})), False
        if winner_name.startswith("LightGBM"):
            return _lgbm(use_gpu=use_gpu and winner_name.endswith("GPU"),
                         **(best_params or {})), False
        if winner_name == "HistGBM":
            return HistGradientBoostingClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE, **(best_params or {})), False
        if winner_name == "RandomForest":
            return RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE,
                n_jobs=-1, **(best_params or {})), True
        raise ValueError(winner_name)

    aucs: list[float] = []
    for seed in range(1, n_seeds + 1):
        est, need_impute = _build()
        auc, _ = cv_macro_auc(est, X_use, y, seed=seed, need_impute=need_impute)
        aucs.append(auc)
        print(f"  seed={seed}  AUC={auc:.4f}")
    arr = np.asarray(aucs)
    print(f"\n  mean={arr.mean():.4f}  std={arr.std(ddof=1):.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}")
    return {"seeds": list(range(1, n_seeds + 1)), "aucs": aucs,
            "mean": float(arr.mean()), "std": float(arr.std(ddof=1))}


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=list("ABCDE") + ["all"], default="all",
                        help="Which phase to run (default: all)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration (CPU-only fallback).")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of seeds for Phase E (default 5)")
    args = parser.parse_args()
    use_gpu = not args.no_gpu

    print(f"[auc_sweep] GPU={'ON' if use_gpu else 'OFF'}")
    print("[auc_sweep] loading kitchen-sink features...")
    df = load_all_features()
    df = build_presence_flags(df)
    feature_cols = discover_feature_cols(df)
    X, y, le = prepare_dataset(df, high_confidence_only=False,
                               feature_cols=feature_cols)
    counts = {le.classes_[i]: int((y == i).sum()) for i in range(len(le.classes_))}
    print(f"[auc_sweep] n={len(y)} features={X.shape[1]} classes={counts}")

    summary: dict = {
        "n":          int(len(y)),
        "n_features": int(X.shape[1]),
        "classes":    list(le.classes_),
        "class_counts": counts,
        "gpu_enabled": use_gpu,
    }

    # Phase A
    a = phase_a_model_sweep(X, y, use_gpu)
    summary["phase_a"] = a
    winner_name = a["winner"]

    if args.phase == "A":
        _write(summary); return

    # Phase B
    b = phase_b_tune_winner(winner_name, X, y, use_gpu)
    summary["phase_b"] = b
    best_params = b.get("best_params") or {}

    if args.phase == "B":
        _write(summary); return

    # Phase C
    c = phase_c_feature_selection(winner_name, best_params, X, y, use_gpu)
    summary["phase_c"] = c
    kept_features = c["kept_features"] if c["winner"] == "filtered" else None

    if args.phase == "C":
        _write(summary); return

    # Phase D
    d = phase_d_stacking(winner_name, best_params, X, y, use_gpu)
    summary["phase_d"] = d

    if args.phase == "D":
        _write(summary); return

    # Phase E
    e = phase_e_multiseed(winner_name, best_params, X, y, use_gpu,
                          kept_features, n_seeds=args.n_seeds)
    summary["phase_e"] = e

    # Final ranking
    print("\n" + "=" * 72)
    print("  FINAL SUMMARY")
    print("=" * 72)
    print(f"  Baseline RF (from train_model): ~0.688")
    print(f"  Phase A winner                : {winner_name}  AUC={a['results'][winner_name]['auc']:.4f}")
    print(f"  Phase B tuned                 : AUC={b.get('best_auc', float('nan')):.4f}  params={b.get('best_params')}")
    print(f"  Phase C feature selection     : winner={c['winner']}  AUC_full={c['auc_full']:.4f}  AUC_filt={c['auc_filtered']:.4f}")
    print(f"  Phase D stacking              : AUC={d['auc']:.4f}")
    print(f"  Phase E multi-seed verify     : {e['mean']:.4f} +/- {e['std']:.4f}  ({args.n_seeds} seeds)")
    _write(summary)


def _write(summary: dict) -> None:
    with open(SWEEP_OUT, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"\n[auc_sweep] wrote {SWEEP_OUT}")


if __name__ == "__main__":
    main()
