"""Round-2 AUC maximization sweep.

Starting point: v1 sweep found CatBoost_GPU at macro-OvR ROC-AUC ≈ 0.7355
(seed=42 headline) / 0.7181 ± 0.008 (5-seed honest) on 505 users × 121
features. This script adds three new signal channels and stacks them:

  Phase F — MentalBERT-embedding classifier
      Load cached per-post [CLS] vectors (reddit-mental-health/data/
      mentalbert_embeddings.npz) → mean-pool per user windowed by
      baseline / pre_4w / pre_2w / pre_1w → 4×768 = 3072-dim features →
      LogisticRegression L2 with class_weight='balanced', standardized.
      5-fold stratified CV, pooled OOF probas.

  Phase G — TF-IDF text classifier
      Concatenate each labelled user's post titles+bodies into one
      document. Fit char_wb(3,5) and word(1,2) TF-IDF in a
      FeatureUnion, logistic regression with L2 and balanced weights,
      5-fold stratified CV.

  Phase H — Optuna-tuned CatBoost (GPU)
      Bayesian search over {iterations, depth, lr, l2_leaf_reg,
      bagging_temperature, border_count}. 50 trials, 5-fold OOF macro
      AUC objective. GPU task_type.

  Phase I — Stacking
      Logistic-regression meta-learner on the concatenation of OOF
      probas from (best Phase H CatBoost, Phase F embedding-LR,
      Phase G TF-IDF-LR). 5-fold stratified CV on the meta-features,
      so the stack AUC is honestly cross-validated.

  Phase J — Simple-average blend
      Average the three OOF proba matrices, report macro AUC. Sanity
      check vs. the learned stacker.

  Phase K — Multi-seed verification
      Re-run Phase I across 5 CV seeds to get mean ± std of the
      stacker. Matches v1 Phase E methodology so the numbers are
      directly comparable.

Outputs
-------
  data/auc_sweep_v2_results.json  — full phase-by-phase dump
  data/auc_sweep_v2_run.log       — stdout log if invoked via watcher

Run
---
  python src/auc_sweep_v2.py                    # full sweep
  python src/auc_sweep_v2.py --no-gpu           # CPU CatBoost (slow)
  python src/auc_sweep_v2.py --n-trials 30      # fewer Optuna trials
  python src/auc_sweep_v2.py --skip-optuna      # use v1's best CatBoost params
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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

warnings.filterwarnings("ignore", category=UserWarning)

RESULTS_OUT = DATA_DIR / "auc_sweep_v2_results.json"
EMB_NPZ     = DATA_DIR / "mentalbert_embeddings.npz"
RAW_POSTS   = DATA_DIR / "raw_posts.parquet"
LABELS_IN   = DATA_DIR / "user_labels.parquet"

WINDOW_WEEKS = 4
WINDOWS      = ["baseline", "pre_4w", "pre_2w", "pre_1w"]


# ── Metric ────────────────────────────────────────────────────────────────

def macro_ovr_auc(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> float:
    aucs: list[float] = []
    for i in range(n_classes):
        pos = (y_true == i).astype(int)
        if pos.sum() == 0 or pos.sum() == len(pos):
            continue
        aucs.append(roc_auc_score(pos, y_proba[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def stratified_oof_proba(estimator_factory, X, y, seed: int = RANDOM_STATE,
                         n_splits: int = N_FOLDS) -> np.ndarray:
    """Return pooled OOF predict_proba matching StratifiedKFold order.

    estimator_factory: callable -> fresh estimator each fold (avoids state leak).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_classes = len(np.unique(y))
    oof = np.zeros((len(y), n_classes), dtype=np.float64)
    for fold_idx, (tr, va) in enumerate(cv.split(np.zeros(len(y)), y)):
        est = estimator_factory()
        if sparse.issparse(X):
            Xtr, Xva = X[tr], X[va]
        elif isinstance(X, np.ndarray):
            Xtr, Xva = X[tr], X[va]
        else:  # pandas DataFrame
            Xtr, Xva = X.iloc[tr], X.iloc[va]
        est.fit(Xtr, y[tr])
        # Columns of predict_proba align with np.unique(y) because we're
        # using the global label encoding; train-fold classes cover all 3.
        proba = est.predict_proba(Xva)
        # Map fold's class order to the global order (0..n_classes-1)
        classes = est.classes_
        if len(classes) == n_classes and np.array_equal(classes, np.arange(n_classes)):
            oof[va] = proba
        else:
            # Fold missing a class — spread zeros
            for j, c in enumerate(classes):
                oof[va, int(c)] = proba[:, j]
    return oof


# ── Phase F: MentalBERT-embedding LR ──────────────────────────────────────

def _load_post_embeddings() -> dict[str, np.ndarray]:
    if not EMB_NPZ.exists():
        raise FileNotFoundError(
            f"Missing {EMB_NPZ}. Run `python src/extract_mentalbert.py` first."
        )
    arr = np.load(EMB_NPZ, allow_pickle=False)
    ids, vecs = arr["ids"], arr["vecs"]
    return {str(pid): vec for pid, vec in zip(ids, vecs)}


def _window_mask(ts: pd.Series, tp_date: pd.Timestamp, window: str) -> pd.Series:
    c4 = tp_date - pd.Timedelta(weeks=4)
    c2 = tp_date - pd.Timedelta(weeks=2)
    c1 = tp_date - pd.Timedelta(weeks=1)
    if window == "baseline":
        return ts < c4
    if window == "pre_4w":
        return (ts >= c4) & (ts < tp_date)
    if window == "pre_2w":
        return (ts >= c2) & (ts < tp_date)
    if window == "pre_1w":
        return (ts >= c1) & (ts < tp_date)
    raise ValueError(window)


def build_user_embedding_matrix(authors: list[str],
                                mode: str = "mean") -> np.ndarray:
    """For each labelled user, build an embedding row vector.

    mode="mean"   → single mean-pooled 768-dim vector over all user posts
    mode="window" → stacked 4-window (baseline/pre_4w/pre_2w/pre_1w) = 3072-dim
    """
    print(f"[phase_F] loading post embeddings (mode={mode})...", flush=True)
    emb = _load_post_embeddings()
    print(f"[phase_F]   loaded {len(emb)} post vectors", flush=True)
    posts = pd.read_parquet(RAW_POSTS, columns=["author", "id", "created_utc"])
    dim = next(iter(emb.values())).shape[0]  # 768

    if mode == "mean":
        out = np.zeros((len(authors), dim), dtype=np.float32)
        missing = 0
        grouped = posts.groupby("author")["id"].apply(list).to_dict()
        for i, author in enumerate(authors):
            ids = grouped.get(author, [])
            vecs = [emb[pid] for pid in ids if pid in emb]
            if vecs:
                out[i] = np.mean(np.stack(vecs), axis=0)
            else:
                missing += 1
        print(f"[phase_F]   user × 768 matrix: {out.shape}  "
              f"(users with no embeddings: {missing})", flush=True)
        return out

    # windowed mode (kept for ablation)
    labels = pd.read_parquet(LABELS_IN).set_index("author")
    posts["created_utc"] = pd.to_datetime(
        posts["created_utc"], unit="s", errors="coerce",
    )
    out = np.zeros((len(authors), dim * len(WINDOWS)), dtype=np.float32)
    missing_windows = 0
    for i, author in enumerate(authors):
        user_posts = posts[posts["author"] == author]
        tp_date = labels.loc[author, "tp_date"] if author in labels.index else pd.NaT
        if pd.isnull(tp_date):
            tp_date = user_posts["created_utc"].max()
        tp_date = pd.to_datetime(tp_date)
        row_vec = []
        for w in WINDOWS:
            mask = _window_mask(user_posts["created_utc"], tp_date, w)
            ids = user_posts.loc[mask, "id"].tolist()
            vecs = [emb[pid] for pid in ids if pid in emb]
            if vecs:
                row_vec.append(np.mean(np.stack(vecs), axis=0))
            else:
                row_vec.append(np.zeros(dim, dtype=np.float32))
                missing_windows += 1
        out[i] = np.concatenate(row_vec)
    print(f"[phase_F]   user × (4×768) matrix: {out.shape}  "
          f"(missing windows zero-padded: {missing_windows})", flush=True)
    return out


def phase_F(authors: list[str], y: np.ndarray, seed: int = RANDOM_STATE) -> dict:
    X_emb = build_user_embedding_matrix(authors)
    t0 = time.time()

    def _make():
        return Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(
                penalty="l2", C=1.0, max_iter=2000,
                class_weight="balanced", solver="lbfgs", n_jobs=None,
            )),
        ])

    oof = stratified_oof_proba(_make, X_emb, y, seed=seed)
    auc = macro_ovr_auc(y, oof, len(np.unique(y)))
    print(f"[phase_F] MentalBERT-emb LR AUC = {auc:.4f}  ({time.time()-t0:.1f}s)",
          flush=True)
    return {"auc": auc, "oof_proba": oof, "X_shape": list(X_emb.shape)}


# ── Phase G: TF-IDF text classifier ───────────────────────────────────────

def build_user_text(authors: list[str]) -> list[str]:
    print("[phase_G] loading raw posts...", flush=True)
    posts = pd.read_parquet(RAW_POSTS, columns=["author", "title", "body"])
    posts["doc"] = (posts["title"].fillna("") + " "
                    + posts["body"].fillna("")).str.replace(r"\s+", " ", regex=True)
    grouped = posts.groupby("author")["doc"].apply(lambda s: " ".join(s.tolist()))
    docs = []
    for a in authors:
        docs.append(grouped.get(a, ""))
    print(f"[phase_G]   {len(docs)} user docs  (avg len chars: "
          f"{int(np.mean([len(d) for d in docs]))})", flush=True)
    return docs


def phase_G(authors: list[str], y: np.ndarray, seed: int = RANDOM_STATE) -> dict:
    docs = build_user_text(authors)
    t0 = time.time()

    def _make():
        return Pipeline([
            ("tfidf", FeatureUnion([
                ("word", TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2), min_df=3,
                    max_features=20000, sublinear_tf=True,
                )),
                ("char", TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                    max_features=20000, sublinear_tf=True,
                )),
            ])),
            ("lr", LogisticRegression(
                penalty="l2", C=1.0, max_iter=4000,
                class_weight="balanced", solver="lbfgs", n_jobs=None,
            )),
        ])

    oof = stratified_oof_proba(_make, np.array(docs, dtype=object), y, seed=seed)
    auc = macro_ovr_auc(y, oof, len(np.unique(y)))
    print(f"[phase_G] TF-IDF LR AUC = {auc:.4f}  ({time.time()-t0:.1f}s)", flush=True)
    return {"auc": auc, "oof_proba": oof, "n_docs": len(docs)}


# ── Phase H: Optuna CatBoost ─────────────────────────────────────────────-

def phase_H(X, y, use_gpu: bool = True, n_trials: int = 50,
            seed: int = RANDOM_STATE) -> dict:
    import optuna
    from catboost import CatBoostClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.trial.Trial") -> float:
        params = dict(
            iterations=trial.suggest_int("iterations", 200, 800, step=100),
            depth=trial.suggest_int("depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
            border_count=trial.suggest_int("border_count", 32, 254),
            loss_function="MultiClass",
            task_type="GPU" if use_gpu else "CPU",
            devices="0",
            verbose=False,
            random_seed=seed,
            allow_writing_files=False,
        )

        def _make():
            return CatBoostClassifier(**params)

        try:
            oof = stratified_oof_proba(_make, X, y, seed=seed)
        except Exception as e:
            print(f"[phase_H] trial failed: {e}", flush=True)
            return 0.0
        return macro_ovr_auc(y, oof, len(np.unique(y)))

    print(f"[phase_H] starting Optuna (n_trials={n_trials}, gpu={use_gpu})",
          flush=True)
    t0 = time.time()
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"[phase_H]   done in {time.time()-t0:.0f}s   "
          f"best AUC = {study.best_value:.4f}", flush=True)
    print(f"[phase_H]   best params = {study.best_params}", flush=True)

    # Recompute OOF with best params to get predictions for stacking
    best_params = {**study.best_params,
                   "loss_function": "MultiClass",
                   "task_type": "GPU" if use_gpu else "CPU",
                   "devices": "0",
                   "verbose": False,
                   "random_seed": seed,
                   "allow_writing_files": False}

    def _make_best():
        return CatBoostClassifier(**best_params)

    oof = stratified_oof_proba(_make_best, X, y, seed=seed)
    auc = macro_ovr_auc(y, oof, len(np.unique(y)))

    history = [{"number": t.number,
                "params": t.params,
                "auc": float(t.value) if t.value is not None else None}
               for t in study.trials]

    return {
        "best_auc":    auc,
        "best_params": study.best_params,
        "n_trials":    n_trials,
        "oof_proba":   oof,
        "history":     history,
    }


# ── Phase H (skip variant): use v1's best CatBoost params ────────────────

def phase_H_v1_fallback(X, y, use_gpu: bool, seed: int = RANDOM_STATE) -> dict:
    from catboost import CatBoostClassifier
    best_params = {
        "iterations":     300,
        "depth":          6,
        "learning_rate":  0.1,
        "loss_function":  "MultiClass",
        "task_type":      "GPU" if use_gpu else "CPU",
        "devices":        "0",
        "verbose":        False,
        "random_seed":    seed,
        "allow_writing_files": False,
    }
    def _make():
        return CatBoostClassifier(**best_params)
    t0 = time.time()
    oof = stratified_oof_proba(_make, X, y, seed=seed)
    auc = macro_ovr_auc(y, oof, len(np.unique(y)))
    print(f"[phase_H*] v1-params CatBoost AUC = {auc:.4f}  ({time.time()-t0:.1f}s)",
          flush=True)
    return {"best_auc": auc,
            "best_params": best_params,
            "n_trials": 0,
            "oof_proba": oof}


# ── Phase I: Stacking ─────────────────────────────────────────────────────

def phase_I(y: np.ndarray, proba_cat: np.ndarray, proba_emb: np.ndarray,
            proba_tfidf: np.ndarray, seed: int = RANDOM_STATE) -> dict:
    """LR meta on stacked OOF probas, honestly CV'd."""
    X_meta = np.hstack([proba_cat, proba_emb, proba_tfidf])
    t0 = time.time()

    def _make():
        return LogisticRegression(
            penalty="l2", C=1.0, max_iter=2000,
            class_weight="balanced", solver="lbfgs",
        )

    oof = stratified_oof_proba(_make, X_meta, y, seed=seed)
    auc = macro_ovr_auc(y, oof, len(np.unique(y)))
    print(f"[phase_I] stacked-LR AUC = {auc:.4f}  ({time.time()-t0:.1f}s)",
          flush=True)
    return {"auc": auc, "meta_shape": list(X_meta.shape), "oof_proba": oof}


# ── Phase J: Simple-average blend ─────────────────────────────────────────

def phase_J(y: np.ndarray, *probas: np.ndarray) -> dict:
    avg = np.mean(np.stack(probas, axis=0), axis=0)
    auc = macro_ovr_auc(y, avg, avg.shape[1])
    print(f"[phase_J] avg-blend AUC = {auc:.4f}  (n_components={len(probas)})",
          flush=True)
    return {"auc": auc, "n_components": len(probas)}


# ── Phase K: multi-seed verification of stacker ───────────────────────────

def phase_K(X_feat, authors: list[str], y: np.ndarray,
            docs: list[str], X_emb: np.ndarray,
            best_cat_params: dict, use_gpu: bool,
            seeds: list[int]) -> dict:
    """Re-run the full stacked pipeline across CV seeds."""
    from catboost import CatBoostClassifier
    aucs: list[float] = []
    for s in seeds:
        print(f"\n[phase_K] seed={s}", flush=True)
        def _make_cat():
            return CatBoostClassifier(**{**best_cat_params, "random_seed": s,
                                         "task_type": "GPU" if use_gpu else "CPU",
                                         "devices": "0",
                                         "verbose": False,
                                         "loss_function": "MultiClass",
                                         "allow_writing_files": False})
        oof_c = stratified_oof_proba(_make_cat, X_feat, y, seed=s)

        def _make_emb():
            return Pipeline([
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(
                    penalty="l2", C=1.0, max_iter=2000,
                    class_weight="balanced", solver="lbfgs")),
            ])
        oof_e = stratified_oof_proba(_make_emb, X_emb, y, seed=s)

        def _make_tfidf():
            return Pipeline([
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
        oof_t = stratified_oof_proba(_make_tfidf,
                                     np.array(docs, dtype=object), y, seed=s)

        X_meta = np.hstack([oof_c, oof_e, oof_t])

        def _make_meta():
            return LogisticRegression(penalty="l2", C=1.0, max_iter=2000,
                                      class_weight="balanced", solver="lbfgs")
        oof_meta = stratified_oof_proba(_make_meta, X_meta, y, seed=s)
        auc = macro_ovr_auc(y, oof_meta, len(np.unique(y)))
        aucs.append(auc)
        print(f"[phase_K]   seed={s}  stacked AUC = {auc:.4f}", flush=True)

    return {"seeds": seeds,
            "aucs":  aucs,
            "mean":  float(np.mean(aucs)),
            "std":   float(np.std(aucs, ddof=1))}


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gpu", action="store_true",
                        help="Use CPU CatBoost (slow)")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Optuna trials for Phase H (default 50)")
    parser.add_argument("--skip-optuna", action="store_true",
                        help="Skip Phase H Optuna, use v1's best CatBoost params")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of seeds for Phase K verification (default 5)")
    parser.add_argument("--skip-phase-k", action="store_true",
                        help="Skip the multi-seed verification phase (fast)")
    args = parser.parse_args()

    use_gpu = not args.no_gpu
    print(f"[auc_sweep_v2] GPU = {'ON' if use_gpu else 'OFF'}", flush=True)

    print("[auc_sweep_v2] loading tabular features (kitchen-sink)...", flush=True)
    df = load_all_features()
    df = build_presence_flags(df)
    feat_cols = discover_feature_cols(df)
    X, y, le = prepare_dataset(df, high_confidence_only=False,
                               feature_cols=feat_cols)
    # `prepare_dataset` returns X as DataFrame whose .index aligns with the
    # rows of `df` it was built from. Pull authors in matching order.
    if hasattr(X, "index"):
        authors = df.loc[X.index, "author"].tolist()
    else:
        authors = df["author"].tolist()
    assert len(authors) == len(y), (
        f"Author alignment mismatch: {len(authors)} vs {len(y)}"
    )

    classes = list(le.classes_)
    n_classes = len(classes)
    print(f"[auc_sweep_v2] n={len(y)}  features={X.shape[1]}  classes={classes}",
          flush=True)
    print(f"[auc_sweep_v2] class counts = "
          f"{ {c: int((y==i).sum()) for i,c in enumerate(classes)} }",
          flush=True)

    results: dict[str, Any] = {
        "n":            int(len(y)),
        "n_features":   int(X.shape[1]),
        "classes":      classes,
        "class_counts": {c: int((y == i).sum()) for i, c in enumerate(classes)},
        "gpu_enabled":  use_gpu,
    }

    # Phase F — MentalBERT-embedding LR
    print("\n=== Phase F: MentalBERT-embedding LR ===", flush=True)
    F = phase_F(authors, y)
    results["phase_f"] = {"auc": F["auc"], "X_shape": F["X_shape"]}

    # Phase G — TF-IDF LR
    print("\n=== Phase G: TF-IDF LR ===", flush=True)
    G = phase_G(authors, y)
    results["phase_g"] = {"auc": G["auc"], "n_docs": G["n_docs"]}

    # Phase H — Optuna CatBoost (or v1-fallback)
    print("\n=== Phase H: Optuna CatBoost ===", flush=True)
    if args.skip_optuna:
        H = phase_H_v1_fallback(X, y, use_gpu=use_gpu)
        results["phase_h"] = {
            "best_auc":    H["best_auc"],
            "best_params": H["best_params"],
            "n_trials":    0,
            "skipped":     True,
        }
    else:
        H = phase_H(X, y, use_gpu=use_gpu, n_trials=args.n_trials)
        results["phase_h"] = {
            "best_auc":    H["best_auc"],
            "best_params": H["best_params"],
            "n_trials":    H["n_trials"],
            "history":     H["history"],
        }

    # Phase I — stacked LR meta
    print("\n=== Phase I: Stacked-LR meta ===", flush=True)
    I = phase_I(y, H["oof_proba"], F["oof_proba"], G["oof_proba"])
    results["phase_i"] = {"auc": I["auc"], "meta_shape": I["meta_shape"]}

    # Phase J — simple-average blend
    print("\n=== Phase J: Simple-average blend ===", flush=True)
    J = phase_J(y, H["oof_proba"], F["oof_proba"], G["oof_proba"])
    results["phase_j"] = J

    # Phase K — multi-seed stacker verification
    if not args.skip_phase_k:
        print(f"\n=== Phase K: Multi-seed stacker verification "
              f"(n_seeds={args.n_seeds}) ===", flush=True)
        seeds = list(range(1, args.n_seeds + 1))
        # Cache the v1 sweep's Phase E seeds = [1..5] for direct comparability
        K = phase_K(X, authors, y, build_user_text(authors),
                    build_user_embedding_matrix(authors),
                    H["best_params"], use_gpu=use_gpu, seeds=seeds)
        results["phase_k"] = K
    else:
        results["phase_k"] = {"skipped": True}

    # ── Summary ───────────────────────────────────────────────────────────
    v1_baseline_seeded = 0.7355
    v1_baseline_multi  = 0.7181
    print("\n" + "=" * 60)
    print("  ROUND-2 AUC SWEEP SUMMARY")
    print("=" * 60)
    print(f"  v1 CatBoost (seed=42)   = {v1_baseline_seeded:.4f}")
    print(f"  v1 CatBoost (5-seed)    = {v1_baseline_multi:.4f} ± 0.008")
    print(f"  Phase F (emb LR)        = {results['phase_f']['auc']:.4f}")
    print(f"  Phase G (TF-IDF LR)     = {results['phase_g']['auc']:.4f}")
    print(f"  Phase H (Optuna CatB)   = {results['phase_h']['best_auc']:.4f}")
    print(f"  Phase I (stacked LR)    = {results['phase_i']['auc']:.4f}")
    print(f"  Phase J (avg blend)     = {results['phase_j']['auc']:.4f}")
    if not args.skip_phase_k:
        print(f"  Phase K (stack, {args.n_seeds}-seed) = "
              f"{results['phase_k']['mean']:.4f} ± "
              f"{results['phase_k']['std']:.4f}")
    print("=" * 60)

    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_OUT, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\n[auc_sweep_v2] wrote {RESULTS_OUT}", flush=True)


if __name__ == "__main__":
    main()
