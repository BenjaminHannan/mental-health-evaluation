"""Train classifiers to distinguish crisis / recovery / neither users.

Stage 4 of the pipeline. Reads ``data/features.parquet`` and trains:
  - Logistic Regression  (baseline, interpretable)
  - Random Forest        (non-linear, feature importances)

Evaluation
----------
5-fold stratified cross-validation. Out-of-fold predictions are pooled
to compute per-class precision, recall, F1, and one-vs-rest ROC-AUC.
Macro- and weighted-average summaries are also reported.

Sensitivity analysis
--------------------
Runs are repeated on two datasets:
  full          : all 132 labelled users + 449 neither  (581 total)
  high_conf     : 33 low-confidence "made it" recovery
                  users removed  (548 total)

A clean comparison table is printed at the end.

Outputs
-------
  data/model_results.json   — full metrics dict (used by visualize.py)

Run:
    python src/train_model.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Optional gradient-boosting backends ───────────────────────────────────
try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
FEATURES_IN = DATA_DIR / "features.parquet"
RESULTS_OUT = DATA_DIR / "model_results.json"

# Columns that are metadata, not features. Everything else numeric is a feature.
METADATA_COLS = {
    "author", "label", "low_confidence", "n_posts",
    "tp_date", "n_baseline_buckets",
}

RANDOM_STATE = 42
N_FOLDS      = 5

# ── Feature columns ────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "sentiment_mean", "ttr", "avg_sent_len",
    "post_freq", "fp_pronoun_rate", "neg_affect_rate", "avg_post_len",
]
WINDOWS = ["baseline", "pre_4w", "pre_2w", "pre_1w"]
DELTA_WINDOWS = ["pre_4w", "pre_2w", "pre_1w"]

# Raw window columns
RAW_COLS = [f"{f}_{w}" for f in FEATURE_NAMES for w in WINDOWS]
# Delta columns (pre_window − baseline)
DELTA_COLS = [f"{f}_delta_{w}" for f in FEATURE_NAMES for w in DELTA_WINDOWS]
# Presence-of-posts flags: silence in a window is itself informative
PRESENCE_COLS = [f"has_posts_{w}" for w in ("pre_4w", "pre_2w", "pre_1w")]

ALL_FEATURE_COLS = RAW_COLS + DELTA_COLS + PRESENCE_COLS

LABEL_ORDER = ["crisis", "recovery", "neither"]


# ── Data preparation ───────────────────────────────────────────────────────

def build_presence_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary flags: 1 if the user had >=1 post in that window, else 0.

    Only added if not already present (z-norm mode emits them directly).
    """
    df = df.copy()
    for win in ("pre_4w", "pre_2w", "pre_1w"):
        col = f"has_posts_{win}"
        if col in df.columns:
            continue
        freq_col = f"post_freq_{win}"
        if freq_col in df.columns:
            df[col] = (df[freq_col] > 0).astype(float)
    return df


def discover_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature column names: all numeric columns minus metadata."""
    numeric = df.select_dtypes(include=["number", "bool"]).columns
    return [c for c in numeric if c not in METADATA_COLS]


def prepare_dataset(
    df: pd.DataFrame,
    high_confidence_only: bool = False,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """Return feature matrix X, encoded labels y, and the fitted LabelEncoder.

    Parameters
    ----------
    df : full features dataframe
    high_confidence_only : if True, drop low_confidence recovery users
    """
    df = build_presence_flags(df)
    if high_confidence_only:
        mask = ~((df["label"] == "recovery") & (df["low_confidence"] == True))
        df = df[mask].copy()

    le = LabelEncoder()
    le.fit(LABEL_ORDER)
    y = le.transform(df["label"])

    if feature_cols is None:
        # Back-compat: use the hard-coded raw-feature list if available,
        # otherwise auto-discover from the dataframe.
        if all(c in df.columns for c in ALL_FEATURE_COLS):
            feature_cols = ALL_FEATURE_COLS
        else:
            feature_cols = discover_feature_cols(df)
    X = df[feature_cols].astype(float)
    return X, y, le


# ── Model pipelines ────────────────────────────────────────────────────────

def make_lr() -> Pipeline:
    """Logistic Regression pipeline: impute → scale → classify."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=0.5,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )),
    ])


def make_rf() -> Pipeline:
    """Random Forest pipeline: impute → classify (no scaling needed)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf",     RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def make_catboost() -> Pipeline:
    """CatBoost pipeline.  Handles NaNs internally, so no imputer needed,
    but we keep one for consistency with the other pipelines."""
    if not _HAS_CATBOOST:
        raise RuntimeError("catboost not installed (pip install catboost)")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", CatBoostClassifier(
            iterations=500,
            depth=5,
            learning_rate=0.05,
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            verbose=0,
            allow_writing_files=False,
        )),
    ])


def make_xgboost() -> Pipeline:
    """XGBoost pipeline.  Multiclass softprob objective."""
    if not _HAS_XGBOOST:
        raise RuntimeError("xgboost not installed (pip install xgboost)")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )),
    ])


def make_lightgbm() -> Pipeline:
    """LightGBM pipeline.  Class-weight balanced for imbalance."""
    if not _HAS_LIGHTGBM:
        raise RuntimeError("lightgbm not installed (pip install lightgbm)")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", LGBMClassifier(
            n_estimators=400,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.05,
            objective="multiclass",
            num_class=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )),
    ])


def make_stacking() -> Pipeline:
    """Stacking ensemble: LR + RF + (CatBoost|XGBoost) → LR meta-model.

    Uses whichever boosting library is available as the third base model.
    Out-of-fold predictions from the base learners become inputs to a
    penalised logistic regression meta-learner.
    """
    base_estimators = [
        ("lr", LogisticRegression(
            max_iter=1000, class_weight="balanced", C=0.5,
            solver="lbfgs", random_state=RANDOM_STATE,
        )),
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=6, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ]
    if _HAS_CATBOOST:
        base_estimators.append(("cb", CatBoostClassifier(
            iterations=500, depth=5, learning_rate=0.05,
            loss_function="MultiClass", auto_class_weights="Balanced",
            random_seed=RANDOM_STATE, verbose=0, allow_writing_files=False,
        )))
    elif _HAS_XGBOOST:
        base_estimators.append(("xgb", XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
        )))

    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(
            max_iter=1000, C=0.5, class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        passthrough=False,
        n_jobs=1,
    )

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     stack),
    ])


# ── Hyperparameter search ─────────────────────────────────────────────────

def _run_grid_search(
    name: str,
    base_pipeline: Pipeline,
    param_grid: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    n_jobs: int = 1,
) -> Pipeline:
    """GridSearchCV on the ``clf__`` params of a pipeline, 5-fold CV, macro-AUC."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        base_pipeline,
        param_grid,
        scoring="roc_auc_ovr",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )
    print(f"[hyperparam] {name}: grid size = "
          f"{int(np.prod([len(v) for v in param_grid.values()]))}")
    gs.fit(X, y)
    print(f"[hyperparam] {name}: best params = {gs.best_params_}")
    print(f"[hyperparam] {name}: best macro AUC = {gs.best_score_:.4f}")
    return gs.best_estimator_


def tune_lr(X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    return _run_grid_search("LogReg", make_lr(), {
        "clf__C": [0.1, 0.3, 0.5, 1.0, 3.0],
    }, X, y)


def tune_rf(X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    return _run_grid_search("RandForest", make_rf(), {
        "clf__n_estimators": [200, 500, 1000],
        "clf__max_depth":    [3, 6, 10, None],
        "clf__min_samples_leaf": [1, 3, 5],
    }, X, y, n_jobs=-1)


def tune_catboost(X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    if not _HAS_CATBOOST:
        raise RuntimeError("catboost not installed")
    return _run_grid_search("CatBoost", make_catboost(), {
        "clf__depth": [3, 5, 7],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__iterations": [300, 500, 800],
    }, X, y)


def tune_xgboost(X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    if not _HAS_XGBOOST:
        raise RuntimeError("xgboost not installed")
    return _run_grid_search("XGBoost", make_xgboost(), {
        "clf__max_depth": [3, 4, 6],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__n_estimators": [200, 400, 800],
    }, X, y, n_jobs=-1)


# ── Cross-validation evaluation ────────────────────────────────────────────

def evaluate_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    le: LabelEncoder,
) -> dict:
    """Run 5-fold CV and return pooled per-class + macro/weighted metrics.

    Uses cross_val_predict to pool out-of-fold predictions so that
    ROC-AUC is computed over the full dataset rather than averaged over
    fold estimates (more stable with small class counts).
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Pooled OOF hard predictions and probabilities
    y_pred  = cross_val_predict(pipeline, X, y, cv=cv, method="predict")
    y_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")

    classes = le.classes_  # ["crisis", "recovery", "neither"]
    n_classes = len(classes)

    # Per-class precision / recall / F1
    prec, rec, f1, support = precision_recall_fscore_support(
        y, y_pred, labels=list(range(n_classes)), zero_division=0
    )

    # Per-class one-vs-rest ROC-AUC
    per_class_auc: list[float] = []
    for i in range(n_classes):
        try:
            auc = roc_auc_score((y == i).astype(int), y_proba[:, i])
        except ValueError:
            auc = float("nan")
        per_class_auc.append(auc)

    # Macro OvR AUC (only include non-NaN classes)
    valid_aucs = [a for a in per_class_auc if not np.isnan(a)]
    macro_auc  = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

    # Macro / weighted averages
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y, y_pred, average="macro", zero_division=0
    )
    wt_prec, wt_rec, wt_f1, _ = precision_recall_fscore_support(
        y, y_pred, average="weighted", zero_division=0
    )

    per_class: dict[str, dict] = {}
    for i, cls in enumerate(classes):
        per_class[cls] = {
            "precision": round(float(prec[i]), 4),
            "recall":    round(float(rec[i]),  4),
            "f1":        round(float(f1[i]),   4),
            "roc_auc":   round(per_class_auc[i], 4),
            "support":   int(support[i]),
        }

    return {
        "per_class": per_class,
        "macro": {
            "precision": round(float(macro_prec), 4),
            "recall":    round(float(macro_rec),  4),
            "f1":        round(float(macro_f1),   4),
            "roc_auc":   round(macro_auc, 4),
        },
        "weighted": {
            "precision": round(float(wt_prec), 4),
            "recall":    round(float(wt_rec),  4),
            "f1":        round(float(wt_f1),   4),
        },
    }


# ── Feature importance (Random Forest) ────────────────────────────────────

def rf_feature_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    top_n: int = 15,
) -> pd.DataFrame:
    """Fit RF on full dataset and return top-N feature importances."""
    pipeline.fit(X, y)
    clf = pipeline.named_steps["clf"]
    imp = clf.feature_importances_
    # After imputation the feature names are preserved. X.columns is
    # authoritative whether we passed raw or z-norm features.
    names = list(X.columns)
    df_imp = pd.DataFrame({"feature": names, "importance": imp})
    return df_imp.sort_values("importance", ascending=False).head(top_n)


# ── Pretty printing ────────────────────────────────────────────────────────

def _rule(width: int = 96) -> str:
    return "-" * width


def print_results_table(all_results: dict) -> None:
    """Print a formatted comparison table of all model × dataset runs."""
    col_w = 12
    W = 96

    print("\n" + "=" * W)
    print("  CLASSIFICATION RESULTS  (5-fold stratified CV, pooled OOF predictions)")
    print("=" * W)

    header = (
        f"  {'Model':<14}{'Dataset':<16}{'Class':<12}"
        f"{'Precision':>{col_w}}{'Recall':>{col_w}}{'F1':>{col_w}}{'ROC-AUC':>{col_w}}"
        f"{'Support':>{col_w}}"
    )
    print(header)
    print(_rule(W))

    for model_name, datasets in all_results.items():
        for dataset_name, metrics in datasets.items():
            first_row = True
            for cls in [*LABEL_ORDER, "macro", "weighted"]:
                is_avg = cls in ("macro", "weighted")
                row_data = metrics["macro"] if cls == "macro" else (
                    metrics["weighted"] if cls == "weighted"
                    else metrics["per_class"].get(cls, {})
                )
                if not row_data:
                    continue
                label_disp = f"[{cls}]" if is_avg else cls
                auc_disp = (
                    f"{row_data.get('roc_auc', float('nan')):>{col_w}.4f}"
                    if not is_avg or cls == "macro"
                    else f"{'--':>{col_w}}"
                )
                support_disp = (
                    f"{row_data.get('support', ''):>{col_w}}"
                    if not is_avg else f"{'':>{col_w}}"
                )
                m_disp = model_name if first_row else ""
                d_disp = dataset_name if first_row else ""
                first_row = False

                print(
                    f"  {m_disp:<14}{d_disp:<16}{label_disp:<12}"
                    f"{row_data.get('precision', float('nan')):>{col_w}.4f}"
                    f"{row_data.get('recall',    float('nan')):>{col_w}.4f}"
                    f"{row_data.get('f1',        float('nan')):>{col_w}.4f}"
                    f"{auc_disp}"
                    f"{support_disp}"
                )
            print(_rule(W))

    print()


def print_sensitivity_delta(all_results: dict) -> None:
    """Print the F1 / AUC change between full and high_conf datasets."""
    W = 72
    print("=" * W)
    print("  SENSITIVITY ANALYSIS  (full vs high_confidence, F1 delta)")
    print("  Positive = score improves when 33 low-confidence users are removed")
    print("=" * W)
    header = f"  {'Model':<14}{'Class':<14}{'dF1':>10}{'dAUC':>10}"
    print(header)
    print(_rule(W))
    for model_name, datasets in all_results.items():
        if "full" not in datasets or "high_conf" not in datasets:
            continue
        full     = datasets["full"]
        high     = datasets["high_conf"]
        first_row = True
        for cls in [*LABEL_ORDER, "macro"]:
            is_avg = cls == "macro"
            full_d = full["macro"] if is_avg else full["per_class"].get(cls, {})
            high_d = high["macro"] if is_avg else high["per_class"].get(cls, {})
            if not full_d or not high_d:
                continue
            d_f1  = high_d["f1"]       - full_d["f1"]
            d_auc = high_d.get("roc_auc", float("nan")) - full_d.get("roc_auc", float("nan"))
            m_disp = model_name if first_row else ""
            first_row = False
            label_disp = f"[{cls}]" if is_avg else cls
            print(f"  {m_disp:<14}{label_disp:<14}{d_f1:>+10.4f}{d_auc:>+10.4f}")
        print(_rule(W))
    print()


def print_feature_importance(imp_df: pd.DataFrame, model_label: str) -> None:
    """Print top feature importances from the Random Forest."""
    print(f"  Top features ({model_label}, Random Forest, Gini importance):")
    print(f"  {'Feature':<40}{'Importance':>12}")
    print("  " + "-" * 52)
    for _, row in imp_df.iterrows():
        print(f"  {row['feature']:<40}{row['importance']:>12.4f}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────

def _jsonify(obj):
    """Recursively convert numpy scalars to Python types for JSON output."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    return obj


def _build_model_dict(model_names: list[str] | None) -> dict[str, Pipeline]:
    """Resolve a list of short names to fresh pipeline instances."""
    if model_names is None:
        model_names = ["LogReg", "RandForest"]
    registry = {
        "LogReg":     make_lr,
        "RandForest": make_rf,
    }
    if _HAS_CATBOOST:
        registry["CatBoost"] = make_catboost
    if _HAS_XGBOOST:
        registry["XGBoost"]  = make_xgboost
    if _HAS_LIGHTGBM:
        registry["LightGBM"] = make_lightgbm
    registry["Stacking"] = make_stacking

    models: dict[str, Pipeline] = {}
    for name in model_names:
        if name not in registry:
            print(f"[train_model] WARNING: unknown model {name!r}, skipping "
                  f"(available: {list(registry)})")
            continue
        models[name] = registry[name]()
    return models


def run_experiment(
    features_path: Path | None,
    results_path: Path,
    label: str = "raw",
    feature_cols: list[str] | None = None,
    df: pd.DataFrame | None = None,
    model_names: list[str] | None = None,
    hyperparam_search: bool = False,
) -> dict:
    """Train LR + RF with 5-fold CV on the given feature matrix.

    Parameters
    ----------
    features_path : path to a features parquet (ignored if ``df`` given)
    results_path  : where to save the metrics JSON
    label         : short tag for print-outs (e.g. 'raw', 'znorm', 'deltas')
    feature_cols  : explicit feature columns; if None, auto-discover
    df            : pre-built features dataframe (bypass file read)
    """
    print(f"[train_model] ===== experiment: {label} =====")
    if df is None:
        print(f"[train_model] Loading {features_path}...")
        df = pd.read_parquet(features_path)
    else:
        print(f"[train_model] Using pre-built dataframe ({len(df)} rows, "
              f"{df.shape[1]} cols)")
    print(f"[train_model] {len(df)} users, label counts: "
          f"{df['label'].value_counts().to_dict()}")
    print(f"[train_model] low_confidence users: {int(df['low_confidence'].sum())}")

    datasets: dict[str, tuple] = {
        "full":      prepare_dataset(df, high_confidence_only=False,
                                     feature_cols=feature_cols),
        "high_conf": prepare_dataset(df, high_confidence_only=True,
                                     feature_cols=feature_cols),
    }
    for name, (X, y, le) in datasets.items():
        counts = {le.classes_[i]: int((y == i).sum()) for i in range(len(le.classes_))}
        print(f"[train_model] dataset={name:12s}  n={len(y)}  "
              f"n_features={X.shape[1]}  {counts}")

    models = _build_model_dict(model_names)

    # Optional hyperparameter search — fit on the full dataset, then use the
    # tuned estimators in place of the default pipelines below.
    if hyperparam_search:
        print(f"\n[hyperparam] Running grid search on the full dataset "
              f"({len(datasets['full'][1])} users)...")
        X_full, y_full, _ = datasets["full"]
        if "LogReg" in models:
            models["LogReg"] = tune_lr(X_full, y_full)
        if "RandForest" in models:
            models["RandForest"] = tune_rf(X_full, y_full)
        if "CatBoost" in models:
            models["CatBoost"] = tune_catboost(X_full, y_full)
        if "XGBoost" in models:
            models["XGBoost"] = tune_xgboost(X_full, y_full)

    all_results: dict[str, dict] = {}
    for model_name, pipeline in models.items():
        all_results[model_name] = {}
        for dataset_name, (X, y, le) in datasets.items():
            print(f"[train_model] CV: {model_name} / {dataset_name} "
                  f"(n={len(y)})...", flush=True)
            metrics = evaluate_cv(pipeline, X, y, le)
            all_results[model_name][dataset_name] = metrics
            print(f"[train_model]   macro F1={metrics['macro']['f1']:.4f}  "
                  f"ROC-AUC={metrics['macro']['roc_auc']:.4f}")

    print("[train_model] Fitting RF on full dataset for feature importance...")
    X_full, y_full, _ = datasets["full"]
    imp_df = rf_feature_importance(make_rf(), X_full, y_full)

    print_results_table(all_results)
    print_sensitivity_delta(all_results)
    print_feature_importance(imp_df, f"{label} dataset")

    save_payload = {
        "label":   label,
        "results": _jsonify(all_results),
        "feature_importance": {
            "feature":    imp_df["feature"].tolist(),
            "importance": imp_df["importance"].round(6).tolist(),
        },
    }
    with open(results_path, "w") as fh:
        json.dump(save_payload, fh, indent=2)
    print(f"[train_model] Saved results to {results_path}")
    return save_payload


def load_raw_plus_temporal_features() -> pd.DataFrame:
    """Merge raw linguistic features with temporal posting-behaviour features."""
    raw  = pd.read_parquet(FEATURES_IN)
    temp = pd.read_parquet(DATA_DIR / "features_temporal.parquet")
    drop = {"label", "low_confidence", "n_posts", "tp_date"}
    temp = temp.drop(columns=[c for c in drop if c in temp.columns])
    return raw.merge(temp, on="author", how="inner")


def load_raw_plus_mentalbert_features() -> pd.DataFrame:
    """Merge raw linguistic features with MentalBERT semantic-shift features."""
    raw = pd.read_parquet(FEATURES_IN)
    mb  = pd.read_parquet(DATA_DIR / "features_mentalbert.parquet")
    drop = {"label", "low_confidence", "n_posts", "tp_date"}
    mb   = mb.drop(columns=[c for c in drop if c in mb.columns])
    return raw.merge(mb, on="author", how="inner")


def load_all_features() -> pd.DataFrame:
    """Merge raw+znorm+temporal+mentalbert features on author (kitchen-sink run)."""
    raw  = pd.read_parquet(FEATURES_IN)
    zn   = pd.read_parquet(DATA_DIR / "features_znorm.parquet")
    temp = pd.read_parquet(DATA_DIR / "features_temporal.parquet")
    mb_path = DATA_DIR / "features_mentalbert.parquet"
    drop = {"label", "low_confidence", "n_posts", "tp_date"}
    zn   = zn.drop(columns=[c for c in drop if c in zn.columns])
    temp = temp.drop(columns=[c for c in drop if c in temp.columns])
    merged = raw.merge(zn, on="author", how="inner")
    merged = merged.merge(temp, on="author", how="inner")
    if mb_path.exists():
        mb = pd.read_parquet(mb_path)
        mb = mb.drop(columns=[c for c in drop if c in mb.columns])
        merged = merged.merge(mb, on="author", how="inner")
    return merged


def load_raw_plus_bonus_features() -> pd.DataFrame:
    """Merge raw linguistic features with bonus features (readability, punct, pronouns)."""
    raw   = pd.read_parquet(FEATURES_IN)
    bonus = pd.read_parquet(DATA_DIR / "features_bonus.parquet")
    drop  = {"label", "low_confidence", "n_posts", "tp_date"}
    bonus = bonus.drop(columns=[c for c in drop if c in bonus.columns])
    return raw.merge(bonus, on="author", how="inner")


def load_all_features_plus_bonus() -> pd.DataFrame:
    """Kitchen-sink+bonus: raw + znorm + temporal + mentalbert + bonus."""
    merged    = load_all_features()
    bonus_p   = DATA_DIR / "features_bonus.parquet"
    drop      = {"label", "low_confidence", "n_posts", "tp_date"}
    if bonus_p.exists():
        bonus = pd.read_parquet(bonus_p)
        bonus = bonus.drop(columns=[c for c in drop if c in bonus.columns])
        merged = merged.merge(bonus, on="author", how="inner")
    return merged


def load_combined_features() -> pd.DataFrame:
    """Merge raw + z-norm feature parquets on author.

    Keeps metadata from raw (label, low_confidence, n_posts, tp_date) and
    drops duplicate metadata from the z-norm side, but retains the
    z-norm-only ``n_baseline_buckets`` column as diagnostic metadata.
    The resulting frame carries both style (raw) and change (znorm)
    signals in a single row per user.
    """
    raw = pd.read_parquet(FEATURES_IN)
    zn  = pd.read_parquet(DATA_DIR / "features_znorm.parquet")

    # Columns unique to the znorm frame (drop duplicated metadata first)
    drop = {"label", "low_confidence", "n_posts", "tp_date"}
    zn = zn.drop(columns=[c for c in drop if c in zn.columns])

    merged = raw.merge(zn, on="author", how="inner")
    return merged


def print_experiment_comparison(experiments: dict[str, dict]) -> None:
    """Side-by-side comparison of macro metrics across experiments."""
    W = 88
    print("\n" + "=" * W)
    print("  EXPERIMENT COMPARISON  (macro-averaged, 5-fold CV)")
    print("=" * W)
    header = (
        f"  {'Experiment':<16}{'Dataset':<14}{'Model':<14}"
        f"{'Macro F1':>10}{'Macro AUC':>12}"
    )
    print(header)
    print("-" * W)
    for exp_label, payload in experiments.items():
        for model_name, datasets in payload["results"].items():
            for dataset_name, metrics in datasets.items():
                print(
                    f"  {exp_label:<16}{dataset_name:<14}{model_name:<14}"
                    f"{metrics['macro']['f1']:>10.4f}"
                    f"{metrics['macro']['roc_auc']:>12.4f}"
                )
        print("-" * W)
    print()


def main() -> None:
    """Run stage 4: train, evaluate, and support --znorm / --all modes."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_IN,
                        help="Features parquet to use (default: raw)")
    parser.add_argument("--results",  type=Path, default=RESULTS_OUT,
                        help="Output JSON path for metrics")
    parser.add_argument("--label",    type=str, default="raw",
                        help="Tag for this experiment in printed output")
    parser.add_argument("--znorm",    action="store_true",
                        help="Shortcut: run on features_znorm.parquet")
    parser.add_argument("--combined", action="store_true",
                        help="Shortcut: run on merged raw+znorm features")
    parser.add_argument("--deltas",   action="store_true",
                        help="Shortcut: run on delta + presence flags only "
                             "(ablation for within-user change signal)")
    parser.add_argument("--temporal", action="store_true",
                        help="Shortcut: run on raw+temporal merged features")
    parser.add_argument("--mentalbert", action="store_true",
                        help="Shortcut: run on raw+mentalbert merged features")
    parser.add_argument("--kitchen-sink", dest="kitchen_sink",
                        action="store_true",
                        help="Shortcut: run on raw+znorm+temporal+mentalbert merged features")
    parser.add_argument("--bonus", action="store_true",
                        help="Shortcut: run on raw+bonus merged features")
    parser.add_argument("--everything", action="store_true",
                        help="Kitchen-sink + bonus features (all feature groups)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated list of models to run. "
                             "Choices: LogReg,RandForest,CatBoost,XGBoost,"
                             "LightGBM,Stacking. Default: LogReg,RandForest")
    parser.add_argument("--hyperparam-search", dest="hyperparam_search",
                        action="store_true",
                        help="Run grid-search hyperparameter tuning for each model")
    parser.add_argument("--all",      action="store_true",
                        help="Run all variants (raw, znorm, deltas, "
                             "combined, temporal, kitchen_sink) side-by-side")
    args = parser.parse_args()

    model_names = None
    if args.models:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    common = {"model_names": model_names,
              "hyperparam_search": args.hyperparam_search}

    if args.all:
        experiments: dict[str, dict] = {}
        experiments["raw"] = run_experiment(
            FEATURES_IN,
            RESULTS_OUT,
            label="raw",
            **common,
        )
        experiments["znorm"] = run_experiment(
            DATA_DIR / "features_znorm.parquet",
            DATA_DIR / "model_results_znorm.json",
            label="znorm",
            **common,
        )
        combined_df = load_combined_features()
        combined_cols = discover_feature_cols(build_presence_flags(combined_df))
        experiments["combined"] = run_experiment(
            None,
            DATA_DIR / "model_results_combined.json",
            label="combined",
            df=combined_df,
            feature_cols=combined_cols,
            **common,
        )
        experiments["deltas"] = run_experiment(
            FEATURES_IN,
            DATA_DIR / "model_results_deltas.json",
            label="deltas",
            feature_cols=DELTA_COLS + PRESENCE_COLS,
            **common,
        )
        temp_df = load_raw_plus_temporal_features()
        temp_cols = discover_feature_cols(build_presence_flags(temp_df))
        experiments["temporal"] = run_experiment(
            None,
            DATA_DIR / "model_results_temporal.json",
            label="temporal",
            df=temp_df,
            feature_cols=temp_cols,
            **common,
        )
        mb_df = load_raw_plus_mentalbert_features()
        mb_cols = discover_feature_cols(build_presence_flags(mb_df))
        experiments["mentalbert"] = run_experiment(
            None,
            DATA_DIR / "model_results_mentalbert.json",
            label="mentalbert",
            df=mb_df,
            feature_cols=mb_cols,
            **common,
        )
        all_df = load_all_features()
        all_cols = discover_feature_cols(build_presence_flags(all_df))
        experiments["kitchen_sink"] = run_experiment(
            None,
            DATA_DIR / "model_results_kitchen_sink.json",
            label="kitchen_sink",
            df=all_df,
            feature_cols=all_cols,
            **common,
        )
        print_experiment_comparison(experiments)
        return

    if args.deltas:
        run_experiment(
            FEATURES_IN,
            DATA_DIR / "model_results_deltas.json",
            label="deltas",
            feature_cols=DELTA_COLS + PRESENCE_COLS,
            **common,
        )
        return

    if args.temporal:
        temp_df = load_raw_plus_temporal_features()
        temp_cols = discover_feature_cols(build_presence_flags(temp_df))
        run_experiment(
            None,
            DATA_DIR / "model_results_temporal.json",
            label="temporal",
            df=temp_df,
            feature_cols=temp_cols,
            **common,
        )
        return

    if args.mentalbert:
        mb_df = load_raw_plus_mentalbert_features()
        mb_cols = discover_feature_cols(build_presence_flags(mb_df))
        run_experiment(
            None,
            DATA_DIR / "model_results_mentalbert.json",
            label="mentalbert",
            df=mb_df,
            feature_cols=mb_cols,
            **common,
        )
        return

    if args.kitchen_sink:
        all_df = load_all_features()
        all_cols = discover_feature_cols(build_presence_flags(all_df))
        run_experiment(
            None,
            DATA_DIR / "model_results_kitchen_sink.json",
            label="kitchen_sink",
            df=all_df,
            feature_cols=all_cols,
            **common,
        )
        return

    if args.bonus:
        bonus_df = load_raw_plus_bonus_features()
        bonus_cols = discover_feature_cols(build_presence_flags(bonus_df))
        run_experiment(
            None,
            DATA_DIR / "model_results_bonus.json",
            label="bonus",
            df=bonus_df,
            feature_cols=bonus_cols,
            **common,
        )
        return

    if args.everything:
        everything_df = load_all_features_plus_bonus()
        everything_cols = discover_feature_cols(build_presence_flags(everything_df))
        run_experiment(
            None,
            DATA_DIR / "model_results_everything.json",
            label="everything",
            df=everything_df,
            feature_cols=everything_cols,
            **common,
        )
        return

    if args.combined:
        combined_df = load_combined_features()
        combined_cols = discover_feature_cols(build_presence_flags(combined_df))
        run_experiment(
            None,
            DATA_DIR / "model_results_combined.json",
            label="combined",
            df=combined_df,
            feature_cols=combined_cols,
            **common,
        )
        return

    if args.znorm:
        features_path = DATA_DIR / "features_znorm.parquet"
        results_path  = DATA_DIR / "model_results_znorm.json"
        label         = "znorm"
    else:
        features_path = args.features
        results_path  = args.results
        label         = args.label

    run_experiment(features_path, results_path, label=label, **common)


if __name__ == "__main__":
    main()
