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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
FEATURES_IN = DATA_DIR / "features.parquet"
RESULTS_OUT = DATA_DIR / "model_results.json"

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
    """Add binary flags: 1 if the user had >=1 post in that window, else 0."""
    df = df.copy()
    for win in ("pre_4w", "pre_2w", "pre_1w"):
        # post_freq is always 0 (not NaN) when window is empty, so use that
        df[f"has_posts_{win}"] = (df[f"post_freq_{win}"] > 0).astype(float)
    return df


def prepare_dataset(
    df: pd.DataFrame,
    high_confidence_only: bool = False,
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
    X = df[ALL_FEATURE_COLS].astype(float)
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
    # After imputation the feature names are preserved
    names = ALL_FEATURE_COLS
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

def main() -> None:
    """Run stage 4: train, evaluate, sensitivity analysis, print results."""
    print("[train_model] Loading feature matrix...")
    df = pd.read_parquet(FEATURES_IN)
    print(f"[train_model] {len(df)} users, label counts: "
          f"{df['label'].value_counts().to_dict()}")
    print(f"[train_model] low_confidence users: {df['low_confidence'].sum()}")

    datasets: dict[str, tuple] = {
        "full":      prepare_dataset(df, high_confidence_only=False),
        "high_conf": prepare_dataset(df, high_confidence_only=True),
    }
    for name, (X, y, le) in datasets.items():
        counts = {le.classes_[i]: int((y == i).sum()) for i in range(len(le.classes_))}
        print(f"[train_model] dataset={name:12s}  n={len(y)}  {counts}")

    models: dict[str, Pipeline] = {
        "LogReg": make_lr(),
        "RandForest": make_rf(),
    }

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

    # Feature importance on full dataset
    print("[train_model] Fitting RF on full dataset for feature importance...")
    X_full, y_full, le_full = datasets["full"]
    imp_df = rf_feature_importance(make_rf(), X_full, y_full)

    # Print results
    print_results_table(all_results)
    print_sensitivity_delta(all_results)
    print_feature_importance(imp_df, "full dataset")

    # Save results for visualize.py
    # Convert numpy types for JSON serialisation
    def _jsonify(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonify(v) for v in obj]
        return obj

    save_payload = {
        "results":       _jsonify(all_results),
        "feature_importance": {
            "feature":    imp_df["feature"].tolist(),
            "importance": imp_df["importance"].round(6).tolist(),
        },
    }
    with open(RESULTS_OUT, "w") as fh:
        json.dump(save_payload, fh, indent=2)
    print(f"[train_model] Saved results to {RESULTS_OUT}")


if __name__ == "__main__":
    main()
