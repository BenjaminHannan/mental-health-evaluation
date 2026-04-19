"""Generate publication-quality figures for the paper.

Stage 5 of the pipeline. Produces four PDFs in ``paper/figures/``:

  sentiment_trajectory.pdf
      Mean VADER sentiment across time windows (baseline -> pre_1w) for
      crisis, recovery, and neither users, with 95% bootstrap CIs.

  feature_importance.pdf
      Top-15 Random Forest Gini importances, horizontal bar chart.

  roc_curves.pdf
      One-vs-rest ROC curves for both classifiers on the full dataset,
      2-panel figure (LogReg left, Random Forest right).

  ablation_comparison.pdf
      Horizontal bar chart of all 7 supervised feature configurations
      plus the unsupervised PELT baseline, ranked by macro ROC-AUC
      with 5-fold CV error bars.

OOF probabilities for the ROC curves are recomputed here with the same
pipeline and random seed used in train_model.py.

Run:
    python src/visualize.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless – no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Import shared helpers from train_model
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from train_model import (         # noqa: E402
    make_lr, make_rf,
    prepare_dataset,
    build_presence_flags,
    discover_feature_cols,
    load_combined_features,
    load_raw_plus_temporal_features,
    load_raw_plus_mentalbert_features,
    load_all_features,
    LABEL_ORDER,
    FEATURE_NAMES,
    N_FOLDS,
    RANDOM_STATE,
    DELTA_COLS,
    PRESENCE_COLS,
    FEATURES_IN as RAW_FEATURES_IN,
)

DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
PAPER_DIR   = Path(__file__).resolve().parent.parent / "paper"
FIGURES_DIR = PAPER_DIR / "figures"
FEATURES_IN = DATA_DIR / "features.parquet"
RESULTS_IN  = DATA_DIR / "model_results.json"

# ── Matplotlib style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "figure.dpi":        150,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
})

# Colour palette (colour-blind safe)
PALETTE = {
    "crisis":   "#d62728",   # red
    "recovery": "#2ca02c",   # green
    "neither":  "#7f7f7f",   # grey
}
LABEL_DISPLAY = {
    "crisis":   "Crisis",
    "recovery": "Recovery",
    "neither":  "Neither (control)",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap CI for the mean of ``values``."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)
    if len(values) == 0 or np.all(np.isnan(values)):
        return float("nan"), float("nan")
    vals = values[~np.isnan(values)]
    if len(vals) < 2:
        return float(vals[0]), float(vals[0])
    boot_means = np.array([
        rng.choice(vals, size=len(vals), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_means, 100 * alpha)), \
           float(np.percentile(boot_means, 100 * (1 - alpha)))


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path)
    print(f"[visualize] Saved {path}")
    plt.close(fig)


# ── Figure 1 : Sentiment trajectory ─────────────────────────────────────────

def plot_sentiment_trajectory(features: pd.DataFrame) -> None:
    """Line plot of mean sentiment across 4 time windows, with 95% CI bands."""
    windows    = ["baseline", "pre_4w", "pre_2w", "pre_1w"]
    x_labels   = ["Baseline\n(>4 wks)", "Pre-4 wk\nwindow", "Pre-2 wk\nwindow", "Pre-1 wk\nwindow"]
    x_pos      = np.arange(len(windows))
    rng        = np.random.default_rng(RANDOM_STATE)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    for label in LABEL_ORDER:
        subset = features[features["label"] == label]
        means, lowers, uppers = [], [], []
        for win in windows:
            col  = f"sentiment_mean_{win}"
            vals = subset[col].dropna().values
            m    = float(np.nanmean(vals)) if len(vals) > 0 else float("nan")
            lo, hi = _bootstrap_ci(vals, rng=rng)
            means.append(m); lowers.append(lo); uppers.append(hi)

        color = PALETTE[label]
        ax.plot(x_pos, means, marker="o", linewidth=2.0,
                markersize=6, color=color, label=LABEL_DISPLAY[label], zorder=3)
        ax.fill_between(
            x_pos,
            lowers, uppers,
            alpha=0.15, color=color, linewidth=0,
        )

    ax.axhline(0, color="black", linewidth=0.6, linestyle=":", zorder=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Time window relative to turning-point post")
    ax.set_ylabel("Mean VADER compound sentiment")
    ax.set_title("Sentiment trajectory approaching the turning-point post")
    ax.legend(loc="lower left", framealpha=0.9)

    # Annotate n per group
    counts = features["label"].value_counts()
    note = "  ".join(
        f"n({LABEL_DISPLAY[l]})={counts.get(l,0)}" for l in LABEL_ORDER
    )
    fig.text(0.5, -0.04, f"Note: 'Neither' users use last post as pseudo-TP.  {note}",
             ha="center", fontsize=8.5, color="#555555")

    _save(fig, "sentiment_trajectory.pdf")


# ── Figure 2 : Feature importance ───────────────────────────────────────────

# Human-readable labels for feature columns
_FEAT_READABLE = {
    "ttr":             "Type-token ratio",
    "sentiment_mean":  "VADER sentiment",
    "avg_sent_len":    "Avg sentence length",
    "post_freq":       "Posting frequency",
    "fp_pronoun_rate": "1st-person pronoun rate",
    "neg_affect_rate": "Neg. affect word rate",
    "avg_post_len":    "Avg post length",
}
_WIN_READABLE = {
    "baseline":       "(baseline)",
    "pre_4w":         "(pre-4 wk)",
    "pre_2w":         "(pre-2 wk)",
    "pre_1w":         "(pre-1 wk)",
    "delta_pre_4w":   "delta, 4-wk",
    "delta_pre_2w":   "delta, 2-wk",
    "delta_pre_1w":   "delta, 1-wk",
}


def _readable_feature(col: str) -> str:
    """Convert raw column name to a short readable label."""
    for feat_key, feat_label in _FEAT_READABLE.items():
        if col.startswith(feat_key):
            suffix = col[len(feat_key):].lstrip("_")
            win_label = _WIN_READABLE.get(suffix, suffix.replace("_", " "))
            return f"{feat_label} {win_label}"
    return col


def plot_feature_importance(results: dict) -> None:
    """Horizontal bar chart of top-15 RF feature importances."""
    feats = results["feature_importance"]["feature"]
    imps  = results["feature_importance"]["importance"]
    df    = pd.DataFrame({"feature": feats, "importance": imps})
    df    = df.sort_values("importance").tail(15)   # ascending for barh

    labels    = [_readable_feature(f) for f in df["feature"]]
    values    = df["importance"].values
    colors    = ["#4c72b0" if "baseline" in f else
                 "#dd8452" if "delta" in f else
                 "#55a868"
                 for f in df["feature"]]

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    bars = ax.barh(range(len(labels)), values, color=colors,
                   edgecolor="white", linewidth=0.4)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=9)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean decrease in impurity (Gini importance)")
    ax.set_title("Top-15 Random Forest feature importances (full dataset)")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # Legend for colour coding
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#4c72b0", label="Baseline"),
        Patch(facecolor="#dd8452", label="Delta (pre-window minus baseline)"),
        Patch(facecolor="#55a868", label="Pre-window (raw)"),
    ]
    ax.legend(handles=legend_els, loc="lower right", fontsize=9, framealpha=0.9)

    _save(fig, "feature_importance.pdf")


# ── Figure 3 : ROC curves ────────────────────────────────────────────────────

def _compute_oof_proba(
    pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
) -> np.ndarray:
    """Return out-of-fold predicted probabilities via cross_val_predict."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")


def _plot_roc_panel(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str,
) -> None:
    """Draw per-class OvR ROC curves + macro-average on a given Axes."""
    n_classes = len(LABEL_ORDER)
    y_bin     = label_binarize(y_true, classes=list(range(n_classes)))

    all_fpr = np.linspace(0, 1, 200)
    interp_tprs: list[np.ndarray] = []

    for i, cls in enumerate(LABEL_ORDER):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=PALETTE[cls], linewidth=1.8,
                label=f"{LABEL_DISPLAY[cls]} (AUC={roc_auc:.3f})")
        interp_tprs.append(np.interp(all_fpr, fpr, tpr))

    # Macro-average
    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color="black", linewidth=2.2,
            linestyle="--", label=f"Macro avg (AUC={mean_auc:.3f})", zorder=4)

    ax.plot([0, 1], [0, 1], color="#aaaaaa", linewidth=1.0,
            linestyle=":", label="Chance")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.01])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9)


def plot_roc_curves(features: pd.DataFrame) -> None:
    """2-panel ROC figure: LogReg (left) and Random Forest (right)."""
    print("[visualize] Recomputing OOF probabilities for ROC curves...")
    X, y, le = prepare_dataset(features, high_confidence_only=False)

    model_specs = [
        ("Logistic Regression", make_lr()),
        ("Random Forest",       make_rf()),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, (name, pipeline) in zip(axes, model_specs):
        print(f"[visualize]   Running CV for {name}...", flush=True)
        y_proba = _compute_oof_proba(pipeline, X, y)
        _plot_roc_panel(ax, y, y_proba, title=name)

    fig.suptitle(
        "One-vs-rest ROC curves (5-fold CV, pooled out-of-fold predictions)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    _save(fig, "roc_curves.pdf")


# ── Figure 4 : Ablation comparison ──────────────────────────────────────────

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def _per_fold_macro_aucs(pipeline, X: pd.DataFrame, y: np.ndarray) -> list[float]:
    """Run 5-fold stratified CV and return one macro OvR AUC per fold.

    Unlike the pooled OOF AUC used in train_model.py (which gives a
    single stable number), we need per-fold AUCs to estimate the
    error bar for the ablation-comparison figure.
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs: list[float] = []
    classes = np.array(sorted(np.unique(y)))
    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        pipeline.fit(X_tr, y_tr)
        proba = pipeline.predict_proba(X_te)
        # One-vs-rest macro AUC, ignoring any classes missing from this fold
        try:
            auc_val = roc_auc_score(
                y_te, proba, multi_class="ovr", average="macro",
                labels=classes,
            )
        except ValueError:
            auc_val = float("nan")
        aucs.append(float(auc_val))
    return aucs


def _load_config_dataset(config: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Return (X, y, feature_cols) for a named ablation configuration."""
    if config == "deltas":
        df = pd.read_parquet(RAW_FEATURES_IN)
        cols = DELTA_COLS + PRESENCE_COLS
        X, y, _ = prepare_dataset(df, feature_cols=cols)
        return X, y, cols
    if config == "znorm":
        df = pd.read_parquet(DATA_DIR / "features_znorm.parquet")
        df = build_presence_flags(df)
        cols = discover_feature_cols(df)
        X, y, _ = prepare_dataset(df, feature_cols=cols)
        return X, y, cols
    if config == "raw":
        df = pd.read_parquet(RAW_FEATURES_IN)
        X, y, _ = prepare_dataset(df)
        return X, y, list(X.columns)
    if config == "raw+embeddings":
        df = load_raw_plus_mentalbert_features()
        df = build_presence_flags(df)
        cols = discover_feature_cols(df)
        X, y, _ = prepare_dataset(df, feature_cols=cols)
        return X, y, cols
    if config == "raw+znorm":
        df = load_combined_features()
        df = build_presence_flags(df)
        cols = discover_feature_cols(df)
        X, y, _ = prepare_dataset(df, feature_cols=cols)
        return X, y, cols
    if config == "raw+temporal":
        df = load_raw_plus_temporal_features()
        df = build_presence_flags(df)
        cols = discover_feature_cols(df)
        X, y, _ = prepare_dataset(df, feature_cols=cols)
        return X, y, cols
    if config == "kitchen_sink":
        df = load_all_features()
        df = build_presence_flags(df)
        cols = discover_feature_cols(df)
        X, y, _ = prepare_dataset(df, feature_cols=cols)
        return X, y, cols
    raise ValueError(config)


# Order matches paper Table 4; each entry: (internal id, display label)
_ABLATION_CONFIGS = [
    ("deltas",         "Deltas only"),
    ("znorm",          "z-norm only"),
    ("raw",            "Raw (style + deltas)"),
    ("raw+embeddings", "Raw + embeddings"),
    ("raw+znorm",      "Raw + z-norm (combined)"),
    ("raw+temporal",   "Raw + temporal"),
    ("kitchen_sink",   "All (kitchen sink)"),
]


def plot_ablation_comparison() -> None:
    """Horizontal bar chart of all 8 ablation rows ranked by macro AUC.

    For each supervised configuration we run 5-fold stratified CV for
    both LogReg and Random Forest and report the stronger model's
    mean macro AUC +/- std across folds. The unsupervised PELT
    baseline is shown at the bottom as its hit-rate-at-+-2-weeks,
    with the random-CP null rate annotated for reference.
    """
    print("[visualize] Figure 4: computing per-fold AUCs for ablation...")
    rows: list[dict] = []

    for config_id, display in _ABLATION_CONFIGS:
        print(f"[visualize]   config={config_id}", flush=True)
        X, y, _ = _load_config_dataset(config_id)
        lr_aucs = _per_fold_macro_aucs(make_lr(), X, y)
        rf_aucs = _per_fold_macro_aucs(make_rf(), X, y)
        lr_mean = float(np.nanmean(lr_aucs))
        rf_mean = float(np.nanmean(rf_aucs))
        if rf_mean >= lr_mean:
            best_model, best_aucs = "RF", rf_aucs
        else:
            best_model, best_aucs = "LR", lr_aucs
        rows.append({
            "config":     display,
            "best_model": best_model,
            "mean_auc":   float(np.nanmean(best_aucs)),
            "std_auc":    float(np.nanstd(best_aucs, ddof=1)),
            "lr_mean":    lr_mean,
            "rf_mean":    rf_mean,
            "n_feats":    X.shape[1],
        })

    # Load PELT results -> shown as an unsupervised reference row
    with open(DATA_DIR / "pelt_baseline.json") as fh:
        pelt = json.load(fh)
    pelt_row = {
        "config":     "PELT hit@\u00b12w (unsup.)",
        "best_model": "--",
        "mean_auc":   float(pelt["hit_rate_all"]["2w"]),
        "std_auc":    0.0,
        "lr_mean":    float("nan"),
        "rf_mean":    float("nan"),
        "n_feats":    0,
        "is_pelt":    True,
        "null_rate":  float(pelt["random_null"]["2w"]),
    }

    # Rank supervised rows by mean AUC (ascending -> barh bottom-up)
    rows = sorted(rows, key=lambda r: r["mean_auc"])
    # PELT goes at the very bottom (worst)
    rows = [pelt_row] + rows

    # ── Plot ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    y_pos = np.arange(len(rows))
    means = np.array([r["mean_auc"] for r in rows])
    stds  = np.array([r["std_auc"]  for r in rows])

    colors = []
    for r in rows:
        if r.get("is_pelt"):
            colors.append("#999999")
        elif r["best_model"] == "RF":
            colors.append("#4c72b0")
        else:
            colors.append("#dd8452")

    ax.barh(
        y_pos, means,
        xerr=stds,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        error_kw={"ecolor": "#333333", "capsize": 3.0, "elinewidth": 1.0},
    )

    # Chance line at 0.5
    ax.axvline(0.5, color="black", linewidth=0.6, linestyle=":",
               zorder=1)
    ax.text(0.5, len(rows) - 0.35, " chance (0.50)",
            fontsize=8.5, color="#555555", va="center")

    # PELT null baseline
    if rows[0].get("is_pelt"):
        null = rows[0]["null_rate"]
        ax.axvline(null, color="#999999", linewidth=0.8, linestyle="--",
                   zorder=1)
        ax.text(null, -0.55, f" PELT null = {null:.3f}",
                fontsize=8, color="#777777", va="center")

    # Bar-end labels
    for i, r in enumerate(rows):
        suffix = ""
        if not r.get("is_pelt"):
            suffix = f"  [{r['best_model']}, n={r['n_feats']}]"
        ax.text(
            r["mean_auc"] + (r["std_auc"] or 0) + 0.006,
            i,
            f"{r['mean_auc']:.3f}{suffix}",
            va="center", ha="left", fontsize=9,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r["config"] for r in rows])
    ax.set_xlabel("Macro one-vs-rest ROC-AUC (5-fold CV, mean $\\pm$ std)")
    ax.set_title(
        "Feature-group ablation: ranked by classifier AUC",
        fontsize=12,
    )
    ax.set_xlim(0.0, 0.80)

    # Legend for colour coding
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#4c72b0", label="Random Forest (best)"),
        Patch(facecolor="#dd8452", label="Logistic Regression (best)"),
        Patch(facecolor="#999999", label="Unsupervised (PELT hit@$\\pm$2w)"),
    ]
    ax.legend(handles=legend_els, loc="lower right",
              fontsize=9, framealpha=0.9)

    # Footnote
    fig.text(
        0.5, -0.02,
        "Error bars show per-fold std; PELT is a single deterministic hit rate (no CV). "
        "n = feature count per configuration.",
        ha="center", fontsize=8, color="#555555",
    )

    _save(fig, "ablation_comparison.pdf")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    """Generate all four paper figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("[visualize] Loading data...")
    features = pd.read_parquet(FEATURES_IN)
    with open(RESULTS_IN) as fh:
        results = json.load(fh)

    print("[visualize] Figure 1: sentiment trajectory...")
    plot_sentiment_trajectory(features)

    print("[visualize] Figure 2: feature importance...")
    plot_feature_importance(results)

    print("[visualize] Figure 3: ROC curves...")
    plot_roc_curves(features)

    print("[visualize] Figure 4: ablation comparison...")
    plot_ablation_comparison()

    print("[visualize] All figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
