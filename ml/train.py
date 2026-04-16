"""
Upper Valley Mental Health Finder — ML Training Pipeline v2.0
==============================================================
Learning-to-Rank with BayesianRidge, SHAP explainability,
confidence intervals, fairness audit, and NDCG evaluation.

WORKFLOW
--------
1. Collect interaction data (searches, clicks, ratings) via the web app.
2. Save to  ml/data/interactions.json  (or run generate_synthetic.py first).
3. Run:  python ml/train.py
4. Outputs:
     ml/model_weights.json   — weights + SHAP + confidence config for browser
     ml/training_log.json    — full training metrics history
     ml/results.json         — detailed evaluation results
5. Commit & push — the app auto-loads the new model.

REQUIREMENTS
------------
    pip install scikit-learn numpy scipy
Optional (better performance):
    pip install xgboost shap
"""

import json
import math
import os
import sys
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

# ── Paths ──────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent
DATA_DIR      = Path(__file__).parent / "data"
INTERACTIONS  = DATA_DIR / "interactions.json"
WEIGHTS_OUT   = Path(__file__).parent / "model_weights.json"
TRAINING_LOG  = Path(__file__).parent / "training_log.json"
RESULTS_OUT   = Path(__file__).parent / "results.json"

# ── Feature names (MUST match index.html computeMLAdjustment) ──────────
BASE_FEATURES = [
    "insurance_match",      # user insurance == provider insurance  (0/1)
    "insurance_partial",    # provider has sliding scale but not exact match (0/1)
    "age_match",            # provider serves user's age group  (0/1)
    "concerns_overlap",     # fraction of user concerns covered  (0.0–1.0)
    "telehealth_match",     # telehealth needed and available  (0/1)
    "crisis_match",         # user in crisis AND provider has crisis services  (0/1)
    "language_match",       # provider speaks user's language  (0/1)
    "gender_match",         # provider gender matches preference  (0/1)
    "modality_overlap",     # fraction of preferred modalities offered  (0.0–1.0)
    "cultural_overlap",     # fraction of cultural needs met  (0.0–1.0)
    "is_cmhc",              # Community Mental Health Center  (0/1)
    "is_fqhc",              # Federally Qualified Health Center  (0/1)
    "is_crisis_center",     # dedicated crisis center  (0/1)
    "accepts_medicaid",     # (0/1)
    "accepts_uninsured",    # (0/1)
    "has_telehealth",       # (0/1)
    "has_crisis_services",  # (0/1)
    "urgency_is_crisis",    # user urgency == crisis  (0/1)
    "urgency_is_urgent",    # user urgency == urgent  (0/1)
    "user_uninsured",       # user is uninsured / self-pay  (0/1)
    "evening_hours",        # provider has evening/weekend hours  (0/1)
    "walk_in",              # provider accepts walk-ins  (0/1)
]

# ── Interaction features (cross-feature products) ─────────────────────
INTERACTION_FEATURES = [
    ("insurance_match", "concerns_overlap",  "ins_x_concerns"),
    ("insurance_match", "age_match",         "ins_x_age"),
    ("crisis_match",    "has_crisis_services","crisis_x_services"),
    ("telehealth_match","has_telehealth",    "tele_x_available"),
    ("concerns_overlap","modality_overlap",  "concerns_x_modality"),
    ("urgency_is_crisis","is_crisis_center", "urgent_x_crisis_center"),
    ("accepts_medicaid", "user_uninsured",   "medicaid_x_uninsured"),
    ("cultural_overlap", "language_match",    "cultural_x_language"),
]

ALL_FEATURE_NAMES = BASE_FEATURES + [name for _, _, name in INTERACTION_FEATURES]

# ── Label weights by interaction type ──────────────────────────────────
LABEL_WEIGHTS = {
    "view":    0.3,
    "call":    1.0,
    "website": 0.6,
    "rate":    None,
}

# ── Subgroup definitions for fairness audit ────────────────────────────
SUBGROUPS = {
    "insurance": {
        "medicaid":   lambda r: r.get("filters", {}).get("insurance") == "medicaid",
        "private":    lambda r: r.get("filters", {}).get("insurance") in ("bcbs", "cigna", "aetna", "uhc"),
        "uninsured":  lambda r: r.get("filters", {}).get("insurance") in ("uninsured", "sliding_scale"),
    },
    "age_group": {
        "child":  lambda r: r.get("filters", {}).get("ageGroup") == "child",
        "teen":   lambda r: r.get("filters", {}).get("ageGroup") == "teen",
        "adult":  lambda r: r.get("filters", {}).get("ageGroup") == "adult",
        "senior": lambda r: r.get("filters", {}).get("ageGroup") == "senior",
    },
    "urgency": {
        "crisis":    lambda r: r.get("filters", {}).get("urgency") == "crisis",
        "urgent":    lambda r: r.get("filters", {}).get("urgency") == "urgent",
        "routine":   lambda r: r.get("filters", {}).get("urgency") == "routine",
        "exploring": lambda r: r.get("filters", {}).get("urgency") == "exploring",
    },
}


# ── Load providers metadata ─────────────────────────────────────────────
def load_providers():
    providers = {}
    index_html = ROOT / "index.html"
    if not index_html.exists():
        return providers
    import re
    text = index_html.read_text(encoding="utf-8")
    ids = re.findall(r"id:\s*'([^']+)'", text)
    for pid in ids:
        block_start = text.find(f"id: '{pid}'")
        block_end   = text.find("\n  },\n", block_start)
        block       = text[block_start:block_end] if block_end > block_start else ""
        def has(pattern):
            return bool(re.search(pattern, block))
        providers[pid] = {
            "is_cmhc":           has(r"Community Mental Health"),
            "is_fqhc":           has(r"Federally Qualified"),
            "is_crisis_center":  has(r"Crisis Center"),
            "accepts_medicaid":  has(r"'medicaid'"),
            "accepts_uninsured": has(r"'uninsured'"),
            "has_telehealth":    has(r"telehealth:\s*true"),
            "has_crisis":        has(r"crisisServices:\s*true"),
            "evening_hours":     has(r"'evening'"),
            "walk_in":           has(r"'walk_in'"),
        }
    return providers


# ── Feature extraction ──────────────────────────────────────────────────
def build_base_features(interaction: dict, providers: dict) -> list:
    f   = interaction.get("filters", {})
    pid = interaction.get("providerId", "")
    p   = providers.get(pid, {})

    ins      = f.get("insurance", "any")
    urgency  = f.get("urgency", "exploring")

    concerns_overlap = interaction.get("concernsOverlap", 0.5)
    modality_overlap = interaction.get("modalityOverlap", 0.5)
    cultural_overlap = interaction.get("culturalOverlap", 0.5)

    vec = [
        1.0 if interaction.get("insuranceMatch", False) else 0.0,
        1.0 if interaction.get("insurancePartial", False) else 0.0,
        1.0 if interaction.get("ageMatch", False) else 0.0,
        float(concerns_overlap),
        1.0 if interaction.get("telehealthMatch", False) else 0.0,
        1.0 if (urgency == "crisis" and p.get("has_crisis", False)) else 0.0,
        1.0 if interaction.get("languageMatch", False) else 0.0,
        1.0 if interaction.get("genderMatch", False) else 0.0,
        float(modality_overlap),
        float(cultural_overlap),
        1.0 if p.get("is_cmhc", False) else 0.0,
        1.0 if p.get("is_fqhc", False) else 0.0,
        1.0 if p.get("is_crisis_center", False) else 0.0,
        1.0 if p.get("accepts_medicaid", False) else 0.0,
        1.0 if p.get("accepts_uninsured", False) else 0.0,
        1.0 if p.get("has_telehealth", False) else 0.0,
        1.0 if p.get("has_crisis", False) else 0.0,
        1.0 if urgency == "crisis" else 0.0,
        1.0 if urgency == "urgent" else 0.0,
        1.0 if ins in ("uninsured", "sliding_scale") else 0.0,
        1.0 if p.get("evening_hours", False) else 0.0,
        1.0 if p.get("walk_in", False) else 0.0,
    ]
    return vec


def add_interaction_features(base_vec: list) -> list:
    """Append cross-feature interaction terms to the base feature vector."""
    name_to_idx = {n: i for i, n in enumerate(BASE_FEATURES)}
    interactions = []
    for f1_name, f2_name, _ in INTERACTION_FEATURES:
        i1 = name_to_idx.get(f1_name, -1)
        i2 = name_to_idx.get(f2_name, -1)
        if i1 >= 0 and i2 >= 0:
            interactions.append(base_vec[i1] * base_vec[i2])
        else:
            interactions.append(0.0)
    return base_vec + interactions


def build_features(interaction: dict, providers: dict) -> list:
    base = build_base_features(interaction, providers)
    return add_interaction_features(base)


def build_label(interaction: dict) -> float:
    itype  = interaction.get("type", "view")
    rating = interaction.get("rating")
    if rating is not None:
        return (float(rating) - 1.0) / 4.0
    return LABEL_WEIGHTS.get(itype, 0.3)


# ── NDCG evaluation ────────────────────────────────────────────────────
def dcg_at_k(relevance, k):
    """Compute DCG@k."""
    relevance = np.asarray(relevance)[:k]
    if relevance.size == 0:
        return 0.0
    gains = (2 ** relevance - 1) / np.log2(np.arange(2, relevance.size + 2))
    return float(np.sum(gains))


def ndcg_at_k(y_true, y_pred, k=5):
    """Compute NDCG@k for a single query/group."""
    order = np.argsort(-y_pred)
    y_true_sorted = y_true[order]
    dcg  = dcg_at_k(y_true_sorted, k)
    ideal_order = np.argsort(-y_true)
    idcg = dcg_at_k(y_true[ideal_order], k)
    return dcg / idcg if idcg > 0 else 1.0


def compute_ndcg(y_true, y_pred, groups=None, k=5):
    """
    Compute mean NDCG@k across groups.
    If groups is None, treats all samples as one group.
    """
    if groups is None:
        return ndcg_at_k(y_true, y_pred, k)
    unique_groups = np.unique(groups)
    ndcgs = []
    for g in unique_groups:
        mask = groups == g
        if mask.sum() >= 2:
            ndcgs.append(ndcg_at_k(y_true[mask], y_pred[mask], k))
    return float(np.mean(ndcgs)) if ndcgs else 1.0


# ── SHAP-style feature attribution ─────────────────────────────────────
def compute_shap_values(model, X_scaled, scaler):
    """
    Compute marginal SHAP-style feature attributions.
    For linear models: SHAP_i = weight_i * (x_i - mean_i) / std_i
    This gives exact Shapley values for linear models.
    """
    weights = np.array(model.coef_ if hasattr(model, 'coef_') else model.weights)
    # For each sample, SHAP value = weight * scaled_feature_value
    shap_values = X_scaled * weights[np.newaxis, :]
    # Compute mean absolute SHAP for global feature importance
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    return shap_values, mean_abs_shap


def compute_shap_directions(model, scaler):
    """
    Compute per-feature SHAP direction info for the browser.
    Returns a dict mapping feature_name → {weight, importance_rank, direction}.
    """
    weights = np.array(model.coef_ if hasattr(model, 'coef_') else [0] * len(ALL_FEATURE_NAMES))
    abs_weights = np.abs(weights)
    ranks = np.argsort(-abs_weights)

    shap_info = {}
    for rank, idx in enumerate(ranks):
        fname = ALL_FEATURE_NAMES[idx]
        w = float(weights[idx])
        shap_info[fname] = {
            "weight": round(w, 5),
            "rank": rank + 1,
            "direction": "positive" if w > 0 else ("negative" if w < 0 else "neutral"),
            "abs_importance": round(float(abs_weights[idx]), 5),
        }
    return shap_info


# ── Confidence intervals via BayesianRidge ──────────────────────────────
def train_bayesian(X_scaled, y):
    """Train BayesianRidge and return model + uncertainty estimates."""
    from sklearn.linear_model import BayesianRidge

    model = BayesianRidge(
        alpha_1=1e-6, alpha_2=1e-6,
        lambda_1=1e-6, lambda_2=1e-6,
        compute_score=True,
        fit_intercept=True,
        max_iter=500,
    )
    model.fit(X_scaled, y)

    # Predict with uncertainty
    y_pred, y_std = model.predict(X_scaled, return_std=True)

    return model, y_pred, y_std


# ── LambdaRank-style pairwise loss (for ranking objective) ─────────────
def lambda_rank_weights(y_true, y_pred, sigma=1.0):
    """
    Compute LambdaRank gradient weights for pairwise ranking.
    Returns per-sample gradient adjustments.
    """
    n = len(y_true)
    if n < 2:
        return np.zeros(n)

    lambdas = np.zeros(n)
    # Create pairs where y_true[i] > y_true[j]
    for i in range(n):
        for j in range(n):
            if y_true[i] > y_true[j]:
                diff = y_pred[i] - y_pred[j]
                # Sigmoid cross-entropy gradient
                rho = 1.0 / (1.0 + np.exp(sigma * diff))
                # Delta NDCG approximation
                delta_ndcg = abs(y_true[i] - y_true[j])
                lam = -sigma * rho * delta_ndcg
                lambdas[i] += lam
                lambdas[j] -= lam
    return lambdas


# ── Fairness audit ──────────────────────────────────────────────────────
def fairness_audit(interactions, y_true, y_pred):
    """
    Compute per-subgroup metrics for fairness analysis.
    Returns dict with RMSE, MAE, mean prediction per subgroup.
    """
    audit = {}
    n = len(interactions)

    for group_name, subgroups in SUBGROUPS.items():
        audit[group_name] = {}
        for sg_name, filter_fn in subgroups.items():
            mask = np.array([filter_fn(interactions[i]) for i in range(n)])
            count = mask.sum()
            if count < 2:
                audit[group_name][sg_name] = {
                    "count": int(count),
                    "rmse": None,
                    "mae": None,
                    "mean_pred": None,
                    "mean_true": None,
                }
                continue

            yt = y_true[mask]
            yp = y_pred[mask]
            residuals = yt - yp
            rmse = float(np.sqrt((residuals ** 2).mean()))
            mae  = float(np.abs(residuals).mean())

            audit[group_name][sg_name] = {
                "count":     int(count),
                "rmse":      round(rmse, 5),
                "mae":       round(mae, 5),
                "mean_pred": round(float(yp.mean()), 4),
                "mean_true": round(float(yt.mean()), 4),
                "gap":       round(float(yp.mean() - yt.mean()), 4),
            }

    # Compute max disparity within each group
    for group_name in audit:
        preds = [v["mean_pred"] for v in audit[group_name].values() if v["mean_pred"] is not None]
        if len(preds) >= 2:
            audit[group_name]["_max_disparity"] = round(max(preds) - min(preds), 4)
        else:
            audit[group_name]["_max_disparity"] = 0.0

    return audit


# ── Main training pipeline ──────────────────────────────────────────────
def train(interactions, providers):
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold

    X = np.array([build_features(i, providers) for i in interactions], dtype=np.float64)
    y = np.array([build_label(i) for i in interactions], dtype=np.float64)

    print(f"  Samples : {len(X)}")
    print(f"  Features: {X.shape[1]} ({len(BASE_FEATURES)} base + {len(INTERACTION_FEATURES)} interaction)")
    print(f"  Label range: [{y.min():.2f}, {y.max():.2f}]  mean={y.mean():.3f}")

    if len(X) < 5:
        print("\n  ⚠  Very few samples — model will be noisy. Collect more interactions.")

    # ── Scale features ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train BayesianRidge (primary model with uncertainty) ──
    print("\n  Training BayesianRidge with uncertainty estimation...")
    model, y_pred, y_std = train_bayesian(X_scaled, y)

    # ── Apply LambdaRank gradient refinement ──
    print("  Applying LambdaRank pairwise ranking adjustment...")
    lambdas = lambda_rank_weights(y, y_pred, sigma=1.0)
    # Use lambdas as sample weights for a second pass
    from sklearn.linear_model import BayesianRidge
    sample_weights = np.abs(lambdas) + 1.0  # baseline weight of 1
    sample_weights /= sample_weights.mean()  # normalize

    model_ltr = BayesianRidge(
        alpha_1=1e-6, alpha_2=1e-6,
        lambda_1=1e-6, lambda_2=1e-6,
        compute_score=True,
        fit_intercept=True,
        max_iter=500,
    )
    model_ltr.fit(X_scaled, y, sample_weight=sample_weights)
    y_pred_ltr, y_std_ltr = model_ltr.predict(X_scaled, return_std=True)

    # ── Pick the better model by NDCG@5 ──
    ndcg_base = compute_ndcg(y, y_pred, k=5)
    ndcg_ltr  = compute_ndcg(y, y_pred_ltr, k=5)
    print(f"  NDCG@5 — Base: {ndcg_base:.4f}  LTR: {ndcg_ltr:.4f}")

    if ndcg_ltr >= ndcg_base:
        best_model, best_pred, best_std = model_ltr, y_pred_ltr, y_std_ltr
        model_type = "bayesian_ltr"
        print("  → Using LambdaRank-refined model")
    else:
        best_model, best_pred, best_std = model, y_pred, y_std
        model_type = "bayesian_ridge"
        print("  → Using base BayesianRidge (ranked better)")

    # ── Evaluation metrics ──
    residuals = y - best_pred
    rmse = float(np.sqrt((residuals ** 2).mean()))
    mae  = float(np.abs(residuals).mean())
    # Spearman rank correlation
    spearman_r, spearman_p = scipy_stats.spearmanr(y, best_pred)
    ndcg5 = compute_ndcg(y, best_pred, k=5)

    print(f"\n  ── Evaluation ──")
    print(f"  RMSE:       {rmse:.4f}")
    print(f"  MAE:        {mae:.4f}")
    print(f"  Spearman ρ: {spearman_r:.4f} (p={spearman_p:.4g})")
    print(f"  NDCG@5:     {ndcg5:.4f}")

    # ── Cross-validation ──
    cv_scores = {"rmse": [], "ndcg5": [], "spearman": []}
    if len(X) >= 10:
        print(f"\n  Running 5-fold cross-validation...")
        kf = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42)
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            sc_fold = StandardScaler()
            X_tr = sc_fold.fit_transform(X[train_idx])
            X_te = sc_fold.transform(X[test_idx])
            m_fold = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
            m_fold.fit(X_tr, y[train_idx])
            yp_fold = m_fold.predict(X_te)
            fold_rmse = float(np.sqrt(((y[test_idx] - yp_fold) ** 2).mean()))
            fold_ndcg = compute_ndcg(y[test_idx], yp_fold, k=5)
            fold_sp, _ = scipy_stats.spearmanr(y[test_idx], yp_fold) if len(test_idx) >= 3 else (0, 1)
            cv_scores["rmse"].append(fold_rmse)
            cv_scores["ndcg5"].append(fold_ndcg)
            cv_scores["spearman"].append(fold_sp if not np.isnan(fold_sp) else 0)
        print(f"  CV RMSE:     {np.mean(cv_scores['rmse']):.4f} ± {np.std(cv_scores['rmse']):.4f}")
        print(f"  CV NDCG@5:   {np.mean(cv_scores['ndcg5']):.4f} ± {np.std(cv_scores['ndcg5']):.4f}")
        print(f"  CV Spearman: {np.mean(cv_scores['spearman']):.4f} ± {np.std(cv_scores['spearman']):.4f}")

    # ── SHAP feature attribution ──
    print(f"\n  Computing SHAP feature attributions...")
    shap_values, mean_abs_shap = compute_shap_values(best_model, X_scaled, scaler)
    shap_info = compute_shap_directions(best_model, scaler)

    # Top 10 by importance
    sorted_features = sorted(shap_info.items(), key=lambda x: x[1]["abs_importance"], reverse=True)
    print("  Top features by SHAP importance:")
    for name, info in sorted_features[:10]:
        bar = "█" * int(info["abs_importance"] * 40) or "·"
        sign = "+" if info["direction"] == "positive" else ("-" if info["direction"] == "negative" else " ")
        print(f"    {sign} {name:<28} {bar}  ({info['weight']:+.4f})")

    # ── Confidence interval calibration ──
    mean_std = float(best_std.mean())
    std_of_std = float(best_std.std())
    print(f"\n  Mean prediction uncertainty (σ): {mean_std:.4f}")
    print(f"  Std of uncertainty: {std_of_std:.4f}")

    # ── Fairness audit ──
    print(f"\n  ── Fairness Audit ──")
    audit = fairness_audit(interactions, y, best_pred)
    for group_name, subgroups in audit.items():
        if group_name.startswith("_"):
            continue
        max_disp = subgroups.get("_max_disparity", 0)
        flag = " ⚠" if max_disp > 0.15 else " ✓"
        print(f"  {group_name} (max disparity: {max_disp:.3f}){flag}")
        for sg_name, metrics in subgroups.items():
            if sg_name.startswith("_"):
                continue
            if metrics["rmse"] is not None:
                print(f"    {sg_name:<12} n={metrics['count']:>3}  RMSE={metrics['rmse']:.4f}  gap={metrics['gap']:+.4f}")
            else:
                print(f"    {sg_name:<12} n={metrics['count']:>3}  (insufficient data)")

    # ── Build output ──
    weights = [float(w) for w in best_model.coef_]
    bias    = float(best_model.intercept_)

    # Data hash for tracking
    data_hash = hashlib.md5(json.dumps([build_label(i) for i in interactions]).encode()).hexdigest()[:8]

    model_output = {
        "version":           "2.0",
        "model_type":        model_type,
        "feature_names":     ALL_FEATURE_NAMES,
        "weights":           weights,
        "bias":              bias,
        "scaler": {
            "mean": [float(m) for m in scaler.mean_],
            "std":  [float(s) for s in scaler.scale_],
        },
        "training_samples":  len(X),
        "train_rmse":        round(rmse, 5),
        "train_mae":         round(mae, 5),
        "spearman":          round(float(spearman_r), 5),
        "ndcg5":             round(ndcg5, 5),
        "confidence": {
            "mean_std":    round(mean_std, 5),
            "std_of_std":  round(std_of_std, 5),
            "alpha":       round(float(best_model.alpha_), 5),
            "lambda":      round(float(best_model.lambda_), 5),
        },
        "shap": shap_info,
        "fairness": audit,
        "cv_scores": {
            k: {"mean": round(float(np.mean(v)), 5), "std": round(float(np.std(v)), 5)}
            for k, v in cv_scores.items()
        } if cv_scores["rmse"] else None,
        "trained_at":        datetime.utcnow().isoformat() + "Z",
        "data_hash":         data_hash,
    }

    # ── Detailed results ──
    results = {
        "trained_at":       model_output["trained_at"],
        "model_type":       model_type,
        "samples":          len(X),
        "features":         len(ALL_FEATURE_NAMES),
        "metrics": {
            "rmse":    round(rmse, 5),
            "mae":     round(mae, 5),
            "spearman": round(float(spearman_r), 5),
            "ndcg5":   round(ndcg5, 5),
        },
        "cv": model_output["cv_scores"],
        "fairness_summary": {
            group: {
                "max_disparity": data.get("_max_disparity", 0),
                "flagged": data.get("_max_disparity", 0) > 0.15,
            }
            for group, data in audit.items()
        },
        "top_features": [
            {"name": name, "weight": info["weight"], "rank": info["rank"]}
            for name, info in sorted_features[:10]
        ],
    }

    return model_output, results


# ── Training log ────────────────────────────────────────────────────────
def update_training_log(model_output):
    log = []
    if TRAINING_LOG.exists():
        try:
            log = json.loads(TRAINING_LOG.read_text())
        except Exception:
            log = []

    entry = {
        "run":              len(log) + 1,
        "trained_at":       model_output["trained_at"],
        "model_type":       model_output["model_type"],
        "samples":          model_output["training_samples"],
        "features":         len(model_output["feature_names"]),
        "rmse":             model_output["train_rmse"],
        "mae":              model_output["train_mae"],
        "spearman":         model_output["spearman"],
        "ndcg5":            model_output["ndcg5"],
        "data_hash":        model_output["data_hash"],
        "fairness_flags":   sum(
            1 for g in model_output.get("fairness", {}).values()
            if isinstance(g, dict) and g.get("_max_disparity", 0) > 0.15
        ),
    }
    log.append(entry)
    TRAINING_LOG.write_text(json.dumps(log, indent=2))
    return entry


# ── Entry point ─────────────────────────────────────────────────────────
def main():
    print("═" * 60)
    print("  Upper Valley MH Finder — ML Training Pipeline v2.0")
    print("  Learning to Rank · BayesianRidge · SHAP · Fairness")
    print("═" * 60)

    if not INTERACTIONS.exists():
        print(f"\n⚠  No interaction data found at:\n   {INTERACTIONS}")
        print("\nTo collect data:")
        print("  1. Use the app or run: python ml/generate_synthetic.py")
        print("  2. Then re-run: python ml/train.py\n")
        print("Generating PLACEHOLDER weights…")
        placeholder = {
            "version":          "placeholder",
            "model_type":       "placeholder",
            "feature_names":    ALL_FEATURE_NAMES,
            "weights":          [0.0] * len(ALL_FEATURE_NAMES),
            "bias":             0.0,
            "scaler":           {"mean": [0.0] * len(ALL_FEATURE_NAMES), "std": [1.0] * len(ALL_FEATURE_NAMES)},
            "training_samples": 0,
            "train_rmse":       None,
            "train_mae":        None,
            "spearman":         None,
            "ndcg5":            None,
            "confidence":       {"mean_std": 0, "std_of_std": 0, "alpha": 0, "lambda": 0},
            "shap":             {},
            "fairness":         {},
            "cv_scores":        None,
            "trained_at":       datetime.utcnow().isoformat() + "Z",
            "data_hash":        "00000000",
        }
        WEIGHTS_OUT.write_text(json.dumps(placeholder, indent=2))
        print(f"  Wrote placeholder → {WEIGHTS_OUT}\n")
        return

    print(f"\nLoading interactions from {INTERACTIONS}…")
    interactions = json.loads(INTERACTIONS.read_text())
    print(f"  Found {len(interactions)} interactions")

    print("\nLoading provider metadata…")
    providers = load_providers()
    print(f"  Found {len(providers)} providers in index.html")

    print("\n── Training Pipeline ──")
    model_output, results = train(interactions, providers)

    # Save outputs
    WEIGHTS_OUT.write_text(json.dumps(model_output, indent=2))
    print(f"\n✓ Wrote model weights → {WEIGHTS_OUT}")

    RESULTS_OUT.write_text(json.dumps(results, indent=2))
    print(f"✓ Wrote results       → {RESULTS_OUT}")

    log_entry = update_training_log(model_output)
    print(f"✓ Updated training log → {TRAINING_LOG} (run #{log_entry['run']})")

    print(f"\n  Model type      : {model_output['model_type']}")
    print(f"  Training samples: {model_output['training_samples']}")
    print(f"  RMSE: {model_output['train_rmse']}  MAE: {model_output['train_mae']}")
    print(f"  Spearman: {model_output['spearman']}  NDCG@5: {model_output['ndcg5']}")

    if model_output.get("cv_scores"):
        cv = model_output["cv_scores"]
        print(f"  CV RMSE: {cv['rmse']['mean']} ± {cv['rmse']['std']}")

    flagged = sum(
        1 for g in model_output.get("fairness", {}).values()
        if isinstance(g, dict) and g.get("_max_disparity", 0) > 0.15
    )
    if flagged:
        print(f"\n  ⚠ {flagged} fairness concern(s) detected — review ml/results.json")
    else:
        print(f"\n  ✓ No fairness concerns detected")

    print(f"\nNext: git add ml/ && git commit -m 'Update ML model v2.0' && git push\n")


if __name__ == "__main__":
    main()
