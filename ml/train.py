"""
Upper Valley Mental Health Finder — ML Training Script
=======================================================
Trains a provider-matching model on interaction data collected by the web app.

WORKFLOW
--------
1. Use the app and click providers, call them, rate them.
2. Click "Export training data" in the app — copy the JSON.
3. Save it to  ml/data/interactions.json
4. Run:  python ml/train.py
5. This writes  ml/model_weights.json
6. Redeploy (git commit + push) — the app auto-loads the weights.

FEATURES
--------
The feature vector is shared between this script and the browser inference
code in index.html (see computeMLAdjustment). If you add features here,
add the matching case to the JS switch statement.

REQUIREMENTS
------------
    pip install scikit-learn numpy
Optional (better performance):
    pip install xgboost shap
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent
INTERACTIONS  = Path(__file__).parent / "data" / "interactions.json"
WEIGHTS_OUT   = Path(__file__).parent / "model_weights.json"

# ── Feature names (MUST match index.html computeMLAdjustment) ──────────
FEATURE_NAMES = [
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

# ── Label weights by interaction type ──────────────────────────────────
# These encode how strongly each action signals "this was a good match."
LABEL_WEIGHTS = {
    "view":    0.3,   # expanded the card → mild positive
    "call":    1.0,   # called the provider → strong positive
    "website": 0.6,   # visited website → moderate positive
    "rate":    None,  # explicit rating overrides weight (see below)
}


# ── Load providers metadata ─────────────────────────────────────────────

def load_providers():
    """
    Parse provider records out of index.html JS.
    Falls back to an empty dict — features that need provider data will be 0.
    For best results, keep this in sync with the PROVIDERS array in index.html,
    or extract it to a separate ml/data/providers.json file.
    """
    providers = {}
    index_html = ROOT / "index.html"
    if not index_html.exists():
        return providers
    # Simple extraction — looks for id: 'xxx' blocks
    import re
    text = index_html.read_text(encoding="utf-8")
    # Extract provider IDs and some key flags
    ids = re.findall(r"id:\s*'([^']+)'", text)
    for pid in ids:
        # We only need rough boolean flags; extract what we can
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

def build_features(interaction: dict, providers: dict) -> list[float]:
    """Convert one interaction record to a feature vector."""
    f  = interaction.get("filters", {})
    pid = interaction.get("providerId", "")
    p  = providers.get(pid, {})

    ins        = f.get("insurance", "any")
    age        = f.get("ageGroup", "any")
    telehealth = f.get("telehealth", "any")
    urgency    = f.get("urgency", "exploring")
    language   = f.get("language", "any")
    gen_pref   = f.get("providerGender", "any")
    concerns   = f.get("concerns", [])
    modalities = f.get("modalities", [])
    cultural   = f.get("cultural", [])

    # Simplified overlap ratios — without full provider data we use proxy flags
    concerns_overlap  = interaction.get("concernsOverlap", 0.5)
    modality_overlap  = interaction.get("modalityOverlap", 0.5)
    cultural_overlap  = interaction.get("culturalOverlap", 0.5)

    vec = [
        1.0 if interaction.get("insuranceMatch", False) else 0.0,  # insurance_match
        1.0 if interaction.get("insurancePartial", False) else 0.0, # insurance_partial
        1.0 if interaction.get("ageMatch", False) else 0.0,         # age_match
        float(concerns_overlap),                                     # concerns_overlap
        1.0 if interaction.get("telehealthMatch", False) else 0.0,  # telehealth_match
        1.0 if (urgency == "crisis" and p.get("has_crisis", False)) else 0.0,  # crisis_match
        1.0 if interaction.get("languageMatch", False) else 0.0,    # language_match
        1.0 if interaction.get("genderMatch", False) else 0.0,      # gender_match
        float(modality_overlap),                                     # modality_overlap
        float(cultural_overlap),                                     # cultural_overlap
        1.0 if p.get("is_cmhc", False) else 0.0,                    # is_cmhc
        1.0 if p.get("is_fqhc", False) else 0.0,                    # is_fqhc
        1.0 if p.get("is_crisis_center", False) else 0.0,           # is_crisis_center
        1.0 if p.get("accepts_medicaid", False) else 0.0,           # accepts_medicaid
        1.0 if p.get("accepts_uninsured", False) else 0.0,          # accepts_uninsured
        1.0 if p.get("has_telehealth", False) else 0.0,             # has_telehealth
        1.0 if p.get("has_crisis", False) else 0.0,                 # has_crisis_services
        1.0 if urgency == "crisis" else 0.0,                        # urgency_is_crisis
        1.0 if urgency == "urgent" else 0.0,                        # urgency_is_urgent
        1.0 if ins in ("uninsured", "sliding_scale") else 0.0,      # user_uninsured
        1.0 if p.get("evening_hours", False) else 0.0,              # evening_hours
        1.0 if p.get("walk_in", False) else 0.0,                    # walk_in
    ]
    assert len(vec) == len(FEATURE_NAMES), \
        f"Feature count mismatch: {len(vec)} vs {len(FEATURE_NAMES)}"
    return vec


def build_label(interaction: dict) -> float:
    """
    Returns a label in [0, 1] representing how positive this interaction was.
    Explicit ratings (1–5 stars) are normalised to [0, 1].
    Action type provides the default signal.
    """
    itype  = interaction.get("type", "view")
    rating = interaction.get("rating")
    if rating is not None:
        return (float(rating) - 1.0) / 4.0   # 1→0, 3→0.5, 5→1
    return LABEL_WEIGHTS.get(itype, 0.3)


# ── Training ────────────────────────────────────────────────────────────

def train(interactions: list[dict], providers: dict):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    X = np.array([build_features(i, providers) for i in interactions], dtype=np.float32)
    y = np.array([build_label(i)               for i in interactions], dtype=np.float32)

    print(f"  Samples : {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Label range: [{y.min():.2f}, {y.max():.2f}]  mean={y.mean():.3f}")

    if len(X) < 5:
        print("\n⚠  Very few samples — model will be noisy. Collect more interactions.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)
    pred = model.predict(X_scaled)
    residuals = y - pred
    rmse = math.sqrt((residuals ** 2).mean())
    print(f"  Train RMSE: {rmse:.4f}")

    # Try XGBoost if available (usually better with small data)
    try:
        import xgboost as xgb
        xmodel = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", random_state=42, verbosity=0,
        )
        xmodel.fit(X_scaled, y)
        xpred = xmodel.predict(X_scaled)
        xrmse = math.sqrt(((y - xpred) ** 2).mean())
        print(f"  XGBoost RMSE: {xrmse:.4f} (using XGBoost)")
        if xrmse < rmse:
            # Export XGBoost leaf-value approximation as linear weights
            importances = xmodel.feature_importances_
            # Use linear approximation: weight each feature by importance × correlation with residual
            weights = []
            for j in range(X_scaled.shape[1]):
                corr = np.corrcoef(X_scaled[:, j], y)[0, 1] if X_scaled[:, j].std() > 0 else 0.0
                weights.append(float(importances[j] * corr * 2.0))
            bias = float(y.mean() - np.dot(np.array(weights), X_scaled.mean(axis=0)))
            model_type = "xgboost_approx"
        else:
            weights = [float(w) for w in model.coef_]
            bias    = float(model.intercept_)
            model_type = "ridge"
    except ImportError:
        weights = [float(w) for w in model.coef_]
        bias    = float(model.intercept_)
        model_type = "ridge"

    return {
        "version":          "1.0",
        "model_type":       model_type,
        "feature_names":    FEATURE_NAMES,
        "weights":          weights,
        "bias":             bias,
        "scaler": {
            "mean": [float(m) for m in scaler.mean_],
            "std":  [float(s) for s in scaler.scale_],
        },
        "training_samples": len(X),
        "train_rmse":       round(rmse, 5),
    }


# ── Feature importance ──────────────────────────────────────────────────

def print_importance(model_data: dict):
    pairs = sorted(zip(model_data["feature_names"], model_data["weights"]),
                   key=lambda x: abs(x[1]), reverse=True)
    print("\n  Feature importances (by |weight|):")
    for name, w in pairs[:10]:
        bar = "█" * int(abs(w) * 40) or "·"
        sign = "+" if w >= 0 else "-"
        print(f"    {sign} {name:<28} {bar}  ({w:+.4f})")


# ── Entry point ─────────────────────────────────────────────────────────

def main():
    print("═" * 55)
    print("  Upper Valley MH Finder — ML Training")
    print("═" * 55)

    if not INTERACTIONS.exists():
        print(f"\n⚠  No interaction data found at:\n   {INTERACTIONS}")
        print("\nTo collect data:")
        print("  1. Use the app at https://benjaminhannan.github.io/uv-mental-health-finder/")
        print("  2. Click providers, call numbers, rate results")
        print("  3. Click 'Export training data' → copy JSON")
        print(f"  4. Save to  {INTERACTIONS}")
        print("  5. Re-run this script\n")
        print("Generating PLACEHOLDER weights instead (uniform zeros)…")
        placeholder = {
            "version":          "placeholder",
            "model_type":       "placeholder",
            "feature_names":    FEATURE_NAMES,
            "weights":          [0.0] * len(FEATURE_NAMES),
            "bias":             0.0,
            "scaler":           {"mean": [0.0] * len(FEATURE_NAMES), "std": [1.0] * len(FEATURE_NAMES)},
            "training_samples": 0,
            "train_rmse":       None,
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

    print("\nTraining…")
    model_data = train(interactions, providers)

    print_importance(model_data)

    WEIGHTS_OUT.write_text(json.dumps(model_data, indent=2))
    print(f"\n✓ Wrote model weights → {WEIGHTS_OUT}")
    print(f"  Model type     : {model_data['model_type']}")
    print(f"  Training samples: {model_data['training_samples']}")
    print(f"  Train RMSE     : {model_data['train_rmse']}")
    print("\nNext steps:")
    print("  git add ml/model_weights.json && git commit -m 'Update ML weights'")
    print("  git push origin main")
    print("  The app will load the new weights automatically.\n")


if __name__ == "__main__":
    main()
