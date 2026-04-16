"""
Upper Valley MH Finder — Synthetic Training Data Generator
============================================================
Generates realistic interaction data via weak supervision to bootstrap
the ML model from zero real user data.

The generator creates synthetic "sessions" where a simulated user with
random filter preferences interacts with providers. The rule-based scoring
system provides noisy labels, simulating what real engagement would look like.

Usage:
    python ml/generate_synthetic.py          # generates 500 interactions
    python ml/generate_synthetic.py 1000     # generates 1000 interactions
    python ml/train.py                       # then train on the synthetic data
"""

import json
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = Path(__file__).parent / "data"
OUTPUT = DATA_DIR / "interactions.json"

# ── Provider catalog (must match index.html PROVIDERS array) ──────────
PROVIDERS = [
    {"id": "wcbh",       "insurance": ["medicaid", "medicare", "bcbs", "sliding_scale"], "ages": ["child", "teen", "adult", "senior"], "telehealth": True,  "crisis": True,  "concerns": ["depression", "anxiety", "bipolar", "schizophrenia", "trauma", "substance_use", "ocd"], "modalities": ["cbt", "dbt", "mi", "trauma_focused"], "cultural": ["lgbtq_affirming", "veteran_friendly"], "type": "cmhc", "languages": ["english", "spanish"]},
    {"id": "hcrs",       "insurance": ["medicaid", "medicare", "bcbs", "cigna", "sliding_scale"], "ages": ["child", "teen", "adult", "senior"], "telehealth": True,  "crisis": True,  "concerns": ["depression", "anxiety", "trauma", "substance_use", "grief", "family"], "modalities": ["cbt", "mi", "family_therapy"], "cultural": ["lgbtq_affirming"], "type": "cmhc", "languages": ["english"]},
    {"id": "headrest",   "insurance": ["medicaid", "medicare", "sliding_scale", "uninsured"], "ages": ["teen", "adult", "senior"], "telehealth": False, "crisis": True,  "concerns": ["substance_use", "addiction", "opioid_use"], "modalities": ["mi", "group_therapy", "mat"], "cultural": [], "type": "crisis", "languages": ["english"]},
    {"id": "wp",         "insurance": ["medicaid", "bcbs", "cigna", "aetna", "uhc"], "ages": ["child", "teen", "adult"], "telehealth": True,  "crisis": False, "concerns": ["depression", "anxiety", "adhd", "autism", "learning_disabilities"], "modalities": ["cbt", "play_therapy", "aba"], "cultural": [], "type": "practice", "languages": ["english"]},
    {"id": "dhmc_psych", "insurance": ["bcbs", "cigna", "aetna", "uhc", "medicare"], "ages": ["adult", "senior"], "telehealth": True,  "crisis": True,  "concerns": ["depression", "anxiety", "bipolar", "schizophrenia", "ocd", "eating_disorders", "trauma"], "modalities": ["cbt", "dbt", "medication", "emdr"], "cultural": [], "type": "hospital", "languages": ["english", "spanish", "mandarin"]},
    {"id": "uwm",        "insurance": ["medicaid", "medicare", "bcbs", "sliding_scale"], "ages": ["child", "teen", "adult"], "telehealth": True,  "crisis": False, "concerns": ["depression", "anxiety", "trauma", "adhd", "family"], "modalities": ["cbt", "mi", "family_therapy"], "cultural": ["lgbtq_affirming"], "type": "cmhc", "languages": ["english"]},
    {"id": "vfp",        "insurance": ["bcbs", "cigna", "aetna"], "ages": ["adult"], "telehealth": True,  "crisis": False, "concerns": ["depression", "anxiety", "relationship", "grief", "self_esteem"], "modalities": ["cbt", "psychodynamic", "mindfulness"], "cultural": ["lgbtq_affirming", "feminist"], "type": "practice", "languages": ["english"]},
    {"id": "cfc",        "insurance": ["medicaid", "bcbs", "sliding_scale"], "ages": ["child", "teen"], "telehealth": True,  "crisis": False, "concerns": ["anxiety", "adhd", "autism", "behavioral", "family", "trauma"], "modalities": ["cbt", "play_therapy", "family_therapy", "aba"], "cultural": [], "type": "practice", "languages": ["english"]},
    {"id": "nami_vt",    "insurance": ["uninsured"], "ages": ["teen", "adult", "senior"], "telehealth": True,  "crisis": False, "concerns": ["depression", "anxiety", "bipolar", "schizophrenia", "family"], "modalities": ["peer_support", "group_therapy", "psychoeducation"], "cultural": ["veteran_friendly"], "type": "nonprofit", "languages": ["english"]},
    {"id": "giv",        "insurance": ["medicaid", "medicare", "bcbs", "cigna", "sliding_scale", "uninsured"], "ages": ["child", "teen", "adult", "senior"], "telehealth": True,  "crisis": True,  "concerns": ["crisis", "suicidal_ideation", "depression", "anxiety", "trauma"], "modalities": ["crisis_intervention"], "cultural": [], "type": "crisis", "languages": ["english"]},
    {"id": "uvbh",       "insurance": ["bcbs", "cigna", "aetna", "uhc"], "ages": ["adult", "senior"], "telehealth": True,  "crisis": False, "concerns": ["depression", "anxiety", "ocd", "trauma", "relationship", "anger"], "modalities": ["cbt", "emdr", "mindfulness", "dbt"], "cultural": ["lgbtq_affirming"], "type": "practice", "languages": ["english"]},
    {"id": "bwc",        "insurance": ["bcbs", "cigna", "sliding_scale"], "ages": ["adult"], "telehealth": True,  "crisis": False, "concerns": ["depression", "anxiety", "women_issues", "perinatal", "trauma", "grief"], "modalities": ["cbt", "emdr", "somatic"], "cultural": ["feminist"], "type": "practice", "languages": ["english"]},
    {"id": "tms",        "insurance": ["bcbs", "uhc", "medicare"], "ages": ["adult", "senior"], "telehealth": False, "crisis": False, "concerns": ["depression", "ocd", "anxiety"], "modalities": ["tms", "medication"], "cultural": [], "type": "clinic", "languages": ["english"]},
    {"id": "rcpc",       "insurance": ["bcbs", "cigna", "aetna", "uhc"], "ages": ["adult"], "telehealth": True,  "crisis": False, "concerns": ["substance_use", "addiction", "depression", "anxiety", "trauma"], "modalities": ["cbt", "mi", "mat", "group_therapy"], "cultural": ["veteran_friendly"], "type": "practice", "languages": ["english"]},
    {"id": "vnh",        "insurance": ["medicaid", "medicare", "sliding_scale"], "ages": ["senior"], "telehealth": False, "crisis": False, "concerns": ["depression", "anxiety", "dementia", "grief", "isolation"], "modalities": ["supportive", "reminiscence"], "cultural": [], "type": "home_health", "languages": ["english"]},
]

# ── Simulation parameters ─────────────────────────────────────────────
INSURANCE_OPTIONS = ["medicaid", "medicare", "bcbs", "cigna", "aetna", "uhc", "uninsured", "sliding_scale"]
INSURANCE_WEIGHTS = [0.25, 0.10, 0.20, 0.12, 0.08, 0.10, 0.10, 0.05]

AGE_OPTIONS = ["child", "teen", "adult", "senior"]
AGE_WEIGHTS = [0.10, 0.15, 0.55, 0.20]

URGENCY_OPTIONS = ["crisis", "urgent", "routine", "exploring"]
URGENCY_WEIGHTS = [0.08, 0.20, 0.45, 0.27]

TELEHEALTH_OPTIONS = ["required", "preferred", "in_person", "any"]
TELEHEALTH_WEIGHTS = [0.15, 0.30, 0.20, 0.35]

FORMAT_OPTIONS = ["individual", "group", "family", "couples", "any"]
FORMAT_WEIGHTS = [0.50, 0.05, 0.10, 0.05, 0.30]

LANGUAGE_OPTIONS = ["english", "spanish", "any"]
LANGUAGE_WEIGHTS = [0.70, 0.10, 0.20]

GENDER_OPTIONS = ["woman", "man", "nonbinary", "any"]
GENDER_WEIGHTS = [0.25, 0.10, 0.05, 0.60]

ALL_CONCERNS = [
    "depression", "anxiety", "trauma", "substance_use", "ocd", "bipolar",
    "adhd", "eating_disorders", "grief", "relationship", "family",
    "schizophrenia", "autism", "crisis", "suicidal_ideation", "anger",
    "self_esteem", "women_issues", "perinatal",
]
CONCERN_WEIGHTS = [
    0.22, 0.25, 0.12, 0.08, 0.05, 0.04,
    0.06, 0.03, 0.04, 0.03, 0.02,
    0.01, 0.01, 0.02, 0.01, 0.01,
    0.02, 0.01, 0.01,
]

ALL_MODALITIES = ["cbt", "dbt", "emdr", "mi", "mindfulness", "psychodynamic", "family_therapy", "group_therapy", "medication", "play_therapy"]
ALL_CULTURAL = ["lgbtq_affirming", "veteran_friendly", "feminist", "bipoc_affirming"]
ALL_ACCESSIBILITY = ["sliding_scale", "evening", "weekend", "walk_in", "wheelchair"]

INTERACTION_TYPES = ["view", "call", "website", "rate"]
TYPE_WEIGHTS = [0.45, 0.20, 0.20, 0.15]


def weighted_choice(options, weights):
    return random.choices(options, weights=weights, k=1)[0]


def random_subset(options, weights=None, min_k=1, max_k=3):
    k = random.randint(min_k, min(max_k, len(options)))
    if weights:
        chosen = set()
        while len(chosen) < k:
            chosen.add(weighted_choice(options, weights))
        return list(chosen)
    return random.sample(options, k)


def compute_rule_score(provider, filters):
    """Compute a simplified rule-based score (mimics the app's scoring)."""
    score = 0
    total = 100

    # Insurance (26pts)
    user_ins = filters["insurance"]
    if user_ins == "any" or user_ins in provider["insurance"]:
        score += 26
    elif "sliding_scale" in provider["insurance"] or "uninsured" in provider["insurance"]:
        score += 8

    # Age (18pts)
    if filters["ageGroup"] == "any" or filters["ageGroup"] in provider["ages"]:
        score += 18

    # Concerns (17pts)
    user_concerns = set(filters["concerns"])
    prov_concerns = set(provider["concerns"])
    if user_concerns:
        overlap = len(user_concerns & prov_concerns) / len(user_concerns)
        score += int(17 * overlap)
    else:
        score += 8

    # Telehealth (11pts)
    tele = filters["telehealth"]
    if tele == "any":
        score += 11
    elif tele == "required" and provider["telehealth"]:
        score += 11
    elif tele == "preferred" and provider["telehealth"]:
        score += 11
    elif tele == "preferred" and not provider["telehealth"]:
        score += 4
    elif tele == "in_person":
        score += 11

    # Urgency (9pts)
    urg = filters["urgency"]
    if urg == "crisis" and provider["crisis"]:
        score += 9
    elif urg == "urgent" and provider["crisis"]:
        score += 7
    elif urg in ("routine", "exploring"):
        score += 9

    # Session format (5pts) — simplified
    score += 5

    # Language (4pts)
    if filters["language"] == "any" or filters["language"] in provider.get("languages", ["english"]):
        score += 4

    # Gender (3pts) — simplified
    if filters["providerGender"] == "any":
        score += 3

    # Modalities (4pts)
    user_mods = set(filters.get("modalities", []))
    prov_mods = set(provider.get("modalities", []))
    if user_mods:
        mod_overlap = len(user_mods & prov_mods) / len(user_mods)
        score += int(4 * mod_overlap)
    else:
        score += 2

    # Cultural (2pts)
    user_cult = set(filters.get("cultural", []))
    prov_cult = set(provider.get("cultural", []))
    if user_cult:
        cult_overlap = len(user_cult & prov_cult) / len(user_cult)
        score += int(2 * cult_overlap)
    else:
        score += 1

    # Accessibility (1pt)
    score += 1

    return min(score, 100)


def compute_features(provider, filters, score):
    """Compute ML training features from a provider-filter pair."""
    user_ins = filters["insurance"]
    user_concerns = set(filters["concerns"])
    prov_concerns = set(provider["concerns"])
    user_mods = set(filters.get("modalities", []))
    prov_mods = set(provider.get("modalities", []))
    user_cult = set(filters.get("cultural", []))
    prov_cult = set(provider.get("cultural", []))

    concerns_overlap = len(user_concerns & prov_concerns) / len(user_concerns) if user_concerns else 0.5
    modality_overlap = len(user_mods & prov_mods) / len(user_mods) if user_mods else 0.5
    cultural_overlap = len(user_cult & prov_cult) / len(user_cult) if user_cult else 0.0

    return {
        "insuranceMatch":  user_ins == "any" or user_ins in provider["insurance"],
        "insurancePartial": user_ins not in provider["insurance"] and ("sliding_scale" in provider["insurance"] or "uninsured" in provider["insurance"]),
        "ageMatch":        filters["ageGroup"] == "any" or filters["ageGroup"] in provider["ages"],
        "concernsOverlap": round(concerns_overlap, 3),
        "telehealthMatch": filters["telehealth"] in ("any", "in_person") or (filters["telehealth"] in ("required", "preferred") and provider["telehealth"]),
        "languageMatch":   filters["language"] == "any" or filters["language"] in provider.get("languages", ["english"]),
        "genderMatch":     filters["providerGender"] == "any",
        "modalityOverlap": round(modality_overlap, 3),
        "culturalOverlap": round(cultural_overlap, 3),
    }


def simulate_engagement(score, interaction_type):
    """
    Simulate user engagement based on the rule-based score.
    Higher scores → more likely positive engagement.
    Adds realistic noise to simulate real user behavior.
    """
    # Base engagement probability from score
    base_prob = score / 100.0

    # Noise factor — real users are unpredictable
    noise = random.gauss(0, 0.15)
    engagement = base_prob + noise

    if interaction_type == "rate":
        # Convert engagement to 1-5 rating
        raw_rating = engagement * 4 + 1  # map [0,1] → [1,5]
        raw_rating += random.gauss(0, 0.8)  # rating noise
        rating = max(1, min(5, round(raw_rating)))
        return rating
    else:
        return None


def generate_interaction(timestamp):
    """Generate one synthetic interaction record."""
    # Random user preferences
    filters = {
        "insurance":      weighted_choice(INSURANCE_OPTIONS, INSURANCE_WEIGHTS),
        "ageGroup":       weighted_choice(AGE_OPTIONS, AGE_WEIGHTS),
        "telehealth":     weighted_choice(TELEHEALTH_OPTIONS, TELEHEALTH_WEIGHTS),
        "urgency":        weighted_choice(URGENCY_OPTIONS, URGENCY_WEIGHTS),
        "sessionFormat":  weighted_choice(FORMAT_OPTIONS, FORMAT_WEIGHTS),
        "language":       weighted_choice(LANGUAGE_OPTIONS, LANGUAGE_WEIGHTS),
        "providerGender": weighted_choice(GENDER_OPTIONS, GENDER_WEIGHTS),
        "concerns":       random_subset(ALL_CONCERNS, CONCERN_WEIGHTS, 1, 4),
        "modalities":     random_subset(ALL_MODALITIES, None, 0, 3) if random.random() > 0.4 else [],
        "cultural":       random_subset(ALL_CULTURAL, None, 0, 2) if random.random() > 0.7 else [],
        "accessibility":  random_subset(ALL_ACCESSIBILITY, None, 0, 2) if random.random() > 0.8 else [],
    }

    # Pick a provider (weighted by how well they match — users are more likely to interact with good matches)
    scored = [(p, compute_rule_score(p, filters)) for p in PROVIDERS]
    # Softmax-style selection: higher scores → more likely to be clicked
    temperatures = [math.exp(s / 30.0) for _, s in scored]
    total_temp = sum(temperatures)
    probs = [t / total_temp for t in temperatures]
    idx = random.choices(range(len(scored)), weights=probs, k=1)[0]
    provider, score = scored[idx]

    # Interaction type
    itype = weighted_choice(INTERACTION_TYPES, TYPE_WEIGHTS)
    rating = simulate_engagement(score, itype)

    # Compute features
    features = compute_features(provider, filters, score)

    record = {
        "ts":              timestamp.isoformat() + "Z",
        "type":            itype,
        "providerId":      provider["id"],
        "score":           score,
        **features,
        "filters":         filters,
    }
    if rating is not None:
        record["rating"] = rating

    return record


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    random.seed(42)

    print("═" * 55)
    print("  Synthetic Training Data Generator")
    print("═" * 55)
    print(f"\n  Generating {n} synthetic interactions…")

    start_time = datetime(2026, 1, 1)
    interactions = []

    for i in range(n):
        # Spread interactions over simulated time
        ts = start_time + timedelta(
            days=random.randint(0, 90),
            hours=random.randint(8, 22),
            minutes=random.randint(0, 59),
        )
        record = generate_interaction(ts)
        interactions.append(record)

    # Sort by timestamp
    interactions.sort(key=lambda r: r["ts"])

    # Stats
    types = {}
    for r in interactions:
        types[r["type"]] = types.get(r["type"], 0) + 1

    ratings = [r["rating"] for r in interactions if "rating" in r]
    scores  = [r["score"] for r in interactions]

    print(f"\n  ── Stats ──")
    print(f"  Total interactions: {len(interactions)}")
    for t, c in sorted(types.items()):
        print(f"    {t:<10} {c:>4} ({c/len(interactions)*100:.0f}%)")
    print(f"  Score range: [{min(scores)}, {max(scores)}]  mean={sum(scores)/len(scores):.1f}")
    if ratings:
        print(f"  Rating range: [{min(ratings)}, {max(ratings)}]  mean={sum(ratings)/len(ratings):.1f}")

    # Provider distribution
    prov_counts = {}
    for r in interactions:
        prov_counts[r["providerId"]] = prov_counts.get(r["providerId"], 0) + 1
    print(f"\n  Provider distribution:")
    for pid, c in sorted(prov_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {pid:<15} {c:>3} interactions")

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(interactions, indent=2))
    print(f"\n✓ Wrote {len(interactions)} interactions → {OUTPUT}")
    print(f"\nNext: python ml/train.py\n")


if __name__ == "__main__":
    main()
