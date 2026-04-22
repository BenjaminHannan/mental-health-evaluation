"""Compare every model & feature-set combination we now have, ranked by AUC."""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

# ── Collect everything ───────────────────────────────────────────────────
rows: list[tuple[str, str, float, float]] = []   # (model, features, auc, f1)

feature_sets = {
    "raw":          "model_results.json",
    "deltas":       "model_results_deltas.json",
    "znorm":        "model_results_znorm.json",
    "combined":     "model_results_combined.json",
    "temporal":     "model_results_temporal.json",
    "mentalbert":   "model_results_mentalbert.json",
    "bonus":        "model_results_bonus.json",
    "kitchen_sink": "model_results_kitchen_sink.json",
    "everything":   "model_results_everything.json",
}

for feat_name, fn in feature_sets.items():
    p = DATA / fn
    if not p.exists():
        continue
    d = json.loads(p.read_text())["results"]
    for model_name, datasets in d.items():
        m = datasets["full"]["macro"]
        rows.append((model_name, feat_name, m["roc_auc"], m["f1"]))

# Sequence model
seq_p = DATA / "sequence_model_results.json"
if seq_p.exists():
    s = json.loads(seq_p.read_text())["results"]
    rows.append(("BiLSTM-Attn", "weekly_seq", s["macro_auc"], s["macro_f1"]))

# PELT baseline (unsupervised)
pelt_p = DATA / "pelt_baseline.json"
if pelt_p.exists():
    pelt = json.loads(pelt_p.read_text())
    # PELT is not an AUC; express hit-rate as a pseudo-AUC for sorting
    rows.append(("PELT(unsup)", "weekly_sent",
                 pelt["hit_rate_all"]["2w"], float("nan")))

# ── Sort by AUC descending ───────────────────────────────────────────────
rows.sort(key=lambda r: r[2], reverse=True)

print(f"{'Rank':>4} {'Model':<14}{'Feature set':<16}{'macro AUC':>12}{'macro F1':>11}")
print("-" * 60)
for i, (model, feat, auc, f1) in enumerate(rows, 1):
    f1_disp = f"{f1:>11.4f}" if not (f1 != f1) else f"{'--':>11}"    # nan check
    print(f"{i:>4} {model:<14}{feat:<16}{auc:>12.4f}{f1_disp}")
