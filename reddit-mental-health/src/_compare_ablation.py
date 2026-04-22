"""Compare OLD (mpnet fallback) vs NEW (real MentalBERT) ablation AUCs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def auc(fn):
    d = json.loads((ROOT / "data" / fn).read_text())["results"]
    return (d["LogReg"]["full"]["macro"]["roc_auc"],
            d["RandForest"]["full"]["macro"]["roc_auc"])


rows = [
    ("raw",          "model_results.json"),
    ("deltas",       "model_results_deltas.json"),
    ("znorm",        "model_results_znorm.json"),
    ("combined",     "model_results_combined.json"),
    ("temporal",     "model_results_temporal.json"),
    ("mentalbert",   "model_results_mentalbert.json"),
    ("kitchen_sink", "model_results_kitchen_sink.json"),
]

old = json.loads((ROOT / "data" / "ablation_summary.json").read_text())

hdr = "config          OLD LR   NEW LR      dLR   OLD RF   NEW RF      dRF"
print(hdr)
print("-" * len(hdr))
for name, fn in rows:
    new_lr, new_rf = auc(fn)
    old_lr = old[name]["LogReg"]["full"]["auc"]
    old_rf = old[name]["RandForest"]["full"]["auc"]
    d_lr = new_lr - old_lr
    d_rf = new_rf - old_rf
    print(f"{name:<15}  {old_lr:>.4f}   {new_lr:>.4f}  {d_lr:>+.4f}   "
          f"{old_rf:>.4f}   {new_rf:>.4f}  {d_rf:>+.4f}")
