# Model-improvements plan — pre-arXiv hardening

## Context

The knowledge-base audit (docs/knowledge-base.md) identified concrete methodological upgrades that would (a) close gaps CLPsych reviewers will definitely raise and (b) close gaps reviewers *might* raise. This plan sequences those changes by impact × effort, grounded in what the code actually does today (see findings at the bottom).

**Goal:** ship a version of the Random Forest + 127-feature pipeline that is defensible at CLPsych/arXiv — proper uncertainty quantification, preempt the OCD-dominance objection, add the MentalBERT baseline reviewers expect, and tighten the paper's framing from "demonstrates" to "pilot feasibility study."

**Out of scope:** hierarchical time-aware transformer (Hills et al. 2024 SoTA), open-vocabulary / LDA features, mixed-effects model, twitter-roberta sentiment. All deferred to a future-work section.

---

## Phase 1 — Metric additions (no model re-training)

These are pure reporting changes on existing predictions. Highest impact per line of code.

1. **Bootstrap 95% CIs on macro AUC and macro F1.**
   New file: `src/evaluate_uncertainty.py`.
   Resample the pooled OOF predictions with replacement (1000×), recompute macro AUC / F1 each time, report 2.5/97.5 percentile band. Write to `data/bootstrap_cis.json`.

2. **Label-permutation p-value on macro AUC.**
   Same file. Permute `y` 1000×, re-run 5-fold CV of the RF pipeline on each permutation (expensive — batch once, save). Report p = fraction of permuted AUCs ≥ observed. Write to `data/permutation_test.json`.

3. **PR-AUC per class + macro PR-AUC.**
   Add `average_precision_score` alongside `roc_auc_score` in `evaluate_cv()` (src/train_model.py, lines 199–210). Baseline is class prevalence (not 0.5) — note that in the output.

4. **Confusion matrix JSON + figure.**
   Save `sklearn.metrics.confusion_matrix(y, y_pred)` to `data/confusion_matrix.json` inside `evaluate_cv()`. Add a heatmap render to `src/visualize.py` → `paper/figures/confusion_matrix.pdf`.

5. **Permutation feature importance (replace / augment Gini).**
   In `rf_feature_importance()` (src/train_model.py, lines 248–262), use `sklearn.inspection.permutation_importance` with `n_repeats=30`. Gini is biased toward high-cardinality features — KB flags this. Keep Gini column for comparison, make permutation the primary.

**Deliverables:** `data/bootstrap_cis.json`, `data/permutation_test.json`, updated `data/model_results_kitchen_sink.json` with PR-AUC fields, `paper/figures/confusion_matrix.pdf`, updated feature-importance figure.

**Estimated runtime:** bootstrap ~1 min, permutation test ~15–30 min, everything else seconds.

---

## Phase 2 — Leave-one-subreddit-out CV

This preempts the #1 reviewer objection: "46% of labels are r/OCD, so you're detecting OCD-speak, not crisis."

1. **Add `dominant_subreddit` to `user_labels.parquet`.**
   In `src/label_users.py`: for each user, compute the mode of `subreddit` across their posts (breaking ties by post count → recency). For crisis/recovery users this will usually match `tp_subreddit`. For neither users this is how we assign a subreddit for LOSO.

2. **Propagate `dominant_subreddit` through to features parquets.**
   Merge it into `features.parquet`, `features_znorm.parquet`, `features_temporal.parquet`, `features_mentalbert.parquet` in their respective extract scripts.

3. **Add `--loso` mode to `train_model.py`.**
   Replace `StratifiedKFold` with `GroupKFold` grouped on `dominant_subreddit` (k = n_subreddits = 5). Report per-fold macro AUC/F1 with subreddit name. Save to `data/loso_results.json`.

4. **Train-on-OCD / test-on-rest (and vice versa).**
   Explicit two-way split. Saves to `data/ocd_transfer.json`.

**Deliverable:** `data/loso_results.json`, `data/ocd_transfer.json`, bar chart of per-subreddit macro F1 → `paper/figures/loso_performance.pdf`.

---

## Phase 3 — MentalBERT baseline verification

1. **Test each candidate in `MODEL_CANDIDATES`** (src/extract_mentalbert.py lines 46–51) explicitly and log which loads. If any MentalBERT variant is now ungated, re-run `extract_mentalbert.py` with it.

2. **Document in paper** the model actually used. If MentalBERT loaded: report MentalBERT results as primary. If still falling back to MPNet: keep MPNet but cite `ji2022mentalbert` and add a limitations line ("MentalBERT remained gated as of {date}; a MentalBERT-specific fine-tune is left to future work").

3. **Fine-tuned MentalBERT classifier** (stretch, only if time permits). Rather than using MentalBERT only for semantic-shift features, fine-tune the full MentalBERT-base-uncased on the user-level crisis/recovery/neither task with a simple [CLS] → linear-head setup over concatenated user text. 10 epochs, AdamW, lr=2e-5. This is the "neural baseline" reviewers expect.

**Deliverable:** if ungated — updated `features_mentalbert.parquet` + mention in methods; if stretch done — new model results row in `data/model_results_mentalbert_finetuned.json`.

---

## Phase 4 — Baseline-bucket sensitivity

Knowledge base says minimum 20–30 posts for reliable per-user z-score. Today there's no floor, and NaN σ cascades into NaN z-scores (quietly).

1. **Add a distribution plot** of `n_baseline_buckets` across users to `src/visualize.py`. How many users have fewer than 3 / 5 / 10?
2. **Sensitivity run:** rerun z-norm experiment with `n_baseline_buckets >= k` for k ∈ {3, 5, 10}. Report how AUC changes. Add a short paragraph to the paper.

**Deliverable:** `data/baseline_bucket_sensitivity.json`, `paper/figures/baseline_buckets_hist.pdf`.

---

## Phase 5 — Paper framing updates (paper/main.tex)

Re-frame from "demonstrates" to "pilot feasibility study." Cite the new `.bib` entries added in commit 1253aa0.

1. **Introduction** — cite `dechoudhury2016discovering` as direct methodological precedent. Reframe first-person-pronoun prediction using `tackman2019depression` (I-talk ≈ negative emotionality, not depression-specific).

2. **Related Work** — add paragraph on construct-validity tradition (`chancellor2020methods`, `ernala2019methodological`).

3. **Results** — report bootstrap CIs alongside point estimates; add PR-AUC; report LOSO performance; include confusion matrix figure.

4. **Limitations** — 5 new paragraphs:
   - Construct validity (`ernala2019methodological`, `chancellor2020methods`)
   - Sample size (`varoquaux2018cross`, `riley2020calculating`) — explicit 95% CI width
   - Cross-subreddit generalization (`harrigian2020models`) — report LOSO numbers
   - Proxy labels (keyword heuristics ≠ clinical)
   - Model card / out-of-scope uses

5. **Author block + Acknowledgments** — Benjamin Hannan + email; acknowledge Taka Khoo and Aditya Gaur.

6. **Replace wording** — a final pass for "demonstrates" / "detects" / "predictive features" per the KB §7.2 table.

---

## Phase 6 — Run everything + regenerate figures

1. Re-run `python src/train_model.py --all` with the updated `evaluate_cv`.
2. Run LOSO and save results.
3. Run bootstrap + permutation test.
4. Regenerate all `paper/figures/*.pdf`.
5. Commit with clear phase tags.

---

## Verification

- `python src/train_model.py --kitchen-sink` produces a `model_results_kitchen_sink.json` that includes `pr_auc` fields and a `confusion_matrix` key.
- `python src/evaluate_uncertainty.py` writes valid `bootstrap_cis.json` and `permutation_test.json`.
- `python src/train_model.py --loso` prints one row per held-out subreddit and saves `loso_results.json`.
- `pdflatex paper/main.tex` compiles without errors after citation insertions.
- `git log --oneline` shows one commit per phase, tagged `analysis:` or `paper:`.

---

## Findings from pre-plan audit (for reference)

- **Z-score leakage:** absent. Baseline stats computed from posts strictly before `tp_date - 4w`. No emergency fix needed.
- **Nested CV:** not strictly needed — hyperparameters hardcoded; no selection bias. Plan to document this explicitly in the paper rather than adding a grid search.
- **`tp_subreddit`:** exists per-user for crisis/recovery but not for "neither" users. `dominant_subreddit` (mode of user's post history) added in Phase 2 solves this.
- **MentalBERT fallback chain:** already tries 3 variants before MPNet. Just needs re-testing whether any of the 3 variants is now ungated.
- **Baseline buckets:** no minimum enforced; NaN σ silently propagates. Surfaced in Phase 4.

---

## Ordering rationale

Phase 1 is pure reporting — small code, big framing impact. Phase 2 needs feature extraction changes. Phase 3 might be a no-op (if MentalBERT still gated) or a full fine-tune (stretch). Phase 4 is diagnostic. Phase 5 ties it together in the paper. Phase 6 is the final rerun.

Do Phase 1 first — it unblocks honest reporting of everything else.
