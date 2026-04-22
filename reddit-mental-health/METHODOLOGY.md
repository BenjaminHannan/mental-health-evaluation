# Methodology and Approach
## Longitudinal Linguistic Markers of Mental Health Deterioration: A Multi-Platform Sliding-Window Feature Study

**Author:** Benjamin Hannan  
**Target:** CLPsych Workshop (ACL) + arXiv preprint  

---

## 1. Research Question

> Can we detect statistically significant shifts in linguistic patterns in a user's posting history in the 2–4 weeks preceding a self-reported mental health crisis or recovery turning point — compared to their baseline posting behaviour — while controlling for stable individual writing style, so that we measure *change* rather than *identity*?

The core tension the paper resolves: a classifier that detects crisis or recovery users might be learning *who these people are* (their stable writing style, which subreddits they post in), or it might be learning *how they change* in the lead-up to the event. These are not the same thing. The methodology is designed to measure both and quantify their relative contributions.

---

## 2. Data

### 2.1 Sources

Data is drawn from two platforms to improve coverage and dataset size.

**Reddit** — The `solomonk/reddit_mental_health_posts` dataset, accessed via the HuggingFace `datasets` library. It contains **151,288 posts** from five subreddits spanning July 2019 – December 2021:

| Subreddit | Posts |
|---|---|
| r/ADHD | 26,904 |
| r/OCD | 24,902 |
| r/depression | 15,828 |
| r/aspergers | 14,542 |
| r/PTSD | 13,897 |

Additional Reddit user post histories are collected via the **Arctic Shift API** (`arctic-shift.photon-reddit.com`), which archives the full Pushshift Reddit corpus. For each target username, all available submissions and comments are retrieved across all subreddits and saved to a unified timeline.

**Tumblr** — Post histories are collected via the **Tumblr API v2** (`api.tumblr.com`). Users are discovered by searching specific mental health tags and their entire text-post history across all tags is retrieved. This provides a cross-platform validation sample with independent labelling.

### 2.2 Preprocessing

Posts are dropped if:
- Author field is `[deleted]` or `[removed]`
- Body text is empty, null, or removed
- Author or body field is null

### 2.3 Reddit Cohort Construction

**Inclusion criterion:** Each author must have **more than 10 posts** in the dataset.

**Turning-point labelling:** For each author, the first post whose concatenated title + body matches a crisis or recovery keyword phrase (case-insensitive) defines the turning-point date T:

| Label | Phrases |
|---|---|
| **Crisis** | "want to die", "wanting to die", "ending it", "end it all", "can't do this", "cant do this", "goodbye", "no point", "kill myself", "killing myself", "take my own life", "not worth living" |
| **Recovery** | "got help", "getting help", "starting therapy", "started therapy", "feeling better", "things are improving", "made it", "on the road to recovery", "doing better", "finally better" |

The user must also have at least one post in the baseline period (>4 weeks before T).

**Final Reddit cohort:**

| Label | N |
|---|---|
| Crisis | 73 |
| Recovery | 59 |
| Neither (control) | 449 |
| **Total** | **581** |

**Low-confidence flag:** The phrase "made it" is highly ambiguous. 33 of 59 recovery users were labelled solely via this phrase and are flagged `low_confidence` for sensitivity analysis.

### 2.4 Tumblr Cohort Construction

**Label assignment via distant supervision:** Users are discovered through tag searches and labelled according to which tags they post under, following the standard distant supervision approach used in CLPsych and related work:

| Tag group | Label | Tags used |
|---|---|---|
| Crisis | `crisis` | #suicidewatch, #suicidal, #suicidal thoughts, #want to die, #active self harm |
| Recovery | `recovery` | #mentalillnessrecovery, #mental health recovery, #depression recovery, #anxiety recovery, #selfharm recovery, #healing journey, #in recovery |
| Control | `neither` | #photography, #books, #art, #travel, #cooking, #gaming, #movies, #music, #nature |

Only unambiguous, high-signal tags are used. Broad diagnostic tags such as `#depression` or `#bpd` are deliberately excluded as they are too ambiguous to serve as reliable distant labels.

### 2.5 MentalBERT Label Verification

Tag-based labels are inherently noisy — a user tagging `#suicidal` could be writing from a position of lived experience, advocacy, or educational commentary. To mitigate this, each Tumblr user's label is cross-validated against the content of their posts using **zero-shot embedding similarity** via MentalBERT.

**Method:**
1. Embed all of a user's posts with MentalBERT and mean-pool to a single user vector
2. Embed a set of hand-written prototype sentences for each class (e.g. *"I want to kill myself tonight"* for crisis; *"therapy has been helping me cope"* for recovery) and mean-pool per class
3. Compute cosine similarity between the user vector and each class prototype
4. Assign a content-based label = argmax(similarity)
5. Compute a confidence margin = softmax difference between top two classes

A user is **kept** only if their tag-based label and content-based label agree AND the margin exceeds a minimum threshold. Users that fail are either dropped (strict mode) or flagged `low_confidence` (lenient mode).

This yields a **label verification report** with per-class agreement rates and a confusion matrix of tag labels vs. content labels.

---

## 3. Time Windows

For each user, four non-overlapping time windows are defined relative to T:

```
Past ←─────────────────────────────────────────────────→ T
│         BASELINE         │  pre_4w  │ pre_2w │ pre_1w │
│  (everything before T-4w)│ [T-4w,T) │[T-2w,T)│[T-1w,T)│
```

| Window | Definition |
|---|---|
| **Baseline** | All posts with timestamp < T − 4 weeks |
| **Pre-4w** | Posts in [T−4w, T) |
| **Pre-2w** | Posts in [T−2w, T) |
| **Pre-1w** | Posts in [T−1w, T) |

Pre-2w and pre-1w are subsets of pre-4w — they zoom in progressively closer to the turning point. Windows with no posts yield NaN values, handled by median imputation inside the classifier pipeline.

---

## 4. Feature Engineering

Six distinct feature groups are extracted, each capturing a different aspect of the signal.

### 4.1 Linguistic Features (7 per window × 4 windows = 28 features)

All computed from concatenated title + body text:

| Feature | Definition | Why it matters |
|---|---|---|
| `sentiment_mean` | Mean VADER compound score (−1 to +1) | Most direct mood proxy |
| `ttr` | Type-token ratio: unique tokens / total tokens | Lexical diversity; lower = more repetitive (associated with depression) |
| `avg_sent_len` | Mean words per sentence | Longer sentences in crisis (rumination), shorter in recovery |
| `post_freq` | Posts per week within the window | Withdrawal from posting activity precedes both crisis and recovery |
| `fp_pronoun_rate` | I/me/my/mine/myself as fraction of total tokens | Classic depression marker (self-focus) |
| `neg_affect_rate` | LIWC-inspired 120-word negative-affect list as fraction of total tokens | Emotional valence beyond sentiment |
| `avg_post_len` | Mean post length in words | General engagement level |

### 4.2 Delta Features (7 per pre-window × 3 pre-windows = 21 features)

For each pre-window w:

$$\Delta_w(f) = f_w - f_{\text{baseline}}$$

These capture *how much each feature changed* from the user's typical baseline. Three binary presence flags (`has_posts_pre_4w`, `has_posts_pre_2w`, `has_posts_pre_1w`) also capture information-in-silence.

### 4.3 Per-User Z-Score Normalisation (21 features)

**The problem with deltas:** Δ = pre − baseline ignores *how variable* the baseline is.

**The solution:** Split each user's baseline into non-overlapping 1-week buckets and z-score each pre-window value against bucket statistics:

$$z_w(f) = \frac{f_w - \mu_{\text{baseline buckets}}}{\sigma_{\text{baseline buckets}}}$$

This yields a **within-user normalised** representation, stripping out between-user style differences.

### 4.4 Temporal Posting-Behaviour Features (24 features + 18 deltas = 42 total)

Capture *when* a user posts, motivated by clinical findings linking disrupted circadian rhythm to depression and anxiety:

| Feature | Definition |
|---|---|
| `hour_entropy` | Normalised Shannon entropy of the hour-of-day distribution |
| `late_night_rate` | Fraction of posts between 00:00–04:00 UTC |
| `interval_mean_hr` | Mean time between consecutive posts (hours) |
| `interval_std_hr` | Standard deviation of inter-post intervals |
| `max_gap_hr` | Longest gap between consecutive posts |
| `weekend_rate` | Fraction of posts on Saturday or Sunday |

**Key finding:** `hour_entropy_baseline` is the single most important feature in every configuration that includes it, outranking all linguistic features.

### 4.5 Semantic-Shift Features from MentalBERT Embeddings (6 features)

Each post is embedded as a 768-dimensional vector using **MentalBERT** (`mental/mental-bert-base-uncased`), a BERT model fine-tuned on a large corpus of mental health text from Reddit. For each window, posts are mean-pooled to a centroid and the drift from the baseline centroid is measured:

$$\text{cos\_sim}_w = \frac{\mathbf{c}_{\text{baseline}} \cdot \mathbf{c}_w}{\|\mathbf{c}_{\text{baseline}}\| \|\mathbf{c}_w\|}$$
$$\text{l2\_dist}_w = \|\mathbf{c}_{\text{baseline}} - \mathbf{c}_w\|_2$$

This gives 6 scalar features (cos_sim and l2_dist for each of pre_4w, pre_2w, pre_1w) capturing whether a user is writing about meaningfully different topics in the pre-crisis period.

### 4.6 Bonus Hand-Engineered Features (8 per window × 4 windows = 32 + 24 deltas = 56 features)

Eight additional features motivated by the psycholinguistics literature:

| Feature | Definition | Why it matters |
|---|---|---|
| `flesch_reading_ease` | Flesch reading ease score (higher = easier) | Readability drops in acute crisis (shorter sentences, simpler words) |
| `flesch_kincaid_grade` | Flesch-Kincaid US grade level | Cross-validates reading ease |
| `exclaim_rate` | `!` per word | Exclamation spikes are common pre-crisis affect markers |
| `question_rate` | `?` per word | Increased questioning associated with rumination |
| `ellipsis_rate` | `...` occurrences per post | Ellipsis use signals trailing, unresolved thought |
| `caps_word_rate` | ALL-CAPS words (≥2 chars) per word | Informal intensity marker on social media |
| `i_vs_we_ratio` | log((#I+1) / (#we+1)) | Isolation signal: higher = more self-focused than collective |
| `i_vs_you_ratio` | log((#I+1) / (#you+1)) | Self-focus signal: higher = more self-focused than other-directed |

Delta features (pre-window minus baseline) are also computed for all 8.

**Key finding:** `flesch_kincaid_grade_pre_4w`, `i_vs_we_ratio_baseline`, and `question_rate_baseline` all rank in the top-15 Random Forest feature importances, confirming that isolation and readability signals are complementary to the core linguistic features.

---

## 5. Modelling

### 5.1 Task Framing

Three-class classification: **crisis / recovery / neither**.

This is harder than binary (crisis vs. not) because recovery and crisis are both minority classes, they are conceptually opposite, and the majority class (neither) is ~77% of the cohort.

### 5.2 Classifiers

Six classifiers are trained and compared:

**Logistic Regression (LR):**
- Multinomial LR with L2 regularisation, C = 0.5
- `class_weight = balanced`
- Features median-imputed then z-score scaled
- Interpretable; linear baseline

**Random Forest (RF):**
- 300 trees, max depth = 6
- `class_weight = balanced`
- Captures non-linear interactions

**CatBoost:**
- 500 iterations, depth = 5
- `auto_class_weights = Balanced`
- Native handling of missing values; strong on tabular data

**XGBoost:**
- 400 estimators, max depth = 4
- `tree_method = hist` for speed
- Class-weight scaling via `scale_pos_weight`

**LightGBM:**
- 400 estimators, 31 leaves
- `class_weight = balanced`
- Fastest of the gradient boosters; comparable AUC to XGBoost

**Stacking Ensemble:**
- Base learners: LR + RF + CatBoost (5-fold cross-val predictions)
- Meta-learner: Logistic Regression
- The meta-learner re-balances predicted probabilities across base learner disagreements, producing the best macro F1 of any single configuration

### 5.3 Hyperparameter Search

Optional grid search via `--hyperparam-search` flag, using `GridSearchCV` with `roc_auc_ovr` scoring and 3-fold inner CV. Grids cover:
- LR: C ∈ {0.1, 0.5, 1.0, 5.0}
- RF: n_estimators ∈ {200, 400}, max_depth ∈ {4, 6, 8}
- CatBoost/XGBoost: iterations ∈ {300, 500}, depth ∈ {4, 6}

### 5.4 Sequence Model (BiLSTM + Attention)

In addition to tabular classifiers, a deep sequence model is trained on the **weekly temporal ordering** of features — testing whether the trajectory of change (not just the aggregate delta) carries signal.

**Architecture:**
- Per user: a (max_weeks=40, 7) matrix of weekly feature vectors in chronological order
- Bidirectional LSTM (hidden size 64) → masked attention pooling (ignores empty weeks) → dropout(0.3) → dense(3)
- Per-user z-score standardisation computed on non-zero weeks
- Class-weighted cross-entropy loss
- 5-fold stratified CV, pooled OOF predictions

**Result:** BiLSTM+Attention achieves 0.6463 macro AUC — below the best tabular models. This is the expected pattern on small datasets; sequence models benefit from thousands of examples. The high fold-to-fold variance (0.47–0.75) confirms that 5-fold CV on ~500 users is noisy for a deep model. This is reported as a negative result, confirming that tabular gradient boosting is more appropriate at this dataset scale.

### 5.5 Evaluation

**5-fold stratified cross-validation** with pooled out-of-fold predictions.

**Primary metric:** Macro-averaged one-vs-rest ROC-AUC.

**Secondary metrics:** Per-class precision, recall, F1, AUC.

---

## 6. Ablation Study

Nine feature configurations are tested to isolate the contribution of each feature group:

| Configuration | Features included | What it tests |
|---|---|---|
| Deltas only | Δ features (21) + flags (3) | Pure within-user change signal |
| Z-norm only | Z-scored pre-window values (21) + flags (3) | Change signal, normalised for style |
| Raw | Raw linguistic (28) + deltas (21) + flags (3) | Combined style + change (main baseline) |
| Raw + embeddings | Raw (52) + semantic shift (6) | Does topic drift add? |
| Raw + z-norm | Raw (52) + z-norm (21) | Do normalised deltas add over raw? |
| Raw + temporal | Raw (52) + temporal (42) | Does posting behaviour help? |
| Kitchen sink | All of the above (~127 features) | Feature upper bound |
| Bonus | Raw + temporal + bonus (56) | Readability + pronoun ratios |
| Everything | Kitchen sink + bonus | Full information upper bound |

For each configuration, all six classifiers are evaluated. Results are ranked by macro AUC.

---

## 7. Unsupervised Baseline (PELT)

**PELT** (Pruned Exact Linear Time) change-point detection is tested as an unsupervised alternative. A weekly VADER sentiment time series is built per user; PELT detects change-points; hit rate at ±2 weeks of T is computed.

**Result:** PELT hit@±2w: **15.3%** vs. random null **11.4%** — only weakly above chance, motivating the supervised approach.

---

## 8. Sensitivity Analysis

All experiments are repeated on a **high-confidence dataset** excluding the 33 ambiguous "made it" recovery users (548 total).

**Key finding:** Recovery F1 drops sharply without the "made it" users, showing that (a) those users contributed most of the recovery signal, and (b) the recovery signal in the full dataset should be interpreted cautiously.

---

## 9. Pipeline Architecture

```
src/load_data.py                 → data/raw_posts.parquet
src/label_users.py               → data/user_timelines.parquet
                                   data/user_labels.parquet
src/collect_data.py              → data/user_timelines.parquet  (Arctic Shift / Reddit)
src/collect_tumblr.py            → data/user_timelines_tumblr.parquet
                                   data/user_labels_tumblr.parquet
src/verify_labels_mentalbert.py  → data/user_labels_tumblr_verified.parquet
                                   data/label_verification_report.json
src/merge_sources.py             → data/user_timelines_merged.parquet
                                   data/user_labels_merged.parquet
src/extract_features.py          → data/features_raw.parquet
src/extract_temporal.py          → data/features_temporal.parquet
src/extract_mentalbert.py        → data/mentalbert_embeddings.npz
                                   data/features_mentalbert.parquet
src/extract_bonus_features.py    → data/features_bonus.parquet
src/train_model.py               → data/model_results_*.json
src/sequence_model.py            → data/sequence_model_results.json
src/pelt_baseline.py             → data/pelt_baseline.json
src/visualize.py                 → paper/figures/*.pdf
src/_final_compare.py            → ranked AUC table (all models × all feature sets)
```

**Technology stack:** Python 3.10, `datasets` (HuggingFace), `pandas`, `numpy`, `nltk`, `vaderSentiment`, `scikit-learn`, `catboost`, `xgboost`, `lightgbm`, `torch` (CUDA), `transformers`, `sentence-transformers`, `textstat`, `ruptures`, `matplotlib`, `seaborn`.

---

## 10. Results Summary

Full ranking across all model × feature-set combinations (Reddit-only dataset, n=505):

| Rank | Model | Feature set | Macro AUC | Macro F1 |
|---|---|---|---|---|
| 1 | Random Forest | temporal | **0.7098** | 0.3866 |
| 2 | XGBoost | everything | 0.7048 | 0.3952 |
| 3 | CatBoost | everything | 0.7043 | 0.4063 |
| 4 | LightGBM | everything | 0.7040 | 0.3852 |
| 5 | Random Forest | kitchen_sink | 0.6995 | 0.3668 |
| 6 | Stacking | everything | 0.6839 | **0.4243** |
| 7 | Random Forest | everything | 0.6806 | 0.3414 |
| 8 | Random Forest | raw | 0.6762 | 0.3599 |
| 15 | BiLSTM+Attn | weekly_seq | 0.6463 | 0.3681 |
| 24 | PELT (unsup.) | weekly_sent | 0.1311 | — |

**Key takeaways:**

- **Headline number: 0.7098 macro AUC** (RF, temporal features) — a +0.034 improvement over the raw-features baseline (0.676)
- **Gradient boosters are the most consistent performers:** CatBoost, XGBoost, and LightGBM all cluster at ~0.704 on the full "everything" feature set — three independent implementations converging suggests this is a real signal, not variance
- **Stacking wins on macro F1 (0.4243):** the meta-learner redistributes probability mass toward minority classes, improving the clinical utility of the predictions
- **MentalBERT embeddings contribute modestly:** the mentalbert-only configuration reaches 0.663 AUC; it adds incremental value in the kitchen-sink and everything configurations
- **BiLSTM underperforms tabular models on this dataset size** — a clear negative result worth reporting; sequence models require larger corpora
- **Temporal/circadian features are the strongest single group:** `hour_entropy_baseline` tops feature importance rankings in every configuration that includes it

**Baseline comparison:**

| Dataset | Best AUC | Improvement over raw |
|---|---|---|
| Reddit-only (n=505) | 0.7098 | +0.034 |
| Reddit + Tumblr (pending) | TBD | TBD |

---

## 11. Ethical Considerations

- **No user identifiers** are published. All released artefacts are aggregate statistics or derived feature matrices only.
- **Keyword labels and tag-based labels are not clinical diagnoses.** Labels come from substring matching and distant supervision, not expert annotation. They must never be used to make inferences about specific individuals.
- **MentalBERT label verification** is applied to Tumblr data to reduce noise, but does not constitute clinical validation.
- **These are population-level correlates**, not individual risk scores. The model outputs are statistical patterns, not clinical predictions.
- **Potential for misuse.** A similar model could be used to surveil users without consent. The authors strongly oppose this application. Any live deployment would require institutional ethics approval, informed consent, and clinical oversight.
- **Data spans the COVID-19 pandemic** (July 2019 – December 2021 for Reddit), which may confound baseline behaviour and turning-point triggers in ways the study cannot disentangle.
- **Tumblr collection** retrieves only publicly visible posts. No private or direct-message content is accessed.
