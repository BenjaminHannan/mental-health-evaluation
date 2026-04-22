# Knowledge Base: Longitudinal Linguistic Markers of Mental Health Deterioration

**Project:** *A Multi-Platform Sliding-Window Feature Study* (Hannan, target CLPsych/ACL)
**Purpose:** Deep-notes reference for related work, comparisons, discussion, and limitations.
**Structure:** Each entry has citation, what it does, methods/datasets, metrics, quotable findings, how it connects to this project, limitations, and a "use in paper" tag.

---

## Table of contents

1. [CLPsych shared tasks and canonical benchmarks](#1-clpsych-shared-tasks-and-canonical-benchmarks)
2. [Foundational Reddit datasets and distant supervision](#2-foundational-reddit-datasets-and-distant-supervision)
3. [Longitudinal and temporal user modelling](#3-longitudinal-and-temporal-user-modelling)
4. [Change-point detection and diachronic methods](#4-change-point-detection-and-diachronic-methods)
5. [Clinical circadian and sleep evidence](#5-clinical-circadian-and-sleep-evidence)
6. [Domain-adaptive language models and generalisation](#6-domain-adaptive-language-models-and-generalisation)
7. [Gradient-boosted trees vs. deep tabular models](#7-gradient-boosted-trees-vs-deep-tabular-models)
8. [Ethics, distant supervision, reproducibility](#8-ethics-distant-supervision-reproducibility)
9. [COVID-19 and subreddit-level confounds](#9-covid-19-and-subreddit-level-confounds)
10. [Psycholinguistics foundations (LIWC, pronouns, TTR, VADER)](#10-psycholinguistics-foundations-liwc-pronouns-ttr-vader)
11. [Supporting references (condensed)](#11-supporting-references-condensed)
12. [Cross-cutting themes and tensions](#12-cross-cutting-themes-and-tensions)
13. [Citation playbook for each section of the paper](#13-citation-playbook-for-each-section-of-the-paper)

---

## 1. CLPsych shared tasks and canonical benchmarks

### 1.1 Coppersmith, Dredze, Harman, Hollingshead & Mitchell (2015) — CLPsych 2015 shared task

- **Citation:** Coppersmith, G., Dredze, M., Harman, C., Hollingshead, K., & Mitchell, M. (2015). CLPsych 2015 shared task: Depression and PTSD on Twitter. *Proceedings of the 2nd Workshop on Computational Linguistics and Clinical Psychology*, 31–39. ACL W15-1204.
- **Link:** https://aclanthology.org/W15-1204/
- **What it does:** Defines three binary classification tasks on Twitter — depression vs control, PTSD vs control, depression vs PTSD — using self-reported diagnosis regex ("I was diagnosed with X") with demographically matched controls.
- **Method / data:** ~1,800 diagnosed users + matched controls. Self-reported-diagnosis regex filtering. All public tweets from each user aggregated; features included unigram, character n-gram, LIWC, and topic features.
- **Metrics:** Average precision; best ~0.87 AP for depression-vs-control using mixture models with LDA topics.
- **Key finding:** Even simple linguistic features discriminate self-reporting users from controls with surprisingly high AP — but this signal conflates writing style with condition.
- **Relevance to this project:** Foundation for the self-reported-diagnosis distant-supervision paradigm that every Reddit mental-health dataset (RSDD, SMHD, this project's keyword labelling) inherits. The *failure mode* this project addresses — classifiers learning "who these people are" vs. "how they change" — was visible here but not resolved.
- **Limitations:** Only binary tasks; no temporal modelling; no recovery class; demographic matching but no clinical ground truth.
- **Use in paper:** Cite in Section 1 (motivation for self-report paradigm) and Section 2 (data lineage). Use as *foil* for the longitudinal framing.
- **Tag:** `citation`

### 1.2 Milne, Pink, Hachey & Calvo (2016) — CLPsych 2016 triage

- **Citation:** Milne, D. N., Pink, G., Hachey, B., & Calvo, R. A. (2016). CLPsych 2016 shared task: Triaging content in online peer-support forums. *CLPsych @ NAACL 2016*, 118–127. ACL W16-0312.
- **Link:** https://aclanthology.org/W16-0312/
- **What it does:** Four-level triage (green, amber, red, crisis) of ReachOut.com youth mental-health forum posts, framed as support for human moderators.
- **Method / data:** 947 train / 280 test labelled messages. 15 teams, 60 systems.
- **Metrics:** Macro F1 over amber/red/crisis; best team ~42 macro F1.
- **Key finding:** Crisis-class recall is the hardest bottleneck; imbalance dominates. Lexicon features plus SVM competitive with neural approaches at this dataset size.
- **Relevance:** The project's three-class (crisis/recovery/neither) framing directly echoes this multi-level urgency framing, with 77% "neither" ≈ ReachOut's green class imbalance.
- **Limitations:** Small, privately held dataset; post-level not user-level; no trajectory.
- **Use in paper:** Compare class-imbalance handling strategies in Section 5; cite as precedent for multi-class urgency.
- **Tag:** `comparison`

### 1.3 Shing et al. (2018) — UMD Reddit Suicidality dataset

- **Citation:** Shing, H.-C., Nair, S., Zirikly, A., Friedenberg, M., Daumé III, H., & Resnik, P. (2018). Expert, crowdsourced, and machine assessment of suicide risk via online postings. *CLPsych @ NAACL 2018*, 25–36. ACL W18-0603.
- **Link:** https://aclanthology.org/W18-0603/
- **What it does:** Introduces UMD Reddit Suicidality v1 with a four-point rubric (no/low/moderate/severe). Demonstrates meaningful clinician IRR on social-media posts for the first time.
- **Method / data:** 934 annotated users sampled from r/SuicideWatch and control subreddits. Per-user labels from expert rubric. Machine-assisted annotation workflow compared to crowd.
- **Metrics:** Expert-pair Krippendorff α ≈ 0.41–0.49 on 4-level scale; crowdsourced lower.
- **Key finding:** Expert agreement is moderate but real for suicide risk on Reddit posts; crowdsourcing can be calibrated with pair-wise rubrics; clinician-grade labels are feasible at scale.
- **Relevance:** Gold-standard precedent this project's *distant* labels (keyword and tag) should be positioned against. Justifies low_confidence flag and sensitivity analysis.
- **Limitations:** No longitudinal labels; single platform; English-only; consent practices still evolving.
- **Use in paper:** Cite in Section 2 (labelling) as the gold-standard contrast; cite in ethics (Section 11) as the consent/IRB model.
- **Tag:** `citation`

### 1.4 Zirikly, Resnik, Uzuner & Hollingshead (2019) — CLPsych 2019 suicide risk

- **Citation:** Zirikly, A., Resnik, P., Uzuner, Ö., & Hollingshead, K. (2019). CLPsych 2019 shared task: Predicting the degree of suicide risk in Reddit posts. *CLPsych @ NAACL 2019*, 24–33. ACL W19-3003.
- **Link:** https://aclanthology.org/W19-3003/
- **What it does:** Four-level suicide-risk prediction on UMD Reddit Suicidality, with subtasks A (using SuicideWatch posts only), B (non-SuicideWatch only), C (both).
- **Method / data:** 496 annotated users. 15 teams, 51 submissions.
- **Metrics:** Macro F1; best ~50–55% macro F1.
- **Key finding:** Non-SuicideWatch posts (subtask B) are much harder than SuicideWatch posts — a direct finding that *detecting change in ordinary behaviour* is fundamentally harder than detecting overt crisis posts.
- **Relevance:** Subtask B's difficulty motivates this project's baseline-normalised delta features — the signal is subtle and individual-variable. Comparable F1 ranges.
- **Limitations:** Static user-level labels (no trajectory); no recovery class.
- **Use in paper:** Primary baseline comparison; cite as precedent showing change-signal is low-SNR.
- **Tag:** `both`

### 1.5 Tsakalidis et al. (2022) — CLPsych 2022 moments of change

- **Citation:** Tsakalidis, A., Chim, J., Bilal, I. M., Zirikly, A., Atzil-Slonim, D., Nanni, F., Resnik, P., Gaur, M., Roy, K., Inkster, B., Leintz, J., & Liakata, M. (2022). Overview of the CLPsych 2022 shared task: Capturing moments of change in longitudinal user posts. *CLPsych @ NAACL 2022*, 184–198. ACL 2022.clpsych-1.16.
- **Link:** https://aclanthology.org/2022.clpsych-1.16/
- **What it does:** Defines *Switches* (abrupt mood change) and *Escalations* (gradual mood change) on TalkLife user timelines. Subtask A: post-level mood-change labelling. Subtask B: user-level suicide risk using mood-change features.
- **Method / data:** 500 annotated TalkLife timelines, ~18,700 posts. Gold annotations from psychology researchers.
- **Metrics:** Precision/recall/F1 for Switch and Escalation labels; macro F1 for user-level risk.
- **Key finding:** BERT+BiLSTM context models outperform per-post baselines. Mood-change features transfer to improving suicide-risk prediction.
- **Relevance:** **Closest prior shared task to this project.** Directly validates the sliding-window / within-user-change framing. The project extends it in three ways: (a) self-reported turning point instead of annotated mood change, (b) Reddit + Tumblr instead of TalkLife, (c) adds recovery class.
- **Limitations:** TalkLife is paywalled / hard to access; annotations are researcher-defined not self-reported.
- **Use in paper:** **Central related work.** Cite in Section 1 (motivation), Section 3 (time windows), Section 5 (modelling). Position this project as complementary: self-report turning-point signal vs. annotator-labelled mood change.
- **Tag:** `both` — primary comparison.

### 1.6 Tsakalidis, Liakata, Damoulas & Cristea (2022) — Identifying moments of change

- **Citation:** Tsakalidis, A., Nanni, F., Hills, A., Chim, J., Song, J., & Liakata, M. (2022). Identifying moments of change from longitudinal user text. *ACL 2022*, 4647–4660. arXiv:2205.05593.
- **Link:** https://arxiv.org/abs/2205.05593
- **What it does:** Methodological companion to the 2022 shared task: formalises Switches/Escalations, introduces the 500-timeline TalkLife dataset, and benchmarks models with contextual encoding.
- **Method / data:** BERT + BiLSTM with contextual aggregation over post sequences; ablations across context window sizes.
- **Metrics:** Precision/recall/F1 per class. Reports strong gains from including *neighbouring-post context*.
- **Key finding:** Context windows matter: a post's mood label depends on its neighbours, not just its content.
- **Relevance:** Methodological blueprint for this project's BiLSTM+Attention sequence model. Confirms that aggregated per-user features can outperform post-level prediction at small *n*.
- **Limitations:** 500 users is still small; TalkLife is a single, moderated platform.
- **Use in paper:** Cite in Section 5.4 (BiLSTM justification and negative-result framing).
- **Tag:** `both`

### 1.7 Chim et al. (2024) — CLPsych 2024 evidence extraction

- **Citation:** Chim, J., Tsakalidis, A., Gkoumas, D., Hills, A., Zirikly, A., Gaur, M., & Liakata, M. (2024). Overview of the CLPsych 2024 shared task: Leveraging large language models to identify evidence of suicidality risk in online posts. *CLPsych @ EACL 2024*. ACL 2024.clpsych-1.15.
- **Link:** https://aclanthology.org/2024.clpsych-1.15/
- **What it does:** Evidence-highlight extraction and summarisation from r/SuicideWatch user histories given pre-assigned risk levels (Low/Moderate/High).
- **Method / data:** 125 annotated users. LLM-centric submissions (GPT-3.5/4, Llama).
- **Metrics:** BERTScore, NLI-consistency, recall against gold highlights.
- **Key finding:** LLMs can extract evidence but hallucinate summaries; NLI-consistency metric is essential for reliability.
- **Relevance:** Frames the *interpretability* angle — this project's feature-importance analysis (e.g., hour_entropy_baseline) is a lighter-weight form of evidence highlighting.
- **Limitations:** Very small evaluation set; closed LLMs confound reproducibility.
- **Use in paper:** Cite briefly in Section 1 (recent CLPsych trajectory); cite in discussion (interpretability comparison).
- **Tag:** `citation`

### 1.8 Tseriotou, Chim et al. (2025) — CLPsych 2025 mental-health dynamics

- **Citation:** Tseriotou, T., Chim, J., Klein, A., Shamir, A., Dvir, G., Ali, I., Kennedy, C., Kohli, G. S., Hills, A., Zirikly, A., Atzil-Slonim, D., & Liakata, M. (2025). Overview of the CLPsych 2025 shared task: Capturing mental health dynamics from social media timelines. *CLPsych 2025* @ NAACL, Albuquerque, 193–217.
- **Link:** https://aclanthology.org/2025.clpsych-1/
- **What it does:** Self-state identification and classification on user timelines — who is the post about, what state are they in.
- **Method / data:** Annotated TalkLife/Reddit timelines; LLM and classical baselines.
- **Key finding:** Self-state framing adds orthogonal signal to mood-change detection.
- **Relevance:** Confirms the field is consolidating around *longitudinal, within-user dynamics* as the frontier — directly supports this project's positioning.
- **Use in paper:** Cite in Section 1 to show longitudinal user modelling is the live research frontier.
- **Tag:** `citation`

> **Caveat to avoid:** There was **no CLPsych 2023 shared task**. The sequence is 2015 → 2016 → 2017 → 2018 → 2019 → 2021 (community task, MacAvaney et al.) → 2022 → 2024 → 2025. Do not invent a 2023 task.

---

## 2. Foundational Reddit datasets and distant supervision

### 2.1 Yates, Cohan & Goharian (2017) — RSDD

- **Citation:** Yates, A., Cohan, A., & Goharian, N. (2017). Depression and self-harm risk assessment in online forums. *EMNLP 2017*, 2968–2978. arXiv:1709.01848. DOI 10.18653/v1/D17-1322.
- **Link:** https://arxiv.org/abs/1709.01848
- **What it does:** Introduces **Reddit Self-reported Depression Diagnosis (RSDD)** via high-precision regex filtering ("I was diagnosed with depression") + matched controls.
- **Method / data:** 9,210 diagnosed users + 107,274 controls, full post histories. CNN and feature-based baselines.
- **Metrics:** F1 on per-user depression classification; best CNN ~51 F1; inter-annotator validation of regex precision.
- **Key finding:** Diagnosis-regex filtering has ~90% precision when patterns are tight enough; CNN over user post sequences beats bag-of-words.
- **Relevance:** RSDD is the reference for distant-supervision regex labelling on Reddit — this project's keyword phrases ("want to die", "started therapy") use the same design pattern.
- **Limitations:** Labels are diagnoses, not states; no temporal segmentation; self-reports conflate disclosure style with condition.
- **Use in paper:** Cite in Section 2 (labelling methodology) and Section 11 (ethics — regex labels ≠ clinical diagnoses).
- **Tag:** `citation`

### 2.2 Cohan et al. (2018) — SMHD

- **Citation:** Cohan, A., Desmet, B., Yates, A., Soldaini, L., MacAvaney, S., & Goharian, N. (2018). SMHD: A large-scale resource for exploring online language usage for multiple mental health conditions. *COLING 2018*, 1485–1497. arXiv:1806.05258. ACL C18-1126.
- **Link:** https://arxiv.org/abs/1806.05258
- **What it does:** Extends RSDD from 1 to **9 mental-health conditions** (depression, ADHD, anxiety, bipolar, PTSD, autism, OCD, schizophrenia, eating disorder) using tightened diagnosis patterns.
- **Method / data:** 350K+ users; matched controls. Linguistic feature analysis + classification baselines.
- **Metrics:** Per-condition F1; multi-label variants. Depression ~53 F1; rarer conditions lower.
- **Key finding:** Conditions co-occur and language overlaps substantially; multi-label is harder than per-condition binary.
- **Relevance:** SMHD's conditions overlap exactly with this project's subreddits (ADHD, OCD, depression, PTSD, autism/aspergers). Cite as the scale benchmark.
- **Limitations:** Self-reports only; no temporal axis; no ground-truth clinical labels.
- **Use in paper:** Cite in Section 2.1 (subreddit choice) and compare scale (~350K SMHD users vs. this project's ~581 Reddit cohort).
- **Tag:** `citation`

### 2.3 Turcan & McKeown (2019) — Dreaddit

- **Citation:** Turcan, E., & McKeown, K. (2019). Dreaddit: A Reddit dataset for stress analysis in social media. *LOUHI @ EMNLP 2019*, 97–107. arXiv:1911.00133. ACL D19-6213.
- **Link:** https://arxiv.org/abs/1911.00133
- **What it does:** 190K Reddit posts from 5 community categories (interpersonal, financial, PTSD, social-anxiety, abuse) with 3,553 MTurk-labelled binary stress segments.
- **Method / data:** BERT, LR+unigrams, LR+LIWC+features baselines.
- **Metrics:** Best ~80 F1 with BERT.
- **Key finding:** Domain-specific stress detection is tractable with BERT; LIWC + linguistic features competitive.
- **Relevance:** Dreaddit is the canonical Reddit stress benchmark — cite as precedent for post-level Reddit mental-health labelling. MentalBERT is pretrained on Dreaddit-adjacent data.
- **Limitations:** Segment-level, not user-level; annotator reliability on MTurk; stress ≠ crisis.
- **Use in paper:** Cite in Section 2 (Reddit mental-health corpora) and Section 4 (LIWC feature justification).
- **Tag:** `citation`

### 2.4 De Choudhury & De (2014) — Reddit self-disclosure

- **Citation:** De Choudhury, M., & De, S. (2014). Mental health discourse on Reddit: Self-disclosure, social support, and anonymity. *ICWSM 2014*. DOI 10.1609/icwsm.v8i1.14526.
- **Link:** https://ojs.aaai.org/index.php/ICWSM/article/view/14526
- **What it does:** Early characterisation of mental-health subreddits; shows throwaway-account use correlates with deeper self-disclosure.
- **Method / data:** Posts from r/depression, r/mentalhealth, r/suicidewatch. LIWC + topic features.
- **Key finding:** Anonymity → deeper disclosure; subreddit norms differ markedly.
- **Relevance:** Motivates Reddit as a venue; explains why keyword matching on "want to die" is detectable on Reddit but would be less so on Facebook. Supports the Section 11 ethics argument.
- **Use in paper:** Section 1 (motivation) and Section 11 (ethics framing).
- **Tag:** `citation`

### 2.5 De Choudhury, Kiciman, Dredze, Coppersmith & Kumar (2016) — Discovering shifts to suicidal ideation

- **Citation:** De Choudhury, M., Kiciman, E., Dredze, M., Coppersmith, G., & Kumar, M. (2016). Discovering shifts to suicidal ideation from mental health content in social media. *CHI 2016*, 2098–2110. DOI 10.1145/2858036.2858207.
- **Link:** https://dl.acm.org/doi/10.1145/2858036.2858207
- **What it does:** **The closest prior Reddit longitudinal-shift study to this project.** Uses propensity-score matching on Reddit mental-health users to identify those who transition to r/SuicideWatch, then characterises linguistic precursors.
- **Method / data:** ~20K Reddit users. Propensity-matched case/control design. Features: LIWC categories, interpersonal pronouns, lexical complexity, self-focus, social engagement.
- **Metrics:** ~81% classification accuracy for predicting shift vs non-shift.
- **Key findings:** Pre-shift users show (a) **reduced linguistic coordination with interlocutors**, (b) **increased self-focus** (I/me/my), (c) **decreased social engagement**, (d) **increased hopelessness** and (e) rising **nocturnal activity**.
- **Relevance:** **This project's intellectual ancestor.** The temporal windows, within-user baseline concept, and first-person pronoun focus all originate here. Position this project as *methodological extension* (adds z-score normalisation, recovery class, circadian entropy, multi-platform).
- **Limitations:** Binary shift vs no-shift (no recovery); single platform; subreddit-move as proxy for actual ideation; confounded by posting selection.
- **Use in paper:** **Primary citation in Section 1 motivation and Section 4 feature justification.** Discussion should note this project's improvements over its static-delta framing.
- **Tag:** `both` — foundational comparison.

### 2.6 Losada & Crestani (2016) — eRisk foundation

- **Citation:** Losada, D. E., & Crestani, F. (2016). A test collection for research on depression and language use. *CLEF 2016*, LNCS 9822, 28–39. DOI 10.1007/978-3-319-44564-9_3.
- **Link:** https://link.springer.com/chapter/10.1007/978-3-319-44564-9_3
- **What it does:** Introduces the sequential Reddit depression test collection and the ERDE latency-aware evaluation metric (penalises late predictions).
- **Method / data:** ~900 users; Reddit post histories in chronological order.
- **Key finding:** Early detection is a distinct task from post-hoc classification — evaluation must reward timeliness.
- **Relevance:** Closest cousin task to this project's pre-window prediction. ERDE concept supports framing predictions in the 2–4 weeks *before* T as more valuable than post-hoc.
- **Use in paper:** Cite in Section 5.5 (metric discussion) as an alternative metric worth reporting in future work.
- **Tag:** `citation`

### 2.7 Baumgartner, Zannettou, Keegan, Squire & Blackburn (2020) — Pushshift

- **Citation:** Baumgartner, J., Zannettou, S., Keegan, B., Squire, M., & Blackburn, J. (2020). The Pushshift Reddit dataset. *ICWSM 2020*, 830–839. DOI 10.1609/icwsm.v14i1.7347.
- **Link:** https://ojs.aaai.org/index.php/ICWSM/article/view/7347
- **What it does:** Documents the ingestion and backend of the Pushshift archive (651M submissions + 5.6B comments, 2005–2019) that powers almost all Reddit research.
- **Relevance:** This project uses the Arctic Shift API, which mirrors Pushshift data. Cite for data provenance.
- **Limitations:** Since 2023, Reddit API changes have complicated Pushshift access; Arctic Shift is a practical mirror.
- **Use in paper:** Section 2.1 data provenance and Section 9 (pipeline) reproducibility.
- **Tag:** `citation`

### 2.8 Coppersmith, Dredze & Harman (2014) — Quantifying mental health signals on Twitter

- **Citation:** Coppersmith, G., Dredze, M., & Harman, C. (2014). Quantifying mental health signals in Twitter. *CLPsych @ ACL 2014*, 51–60. ACL W14-3207.
- **Link:** https://aclanthology.org/W14-3207/
- **What it does:** Originator of the self-reported-diagnosis regex paradigm ("I was diagnosed with ..."). Applied to Twitter, later transferred to Reddit.
- **Relevance:** Intellectual origin of distant-supervision labelling in mental-health NLP.
- **Use in paper:** Section 2 (labelling).
- **Tag:** `citation`

### 2.9 Gkotsis, Oellrich, Velupillai, Liakata, Hubbard, Dobson & Dutta (2017) — Characterisation of mental health conditions on Reddit

- **Citation:** Gkotsis, G., Oellrich, A., Velupillai, S., Liakata, M., Hubbard, T., Dobson, R., & Dutta, R. (2017). Characterisation of mental health conditions in social media using informed deep learning. *Scientific Reports 7*, 45141. DOI 10.1038/srep45141.
- **Link:** https://www.nature.com/articles/srep45141
- **What it does:** Early Reddit-wide deep-learning characterisation of 11 mental-health subreddits using CNN with clinical lexicon augmentation.
- **Key finding:** Domain-informed CNN outperforms vanilla CNN; lexicons still matter in deep-learning era.
- **Relevance:** Precedent for using subreddit posts as weak labels for condition-typical language. Justifies the choice of r/depression, r/PTSD, r/OCD, r/ADHD.
- **Use in paper:** Section 2.1 (subreddit selection rationale).
- **Tag:** `citation`

---

## 3. Longitudinal and temporal user modelling

### 3.1 Sawhney, Joshi, Gandhi & Shah (2020) — STATENet

- **Citation:** Sawhney, R., Joshi, H., Gandhi, S., & Shah, R. R. (2020). A time-aware transformer based model for suicide ideation detection on social media. *EMNLP 2020*, 7685–7697. ACL 2020.emnlp-main.619.
- **Link:** https://aclanthology.org/2020.emnlp-main.619/
- **What it does:** **The canonical temporal baseline** for user-history mental-health modelling. Encodes target tweet with SentenceBERT; encodes historical tweets via Plutchik-emotion transformer + **Time-Aware LSTM** (T-LSTM) that weights historical posts by recency.
- **Method / data:** Twitter suicide-risk classification. Constructs a dataset of "at-risk" tweets + user histories.
- **Metrics:** Outperforms prior art on precision, recall, F1 for suicide-ideation detection.
- **Key finding:** *Time decay over post history improves performance substantially* — the distance from a post to the target tweet matters.
- **Relevance:** **Primary architectural baseline** this project's BiLSTM+Attention echoes. The project's negative BiLSTM result should be discussed in light of STATENet's success — likely due to dataset size (n=505 vs. STATENet's larger Twitter set) and per-user aggregation granularity.
- **Limitations:** Twitter, not Reddit; binary classification; no recovery class; requires a flagged "target tweet" as anchor.
- **Use in paper:** **Central comparison in Section 5.4.** Cite to explain why the BiLSTM negative result is not surprising at n=505.
- **Tag:** `both` — primary comparison.

### 3.2 Sawhney, Joshi, Gandhi & Shah (2021) — SISMO (ordinal)

- **Citation:** Sawhney, R., Joshi, H., Gandhi, S., & Shah, R. R. (2021). Towards ordinal suicide ideation detection on social media. *WSDM 2021*, 22–30. DOI 10.1145/3437963.3441805.
- **Link:** https://dl.acm.org/doi/10.1145/3437963.3441805
- **What it does:** Hierarchical attention over longitudinal Reddit C-SSRS-annotated user timelines, with ordinal regression loss (not one-hot classification).
- **Key finding:** Ordinal loss outperforms categorical cross-entropy when classes have a natural order (none → low → moderate → severe).
- **Relevance:** This project's three classes (crisis / recovery / neither) are *not* ordinal (recovery and crisis are opposite poles), so categorical cross-entropy is correct — but SISMO shows hierarchical attention works on Reddit longitudinal data.
- **Use in paper:** Section 5.4 as a relevant Reddit hierarchical attention baseline that outperforms flat attention.
- **Tag:** `comparison`

### 3.3 Sawhney, Joshi, Flek & Shah (2021) — PHASE (emotional phase-aware)

- **Citation:** Sawhney, R., Joshi, H., Flek, L., & Shah, R. R. (2021). PHASE: Learning emotional phase-aware representations for suicide ideation detection on social media. *EACL 2021*, 2415–2428.
- **Link:** https://aclanthology.org/2021.eacl-main.209/
- **What it does:** Encodes users' emotional *phase* (positive/neutral/negative trajectory pattern) via hyperbolic embeddings; exploits hierarchical structure of emotional states.
- **Relevance:** Phase-aware modelling is the conceptual cousin of PELT change-point detection — both look at piecewise emotional regimes rather than single-point signal.
- **Use in paper:** Section 7 (PELT baseline discussion) as a deep-learning analogue.
- **Tag:** `comparison`

### 3.4 Matero et al. (2019) — Dual-context BERT

- **Citation:** Matero, M., Idnani, A., Son, Y., Giorgi, S., Vu, H., Zamani, M., Limbachiya, P., Guntuku, S. C., & Schwartz, H. A. (2019). Suicide risk assessment with multi-level dual-context language and BERT. *CLPsych 2019*, 39–44. ACL W19-3005.
- **Link:** https://aclanthology.org/W19-3005/
- **What it does:** **Best system in CLPsych 2019.** Dual RNN over BERT embeddings separating SuicideWatch context from general context, fused with user-factor adaptation (age/gender) and psychological-theory features.
- **Metrics:** Best macro F1 on CLPsych 2019 suicide-risk task.
- **Key finding:** Separating context streams and augmenting with user factors improves performance meaningfully.
- **Relevance:** Dual-context idea could be extended to *baseline vs pre-window* context streams in this project — a natural future-work extension.
- **Use in paper:** Section 5 (related work on Reddit suicide risk) and Discussion (future work: baseline/pre-window dual-context).
- **Tag:** `comparison`

### 3.5 Cao et al. (2019) — Layered attention with suicide-oriented embeddings

- **Citation:** Cao, L., Zhang, H., Feng, L., Wei, Z., Wang, X., Li, N., & He, X. (2019). Latent suicide risk detection on microblog via suicide-oriented word embeddings and layered attention. *EMNLP-IJCNLP 2019*. arXiv:1910.12038.
- **Link:** https://arxiv.org/abs/1910.12038
- **What it does:** Learns suicide-oriented word embeddings on Weibo; applies two-level attention (word + post) over user blog streams.
- **Relevance:** Word-level attention could identify which crisis keywords drive classification. Relevant for interpretability discussion.
- **Use in paper:** Briefly cite in Section 5 as cross-platform precedent.
- **Tag:** `comparison`

### 3.6 Amir, Coppersmith, Carvalho, Silva & Wallace (2017) — User2Vec

- **Citation:** Amir, S., Coppersmith, G., Carvalho, P., Silva, M. J., & Wallace, B. C. (2017). Quantifying mental health from social media with neural user embeddings. *MLHC 2017*, PMLR 68. arXiv:1705.00335.
- **Link:** https://arxiv.org/abs/1705.00335
- **What it does:** Learns per-user embeddings (User2Vec) from Twitter post histories. Embeddings capture PTSD/depression homophily and improve downstream classifiers.
- **Key finding:** User-level embeddings outperform post-aggregated features for some mental-health tasks.
- **Relevance:** This project's per-user feature aggregation (mean-pool MentalBERT embeddings per window) is a simpler version of User2Vec. Cite as justification for user-level representation.
- **Limitations:** Requires many posts per user to learn stable embedding; static (no temporal decomposition).
- **Use in paper:** Section 4.5 (MentalBERT centroid justification).
- **Tag:** `citation`

### 3.7 Benton, Mitchell & Hovy (2017) — Multi-task learning

- **Citation:** Benton, A., Mitchell, M., & Hovy, D. (2017). Multitask learning for mental health conditions with limited social media data. *EACL 2017*, 152–162. arXiv:1712.03538.
- **Link:** https://aclanthology.org/E17-1015/
- **What it does:** Hard-parameter-sharing MTL for suicide attempt + multiple mental-health conditions + gender as auxiliary tasks.
- **Key finding:** MTL gives largest gains on low-data conditions.
- **Relevance:** This project's three classes could be framed as MTL (crisis vs recovery vs neither, plus auxiliary like condition type). Worth mentioning as future work.
- **Use in paper:** Discussion (future work: MTL for the small recovery class).
- **Tag:** `citation`

### 3.8 Orabi, Buddhitha, Orabi & Inkpen (2018) — CNN/BiLSTM depression

- **Citation:** Orabi, A. H., Buddhitha, P., Orabi, M. H., & Inkpen, D. (2018). Deep learning for depression detection of Twitter users. *CLPsych @ NAACL 2018*, 88–97. ACL W18-0609.
- **Link:** https://aclanthology.org/W18-0609/
- **What it does:** Benchmarks CNN and BiLSTM with pretrained/learned embeddings on CLPsych Twitter data.
- **Relevance:** Precedent for BiLSTM baselines in mental-health NLP. Useful reference for this project's negative BiLSTM result.
- **Use in paper:** Section 5.4 (BiLSTM related work).
- **Tag:** `comparison`

### 3.9 Reimers & Gurevych (2019) — Sentence-BERT

- **Citation:** Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP-IJCNLP 2019*. arXiv:1908.10084.
- **Link:** https://arxiv.org/abs/1908.10084
- **What it does:** Mean-pooled Siamese-tuned BERT embeddings for semantic similarity. Outperforms CLS tokens.
- **Relevance:** This project mean-pools MentalBERT embeddings per window — SBERT is the methodological precedent for mean-pooling over Siamese encoders.
- **Use in paper:** Section 4.5 (pooling justification).
- **Tag:** `citation`

### 3.10 Trotzek, Koitka & Friedrich (2020) — eRisk-2018 top system

- **Citation:** Trotzek, M., Koitka, S., & Friedrich, C. M. (2020). Utilizing neural networks and linguistic metadata for early detection of depression indications in text sequences. *IEEE TKDE 32(3)*, 588–601. DOI 10.1109/TKDE.2018.2885515.
- **Link:** https://ieeexplore.ieee.org/document/8581374
- **What it does:** Top eRisk-2018 system: CNN over fastText + LIWC/metadata, ensembled.
- **Key finding:** LIWC + linguistic metadata remains competitive alongside deep learning for small-data mental-health detection.
- **Relevance:** Justifies this project's hand-engineered feature approach (Section 4.6 bonus features).
- **Use in paper:** Section 4.6 and Section 5.
- **Tag:** `comparison`

---

## 4. Change-point detection and diachronic methods

### 4.1 Killick, Fearnhead & Eckley (2012) — PELT

- **Citation:** Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. *Journal of the American Statistical Association 107(500)*, 1590–1598. DOI 10.1080/01621459.2012.737745. arXiv:1101.1438.
- **Link:** https://arxiv.org/abs/1101.1438
- **What it does:** **The PELT algorithm.** Pruning-based exact dynamic programming for penalised change-point detection with linear expected cost.
- **Method:** Penalised likelihood objective with pruning step that discards candidate change-points that cannot be part of an optimal solution. O(n) expected time vs O(n³) for naive exact methods.
- **Key finding:** Exact, optimal, and fast — dominates approximate methods when the cost function is additive.
- **Relevance:** **The core algorithm this project's Section 7 unsupervised baseline uses.** Cite every time PELT is mentioned.
- **Limitations:** Requires a penalty choice (BIC commonly used); offline only; assumes independent segments.
- **Use in paper:** Section 7 (primary citation).
- **Tag:** `citation`

### 4.2 Truong, Oudre & Vayatis (2020) — ruptures library and CPD survey

- **Citation:** Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of offline change point detection methods. *Signal Processing 167*, 107299. DOI 10.1016/j.sigpro.2019.107299. arXiv:1801.00718. Accompanying library: `ruptures` (arXiv:1801.00826).
- **Link:** https://arxiv.org/abs/1801.00826
- **What it does:** Unifies offline CPD algorithms by (a) cost function, (b) search strategy, (c) constraint. Provides the `ruptures` Python package used in this project's pipeline.
- **Relevance:** Methodological spine for Section 7. Cite for software reproducibility.
- **Use in paper:** Section 7 and Section 9 (pipeline, tech stack).
- **Tag:** `citation`

### 4.3 Adams & MacKay (2007) — BOCPD

- **Citation:** Adams, R. P., & MacKay, D. J. C. (2007). Bayesian online changepoint detection. arXiv:0710.3742.
- **Link:** https://arxiv.org/abs/0710.3742
- **What it does:** Exact online inference over run-length posteriors via message passing with conjugate exponential-family updates.
- **Relevance:** BOCPD is the natural *online* alternative to PELT's offline batch detection. If this project is extended to streaming, BOCPD is the next step.
- **Use in paper:** Section 7 discussion (online alternatives).
- **Tag:** `comparison`

### 4.4 Hamilton, Leskovec & Jurafsky (2016) — Diachronic embeddings

- **Citation:** Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). Diachronic word embeddings reveal statistical laws of semantic change. *ACL 2016*, 1489–1501. arXiv:1605.09096.
- **Link:** https://arxiv.org/abs/1605.09096
- **What it does:** Procrustes-aligned SGNS/PPMI/SVD embeddings across time buckets; formalises "law of conformity" and "law of innovation".
- **Relevance:** This project's MentalBERT centroid-drift features (cosine, L2 from baseline) are a user-scale version of diachronic embedding drift.
- **Use in paper:** Section 4.5 (semantic-shift feature justification).
- **Tag:** `citation`

### 4.5 Kutuzov, Øvrelid, Szymanski & Velldal (2018) — Diachronic survey

- **Citation:** Kutuzov, A., Øvrelid, L., Szymanski, T., & Velldal, E. (2018). Diachronic word embeddings and semantic shifts: A survey. *COLING 2018*, 1384–1397. arXiv:1806.03537.
- **Link:** https://aclanthology.org/C18-1117/
- **What it does:** Organises embedding-based semantic-shift detection by method and evaluation, including cultural-shift applications.
- **Relevance:** Context for treating per-user MentalBERT centroid drift as a semantic-shift measurement.
- **Use in paper:** Section 4.5.
- **Tag:** `citation`

---

## 5. Clinical circadian and sleep evidence

### 5.1 Ballard et al. (2016) — Nocturnal wakefulness → next-day suicidal ideation

- **Citation:** Ballard, E. D., Vande Voort, J. L., Bernert, R. A., Luckenbaugh, D. A., Richards, E. M., Niciu, M. J., Furey, M. L., Duncan, W. C. Jr., & Zarate, C. A. Jr. (2016). Nocturnal wakefulness is associated with next-day suicidal ideation in major depressive disorder and bipolar disorder. *Journal of Clinical Psychiatry 77(6)*, 825–831. DOI 10.4088/JCP.15m09943. PMC5103284.
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC5103284/
- **What it does:** **Core clinical finding for this project's circadian features.** Polysomnography in 65 MDD/BD patients correlated with next-day Hamilton suicide-item scores.
- **Method:** PSG measurement of wakefulness per hour; next-day suicidal-ideation rating.
- **Key finding:** **Wakefulness in the 4:00–4:59 am hour predicted next-day suicidal ideation (standardized β = 0.31, p = .008)**, independent of depression severity.
- **Relevance:** **Direct clinical justification for the `late_night_rate` (00:00–04:00 UTC) and `hour_entropy` features.** The empirical finding that 4 am wakefulness predicts suicidal ideation is why this project's temporal features outperform all linguistic ones.
- **Limitations:** Small clinical sample; PSG not social-media proxy; does not address *why* 4 am specifically.
- **Use in paper:** **Central citation in Section 4.4.** Cite every time hour_entropy_baseline is discussed as the top feature.
- **Tag:** `citation`

### 5.2 Perlis et al. (2016) — Nocturnal wakefulness as suicide risk

- **Citation:** Perlis, M. L., Grandner, M. A., Chakravorty, S., Bernert, R. A., Brown, G. K., & Thase, M. E. (2016). Nocturnal wakefulness: A previously unrecognized risk factor for suicide. *Journal of Clinical Psychiatry 77(6)*, e726–e733. DOI 10.4088/JCP.15m10131. PMC6314836.
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6314836/
- **What it does:** Combines NVDRS (2003–2010) fatal-injury-time data with American Time Use Survey wakefulness estimates to compute hour-by-hour suicide incidence rate ratios.
- **Key finding:** **Suicide between 00:00–05:59 is 3.6× higher than expected by chance**, peaking at 02:00.
- **Relevance:** Epidemiological-scale confirmation of Ballard et al. Directly supports late_night_rate (00:00–04:00 UTC) as a feature.
- **Use in paper:** Section 4.4 alongside Ballard 2016.
- **Tag:** `citation`

### 5.3 Tubbs et al. (2020) — Robustness of nocturnal suicide finding

- **Citation:** Tubbs, A. S., Perlis, M. L., Basner, M., Chakravorty, S., Khader, W., Fernandez, F., & Grandner, M. A. (2020). Relationship of nocturnal wakefulness to suicide risk across months and methods of suicide. *Journal of Clinical Psychiatry 81(2)*, 19m12964. DOI 10.4088/JCP.19m12964.
- **Link:** https://www.psychiatrist.com/jcp/nocturnal-wakefulness-and-season-and-method-of-suicide/
- **What it does:** Extends Perlis 2016 across 35,338 suicides. Demonstrates that nocturnal IRRs are consistently elevated across demographics, months, and methods (mean IRR ≈ 3.09).
- **Key finding:** The "Mind after Midnight" effect is robust — not an artefact of season, method, or demographic subgroup.
- **Relevance:** Robustness citation. Use to pre-empt the "your circadian features are a seasonal artefact" reviewer objection.
- **Use in paper:** Section 4.4 robustness argument.
- **Tag:** `citation`

### 5.4 Bernert et al. (2017) — Sleep architecture biomarker

- **Citation:** Bernert, R. A., Luckenbaugh, D. A., Duncan, W. C. Jr., Iwata, N. G., Ballard, E. D., & Zarate, C. A. Jr. (2017). Sleep architecture parameters as a putative biomarker of suicidal ideation in treatment-resistant depression. *Journal of Affective Disorders 208*, 309–315. PMID 27810712.
- **Link:** https://pubmed.ncbi.nlm.nih.gov/27810712/
- **What it does:** PSG study of treatment-resistant depression; suicidal ideation associated with less NREM Stage 4 sleep and higher nocturnal wakefulness.
- **Relevance:** Independent confirmation that sleep-architecture deficits predict suicide risk. Supports the broader argument that circadian/sleep features are clinically informative beyond depression severity.
- **Use in paper:** Section 4.4 (secondary clinical citation).
- **Tag:** `citation`

### 5.5 Golder & Macy (2011) — Diurnal and seasonal mood on Twitter

- **Citation:** Golder, S. A., & Macy, M. W. (2011). Diurnal and seasonal mood vary with work, sleep, and daylength across diverse cultures. *Science 333(6051)*, 1878–1881. DOI 10.1126/science.1202775.
- **Link:** https://www.science.org/doi/10.1126/science.1202775
- **What it does:** **Landmark large-scale circadian social-media study.** 509M tweets from 2.4M users across 84 countries.
- **Key findings:** (a) Positive affect peaks in early morning and on weekends, (b) morning peak is delayed ~2 h on weekends (sleep effect), (c) seasonal affect tracks daylength change.
- **Relevance:** **Foundational citation for using posting-time distributions as a psychological signal.** Justifies the `hour_entropy`, `weekend_rate`, and `late_night_rate` features.
- **Use in paper:** **Primary citation in Section 4.4** alongside Ballard 2016.
- **Tag:** `both`

---

## 6. Domain-adaptive language models and generalisation

### 6.1 Ji, Zhang, Ansari, Fu, Tiwari & Cambria (2022) — MentalBERT

- **Citation:** Ji, S., Zhang, T., Ansari, L., Fu, J., Tiwari, P., & Cambria, E. (2022). MentalBERT: Publicly available pretrained language models for mental healthcare. *LREC 2022*, 7184–7190. arXiv:2110.15621.
- **Link:** https://arxiv.org/abs/2110.15621
- **What it does:** **The MentalBERT paper.** Continues BERT-base-uncased and RoBERTa pretraining on ~13.6M Reddit mental-health sentences (r/depression, SuicideWatch, Anxiety, offmychest, bipolar, mentalillness). Releases MentalBERT and MentalRoBERTa.
- **Method / data:** Masked-language-modelling continued pretraining; 8 downstream benchmarks (Dreaddit, eRisk, CLPsych, UMD, T-SID, SWMH, SAD, Depression_Reddit).
- **Metrics:** Consistent 1–5 F1 improvement over base BERT/RoBERTa across all benchmarks.
- **Key finding:** Domain-adaptive pretraining on Reddit mental-health text is substantially beneficial.
- **Relevance:** **The exact model used for this project's Section 4.5 semantic-shift features and Section 2.5 label verification.** Cite as the source of `mental/mental-bert-base-uncased`.
- **Limitations:** Trained on Reddit — may not generalise perfectly to Tumblr; no explicit fine-tuning for longitudinal tasks.
- **Use in paper:** **Primary citation in Section 2.5 and Section 4.5.**
- **Tag:** `both`

### 6.2 Vajre, Naylor, Kamath & Shehu (2021) — PsychBERT

- **Citation:** Vajre, V., Naylor, M., Kamath, U., & Shehu, A. (2021). PsychBERT: A mental health language model for social media mental health behavioral analysis. *IEEE BIBM 2021*. DOI 10.1109/BIBM52615.2021.9669469.
- **Link:** https://ieeexplore.ieee.org/document/9669469
- **What it does:** BERT adapted on ~40K PubMed psychology/psychiatry articles + ~200K social-media mental-health conversations, with a two-stage discriminate-then-classify framework.
- **Relevance:** Alternative domain LM. Mention as viable alternative embedder — future work could compare PsychBERT vs MentalBERT for semantic-shift features.
- **Use in paper:** Discussion (model alternatives).
- **Tag:** `comparison`

### 6.3 Alsentzer et al. (2019) — ClinicalBERT

- **Citation:** Alsentzer, E., Murphy, J., Boag, W., Weng, W.-H., Jindi, D., Naumann, T., & McDermott, M. (2019). Publicly available clinical BERT embeddings. *Clinical NLP Workshop @ NAACL 2019*, 72–78. arXiv:1904.03323.
- **Link:** https://arxiv.org/abs/1904.03323
- **What it does:** ClinicalBERT / Discharge-Summary-BERT trained on MIMIC-III clinical notes. Canonical example of domain-adaptive BERT pretraining.
- **Relevance:** Methodological precedent for MentalBERT. Cite when justifying domain-adaptive pretraining.
- **Use in paper:** Section 4.5 background.
- **Tag:** `citation`

### 6.4 Harrigian, Aguirre & Dredze (2020) — Generalisation failure

- **Citation:** Harrigian, K., Aguirre, C., & Dredze, M. (2020). Do models of mental health based on social media data generalize? *Findings of EMNLP 2020*, 3774–3788. ACL 2020.findings-emnlp.337.
- **Link:** https://aclanthology.org/2020.findings-emnlp.337/
- **What it does:** Trains depression classifiers on 6 proxy-labelled corpora (CLPsych, RSDD, SMHD, Topic-Restricted, Multi-Task, Shen Twitter) and evaluates cross-platform/cross-proxy transfer.
- **Key finding:** **Large F1 drops under cross-corpus transfer** (often 30–50 points); much of in-domain performance attributable to confounds (posting style, subreddit moderation effects, topic).
- **Relevance:** **Central limitation-framing citation.** Any reviewer will ask whether your model generalises. This project's multi-platform (Reddit + Tumblr) design is a *partial* answer; Harrigian 2020 explains why this is hard.
- **Use in paper:** **Section 11 limitations (critical)** and Section 5 ablation discussion.
- **Tag:** `citation`

### 6.5 Harrigian, Aguirre & Dredze (2021) — State of social-media mental health data

- **Citation:** Harrigian, K., Aguirre, C., & Dredze, M. (2021). On the state of social media data for mental health research. *CLPsych @ NAACL 2021*, 15–24. arXiv:2011.05233.
- **Link:** https://arxiv.org/abs/2011.05233
- **What it does:** Introduces a standardised directory of mental-health datasets and quantifies bottlenecks in the field (access, size, proxy validity).
- **Relevance:** Meta-framing. Cite when discussing scope of the field and data-access problems.
- **Use in paper:** Section 1 introduction and Section 11.
- **Tag:** `citation`

### 6.6 Aguirre, Harrigian & Dredze (2021) — Fairness

- **Citation:** Aguirre, C. A., Harrigian, K., & Dredze, M. (2021). Gender and racial fairness in depression research using social media. *EACL 2021*, 2932–2949. ACL 2021.eacl-main.256.
- **Link:** https://aclanthology.org/2021.eacl-main.256/
- **What it does:** Shows depression classifiers perform unequally across gender and racial groups. Proposes mitigations.
- **Relevance:** Must-cite fairness paper. This project does not measure demographics (no IRB), so cite as explicit limitation.
- **Use in paper:** Section 11 ethics.
- **Tag:** `citation`

---

## 7. Gradient-boosted trees vs. deep tabular models

### 7.1 Chen & Guestrin (2016) — XGBoost

- **Citation:** Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*, 785–794. arXiv:1603.02754.
- **Link:** https://arxiv.org/abs/1603.02754
- **What it does:** **The XGBoost paper.** Sparsity-aware split-finding, weighted quantile sketch, cache-aware column-block design, shrinkage and column subsampling regularisation.
- **Relevance:** Cite as the source of XGBoost, used in Section 5.2.
- **Use in paper:** Section 5.2 model list.
- **Tag:** `citation`

### 7.2 Ke et al. (2017) — LightGBM

- **Citation:** Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS 30*, 3149–3157.
- **Link:** https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree
- **What it does:** **The LightGBM paper.** Gradient-based One-Side Sampling (GOSS), Exclusive Feature Bundling (EFB), leaf-wise tree growth. Up to ~20× faster than prior GBDT at comparable accuracy.
- **Relevance:** Source citation for LightGBM in Section 5.2.
- **Use in paper:** Section 5.2.
- **Tag:** `citation`

### 7.3 Prokhorenkova, Gusev, Vorobev, Dorogush & Gulin (2018) — CatBoost

- **Citation:** Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. *NeurIPS 31*, 6639–6649. arXiv:1706.09516.
- **Link:** https://arxiv.org/abs/1706.09516
- **What it does:** **The CatBoost paper.** Ordered boosting + ordered target statistics address prediction shift and target leakage. Best-in-class on mixed-type tabular data.
- **Relevance:** Source citation for CatBoost in Section 5.2.
- **Use in paper:** Section 5.2.
- **Tag:** `citation`

### 7.4 Grinsztajn, Oyallon & Varoquaux (2022) — Trees beat DL on tabular

- **Citation:** Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS 2022 Datasets & Benchmarks*. arXiv:2207.08815.
- **Link:** https://arxiv.org/abs/2207.08815
- **What it does:** **The definitive 2022 benchmark.** 45 tabular datasets, systematic comparison of trees (XGB, RF) vs neural (TabNet, SAINT, FT-Transformer, etc.).
- **Key findings:** (a) Trees dominate at ~10K samples; (b) neural nets are sensitive to uninformative features, (c) neural nets have implicit rotation-invariance and smoothness bias that hurt on tabular data.
- **Relevance:** **This is the citation that justifies why this project's tabular features beat the BiLSTM at n=505.** Essential for Section 5.4 negative-result framing.
- **Use in paper:** **Section 5.4** (central citation for the BiLSTM negative result).
- **Tag:** `citation`

### 7.5 Shwartz-Ziv & Armon (2022) — Deep learning is not all you need on tabular

- **Citation:** Shwartz-Ziv, R., & Armon, A. (2022). Tabular data: Deep learning is not all you need. *Information Fusion 81*, 84–90. arXiv:2106.03253.
- **Link:** https://arxiv.org/abs/2106.03253
- **What it does:** XGBoost beats TabNet, NODE, DNF-Net, 1D-CNN on 11 datasets with less tuning. Ensembling gives small additional gains.
- **Relevance:** Secondary citation for tabular vs DL. Also justifies *stacking* (ensembles give small gains).
- **Use in paper:** Section 5.2 and 5.4.
- **Tag:** `citation`

### 7.6 Wolpert (1992) — Stacked generalization

- **Citation:** Wolpert, D. H. (1992). Stacked generalization. *Neural Networks 5(2)*, 241–259. DOI 10.1016/S0893-6080(05)80023-1.
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231
- **What it does:** Foundational paper introducing stacked generalization — meta-learner trained on out-of-fold predictions of level-0 models.
- **Relevance:** Cite as source of stacking methodology (Section 5.2 stacking ensemble).
- **Use in paper:** Section 5.2.
- **Tag:** `citation`

---

## 8. Ethics, distant supervision, reproducibility

### 8.1 Benton, Coppersmith & Dredze (2017) — Ethics protocols

- **Citation:** Benton, A., Coppersmith, G., & Dredze, M. (2017). Ethical research protocols for social media health research. *EthNLP @ EACL 2017*, 94–102. ACL W17-1612.
- **Link:** https://aclanthology.org/W17-1612/
- **What it does:** **Practical ethics checklist** — IRB approach, consent, data sharing, de-identification, risk disclosure, preregistration.
- **Relevance:** **The go-to ethics citation** for any Reddit mental-health paper. This project's Section 11 aligns with these protocols.
- **Use in paper:** **Section 11 (primary ethics citation).**
- **Tag:** `citation`

### 8.2 Chancellor, Birnbaum, Caine, Silenzio & De Choudhury (2019) — Ethical taxonomy

- **Citation:** Chancellor, S., Birnbaum, M. L., Caine, E. D., Silenzio, V. M. B., & De Choudhury, M. (2019). A taxonomy of ethical tensions in inferring mental health states from social media. *FAT\* 2019*, 79–88. DOI 10.1145/3287560.3287587.
- **Link:** https://dl.acm.org/doi/10.1145/3287560.3287587
- **What it does:** Maps conflicts across IRB exemption, construct validity, stakeholder harms, and ML opacity in predicting mental health from social media.
- **Relevance:** Must-cite for ethics framing. This project's explicit "population-level correlates, not individual risk scores" disclaimer aligns with this taxonomy.
- **Use in paper:** Section 11.
- **Tag:** `citation`

### 8.3 Chancellor & De Choudhury (2020) — Critical methods review

- **Citation:** Chancellor, S., & De Choudhury, M. (2020). Methods in predictive techniques for mental health status on social media: A critical review. *npj Digital Medicine 3*, 43. DOI 10.1038/s41746-020-0233-7.
- **Link:** https://www.nature.com/articles/s41746-020-0233-7
- **What it does:** Systematic review of 75 studies (2013–2018) documenting pervasive gaps in construct validity, annotation standards, and reproducibility.
- **Relevance:** **Essential methodological self-check.** Cite when discussing labelling limitations (Section 2) and reproducibility (Section 9).
- **Use in paper:** Section 2.2 labelling rationale and Section 11.
- **Tag:** `citation`

### 8.4 Ernala et al. (2019) — Methodological gaps

- **Citation:** Ernala, S. K., Birnbaum, M. L., Candan, K. A., Rizvi, A. F., Sterling, W. A., Kane, J. M., & De Choudhury, M. (2019). Methodological gaps in predicting mental health states from social media: Triangulating diagnostic signals. *CHI 2019*. DOI 10.1145/3290605.3300364.
- **Link:** https://dl.acm.org/doi/10.1145/3290605.3300364
- **What it does:** Empirically tests proxy diagnostic signals (subreddit membership, self-reports) against clinician-verified schizophrenia diagnoses.
- **Key finding:** Proxy signals show **strong internal but poor external validity** — they predict themselves well but don't transfer to clinical diagnoses.
- **Relevance:** **Critical limitation citation.** Directly frames the "your labels aren't clinical" objection. This project's keyword labels have exactly this problem.
- **Use in paper:** **Section 11 (critical).**
- **Tag:** `citation`

### 8.5 Ajmani, Chancellor, Mehta, Fiesler, Zimmer & De Choudhury (2023) — Ethics disclosure review

- **Citation:** Ajmani, L., Chancellor, S., Mehta, B., Fiesler, C., Zimmer, M., & De Choudhury, M. (2023). A systematic review of ethics disclosures in predictive mental health research. *FAccT 2023*. DOI 10.1145/3593013.3594082.
- **Link:** https://dl.acm.org/doi/10.1145/3593013.3594082
- **What it does:** Follow-up to Chancellor 2019, finds ethics reporting remains inconsistent five years later.
- **Relevance:** Recent ethics-reporting benchmark to hold this project to.
- **Use in paper:** Section 11.
- **Tag:** `citation`

---

## 9. COVID-19 and subreddit-level confounds

### 9.1 Saha, Torous, Caine & De Choudhury (2020) — COVID psychosocial effects

- **Citation:** Saha, K., Torous, J., Caine, E. D., & De Choudhury, M. (2020). Psychosocial effects of the COVID-19 pandemic: Large-scale quasi-experimental study on social media. *JMIR 22(11)*, e22600. DOI 10.2196/22600.
- **Link:** https://www.jmir.org/2020/11/e22600/
- **What it does:** Compares 60M Twitter posts (March–May 2020) with 40M 2019 controls.
- **Key finding:** **Mental-health symptomatic expressions rose ~14%** and support expressions ~5% during early COVID; topical anchoring to COVID contexts.
- **Relevance:** Documents the scale of the COVID confound on social media. This project's data spans July 2019 – December 2021 — right through COVID.
- **Use in paper:** **Section 11 (critical COVID confound citation).**
- **Tag:** `citation`

### 9.2 Low, Rumker, Talkar, Torous, Cecchi & Ghosh (2020) — Reddit COVID mental-health dataset

- **Citation:** Low, D. M., Rumker, L., Talkar, T., Torous, J., Cecchi, G., & Ghosh, S. S. (2020). Natural language processing reveals vulnerable mental health support groups and heightened health anxiety on Reddit during COVID-19: Observational study. *JMIR 22(10)*, e22635. DOI 10.2196/22635.
- **Link:** https://www.jmir.org/2020/10/e22635/
- **What it does:** Releases the **Reddit Mental Health Dataset (826,961 users, 2018–2020)** — exactly the kind of dataset this project's solomonk/reddit_mental_health_posts resembles. Analyses 15 mental-health subreddits + 11 controls.
- **Key findings:** (a) r/HealthAnxiety spiked ~2 months before other groups, (b) ADHD, eating-disorders, and anxiety showed the most negative semantic change, (c) suicidality and loneliness clusters doubled in post count.
- **Relevance:** **The foundational COVID-era Reddit mental-health reference.** Must be cited as both a data provenance note and a COVID-confound framing.
- **Use in paper:** **Section 2.1 (data lineage) and Section 11 (COVID confound).**
- **Tag:** `both`

### 9.3 Biester, Matton, Rajendran, Provost & Mihalcea (2021) — Reddit COVID forum shifts

- **Citation:** Biester, L., Matton, K., Rajendran, J., Provost, E. M., & Mihalcea, R. (2021). Understanding the impact of COVID-19 on online mental health forums. *ACM TMIS 12(4)*, 1–28.
- **Link:** https://dl.acm.org/doi/10.1145/3458770
- **What it does:** Companion analysis showing community-level linguistic shifts (topic, negative emotion, isolation vocabulary) across Reddit mental-health forums during the pandemic.
- **Relevance:** Secondary COVID-confound citation.
- **Use in paper:** Section 11.
- **Tag:** `citation`

### 9.4 Saha, Kotakonda & De Choudhury (2025) — College students longitudinal COVID

- **Citation:** Saha, K., Kotakonda, S., & De Choudhury, M. (2025). Mental health impact of the COVID-19 pandemic on college students: A quasi-experimental study on social media. *ICWSM 2025*. DOI 10.1609/icwsm.v19i1.35899.
- **Link:** https://ojs.aaai.org/index.php/ICWSM/article/view/35899
- **What it does:** Longitudinal causal analysis (pre-pandemic, lockdown, vaccination periods) on college-subreddit discourse.
- **Relevance:** Demonstrates quasi-experimental design to handle the COVID confound — a future-work direction for this project.
- **Use in paper:** Section 11 or Discussion (future work on causal framing).
- **Tag:** `citation`

---

## 10. Psycholinguistics foundations (LIWC, pronouns, TTR, VADER)

### 10.1 Pennebaker, Mehl & Niederhoffer (2003) — Psychological aspects of word use

- **Citation:** Pennebaker, J. W., Mehl, M. R., & Niederhoffer, K. G. (2003). Psychological aspects of natural language use: Our words, our selves. *Annual Review of Psychology 54*, 547–577. DOI 10.1146/annurev.psych.54.101601.145041.
- **Link:** https://www.annualreviews.org/doi/10.1146/annurev.psych.54.101601.145041
- **What it does:** **The Pennebaker keystone review.** Establishes that function words (pronouns, articles, conjunctions) reflect psychological state more reliably than content words.
- **Key finding:** Frequent first-person singular pronoun use correlates with depression, honesty, and status.
- **Relevance:** **Foundational citation for Section 4.1** (`fp_pronoun_rate`, `i_vs_we_ratio`, `i_vs_you_ratio`).
- **Use in paper:** Section 4.1 (feature justification).
- **Tag:** `citation`

### 10.2 Rude, Gortner & Pennebaker (2004) — Depression in essays

- **Citation:** Rude, S., Gortner, E.-M., & Pennebaker, J. (2004). Language use of depressed and depression-vulnerable college students. *Cognition and Emotion 18(8)*, 1121–1133. DOI 10.1080/02699930441000030.
- **Link:** https://www.tandfonline.com/doi/abs/10.1080/02699930441000030
- **What it does:** **Empirically demonstrates** that currently-depressed students use the first-person singular ("I") substantially more than never-depressed students in free writing.
- **Key finding:** "I" use tracks current depression more reliably than content words.
- **Relevance:** **Direct empirical foundation for fp_pronoun_rate.**
- **Use in paper:** **Section 4.1 primary citation for pronouns.**
- **Tag:** `citation`

### 10.3 Tausczik & Pennebaker (2010) — LIWC

- **Citation:** Tausczik, Y. R., & Pennebaker, J. W. (2010). The psychological meaning of words: LIWC and computerized text analysis methods. *Journal of Language and Social Psychology 29(1)*, 24–54. DOI 10.1177/0261927X09351676.
- **Link:** https://journals.sagepub.com/doi/10.1177/0261927X09351676
- **What it does:** Comprehensive review of LIWC methodology, categories, and validation.
- **Relevance:** **The LIWC reference citation.** This project uses a LIWC-inspired 120-word negative-affect list; cite to justify.
- **Use in paper:** Section 4.1 (`neg_affect_rate`).
- **Tag:** `citation`

### 10.4 Hutto & Gilbert (2014) — VADER

- **Citation:** Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *ICWSM 2014*. AAAI 14550.
- **Link:** https://ojs.aaai.org/index.php/ICWSM/article/view/14550
- **What it does:** **The VADER paper.** Rule-based sentiment lexicon + grammatical heuristics tuned on social-media text. Returns compound score in [-1, +1].
- **Key finding:** VADER outperforms individual human raters on tweet-level sentiment labelling (F1 = 0.96 vs 0.84).
- **Relevance:** **Primary citation for `sentiment_mean` feature** (Section 4.1) and the PELT weekly-sentiment time series (Section 7).
- **Use in paper:** **Section 4.1 and Section 7.**
- **Tag:** `citation`

### 10.5 Templin (1957) — Type-token ratio

- **Citation:** Templin, M. C. (1957). *Certain Language Skills in Children: Their Development and Interrelationships.* University of Minnesota Press.
- **What it does:** Classic reference establishing type-token ratio as a lexical-diversity measure.
- **Relevance:** Foundational citation for `ttr` feature.
- **Use in paper:** Section 4.1 (lexical diversity). If preferred, cite McCarthy & Jarvis (2010) on MTLD as modern alternative.
- **Tag:** `citation`

### 10.6 McCarthy & Jarvis (2010) — MTLD and TTR alternatives

- **Citation:** McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment. *Behavior Research Methods 42(2)*, 381–392. DOI 10.3758/BRM.42.2.381.
- **Link:** https://link.springer.com/article/10.3758/BRM.42.2.381
- **What it does:** Shows that raw TTR is length-sensitive and proposes MTLD as a length-invariant alternative.
- **Relevance:** Critical caveat for this project's TTR feature — TTR varies with post length. Should discuss MTLD as future-work alternative or bucket posts by length.
- **Use in paper:** Section 4.1 (TTR limitation) or Discussion.
- **Tag:** `citation`

### 10.7 Flesch (1948) — Reading ease

- **Citation:** Flesch, R. (1948). A new readability yardstick. *Journal of Applied Psychology 32(3)*, 221–233. DOI 10.1037/h0057532.
- **What it does:** Introduces the Flesch Reading Ease formula.
- **Relevance:** Section 4.6 `flesch_reading_ease` feature.
- **Use in paper:** Section 4.6.
- **Tag:** `citation`

### 10.8 Kincaid, Fishburne, Rogers & Chissom (1975) — Flesch-Kincaid Grade

- **Citation:** Kincaid, J. P., Fishburne, R. P. Jr., Rogers, R. L., & Chissom, B. S. (1975). Derivation of new readability formulas for Navy enlisted personnel. Research Branch Report 8-75.
- **Relevance:** Source of Flesch-Kincaid Grade Level.
- **Use in paper:** Section 4.6.
- **Tag:** `citation`

### 10.9 Shannon (1948) — A mathematical theory of communication

- **Citation:** Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal 27*, 379–423, 623–656.
- **Relevance:** Origin of Shannon entropy used in `hour_entropy` feature.
- **Use in paper:** Section 4.4.
- **Tag:** `citation`

---

## 11. Supporting references (condensed)

These are worth citing selectively but don't need deep notes.

- **Gui, Zhu, Xu, Peng & Huang (2019)** — Cooperative multimodal approach to depression detection in Twitter. *AAAI 33*. DOI 10.1609/aaai.v33i01.3301110. *Multimodal depression baseline.*
- **Gaur et al. (2019)** — Knowledge-aware assessment of severity of suicide risk. *WWW 2019*. *Knowledge-augmented Reddit suicide-risk.*
- **Pavalanathan & De Choudhury (2015)** — Identity management and mental-health discourse on Reddit. *ICWSM 2015*. *Reddit throwaway-account norm.*
- **Coppersmith, Ngo, Leary & Wood (2016)** — Exploratory analysis of social media prior to a suicide attempt. *CLPsych 2016*. *Pre-event signal baseline.*
- **Conway & O'Connor (2016)** — Social media, big data, and mental health: current advances and ethical implications. *Current Opinion in Psychology 9*. *Ethics survey.*
- **Seabrook, Kern & Rickard (2016)** — Social networking sites, depression, and anxiety: a systematic review. *JMIR Mental Health 3(4)*. *Scoping review.*
- **Jashinsky et al. (2014)** — Tracking suicide risk factors through Twitter in the US. *Crisis 35(1)*. *Epidemiological Twitter suicide mapping.*
- **Jamil et al. (2017)** — Monitoring tweets for depression to detect at-risk users. *CLPsych @ ACL 2017*. *Per-user Twitter depression monitoring.*
- **MacAvaney, Mittu, Coppersmith, Leintz & Resnik (2021)** — Community-level research on suicidality prediction in a secure environment. *CLPsych 2021 shared task*. *CLPsych 2021 reference.*
- **Tsakalidis, Liakata et al. (2021) — CLPsych 2021 NPS-Chat baseline paper.* *Mood-timeline preliminaries.*
- **Zhang, Lyu, Sun, Tao & Jin (2022)** — A survey of mental-health-oriented pretrained language models. *arXiv:2209.05552.* *Domain-LM survey.*

---

## 12. Cross-cutting themes and tensions

### Theme A: The identity-vs-change distinction
Coppersmith 2015, Yates 2017, and Cohan 2018 all build classifiers that learn *who* mental-health users are. De Choudhury 2016, Tsakalidis 2022, and Sawhney 2020 shift to asking *how* users change. This project sits squarely in the second camp and should frame itself against Coppersmith 2015 as "the static-identity paradigm that we explicitly depart from."

### Theme B: Temporal/circadian signal vs. linguistic content
Ballard 2016, Perlis 2016, Tubbs 2020, Golder & Macy 2011 provide strong clinical/empirical evidence that posting *time* — independent of content — predicts mental state. Most CLPsych systems (Matero 2019, Sawhney 2020) still centre text. This project's finding that `hour_entropy_baseline` tops feature importance directly validates the circadian literature — and is a novel contribution to CLPsych.

### Theme C: Tabular vs. deep learning at small n
Grinsztajn 2022 and Shwartz-Ziv 2022 establish that trees beat DL on tabular data at ~10K samples. This project has n=505 and sees BiLSTM underperform RF/XGB/LGB — exactly predicted. Use this as cover for the negative result rather than apologising for it.

### Theme D: Distant supervision's validity ceiling
Chancellor 2020, Ernala 2019, and Harrigian 2020 all argue that proxy/regex/tag labels have limited external validity. This project's low_confidence flag (for ambiguous "made it") and MentalBERT label verification are partial mitigations. Section 11 must explicitly acknowledge this.

### Theme E: COVID as a baseline confound
Saha 2020, Low 2020, and Biester 2021 document large COVID-era shifts in Reddit mental-health discourse. This project's data window (July 2019 – Dec 2021) includes the entire COVID onset and first two waves. Cannot disentangle; must acknowledge.

### Theme F: Reddit vs. Tumblr as platforms
Reddit has been studied at scale (Pushshift, RSDD, SMHD, Low 2020); Tumblr has barely been studied (almost no peer-reviewed Tumblr mental-health NLP exists beyond blog-level analyses). This project's multi-platform design is genuinely novel, and the MentalBERT-based label verification is a principled response to Tumblr's tag-noise.

---

## 13. Citation playbook for each section of the paper

| Paper section | Primary citations | Secondary citations |
|---|---|---|
| **1. Research Question** | De Choudhury 2016, Tsakalidis 2022, Coppersmith 2015 | Harrigian 2020, Chancellor 2020 |
| **2.1 Data sources** | Baumgartner 2020 (Pushshift), Low 2020 (RSDD-adjacent dataset), Yates 2017 (RSDD), Cohan 2018 (SMHD) | Turcan 2019 (Dreaddit) |
| **2.3 Reddit cohort / keyword labels** | Coppersmith 2014, Yates 2017 | Chancellor 2020 |
| **2.4 Tumblr distant supervision** | Benton 2017 (ethics), Chancellor 2020 | — |
| **2.5 MentalBERT label verification** | Ji 2022 (MentalBERT), Reimers 2019 (SBERT) | Alsentzer 2019 (ClinicalBERT precedent) |
| **3. Time Windows** | Tsakalidis 2022, De Choudhury 2016 | Sawhney 2020 (STATENet time-awareness) |
| **4.1 Linguistic features** | Pennebaker 2003, Rude 2004, Tausczik & Pennebaker 2010, Hutto 2014 (VADER), Templin 1957 | McCarthy & Jarvis 2010 (TTR caveat) |
| **4.2–4.3 Deltas & z-norm** | De Choudhury 2016 (within-user change precedent) | — |
| **4.4 Temporal/circadian** | Golder & Macy 2011, Ballard 2016, Perlis 2016, Tubbs 2020, Shannon 1948 | Bernert 2017 |
| **4.5 Semantic shift** | Ji 2022 (MentalBERT), Hamilton 2016, Kutuzov 2018, Reimers 2019 | Amir 2017 (User2Vec) |
| **4.6 Bonus hand-engineered features** | Flesch 1948, Kincaid 1975, Pennebaker 2003 | Trotzek 2020 |
| **5.1 Task framing** | Milne 2016, Zirikly 2019, Tsakalidis 2022 | — |
| **5.2 Classifiers** | Chen & Guestrin 2016, Ke 2017, Prokhorenkova 2018, Wolpert 1992 | Shwartz-Ziv 2022 |
| **5.3 Hyperparameter search** | — (pedagogical citation for GridSearchCV) | — |
| **5.4 Sequence model (BiLSTM+Attn)** | Tsakalidis 2022 (ACL), Sawhney 2020 (STATENet), Grinsztajn 2022, Shwartz-Ziv 2022 | Orabi 2018, Sawhney 2021a (SISMO) |
| **5.5 Evaluation** | Losada & Crestani 2016 (ERDE as future alternative) | — |
| **6. Ablation study** | Harrigian 2020 (each feature group tests generalisation) | — |
| **7. PELT baseline** | Killick 2012 (PELT), Truong 2020 (ruptures), Hutto 2014 (VADER time series) | Adams & MacKay 2007 (BOCPD alternative) |
| **8. Sensitivity analysis** | Chancellor 2020, Ernala 2019 | — |
| **9. Pipeline architecture** | Baumgartner 2020, Truong 2020 | — |
| **10. Results summary** | Sawhney 2020, Matero 2019 (as benchmarks) | — |
| **11. Ethical considerations** | Benton 2017, Chancellor 2019, Chancellor 2020, Ernala 2019, Ajmani 2023, Aguirre 2021, Saha 2020, Low 2020 | Conway & O'Connor 2016 |

---

## Appendix: BibTeX skeleton

Starter BibTeX keys for the must-cite set. Fill in missing fields before submission.

```bibtex
@inproceedings{coppersmith2015clpsych,
  title={{CLPsych} 2015 shared task: Depression and {PTSD} on {T}witter},
  author={Coppersmith, Glen and Dredze, Mark and Harman, Craig and Hollingshead, Kristy and Mitchell, Margaret},
  booktitle={Proc. 2nd Workshop on Computational Linguistics and Clinical Psychology},
  pages={31--39},
  year={2015},
  publisher={ACL}
}

@inproceedings{dechoudhury2016shifts,
  title={Discovering shifts to suicidal ideation from mental health content in social media},
  author={De Choudhury, Munmun and Kiciman, Emre and Dredze, Mark and Coppersmith, Glen and Kumar, Mrinal},
  booktitle={Proc. CHI},
  pages={2098--2110},
  year={2016}
}

@inproceedings{tsakalidis2022clpsych,
  title={Overview of the {CLP}sych 2022 shared task: Capturing moments of change in longitudinal user posts},
  author={Tsakalidis, Adam and Chim, Jenny and Bilal, Iman Munire and Zirikly, Ayah and others},
  booktitle={Proc. CLPsych @ NAACL},
  pages={184--198},
  year={2022}
}

@inproceedings{sawhney2020statenet,
  title={A time-aware transformer based model for suicide ideation detection on social media},
  author={Sawhney, Ramit and Joshi, Harshit and Gandhi, Saumya and Shah, Rajiv Ratn},
  booktitle={EMNLP},
  pages={7685--7697},
  year={2020}
}

@article{ji2022mentalbert,
  title={{MentalBERT}: Publicly available pretrained language models for mental healthcare},
  author={Ji, Shaoxiong and Zhang, Tianlin and Ansari, Luna and Fu, Jie and Tiwari, Prayag and Cambria, Erik},
  booktitle={LREC},
  pages={7184--7190},
  year={2022}
}

@article{ballard2016nocturnal,
  title={Nocturnal wakefulness is associated with next-day suicidal ideation in major depressive disorder and bipolar disorder},
  author={Ballard, Elizabeth D and Vande Voort, Jennifer L and Bernert, Rebecca A and others},
  journal={Journal of Clinical Psychiatry},
  volume={77},
  number={6},
  pages={825--831},
  year={2016}
}

@article{perlis2016nocturnal,
  title={Nocturnal wakefulness: A previously unrecognized risk factor for suicide},
  author={Perlis, Michael L and Grandner, Michael A and Chakravorty, Subhajit and others},
  journal={Journal of Clinical Psychiatry},
  volume={77},
  number={6},
  pages={e726--e733},
  year={2016}
}

@article{golder2011diurnal,
  title={Diurnal and seasonal mood vary with work, sleep, and daylength across diverse cultures},
  author={Golder, Scott A and Macy, Michael W},
  journal={Science},
  volume={333},
  number={6051},
  pages={1878--1881},
  year={2011}
}

@article{killick2012pelt,
  title={Optimal detection of changepoints with a linear computational cost},
  author={Killick, Rebecca and Fearnhead, Paul and Eckley, Idris A},
  journal={Journal of the American Statistical Association},
  volume={107},
  number={500},
  pages={1590--1598},
  year={2012}
}

@article{grinsztajn2022trees,
  title={Why do tree-based models still outperform deep learning on typical tabular data?},
  author={Grinsztajn, L{\'e}o and Oyallon, Edouard and Varoquaux, Ga{\"e}l},
  journal={NeurIPS Datasets and Benchmarks},
  year={2022}
}

@inproceedings{harrigian2020generalize,
  title={Do models of mental health based on social media data generalize?},
  author={Harrigian, Keith and Aguirre, Carlos and Dredze, Mark},
  booktitle={Findings of EMNLP},
  pages={3774--3788},
  year={2020}
}

@inproceedings{chancellor2019taxonomy,
  title={A taxonomy of ethical tensions in inferring mental health states from social media},
  author={Chancellor, Stevie and Birnbaum, Michael L and Caine, Eric D and Silenzio, Vincent MB and De Choudhury, Munmun},
  booktitle={Proc. FAT*},
  pages={79--88},
  year={2019}
}

@article{chancellor2020npj,
  title={Methods in predictive techniques for mental health status on social media: A critical review},
  author={Chancellor, Stevie and De Choudhury, Munmun},
  journal={npj Digital Medicine},
  volume={3},
  number={1},
  pages={43},
  year={2020}
}

@article{low2020reddit,
  title={Natural language processing reveals vulnerable mental health support groups and heightened health anxiety on {R}eddit during {COVID-19}: Observational study},
  author={Low, Daniel M and Rumker, Laurie and Talkar, Tanya and Torous, John and Cecchi, Guillermo and Ghosh, Satrajit S},
  journal={Journal of Medical Internet Research},
  volume={22},
  number={10},
  pages={e22635},
  year={2020}
}

@inproceedings{hutto2014vader,
  title={{VADER}: A parsimonious rule-based model for sentiment analysis of social media text},
  author={Hutto, Clayton J and Gilbert, Eric},
  booktitle={Proc. ICWSM},
  year={2014}
}

@article{rude2004language,
  title={Language use of depressed and depression-vulnerable college students},
  author={Rude, Stephanie and Gortner, Eva-Maria and Pennebaker, James},
  journal={Cognition and Emotion},
  volume={18},
  number={8},
  pages={1121--1133},
  year={2004}
}
```

---

*End of knowledge base. Maintain this as a living document — add new entries as you encounter them during drafting, and update the citation playbook (Section 13) as section numbers evolve.*
