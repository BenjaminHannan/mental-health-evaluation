# Knowledge Base: Longitudinal Linguistic Markers of Mental Health Deterioration in Online Communities

*Prepared for Benjamin Hannan, Hanover High School / mentored by Taka Khoo (Dartmouth ML MS) and Aditya Gaur (Dartmouth '29). Target: arXiv cs.CL by August 15, 2026; potential CLPsych 2027 submission.*

This knowledge base synthesizes the foundational history and 2023–2026 frontier of computational linguistics applied to mental health. It is organized to map directly onto the paper's sections (Introduction, Related Work, Methods, Ethics, Limitations) and reviewer expectations at ACL/EMNLP/NAACL/CLPsych.

---

## 1. Dataset and methodology context

### 1.1 The solomonk/reddit_mental_health_posts dataset

**What the dataset actually is.** The dataset lives at `huggingface.co/datasets/solomonk/reddit_mental_health_posts`. The HuggingFace dataset viewer reports **151k rows** (larger than the 96,073 number the author is working with, so the author is using a filtered subset), 5 subreddit classes (r/depression, r/OCD, r/PTSD, r/ADHD, r/aspergers), and columns `author, body, created_utc, id, num_comments, score, subreddit, title, upvote_ratio, url`. Posts date from roughly **mid-2019 through late-2021** (the viewer shows 2021-12-22 timestamps). There is **no published paper, no datasheet, and no formally documented collection methodology**; the uploader is an individual HuggingFace user ("solomonk"), not an institutional research group.

**Critical implications.** Because the dataset has no datasheet (per Gebru et al. 2021, *Commun. ACM*), the author should treat it as "scraped public Reddit data, likely via Pushshift" and cite **Gaffney & Matias 2018** (*PLOS ONE* 13(7):e0200162) on known missing-data problems in commonly used Reddit corpora (~0.04% of comments missing, systematically biased toward certain subreddits — a threat to longitudinal claims). The author should also explicitly note **DSM-5's 2013 reclassification of Asperger's into Autism Spectrum Disorder**, which means r/aspergers labels do not map to any current clinical diagnostic category and should be framed as "self-identified community membership" rather than a diagnostic group.

**The paper must include a datasheet-lite section**: source, time window, collection method (inferred), number of users, number of posts per user distribution, preprocessing steps, and a statement that the author did not personally scrape the data but used the existing HuggingFace release.

### 1.2 How the field uses Reddit mental-health subreddits

Reddit mental-health NLP runs on a small number of foundational datasets and papers. The **five must-cite Reddit-MH datasets/papers**:

1. **De Choudhury, Kiciman, Dredze, Coppersmith & Kumar (2016)** "Discovering shifts to suicidal ideation from mental health content in social media." *CHI 2016*, pp. 2098–2110. DOI: 10.1145/2858036.2858207. **This is the single most direct methodological precedent for this paper** — it uses r/depression and related mental-health subreddits to study transitions to r/SuicideWatch.
2. **Cohan, Desmet, Yates, Soldaini, MacAvaney & Goharian (2018)** "SMHD: A Large-Scale Resource for Exploring Online Language Usage for Multiple Mental Health Conditions." *COLING 2018*, pp. 1485–1497 (ACL C18-1126, arXiv:1806.05258). Self-reported-diagnosis corpus covering 9 conditions including depression, OCD, PTSD, ADHD, Asperger's/autism — maps almost exactly onto this paper's subreddit set.
3. **Shing, Nair, Zirikly, Friedenberg, Daumé III & Resnik (2018)** "Expert, Crowdsourced, and Machine Assessment of Suicide Risk via Online Postings." *CLPsych 2018*, pp. 25–36. DOI: 10.18653/v1/W18-0603. The UMD Reddit Suicidality Dataset — the gold standard for clinically-anchored Reddit labels, distributed via formal DUA with the American Association of Suicidology.
4. **Coppersmith, Dredze & Harman (2014)** "Quantifying Mental Health Signals in Twitter." *CLPsych 2014*, pp. 51–60. The progenitor paper for automatic mental-health signal extraction from social text; introduced the self-report regex labeling approach (e.g., "I was diagnosed with depression") that underlies SMHD and RSDD.
5. **Losada & Crestani (2016)** and the subsequent **eRisk shared task at CLEF (2017–present)**: test collections for early risk prediction of depression, anorexia, self-harm, and pathological gambling from Reddit. Cite the most recent **Parapar et al. (2023, 2024) eRisk overviews** for current state of the art.

### 1.3 Turning-point / moments-of-change labeling practice

The gold-standard rigorous approach is the **Moments of Change (MoC)** framework developed by Tsakalidis, Liakata, and colleagues:

- **Tsakalidis, Nanni, Hills, Chim, Song & Liakata (2022)** "Identifying Moments of Change from Longitudinal User Text." *ACL 2022*, pp. 4647–4660. DOI: 10.18653/v1/2022.acl-long.318 (arXiv:2205.05593). Defines two change types: **Switches** (sudden shifts in mood) and **Escalations** (gradual mood progressions). **500 manually annotated user timelines, 18.7K posts**, with detailed annotation guidelines and temporally-sensitive evaluation metrics.
- **Tsakalidis, Chim, Bilal, Zirikly et al. (2022)** "Overview of the CLPsych 2022 Shared Task: Capturing Moments of Change in Longitudinal User Posts." *CLPsych 2022*, pp. 184–198. DOI: 10.18653/v1/2022.clpsych-1.16. Reddit subset of the MoC corpus.
- **Hills, Tsakalidis, Nanni, Zachos & Liakata (2023)** "Creation and evaluation of timelines for longitudinal user posts." *EACL 2023*, pp. 3791–3804. DOI: 10.18653/v1/2023.eacl-main.274. How to *segment* user timelines to surface meaningful change windows — uses BOCPD (Adams & MacKay 2007) as one baseline.
- **Hills, Tseriotou, Miscouridou, Tsakalidis & Liakata (2024)** "Exciting Mood Changes: A Time-aware Hierarchical Transformer for Change Detection Modelling." *Findings of ACL 2024*. Current SoTA on the MoC task; the hierarchical transformer with time-aware context is the model to benchmark against if the paper aims at CLPsych 2027.
- **Chim, Tsakalidis, Gkoumas, Atzil-Slonim, Ophir, Zirikly, Resnik & Liakata (2024)** "Overview of the CLPsych 2024 Shared Task: Leveraging LLMs to Identify Evidence of Suicidality Risk in Online Posts." *CLPsych 2024*, pp. 177–190. Shift toward evidence-generation with LLMs.

**Rigorous labeling uses human annotation with published guidelines, not keyword heuristics.** Inter-annotator agreement (Krippendorff's α or Cohen's κ) is reported, typically κ ≈ 0.5–0.7 for these subjective mental-state labels.

### 1.4 Known issues with keyword-based crisis/recovery labeling

Keyword-based labels (e.g., tagging a user as "crisis" because a post contains "kill myself" or "want to die") suffer four documented failure modes:

1. **Hyperbolic everyday usage**: "I want to die, these exams are killing me." Benton, Mitchell & Hovy (2017) "Multi-Task Learning for Mental Health" discuss this.
2. **Sarcasm and negation**: VADER/lexicon methods misclassify ~15–25% of mental-health posts due to negation (Hutto & Gilbert 2014 caveats).
3. **Community-specific idioms**: r/OCD uses "intrusive thoughts," "checking," "reassurance-seeking" in technical meanings.
4. **Temporal ambiguity**: past-tense recovery narratives ("last year I wanted to die, now I'm in therapy") get flagged as current crisis.

**Framing advice.** The paper must disclose exactly how "crisis" and "recovery" labels were generated (keyword list, context window, validation). If labels are unvalidated heuristics, frame the entire contribution as a **feasibility study on heuristic labels**, not as detection of clinical states. This is the #1 thing reviewers will hammer.

**Open question the field has not answered**: *What is the minimum level of human validation required for a proxy label to be defensible?* Ernala et al. (2019) show that proxies without clinician triangulation fail out-of-distribution; no published lower bound exists.

---

## 2. Prior work in linguistic mental health detection

### 2.1 Foundational work (1980s–2010s)

The linguistic-markers-of-depression literature predates NLP by about two decades. Key citations, in historical order:

- **Gottschalk, L. A. & Gleser, G. C. (1969)** *The Measurement of Psychological States Through the Content Analysis of Verbal Behavior*. University of California Press. The Gottschalk-Gleser Content Analysis Scales quantify anxiety, hostility, and cognitive impairment from verbal samples — pre-computational ancestor of LIWC. Still cited in clinical linguistics.
- **Pennebaker, J. W. & Francis, M. E. (1996)** "Cognitive, emotional, and language processes in disclosure." *Cognition & Emotion*, 10(6), 601–626. First operationalization of LIWC categories.
- **Stirman, S. W. & Pennebaker, J. W. (2001)** "Word use in the poetry of suicidal and non-suicidal poets." *Psychosomatic Medicine*, 63(4), 517–522. DOI: 10.1097/00006842-200107000-00001. Famous finding: suicidal poets used more first-person singular and fewer social-references. Cited in essentially every linguistic-suicidality paper since.
- **Rude, S., Gortner, E.-M. & Pennebaker, J. W. (2004)** "Language use of depressed and depression-vulnerable college students." *Cognition & Emotion*, 18(8), 1121–1133. DOI: 10.1080/02699930441000030. **The single most important citation for any first-person-pronoun argument.** Currently-depressed and formerly-depressed college students used more first-person singular pronouns ("I," "me," "my"), more negative emotion words, and fewer positive emotion words than never-depressed students. Sample size was modest (N ≈ 124) but the finding has replicated repeatedly.
- **Pennebaker, J. W., Mehl, M. R. & Niederhoffer, K. G. (2003)** "Psychological Aspects of Natural Language Use: Our Words, Our Selves." *Annual Review of Psychology*, 54, 547–577. DOI: 10.1146/annurev.psych.54.101601.145041. The canonical narrative review linking function-word usage to personality and psychopathology.
- **Chung, C. K. & Pennebaker, J. W. (2007)** "The psychological functions of function words." In *Social Communication*. Frontiers of Social Psychology, pp. 343–359. Extended theoretical framing — pronouns as "attentional focus" markers.
- **Tausczik, Y. R. & Pennebaker, J. W. (2010)** "The psychological meaning of words: LIWC and computerized text analysis methods." *Journal of Language and Social Psychology*, 29(1), 24–54. DOI: 10.1177/0261927X09351676. **The canonical LIWC review.** Must be cited for any LIWC-based feature.

### 2.2 Why first-person pronoun rate is linked to depression

The theoretical mechanism is **self-focused attention / rumination** (Pyszczynski & Greenberg 1987; Nolen-Hoeksema 1991). Depression narrows attention inward; repeated self-referencing in language reflects and reinforces this narrowing. The empirical literature has matured into two large-sample meta-analyses that are essential reading:

- **Edwards, T. S. & Holtzman, N. S. (2017)** "A meta-analysis of correlations between depression and first-person singular pronoun use." *Journal of Research in Personality*, 68, 63–68. DOI: 10.1016/j.jrp.2017.02.005. Pooled effect size r ≈ 0.13 across multiple studies. Small but reliable.
- **Tackman, A. M., Sbarra, D. A., Carey, A. L., Donnellan, M. B., Horn, A. B., Holtzman, N. S., Edwards, T. S., Pennebaker, J. W. & Mehl, M. R. (2019)** "Depression, negative emotionality, and self-referential language: A multi-lab, multi-measure, and multi-language-task research synthesis." *Journal of Personality and Social Psychology*, 116(5), 817–834. DOI: 10.1037/pspp0000187. **N = 4,754 participants, 6 labs, 2 countries, preregistered.** Pooled r ≈ 0.10 between depression and I-talk. Crucially: **the depression–I-talk effect largely disappears when controlling for negative emotionality**, but negative emotionality–I-talk survives controlling for depression. Interpretation: I-talk is a marker of *general distress / negative emotionality*, NOT specifically depression.

**The paper's framing.** This is a critical nuance. The author should **not** write "first-person pronouns are a marker of depression"; they should write "**elevated I-talk has been robustly linked to general negative emotionality** (Tackman et al., 2019, *JPSP*), with a small direct correlation to depression specifically (r ≈ 0.10–0.13). Our task therefore expects I-talk to be elevated across crisis states regardless of specific disorder." This framing also defends against reviewer pushback that OCD/PTSD/ADHD "crisis" users shouldn't all show the same pronoun pattern.

**Open question.** How does I-talk evolve *within-person* over time around crisis moments? The cross-sectional literature is extensive; longitudinal within-person evidence is thin. **This is exactly the gap the paper addresses — emphasize it in the Introduction.**

### 2.3 Seminal CLPsych papers 2014–2025 on longitudinal analysis

CLPsych (Workshop on Computational Linguistics and Clinical Psychology) is the central venue. The ACL Anthology lists ten workshops through 2025. The most consequential papers for this project:

- **Coppersmith, Dredze & Harman (2014)** "Quantifying Mental Health Signals in Twitter." CLPsych 2014. Founding paper of the subfield. Self-report regex labeling; LIWC feature differences between self-reported-diagnosis and control users.
- **Coppersmith, Dredze, Harman, Hollingshead & Mitchell (2015)** "CLPsych 2015 Shared Task: Depression and PTSD on Twitter." CLPsych 2015.
- **Coppersmith, Leary, Crutchley & Fine (2018)** "Natural Language Processing of Social Media as Screening for Suicide Risk." *Biomedical Informatics Insights*, 10. Operational deployment questions.
- **Zirikly, Resnik, Uzuner & Hollingshead (2019)** "CLPsych 2019 Shared Task: Predicting the Degree of Suicide Risk in Reddit Posts." CLPsych 2019, pp. 24–33. DOI: 10.18653/v1/W19-3003. **The canonical 4-level suicide risk task on the UMD dataset.** Essential citation.
- **Tsakalidis et al. (2022)** ACL long + CLPsych shared task overview — cited in §1.3.
- **MacAvaney, Mittu, Coppersmith, Leintz & Resnik (2021)** "Community-level Research on Suicidality Prediction in a Secure Environment." CLPsych 2021.
- **Chim et al. (2024)** CLPsych 2024 shared task — LLM-based evidence extraction.
- **CLPsych 2025** shared task (Zirikly, Yates co-chairs; NAACL 2025, Albuquerque) focused on the **MIND framework (Atzil-Slonim 2024)** — self-states as Affect/Behavior/Cognition/Desire (ABCD) combinations with fine-grained transitions.
- **CLPsych 2026** shared task (per clpsych.org) builds on MIND and targets adaptive vs maladaptive self-state dimensions over timelines. Relevant if the author submits to CLPsych 2027.

### 2.4 VADER for mental health applications and its limitations

- **Hutto, C. J. & Gilbert, E. (2014)** "VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text." *ICWSM 2014*. The canonical VADER paper. Lexicon + rules (negation, intensifiers, punctuation, capitalization). Trained on social-media-style text.

**Known limitations for mental-health text.** (i) VADER was validated on movie reviews, tweets, news, product reviews — **not clinical or mental-health text**. (ii) Clinical terms ("manic," "panic," "anxiety") have affective polarity that misfires in first-person symptom reporting. (iii) Sarcasm and irony are explicitly unhandled. (iv) Context-free: "not happy" is handled by a negation rule, but "happy that I'm not suicidal anymore" is not. (v) Recent replications (Zhang et al. 2023; Ribeiro et al. 2016 on sentiment benchmark robustness) show VADER underperforms BERT-based sentiment models on informal social text by 10–15 F1.

**Framing.** The paper should cite VADER as a **baseline / coarse signal** and acknowledge in the limitations that for fine-grained mental-health mood detection, transformer-based sentiment (e.g., twitter-roberta-base-sentiment) or NRC-VAD is preferable.

### 2.5 MentalBERT, MentaLLaMA, and domain-specific LMs

- **Ji, S., Zhang, T., Ansari, L., Fu, J., Tiwari, P. & Cambria, E. (2021)** "MentalBERT: Publicly Available Pretrained Language Models for Mental Healthcare." arXiv:2110.15621 (later published at LREC 2022). Continued pretraining of BERT/RoBERTa on Reddit mental-health subreddits. **The expected neural baseline for any Reddit mental-health paper since 2022.**
- **Yang, K., Zhang, T., Kuang, Z., Xie, Q., Huang, J. & Ananiadou, S. (2024)** "MentaLLaMA: Interpretable Mental Health Analysis on Social Media with Large Language Models." *WWW 2024*. arXiv:2309.13567. First instruction-tuned mental-health LLaMA; introduces interpretable analysis.
- **Xu, X., Yao, B., Dong, Y., Gabriel, S., Yu, H., Hendler, J., Ghassemi, M., Dey, A. K. & Wang, D. (2024)** "Mental-LLM: Leveraging Large Language Models for Mental Health Prediction via Online Text Data." *Proc. ACM IMWUT*, 8(1), Article 31. DOI: 10.1145/3643540. Comprehensive LLM benchmarking on 10 mental-health datasets.
- **Vajre, V., Ryan, M., Bhatia, A., Gopalakrishnan, A. & Abinaya, K. (2021)** "PsychBERT: A Mental Health Language Model for Social Media Mental Health Behavioral Analysis." *IEEE BIBM 2021*. Alternative domain-adapted model; less cited than MentalBERT but worth noting.
- **Harrigian, K., Aguirre, C. & Dredze, M. (2020)** "Do Models of Mental Health Based on Social Media Data Generalize?" *Findings of EMNLP 2020*. DOI: 10.18653/v1/2020.findings-emnlp.337. Essential caveat: domain-adapted models drop 10–30 F1 across corpora/platforms.

**Framing.** Reviewers will expect **at least one fine-tuned MentalBERT or MentalRoBERTa baseline** to accompany the Random Forest + 127-feature model. A paper without this baseline, published in 2026, reads as out-of-date.

**Open question.** Do LLMs (GPT-4, LLaMA-3, Claude) outperform domain-specific BERTs on longitudinal tasks, or do the latter retain advantages via temporal modeling? Xu et al. 2024 suggest mixed results; this is an active 2025–2026 research front.

---

## 3. Key methods — deep background

### 3.1 PELT change-point detection

**Killick, R., Fearnhead, P. & Eckley, I. A. (2012)** "Optimal Detection of Changepoints With a Linear Computational Cost." *Journal of the American Statistical Association*, 107(500), 1590–1598. DOI: 10.1080/01621459.2012.737745 (arXiv:1101.1438). PELT minimizes a penalized cost function ∑C(y_{τᵢ+1:τᵢ₊₁}) + β·f(m) over all segmentations. Cost C is typically negative log-likelihood per segment (Gaussian mean/variance, etc.); penalty β controls the number of changepoints (BIC: β = p·log n; AIC: β = 2p). The key innovation is **inequality-based pruning**: candidate changepoints that cannot improve future cost are permanently discarded. Under the assumption that expected number of changepoints grows linearly with n, expected time is **O(n)** vs O(n²) for Optimal Partitioning.

**When PELT works**: piecewise-stationary signals with many changepoints (≈linear in n); i.i.d. within-segment observations; well-specified likelihood; correct penalty scale.

**When PELT fails**: (i) few or no real changepoints — pruning rarely fires, complexity degrades to O(n²); (ii) within-segment autocorrelation violates i.i.d.; (iii) heavy-tailed noise with Gaussian cost; (iv) **misspecified penalty** — too large β → under-segmentation (misses subtle shifts); too small β → over-segmentation; (v) short segments relative to noise variance. For per-user linguistic time series in noisy Reddit text, all of these failure modes are plausibly active.

**Truong, C., Oudre, L. & Vayatis (2020)** "Selective review of offline change point detection methods." *Signal Processing*, 167, 107299. DOI: 10.1016/j.sigpro.2019.107299 (arXiv:1801.00718). The canonical review. Provides the reference `ruptures` Python package. Organize along three axes: cost function (L2, L1, rank, kernel/RBF, autoregressive), search method (exact: OP, PELT, pDPA; approximate: BinSeg, Bottom-Up, Window), and constraint on number of changepoints.

**Adams, R. P. & MacKay, D. J. C. (2007)** "Bayesian Online Changepoint Detection." arXiv:0710.3742. BOCPD models the posterior over "run length" rₜ (time since last changepoint). Online, probabilistic, modular. Used as a baseline by Tsakalidis et al. for timeline selection. Fails under model mismatch, small samples, and abrupt regime shifts followed by sparse data.

**Top 3 must-cite**: Killick 2012, Truong 2020, Adams & MacKay 2007.

### 3.2 Why supervised > unsupervised change-point for subtle linguistic shifts

Unsupervised CPD detects *distributional* breaks given a cost function; it is blind to clinical meaningfulness. For noisy short social-media text, subtle LIWC/NRC shifts are (a) small relative to within-person variance, (b) confounded by topic/context drift, (c) heavy-tailed, and (d) correlated across features. Information-theoretic penalties (BIC) set detection thresholds that routinely miss subtle-but-meaningful shifts.

When outcome labels exist, **supervised learning leverages label information that unsupervised CPD cannot use**: it learns class-conditional feature weightings, inherits regularization for d ≫ effect-size regimes, and benefits from ensembling. **Aminikhanghahi, S. & Cook, D. J. (2017)** "A survey of methods for time series change point detection." *Knowledge and Information Systems*, 51(2), 339–367. DOI: 10.1007/s10115-016-0987-z — explicit comparison arguing supervised CPD outperforms when labels are available.

**Framing for this paper.** With n=132 users and 127 features and labels available, supervised classification (Random Forest) is the right primary tool. PELT should enter as either (a) a **feature-engineering step** (segment-level aggregates, e.g., "number of PELT-detected changepoints in the 4 weeks preceding label") or (b) a **baseline against which supervised performance is compared**. Do not claim PELT "detects crises" — that would require clinical validation PELT cannot provide.

### 3.3 Per-user z-score normalization

**Statistical rationale.** Transforming each user's feature x to z = (x − μ_user)/σ_user removes between-person differences in baseline level/scale, isolating **within-person** deviations. This is standard in digital phenotyping and affective computing (Sano & Picard 2013, ACII). It addresses the fact that baseline linguistic styles vary enormously across individuals.

**Assumptions.** (i) Approximate normality/symmetry per user (z-scores most interpretable under Gaussianity; violated by pronoun rates and other counts — consider robust MAD-based z or rank transforms). (ii) **Stationarity within-user** over the baseline window — if the user is currently undergoing the very shift you want to detect, z-scoring anchored to a contaminated baseline shrinks the signal. (iii) Sufficient per-user sample size (~20–30 posts minimum) to estimate μ, σ reliably; below this, σ is noisy and z-scores unstable. (iv) Observations approximately exchangeable within user (violated by strong autocorrelation, common in posting streams).

**Limitations.** Removes all between-person signal — if the *level* of a feature (not its deviation) predicts outcome, per-user z destroys that. Data-leakage risk in CV if μ_user/σ_user are computed on the full series including test windows — must be computed on train-only. Amplifies noise in sparse posters. Under non-stationarity, produces spurious deviations anchored to contaminated baselines. Can remove the very regime-shift effect of interest in small longitudinal samples.

**Framing.** The paper should (a) state the windowing scheme (e.g., "z-scores computed on each user's first N posts, excluding the pre-label week"), (b) acknowledge the stationarity assumption explicitly, (c) report sensitivity to sparse posters (posts per user distribution), (d) consider **mixed-effects models** (user random intercept) as a principled alternative for future work.

### 3.4 LIWC: what it measures, validation, critiques

Versions: **LIWC2001, LIWC2007, LIWC2015, LIWC-22** (Boyd, Ashokkumar, Seraj & Pennebaker 2022). LIWC2015 has ~90 categories, 6,549 dictionary entries, and four summary variables (Analytical Thinking, Clout, Authenticity, Emotional Tone). Categories: linguistic processes (pronouns, function words), psychological processes (affect, cognitive, social), personal concerns (work, leisure, money, death), spoken-language markers.

**Canonical review**: **Tausczik & Pennebaker 2010** (*J. Lang. Soc. Psychol.*), cited above.

**Critiques**. **Schwartz, H. A., Eichstaedt, J. C., Kern, M. L. et al. (2013)** "Personality, gender, and age in the language of social media: The open-vocabulary approach." *PLOS ONE*, 8(9), e73791. DOI: 10.1371/journal.pone.0073791. On 75,000 Facebook users, open-vocabulary DLA (unigrams, bigrams, LDA topics) substantially outperforms closed-vocabulary LIWC for personality/age/gender prediction. Critique themes: (i) closed a priori categories miss unanticipated markers; (ii) category coherence is questionable (articles grouped by POS, not semantics); (iii) coverage incomplete (6,549 words — slang and mental-health jargon missed); (iv) context-insensitive, cannot handle negation; (v) non-orthogonal categories; (vi) proprietary summary variables with opaque derivations.

**Eichstaedt, J. C., Kern, M. L., Yaden, D. B. et al. (2021)** "Closed- and open-vocabulary approaches to text analysis." *Psychological Methods*, 26(4), 398–427. DOI: 10.1037/met0000349. Direct comparison; recommends both for different purposes.

**Framing.** LIWC is interpretable and comparable across studies — valuable. But for a 2026 paper, the author should also cite Schwartz 2013 / Eichstaedt 2021, acknowledge closed-vocabulary limits, and consider including open-vocabulary or embedding-based features (the paper already does via MPNet embeddings, good).

### 3.5 NRC Emotion Lexicon

- **Mohammad & Turney (2013)** "Crowdsourcing a word–emotion association lexicon." *Computational Intelligence*, 29(3), 436–465. DOI: 10.1111/j.1467-8640.2012.00460.x (arXiv:1308.6297). ~14,000 English words × **Plutchik's 8 emotions** (joy, sadness, anger, fear, trust, disgust, anticipation, surprise) + positive/negative. Binary labels from Amazon MTurk.
- **Mohammad (2018)** "Obtaining reliable human ratings of valence, arousal, and dominance for 20,000 English words." *ACL 2018*, 174–184. DOI: 10.18653/v1/P18-1017. **NRC VAD Lexicon** — 20k words rated on continuous V/A/D via **Best–Worst Scaling**, substantially more reliable than Likert.

**Critiques**. Context-insensitive (no negation, sarcasm, intensifiers). Demographic bias in crowdworker annotations (Mohammad 2022 "Ethics sheet for automatic emotion recognition," arXiv:2011.03492). Coverage gaps for clinical/slang vocabulary. Binary association in EmoLex loses graded information. Domain shift from general MTurk to clinical populations is a threat.

**Framing.** Cite both Mohammad & Turney 2013 (discrete) and Mohammad 2018 (dimensional). Acknowledge lexical context-blindness in limitations.

### 3.6 Cross-validation for small, imbalanced datasets

The combination of **n=132 users, 127 features, and 3 classes (73/59/~remaining-from-449)** is severely under-powered. Must-cite:

- **Cawley, G. C. & Talbot, N. L. C. (2010)** "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation." *JMLR*, 11, 2079–2107. **Tuning hyperparameters on the same CV folds used for performance reporting introduces optimistic selection bias** comparable to between-algorithm differences. Recommends **nested CV** (outer for performance, inner for tuning).
- **Varoquaux, G. (2018)** "Cross-validation failure: Small sample sizes lead to large error bars." *NeuroImage*, 180(A), 68–77. DOI: 10.1016/j.neuroimage.2017.06.061 (arXiv:1706.07581). At n ≈ 100, the 95% CI on CV-accuracy is ≈ ±10 percentage points — much wider than standard error across folds suggests. Folds are correlated; between-fold SE underestimates true sampling variance.
- **Ojala, M. & Garriga, G. C. (2010)** "Permutation tests for studying classifier performance." *JMLR*, 11, 1833–1863. Non-parametric permutation tests (label permutation, feature permutation) establish significance vs chance without distributional assumptions. Essential for small n.
- **Riley, R. D., Ensor, J., Snell, K. I. E. et al. (2020)** "Calculating the sample size required for developing a clinical prediction model." *BMJ*, 368, m441. DOI: 10.1136/bmj.m441. Superseded the old "10 events per variable" rule. For 127 candidate predictors, realistic prevalence, and a multinomial outcome, needed n is in the **high hundreds to low thousands** for a stable model (per Pate, Riley et al. 2023 multinomial extension, *Stat Methods Med Res* 32(3):555–571).
- **Vabalas, A., Gowen, E., Poliakoff, E. & Casson, A. J. (2019)** "Machine learning algorithm validation with a limited sample size." *PLOS ONE*. Shows K-fold CV produces optimistic estimates below n ≈ 1000.

**Framing.** With n=132 and 127 features, effective events-per-predictor ≈ 1 (a Riley-threshold violation by ~10×). The paper must (a) use **nested CV** or a held-out test set, (b) report **bootstrap confidence intervals** on AUC/F1, (c) report **permutation p-values**, (d) frame results as **exploratory / pilot** (not "demonstrates"), (e) show a **learning curve** (n=30/60/90/132) to expose instability.

### 3.7 ROC-AUC interpretation for imbalanced multiclass

- **Hand, D. J. & Till, R. J. (2001)** "A simple generalisation of the area under the ROC curve for multiple class classification problems." *Machine Learning*, 45(2), 171–186. DOI: 10.1023/A:1010920819831. Macro-averaged one-vs-one AUC: M = (2/(C(C−1))) · ∑_{i<j} A(i,j). This is what scikit-learn's `roc_auc_score(..., multi_class='ovo', average='macro')` computes. Equal class weighting — appropriate when rare classes matter.
- **Davis, J. & Goadrich, M. (2006)** "The relationship between precision-recall and ROC curves." *ICML 2006*. A curve dominates in ROC space iff it dominates in PR space, but PR curves are far more sensitive to imbalance because precision is prediction-normalized.
- **Saito, T. & Rehmsmeier, M. (2015)** "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets." *PLOS ONE*, 10(3), e0118432. DOI: 10.1371/journal.pone.0118432. Baseline PR-AUC equals positive class prevalence, not 0.5.
- **Fawcett, T. (2006)** "An introduction to ROC analysis." *Pattern Recognition Letters*, 27(8), 861–874. Comprehensive ROC reference.

**What macro AUC 0.688 + macro F1 0.396 actually means.** Macro AUC 0.688 = **moderate ranking ability** (randomly chosen pair from classes i and j ranked correctly ~69% of the time on average). Meaningfully above chance (0.5) but well below the 0.8+ typically called "good." At n=132 the 95% CI is plausibly ±0.06–0.10, so the lower bound may approach chance. Macro F1 0.396 = **poor threshold/calibration**, especially on rare classes where precision and recall are each limited by few positives. **The gap between AUC 0.688 and F1 0.396 is the diagnostic signature of a classifier that learns some signal but whose decision threshold is not calibrated to the multi-class imbalance.**

**Remedies**: threshold tuning per class, probability calibration (Platt scaling, isotonic regression — Niculescu-Mizil & Caruana 2005 ICML), class-weighted Random Forest or resampling, **reporting PR-AUC alongside ROC-AUC**.

---

## 4. Ethics and research design

### 4.1 Ethical frameworks for Reddit mental-health research

**Consent asymmetry** is the core issue: posts are public, but users did not consent to research use. Must-cite framework:

- **Nissenbaum, H. (2004, 2010)** "Privacy as Contextual Integrity," *Washington Law Review* 79(1), and *Privacy in Context*, Stanford UP. **Contextual integrity**: privacy violations occur when information flows violate context-specific informational norms. A r/depression peer-support post has very different norms from the same text aggregated into an ML corpus on arXiv.
- **Zimmer, M. (2010)** "But the data is already public." *Ethics and Information Technology*, 12(4), 313–325. DOI: 10.1007/s10676-010-9227-5. Canonical counter to the "public = fair game" argument. De-identification routinely fails.
- **Fiesler, C. & Proferes, N. (2018)** "'Participant' Perceptions of Twitter Research Ethics." *Social Media + Society*, 4(1). DOI: 10.1177/2056305118763366. Most users surveyed were unaware their posts could be used for research and felt researchers should seek consent, especially for sensitive topics.
- **Benton, A., Coppersmith, G. & Dredze, M. (2017)** "Ethical Research Protocols for Social Media Health Research." *ACL EthNLP Workshop*, W17-1612. The de facto operational ethics checklist: minimize PII, paraphrase examples, restrict data redistribution, etc.
- **Fiesler, C., Zimmer, M., Proferes, N., Gilbert, S. & Jones, N. (2024)** "Remember the Human: A Systematic Review of Ethical Considerations in Reddit Research." *Proc. ACM HCI*, 8(GROUP), Article 5. DOI: 10.1145/3633070. Most recent, Reddit-specific systematic review.
- **Ajmani, L. H., Chancellor, S., Mehta, B., Fiesler, C., Zimmer, M. & De Choudhury, M. (2023)** "A Systematic Review of Ethics Disclosures in Predictive Mental Health Research." *FAccT 2023*, 1311–1323. DOI: 10.1145/3593013.3594082. The checklist for 2024+ ethics sections.

**Concrete sentences for the paper**:
> "Although all posts analyzed were publicly visible, we follow Nissenbaum (2004, 2010) in recognizing that contextual integrity—not mere accessibility—is the appropriate privacy norm. Following Zimmer (2010) and Fiesler & Proferes (2018), we do not treat 'public' as equivalent to consent. We implement the protocol of Benton, Coppersmith & Dredze (2017): we paraphrase all example posts to prevent reidentification, we report only aggregate statistics, and we do not release the raw user-level dataset."

### 4.2 CLPsych-specific ethics requirements

The UMD Reddit Suicidality Dataset (Shing et al. 2018) is the template. It is shared only via a **formal application + signed Data Use Agreement** through the American Association of Suicidology, requiring: PI affiliation, commitment to Benton 2017 principles, citation of Shing 2018 & Zirikly 2019, and a data-management plan.

**For a high-school researcher** who cannot sign institutional DUAs, the defensible posture is: use only already-public, non-gated Reddit data; state explicitly that UMD-RSD and SMHD were not used because they require DUAs unavailable to individual researchers; voluntarily adhere to their ethical principles anyway; paraphrase all quoted examples; do not release raw text; include a "Data Availability Statement" specifying code and aggregate stats only.

Per CLPsych 2025 CFP: **"papers without a limitations section will be desk-rejected"** (NAACL policy). Ethics + limitations are mandatory, not optional.

### 4.3 Misuse prevention

Chancellor et al. (2019, FAT*) catalog dual-use harms: insurance/employment discrimination, government surveillance, unsanctioned platform-side intervention, stigma amplification via false positives. Must-cite:

- **Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D. & Gebru, T. (2019)** "Model Cards for Model Reporting." *FAT* 2019*, 220–229. DOI: 10.1145/3287560.3287596.
- **Gebru, T., Morgenstern, J., Vecchione, B., Wortman Vaughan, J., Wallach, H., Daumé III, H. & Crawford, K. (2021)** "Datasheets for Datasets." *Commun. ACM*, 64(12), 86–92. DOI: 10.1145/3458723.

**Concrete sentence**: "We publish a model card specifying intended use (exploratory research), **out-of-scope uses** (clinical diagnosis, insurance underwriting, employment screening, law-enforcement monitoring, and any application to users who did not self-select into a mental-health community), and known degradations across demographic and subreddit groups."

### 4.4 Detection vs clinical diagnosis

**Why this matters legally and ethically.** If a Reddit-trained classifier claims "depression detection" and a third party commercializes it as screening, **FDA may regulate it as Software as a Medical Device (SaMD)** under the 2019/2022 Clinical Decision Support guidance. HIPAA applies once outputs are linked to an identified patient by a covered entity. Construct validity is the core issue: Reddit classifiers predict *self-disclosure of a label on a specific platform*, not DSM-5 diagnoses.

**Must-cite**: **Ernala, S. K., Birnbaum, M. L., Candan, K. A., Rizvi, A. F., Sterling, W. A., Kane, J. M. & De Choudhury, M. (2019)** "Methodological Gaps in Predicting Mental Health States from Social Media: Triangulating Diagnostic Signals." *CHI 2019*, Paper 134. DOI: 10.1145/3290605.3300364. Empirical proof that proxy-trained models (self-reports, subreddit membership, clinician appraisal of public posts) have **strong internal validity but poor external validity** on clinically diagnosed patients.

**Concrete sentence**: "We emphasize that our classifier detects **self-reported linguistic markers associated with membership in a mental-health subreddit**, not clinical diagnoses. Following Ernala et al. (2019), we expect substantial degradation when applied to populations that do not self-select into mental-health communities. This model is not a screening, diagnostic, or triage tool and is not validated for clinical use."

### 4.5 Chancellor & De Choudhury's critiques — the central engagement

These are the references reviewers will check for:

1. **Chancellor, S. & De Choudhury, M. (2020)** "Methods in predictive techniques for mental health status on social media: a critical review." *npj Digital Medicine*, 3, 43. DOI: 10.1038/s41746-020-0233-7. **Systematic review of 75 papers.** Core finding: "concerning trends around construct validity, and a lack of reflection in the methods used to operationalize and identify mental health status." Proposes reporting standards. **Any 2024+ mental-health paper that does not cite this reads as not having engaged the field.**
2. **Chancellor, S., Birnbaum, M. L., Caine, E. D., Silenzio, V. M. B. & De Choudhury, M. (2019)** "A Taxonomy of Ethical Tensions in Inferring Mental Health States from Social Media." *FAT* 2019*, 79–88. DOI: 10.1145/3287560.3287587. Three tension categories: ethics-committee gap, validity/data/ML, stakeholder-level implications.
3. **De Choudhury, M., Gamon, M., Counts, S. & Horvitz, E. (2013)** "Predicting Depression via Social Media." *ICWSM 2013*, 128–137. DOI: 10.1609/icwsm.v7i1.14432. The progenitor paper.
4. **De Choudhury et al. (2016)** CHI — cited in §1.2. **The single most direct methodological precedent.**
5. **Ernala et al. (2019)** — cited above. **The single most important limitations citation.**
6. **Chancellor, S., Baumer, E. P. S. & De Choudhury, M. (2019)** "Who is the 'Human' in Human-Centered Machine Learning: The Case of Predicting Mental Health from Social Media." *Proc. ACM HCI*, 3(CSCW), Article 147. DOI: 10.1145/3359249.

### 4.6 Published critiques of Reddit-based research

- **Sampling bias**: Reddit users ≠ general population; mental-health subreddit posters ≠ Reddit users (younger, more male, Western, English-speaking, digitally literate). **Olteanu, A., Castillo, C., Diaz, F. & Kıcıman, E. (2019)** "Social Data: Biases, Methodological Pitfalls, and Ethical Boundaries." *Frontiers in Big Data*, 2, 13.
- **Self-diagnosis vs clinical**: SMHD's regex labels ("I was diagnosed with X") are noisy. Subreddit membership is weaker — users post for many reasons.
- **Ground-truth problem**: Ernala et al. 2019.
- **Temporal confounders**: status changes over time (MacAvaney et al. 2018 RSDD-Time, DOI: 10.18653/v1/W18-0618); platform/algorithm drift; language drift; COVID-19 regime shifts (directly relevant — this dataset spans 2019–2021); **Gaffney & Matias 2018** on Pushshift missing data.
- **Cross-platform generalization**: **Harrigian, Aguirre & Dredze 2020** — mental-health models generalize poorly across time and platforms.

---

## 5. Temporal and behavioral features

### 5.1 Posting-time patterns as psychological signals

- **De Choudhury, Gamon, Counts & Horvitz (2013)** ICWSM — cited above. Foundational paper using posting-time features (volume, diurnal patterns, insomnia index) for depression prediction from Twitter.
- **De Choudhury, Counts & Horvitz (2013)** "Social media as a measurement tool of depression in populations." *WebSci 2013*. Volume, engagement, and diurnal variation as depression signals.
- **De Choudhury, Counts, Horvitz & Hoff (2014)** "Characterizing and predicting postpartum depression from shared Facebook data." *CSCW 2014*. Temporal features + linguistic features.
- **Golder, S. A. & Macy, M. W. (2011)** "Diurnal and Seasonal Mood Vary with Work, Sleep, and Daylength Across Diverse Cultures." *Science*, 333(6051), 1878–1881. DOI: 10.1126/science.1202775. **Canonical Twitter diurnal-mood paper.** Establishes that posting time reflects circadian mood variation across cultures.
- **Reece, A. G. & Danforth, C. M. (2017)** "Instagram photos reveal predictive markers of depression." *EPJ Data Science*, 6, 15. Temporal + visual features.
- **Saha, K. & De Choudhury, M. (2017)** "Modeling Stress with Social Media Around Incidents of Gun Violence on College Campuses." *Proc. ACM HCI*, 1(CSCW). Temporal stress signals.

### 5.2 Sleep disruption as linguistic/behavioral marker

Sleep disruption is a core feature of depression (Harvey, A. G. 2008, "Sleep and circadian rhythms in bipolar disorder." *American Journal of Psychiatry*, 165(7), 820–829) and mania (Wehr, T. A. et al. 1987, sleep deprivation as mania trigger). In social media:

- **Jamison-Powell, S., Linehan, C., Daley, L., Garbett, A. & Lawson, S. (2012)** "'I can't get no sleep': discussing #insomnia on Twitter." *CHI 2012*, 1501–1510. DOI: 10.1145/2207676.2208612. Canonical social-media sleep-complaint paper.
- **Suhara, Y., Xu, Y. & Pentland, A. (2017)** "DeepMood: Forecasting Depressed Mood Based on Self-Reported Histories via Recurrent Neural Networks." *WWW 2017*, 715–724. DOI: 10.1145/3038912.3052676. Shows sleep + prior-mood temporal patterns strongly predict next-day depressed mood.
- **Seabrook, E. M., Kern, M. L., Fulcher, B. D. & Rickard, N. S. (2018)** "Predicting Depression from Language-Based Emotion Dynamics." *JMIR Mental Health*, 5(1), e4. Integrates linguistic emotion dynamics over time.

Clinical anchor: **American Psychiatric Association DSM-5** lists "insomnia or hypersomnia nearly every day" as a core MDD criterion.

### 5.3 "When you post" vs "what you post"

The growing consensus in digital phenotyping is that **behavioral features (timing, frequency, burstiness) add non-redundant signal to linguistic features**. De Choudhury 2013, Reece & Danforth 2017, and the Tsakalidis et al. 2022 MoC corpus all demonstrate this. For this paper, the temporal features (hour entropy, late-night rate, inter-post intervals, weekend rate) are well-motivated by this literature — but the author should cite at least **De Choudhury 2013, Golder & Macy 2011, and Jamison-Powell 2012** explicitly when introducing these features.

**Open question**. The literature has not cleanly isolated the causal direction: does disrupted sleep → more late-night posting → depression, or does depression cause both? Nor has it established whether timing features add predictive value *after* controlling for linguistic content in longitudinal within-person designs. This is another gap the paper can claim to explore.

---

## 6. Venues and citation strategy

### 6.1 arXiv category

**Primary: cs.CL** (Computation and Language) is correct and expected. **Recommended cross-listings**:
- **cs.LG** (Machine Learning) — appropriate given the Random Forest / supervised-learning focus and feature engineering.
- **cs.CY** (Computers and Society) — strongly recommended given the ethics section and dual-use considerations; signals engagement with responsibility frameworks.
- **cs.SI** (Social and Information Networks) — appropriate given Reddit network context and temporal posting analysis.

Typical pattern for CLPsych-adjacent papers on arXiv: cs.CL primary + cs.LG + cs.CY. Avoid q-bio.QM unless the paper makes explicit clinical/quantitative-biology claims.

### 6.2 CLPsych 2027 timeline

CLPsych has been co-located with NAACL/ACL/EACL annually. Based on the historical pattern:

- **CLPsych 2022**: with NAACL 2022, Seattle, July 2022.
- **CLPsych 2024**: with EACL 2024, St. Julians, Malta, March 2024.
- **CLPsych 2025**: with NAACL 2025, Albuquerque, May 2025. Submission deadline Jan 30, 2025; notification Mar 1; camera-ready Mar 10.
- **CLPsych 2026**: per clpsych.org, shared task focuses on MIND/ABCD self-state modeling.
- **CLPsych 2027**: likely co-located with NAACL 2027 or ACL 2027. Expect **submission deadline in Jan–Feb 2027** for a spring/early-summer workshop. Both long (8 pp.) and short (4 pp.) tracks accepted; limitations section is mandatory.

**Strategy for the author**. The August 2026 arXiv release positions the paper well for a CLPsych 2027 submission: the arXiv preprint establishes priority; the workshop version can be expanded with mentor-led extensions (e.g., a MentalBERT baseline, cross-subreddit generalization experiments, CLPsych 2025/2026 shared-task alignment). CLPsych is unusually welcoming to student-led work and interdisciplinary contributions.

### 6.3 Other venues

- **ACL Student Research Workshop (SRW)** — **strongly recommended for a high-school author.** ACL SRW accepts pre-doctoral work, provides mentorship, and has an excellent track record of publishing undergraduate and, occasionally, secondary-school contributions. The SRW deadline typically precedes ACL by 2–3 months.
- **NAACL SRW** and **EMNLP (no SRW but Findings track)** are alternatives.
- **Workshops**: **LOUHI** (Health Text Mining, at EMNLP), **WASSA** (Subjectivity/Sentiment/Social Media), **WNUT** (Noisy User-generated Text), **SMM4H** (Social Media Mining for Health).
- **Journals**: **JMIR Mental Health**, **npj Digital Medicine**, **JAMIA**, **Journal of Biomedical Informatics**.

### 6.4 Top 15 must-cite papers

Ranked by necessity:

1. **Rude, Gortner & Pennebaker (2004)** *Cognition & Emotion* — first-person pronouns and depression.
2. **Tackman et al. (2019)** *JPSP* — I-talk meta-analysis (specificity vs negative emotionality).
3. **Tausczik & Pennebaker (2010)** *J. Lang. Soc. Psychol.* — LIWC canonical review.
4. **Coppersmith, Dredze & Harman (2014)** CLPsych — Twitter mental-health progenitor.
5. **De Choudhury et al. (2013)** ICWSM — Predicting Depression via Social Media.
6. **De Choudhury et al. (2016)** CHI — Shifts to suicidal ideation on Reddit mental-health subreddits (direct precedent).
7. **Cohan et al. (2018)** COLING — SMHD dataset (same subreddit set).
8. **Shing et al. (2018)** CLPsych — UMD Reddit Suicidality Dataset.
9. **Tsakalidis et al. (2022)** ACL — Moments of Change (longitudinal framework).
10. **Ji et al. (2021)** arXiv:2110.15621 — MentalBERT (expected neural baseline).
11. **Chancellor & De Choudhury (2020)** *npj Digital Medicine* — critical review / reporting standards.
12. **Ernala et al. (2019)** CHI — methodological gaps / construct validity.
13. **Benton, Coppersmith & Dredze (2017)** EACL EthNLP — ethical protocols.
14. **Hutto & Gilbert (2014)** ICWSM — VADER.
15. **Killick, Fearnhead & Eckley (2012)** *JASA* — PELT.

**Also essential, just below the top 15**: Mohammad & Turney 2013 (NRC EmoLex), Mohammad 2018 (NRC VAD), Reimers & Gurevych 2019 *EMNLP* Sentence-BERT (arXiv:1908.10084), Song et al. 2020 *NeurIPS* MPNet (arXiv:2004.09297), Varoquaux 2018 *NeuroImage*, Harrigian, Aguirre & Dredze 2020 *Findings of EMNLP*, Gaffney & Matias 2018 *PLOS ONE*, Mitchell et al. 2019 model cards.

### 6.5 Recent (2023–2026) work the paper must acknowledge

Missing any of these makes a 2026 paper feel out of date:

- **Yang et al. (2024)** MentaLLaMA (*WWW 2024*, arXiv:2309.13567) and **Xu et al. (2024)** Mental-LLM (*IMWUT*, DOI 10.1145/3643540) — LLM-based mental-health analysis.
- **Chim et al. (2024)** CLPsych 2024 shared task — LLM evidence extraction.
- **Hills et al. (2024)** *Findings of ACL* — Exciting Mood Changes, current SoTA on MoC.
- **Ajmani et al. (2023)** FAccT — ethics disclosure review.
- **Tseriotou et al. (2024)** EACL Sig-Networks — signature-based networks for longitudinal language modeling.
- **CLPsych 2025 shared task** (MIND framework, Atzil-Slonim 2024).
- **Fiesler et al. (2024)** *Proc. ACM HCI* — Reddit research ethics systematic review.

---

## 7. Limitations to acknowledge

### 7.1 What reviewers will expect

Reviewing limitations sections across the canonical Reddit mental-health NLP corpus yields a stable reviewer-expectation checklist:

- **Weak ground truth**: self-report regexes / subreddit membership / heuristic labels ≠ clinical diagnosis. **Cite Ernala et al. 2019 and Chancellor & De Choudhury 2020.**
- **Platform/demographic bias**: Reddit skews younger, male, Western, English. **Cite Harrigian et al. 2020 and Olteanu et al. 2019.**
- **Poor cross-dataset generalization**: **Cite Harrigian et al. 2020 explicitly.**
- **Construct validity**: what exactly does "depression" mean in your operationalization?
- **Small samples with absent CIs**: reviewers expect bootstrap CIs and permutation p-values at n < 500.
- **Missing baselines**: majority class, TF-IDF + logistic regression, LIWC + logistic regression, MentalBERT fine-tuned.
- **Temporal leakage**: user-disjoint splits are mandatory; chronological ordering of train→test is strongly preferred.
- **Reporting standards**: Chancellor & De Choudhury 2020 identify 5 minimum standards most papers fail to meet.
- **Ethics + dual-use**: FAccT 2019 taxonomy; model card.

### 7.2 What 132 users can support

**Statistical power.** With 127 features and ~130 labeled users, effective events-per-predictor ≈ 1 — below Riley et al. 2020 thresholds by roughly an order of magnitude. At this n, Hanley–McNeil SE for AUC ≈ 0.69 is ≈ 0.05, giving a **95% CI approximately [0.59, 0.79]** — the lower bound doesn't cleanly separate from chance.

**Defensible claims.**

| Avoid | Use instead |
|---|---|
| "demonstrates / proves" | "is consistent with / suggests" |
| "detects mental health crises" | "classifies users by proxy-labeled linguistic patterns" |
| "state-of-the-art" | "feasibility benchmark on a small labeled sample" |
| "predictive features" | "candidate features warranting validation in larger cohorts" |

**Concrete sentence to adopt**: "Following Varoquaux (2018), at n=132 the standard error across CV folds substantially underestimates true predictive uncertainty, and our macro-AUC of 0.688 should be interpreted as exploratory. Applying Riley et al. (2020) and Pate et al. (2023) to a 3-class model with 127 candidate predictors suggests this dataset is below the threshold for stable multivariable prediction modeling; we therefore frame our work as a pilot study rather than a validated predictive tool."

### 7.3 Generalizability: OCD dominates (61/132 ≈ 46%)

Reviewers will immediately note that nearly half of the labeled signal may reflect **OCD-specific linguistic patterns** (rumination about intrusive thoughts, reassurance-seeking, compulsion/ritual vocabulary — Gkotsis et al. 2017 *Scientific Reports* 7:45141) rather than a generalizable crisis/recovery marker.

**Required ablations.** Per-subreddit performance breakdown; leave-one-subreddit-out cross-validation; feature-importance overlap across disorders; train-on-OCD/test-on-non-OCD (and vice versa).

**Concrete sentence**: "Because 46% of labeled users were drawn from r/OCD, features that appear predictive may reflect OCD-specific linguistic markers rather than a generalized crisis/recovery signal. We do not claim the classifier would transfer to other mental-health subreddits, and prior work (Harrigian et al. 2020) finds that cross-subreddit generalization for depression detection is substantially weaker than within-subreddit performance."

### 7.4 The "we don't know if users actually had crises" problem

This is the deepest vulnerability. The solomonk dataset does not include clinician-rated labels; "crisis" and "recovery" are heuristic operationalizations. **Ernala et al. 2019** shows that proxy-label models with strong internal validity fail on clinically diagnosed patients. Must-cite limitations-paragraph template:

> "A fundamental limitation is construct validity. We do not have clinical confirmation that labeled 'crisis' users experienced clinical crises, nor that 'recovery' users met recovery criteria; labels are derived from [describe procedure] and therefore operationalize a behavioral/linguistic proxy rather than a diagnostic ground truth. As Ernala et al. (2019) show, predictive models built on such proxies can achieve strong internal validity while failing to generalize to clinically diagnosed populations, and Chancellor & De Choudhury (2020) document that this gap is pervasive across the 75 papers they reviewed. Following De Choudhury et al. (2016), we emphasize that 'we can only make a weak inference' about the actual mental-health state of these users. Our contribution is methodological—a feasibility study of feature-based classification over a publicly available self-disclosure corpus—rather than a clinical detection tool."

### 7.5 Exemplary limitations sections to model

1. **De Choudhury et al. 2016 CHI** — explicit disclaimer of diagnostic claims.
2. **Cohan et al. 2018 COLING (SMHD)** — precision/recall tradeoff of self-report regexes; control contamination.
3. **Shing et al. 2018 CLPsych** — small expert-annotated subset; crowd–expert disagreement.
4. **Zirikly et al. 2019 CLPsych** — task-card style: construct, label source, agreement, chance baselines, ethics.
5. **Harrigian et al. 2020 *Findings EMNLP*** — an entire paper structured as field-level limitations; template for platform/subreddit/time/demographic decomposition.

### 7.6 Other reviewer-expected elements

- **Confusion matrix with per-class F1.**
- **Permutation-importance or SHAP** for feature importance (Random Forest's built-in Gini importance is biased toward high-cardinality features).
- **Feature-group ablations**: LIWC vs n-gram vs stylometric vs temporal vs embeddings.
- **Learning curve**: performance at n=30, 60, 90, 132 to visualize instability.
- **Reproducibility**: all 127 features enumerated in an appendix; seeds; library versions; code released.
- **Error analysis**: inspect 10–20 misclassified users, per-subreddit error breakdown.
- **Ethics statement as a standalone section** before references, covering Benton 2017 + Chancellor 2019 (FAccT) + model card + out-of-scope uses.

---

## Conclusion: how this knowledge base should shape the paper

Three narrative pivots convert this project from a potentially desk-rejectable small-sample ML result into a publishable, defensible student-research contribution:

**First, frame the work as a pilot / exploratory feasibility study, not a detection tool.** The construct-validity literature (Chancellor & De Choudhury 2020; Ernala et al. 2019) and sample-size literature (Riley et al. 2020; Varoquaux 2018) converge on the same conclusion: n=132 with 127 features and proxy labels cannot support clinical claims. Framing it honestly as a pilot aligns the paper with the field's current epistemological norms and makes reviewers sympathetic rather than hostile.

**Second, position the longitudinal-within-person angle as the novel contribution, not the classifier's raw performance.** The cross-sectional linguistic-markers literature is vast (Tausczik & Pennebaker 2010; Tackman et al. 2019); the within-person temporal evolution around turning points is the thin spot. The paper's intellectual novelty is in *asking whether pre-label linguistic drift is detectable*, not in achieving high AUC. Frame the AUC 0.688 / F1 0.396 as evidence that such drift exists above chance but is noisy and small — which is exactly what the Tackman 2019 meta-analytic effect sizes (r ≈ 0.10) predict.

**Third, engage directly and prominently with the Chancellor–De Choudhury–Ernala critique tradition in the Introduction, Ethics, and Limitations sections.** This signals to reviewers that the author has read the field's self-criticism and is not repeating known methodological errors. Specifically: cite Chancellor & De Choudhury 2020 as the source of reporting standards; cite Ernala et al. 2019 for construct validity; cite Chancellor et al. 2019 (FAccT) for ethical taxonomy; cite Benton et al. 2017 for operational ethics. Add a model card per Mitchell et al. 2019 with explicit out-of-scope uses.

If the paper follows these framings, cites the top-15 list, includes bootstrap CIs and permutation tests, and provides per-subreddit ablations, it will read as a rigorous student pilot and is well-positioned for arXiv, ACL SRW, and CLPsych 2027. The single most leverage-giving addition to the current pipeline is a **fine-tuned MentalBERT baseline** and a **leave-one-subreddit-out evaluation** — together these preempt the two strongest reviewer objections (missing neural baseline, OCD dominance) and substantially raise the paper's ceiling.