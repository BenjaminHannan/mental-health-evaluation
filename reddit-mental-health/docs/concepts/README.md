# Concepts — Interview Prep

Short explainers for every non-obvious concept in this project, in my own voice. Each file is ≤1 page and aims to be re-readable in under 3 minutes. The goal is that I can answer a cold question about any of these without looking at the paper.

| Concept | One-line hook |
|---|---|
| [Cross-validation](cross-validation.md) | Why 5-fold CV instead of a single train/test split when n=581. |
| [ROC-AUC](roc-auc.md) | Why 0.688 is meaningful when 0.5 is chance, and why macro-F1 0.396 is not as bad as it sounds. |
| [VADER sentiment](vader-sentiment.md) | Lexicon-based sentiment: why I used a rule-based tool instead of a transformer. |
| [Z-score normalization](z-score-normalization.md) | How per-user normalization isolates *change* from baseline writing style. |
| [PELT change-point](pelt-change-point.md) | The unsupervised baseline that barely beat chance — and why that strengthens the paper. |
| [First-person pronouns](first-person-pronouns.md) | The depression marker from Pennebaker's NLP literature, and what I saw in my data. |
| [CLPsych](clpsych-venue.md) | The workshop I'm targeting and why this paper fits. |
| [Ethics of Reddit MH research](ethics-reddit-mh.md) | Public posts ≠ consent, and how that shaped the project. |
| [Baseline style vs temporal shift](baseline-vs-temporal.md) | Why baseline writing style predicts label better than within-user change — and what that means. |
| [Clinical-tool gap](clinical-tool-gap.md) | What would have to change for this to be a real clinical tool. |

## How to use this folder

Before any college interview or CLPsych discussion: read the relevant file, then try to answer the "interviewer push-back" question out loud without looking. If I can't, that's the topic to rehearse.
