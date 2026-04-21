# VADER sentiment

**One-sentence answer:** VADER is a rule-based sentiment analyzer — it looks up words in a curated lexicon, applies rules for negation, intensifiers, and punctuation, and returns a compound score from −1 to +1. I used it because it's fast, interpretable, requires no training, and its behavior is stable over time, which matters when I'm measuring *change* in sentiment.

## What it is
VADER (Valence Aware Dictionary and sEntiment Reasoner, Hutto & Gilbert 2014) is specifically designed for social-media text. Each word in its lexicon has a hand-coded valence score. Rules handle things like "not good" flipping polarity, "very good" amplifying it, and ALL-CAPS boosting intensity. The compound score is the sum of all valences, normalized to [−1, +1].

## Why we used it in this project
Three reasons:
1. **It doesn't need training data.** Our dataset is small and unlabeled at the sentence level; supervised sentiment models would require labels we don't have.
2. **It's deterministic and interpretable.** If a user's compound score drops from +0.2 to −0.4 over four weeks, I can point at specific words that caused the shift. That matters for a paper arguing that a *change* is detectable.
3. **It's stable.** A transformer fine-tuned on sentiment would give different scores depending on the training run and version. VADER gives the same score for the same text forever, which is important when comparing the same user's posts across months.

It's called in `src/extract_features.py` and produces the `sentiment_mean` feature, which shows up in Table 1 of the paper.

## The number that matters
Pre-1w sentiment delta (from the paper, Table of deltas): crisis = −0.070, recovery = +0.216, neither = −0.017. The recovery signal is clearly visible even in raw deltas, without any ML.

## What an interviewer might push on
- **"Wouldn't MentalBERT give better sentiment?"** — Probably yes on *accuracy*. We actually wanted to use it and the checkpoint was gated on HuggingFace, so we fell back to sentence-transformers (mpnet) for embeddings and kept VADER for sentiment. The paper flags this as future work.
- **"Isn't VADER too crude for depression?"** — VADER measures general affect, not clinical depression. That's fine because we're not diagnosing — we're looking for linguistic *shifts*, and general affect shifts are one signal among many.

## The one thing I want to remember
The choice of VADER over a neural model is a deliberate tradeoff for interpretability and reproducibility, not a limitation I was stuck with.
