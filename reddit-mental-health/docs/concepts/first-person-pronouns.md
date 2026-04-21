# First-person pronouns as a depression marker

**One-sentence answer:** James Pennebaker and collaborators have shown across multiple studies that people experiencing depression use first-person singular pronouns ("I," "me," "my") at noticeably higher rates than non-depressed people — it's one of the most-replicated findings in psycholinguistics, and I include it as a feature in this project.

## What it is
The theory: depression narrows attention onto the self. That shift shows up in language as elevated use of "I," "me," "my," "mine," "myself" relative to other function words. The effect is small per-post but reliable across many posts. Pennebaker's book *The Secret Life of Pronouns* (2011) is the best general reference; Rude, Gortner & Pennebaker (2004) is the depression-specific paper.

## Why we used it in this project
It's cheap to compute (just count pronoun tokens divided by total tokens), well-motivated in the literature, and interpretable. It's the `fp_pronoun_rate` feature in `src/extract_features.py`.

## The number that matters
In raw pre-1w deltas (from the paper's delta table), first-person pronoun rate moves +0.004 for crisis, +0.005 for recovery, −0.001 for neither. The absolute differences are tiny, which is why *z-scoring* matters so much for this feature — when you express each user's pronoun rate in units of their own standard deviation, the separation between groups becomes large. (The paper's feature-importance analysis is where this shows up as a top contributor.)

## What an interviewer might push on
- **"Correlation isn't causation — high 'I' use could mean lots of things."** — Correct. We're not claiming pronoun use *causes* or *diagnoses* depression. We're saying it's a reliable correlate at the population level, and it contributes to a classifier that predicts labels above chance.
- **"Doesn't everyone on r/depression use 'I' a lot?"** — Yes, which is why raw rates don't separate groups well. The signal is in the *change* from the user's own baseline (see [z-score-normalization.md](z-score-normalization.md)). That's why per-user normalization matters here specifically.
- **"Why not Linguistic Inquiry and Word Count (LIWC)?"** — LIWC is a paid lexicon. First-person pronoun rate is the one LIWC feature I can reproduce from a public word list, so I used it directly.

## The one thing I want to remember
This feature is here because of a specific cited paper, not because I guessed. If asked, I should be able to name Pennebaker and explain the self-focus theory in one sentence.
