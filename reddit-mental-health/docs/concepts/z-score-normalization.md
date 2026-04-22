# Per-user z-score normalization

**One-sentence answer:** For each feature, I subtract *that user's* baseline mean and divide by *that user's* baseline standard deviation — so a z-score of −2 means the user's current window is two of their own standard deviations below their own baseline, not below the population.

## What it is
Standard z-scoring uses the population mean and std: `(x − μ) / σ`. Per-user z-scoring uses the user's own baseline statistics: `(x_window − μ_user) / σ_user`. The output is in units of that specific user's historical variability, which makes shifts comparable across users who have very different baselines.

## Why we used it in this project
People's writing styles differ a lot. One user naturally writes gloomy posts; another writes upbeat ones. If I compare raw sentiment, I'm mostly measuring *who the person is*, not *how they're changing*. Per-user z-scoring in `src/extract_features.py` strips out the baseline and leaves only the within-user change signal. The paper calls this "isolating within-user change from cross-user style differences."

It produces a 24-feature vector per user: 7 features × 3 pre-windows, z-scored against each user's own baseline.

## The number that matters
Z-norm-only features get **macro AUC 0.559** (RF, 5-fold CV). That's 0.06 above chance — modest but clearly non-zero, meaning within-user change *does* carry signal even after we strip out personal style. But it's much lower than raw features (AUC ~0.65), which tells a different story — see [baseline-vs-temporal.md](baseline-vs-temporal.md).

## What an interviewer might push on
- **"What if a user has only a few baseline posts? σ is unstable."** — Exactly right. Users with too little baseline data get NaN z-scores, which we median-impute inside the classifier pipeline. The paper flags this explicitly.
- **"Why not just use fixed effects / mixed models?"** — Z-scoring is basically a hand-rolled, feature-wise version of subtracting a per-user fixed effect. A mixed model would be more principled but harder to combine with RF / LR feature pipelines. This is a pragmatic choice.

## The one thing I want to remember
Z-scoring is the experimental *knob* that lets me separate "this user writes sadly" from "this user is becoming more sad." The fact that AUC drops from ~0.65 to 0.559 when I z-score isn't a failure of z-scoring — it's the measurement of how much of the signal was personal style vs. real change.
