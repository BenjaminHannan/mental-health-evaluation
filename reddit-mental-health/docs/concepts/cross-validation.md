# Cross-validation

**One-sentence answer:** Cross-validation splits the data into k folds, trains on k−1 and tests on the held-out fold, and repeats — it gives a more stable estimate of how the model generalizes than a single train/test split, which matters a lot when the dataset is small.

## What it is
In k-fold CV, you partition the labeled users into k equal groups ("folds"). For each fold, you train the model on the other k−1 folds and evaluate on the held-out one. You then average the k scores. "Stratified" CV keeps the class proportions (crisis / recovery / neither) roughly equal across folds, so no fold accidentally ends up with zero recovery users.

## Why we used it in this project
We have 581 labeled users total (73 crisis, 59 recovery, 449 neither). A single 80/20 split would put only ~12 recovery users in the test set, and the AUC could swing several points depending on *which* 12. 5-fold stratified CV (in `src/train_model.py`) uses every user for evaluation exactly once, so the AUC estimate is much more stable. We didn't go to 10-fold because each fold would have ~6 recovery users — too few to estimate per-class metrics reliably. We didn't do LOOCV (leave-one-out) because it's expensive and has higher variance in AUC for imbalanced classification.

## The number that matters
All headline numbers in the paper — including macro AUC 0.688 and the 0.559 for z-norm-only — are 5-fold stratified CV averages, not single-split results.

## What an interviewer might push on
- **"Isn't 581 users too few for any of this to be reliable?"** — Fair. That's why I use stratified CV and report pooled out-of-fold predictions rather than a single test set. It's also why the paper is honest about this as a limitation and frames the contribution as a *pilot study* rather than a deployable model.
- **"Why not a proper held-out test set?"** — With this sample size, a held-out test set would waste labeled users I can't easily get more of. If I had 10,000 labeled users I'd absolutely hold some out.

## The one thing I want to remember
CV isn't a magic trick to make a small dataset big. It's a way to get a less noisy estimate from the small dataset you have. The dataset is still small, and I should say so.
