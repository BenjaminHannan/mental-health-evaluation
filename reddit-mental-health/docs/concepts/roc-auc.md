# ROC-AUC

**One-sentence answer:** ROC-AUC is the probability that the model scores a randomly-chosen positive example higher than a randomly-chosen negative one — so 0.5 is a coin flip and 1.0 is perfect, and our 0.688 means the model has real signal even though the classifier's hard predictions are still noisy.

## What it is
A classifier like Random Forest outputs a probability for each class, not a label. If you sweep the decision threshold from 0 to 1, you trace out a curve of true-positive rate vs false-positive rate — that's the ROC curve. AUC is the area under it. It's threshold-independent, which matters because the "right" threshold for a mental-health screener depends on how much you care about false negatives vs false positives, and that's a policy decision, not a modeling one.

## Why we used it in this project
We report **macro AUC** — averaged across the three classes (crisis, recovery, neither) — because the dataset is imbalanced (449 neither vs 73+59 turning-point users) and plain accuracy would be dominated by predicting "neither" on everything.

## The number that matters
Kitchen-sink RF: **macro AUC 0.688, macro F1 0.396** (5-fold CV, n=581). The PELT unsupervised baseline hits only 0.153 at ±2 weeks versus a null of 0.114 — so the supervised lift is large relative to the unsupervised reference.

## What an interviewer might push on
- **"Macro F1 of 0.396 sounds bad — is this model any good?"** — F1 depends on the threshold; AUC doesn't. AUC 0.688 says the model *ranks* users correctly about 69% of the time, which is well above chance. F1 is low partly because with 449 "neither" users, the optimal hard-prediction threshold is conservative and misses recovery users. For a real screener you'd tune the threshold for recall.
- **"Why not report accuracy?"** — On a dataset that's 77% "neither," a model that always predicts neither gets 77% accuracy and is useless. AUC is imbalance-robust.
- **"Is 0.688 impressive?"** — It's modest in absolute terms. But the point of the paper isn't "we built a great classifier" — it's "linguistic change signals are detectable above chance using basic features," and AUC 0.688 with 24–127 interpretable features supports that.

## The one thing I want to remember
0.688 is a *pilot-study* signal, not a production-ready model. I should say that directly instead of defending it as if it were.
