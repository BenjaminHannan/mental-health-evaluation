# "Baseline style dominates over temporal shift features"

**One-sentence answer:** When I compare feature groups head-to-head, classifiers trained on *how a user writes in general* score higher than classifiers trained on *how much a user's writing has changed* — which means a big chunk of the predictive signal is personal style, not deterioration, and I think that's one of the most interesting findings in the paper.

## What it is
The paper compares several feature groups (ablation table):

| Feature set | Macro AUC |
|---|---|
| Z-norm only (change signal, 24 features) | 0.559 |
| Raw + deltas (baseline style + change, 52 feat) | 0.649 |
| Raw + temporal (baseline + posting behavior, ~94 feat) | 0.676 |
| Kitchen sink (all groups, 127 feat) | 0.688 |

Z-norm-only isolates *change* from baseline. Raw + temporal uses baseline writing style + *when* people post. The raw/temporal set wins by a clear margin.

## Why we used it in this project
Running the ablation is the whole point — without it, we can't tell whether the model is predicting deterioration or just predicting "this user writes in a depressed style." The ablation forces us to answer that question.

## The number that matters
**Gap of ~0.12 AUC** between z-norm-only (0.559) and raw+temporal (0.676). The gap is the "baseline style dominates" effect — that much of the signal goes away once you strip out who the user is.

## What an interviewer might push on
- **"Doesn't this mean your model is useless for a real early-warning system?"** — It means a real system has a *cold-start problem*: when a new user joins, we have no personal baseline, so we'd fall back to style-based prediction, which is weaker and risks profiling. It doesn't make the model useless; it tells you where to focus. The z-norm-only AUC of 0.559 is still above chance, so the change signal is real, just small.
- **"Why not just use the best feature set and not talk about the ablation?"** — Because the ablation *is* the scientific contribution. "We built a classifier" isn't interesting. "We quantified how much of the signal is style vs change" is.
- **"Isn't this obvious?"** — The *direction* is unsurprising. The *magnitude* of the gap is new — this is the first time (to my knowledge) that a Reddit MH dataset has been used to quantify it.

## The one thing I want to remember
The finding I'm most proud of is not AUC 0.688 — it's the ~0.12 AUC gap that shows how much of the signal is style vs change. That's the thing I should lead with in interviews.
