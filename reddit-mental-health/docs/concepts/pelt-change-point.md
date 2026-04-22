# PELT change-point detection

**One-sentence answer:** PELT ("Pruned Exact Linear Time") is an unsupervised algorithm that finds points in a time series where the statistical properties change — I used it as a baseline that doesn't know which users were in crisis, and it barely beat chance, which is actually one of the paper's most useful results.

## What it is
Given a 1-D series (e.g. weekly-mean sentiment for one user), PELT searches for a set of change-points that minimize a cost function plus a penalty on the number of change-points. The penalty prevents it from flagging every small wiggle. It runs in linear time in the length of the series, hence the name. Reference: van den Burg & Williams (2020) survey.

## Why we used it in this project
It's the natural *unsupervised* comparison for our research question. If someone says "well, you could just find change-points in sentiment without any labels," PELT is the cleanest way to test that claim. In `src/` we apply PELT with an $\ell_2$ cost to each user's weekly-mean VADER series and check whether the detected change-point falls within ±2 weeks of the actual turning point.

## The number that matters
**PELT hit@±2w = 0.153, versus a random-placement null of 0.114.** At the stricter ±1-week tolerance, PELT hits 7.6% vs null. So PELT is *slightly* above chance but not usefully so.

## What an interviewer might push on
- **"Isn't a barely-above-chance baseline a failure?"** — It's the opposite. It makes the supervised result meaningful. If PELT had been at 0.5, I'd be worried the supervised 0.688 was just picking up the same easy signal. The gap between 0.153 and 0.688 *is* the evidence that the linguistic change signal is there but subtle enough that you need labels + multiple features to extract it.
- **"Why only ℓ₂ cost on sentiment?"** — We picked the cleanest possible configuration so that criticism would be "your PELT was too simple," not "your PELT was cherry-picked." Extending PELT to multivariate series is in the future-work section.

## The one thing I want to remember
This is a *negative result that strengthens the paper*. In science, showing what doesn't work (and then doing the thing that does) is more convincing than only reporting your wins.
