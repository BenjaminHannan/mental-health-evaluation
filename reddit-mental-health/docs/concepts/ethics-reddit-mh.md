# Ethics of using public Reddit data for mental-health research

**One-sentence answer:** Reddit posts are public in the legal sense, but that doesn't mean users consented to being part of a mental-health research dataset — so the project is designed to publish only aggregate findings, never individual-level risk scores, and every design choice flows from that constraint.

## What it is
Public availability and research-ethical use are not the same thing. A person posting in r/depression at 2am is technically broadcasting to the world, but most users don't imagine their post being mined, aggregated, and analyzed alongside thousands of others. The research community (notably Benton et al. 2017, "Ethical Research Protocols for Social Media Health Research") has converged on a set of norms that balance scientific value against user dignity.

## Why we used it in this project
Several concrete constraints follow from these norms:
1. **No re-identification.** No usernames or raw post text are published in the paper's figures or tables. Only aggregate statistics and anonymized feature distributions.
2. **No individual risk scoring.** The paper never describes the model as "diagnosing" or "flagging" a user. It describes detecting population-level linguistic patterns.
3. **Keyword labels ≠ clinical labels.** A post containing the word "suicidal" is a linguistic signal, not a clinical diagnosis. The paper is explicit about this.
4. **No outreach.** I don't contact users, reply to posts, or attempt to verify labels with the authors.
5. **Dataset provenance.** The source dataset (`solomonk/reddit_mental_health_posts` on HuggingFace) is already public; I didn't scrape fresh posts, which would raise additional ethical questions.

## The number that matters
Not a number — a policy. The Ethics section of `paper/main.tex` is the reference document. It shouldn't be weakened in revision.

## What an interviewer might push on
- **"If the data is already on HuggingFace, isn't the ethical question already settled?"** — No. Downstream use of public data still has ethical implications — most notably, whether the analysis could enable harm (e.g. re-identification, targeting) if someone replicated it. My safeguards are about what I *publish*, not just what's available.
- **"Why didn't you go through IRB?"** — This is secondary-analysis of already-public data and I'm not interacting with human subjects or attempting re-identification. That puts it outside standard IRB jurisdiction in most US universities. If this became a clinical study, IRB would be mandatory — see [clinical-tool-gap.md](clinical-tool-gap.md).
- **"Could someone misuse your model?"** — In principle, yes — e.g. to flag users for targeted advertising of "wellness" products. This is why I'm publishing *findings*, not a trained model weights file.

## The one thing I want to remember
The ethics section isn't a checkbox. It's what makes this work credible at CLPsych and defensible in interviews. I should know the Benton et al. norms well enough to name them.
