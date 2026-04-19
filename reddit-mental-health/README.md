# Longitudinal Linguistic Markers of Mental Health Deterioration

NLP study of Reddit posts from mental-health subreddits (r/depression, r/ADHD, r/PTSD, r/OCD, r/aspergers), looking for **linguistic shifts in the weeks preceding a user's crisis or recovery "turning point" post** compared to their baseline posting behavior.

Target venue: CLPsych workshop (Computational Linguistics and Clinical Psychology) + arXiv preprint.

## Research question
Can we detect statistically significant shifts in linguistic patterns in a Reddit user's posting history in the 2–4 weeks preceding a "turning point" post (defined as a post containing crisis or recovery language), compared to their baseline posting behavior?

## Pipeline

```
load_data.py       → pulls HF dataset, saves raw posts parquet
label_users.py     → groups posts by author, finds turning points, labels users
extract_features.py→ sliding-window linguistic features per user
train_model.py     → LR / RF / (BERT) classifiers, CV evaluation
visualize.py       → feature trajectories and paper figures
```

## Dataset
[`solomonk/reddit_mental_health_posts`](https://huggingface.co/datasets/solomonk/reddit_mental_health_posts) — 151k posts, fields include `author`, `body`, `title`, `created_utc`, `subreddit`. Covers r/ADHD, r/aspergers, r/depression, r/OCD, r/PTSD.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
python src/load_data.py
```

## Ethics
This project uses **public** Reddit posts from mental-health subreddits. Data is sensitive even when public:
- No user IDs or raw posts are published in paper figures — only aggregate statistics.
- No attempt is made to re-identify or contact any user.
- Keyword-based crisis labeling is an approximation, not a clinical diagnosis.
- See the Ethics section of the paper (`paper/main.tex`) for the full statement.

## Project layout

```
reddit-mental-health/
├── data/            # raw + processed data (gitignored where large)
├── notebooks/       # exploratory analysis
├── src/
│   ├── load_data.py
│   ├── label_users.py
│   ├── extract_features.py
│   ├── train_model.py
│   └── visualize.py
├── paper/main.tex
├── requirements.txt
├── .gitignore
└── README.md
```
