# Effort Estimator — CA2 Prototype

A Jira-styled Streamlit prototype that uses an LLM to estimate story points,
confidence, and (when needed) a subtask decomposition for software tickets.
Sample tickets are drawn from the TAWOS dataset.

## Setup

```powershell
cd ca2-effort-estimator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # then fill in OPENAI_API_KEY if you have one
streamlit run app.py
```

The app runs without an API key — it falls back to a deterministic heuristic
estimator so the UI is fully demoable offline.

## Layout

- `app.py` — Backlog page (ticket list, estimate detail, comparison chart)
- `pages/About.py` — About page (aim, technology stack, data/training statement)
- `ui/theme.py` — Shared CSS theme, header, and header navigation links
- `estimator.py` — LLM call + offline heuristic fallback
- `data/tawos_sample.csv` — TAWOS-style sample tickets with real story points
- `scripts/analyze_tawos.py` — CLI summary stats for the full TAWOS MySQL dataset

## Pages

The app has two pages: **Backlog** (main estimator) and **About** (project
documentation). Navigate via the header links at the top of each page; the
Streamlit sidebar is hidden.

## Swapping the model provider

Set `OPENAI_BASE_URL` in `.env` to point at any OpenAI-compatible endpoint
(local Ollama, Together, Groq, etc.) and `ESTIMATOR_MODEL` to the model id.

## Replacing the sample data with the full TAWOS dataset

The CSV schema is:

```
issue_key, project, issue_type, title, description, actual_story_points
```

Drop a larger export with these columns into `data/tawos_sample.csv` to scale
up the backlog.

## Dataset analytics

Explore summary statistics for the full TAWOS `Issue` table in MySQL.

**Prerequisite:** MySQL running with the `tawos` database loaded:

```bash
mysql tawos < TAWOS.sql
```

Install dependencies (if not already done), then run:

```bash
pip install -r requirements.txt
python scripts/analyze_tawos.py
```

The script prints ticket counts, missing-field stats, description length
summaries, and priority/story-point distributions. Override the connection
string in `.env` with `DATABASE_URL` if your MySQL host or credentials differ
(see `.env.example`).
