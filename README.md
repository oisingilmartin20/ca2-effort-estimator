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

- `app.py` — Streamlit UI (Jira-style backlog + ticket detail)
- `estimator.py` — LLM call + offline heuristic fallback
- `data/tawos_sample.csv` — TAWOS-style sample tickets with real story points

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
