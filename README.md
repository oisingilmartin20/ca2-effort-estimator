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
- `data/tawos_sample.csv` — small demo backlog (14 sample tickets)
- `data/tawos_with_story_points.csv` — full export of tickets with positive story points
- `data/tawos_balanced_train.csv` — balanced ~20% training subset (Fibonacci labels)
- `data/tawos_balanced_train_with_zero.csv` — balanced ~20% training subset including zero-point tickets
- `scripts/analyze_tawos.py` — CLI summary stats for the full TAWOS MySQL dataset
- `scripts/export_tawos_training_data.py` — export training CSVs from MySQL
- `notebooks/tawos_dataset_analysis.ipynb` — Interactive tables and charts for TAWOS dataset analytics

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

`data/tawos_sample.csv` remains the default small demo backlog. To generate
training datasets from the full TAWOS MySQL database:

```bash
mysql tawos < TAWOS.sql
python scripts/export_tawos_training_data.py
```

This writes three files into `data/`:

| File | Contents |
| ---- | -------- |
| `tawos_with_story_points.csv` | All tickets with story points > 0 and ≤ 100 |
| `tawos_balanced_train.csv` | ~20% balanced subset mapped to Fibonacci scale (higher bracket) |
| `tawos_balanced_train_with_zero.csv` | Same balanced sampling, but includes zero-point tickets |

Story points are mapped to the Fibonacci scale (`1, 2, 3, 5, 8, 13, 21`).
Exact matches are kept; values strictly between two scale points map to the
**higher** bracket (e.g. `10 → 13`, `4 → 5`). Zero-point tickets are labelled `0`.
Exported `title` and `description` fields have TAWOS literal quote wrappers stripped.

Point the Streamlit app at an exported CSV via `.env`:

```
TAWOS_DATA_PATH=data/tawos_balanced_train.csv
```

Or drop a custom export with the schema above into `data/tawos_sample.csv` to
scale up the backlog without changing configuration.

## Dataset analytics

Explore summary statistics for the full TAWOS `Issue` table in MySQL.

**Prerequisite:** MySQL running with the `tawos` database loaded:

```bash
mysql tawos < TAWOS.sql
```

Install dependencies (if not already done), then run either the CLI or the notebook:

```bash
pip install -r requirements.txt
python scripts/analyze_tawos.py
jupyter lab notebooks/tawos_dataset_analysis.ipynb
```

The script prints ticket counts, missing-field stats, description length
summaries, and priority/story-point distributions. The notebook presents the
same metrics as tables and seaborn charts, including story point class
imbalance. Override the connection string in `.env` with `DATABASE_URL` if your
MySQL host or credentials differ (see `.env.example`).
