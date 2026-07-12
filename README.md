# Effort Estimator - CA2 Prototype

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

The app runs without an API key - it falls back to a deterministic heuristic
estimator so the UI is fully demoable offline.

## Layout

- `app.py` - Backlog page (ticket list, estimate detail, comparison chart)
- `pages/About.py` - About page (aim, technology stack, data/training statement)
- `ui/theme.py` - Shared CSS theme, header, and header navigation links
- `estimator.py` - LLM call + offline heuristic fallback
- `data/tawos_sample.csv` - small demo backlog (14 sample tickets)
- `data/tawos_with_story_points.csv` - full export of tickets with positive story points
- `data/tawos_balanced_train.csv` - balanced ~20% training subset (Fibonacci labels)
- `data/tawos_balanced_train_with_zero.csv` - balanced ~20% training subset including zero-point tickets
- `scripts/analyze_tawos.py` - CLI summary stats for the full TAWOS MySQL dataset
- `scripts/export_tawos_training_data.py` - export training CSVs from MySQL
- `scripts/create_train_retrieval_split.py` - 80/20 retrieval corpus vs training holdout split
- `scripts/generate_embeddings.py` - embed retrieval corpus into Postgres pgvector
- `scripts/similarity_search.py` - shared pgvector nearest-neighbour lookup
- `mcp/tawos_similarity_server.py` - MCP tool for nearest-neighbour ticket search
- `docker-compose.yml` - Postgres + pgvector for vector retrieval
- `notebooks/tawos_dataset_analysis.ipynb` - Interactive tables and charts for TAWOS dataset analytics

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
summaries, priority/story-point distributions, and a per-project breakdown.
The notebook presents the same metrics as tables and seaborn charts, including
story point class imbalance and per-project story-point distribution for the
top projects. Override the connection string in `.env` with `DATABASE_URL` if
your MySQL host or credentials differ (see `.env.example`).

## Vector similarity retrieval (Postgres + MCP)

MySQL remains the TAWOS source of truth. Postgres with pgvector stores embedded
ticket descriptions for nearest-neighbour retrieval during estimation or analysis.

### Prerequisites

Requires **Python 3.10+** for the MCP server (`fastmcp`). Other scripts in this
section work on Python 3.9+.

```bash
pip install -r requirements.txt
docker compose up -d
mysql tawos < TAWOS.sql
```

Copy `.env.example` to `.env` and adjust `POSTGRES_URL` / `EMBEDDING_*` if needed.
Local embeddings (`EMBEDDING_PROVIDER=local`) work offline; set
`EMBEDDING_PROVIDER=openai` to use an OpenAI-compatible embedding endpoint.

### 1. Create the 80/20 split

Eligible tickets must have a non-empty description and a non-negative story-point label
(including zero). The retrieval corpus (80%) is embedded; the training holdout
(20%) is reserved for model training/eval and is not embedded by default.

```bash
python scripts/create_train_retrieval_split.py
```

Outputs:

| File | Contents |
| ---- | -------- |
| `data/tawos_retrieval_corpus.csv` | 80% retrieval corpus |
| `data/tawos_train_holdout.csv` | 20% training holdout |

### 2. Embed the retrieval corpus

```bash
python scripts/generate_embeddings.py
```

Use `--limit 100` for a quick smoke test. Use `--force` to re-embed rows for the
current `EMBEDDING_MODEL`.

### 3. MCP similarity search

Register the server in Cursor via `.cursor/mcp.json`, then use the
`find_similar_tickets` tool. It embeds a query description and returns the
closest tickets with `description`, `story_points`, and `similarity`.

Run manually:

```bash
python mcp/tawos_similarity_server.py
```

### 4. Streamlit estimator with vector retrieval

The backlog estimator (`streamlit run app.py`) uses pgvector neighbours as LLM
context when you click **Estimate Effort**:

1. Fetches the 10 nearest tickets from Postgres (configurable via `SIMILAR_TICKETS_LIMIT`)
2. Injects them into the LLM prompt as reference examples
3. Displays **Similar past tickets** in the estimate card

**Requirements:**

- `OPENAI_API_KEY` and `ESTIMATOR_MODEL` set in `.env` (e.g. Olmo via LM Studio)
- Postgres running with embeddings ingested (`generate_embeddings.py`)

**Validation:**

1. Run `streamlit run app.py`
2. Select a ticket and click **Estimate Effort**
3. Confirm **Similar past tickets** appears with story points and similarity scores
4. Check the Source line shows `llm:{model}+retrieval` when neighbours were found

If no neighbours are found, the estimate still runs but Source shows `+no-retrieval`
and an info message suggests running `generate_embeddings.py`.
