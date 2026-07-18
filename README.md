# Effort Estimator - CA2 Prototype

A Jira-styled Streamlit prototype that uses an LLM to estimate story points,
confidence, and (when needed) a subtask decomposition for software tickets.
Users create their own backlog tickets and run AI-assisted estimation on them.

## Setup

```powershell
cd ca2-effort-estimator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # defaults to local Olmo via LM Studio
streamlit run app.py
```

Copy `.env.example` to `.env`. The recommended setup is local LM Studio with
`ESTIMATOR_MODEL=olmo-3-7b-instruct` (see `.env.example`). Cloud Groq endpoints
are rejected by the estimator so evaluation stays offline-capable.

## Layout

- `app.py` - Backlog page (ticket dropdown, estimate detail)
- `pages/Ticket.py` - Ticket creation form (Title, Project, Task Type, Description)
- `pages/About.py` - About page (aim, technology stack, data/training statement)
- `ui/theme.py` - Shared CSS theme, header, and header navigation links
- `ui/components.py` - Shared UI helpers (pills, estimate card rendering)
- `tickets/store.py` - Load/save user tickets to `data/user_tickets.json`
- `estimator.py` - LLM call + offline heuristic fallback
- `data/user_tickets.json` - user-created backlog (starts empty)
- `data/tawos_sample.csv` - legacy sample data (used by scripts/embeddings, not the app backlog)
- `data/tawos_with_story_points.csv` - full export of tickets with positive story points
- `data/tawos_balanced_train.csv` - balanced ~20% training subset (Fibonacci labels)
- `data/tawos_balanced_train_with_zero.csv` - balanced ~20% training subset including zero-point tickets
- `data/tawos_retrieval_corpus.csv` - 80% retrieval corpus (embedded into Postgres)
- `data/tawos_train_holdout.csv` - 20% holdout for fair RAG evaluation (not embedded)
- `scripts/analyze_tawos.py` - CLI summary stats for the full TAWOS MySQL dataset
- `scripts/export_tawos_training_data.py` - export training CSVs from MySQL
- `scripts/create_train_retrieval_split.py` - 80/20 retrieval corpus vs training holdout split
- `scripts/generate_embeddings.py` - embed retrieval corpus into Postgres pgvector
- `scripts/similarity_search.py` - shared pgvector nearest-neighbour lookup
- `mcp/tawos_similarity_server.py` - MCP tool for nearest-neighbour ticket search
- `docker-compose.yml` - Postgres + pgvector for vector retrieval
- `notebooks/tawos_dataset_analysis.ipynb` - Interactive tables and charts for TAWOS dataset analytics

## Pages

The app has three pages: **Ticket** (create backlog items), **Backlog** (select a
ticket and run the estimator), and **About** (project documentation). Navigate
via the header links at the top of each page; the Streamlit sidebar is hidden.

### Workflow

1. Open the **Ticket** tab and fill in Title, Project Name, Task Type (Story, Bug, Task, Epic, Improvement, and other common TAWOS types), and Description.
2. Submit to add the ticket to `data/user_tickets.json` (persists across refreshes).
3. Switch to **Backlog**, pick the ticket from the dropdown, and click **Estimate Effort**.

## Architecture

### High-level workflow

At a high level, the user interacts with the **UI** (Streamlit), which delegates
estimation to the **RAG** pipeline. RAG queries the vector **DB** for similar TAWOS
tickets, computes story points when neighbours exist, and calls the **LLM** for
explanation. The detailed file-level sequence follows below.

```mermaid
sequenceDiagram
    actor User
    participant UI
    participant RAG
    participant DB
    participant LLM

    User->>UI: Create ticket
    UI->>UI: Save to backlog

    User->>UI: Select ticket and request estimate
    UI->>RAG: Send ticket details

    RAG->>DB: Search similar past tickets
    DB-->>RAG: Nearest neighbours with story points

    alt Similar tickets found
        RAG->>RAG: Compute story points from neighbours
        RAG->>LLM: Request explanation and confidence
        LLM-->>RAG: Reasoning and optional subtasks
    else No similar tickets
        RAG->>LLM: Estimate from ticket details only
        LLM-->>RAG: Story points, reasoning, low confidence
    end

    RAG-->>UI: Estimate result
    UI-->>User: Show story points, confidence, and reasoning
```

### Estimation pipeline

The sequence diagrams above show message order; this pipeline view shows **stages,
data stores, and the RAG vs fallback fork** from offline preparation through to the
estimate shown in the UI.

```mermaid
flowchart TB
    subgraph OFFLINE["Offline data preparation"]
        direction TB
        MYSQL[(MySQL TAWOS)]
        EXPORT["export_tawos_training_data.py"]
        SPLIT["create_train_retrieval_split.py"]
        CORPUS["tawos_retrieval_corpus.csv<br/>(80%)"]
        HOLDOUT["tawos_train_holdout.csv<br/>(20%, not embedded)"]
        EMBED_BATCH["generate_embeddings.py"]
        PG[(Postgres pgvector<br/>ticket_embeddings)]

        MYSQL --> EXPORT
        EXPORT --> SPLIT
        SPLIT --> CORPUS
        SPLIT --> HOLDOUT
        CORPUS --> EMBED_BATCH
        EMBED_BATCH --> PG
    end

    subgraph ONLINE["Online estimation (Backlog)"]
        direction TB
        USER([User])
        TICKET["Ticket page<br/>user_tickets.json"]
        BACKLOG["Backlog page<br/>app.py"]
        EST["estimator.py"]
        QUERY_EMBED["Embed query description<br/>embedding.py"]
        SEARCH["find_similar_tickets<br/>similarity_search.py"]
        DECIDE{Neighbours<br/>found?}
        RAG_SP["compute_rag_story_points<br/>weighted avg → Fibonacci"]
        LLM_EXPLAIN["LLM: explain fixed SP<br/>confidence, subtasks"]
        LLM_FULL["LLM: estimate SP + reasoning<br/>low confidence cap"]
        RESULT["Estimate card<br/>SP, confidence, reasoning,<br/>similar tickets, source"]
        SESSION[(Session state<br/>not persisted)]

        USER -->|Create ticket| TICKET
        USER -->|Select + Estimate| BACKLOG
        TICKET --> BACKLOG
        BACKLOG --> EST
        EST --> QUERY_EMBED
        QUERY_EMBED --> SEARCH
        PG -.->|cosine NN lookup| SEARCH
        SEARCH --> DECIDE

        DECIDE -->|Yes| RAG_SP
        RAG_SP --> LLM_EXPLAIN
        LLM_EXPLAIN --> RESULT

        DECIDE -->|No| LLM_FULL
        LLM_FULL --> RESULT

        RESULT --> SESSION
        SESSION --> USER
    end

    OFFLINE -.->|enables retrieval| ONLINE
```

### Estimation sequence (detailed)

When a user clicks **Estimate Effort** on the Backlog page, the app runs a RAG-first
estimation pipeline: embed the ticket description, retrieve similar TAWOS tickets from
Postgres, compute story points deterministically, then ask the LLM to explain the result.

Story points are **not** LLM-generated when retrieval succeeds — they are computed in
`compute_rag_story_points()` from neighbour similarities and snapped to the Fibonacci
scale. The LLM explains the fixed estimate, sets confidence (capped by RAG confidence),
and optionally decomposes into subtasks. Estimates are stored in session state only
(not persisted to disk).

```mermaid
sequenceDiagram
    actor User
    participant Backlog as Streamlit Backlog<br/>app.py
    participant Estimator as estimator.py
    participant Search as similarity_search.py
    participant Embed as embedding.py
    participant PG as Postgres pgvector
    participant LLM as OpenAI-compatible API

    User->>Backlog: Select ticket and click Estimate Effort
    Backlog->>Backlog: Validate OPENAI_API_KEY and ESTIMATOR_MODEL
    Backlog->>Estimator: estimate_ticket(ticket)

    Estimator->>Search: find_similar_tickets(description, limit)
    Search->>Embed: embed_query(description)
    Embed-->>Search: query vector
    Search->>PG: cosine nearest-neighbour query
    PG-->>Search: similar tickets (key, SP, similarity)
    Search-->>Estimator: list of SimilarTicket

    alt Neighbours found (RAG path)
        Estimator->>Estimator: compute_rag_story_points()<br/>weighted avg to Fibonacci SP
        Estimator->>LLM: chat completion (fixed SP, neighbours as context)
        LLM-->>Estimator: JSON: confidence, reasoning, subtasks
        Estimator-->>Backlog: Estimate (source: rag+llm:model)
    else No neighbours (fallback)
        Estimator->>LLM: chat completion (no retrieval context)
        LLM-->>Estimator: JSON: story_points, confidence, reasoning
        Estimator-->>Backlog: Estimate (source: llm:model+no-retrieval)
    end

    Backlog->>Backlog: Store in session_state and render estimate card
    Backlog-->>User: Story points, confidence, similar tickets, reasoning
```

**RAG path:** neighbours found → weighted average of neighbour story points snapped to
Fibonacci → LLM explains with fixed SP (`rag+llm:{model}`).

**Fallback:** no neighbours (empty description, Postgres down, or no embeddings) →
LLM estimates SP from ticket details only, with capped confidence and a UI warning
(`llm:{model}+no-retrieval`).

## Swapping the model provider

Course evaluations use **local Olmo via LM Studio**:

```bash
OPENAI_API_KEY=lm-studio
OPENAI_BASE_URL=http://127.0.0.1:1234/v1
ESTIMATOR_MODEL=olmo-3-7b-instruct
```

`estimator.py` rejects `api.groq.com` so RQ runs cannot silently hit a cloud
provider. Other OpenAI-compatible local endpoints (e.g. Ollama) can be used by
changing `OPENAI_BASE_URL` and `ESTIMATOR_MODEL`.

## TAWOS dataset (scripts and retrieval only)

The Streamlit app no longer loads a prebuilt CSV backlog. TAWOS data files remain
for dataset analytics, training exports, and vector similarity retrieval during
estimation.

```bash
mysql tawos < TAWOS.sql
python scripts/export_tawos_training_data.py
```

This writes three files into `data/`:

| File                                 | Contents                                                        |
| ------------------------------------ | --------------------------------------------------------------- |
| `tawos_with_story_points.csv`        | All tickets with story points > 0 and ≤ 100                     |
| `tawos_balanced_train.csv`           | ~20% balanced subset mapped to Fibonacci scale (higher bracket) |
| `tawos_balanced_train_with_zero.csv` | Same balanced sampling, but includes zero-point tickets         |

Story points are mapped to the Fibonacci scale (`1, 2, 3, 5, 8, 13, 21`).
Exact matches are kept; values strictly between two scale points map to the
**higher** bracket (e.g. `10 → 13`, `4 → 5`). Zero-point tickets are labelled `0`.
Exported `title` and `description` fields have TAWOS literal quote wrappers stripped.

Point exported CSVs at scripts or embeddings workflows as needed. The app backlog
is managed entirely through the Ticket page and `data/user_tickets.json`.

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

| File                              | Contents             |
| --------------------------------- | -------------------- |
| `data/tawos_retrieval_corpus.csv` | 80% retrieval corpus |
| `data/tawos_train_holdout.csv`    | 20% training holdout |

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

The backlog estimator (`streamlit run app.py`) uses a **RAG-first** flow when you
click **Estimate Effort**:

1. Embeds the ticket description and fetches the nearest neighbours from Postgres
   (configurable via `SIMILAR_TICKETS_LIMIT`)
2. **Computes story points** from a similarity-weighted average of neighbour SPs,
   snapped to the Fibonacci scale
3. Sends neighbours + fixed SP to the LLM for **reasoning**, confidence, and
   optional subtask decomposition
4. Displays **Similar past tickets** and a RAG baseline line in the estimate card

If no neighbours are found, the LLM estimates story points from ticket details
only, with **low confidence** and an explicit warning.

**Requirements:**

- `OPENAI_API_KEY` and `ESTIMATOR_MODEL` set in `.env` (e.g. Olmo via LM Studio)
- Postgres running with embeddings ingested (`generate_embeddings.py`)

**Validation:**

1. Run `streamlit run app.py`
2. Create a ticket and click **Estimate Effort**
3. Confirm **Similar past tickets** appears and Source shows `rag+llm:{model}`
4. Confirm the estimate card shows the RAG baseline line (weighted avg → SP)
5. With Postgres down or empty description, confirm low-confidence fallback
   and warning when Source shows `+no-retrieval`


## RQ1 – Estimate closeness to actual story points (evaluation/rq1_analysis.py)

Evaluates how closely the **RAG estimator** (local `olmo-3-7b-instruct`) matches
TAWOS story points on the **training holdout** sample
(`evaluation/results_rag.csv`, stratified from `data/tawos_train_holdout.csv`).

Earlier Groq / prompt-only runs are not comparable and are discarded as baselines.

### What it computes
MAE (Mean Absolute Error) — average point gap between prediction and actual  
RMSE (Root Mean Squared Error) — penalises large misses more heavily  
MMRE (Mean Magnitude of Relative Error) — average error as a proportion of the actual value  
PRED(25) / PRED(50) = Proportion of predictions within 25% / 50% of the actual value  
Exact match rate = Proportion where prediction equals actual exactly  
Spearman ρ - Rank correlation between predictions and actuals (with p-value)  

It also prints the N worst misses (configurable, default 10) sorted by absolute error.

### Fibonacci validation
Before computing metrics, the script checks that every predicted value is a valid Fibonacci scale point. Off-scale predictions are flagged with a warning listing the affected tickets — they are still included in the metrics, so any drift will show up in the numbers.

### Usage

```bash
python evaluation/rq1_analysis.py --results evaluation/results_rag.csv --worst-n 10
```

Both arguments are optional and default to the values above.

## RQ2 – RAG-on vs RAG-off and Justification Faithfulness

Evaluates the **primary RAG estimator** on holdout tickets (retrieval from the
embedded corpus), compares it to a forced **LLM-only** ablation on the same
sample, and scores how faithfully the LLM justification cites neighbours.

### Prerequisites

- Postgres up with embeddings loaded (`docker compose up -d`, then
  `python scripts/generate_embeddings.py`)
- Local LM Studio with `olmo-3-7b-instruct` and `.env` as above
  (`OPENAI_BASE_URL` must not point at Groq)

### 1. Batch runs (same stratified holdout sample)

```bash
# RAG on (primary) — default --data is data/tawos_train_holdout.csv
python evaluation/run_estimator.py --out evaluation/results_rag.csv

# RAG off (fallback baseline) — same tickets, no retrieval
python evaluation/run_estimator.py --no-rag --out evaluation/results_norag.csv
```

Use the same `--per-class` and `--seed` for both runs so `issue_key`s align.
Confirm every `source` value contains `olmo-3-7b-instruct`.

### 2. Ablation metrics

```bash
python evaluation/rq2_rag_ablation.py \
  --rag evaluation/results_rag.csv \
  --norag evaluation/results_norag.csv
```

Prints side-by-side RQ1-style error metrics, paired win rate (RAG closer vs
no-RAG), retrieval stats (neighbour coverage, similarities, SP spread,
same-project-key rate), and MAE by RAG confidence band.

### 3. Justification review

```bash
python evaluation/rq2_prepare_justification.py --results evaluation/results_rag.csv
# Open evaluation/justification_review.csv and score each rubric column 0 or 1:
#   grounded, faithful_to_rag, comparative, no_hallucination, useful
python evaluation/rq2_justification.py --review evaluation/justification_review.csv
```

Automatic checks (no labels needed): cites at least one retrieved issue key,
does not invent keys, and does not contradict the fixed story-point estimate.
Human rates are reported once the rubric columns are filled.

## RQ3 – Complexity detection (evaluation/rq3_complexity.py)

Treats `actual_story_points >= 8` as ground-truth “should decompose”, and
compares that to the estimator’s `complex_flag`. Also reports the gap between
subtask point sums and the top-level estimate.

```bash
python evaluation/rq3_complexity.py --results evaluation/results_rag.csv
```

## RQ4 – Subtask hallucination rate (evaluation/rq4_*.py)

1. Expand complex-ticket subtasks into a labeling sheet:

```bash
python evaluation/rq4_prepare_review.py \
  --results evaluation/results_rag.csv \
  --out evaluation/hallucination_review_rag.csv
```

2. Label each row `grounded` / `inferred` / `hallucinated`
   (optionally assisted by local Olmo via `evaluation/label_hallucinations_olmo.py`),
   save as `evaluation/hallucination_review_rag_labeled.csv`.

3. Crunch rates:

```bash
python evaluation/rq4_hallucination.py \
  --review evaluation/hallucination_review_rag_labeled.csv
```

### Evaluation conventions

| Setting | Value |
| --- | --- |
| LLM | local `olmo-3-7b-instruct` via LM Studio |
| Sample | stratified holdout (`tawos_train_holdout.csv`), zero overlap with embedded corpus |
| Embeddings | local `all-MiniLM-L6-v2`, Postgres `vector(384)` |
| Primary results | `evaluation/results_rag.csv` (also copied to `results.csv`) |
