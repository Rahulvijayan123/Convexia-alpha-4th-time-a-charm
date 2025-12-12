## Drug Asset Discovery (agentic long-tail web discovery)

An agentic long-tail web discovery system for drug assets.

### What it does
- **Input**: a user query like `"<target/pathway> <modality> for <indication>"`
- **Output**: a structured list of drug assets with:
  1) drug name/code, 2) sponsor, 3) target, 4) modality, 5) indication,
  6) development stage, 7) geography, 8) direct source link(s).

### Hard requirements this repo follows
- **OpenAI Responses API** using **model `gpt-5.2`** (and only `gpt-5.2`)
- **OpenAI built-in web search tool** (`web_search`) for retrieval
- **Loop Mode**: 3–4 independent workers, multi-cycle search → extract → plan next queries
- **Hard separation**:
  - Stage 1 (Recall): reckless mention harvesting (no relevance filtering)
  - Stage 2 (Validate): strict verification + structured output with citations/links
- **No giant negative lists** passed into the LLM; dedup happens in code
- **Identifier preservation**: raw mentions preserved; mapping happens later
- **Stopping**: only after multiple successive high-quality rounds produce 0 new validated assets

### Repo layout
- `src/drug_asset_discovery/`: orchestrator + workers + retrieval + extraction + validation + storage
- `prompts/v*/`: versioned prompts
- `configs/v*.json`: versioned configs
- `supabase/schema.sql`: tables for runs/cycles/queries/results/documents/mentions/candidates/validations/final_assets/metrics

---

## Local setup

### 1) Install

```bash
cd "/Users/rahulvijayan/alpha 4th time is a charm"
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### 2) Configure env
This environment blocks `.env*` files, so we ship `env.example`.

```bash
cp env.example .env
```

Set:
- `OPENAI_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

### 3) Create tables in Supabase
Run `supabase/schema.sql` in the Supabase SQL editor.

---

## Run the API

```bash
uvicorn drug_asset_discovery.api.server:app --reload
```

### POST /run
Example:

```bash
curl -X POST "http://127.0.0.1:8000/run" \
  -H "Content-Type: application/json" \
  -d '{"query":"EGFR antibody for NSCLC"}'
```

---

## Run via CLI

```bash
drug-asset-discovery run "EGFR antibody for NSCLC"
```

Replay a previous run deterministically from Supabase:

```bash
drug-asset-discovery replay <run_id>
```

---

## Deployment notes
- This is a standard FastAPI app; deploy behind your preferred API gateway.
- Use a Supabase service role key server-side only (never ship it to clients).
- Monitor `metrics` for marginal returns (new identifiers / new validated assets per round).


