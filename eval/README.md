### Eval module (evaluation-only)

This `/eval` package compares **stored system outputs** against a **local benchmark CSV** and produces offline evaluation reports (no model calls).

#### Hard constraints / safety

- **The benchmark CSV is only read locally** by the evaluation process.
- **No OpenAI/LLM calls are made** by this package.
- The benchmark is **not** used to generate queries or influence runtime behavior (this is evaluation-only).

### Install

From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

### Configure Supabase (for live eval runs)

This repo’s environment may block committing `.env*` files; prefer keeping a local copy outside git.

1) Copy `eval/env.example` to a local file (example: `eval/env.local`).
2) Fill in:

- `EVAL_SUPABASE_URL` (or reuse the repo-wide `SUPABASE_URL`)
- `EVAL_SUPABASE_SERVICE_KEY` (or reuse `SUPABASE_SERVICE_ROLE_KEY`)
- `EVAL_SUPABASE_TABLE` (defaults to `final_assets`)
- `EVAL_SUPABASE_RUN_ID_COLUMN` (defaults to `run_id`)

### Provide matching rules (offline files)

Create these files (do **not** commit sensitive data; these are just match rules):

- `eval/mappings/synonyms.json` (copy from `eval/mappings/synonyms.example.json`)
- `eval/mappings/series_rules.json` (copy from `eval/mappings/series_rules.example.json`)

### Run evaluation

#### Live run (fetch stored outputs from Supabase)

Because `eval` is a shell builtin in bash/zsh, prefer **either**:

- `wd-eval ...` (recommended), or
- `python -m eval ...`, or
- `env eval ...`

Example:

```bash
wd-eval --env_file eval/env.local --run_id 123 --benchmark /abs/path/to/benchmark.csv --version v1.2
```

#### Local run (no Supabase; read predictions from a JSON file)

```bash
wd-eval --run_id local --predictions_json /abs/path/to/predictions.json --benchmark /abs/path/to/benchmark.csv --version v1.2
```

### Outputs

Reports are written to:

- `eval/reports/<version>/run_<run_id>/scorecard.json`
- `eval/reports/<version>/run_<run_id>/details.json` (TP / FP / FN with evidence URLs where available)
- `eval/reports/<version>/run_<run_id>/marginal_gains.json` (plot-ready per-cycle timeline)
- `eval/reports/<version>/run_<run_id>/report.json` (combined)

### Regression tests

Run:

```bash
python -m unittest discover -s eval/tests
```

The regression suite uses fixed fixtures and fails if **weighted recall** drops more than a threshold vs the committed baseline in:

- `eval/tests/baselines/previous.json`

To update the baseline (when intentionally improving evaluation/version):

```bash
python -m eval.tests.update_baseline
```

### GitHub version tagging workflow

Use Git tags like `v1.1`, `v1.2`, etc:

```bash
git tag -a v1.2 -m "eval: v1.2"
git push origin v1.2
```

When tagging a new version:

- run the regression suite
- update `eval/tests/baselines/previous.json` if the new behavior is desired


