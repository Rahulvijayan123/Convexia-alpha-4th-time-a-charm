## Changelog

### v1.6 (recall-first discovery + split recall/precision)
- **Recall/precision split**: surface **Found (unvalidated)** via `draft_assets` and **Validated (evidence-complete)** via `final_assets`.
- **Enrichment in-loop**: enrichment starts during the search loop (top-K per cycle) and uses short identifier-specific micro-queries; a nonzero enrichment guarantee is enforced when any found assets exist.
- **ChatGPT-style iterative querying**: remove static precomputed query lists for v1.6; workers plan micro-queries each cycle using only compact summaries (no giant negative lists).
- **Diagnostics**: emit a single `run_report_v1_6` JSON report (per-worker marginal gains per cycle, top query templates by yield, rejection reasons by stage).
- **Eval**: add stage-wise recall reporting and a benchmark pairing report (eval-only; no benchmark leakage into runtime).

### v1.5 (draft assets + decoupled enrichment)
- Added `draft_assets` as an evidence-anchored discovery layer (never dropped for missing enrichment fields).
- Added budgeted enrichment + late filtering/scoring for promotion to `final_assets`.

### v1.4 (identifier contract + eval diagnostics)
- **Runtime contract (final_assets)**: added evidence-anchored identifier fields:
  - `primary_identifier_raw`, `primary_identifier_canonical`, `identifier_aliases_raw`
  - `evidence_snippet`, `evidence_url`, `evidence_source_type`
- **Hard rule enforced**: `primary_identifier_raw` must appear **verbatim** in `evidence_snippet` (or in locally cached fetched document text for `evidence_url` if present). Assets that cannot be evidence-anchored are rejected.
- **No identifier loss**: canonicalization is stored as an additional field and never overwrites raw identifiers.
- **Typed mention handling**: mentions are classified (`drug_code_like`, `patent_id_like`, `trial_id_like`, `protein_gene_like`, `junk`); only `drug_code_like` and `patent_id_like` become Stage B candidates.
- **Unified canonicalization**: a single `canonicalize_identifier()` is used across runtime and eval (A–Z/0–9 only after uppercasing).
- **Evaluation diagnostics**:
  - fixed prediction identifier extraction (avoid treating Supabase row UUIDs as identifiers)
  - added exact-match + canonical-match TP/FP/FN reporting and near-miss suggestions
  - added attribution diagnostic script `eval/diagnose_pipeline_attribution.py`
- **Debug CLIs**: `python -m tools.dump_run_assets` and `python -m tools.dump_benchmark`

**Note**: v1.4 does not add new search providers and does not include any benchmark-specific prompt/runtime logic. The benchmark remains evaluation-only.


