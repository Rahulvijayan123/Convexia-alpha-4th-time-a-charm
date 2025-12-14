-- Minimal Supabase schema for agentic long-tail discovery runs
-- NOTE: Run this in the Supabase SQL editor.

create extension if not exists "uuid-ossp";

create table if not exists public.runs (
  id uuid primary key default uuid_generate_v4(),
  created_at timestamptz not null default now(),
  started_at timestamptz,
  finished_at timestamptz,
  status text not null default 'running',
  idempotency_key text,
  user_query text not null,
  config_version text not null,
  prompt_version text not null,
  params jsonb not null default '{}'::jsonb,
  summary jsonb not null default '{}'::jsonb
);

create index if not exists runs_created_at_idx on public.runs(created_at desc);
create unique index if not exists runs_idempotency_key_idx on public.runs(idempotency_key) where idempotency_key is not null;

create table if not exists public.cycles (
  id uuid primary key default uuid_generate_v4(),
  created_at timestamptz not null default now(),
  run_id uuid not null references public.runs(id) on delete cascade,
  round_idx int not null,
  worker_id text not null,
  phase text not null, -- recall | validate
  planned_query text,
  success boolean not null default false,
  metrics jsonb not null default '{}'::jsonb
);
create index if not exists cycles_run_round_idx on public.cycles(run_id, round_idx);

create table if not exists public.queries (
  id uuid primary key default uuid_generate_v4(),
  created_at timestamptz not null default now(),
  run_id uuid not null references public.runs(id) on delete cascade,
  cycle_id uuid references public.cycles(id) on delete set null,
  worker_id text,
  phase text not null, -- recall | validate
  query_text text not null,
  query_fingerprint text not null
);
create index if not exists queries_run_id_idx on public.queries(run_id);
create index if not exists queries_fingerprint_idx on public.queries(query_fingerprint);

create table if not exists public.results (
  id uuid primary key default uuid_generate_v4(),
  created_at timestamptz not null default now(),
  run_id uuid not null references public.runs(id) on delete cascade,
  query_id uuid not null references public.queries(id) on delete cascade,
  tool_name text not null,
  response_json jsonb not null,
  urls jsonb not null default '[]'::jsonb
);
create index if not exists results_query_id_idx on public.results(query_id);

create table if not exists public.documents (
  id uuid primary key default uuid_generate_v4(),
  created_at timestamptz not null default now(),
  url text not null,
  content_type text,
  sha256 text,
  fetched_at timestamptz,
  text_content text,
  raw_content bytea
);
create unique index if not exists documents_url_idx on public.documents(url);

create table if not exists public.mentions (
  id uuid primary key default uuid_generate_v4(),
  created_at timestamptz not null default now(),
  run_id uuid not null references public.runs(id) on delete cascade,
  query_id uuid references public.queries(id) on delete set null,
  result_id uuid references public.results(id) on delete set null,
  mention_type text not null,
  raw_text text not null,
  normalized_text text not null,
  -- v1.4: identifier canonicalization + typed mention handling (evaluation/diagnostics)
  canonical_text text,
  mention_class text,
  context text,
  source_url text,
  fingerprint text not null
);
create index if not exists mentions_run_id_idx on public.mentions(run_id);
create index if not exists mentions_fingerprint_idx on public.mentions(fingerprint);

create table if not exists public.candidates (
  id uuid primary key default uuid_generate_v4(),
  created_at timestamptz not null default now(),
  run_id uuid not null references public.runs(id) on delete cascade,
  source_mention_id uuid references public.mentions(id) on delete set null,
  candidate_type text not null,
  raw_identifier text not null,
  normalized_identifier text not null,
  -- v1.4: canonical identifier used for dedup + eval attribution
  canonical_identifier text,
  mention_class text,
  fingerprint text not null,
  status text not null default 'pending', -- pending | validated | rejected
  metadata jsonb not null default '{}'::jsonb
);
create index if not exists candidates_run_id_idx on public.candidates(run_id);
create index if not exists candidates_fingerprint_idx on public.candidates(fingerprint);

create table if not exists public.validations (
  id uuid primary key default uuid_generate_v4(),
  created_at timestamptz not null default now(),
  run_id uuid not null references public.runs(id) on delete cascade,
  candidate_id uuid not null references public.candidates(id) on delete cascade,
  status text not null, -- validated | rejected | error
  model_output jsonb,
  evidence_urls jsonb not null default '[]'::jsonb,
  error text
);
create index if not exists validations_candidate_id_idx on public.validations(candidate_id);

-- -------------------------------------------------------------------
-- v1.5: draft assets (evidence-anchored discovery, decoupled from enrichment)
-- -------------------------------------------------------------------

create table if not exists public.draft_assets (
  id uuid primary key default uuid_generate_v4(),
  created_at timestamptz not null default now(),
  run_id uuid not null references public.runs(id) on delete cascade,

  -- Identifier preservation + canonicalization (store both everywhere)
  identifier_raw text not null,
  identifier_canonical text not null,
  -- Preserve all raw formatting variants as aliases (dedup is by canonical at this layer)
  identifier_aliases_raw jsonb not null default '[]'::jsonb,

  -- Evidence anchoring (minimal acceptance requires these)
  evidence_url text not null,
  evidence_snippet text not null,
  evidence_source_type text not null, -- enum-like: patent|trial|pipeline_pdf|paper|vendor|press_release|other

  -- Provenance
  discovered_by_worker_id text,
  discovered_by_cycle_id uuid references public.cycles(id) on delete set null,
  confidence_discovery double precision,

  -- Enrichment status + optional enrichment fields
  enrichment_status text not null default 'pending', -- pending|partial|complete|failed
  sponsor text,
  target text,
  modality text,
  indication text,
  stage text,
  geography text,

  -- Evidence citations for enrichment (list of {url, snippet})
  citations jsonb not null default '[]'::jsonb,

  -- v1.5: late filtering + diagnostics (do NOT use benchmark knowledge)
  identifier_type text, -- drug_code|target_gene|modality_phrase|trial_id|company|other
  match_scores jsonb, -- {target_match_score, modality_match_score, indication_match_score, overall_match_score, evidence:{...}}
  rejection_reason text, -- identifier_type_not_drug_code|insufficient_target_evidence|insufficient_modality_evidence|insufficient_indication_evidence|incomplete_evidence_anchor|other
  extracted_context jsonb -- optional free-form context for debugging/scoring
);
create index if not exists draft_assets_run_id_idx on public.draft_assets(run_id);
create unique index if not exists draft_assets_run_canonical_idx on public.draft_assets(run_id, identifier_canonical);
-- Index on rejection_reason moved to v1.5 migrations section below to avoid errors
-- when running against older schemas that don't yet have the column.

create table if not exists public.final_assets (
  id uuid primary key default uuid_generate_v4(),
  created_at timestamptz not null default now(),
  run_id uuid not null references public.runs(id) on delete cascade,
  candidate_id uuid references public.candidates(id) on delete set null,
  drug_name_code text not null,
  -- v1.4 identifier contract (evidence-anchored)
  primary_identifier_raw text,
  primary_identifier_canonical text,
  identifier_aliases_raw jsonb not null default '[]'::jsonb,
  evidence_snippet text,
  evidence_url text,
  evidence_source_type text, -- enum-like: patent|trial|pipeline_pdf|paper|vendor|press_release|other
  -- v1.5: final assets may be partially enriched (promotion threshold, not "all fields required")
  sponsor text,
  target text,
  modality text,
  indication text,
  development_stage text,
  geography text,
  sources jsonb not null default '[]'::jsonb,
  fingerprint text not null
);
create index if not exists final_assets_run_id_idx on public.final_assets(run_id);
create index if not exists final_assets_fingerprint_idx on public.final_assets(fingerprint);

create table if not exists public.metrics (
  id uuid primary key default uuid_generate_v4(),
  created_at timestamptz not null default now(),
  run_id uuid not null references public.runs(id) on delete cascade,
  round_idx int,
  name text not null,
  value jsonb not null
);
create index if not exists metrics_run_id_idx on public.metrics(run_id);

-- -------------------------------------------------------------------
-- v1.4 migrations (safe to run multiple times)
-- NOTE: `create table if not exists` does not alter existing tables.
-- Run these ALTERs on existing deployments to add v1.4 columns.
-- -------------------------------------------------------------------

alter table public.mentions add column if not exists canonical_text text;
alter table public.mentions add column if not exists mention_class text;

alter table public.candidates add column if not exists canonical_identifier text;
alter table public.candidates add column if not exists mention_class text;

alter table public.final_assets add column if not exists primary_identifier_raw text;
alter table public.final_assets add column if not exists primary_identifier_canonical text;
alter table public.final_assets add column if not exists identifier_aliases_raw jsonb not null default '[]'::jsonb;
alter table public.final_assets add column if not exists evidence_snippet text;
alter table public.final_assets add column if not exists evidence_url text;
alter table public.final_assets add column if not exists evidence_source_type text;

-- -------------------------------------------------------------------
-- v1.5 migrations (safe to run multiple times)
-- -------------------------------------------------------------------

-- draft_assets table/columns are created above with IF NOT EXISTS.
alter table public.draft_assets add column if not exists identifier_type text;
alter table public.draft_assets add column if not exists match_scores jsonb;
alter table public.draft_assets add column if not exists rejection_reason text;
alter table public.draft_assets add column if not exists extracted_context jsonb;
-- Ensure index on rejection_reason exists after column additions (safe to run repeatedly)
create index if not exists draft_assets_rejection_reason_idx on public.draft_assets(rejection_reason);

-- Relax final_assets required fields for v1.5 promotion rule (>=4 filled fields).
alter table public.final_assets alter column sponsor drop not null;
alter table public.final_assets alter column target drop not null;
alter table public.final_assets alter column modality drop not null;
alter table public.final_assets alter column indication drop not null;
alter table public.final_assets alter column development_stage drop not null;
alter table public.final_assets alter column geography drop not null;


