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

create table if not exists public.final_assets (
  id uuid primary key default uuid_generate_v4(),
  created_at timestamptz not null default now(),
  run_id uuid not null references public.runs(id) on delete cascade,
  candidate_id uuid references public.candidates(id) on delete set null,
  drug_name_code text not null,
  sponsor text not null,
  target text not null,
  modality text not null,
  indication text not null,
  development_stage text not null,
  geography text not null,
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


