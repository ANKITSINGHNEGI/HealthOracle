create table if not exists public.users (
  id uuid primary key default gen_random_uuid(),
  email text unique not null,
  created_at timestamptz default now()
);
create table if not exists public.submissions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references public.users(id) on delete cascade,
  payload jsonb not null,
  created_at timestamptz default now()
);
create table if not exists public.predictions (
  id uuid primary key default gen_random_uuid(),
  submission_id uuid references public.submissions(id) on delete cascade,
  risk jsonb not null,
  shap_contrib jsonb,
  recommendations jsonb,
  model_version text not null,
  created_at timestamptz default now()
);
create index if not exists idx_predictions_risk on public.predictions using gin (risk jsonb_path_ops);
create index if not exists idx_submissions_payload on public.submissions using gin (payload jsonb_path_ops);
