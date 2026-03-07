-- Additional unique constraints needed for upsert operations

-- petri-audit-callback needs to upsert transcripts by run_id + path
CREATE UNIQUE INDEX IF NOT EXISTS idx_audit_transcripts_run_path
  ON audit_transcripts(run_id, path);

-- benchmark_runs needs unique on petri_run_id + benchmark_type for posthoc upsert
CREATE UNIQUE INDEX IF NOT EXISTS idx_benchmark_runs_petri_type
  ON benchmark_runs(petri_run_id, benchmark_type);

-- user_api_keys needs unique on user_id + provider for upsert
CREATE UNIQUE INDEX IF NOT EXISTS idx_user_api_keys_user_provider
  ON user_api_keys(user_id, provider);

-- learning_paths needs unique on user_id + framework_id for upsert
-- (already defined in schema as UNIQUE constraint, but verify)
