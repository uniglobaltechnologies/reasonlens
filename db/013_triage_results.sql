-- 013: Triage results table for guided triage (THE DMI quick start)
-- Stores pillar-level signals and recommendations, NOT assessment results.
-- Deliberately separate from assessment_results to prevent downstream
-- consumers from treating triage signals as assessed maturity levels.

CREATE TABLE IF NOT EXISTS triage_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  framework_id TEXT NOT NULL DEFAULT 'maturity-the',
  respondent_role TEXT NOT NULL,
  respondent_visibility TEXT NOT NULL,
  pillar_signals JSONB NOT NULL,
  perceived_priority_dimension TEXT,
  recommended_pillar TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_triage_results_user ON triage_results(user_id);
CREATE INDEX IF NOT EXISTS idx_triage_results_framework ON triage_results(user_id, framework_id);
