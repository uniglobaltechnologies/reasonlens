-- 014: Interpretation tables for THE DMI AI-powered interpretive reports
-- Depends on: scenario_sessions (006), profiles (001)

-- Open-ended contextual responses collected after assessment completion
CREATE TABLE IF NOT EXISTS interpretation_context (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID NOT NULL REFERENCES scenario_sessions(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  trigger_response TEXT,
  previous_attempts TEXT,
  constraints TEXT[] DEFAULT '{}',
  constraints_detail TEXT,
  success_definition TEXT,
  additional_context TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(session_id)
);

CREATE INDEX IF NOT EXISTS idx_interpretation_context_session
  ON interpretation_context(session_id);

-- Stored generated interpretation sections
CREATE TABLE IF NOT EXISTS interpretive_reports (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID NOT NULL REFERENCES scenario_sessions(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  executive_summary TEXT NOT NULL,
  pillar_tl TEXT NOT NULL,
  pillar_re TEXT NOT NULL,
  pillar_ps TEXT NOT NULL,
  pillar_pg TEXT NOT NULL,
  recommendations TEXT NOT NULL,
  generation_time_ms INTEGER,
  methodology_version TEXT NOT NULL DEFAULT '1.0',
  model_used TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(session_id)
);

CREATE INDEX IF NOT EXISTS idx_interpretive_reports_session
  ON interpretive_reports(session_id);
CREATE INDEX IF NOT EXISTS idx_interpretive_reports_user
  ON interpretive_reports(user_id);
