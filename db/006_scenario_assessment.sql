-- =============================================================================
-- 006: Scenario-Based Assessment (SJT) Schema
-- Adds tables for situational judgement test assessment methodology
-- =============================================================================

-- New enum for scenario session status
CREATE TYPE scenario_session_status AS ENUM ('in_progress', 'completed', 'abandoned');

-- =============================================================================
-- 1. SCENARIO BANK — Scenario stems with metadata
-- =============================================================================

CREATE TABLE scenario_bank (
  scenario_id TEXT PRIMARY KEY,                          -- e.g. "CFT-ETH-AD-01"
  framework_id TEXT NOT NULL,                            -- e.g. "teacher-competency"
  dimension_id TEXT NOT NULL,                            -- e.g. "ethics"
  dimension_name TEXT NOT NULL,                          -- e.g. "Ethics of AI"
  target_boundary TEXT NOT NULL,                         -- e.g. "acquire-deepen"
  target_lower_level TEXT NOT NULL,                      -- e.g. "Acquire"
  target_upper_level TEXT NOT NULL,                      -- e.g. "Deepen"
  stem TEXT NOT NULL,                                    -- The scenario text
  question TEXT NOT NULL DEFAULT 'What would you most likely do?',
  context_tags JSONB NOT NULL DEFAULT '{}',              -- e.g. {"subject": ["universal"], "institution_level": ["any"]}
  status TEXT NOT NULL DEFAULT 'draft'
    CHECK (status IN ('draft', 'active', 'retired')),
  source_attribution JSONB,                             -- Licensing metadata
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_scenario_bank_framework ON scenario_bank(framework_id);
CREATE INDEX idx_scenario_bank_dimension ON scenario_bank(framework_id, dimension_id);
CREATE INDEX idx_scenario_bank_status ON scenario_bank(status);

CREATE TRIGGER trg_scenario_bank_updated_at
  BEFORE UPDATE ON scenario_bank
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 2. SCENARIO RESPONSES — Response options per scenario
-- =============================================================================

CREATE TABLE scenario_responses (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  scenario_id TEXT NOT NULL REFERENCES scenario_bank(scenario_id) ON DELETE CASCADE,
  response_key CHAR(1) NOT NULL CHECK (response_key IN ('A', 'B', 'C', 'D')),
  response_text TEXT NOT NULL,
  maps_to_level_name TEXT NOT NULL,                     -- e.g. "Acquire", "Deepen", "Create"
  maps_to_level_order INTEGER NOT NULL,                 -- 1, 2, or 3
  is_attractive_nuisance BOOLEAN NOT NULL DEFAULT false,
  nuisance_explanation TEXT,                             -- Why over-estimators pick this
  discriminating_presence TEXT,                          -- What this response demonstrates
  discriminating_absence TEXT,                           -- What this response lacks vs higher level
  UNIQUE(scenario_id, response_key)
);

CREATE INDEX idx_scenario_responses_scenario ON scenario_responses(scenario_id);

-- =============================================================================
-- 3. USER ASSESSMENT CONTEXT — Persona/contextual data for personalisation
-- =============================================================================

CREATE TABLE user_assessment_context (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  subject_area TEXT,
  institution_type TEXT,
  institution_level TEXT,
  region TEXT,
  current_ai_tools TEXT[] DEFAULT '{}',
  primary_frustration TEXT,
  years_of_experience TEXT,
  management_responsibility TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(user_id)
);

CREATE TRIGGER trg_user_assessment_context_updated_at
  BEFORE UPDATE ON user_assessment_context
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 4. SCENARIO SESSIONS — Assessment session tracking
-- =============================================================================

CREATE TABLE scenario_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  framework_id TEXT NOT NULL,
  context_snapshot JSONB NOT NULL DEFAULT '{}',          -- Frozen copy of user context at session start
  scenario_ids TEXT[] NOT NULL DEFAULT '{}',             -- Ordered list of scenario IDs in this session
  status scenario_session_status NOT NULL DEFAULT 'in_progress',
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_scenario_sessions_user ON scenario_sessions(user_id);
CREATE INDEX idx_scenario_sessions_framework ON scenario_sessions(framework_id);

CREATE TRIGGER trg_scenario_sessions_updated_at
  BEFORE UPDATE ON scenario_sessions
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 5. SCENARIO ANSWERS — Individual responses within a session
-- =============================================================================

CREATE TABLE scenario_answers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID NOT NULL REFERENCES scenario_sessions(id) ON DELETE CASCADE,
  scenario_id TEXT NOT NULL REFERENCES scenario_bank(scenario_id),
  response_id UUID NOT NULL REFERENCES scenario_responses(id),
  mapped_level TEXT NOT NULL,                            -- Denormalised from scenario_responses
  time_to_respond_seconds NUMERIC(6,1),                 -- Time in seconds (allows decimals)
  answered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(session_id, scenario_id)
);

CREATE INDEX idx_scenario_answers_session ON scenario_answers(session_id);

-- =============================================================================
-- 6. ITEM CALIBRATION — Phase 2 placeholder for IRT parameters
-- =============================================================================

CREATE TABLE item_calibration (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  scenario_id TEXT NOT NULL REFERENCES scenario_bank(scenario_id) ON DELETE CASCADE,
  calibration_date TIMESTAMPTZ NOT NULL DEFAULT now(),
  sample_size INTEGER NOT NULL DEFAULT 0,
  difficulty_b NUMERIC(5,3),                             -- IRT difficulty parameter
  discrimination_a NUMERIC(5,3),                         -- IRT discrimination parameter
  option_scores JSONB,                                   -- Per-option IRT scores
  dif_summary JSONB,                                     -- Differential item functioning
  model_fit JSONB,                                       -- Model fit statistics
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(scenario_id)
);

-- =============================================================================
-- 7. SCHEMA ADDITIONS — Add assessment_method to existing table
-- =============================================================================

ALTER TABLE assessment_results
  ADD COLUMN IF NOT EXISTS assessment_method TEXT NOT NULL DEFAULT 'self_report'
  CHECK (assessment_method IN ('self_report', 'scenario'));
