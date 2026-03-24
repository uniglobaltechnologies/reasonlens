-- =============================================================================
-- 016: Add QS AI Capability context columns to user_assessment_context
-- =============================================================================

ALTER TABLE user_assessment_context
  ADD COLUMN IF NOT EXISTS ai_maturity_baseline TEXT,
  ADD COLUMN IF NOT EXISTS sector_focus TEXT,
  ADD COLUMN IF NOT EXISTS respondent_ai_familiarity TEXT;
