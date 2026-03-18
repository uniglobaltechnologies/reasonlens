-- =============================================================================
-- 011: Expand user assessment context for institutional scenario assessment
-- =============================================================================

ALTER TABLE user_assessment_context
  ADD COLUMN IF NOT EXISTS institution_size TEXT,
  ADD COLUMN IF NOT EXISTS funding_model TEXT,
  ADD COLUMN IF NOT EXISTS respondent_role TEXT,
  ADD COLUMN IF NOT EXISTS respondent_institutional_visibility TEXT,
  ADD COLUMN IF NOT EXISTS digital_infrastructure_baseline TEXT;
