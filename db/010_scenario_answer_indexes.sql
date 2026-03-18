-- 010: Add missing indexes on scenario_answers for JOIN performance and upsert efficiency
-- Safe to run multiple times (IF NOT EXISTS)

-- Supports JOIN to scenario_responses in scenario-session-complete.ts
CREATE INDEX IF NOT EXISTS idx_scenario_answers_response ON scenario_answers(response_id);

-- Supports JOIN to scenario_bank in scenario-session-complete.ts
CREATE INDEX IF NOT EXISTS idx_scenario_answers_scenario ON scenario_answers(scenario_id);

-- Composite index for ON CONFLICT (session_id, scenario_id) upsert in scenario-answers.ts
-- Also covers queries filtering by session_id alone
CREATE INDEX IF NOT EXISTS idx_scenario_answers_session_scenario ON scenario_answers(session_id, scenario_id);
