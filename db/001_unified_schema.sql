-- =============================================================================
-- ReasonLens Unified Database Schema
-- Merges: LearnAI Scope (15 tables) + GlassRoom Lab (11 tables)
-- Target: Azure PostgreSQL Flexible Server 16
-- =============================================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- ENUMS
-- =============================================================================

CREATE TYPE app_role AS ENUM ('admin', 'educator', 'leader', 'student', 'runner', 'viewer');
CREATE TYPE run_status AS ENUM ('queued', 'running', 'completed', 'stopped', 'failed');

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- 1. IDENTITY & AUTH (2 tables)
-- =============================================================================

CREATE TABLE profiles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  auth_provider_id TEXT UNIQUE NOT NULL,
  email TEXT NOT NULL,
  full_name TEXT,
  institution TEXT,
  region TEXT,
  sector TEXT,
  institution_type TEXT,
  comfort_level INTEGER CHECK (comfort_level BETWEEN 1 AND 5),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TRIGGER trg_profiles_updated_at
  BEFORE UPDATE ON profiles
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TABLE user_roles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  role app_role NOT NULL,
  granted_by UUID REFERENCES profiles(id) ON DELETE SET NULL,
  granted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(user_id, role)
);

CREATE INDEX idx_user_roles_user_id ON user_roles(user_id);

-- Role check function (replaces Supabase SECURITY DEFINER pattern)
CREATE OR REPLACE FUNCTION has_role(_user_id UUID, _role app_role)
RETURNS BOOLEAN AS $$
  SELECT EXISTS (
    SELECT 1 FROM user_roles WHERE user_id = _user_id AND role = _role
  );
$$ LANGUAGE sql STABLE;

-- =============================================================================
-- 2. FRAMEWORK ASSESSMENT (5 tables)
-- =============================================================================

CREATE TABLE user_goals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  goal TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(user_id, goal)
);

CREATE INDEX idx_user_goals_user_id ON user_goals(user_id);

CREATE TABLE assessment_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  framework_id TEXT NOT NULL,
  framework_name TEXT NOT NULL,
  question_id TEXT NOT NULL,
  dimension TEXT NOT NULL,
  selected_level TEXT NOT NULL CHECK (selected_level <> ''),
  completed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_assessment_results_user_id ON assessment_results(user_id);
CREATE INDEX idx_assessment_results_framework_id ON assessment_results(framework_id);

CREATE TABLE framework_progress (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  framework_id TEXT NOT NULL,
  framework_name TEXT NOT NULL,
  progress INTEGER NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
  completed_items INTEGER NOT NULL DEFAULT 0,
  total_items INTEGER NOT NULL DEFAULT 0,
  last_activity TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(user_id, framework_id)
);

CREATE INDEX idx_framework_progress_user_id ON framework_progress(user_id);

CREATE TRIGGER trg_framework_progress_updated_at
  BEFORE UPDATE ON framework_progress
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TABLE learning_paths (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  framework_id TEXT NOT NULL,
  framework_name TEXT NOT NULL,
  recommendations JSONB NOT NULL DEFAULT '[]',
  overall_progress INTEGER NOT NULL DEFAULT 0,
  dimension_gaps JSONB NOT NULL DEFAULT '[]',
  ai_recommendations JSONB NOT NULL DEFAULT '[]',
  generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_learning_paths_user_id ON learning_paths(user_id);

CREATE TRIGGER trg_learning_paths_updated_at
  BEFORE UPDATE ON learning_paths
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 3. PORTFOLIO & EVIDENCE (3 tables)
-- =============================================================================

CREATE TABLE portfolio_items (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  description TEXT,
  artifact_type TEXT NOT NULL CHECK (artifact_type IN ('document', 'link', 'reflection', 'video')),
  file_url TEXT,
  visibility TEXT NOT NULL DEFAULT 'private' CHECK (visibility IN ('public', 'private')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_portfolio_items_user_id ON portfolio_items(user_id);
CREATE INDEX idx_portfolio_items_visibility ON portfolio_items(visibility);

CREATE TRIGGER trg_portfolio_items_updated_at
  BEFORE UPDATE ON portfolio_items
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TABLE competency_tags (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  portfolio_item_id UUID NOT NULL REFERENCES portfolio_items(id) ON DELETE CASCADE,
  framework_id TEXT NOT NULL,
  framework_name TEXT NOT NULL,
  dimension TEXT NOT NULL,
  competency_level TEXT NOT NULL CHECK (competency_level <> ''),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_competency_tags_portfolio_item_id ON competency_tags(portfolio_item_id);
CREATE INDEX idx_competency_tags_framework_id ON competency_tags(framework_id);

CREATE TABLE portfolio_shares (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  portfolio_item_id UUID NOT NULL REFERENCES portfolio_items(id) ON DELETE CASCADE,
  shared_with_email TEXT,
  share_token UUID NOT NULL UNIQUE DEFAULT gen_random_uuid(),
  expires_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_portfolio_shares_share_token ON portfolio_shares(share_token);
CREATE INDEX idx_portfolio_shares_portfolio_item_id ON portfolio_shares(portfolio_item_id);

-- =============================================================================
-- 4. POLICY (1 table)
-- =============================================================================

CREATE TABLE policy_drafts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  policy_type TEXT NOT NULL,
  title TEXT NOT NULL,
  content TEXT NOT NULL DEFAULT '',
  framework_id TEXT,
  region TEXT,
  status TEXT NOT NULL DEFAULT 'draft',
  metadata JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TRIGGER trg_policy_drafts_updated_at
  BEFORE UPDATE ON policy_drafts
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 5. CHAT (2 tables)
-- =============================================================================

CREATE TABLE chat_conversations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  title TEXT,
  framework_context TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_chat_conversations_user_id ON chat_conversations(user_id);
CREATE INDEX idx_chat_conversations_updated_at ON chat_conversations(updated_at DESC);

CREATE TRIGGER trg_chat_conversations_updated_at
  BEFORE UPDATE ON chat_conversations
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TABLE chat_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES chat_conversations(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
  content TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_chat_messages_conversation_id ON chat_messages(conversation_id);
CREATE INDEX idx_chat_messages_created_at ON chat_messages(created_at);

-- =============================================================================
-- 6. BADGES & ACHIEVEMENTS (3 tables)
-- =============================================================================

CREATE TABLE badges (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT NOT NULL,
  icon TEXT NOT NULL,
  category TEXT NOT NULL CHECK (category IN ('completion', 'mastery', 'practice', 'portfolio', 'streak', 'social')),
  criteria JSONB NOT NULL,
  points INTEGER NOT NULL DEFAULT 10,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_badges_category ON badges(category);

CREATE TABLE user_badges (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  badge_id TEXT NOT NULL REFERENCES badges(id) ON DELETE CASCADE,
  earned_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  progress JSONB,
  UNIQUE(user_id, badge_id)
);

CREATE INDEX idx_user_badges_user_id ON user_badges(user_id);
CREATE INDEX idx_user_badges_badge_id ON user_badges(badge_id);

CREATE TABLE user_achievements (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  achievement_type TEXT NOT NULL,
  achievement_data JSONB,
  earned_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_user_achievements_user_id ON user_achievements(user_id);

-- =============================================================================
-- 7. AI SAFETY AUDITS (6 tables, from GlassRoom Lab)
-- =============================================================================

CREATE TABLE audit_runs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_by UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  scenario_pack TEXT NOT NULL,
  auditor_model TEXT NOT NULL,
  target_model TEXT NOT NULL,
  judge_model TEXT NOT NULL,
  max_turns INTEGER NOT NULL DEFAULT 10,
  samples_per_scenario INTEGER DEFAULT 1,
  benchmark_packs TEXT[],
  posthoc_packs TEXT[],
  cap_tokens INTEGER,
  cap_cost NUMERIC,
  cost_tokens INTEGER,
  cost_currency NUMERIC,
  status run_status NOT NULL DEFAULT 'queued',
  mode TEXT,
  simple_mode_context JSONB,
  error_message TEXT,
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE audit_transcripts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id UUID NOT NULL REFERENCES audit_runs(id) ON DELETE CASCADE,
  path TEXT NOT NULL,
  content TEXT,
  judge_scores_json JSONB,
  flags TEXT[],
  language TEXT DEFAULT 'en',
  epoch_number INTEGER,
  scenario_id TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_audit_transcripts_run_id ON audit_transcripts(run_id);

CREATE TABLE scenarios (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  pack_id TEXT NOT NULL,
  title TEXT NOT NULL,
  purpose TEXT NOT NULL,
  tests TEXT NOT NULL,
  seed_instruction TEXT NOT NULL,
  typical_flags TEXT[] NOT NULL DEFAULT '{}',
  color TEXT NOT NULL DEFAULT 'primary',
  icon TEXT NOT NULL DEFAULT 'flask-conical',
  is_default BOOLEAN NOT NULL DEFAULT false,
  owner_id UUID REFERENCES profiles(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TRIGGER trg_scenarios_updated_at
  BEFORE UPDATE ON scenarios
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TABLE audit_reports (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id UUID NOT NULL UNIQUE REFERENCES audit_runs(id) ON DELETE CASCADE,
  content_markdown TEXT,
  visuals_json JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TRIGGER trg_audit_reports_updated_at
  BEFORE UPDATE ON audit_reports
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TABLE posthoc_pack_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id UUID NOT NULL REFERENCES audit_runs(id) ON DELETE CASCADE,
  pack_id TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  metrics_json JSONB,
  evidence_json JSONB,
  error_message TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(run_id, pack_id)
);

CREATE TRIGGER trg_posthoc_pack_results_updated_at
  BEFORE UPDATE ON posthoc_pack_results
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TABLE benchmark_runs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_by UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  benchmark_type TEXT NOT NULL,
  target_model TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  config_json JSONB,
  results_json JSONB,
  error_message TEXT,
  petri_run_id UUID REFERENCES audit_runs(id) ON DELETE CASCADE,
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =============================================================================
-- 8. CONFIGURATION (3 tables, from GlassRoom Lab)
-- =============================================================================

CREATE TABLE models (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id TEXT NOT NULL,
  provider_slug TEXT NOT NULL,
  display_name TEXT NOT NULL,
  enabled BOOLEAN NOT NULL DEFAULT true,
  is_free_tier BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE user_api_keys (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  provider TEXT NOT NULL,
  encrypted_key TEXT NOT NULL,
  key_last4 TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE audit_log (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES profiles(id) ON DELETE SET NULL,
  run_id UUID REFERENCES audit_runs(id) ON DELETE SET NULL,
  action TEXT NOT NULL,
  details JSONB,
  ts TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =============================================================================
-- 9. ASSESSMENT EVIDENCE (1 new table — bridges assessment + portfolio + audits)
-- =============================================================================

CREATE TABLE assessment_evidence (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  framework_id TEXT NOT NULL,
  dimension TEXT NOT NULL,
  portfolio_item_id UUID REFERENCES portfolio_items(id) ON DELETE SET NULL,
  audit_run_id UUID REFERENCES audit_runs(id) ON DELETE SET NULL,
  evidence_type TEXT NOT NULL CHECK (evidence_type IN ('portfolio_link', 'url', 'reflection', 'upload', 'audit_result')),
  evidence_url TEXT,
  evidence_text TEXT,
  evidence_title TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_assessment_evidence_user_framework_dim
  ON assessment_evidence(user_id, framework_id, dimension);

CREATE TRIGGER trg_assessment_evidence_updated_at
  BEFORE UPDATE ON assessment_evidence
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- DONE: 27 tables, 2 enums, 2 functions, 10 triggers
-- =============================================================================
