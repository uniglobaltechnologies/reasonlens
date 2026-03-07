-- =============================================================================
-- ReasonLens Seed Data
-- =============================================================================

-- =============================================================================
-- BADGES (18 definitions from LearnAI Scope)
-- =============================================================================

INSERT INTO badges (id, name, description, icon, category, criteria, points) VALUES
-- Completion Badges
('framework_explorer', 'Framework Explorer', 'Complete your first framework assessment', 'Compass', 'completion', '{"type": "assessments_completed", "count": 1}', 10),
('multidisciplinary_master', 'Multidisciplinary Master', 'Complete assessments in 5 different frameworks', 'Award', 'completion', '{"type": "unique_frameworks", "count": 5}', 50),
('complete_journey', 'Complete Journey', 'Complete all 7 framework assessments', 'Trophy', 'completion', '{"type": "unique_frameworks", "count": 7}', 100),

-- Mastery Badges
('creator', 'Creator', 'Achieve "Create" level in any dimension', 'Sparkles', 'mastery', '{"type": "create_level", "count": 1}', 25),
('expert_practitioner', 'Expert Practitioner', 'Achieve "Create" level in 5 dimensions', 'Star', 'mastery', '{"type": "create_level", "count": 5}', 75),
('pedagogical_pioneer', 'Pedagogical Pioneer', 'Achieve "Create" level in 10 dimensions', 'Rocket', 'mastery', '{"type": "create_level", "count": 10}', 150),

-- Practice Badges
('lab_rat', 'Lab Rat', 'Complete your first practice lab', 'FlaskConical', 'practice', '{"type": "labs_completed", "count": 1}', 15),
('hands_on_learner', 'Hands-On Learner', 'Complete 5 practice labs', 'BookOpen', 'practice', '{"type": "labs_completed", "count": 5}', 50),
('scenario_master', 'Scenario Master', 'Complete 10 practice labs', 'GraduationCap', 'practice', '{"type": "labs_completed", "count": 10}', 100),

-- Portfolio Badges
('documenter', 'Documenter', 'Add 5 items to your portfolio', 'FileText', 'portfolio', '{"type": "portfolio_items", "count": 5}', 20),
('storyteller', 'Storyteller', 'Add 15 items to your portfolio', 'BookMarked', 'portfolio', '{"type": "portfolio_items", "count": 15}', 60),
('curator', 'Curator', 'Add 30 items to your portfolio', 'Library', 'portfolio', '{"type": "portfolio_items", "count": 30}', 120),

-- Streak Badges
('committed', 'Committed', 'Log in for 7 consecutive days', 'Flame', 'streak', '{"type": "login_streak", "days": 7}', 30),
('dedicated', 'Dedicated', 'Log in for 30 consecutive days', 'Zap', 'streak', '{"type": "login_streak", "days": 30}', 100),
('unstoppable', 'Unstoppable', 'Log in for 90 consecutive days', 'Crown', 'streak', '{"type": "login_streak", "days": 90}', 250),

-- Social Badges
('collaborator', 'Collaborator', 'Share 3 portfolio items', 'Users', 'social', '{"type": "shares_created", "count": 3}', 25),
('mentor', 'Mentor', 'Share 10 portfolio items', 'UserCheck', 'social', '{"type": "shares_created", "count": 10}', 75),
('community_champion', 'Community Champion', 'Have 5 public portfolio items', 'Heart', 'social', '{"type": "public_portfolio_items", "count": 5}', 50);

-- =============================================================================
-- MODELS and SCENARIOS will be imported from GlassRoom CSV exports
-- using \copy or psql COPY commands after initial setup
-- =============================================================================
