-- =============================================================================
-- 009: Seed THE Digital Maturity Index scenarios
-- Generated from data/the-dmi/scenarios.json
-- 20 child dimensions x 3 boundaries x 2 scenarios = 120 scenarios
-- =============================================================================

BEGIN;

-- Retire legacy THE scenarios that are no longer in the active production bank.
UPDATE scenario_bank
SET status = 'retired', updated_at = now()
WHERE framework_id = 'maturity-the'
  AND scenario_id NOT IN ('THE-TLS-IN-01', 'THE-TLS-IN-02', 'THE-TLS-NI-01', 'THE-TLS-NI-02', 'THE-TLS-IO-01', 'THE-TLS-IO-02', 'THE-TLP-IN-01', 'THE-TLP-IN-02', 'THE-TLP-NI-01', 'THE-TLP-NI-02', 'THE-TLP-IO-01', 'THE-TLP-IO-02', 'THE-TLT-IN-01', 'THE-TLT-IN-02', 'THE-TLT-NI-01', 'THE-TLT-NI-02', 'THE-TLT-IO-01', 'THE-TLT-IO-02', 'THE-TLD-IN-01', 'THE-TLD-IN-02', 'THE-TLD-NI-01', 'THE-TLD-NI-02', 'THE-TLD-IO-01', 'THE-TLD-IO-02', 'THE-TLU-IN-01', 'THE-TLU-IN-02', 'THE-TLU-NI-01', 'THE-TLU-NI-02', 'THE-TLU-IO-01', 'THE-TLU-IO-02', 'THE-RES-IN-01', 'THE-RES-IN-02', 'THE-RES-NI-01', 'THE-RES-NI-02', 'THE-RES-IO-01', 'THE-RES-IO-02', 'THE-REP-IN-01', 'THE-REP-IN-02', 'THE-REP-NI-01', 'THE-REP-NI-02', 'THE-REP-IO-01', 'THE-REP-IO-02', 'THE-RET-IN-01', 'THE-RET-IN-02', 'THE-RET-NI-01', 'THE-RET-NI-02', 'THE-RET-IO-01', 'THE-RET-IO-02', 'THE-RED-IN-01', 'THE-RED-IN-02', 'THE-RED-NI-01', 'THE-RED-NI-02', 'THE-RED-IO-01', 'THE-RED-IO-02', 'THE-REU-IN-01', 'THE-REU-IN-02', 'THE-REU-NI-01', 'THE-REU-NI-02', 'THE-REU-IO-01', 'THE-REU-IO-02', 'THE-PSS-IN-01', 'THE-PSS-IN-02', 'THE-PSS-NI-01', 'THE-PSS-NI-02', 'THE-PSS-IO-01', 'THE-PSS-IO-02', 'THE-PSP-IN-01', 'THE-PSP-IN-02', 'THE-PSP-NI-01', 'THE-PSP-NI-02', 'THE-PSP-IO-01', 'THE-PSP-IO-02', 'THE-PST-IN-01', 'THE-PST-IN-02', 'THE-PST-NI-01', 'THE-PST-NI-02', 'THE-PST-IO-01', 'THE-PST-IO-02', 'THE-PSD-IN-01', 'THE-PSD-IN-02', 'THE-PSD-NI-01', 'THE-PSD-NI-02', 'THE-PSD-IO-01', 'THE-PSD-IO-02', 'THE-PSU-IN-01', 'THE-PSU-IN-02', 'THE-PSU-NI-01', 'THE-PSU-NI-02', 'THE-PSU-IO-01', 'THE-PSU-IO-02', 'THE-PGS-IN-01', 'THE-PGS-IN-02', 'THE-PGS-NI-01', 'THE-PGS-NI-02', 'THE-PGS-IO-01', 'THE-PGS-IO-02', 'THE-PGP-IN-01', 'THE-PGP-IN-02', 'THE-PGP-NI-01', 'THE-PGP-NI-02', 'THE-PGP-IO-01', 'THE-PGP-IO-02', 'THE-PGT-IN-01', 'THE-PGT-IN-02', 'THE-PGT-NI-01', 'THE-PGT-NI-02', 'THE-PGT-IO-01', 'THE-PGT-IO-02', 'THE-PGD-IN-01', 'THE-PGD-IN-02', 'THE-PGD-NI-01', 'THE-PGD-NI-02', 'THE-PGD-IO-01', 'THE-PGD-IO-02', 'THE-PGU-IN-01', 'THE-PGU-IN-02', 'THE-PGU-NI-01', 'THE-PGU-NI-02', 'THE-PGU-IO-01', 'THE-PGU-IO-02');

-- THE-TLS-IN-01 :: Teaching & Learning: Strategy :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLS-IN-01', 'maturity-the', 'the-tl-strategy', 'Teaching & Learning: Strategy', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university has been responding to digital demands in teaching and learning on a case-by-case basis. Several departments have adopted different tools independently. The senior leadership team has recently discussed the need for a more coordinated approach. A deputy vice-chancellor asks you: ''Where are we on digital strategy for teaching and learning?''', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-IN-01', 'A', 'We have a clear digital strategy for teaching and learning that was approved last year and is being implemented across all faculties', 'Intentional', 2, false, NULL, 'This would indicate Intentional if true, but the scenario describes no approved strategy', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-IN-01', 'B', 'We recognise the need and are actively developing a digital strategy for teaching and learning with identified priorities and a governance proposal', 'Intentional', 2, false, NULL, 'Active development of a purposeful strategy with governance indicates transition toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-IN-01', 'C', 'We''ve been meaning to write a strategy but haven''t found the time. Meanwhile, departments are managing things in their own way', 'Incidental', 1, false, NULL, 'Acknowledged need without action and devolved ad-hoc activity is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-IN-01', 'D', 'Our institutional strategic plan mentions digital transformation and we reference that when departments ask for guidance', 'Incidental', 1, true, 'A passing mention in a broader strategy without specific objectives, owners, or resources for teaching and learning is not a purposeful strategy. This is the ''we have a plan'' attractive nuisance', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLS-IN-02 :: Teaching & Learning: Strategy :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLS-IN-02', 'maturity-the', 'the-tl-strategy', 'Teaching & Learning: Strategy', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university''s board has asked for an update on how digital technology supports teaching and learning. You discover that while several successful digital initiatives exist across the institution, they were each initiated by individual champions with no central coordination. The board wants to know what the institutional approach is.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-IN-02', 'A', 'We have a coordinated institutional approach with a strategy document, dedicated budget, and a committee overseeing digital in teaching and learning', 'Intentional', 2, false, NULL, 'This describes Intentional with formal strategy, resources, and governance', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-IN-02', 'B', 'We have some excellent initiatives and we''re now developing a formal strategy to bring them together under a coherent plan', 'Intentional', 2, false, NULL, 'Transitioning from ad-hoc to purposeful planning indicates movement toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-IN-02', 'C', 'We have a lot of innovative activity happening organically. Our approach is to let a thousand flowers bloom and learn from what works', 'Incidental', 1, true, 'Framing lack of strategy as deliberate emergent innovation is a common attractive nuisance. Organic activity without coordination is Incidental', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-IN-02', 'D', 'We don''t really have an institutional approach yet. Individual departments have done their own thing based on local needs', 'Incidental', 1, false, NULL, 'Honest acknowledgement of no institutional approach is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLS-NI-01 :: Teaching & Learning: Strategy :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLS-NI-01', 'maturity-the', 'the-tl-strategy', 'Teaching & Learning: Strategy', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university approved a digital teaching and learning strategy 18 months ago. The strategy has clear objectives and a steering group meets quarterly. However, implementation varies dramatically across faculties. Two faculties are well advanced, three have barely started, and the remaining faculties fall somewhere in between. A new PVC asks how well the strategy is being implemented.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-NI-01', 'A', 'The strategy is fully embedded. All faculties have adopted it and are implementing it consistently with local adaptation', 'Integrated', 3, false, NULL, 'This would indicate Integrated if true, but the scenario contradicts this', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-NI-01', 'B', 'We have strong pockets of implementation and we''re working to bring all faculties up to the standard of our leading areas', 'Intentional', 2, true, '''Strong pockets'' with inconsistent implementation across the institution is characteristic of Intentional. The attractive nuisance is that activity in multiple locations feels like integration', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-NI-01', 'C', 'Implementation is uneven. We have the strategy but we haven''t yet achieved consistent cross-institutional adoption with proper governance and accountability', 'Intentional', 2, false, NULL, 'Honest assessment of uneven implementation describes Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-NI-01', 'D', 'All faculties have operational plans that reference the institutional strategy, with locally adapted targets, and we report on progress to the board termly', 'Integrated', 3, false, NULL, 'Faculty-level plans aligned to institutional strategy with regular reporting indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLS-NI-02 :: Teaching & Learning: Strategy :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLS-NI-02', 'maturity-the', 'the-tl-strategy', 'Teaching & Learning: Strategy', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university has invested in a central digital teaching and learning team of six staff who develop and support digital initiatives. The team runs projects across faculties, but each project requires negotiation with individual faculty leaders for access and cooperation. The team reports to the PVC Education but has no formal authority over faculty-level decisions.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-NI-02', 'A', 'Having a dedicated central team with a clear reporting line shows we have an integrated approach to digital teaching and learning', 'Integrated', 3, true, 'A central team without governance authority over faculties and without faculty-level adoption of institutional strategy operates in an Intentional model. The attractive nuisance is equating central team existence with integration', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-NI-02', 'B', 'We have purposeful investment in digital teaching and learning but the central team operates somewhat independently from faculty planning. We need stronger governance to achieve integration', 'Intentional', 2, false, NULL, 'Purposeful investment without cross-institutional governance is accurately Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-NI-02', 'C', 'The central team''s work is governed by a cross-institutional board with faculty representation, and faculty plans are required to align with the digital strategy', 'Integrated', 3, false, NULL, 'If this were true it would indicate Integrated, but the scenario describes a team negotiating cooperation', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-NI-02', 'D', 'Our digital teaching and learning activity is still at an early stage with no real coordination between the centre and faculties', 'Incidental', 1, false, NULL, 'This understates the scenario, which does describe purposeful investment', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLS-IO-01 :: Teaching & Learning: Strategy :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLS-IO-01', 'maturity-the', 'the-tl-strategy', 'Teaching & Learning: Strategy', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has a well-functioning digital teaching and learning strategy that is implemented consistently across all faculties. Governance is strong, KPIs are reported regularly, and the institution has achieved solid results. The PVC Education wants to know whether the institution can now be considered sector-leading.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"research-intensive","region":"UK"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-IO-01', 'A', 'We regularly benchmark against Russell Group peers and ensure our approach matches best practice. We''ve adopted several approaches we learned from peer institutions', 'Integrated', 3, true, 'Benchmarking by adopting approaches from others is characteristic of Integrated. The attractive nuisance is that benchmarking activity feels like sector leadership, but copying good practice is not the same as setting it', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-IO-01', 'B', 'We review our strategy annually based on data, run a horizon scanning function, publish our approaches, and other institutions regularly visit to learn from us', 'Optimised', 4, false, NULL, 'Evidence-based annual review, horizon scanning, publication, and peer recognition indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-IO-01', 'C', 'Our implementation is strong and consistent but we recognise we''re implementing established good practice rather than innovating ahead of the sector', 'Integrated', 3, false, NULL, 'Honest assessment of implementing good practice without leading innovation is Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-IO-01', 'D', 'We''ve won a national award for one of our digital teaching and learning initiatives and were featured in a trade publication', 'Integrated', 3, true, 'A single award for a specific initiative does not indicate systematic sector leadership. The attractive nuisance is equating one recognition with Optimised maturity', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLS-IO-02 :: Teaching & Learning: Strategy :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLS-IO-02', 'maturity-the', 'the-tl-strategy', 'Teaching & Learning: Strategy', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has strong digital teaching and learning governance and consistent implementation. The sector is now grappling with a new technological development that could significantly impact teaching and learning. Several peer institutions are starting to respond.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-IO-02', 'A', 'We formed an expert panel six months ago to assess the implications and have already piloted approaches. We published a briefing paper that three peer institutions have since adopted', 'Optimised', 4, false, NULL, 'Proactive assessment, early piloting, and sector contribution through publication indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-IO-02', 'B', 'We''re watching what leading institutions are doing and plan to adopt best practice once it becomes clearer what works', 'Integrated', 3, true, 'Waiting to adopt others'' practices is reactive benchmarking, characteristic of Integrated. The attractive nuisance is that this feels prudent and strategic', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-IO-02', 'C', 'We''ve established a task force to develop our response and are developing a pilot programme informed by sector guidance', 'Integrated', 3, false, NULL, 'Developing a response informed by others indicates Integrated responding effectively, not Optimised leading', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLS-IO-02', 'D', 'We''ve been preparing for this for over a year through our horizon scanning process and have an institutional position and implementation plan ready', 'Optimised', 4, false, NULL, 'Anticipating the development through horizon scanning indicates Optimised proactive strategic maturity', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLP-IN-01 :: Teaching & Learning: People & Culture :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLP-IN-01', 'maturity-the', 'the-tl-people', 'Teaching & Learning: People & Culture', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university is assessing digital skills readiness among academic staff. A survey reveals that 15% of academic staff are highly digitally capable and actively innovating, while 60% use basic digital tools but avoid anything more advanced. The remaining 25% actively resist using digital tools. No institutional development programme for digital skills in teaching and learning exists.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-IN-01', 'A', 'We clearly have digital capability. 15% of our academic staff are highly skilled and leading innovation in teaching and learning', 'Incidental', 1, true, 'Concentrated expertise in a small minority without institutional development is the hallmark of Incidental. The attractive nuisance is celebrating individual champions as institutional capability', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-IN-01', 'B', 'We need to start investing in digital skills for academic staff. We''re planning a development programme targeting the 60% in the middle', 'Intentional', 2, false, NULL, 'Planning purposeful investment in workforce development indicates movement toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-IN-01', 'C', 'We don''t have an institutional programme. Skills are developed through informal peer learning and self-study', 'Incidental', 1, false, NULL, 'Informal, self-directed development without institutional programme is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-IN-01', 'D', 'We''ve launched a targeted digital capabilities programme for academic staff with modules mapped to role requirements, and we''re tracking participation', 'Intentional', 2, false, NULL, 'A targeted, tracked programme indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLP-IN-02 :: Teaching & Learning: People & Culture :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLP-IN-02', 'maturity-the', 'the-tl-people', 'Teaching & Learning: People & Culture', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university is recruiting for a senior role in teaching and learning. The job description does not mention digital competencies. The hiring manager says: ''Digital skills aren''t really relevant for this role. They just need to be good at the core functions.'' Meanwhile, a departing staff member who was a digital champion leaves a significant gap in the team''s digital capability.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-IN-02', 'A', 'We recognise this is a gap. We''ve started reviewing all teaching and learning role profiles to include digital competency requirements', 'Intentional', 2, false, NULL, 'Purposeful review of role profiles to embed digital requirements indicates movement toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-IN-02', 'B', 'Digital skills are important but they''re something people develop on the job. We don''t need to specify them in recruitment', 'Incidental', 1, true, 'Assuming digital skills will develop organically is the Incidental pattern. The attractive nuisance is that this sounds like a reasonable, flexible approach', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-IN-02', 'C', 'All our teaching and learning role profiles now include digital competencies and we assess them at interview', 'Intentional', 2, false, NULL, 'Embedding digital in recruitment indicates Intentional or above', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-IN-02', 'D', 'We don''t currently include digital skills in role profiles for teaching and learning positions', 'Incidental', 1, false, NULL, 'Absence of digital in role profiles is characteristic of Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLP-NI-01 :: Teaching & Learning: People & Culture :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLP-NI-01', 'maturity-the', 'the-tl-people', 'Teaching & Learning: People & Culture', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university has been running digital skills training for academic staff for two years. Attendance is good in some departments but poor in others. A few departments have transformed their practice while most continue as before. Performance reviews do not assess digital capability. You are asked whether digital skills development is working.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-NI-01', 'A', 'Absolutely. Training attendance is strong and we''ve seen real transformation in our leading departments', 'Intentional', 2, true, 'Training attendance with patchy uptake and no integration into performance management is Intentional. The attractive nuisance is pointing to leading departments as evidence of institutional maturity', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-NI-01', 'B', 'We''ve made a good start but digital competencies aren''t yet embedded in performance review, career pathways, or promotion criteria across the institution', 'Intentional', 2, false, NULL, 'Honest recognition that development is not yet embedded in HR processes describes Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-NI-01', 'C', 'Digital competencies are integrated into our PDR process. Development pathways exist at every career stage. Innovation is recognised in promotion criteria', 'Integrated', 3, false, NULL, 'Embedded in HR processes and career pathways indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-NI-01', 'D', 'We''ve mandated digital training completion for all academic staff and track compliance centrally', 'Intentional', 2, true, 'Mandatory compliance training can generate resentment rather than culture change. Mandated attendance without HR embedding is still Intentional', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLP-NI-02 :: Teaching & Learning: People & Culture :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLP-NI-02', 'maturity-the', 'the-tl-people', 'Teaching & Learning: People & Culture', 'intentional-integrated', 'Intentional', 'Integrated', 'A head of department at your university approaches you saying their team needs more digital skills support. They''ve been relying on one team member who is the ''digital person'' for all technology-related work. When that person is on leave, digital projects stall. They ask what the institution offers.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-NI-02', 'A', 'We have a comprehensive programme but capability tends to concentrate in enthusiasts. We haven''t yet built distributed competence across all teams', 'Intentional', 2, false, NULL, 'Champion-dependency describes Intentional even with good programmes', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-NI-02', 'B', 'Our competency framework ensures all staff develop digital skills as part of their role. No team should depend on a single person', 'Integrated', 3, false, NULL, 'Institutional competency framework preventing single-person dependency indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-NI-02', 'C', 'That''s exactly what our community of practice and digital champions network is designed to address. We support knowledge sharing across teams', 'Intentional', 2, true, 'A champions network can address symptoms but not root cause. If capability remains concentrated in designated champions, this is Intentional', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-NI-02', 'D', 'We have some training available. I''d suggest sending a couple of their team on the next session', 'Intentional', 2, false, NULL, 'Ad-hoc training referral without systematic capability building is Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLP-IO-01 :: Teaching & Learning: People & Culture :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLP-IO-01', 'maturity-the', 'the-tl-people', 'Teaching & Learning: People & Culture', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has invested heavily in developing digital capabilities across academic staff. Competency frameworks are embedded in HR processes, communities of practice are active, and satisfaction is high. You are asked whether the institution should now focus elsewhere or continue investing.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-IO-01', 'A', 'Our development programmes are comprehensive and well-attended. We can now focus investment elsewhere and maintain current provision', 'Integrated', 3, true, 'Good institutional provision maintained centrally is Integrated. Optimised culture is self-sustaining and peer-driven, not dependent on continued central provision', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-IO-01', 'B', 'Development has become largely self-sustaining. Staff drive their own learning and peer development. We should invest in supporting that culture, not controlling it', 'Optimised', 4, false, NULL, 'Self-sustaining, peer-driven development culture indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-IO-01', 'C', 'We''re well-established internally. We should now contribute to sector workforce development by sharing our frameworks and offering training to other institutions', 'Optimised', 4, false, NULL, 'Sector contribution indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-IO-01', 'D', 'We need to maintain our investment. Without continued institutional programmes, capability will erode', 'Integrated', 3, false, NULL, 'Dependency on institutional programmes for capability maintenance indicates Integrated not Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLP-IO-02 :: Teaching & Learning: People & Culture :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLP-IO-02', 'maturity-the', 'the-tl-people', 'Teaching & Learning: People & Culture', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university is known for strong digital skills among academic staff. A peer institution contacts you asking to learn from your approach. You also notice that your digital innovation often follows trends set by two or three leading institutions rather than originating internally.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-IO-02', 'A', 'We''re happy to share our approach. We''ve developed it by carefully studying and adapting best practice from leading institutions', 'Integrated', 3, true, 'Adapting others'' best practice is Integrated. The attractive nuisance is that sharing your adapted approach feels like leadership', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-IO-02', 'B', 'We generate original approaches that others adopt. Our staff regularly publish and present on novel digital practices they''ve developed', 'Optimised', 4, false, NULL, 'Originating novel approaches adopted by peers indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-IO-02', 'C', 'We''re strong implementers of established good practice. We aren''t really generating new approaches that others follow', 'Integrated', 3, false, NULL, 'Honest recognition of implementing vs leading is Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLP-IO-02', 'D', 'Our approach is distinctive and recognised. We contribute to sector thinking through advisory roles and published research on digital workforce development', 'Optimised', 4, false, NULL, 'Sustained sector contribution through advisory and publication indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLT-IN-01 :: Teaching & Learning: Technology :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLT-IN-01', 'maturity-the', 'the-tl-technology', 'Teaching & Learning: Technology', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university recently discovered that five departments are using three different tools for essentially the same teaching and learning function. No one made a decision to standardise, and each department chose independently. IT maintains all three but has raised concerns about sustainability.', 'What would you most likely do?', '{"institution_size":"medium","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-IN-01', 'A', 'This shows our departments are proactive about adopting technology. We support local autonomy in tool selection', 'Incidental', 1, true, 'Framing fragmentation as autonomy is the Incidental attractive nuisance. Uncoordinated duplication is not empowerment', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-IN-01', 'B', 'We''ve identified this as a problem and are developing a technology roadmap to rationalise and integrate our teaching and learning systems', 'Intentional', 2, false, NULL, 'Developing a purposeful roadmap indicates movement toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-IN-01', 'C', 'This has happened because we don''t have an institutional technology strategy for teaching and learning. Departments filled gaps independently', 'Incidental', 1, false, NULL, 'Honest recognition of uncoordinated adoption is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-IN-01', 'D', 'We have an approved technology roadmap and procurement policy requiring architectural review. New tools must align with our standards', 'Intentional', 2, false, NULL, 'An approved roadmap with procurement governance indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLT-IN-02 :: Teaching & Learning: Technology :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLT-IN-02', 'maturity-the', 'the-tl-technology', 'Teaching & Learning: Technology', 'incidental-intentional', 'Incidental', 'Intentional', 'IT at your university has been asked to prepare a technology investment case for teaching and learning. When they attempt to map the current landscape, they find no documentation of what systems are in use across departments, who owns them, or how they connect.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-IN-02', 'A', 'We know our landscape is complex. We''ve commissioned an audit and will use it to develop a roadmap with integration priorities', 'Intentional', 2, false, NULL, 'Commissioning an audit to inform purposeful planning indicates movement toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-IN-02', 'B', 'We have a comprehensive, documented technology landscape with identified integration points and investment priorities', 'Intentional', 2, false, NULL, 'If true, this indicates Intentional or above', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-IN-02', 'C', 'Our systems work fine individually. We don''t really need a map because each department manages its own technology effectively', 'Incidental', 1, true, 'Defending fragmentation as departmental effectiveness is Incidental', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-IN-02', 'D', 'We recognise we have limited visibility of our technology landscape and it''s never been formally documented', 'Incidental', 1, false, NULL, 'Undocumented, unmanaged technology landscape is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLT-NI-01 :: Teaching & Learning: Technology :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLT-NI-01', 'maturity-the', 'the-tl-technology', 'Teaching & Learning: Technology', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university has invested in modernising its VLE and assessment platform and migrated to cloud hosting. Single sign-on connects the main platforms. However, a review reveals that several key data flows between systems are manual, the architecture has no formal governance, and procurement still happens without architectural review in some faculties.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-NI-01', 'A', 'We''ve achieved integration. Our core systems are in the cloud with SSO and our main platforms are connected', 'Intentional', 2, true, 'Cloud hosting and SSO are positive steps but manual data flows, ungoverned architecture, and uncontrolled procurement indicate Intentional. The attractive nuisance is equating modernisation with integration', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-NI-01', 'B', 'We''ve made good progress on modernisation but we haven''t yet achieved full architectural governance and automated data flows across all systems', 'Intentional', 2, false, NULL, 'Recognising the gap between modernisation and integration accurately describes Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-NI-01', 'C', 'All our teaching and learning systems are governed by enterprise architecture standards, connected through APIs, and procurement requires architectural review institution-wide', 'Integrated', 3, false, NULL, 'Comprehensive architecture governance with API integration indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-NI-01', 'D', 'We have a full integration platform with service level monitoring, planned refresh cycles, and joint IT-domain governance', 'Integrated', 3, false, NULL, 'Integration platform with comprehensive governance indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLT-NI-02 :: Teaching & Learning: Technology :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLT-NI-02', 'maturity-the', 'the-tl-technology', 'Teaching & Learning: Technology', 'intentional-integrated', 'Intentional', 'Integrated', 'All faculties at your university use the same VLE and assessment platform. IT mandated this five years ago. However, faculties configure the system differently, there are no shared standards, and the system was chosen by IT without consulting domain experts. You are asked whether this represents integrated technology.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-NI-02', 'A', 'Absolutely. Everyone uses the same platform, which means our technology is integrated for teaching and learning', 'Intentional', 2, true, 'Mandated shared platform without configuration standards, domain input, or architectural governance is technology standardisation not integration', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-NI-02', 'B', 'We have a shared platform but we recognise it was an IT-led decision without domain governance. We''re now establishing joint governance with configuration standards', 'Intentional', 2, false, NULL, 'Recognising the governance gap and working to address it describes Intentional moving toward Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-NI-02', 'C', 'Our shared platform has institution-wide configuration standards developed jointly by IT and domain stakeholders, with regular review', 'Integrated', 3, false, NULL, 'Jointly governed platform with shared standards indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-NI-02', 'D', 'The platform was chosen without faculty input and each faculty uses it differently. It creates as many problems as it solves', 'Intentional', 2, false, NULL, 'Mandated without governance is not integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLT-IO-01 :: Teaching & Learning: Technology :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLT-IO-01', 'maturity-the', 'the-tl-technology', 'Teaching & Learning: Technology', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has well-governed, integrated technology for teaching and learning. A new technology emerges that could significantly enhance teaching and learning capability. Several peer institutions are evaluating it. You are asked how to respond.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"research-intensive","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-IO-01', 'A', 'We should wait to see how peers implement it and learn from their experience before committing resources', 'Integrated', 3, true, 'Waiting to learn from peers is reactive, characteristic of Integrated. The attractive nuisance is that this feels prudent', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-IO-01', 'B', 'Our sandbox environment is already being used to evaluate this. Our technology futures panel assessed it three months ago and we have a pilot planned', 'Optimised', 4, false, NULL, 'Proactive assessment through established processes indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-IO-01', 'C', 'We should commission a thorough evaluation and develop a business case before proceeding', 'Integrated', 3, false, NULL, 'Thorough evaluation is good practice but reactive evaluation of already-visible technologies is Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-IO-01', 'D', 'We anticipated this development through our horizon scanning. We''ve already published a position paper and are advising sector bodies on implementation approaches', 'Optimised', 4, false, NULL, 'Anticipation through horizon scanning and sector leadership indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLT-IO-02 :: Teaching & Learning: Technology :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLT-IO-02', 'maturity-the', 'the-tl-technology', 'Teaching & Learning: Technology', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university achieves 99.5% uptime on its VLE and assessment platform and has strong user satisfaction scores. A vendor invites you to present your technology approach at their annual conference as a customer success story. You are asked whether this means you are sector-leading.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-IO-02', 'A', 'High availability and vendor recognition confirms we are Optimised in technology for teaching and learning', 'Integrated', 3, true, 'Reliability is Integrated. Vendor marketing invitations are not the same as sector-recognised innovation and leadership', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-IO-02', 'B', 'We''re well-run but reliable operations are table stakes. We need to ask whether our architecture enables innovation and whether we''re advancing practice beyond our own institution', 'Integrated', 3, false, NULL, 'Recognising that reliability alone is not sector leadership accurately assesses Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-IO-02', 'C', 'Our architecture is extensible, we have sandbox environments, we run regular technology innovation cycles, and peer institutions adopt our published architectural patterns', 'Optimised', 4, false, NULL, 'Innovation capacity and adopted patterns indicate Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLT-IO-02', 'D', 'We''re early adopters of new features and always among the first to upgrade to the latest version', 'Integrated', 3, true, 'Early adoption of vendor releases is not the same as architectural innovation and sector contribution', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLD-IN-01 :: Teaching & Learning: Data :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLD-IN-01', 'maturity-the', 'the-tl-data', 'Teaching & Learning: Data', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university needs to produce a report on digital teaching and learning activity for a regulatory body. The data team discovers that relevant metrics are held in spreadsheets by individual departments, each using different definitions and formats. Compiling the report takes three weeks of manual work.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"UK"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-IN-01', 'A', 'We meet all our statutory reporting requirements. Our data management is adequate for external purposes', 'Incidental', 1, true, 'Meeting statutory requirements through manual compilation is Incidental. The attractive nuisance is equating compliance with data maturity', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-IN-01', 'B', 'We''ve identified our core student and learning datasets, standardised definitions, and started systematic collection to replace departmental spreadsheets', 'Intentional', 2, false, NULL, 'Standardised definitions and systematic collection indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-IN-01', 'C', 'Our student and learning data is fragmented across departments with no standard definitions. We rely on manual compilation for reporting', 'Incidental', 1, false, NULL, 'Fragmented data with manual compilation is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-IN-01', 'D', 'Our student and learning data is centrally managed with standardised definitions, automated collection, and dashboards. The report could be produced in hours', 'Intentional', 2, false, NULL, 'This describes at least Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLD-IN-02 :: Teaching & Learning: Data :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLD-IN-02', 'maturity-the', 'the-tl-data', 'Teaching & Learning: Data', 'incidental-intentional', 'Incidental', 'Intentional', 'A dean at your university wants to make a data-informed decision about digital investment in teaching and learning. When they ask for relevant data, they are told it doesn''t exist in any centralised form and would need to be collected manually from multiple sources.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-IN-02', 'A', 'Our student and learning data is comprehensive and available through self-service dashboards', 'Intentional', 2, false, NULL, 'This would indicate at least Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-IN-02', 'B', 'We don''t currently have systematic data collection for teaching and learning. We''re planning to implement it', 'Incidental', 1, false, NULL, 'No systematic collection is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-IN-02', 'C', 'We collect good data but it sits in different systems and teams. We''re working on integrating it into a central platform', 'Intentional', 2, false, NULL, 'Purposeful integration of existing data indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-IN-02', 'D', 'Deans should be able to make these decisions based on their professional judgment. They shouldn''t need a dashboard for everything', 'Incidental', 1, true, 'Dismissing the need for data-informed decisions is characteristic of Incidental. The attractive nuisance is framing this as valuing professional expertise', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLD-NI-01 :: Teaching & Learning: Data :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLD-NI-01', 'maturity-the', 'the-tl-data', 'Teaching & Learning: Data', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university has built dashboards for student and learning data and drafted a data governance policy. However, a data quality audit reveals significant inconsistencies: 30% of key fields have missing or incorrect data, governance compliance varies by department, and most committees still make decisions without consulting the available data.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-NI-01', 'A', 'We have dashboards and governance in place. Data quality will improve over time as people get used to the new systems', 'Intentional', 2, true, 'Dashboards and policy without quality management and actual use for decision-making is Intentional. The attractive nuisance is expecting passive improvement', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-NI-01', 'B', 'We''ve built the infrastructure but haven''t yet achieved reliable quality, consistent governance, or data-informed decision-making across the institution', 'Intentional', 2, false, NULL, 'Recognising the gap between infrastructure and institutional adoption accurately describes Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-NI-01', 'C', 'Our data quality is actively managed with regular audits, governance is operational with compliance monitoring, and committee papers routinely include data analysis', 'Integrated', 3, false, NULL, 'Active quality management, operational governance, and routine data use indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-NI-01', 'D', 'We need to prioritise data quality remediation and embed data use in committee processes before we can call our data mature', 'Intentional', 2, false, NULL, 'Identifying remediation needs accurately places the institution at Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLD-NI-02 :: Teaching & Learning: Data :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLD-NI-02', 'maturity-the', 'the-tl-data', 'Teaching & Learning: Data', 'intentional-integrated', 'Intentional', 'Integrated', 'The planning department at your university produces excellent student and learning reports. However, when you investigate, you find these reports are produced by a small specialist team. Faculty and department leaders do not have self-service access and must request custom reports each time. Data governance relies on the planning team''s expertise rather than institutional processes.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-NI-02', 'A', 'Our reporting is excellent and the planning team ensures data quality. This is an effective model', 'Intentional', 2, true, 'Expert-dependent reporting without self-service or institutional governance is Intentional. The attractive nuisance is that high-quality output feels like maturity', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-NI-02', 'B', 'We produce good reports but data capability is concentrated in one team. We need to democratise access and formalise governance institutionally', 'Intentional', 2, false, NULL, 'Expert dependency without distributed access is Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-NI-02', 'C', 'Self-service analytics are available to authorised users across the institution with institutional data governance ensuring quality', 'Integrated', 3, false, NULL, 'Distributed access with institutional governance indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-NI-02', 'D', 'Leaders can access data through an institutional analytics platform with training support and defined governance roles across all units', 'Integrated', 3, false, NULL, 'Institutional platform with governance indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLD-IO-01 :: Teaching & Learning: Data :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLD-IO-01', 'maturity-the', 'the-tl-data', 'Teaching & Learning: Data', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has comprehensive, well-governed student and learning data with institution-wide dashboards. A vendor offers an AI-powered predictive analytics tool. The PVC Education is excited and wants to implement it immediately, claiming it will make the institution Optimised.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-IO-01', 'A', 'Implementing AI analytics on our well-governed data will make us sector-leading immediately', 'Integrated', 3, true, 'Tool adoption does not equal maturity. Optimised requires proven impact, continuous improvement, and sector contribution, not just tool acquisition', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-IO-01', 'B', 'We should pilot the tool, evaluate impact against outcomes, and publish our findings. If it works, we''ll iterate and share our methodology', 'Optimised', 4, false, NULL, 'Evidence-based evaluation with publication and continuous improvement indicates Optimised approach', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-IO-01', 'C', 'We already use predictive models validated against outcomes with documented impact. We''d evaluate this tool against our existing capabilities', 'Optimised', 4, false, NULL, 'Existing validated predictive capabilities with impact evidence indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-IO-01', 'D', 'We should implement it carefully with proper evaluation. AI tools need to be properly governed before deployment', 'Integrated', 3, false, NULL, 'Careful implementation with governance is good Integrated practice, not Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLD-IO-02 :: Teaching & Learning: Data :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLD-IO-02', 'maturity-the', 'the-tl-data', 'Teaching & Learning: Data', 'integrated-optimised', 'Integrated', 'Optimised', 'A sector body asks your university to contribute to developing new data standards for teaching and learning. You currently have strong internal data governance but have not previously engaged with sector-level data practice.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-IO-02', 'A', 'We''d be delighted to contribute. Our data governance for teaching and learning is strong internally and we''re ready to share our approaches', 'Optimised', 4, false, NULL, 'Willingness and readiness to contribute to sector standards indicates movement toward Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-IO-02', 'B', 'We''re confident our internal data practice is good but we should focus on maintaining what we have rather than taking on sector work', 'Integrated', 3, false, NULL, 'Internal focus without sector contribution is Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-IO-02', 'C', 'We''ve been contributing to sector data standards for several years and our governance framework has been adopted by three peer institutions', 'Optimised', 4, false, NULL, 'Sustained contribution with peer adoption indicates established Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLD-IO-02', 'D', 'We benchmark our data governance against the sector body''s existing standards to ensure we meet best practice', 'Integrated', 3, true, 'Benchmarking against others'' standards is consuming not contributing. The attractive nuisance is that benchmarking feels like sector engagement', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLU-IN-01 :: Teaching & Learning: Utilisation :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLU-IN-01', 'maturity-the', 'the-tl-utilization', 'Teaching & Learning: Utilisation', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university invested in VLE, online assessment, and learning analytics two years ago. Usage data (where available) shows that only 30% of academics and students regularly use the core features, and fewer than 10% use advanced capabilities. Most academics and students continue with previous manual or paper-based approaches.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-IN-01', 'A', 'We''ve made the investment and the tools are available. People will adopt at their own pace', 'Incidental', 1, true, 'Availability without promotion of adoption is Incidental. The attractive nuisance is framing passive availability as a strategy', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-IN-01', 'B', 'We''ve launched a training programme with minimum usage expectations and we''re tracking adoption rates institution-wide', 'Intentional', 2, false, NULL, 'Purposeful promotion with training and tracking indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-IN-01', 'C', 'Adoption is low and we don''t have a plan to address it. The tools are there but people haven''t taken to them', 'Incidental', 1, false, NULL, 'Unaddressed low adoption is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-IN-01', 'D', 'Usage is high and consistent. Over 80% of academics and students regularly use both basic and advanced features', 'Intentional', 2, false, NULL, 'This would indicate at least Intentional but contradicts the scenario', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLU-IN-02 :: Teaching & Learning: Utilisation :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLU-IN-02', 'maturity-the', 'the-tl-utilization', 'Teaching & Learning: Utilisation', 'incidental-intentional', 'Incidental', 'Intentional', 'A few academics and students at your university have developed impressive workflows using VLE, online assessment, and learning analytics and have been nominated for an innovation award. Meanwhile, their immediate colleagues continue using older methods. No one has been asked to adopt the innovative approaches more widely.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-IN-02', 'A', 'Our award nominees demonstrate excellent utilisation across the institution', 'Incidental', 1, true, 'Individual excellence is the Incidental pattern. The attractive nuisance is pointing to champions as evidence of institutional utilisation', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-IN-02', 'B', 'We''ve identified these innovators and are using their approaches to develop institutional training and minimum standards for all academics and students', 'Intentional', 2, false, NULL, 'Converting individual innovation into institutional programme indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-IN-02', 'C', 'These individuals found their own way. We haven''t yet developed an institutional approach to promoting consistent utilisation', 'Incidental', 1, false, NULL, 'Individual adoption without institutional promotion is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-IN-02', 'D', 'We have minimum standards for tool usage and consistent adoption is monitored across the institution', 'Intentional', 2, false, NULL, 'Institutional standards with monitoring indicates at least Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLU-NI-01 :: Teaching & Learning: Utilisation :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLU-NI-01', 'maturity-the', 'the-tl-utilization', 'Teaching & Learning: Utilisation', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university has been actively promoting adoption of VLE, online assessment, and learning analytics with training programmes and published expectations. Adoption has risen to 65% for basic features, with three faculties at over 80% and two below 40%. Impact on outcomes is not measured.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-NI-01', 'A', 'Adoption is growing strongly. We''re approaching integrated utilisation across the institution', 'Intentional', 2, true, 'Significant variation (40-80%) across faculties and no impact measurement is Intentional. The attractive nuisance is that average growth masks inconsistency', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-NI-01', 'B', 'Adoption is growing but not yet consistent. We need institution-wide standards, measurement of outcomes, and intervention in lagging areas', 'Intentional', 2, false, NULL, 'Inconsistent adoption without impact measurement is Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-NI-01', 'C', 'Adoption is consistently above 80% across all units, we measure impact on outcomes, and user feedback informs tool optimisation', 'Integrated', 3, false, NULL, 'Consistent high adoption with impact measurement indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-NI-01', 'D', 'All core teaching and learning processes run on digital workflows and we can demonstrate improvement in outcomes attributable to tool utilisation', 'Integrated', 3, false, NULL, 'Digital workflows as standard with demonstrated impact indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLU-NI-02 :: Teaching & Learning: Utilisation :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLU-NI-02', 'maturity-the', 'the-tl-utilization', 'Teaching & Learning: Utilisation', 'intentional-integrated', 'Intentional', 'Integrated', 'Several departments at your university report that they have ''gone fully digital'' for teaching and learning processes. However, an audit reveals that while forms are digital, they are printed out for review, approvals happen by email rather than through the system, and data entry is duplicated across platforms.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-NI-02', 'A', 'We''ve digitised our processes. All our forms and records are digital now', 'Intentional', 2, true, 'Digitising the form without digitising the workflow is Intentional. The attractive nuisance is equating digital forms with digital processes', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-NI-02', 'B', 'We''ve digitised inputs but not workflows. True utilisation means end-to-end digital processes without manual intervention', 'Intentional', 2, false, NULL, 'Recognising partial digitisation accurately describes Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-NI-02', 'C', 'Our processes run end-to-end digitally with no paper fallbacks, automated routing, and single data entry', 'Integrated', 3, false, NULL, 'End-to-end digital workflows indicate Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-NI-02', 'D', 'We''ve not digitised our processes yet. Most work is still paper-based', 'Incidental', 1, false, NULL, 'This understates the scenario which describes partial digitisation', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLU-IO-01 :: Teaching & Learning: Utilisation :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLU-IO-01', 'maturity-the', 'the-tl-utilization', 'Teaching & Learning: Utilisation', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has consistent, high utilisation of VLE, online assessment, and learning analytics across all units. A vendor approaches you asking to feature your institution as a case study for how well you use their product. You are asked whether this means your utilisation is Optimised.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-IO-01', 'A', 'Vendor recognition confirms we are sector-leading in utilisation of teaching and learning tools', 'Integrated', 3, true, 'Vendor case studies are marketing tools, not independent assessment. Consistent usage of a product as intended is Integrated, not innovation', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-IO-01', 'B', 'We use the tools well as designed. But Optimised utilisation means our users innovate new use cases and drive continuous improvement beyond standard deployment', 'Integrated', 3, false, NULL, 'Distinguishing standard effective use from innovation-driven optimisation is accurate', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-IO-01', 'C', 'Our users have developed novel applications of these tools that the vendor has incorporated into their product roadmap. We continuously optimise based on usage analytics', 'Optimised', 4, false, NULL, 'User innovation influencing vendor roadmaps and continuous optimisation indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-IO-01', 'D', 'We regularly present at user conferences sharing innovative workflows our academics and students have developed, and peer institutions adopt our configurations', 'Optimised', 4, false, NULL, 'Innovative workflows adopted by peers indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-TLU-IO-02 :: Teaching & Learning: Utilisation :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-TLU-IO-02', 'maturity-the', 'the-tl-utilization', 'Teaching & Learning: Utilisation', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university monitors utilisation of VLE, online assessment, and learning analytics through monthly reports. Adoption is consistently above 85% for core features. Users report high satisfaction. A review asks whether there is anything more to achieve.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-IO-02', 'A', 'We''ve achieved consistent high utilisation. We should maintain current levels and focus investment elsewhere', 'Integrated', 3, true, 'Maintenance of current utilisation is Integrated. Optimised means continuous improvement and innovation, not steady-state management', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-IO-02', 'B', 'We should move from monitoring adoption to analysing usage patterns to identify optimisation opportunities and measuring impact on outcomes', 'Optimised', 4, false, NULL, 'Moving from adoption monitoring to optimisation and impact measurement indicates Optimised thinking', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-IO-02', 'C', 'Our users already drive innovation. We analyse usage patterns in real-time, users contribute novel workflows, and we quantify impact on outcomes', 'Optimised', 4, false, NULL, 'User-driven innovation with analytics and impact quantification indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-TLU-IO-02', 'D', 'We should push adoption of advanced features to increase the 85% further', 'Integrated', 3, false, NULL, 'Pursuing higher adoption of existing features is still an Integrated activity, not innovation-driven optimisation', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-RES-IN-01 :: Research: Strategy :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-RES-IN-01', 'maturity-the', 'the-re-strategy', 'Research: Strategy', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university has been responding to digital demands in research on a case-by-case basis. Several departments have adopted different tools independently. The senior leadership team has recently discussed the need for a more coordinated approach. A deputy vice-chancellor asks you: ''Where are we on digital strategy for research?''', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-IN-01', 'A', 'We have a clear digital strategy for research that was approved last year and is being implemented across all faculties', 'Intentional', 2, false, NULL, 'This would indicate Intentional if true, but the scenario describes no approved strategy', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-IN-01', 'B', 'We recognise the need and are actively developing a digital strategy for research with identified priorities and a governance proposal', 'Intentional', 2, false, NULL, 'Active development of a purposeful strategy with governance indicates transition toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-IN-01', 'C', 'We''ve been meaning to write a strategy but haven''t found the time. Meanwhile, departments are managing things in their own way', 'Incidental', 1, false, NULL, 'Acknowledged need without action and devolved ad-hoc activity is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-IN-01', 'D', 'Our institutional strategic plan mentions digital transformation and we reference that when departments ask for guidance', 'Incidental', 1, true, 'A passing mention in a broader strategy without specific objectives, owners, or resources for research is not a purposeful strategy. This is the ''we have a plan'' attractive nuisance', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-RES-IN-02 :: Research: Strategy :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-RES-IN-02', 'maturity-the', 'the-re-strategy', 'Research: Strategy', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university''s board has asked for an update on how digital technology supports research. You discover that while several successful digital initiatives exist across the institution, they were each initiated by individual champions with no central coordination. The board wants to know what the institutional approach is.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-IN-02', 'A', 'We have a coordinated institutional approach with a strategy document, dedicated budget, and a committee overseeing digital in research', 'Intentional', 2, false, NULL, 'This describes Intentional with formal strategy, resources, and governance', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-IN-02', 'B', 'We have some excellent initiatives and we''re now developing a formal strategy to bring them together under a coherent plan', 'Intentional', 2, false, NULL, 'Transitioning from ad-hoc to purposeful planning indicates movement toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-IN-02', 'C', 'We have a lot of innovative activity happening organically. Our approach is to let a thousand flowers bloom and learn from what works', 'Incidental', 1, true, 'Framing lack of strategy as deliberate emergent innovation is a common attractive nuisance. Organic activity without coordination is Incidental', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-IN-02', 'D', 'We don''t really have an institutional approach yet. Individual departments have done their own thing based on local needs', 'Incidental', 1, false, NULL, 'Honest acknowledgement of no institutional approach is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-RES-NI-01 :: Research: Strategy :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-RES-NI-01', 'maturity-the', 'the-re-strategy', 'Research: Strategy', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university approved a digital research strategy 18 months ago. The strategy has clear objectives and a steering group meets quarterly. However, implementation varies dramatically across faculties. Two faculties are well advanced, three have barely started, and the remaining faculties fall somewhere in between. A new PVC asks how well the strategy is being implemented.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-NI-01', 'A', 'The strategy is fully embedded. All faculties have adopted it and are implementing it consistently with local adaptation', 'Integrated', 3, false, NULL, 'This would indicate Integrated if true, but the scenario contradicts this', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-NI-01', 'B', 'We have strong pockets of implementation and we''re working to bring all faculties up to the standard of our leading areas', 'Intentional', 2, true, '''Strong pockets'' with inconsistent implementation across the institution is characteristic of Intentional. The attractive nuisance is that activity in multiple locations feels like integration', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-NI-01', 'C', 'Implementation is uneven. We have the strategy but we haven''t yet achieved consistent cross-institutional adoption with proper governance and accountability', 'Intentional', 2, false, NULL, 'Honest assessment of uneven implementation describes Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-NI-01', 'D', 'All faculties have operational plans that reference the institutional strategy, with locally adapted targets, and we report on progress to the board termly', 'Integrated', 3, false, NULL, 'Faculty-level plans aligned to institutional strategy with regular reporting indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-RES-NI-02 :: Research: Strategy :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-RES-NI-02', 'maturity-the', 'the-re-strategy', 'Research: Strategy', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university has invested in a central digital research team of six staff who develop and support digital initiatives. The team runs projects across faculties, but each project requires negotiation with individual faculty leaders for access and cooperation. The team reports to the PVC Research but has no formal authority over faculty-level decisions.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-NI-02', 'A', 'Having a dedicated central team with a clear reporting line shows we have an integrated approach to digital research', 'Integrated', 3, true, 'A central team without governance authority over faculties and without faculty-level adoption of institutional strategy operates in an Intentional model. The attractive nuisance is equating central team existence with integration', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-NI-02', 'B', 'We have purposeful investment in digital research but the central team operates somewhat independently from faculty planning. We need stronger governance to achieve integration', 'Intentional', 2, false, NULL, 'Purposeful investment without cross-institutional governance is accurately Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-NI-02', 'C', 'The central team''s work is governed by a cross-institutional board with faculty representation, and faculty plans are required to align with the digital strategy', 'Integrated', 3, false, NULL, 'If this were true it would indicate Integrated, but the scenario describes a team negotiating cooperation', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-NI-02', 'D', 'Our digital research activity is still at an early stage with no real coordination between the centre and faculties', 'Incidental', 1, false, NULL, 'This understates the scenario, which does describe purposeful investment', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-RES-IO-01 :: Research: Strategy :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-RES-IO-01', 'maturity-the', 'the-re-strategy', 'Research: Strategy', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has a well-functioning digital research strategy that is implemented consistently across all faculties. Governance is strong, KPIs are reported regularly, and the institution has achieved solid results. The PVC Research wants to know whether the institution can now be considered sector-leading.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"research-intensive","region":"UK"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-IO-01', 'A', 'We regularly benchmark against Russell Group peers and ensure our approach matches best practice. We''ve adopted several approaches we learned from peer institutions', 'Integrated', 3, true, 'Benchmarking by adopting approaches from others is characteristic of Integrated. The attractive nuisance is that benchmarking activity feels like sector leadership, but copying good practice is not the same as setting it', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-IO-01', 'B', 'We review our strategy annually based on data, run a horizon scanning function, publish our approaches, and other institutions regularly visit to learn from us', 'Optimised', 4, false, NULL, 'Evidence-based annual review, horizon scanning, publication, and peer recognition indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-IO-01', 'C', 'Our implementation is strong and consistent but we recognise we''re implementing established good practice rather than innovating ahead of the sector', 'Integrated', 3, false, NULL, 'Honest assessment of implementing good practice without leading innovation is Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-IO-01', 'D', 'We''ve won a national award for one of our digital research initiatives and were featured in a trade publication', 'Integrated', 3, true, 'A single award for a specific initiative does not indicate systematic sector leadership. The attractive nuisance is equating one recognition with Optimised maturity', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-RES-IO-02 :: Research: Strategy :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-RES-IO-02', 'maturity-the', 'the-re-strategy', 'Research: Strategy', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has strong digital research governance and consistent implementation. The sector is now grappling with a new technological development that could significantly impact research. Several peer institutions are starting to respond.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-IO-02', 'A', 'We formed an expert panel six months ago to assess the implications and have already piloted approaches. We published a briefing paper that three peer institutions have since adopted', 'Optimised', 4, false, NULL, 'Proactive assessment, early piloting, and sector contribution through publication indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-IO-02', 'B', 'We''re watching what leading institutions are doing and plan to adopt best practice once it becomes clearer what works', 'Integrated', 3, true, 'Waiting to adopt others'' practices is reactive benchmarking, characteristic of Integrated. The attractive nuisance is that this feels prudent and strategic', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-IO-02', 'C', 'We''ve established a task force to develop our response and are developing a pilot programme informed by sector guidance', 'Integrated', 3, false, NULL, 'Developing a response informed by others indicates Integrated responding effectively, not Optimised leading', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RES-IO-02', 'D', 'We''ve been preparing for this for over a year through our horizon scanning process and have an institutional position and implementation plan ready', 'Optimised', 4, false, NULL, 'Anticipating the development through horizon scanning indicates Optimised proactive strategic maturity', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-REP-IN-01 :: Research: People & Culture :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-REP-IN-01', 'maturity-the', 'the-re-people', 'Research: People & Culture', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university is assessing digital skills readiness among researchers. A survey reveals that 15% of researchers are highly digitally capable and actively innovating, while 60% use basic digital tools but avoid anything more advanced. The remaining 25% actively resist using digital tools. No institutional development programme for digital skills in research exists.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-IN-01', 'A', 'We clearly have digital capability. 15% of our researchers are highly skilled and leading innovation in research', 'Incidental', 1, true, 'Concentrated expertise in a small minority without institutional development is the hallmark of Incidental. The attractive nuisance is celebrating individual champions as institutional capability', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-IN-01', 'B', 'We need to start investing in digital skills for researchers. We''re planning a development programme targeting the 60% in the middle', 'Intentional', 2, false, NULL, 'Planning purposeful investment in workforce development indicates movement toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-IN-01', 'C', 'We don''t have an institutional programme. Skills are developed through informal peer learning and self-study', 'Incidental', 1, false, NULL, 'Informal, self-directed development without institutional programme is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-IN-01', 'D', 'We''ve launched a targeted digital capabilities programme for researchers with modules mapped to role requirements, and we''re tracking participation', 'Intentional', 2, false, NULL, 'A targeted, tracked programme indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-REP-IN-02 :: Research: People & Culture :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-REP-IN-02', 'maturity-the', 'the-re-people', 'Research: People & Culture', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university is recruiting for a senior role in research. The job description does not mention digital competencies. The hiring manager says: ''Digital skills aren''t really relevant for this role. They just need to be good at the core functions.'' Meanwhile, a departing staff member who was a digital champion leaves a significant gap in the team''s digital capability.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-IN-02', 'A', 'We recognise this is a gap. We''ve started reviewing all research role profiles to include digital competency requirements', 'Intentional', 2, false, NULL, 'Purposeful review of role profiles to embed digital requirements indicates movement toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-IN-02', 'B', 'Digital skills are important but they''re something people develop on the job. We don''t need to specify them in recruitment', 'Incidental', 1, true, 'Assuming digital skills will develop organically is the Incidental pattern. The attractive nuisance is that this sounds like a reasonable, flexible approach', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-IN-02', 'C', 'All our research role profiles now include digital competencies and we assess them at interview', 'Intentional', 2, false, NULL, 'Embedding digital in recruitment indicates Intentional or above', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-IN-02', 'D', 'We don''t currently include digital skills in role profiles for research positions', 'Incidental', 1, false, NULL, 'Absence of digital in role profiles is characteristic of Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-REP-NI-01 :: Research: People & Culture :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-REP-NI-01', 'maturity-the', 'the-re-people', 'Research: People & Culture', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university has been running digital skills training for researchers for two years. Attendance is good in some departments but poor in others. A few departments have transformed their practice while most continue as before. Performance reviews do not assess digital capability. You are asked whether digital skills development is working.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-NI-01', 'A', 'Absolutely. Training attendance is strong and we''ve seen real transformation in our leading departments', 'Intentional', 2, true, 'Training attendance with patchy uptake and no integration into performance management is Intentional. The attractive nuisance is pointing to leading departments as evidence of institutional maturity', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-NI-01', 'B', 'We''ve made a good start but digital competencies aren''t yet embedded in performance review, career pathways, or promotion criteria across the institution', 'Intentional', 2, false, NULL, 'Honest recognition that development is not yet embedded in HR processes describes Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-NI-01', 'C', 'Digital competencies are integrated into our PDR process. Development pathways exist at every career stage. Innovation is recognised in promotion criteria', 'Integrated', 3, false, NULL, 'Embedded in HR processes and career pathways indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-NI-01', 'D', 'We''ve mandated digital training completion for all researchers and track compliance centrally', 'Intentional', 2, true, 'Mandatory compliance training can generate resentment rather than culture change. Mandated attendance without HR embedding is still Intentional', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-REP-NI-02 :: Research: People & Culture :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-REP-NI-02', 'maturity-the', 'the-re-people', 'Research: People & Culture', 'intentional-integrated', 'Intentional', 'Integrated', 'A head of department at your university approaches you saying their team needs more digital skills support. They''ve been relying on one team member who is the ''digital person'' for all technology-related work. When that person is on leave, digital projects stall. They ask what the institution offers.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-NI-02', 'A', 'We have a comprehensive programme but capability tends to concentrate in enthusiasts. We haven''t yet built distributed competence across all teams', 'Intentional', 2, false, NULL, 'Champion-dependency describes Intentional even with good programmes', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-NI-02', 'B', 'Our competency framework ensures all staff develop digital skills as part of their role. No team should depend on a single person', 'Integrated', 3, false, NULL, 'Institutional competency framework preventing single-person dependency indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-NI-02', 'C', 'That''s exactly what our community of practice and digital champions network is designed to address. We support knowledge sharing across teams', 'Intentional', 2, true, 'A champions network can address symptoms but not root cause. If capability remains concentrated in designated champions, this is Intentional', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-NI-02', 'D', 'We have some training available. I''d suggest sending a couple of their team on the next session', 'Intentional', 2, false, NULL, 'Ad-hoc training referral without systematic capability building is Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-REP-IO-01 :: Research: People & Culture :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-REP-IO-01', 'maturity-the', 'the-re-people', 'Research: People & Culture', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has invested heavily in developing digital capabilities across researchers. Competency frameworks are embedded in HR processes, communities of practice are active, and satisfaction is high. You are asked whether the institution should now focus elsewhere or continue investing.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-IO-01', 'A', 'Our development programmes are comprehensive and well-attended. We can now focus investment elsewhere and maintain current provision', 'Integrated', 3, true, 'Good institutional provision maintained centrally is Integrated. Optimised culture is self-sustaining and peer-driven, not dependent on continued central provision', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-IO-01', 'B', 'Development has become largely self-sustaining. Staff drive their own learning and peer development. We should invest in supporting that culture, not controlling it', 'Optimised', 4, false, NULL, 'Self-sustaining, peer-driven development culture indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-IO-01', 'C', 'We''re well-established internally. We should now contribute to sector workforce development by sharing our frameworks and offering training to other institutions', 'Optimised', 4, false, NULL, 'Sector contribution indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-IO-01', 'D', 'We need to maintain our investment. Without continued institutional programmes, capability will erode', 'Integrated', 3, false, NULL, 'Dependency on institutional programmes for capability maintenance indicates Integrated not Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-REP-IO-02 :: Research: People & Culture :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-REP-IO-02', 'maturity-the', 'the-re-people', 'Research: People & Culture', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university is known for strong digital skills among researchers. A peer institution contacts you asking to learn from your approach. You also notice that your digital innovation often follows trends set by two or three leading institutions rather than originating internally.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-IO-02', 'A', 'We''re happy to share our approach. We''ve developed it by carefully studying and adapting best practice from leading institutions', 'Integrated', 3, true, 'Adapting others'' best practice is Integrated. The attractive nuisance is that sharing your adapted approach feels like leadership', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-IO-02', 'B', 'We generate original approaches that others adopt. Our staff regularly publish and present on novel digital practices they''ve developed', 'Optimised', 4, false, NULL, 'Originating novel approaches adopted by peers indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-IO-02', 'C', 'We''re strong implementers of established good practice. We aren''t really generating new approaches that others follow', 'Integrated', 3, false, NULL, 'Honest recognition of implementing vs leading is Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REP-IO-02', 'D', 'Our approach is distinctive and recognised. We contribute to sector thinking through advisory roles and published research on digital workforce development', 'Optimised', 4, false, NULL, 'Sustained sector contribution through advisory and publication indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-RET-IN-01 :: Research: Technology :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-RET-IN-01', 'maturity-the', 'the-re-technology', 'Research: Technology', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university recently discovered that five departments are using three different tools for essentially the same research function. No one made a decision to standardise, and each department chose independently. IT maintains all three but has raised concerns about sustainability.', 'What would you most likely do?', '{"institution_size":"medium","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-IN-01', 'A', 'This shows our departments are proactive about adopting technology. We support local autonomy in tool selection', 'Incidental', 1, true, 'Framing fragmentation as autonomy is the Incidental attractive nuisance. Uncoordinated duplication is not empowerment', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-IN-01', 'B', 'We''ve identified this as a problem and are developing a technology roadmap to rationalise and integrate our research systems', 'Intentional', 2, false, NULL, 'Developing a purposeful roadmap indicates movement toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-IN-01', 'C', 'This has happened because we don''t have an institutional technology strategy for research. Departments filled gaps independently', 'Incidental', 1, false, NULL, 'Honest recognition of uncoordinated adoption is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-IN-01', 'D', 'We have an approved technology roadmap and procurement policy requiring architectural review. New tools must align with our standards', 'Intentional', 2, false, NULL, 'An approved roadmap with procurement governance indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-RET-IN-02 :: Research: Technology :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-RET-IN-02', 'maturity-the', 'the-re-technology', 'Research: Technology', 'incidental-intentional', 'Incidental', 'Intentional', 'IT at your university has been asked to prepare a technology investment case for research. When they attempt to map the current landscape, they find no documentation of what systems are in use across departments, who owns them, or how they connect.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-IN-02', 'A', 'We know our landscape is complex. We''ve commissioned an audit and will use it to develop a roadmap with integration priorities', 'Intentional', 2, false, NULL, 'Commissioning an audit to inform purposeful planning indicates movement toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-IN-02', 'B', 'We have a comprehensive, documented technology landscape with identified integration points and investment priorities', 'Intentional', 2, false, NULL, 'If true, this indicates Intentional or above', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-IN-02', 'C', 'Our systems work fine individually. We don''t really need a map because each department manages its own technology effectively', 'Incidental', 1, true, 'Defending fragmentation as departmental effectiveness is Incidental', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-IN-02', 'D', 'We recognise we have limited visibility of our technology landscape and it''s never been formally documented', 'Incidental', 1, false, NULL, 'Undocumented, unmanaged technology landscape is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-RET-NI-01 :: Research: Technology :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-RET-NI-01', 'maturity-the', 'the-re-technology', 'Research: Technology', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university has invested in modernising its research computing and data storage and migrated to cloud hosting. Single sign-on connects the main platforms. However, a review reveals that several key data flows between systems are manual, the architecture has no formal governance, and procurement still happens without architectural review in some faculties.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-NI-01', 'A', 'We''ve achieved integration. Our core systems are in the cloud with SSO and our main platforms are connected', 'Intentional', 2, true, 'Cloud hosting and SSO are positive steps but manual data flows, ungoverned architecture, and uncontrolled procurement indicate Intentional. The attractive nuisance is equating modernisation with integration', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-NI-01', 'B', 'We''ve made good progress on modernisation but we haven''t yet achieved full architectural governance and automated data flows across all systems', 'Intentional', 2, false, NULL, 'Recognising the gap between modernisation and integration accurately describes Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-NI-01', 'C', 'All our research systems are governed by enterprise architecture standards, connected through APIs, and procurement requires architectural review institution-wide', 'Integrated', 3, false, NULL, 'Comprehensive architecture governance with API integration indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-NI-01', 'D', 'We have a full integration platform with service level monitoring, planned refresh cycles, and joint IT-domain governance', 'Integrated', 3, false, NULL, 'Integration platform with comprehensive governance indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-RET-NI-02 :: Research: Technology :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-RET-NI-02', 'maturity-the', 'the-re-technology', 'Research: Technology', 'intentional-integrated', 'Intentional', 'Integrated', 'All faculties at your university use the same research computing and data storage. IT mandated this five years ago. However, faculties configure the system differently, there are no shared standards, and the system was chosen by IT without consulting domain experts. You are asked whether this represents integrated technology.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-NI-02', 'A', 'Absolutely. Everyone uses the same platform, which means our technology is integrated for research', 'Intentional', 2, true, 'Mandated shared platform without configuration standards, domain input, or architectural governance is technology standardisation not integration', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-NI-02', 'B', 'We have a shared platform but we recognise it was an IT-led decision without domain governance. We''re now establishing joint governance with configuration standards', 'Intentional', 2, false, NULL, 'Recognising the governance gap and working to address it describes Intentional moving toward Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-NI-02', 'C', 'Our shared platform has institution-wide configuration standards developed jointly by IT and domain stakeholders, with regular review', 'Integrated', 3, false, NULL, 'Jointly governed platform with shared standards indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-NI-02', 'D', 'The platform was chosen without faculty input and each faculty uses it differently. It creates as many problems as it solves', 'Intentional', 2, false, NULL, 'Mandated without governance is not integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-RET-IO-01 :: Research: Technology :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-RET-IO-01', 'maturity-the', 'the-re-technology', 'Research: Technology', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has well-governed, integrated technology for research. A new technology emerges that could significantly enhance research capability. Several peer institutions are evaluating it. You are asked how to respond.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"research-intensive","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-IO-01', 'A', 'We should wait to see how peers implement it and learn from their experience before committing resources', 'Integrated', 3, true, 'Waiting to learn from peers is reactive, characteristic of Integrated. The attractive nuisance is that this feels prudent', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-IO-01', 'B', 'Our sandbox environment is already being used to evaluate this. Our technology futures panel assessed it three months ago and we have a pilot planned', 'Optimised', 4, false, NULL, 'Proactive assessment through established processes indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-IO-01', 'C', 'We should commission a thorough evaluation and develop a business case before proceeding', 'Integrated', 3, false, NULL, 'Thorough evaluation is good practice but reactive evaluation of already-visible technologies is Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-IO-01', 'D', 'We anticipated this development through our horizon scanning. We''ve already published a position paper and are advising sector bodies on implementation approaches', 'Optimised', 4, false, NULL, 'Anticipation through horizon scanning and sector leadership indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-RET-IO-02 :: Research: Technology :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-RET-IO-02', 'maturity-the', 'the-re-technology', 'Research: Technology', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university achieves 99.5% uptime on its research computing and data storage and has strong user satisfaction scores. A vendor invites you to present your technology approach at their annual conference as a customer success story. You are asked whether this means you are sector-leading.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-IO-02', 'A', 'High availability and vendor recognition confirms we are Optimised in technology for research', 'Integrated', 3, true, 'Reliability is Integrated. Vendor marketing invitations are not the same as sector-recognised innovation and leadership', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-IO-02', 'B', 'We''re well-run but reliable operations are table stakes. We need to ask whether our architecture enables innovation and whether we''re advancing practice beyond our own institution', 'Integrated', 3, false, NULL, 'Recognising that reliability alone is not sector leadership accurately assesses Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-IO-02', 'C', 'Our architecture is extensible, we have sandbox environments, we run regular technology innovation cycles, and peer institutions adopt our published architectural patterns', 'Optimised', 4, false, NULL, 'Innovation capacity and adopted patterns indicate Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RET-IO-02', 'D', 'We''re early adopters of new features and always among the first to upgrade to the latest version', 'Integrated', 3, true, 'Early adoption of vendor releases is not the same as architectural innovation and sector contribution', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-RED-IN-01 :: Research: Data :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-RED-IN-01', 'maturity-the', 'the-re-data', 'Research: Data', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university needs to produce a report on digital research activity for a regulatory body. The data team discovers that relevant metrics are held in spreadsheets by individual departments, each using different definitions and formats. Compiling the report takes three weeks of manual work.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"UK"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-IN-01', 'A', 'We meet all our statutory reporting requirements. Our data management is adequate for external purposes', 'Incidental', 1, true, 'Meeting statutory requirements through manual compilation is Incidental. The attractive nuisance is equating compliance with data maturity', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-IN-01', 'B', 'We''ve identified our core research output and research data datasets, standardised definitions, and started systematic collection to replace departmental spreadsheets', 'Intentional', 2, false, NULL, 'Standardised definitions and systematic collection indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-IN-01', 'C', 'Our research output and research data data is fragmented across departments with no standard definitions. We rely on manual compilation for reporting', 'Incidental', 1, false, NULL, 'Fragmented data with manual compilation is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-IN-01', 'D', 'Our research output and research data data is centrally managed with standardised definitions, automated collection, and dashboards. The report could be produced in hours', 'Intentional', 2, false, NULL, 'This describes at least Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-RED-IN-02 :: Research: Data :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-RED-IN-02', 'maturity-the', 'the-re-data', 'Research: Data', 'incidental-intentional', 'Incidental', 'Intentional', 'A dean at your university wants to make a data-informed decision about digital investment in research. When they ask for relevant data, they are told it doesn''t exist in any centralised form and would need to be collected manually from multiple sources.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-IN-02', 'A', 'Our research output and research data data is comprehensive and available through self-service dashboards', 'Intentional', 2, false, NULL, 'This would indicate at least Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-IN-02', 'B', 'We don''t currently have systematic data collection for research. We''re planning to implement it', 'Incidental', 1, false, NULL, 'No systematic collection is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-IN-02', 'C', 'We collect good data but it sits in different systems and teams. We''re working on integrating it into a central platform', 'Intentional', 2, false, NULL, 'Purposeful integration of existing data indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-IN-02', 'D', 'Deans should be able to make these decisions based on their professional judgment. They shouldn''t need a dashboard for everything', 'Incidental', 1, true, 'Dismissing the need for data-informed decisions is characteristic of Incidental. The attractive nuisance is framing this as valuing professional expertise', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-RED-NI-01 :: Research: Data :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-RED-NI-01', 'maturity-the', 'the-re-data', 'Research: Data', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university has built dashboards for research output and research data data and drafted a data governance policy. However, a data quality audit reveals significant inconsistencies: 30% of key fields have missing or incorrect data, governance compliance varies by department, and most committees still make decisions without consulting the available data.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-NI-01', 'A', 'We have dashboards and governance in place. Data quality will improve over time as people get used to the new systems', 'Intentional', 2, true, 'Dashboards and policy without quality management and actual use for decision-making is Intentional. The attractive nuisance is expecting passive improvement', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-NI-01', 'B', 'We''ve built the infrastructure but haven''t yet achieved reliable quality, consistent governance, or data-informed decision-making across the institution', 'Intentional', 2, false, NULL, 'Recognising the gap between infrastructure and institutional adoption accurately describes Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-NI-01', 'C', 'Our data quality is actively managed with regular audits, governance is operational with compliance monitoring, and committee papers routinely include data analysis', 'Integrated', 3, false, NULL, 'Active quality management, operational governance, and routine data use indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-NI-01', 'D', 'We need to prioritise data quality remediation and embed data use in committee processes before we can call our data mature', 'Intentional', 2, false, NULL, 'Identifying remediation needs accurately places the institution at Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-RED-NI-02 :: Research: Data :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-RED-NI-02', 'maturity-the', 'the-re-data', 'Research: Data', 'intentional-integrated', 'Intentional', 'Integrated', 'The planning department at your university produces excellent research output and research data reports. However, when you investigate, you find these reports are produced by a small specialist team. Faculty and department leaders do not have self-service access and must request custom reports each time. Data governance relies on the planning team''s expertise rather than institutional processes.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-NI-02', 'A', 'Our reporting is excellent and the planning team ensures data quality. This is an effective model', 'Intentional', 2, true, 'Expert-dependent reporting without self-service or institutional governance is Intentional. The attractive nuisance is that high-quality output feels like maturity', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-NI-02', 'B', 'We produce good reports but data capability is concentrated in one team. We need to democratise access and formalise governance institutionally', 'Intentional', 2, false, NULL, 'Expert dependency without distributed access is Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-NI-02', 'C', 'Self-service analytics are available to authorised users across the institution with institutional data governance ensuring quality', 'Integrated', 3, false, NULL, 'Distributed access with institutional governance indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-NI-02', 'D', 'Leaders can access data through an institutional analytics platform with training support and defined governance roles across all units', 'Integrated', 3, false, NULL, 'Institutional platform with governance indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-RED-IO-01 :: Research: Data :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-RED-IO-01', 'maturity-the', 'the-re-data', 'Research: Data', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has comprehensive, well-governed research output and research data data with institution-wide dashboards. A vendor offers an AI-powered predictive analytics tool. The PVC Research is excited and wants to implement it immediately, claiming it will make the institution Optimised.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-IO-01', 'A', 'Implementing AI analytics on our well-governed data will make us sector-leading immediately', 'Integrated', 3, true, 'Tool adoption does not equal maturity. Optimised requires proven impact, continuous improvement, and sector contribution, not just tool acquisition', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-IO-01', 'B', 'We should pilot the tool, evaluate impact against outcomes, and publish our findings. If it works, we''ll iterate and share our methodology', 'Optimised', 4, false, NULL, 'Evidence-based evaluation with publication and continuous improvement indicates Optimised approach', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-IO-01', 'C', 'We already use predictive models validated against outcomes with documented impact. We''d evaluate this tool against our existing capabilities', 'Optimised', 4, false, NULL, 'Existing validated predictive capabilities with impact evidence indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-IO-01', 'D', 'We should implement it carefully with proper evaluation. AI tools need to be properly governed before deployment', 'Integrated', 3, false, NULL, 'Careful implementation with governance is good Integrated practice, not Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-RED-IO-02 :: Research: Data :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-RED-IO-02', 'maturity-the', 'the-re-data', 'Research: Data', 'integrated-optimised', 'Integrated', 'Optimised', 'A sector body asks your university to contribute to developing new data standards for research. You currently have strong internal data governance but have not previously engaged with sector-level data practice.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-IO-02', 'A', 'We''d be delighted to contribute. Our data governance for research is strong internally and we''re ready to share our approaches', 'Optimised', 4, false, NULL, 'Willingness and readiness to contribute to sector standards indicates movement toward Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-IO-02', 'B', 'We''re confident our internal data practice is good but we should focus on maintaining what we have rather than taking on sector work', 'Integrated', 3, false, NULL, 'Internal focus without sector contribution is Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-IO-02', 'C', 'We''ve been contributing to sector data standards for several years and our governance framework has been adopted by three peer institutions', 'Optimised', 4, false, NULL, 'Sustained contribution with peer adoption indicates established Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-RED-IO-02', 'D', 'We benchmark our data governance against the sector body''s existing standards to ensure we meet best practice', 'Integrated', 3, true, 'Benchmarking against others'' standards is consuming not contributing. The attractive nuisance is that benchmarking feels like sector engagement', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-REU-IN-01 :: Research: Utilisation :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-REU-IN-01', 'maturity-the', 'the-re-utilization', 'Research: Utilisation', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university invested in research data systems, computational tools, and collaboration platforms two years ago. Usage data (where available) shows that only 30% of researchers regularly use the core features, and fewer than 10% use advanced capabilities. Most researchers continue with previous manual or paper-based approaches.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-IN-01', 'A', 'We''ve made the investment and the tools are available. People will adopt at their own pace', 'Incidental', 1, true, 'Availability without promotion of adoption is Incidental. The attractive nuisance is framing passive availability as a strategy', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-IN-01', 'B', 'We''ve launched a training programme with minimum usage expectations and we''re tracking adoption rates institution-wide', 'Intentional', 2, false, NULL, 'Purposeful promotion with training and tracking indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-IN-01', 'C', 'Adoption is low and we don''t have a plan to address it. The tools are there but people haven''t taken to them', 'Incidental', 1, false, NULL, 'Unaddressed low adoption is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-IN-01', 'D', 'Usage is high and consistent. Over 80% of researchers regularly use both basic and advanced features', 'Intentional', 2, false, NULL, 'This would indicate at least Intentional but contradicts the scenario', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-REU-IN-02 :: Research: Utilisation :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-REU-IN-02', 'maturity-the', 'the-re-utilization', 'Research: Utilisation', 'incidental-intentional', 'Incidental', 'Intentional', 'A few researchers at your university have developed impressive workflows using research data systems, computational tools, and collaboration platforms and have been nominated for an innovation award. Meanwhile, their immediate colleagues continue using older methods. No one has been asked to adopt the innovative approaches more widely.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-IN-02', 'A', 'Our award nominees demonstrate excellent utilisation across the institution', 'Incidental', 1, true, 'Individual excellence is the Incidental pattern. The attractive nuisance is pointing to champions as evidence of institutional utilisation', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-IN-02', 'B', 'We''ve identified these innovators and are using their approaches to develop institutional training and minimum standards for all researchers', 'Intentional', 2, false, NULL, 'Converting individual innovation into institutional programme indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-IN-02', 'C', 'These individuals found their own way. We haven''t yet developed an institutional approach to promoting consistent utilisation', 'Incidental', 1, false, NULL, 'Individual adoption without institutional promotion is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-IN-02', 'D', 'We have minimum standards for tool usage and consistent adoption is monitored across the institution', 'Intentional', 2, false, NULL, 'Institutional standards with monitoring indicates at least Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-REU-NI-01 :: Research: Utilisation :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-REU-NI-01', 'maturity-the', 'the-re-utilization', 'Research: Utilisation', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university has been actively promoting adoption of research data systems, computational tools, and collaboration platforms with training programmes and published expectations. Adoption has risen to 65% for basic features, with three faculties at over 80% and two below 40%. Impact on outcomes is not measured.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-NI-01', 'A', 'Adoption is growing strongly. We''re approaching integrated utilisation across the institution', 'Intentional', 2, true, 'Significant variation (40-80%) across faculties and no impact measurement is Intentional. The attractive nuisance is that average growth masks inconsistency', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-NI-01', 'B', 'Adoption is growing but not yet consistent. We need institution-wide standards, measurement of outcomes, and intervention in lagging areas', 'Intentional', 2, false, NULL, 'Inconsistent adoption without impact measurement is Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-NI-01', 'C', 'Adoption is consistently above 80% across all units, we measure impact on outcomes, and user feedback informs tool optimisation', 'Integrated', 3, false, NULL, 'Consistent high adoption with impact measurement indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-NI-01', 'D', 'All core research processes run on digital workflows and we can demonstrate improvement in outcomes attributable to tool utilisation', 'Integrated', 3, false, NULL, 'Digital workflows as standard with demonstrated impact indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-REU-NI-02 :: Research: Utilisation :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-REU-NI-02', 'maturity-the', 'the-re-utilization', 'Research: Utilisation', 'intentional-integrated', 'Intentional', 'Integrated', 'Several departments at your university report that they have ''gone fully digital'' for research processes. However, an audit reveals that while forms are digital, they are printed out for review, approvals happen by email rather than through the system, and data entry is duplicated across platforms.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-NI-02', 'A', 'We''ve digitised our processes. All our forms and records are digital now', 'Intentional', 2, true, 'Digitising the form without digitising the workflow is Intentional. The attractive nuisance is equating digital forms with digital processes', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-NI-02', 'B', 'We''ve digitised inputs but not workflows. True utilisation means end-to-end digital processes without manual intervention', 'Intentional', 2, false, NULL, 'Recognising partial digitisation accurately describes Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-NI-02', 'C', 'Our processes run end-to-end digitally with no paper fallbacks, automated routing, and single data entry', 'Integrated', 3, false, NULL, 'End-to-end digital workflows indicate Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-NI-02', 'D', 'We''ve not digitised our processes yet. Most work is still paper-based', 'Incidental', 1, false, NULL, 'This understates the scenario which describes partial digitisation', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-REU-IO-01 :: Research: Utilisation :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-REU-IO-01', 'maturity-the', 'the-re-utilization', 'Research: Utilisation', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has consistent, high utilisation of research data systems, computational tools, and collaboration platforms across all units. A vendor approaches you asking to feature your institution as a case study for how well you use their product. You are asked whether this means your utilisation is Optimised.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-IO-01', 'A', 'Vendor recognition confirms we are sector-leading in utilisation of research tools', 'Integrated', 3, true, 'Vendor case studies are marketing tools, not independent assessment. Consistent usage of a product as intended is Integrated, not innovation', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-IO-01', 'B', 'We use the tools well as designed. But Optimised utilisation means our users innovate new use cases and drive continuous improvement beyond standard deployment', 'Integrated', 3, false, NULL, 'Distinguishing standard effective use from innovation-driven optimisation is accurate', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-IO-01', 'C', 'Our users have developed novel applications of these tools that the vendor has incorporated into their product roadmap. We continuously optimise based on usage analytics', 'Optimised', 4, false, NULL, 'User innovation influencing vendor roadmaps and continuous optimisation indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-IO-01', 'D', 'We regularly present at user conferences sharing innovative workflows our researchers have developed, and peer institutions adopt our configurations', 'Optimised', 4, false, NULL, 'Innovative workflows adopted by peers indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-REU-IO-02 :: Research: Utilisation :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-REU-IO-02', 'maturity-the', 'the-re-utilization', 'Research: Utilisation', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university monitors utilisation of research data systems, computational tools, and collaboration platforms through monthly reports. Adoption is consistently above 85% for core features. Users report high satisfaction. A review asks whether there is anything more to achieve.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-IO-02', 'A', 'We''ve achieved consistent high utilisation. We should maintain current levels and focus investment elsewhere', 'Integrated', 3, true, 'Maintenance of current utilisation is Integrated. Optimised means continuous improvement and innovation, not steady-state management', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-IO-02', 'B', 'We should move from monitoring adoption to analysing usage patterns to identify optimisation opportunities and measuring impact on outcomes', 'Optimised', 4, false, NULL, 'Moving from adoption monitoring to optimisation and impact measurement indicates Optimised thinking', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-IO-02', 'C', 'Our users already drive innovation. We analyse usage patterns in real-time, users contribute novel workflows, and we quantify impact on outcomes', 'Optimised', 4, false, NULL, 'User-driven innovation with analytics and impact quantification indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-REU-IO-02', 'D', 'We should push adoption of advanced features to increase the 85% further', 'Integrated', 3, false, NULL, 'Pursuing higher adoption of existing features is still an Integrated activity, not innovation-driven optimisation', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSS-IN-01 :: Professional Services: Strategy :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSS-IN-01', 'maturity-the', 'the-ps-strategy', 'Professional Services: Strategy', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university has been responding to digital demands in professional services on a case-by-case basis. Several departments have adopted different tools independently. The senior leadership team has recently discussed the need for a more coordinated approach. A deputy vice-chancellor asks you: ''Where are we on digital strategy for professional services?''', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-IN-01', 'A', 'We have a clear digital strategy for professional services that was approved last year and is being implemented across all faculties', 'Intentional', 2, false, NULL, 'This would indicate Intentional if true, but the scenario describes no approved strategy', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-IN-01', 'B', 'We recognise the need and are actively developing a digital strategy for professional services with identified priorities and a governance proposal', 'Intentional', 2, false, NULL, 'Active development of a purposeful strategy with governance indicates transition toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-IN-01', 'C', 'We''ve been meaning to write a strategy but haven''t found the time. Meanwhile, departments are managing things in their own way', 'Incidental', 1, false, NULL, 'Acknowledged need without action and devolved ad-hoc activity is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-IN-01', 'D', 'Our institutional strategic plan mentions digital transformation and we reference that when departments ask for guidance', 'Incidental', 1, true, 'A passing mention in a broader strategy without specific objectives, owners, or resources for professional services is not a purposeful strategy. This is the ''we have a plan'' attractive nuisance', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSS-IN-02 :: Professional Services: Strategy :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSS-IN-02', 'maturity-the', 'the-ps-strategy', 'Professional Services: Strategy', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university''s board has asked for an update on how digital technology supports professional services. You discover that while several successful digital initiatives exist across the institution, they were each initiated by individual champions with no central coordination. The board wants to know what the institutional approach is.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-IN-02', 'A', 'We have a coordinated institutional approach with a strategy document, dedicated budget, and a committee overseeing digital in professional services', 'Intentional', 2, false, NULL, 'This describes Intentional with formal strategy, resources, and governance', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-IN-02', 'B', 'We have some excellent initiatives and we''re now developing a formal strategy to bring them together under a coherent plan', 'Intentional', 2, false, NULL, 'Transitioning from ad-hoc to purposeful planning indicates movement toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-IN-02', 'C', 'We have a lot of innovative activity happening organically. Our approach is to let a thousand flowers bloom and learn from what works', 'Incidental', 1, true, 'Framing lack of strategy as deliberate emergent innovation is a common attractive nuisance. Organic activity without coordination is Incidental', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-IN-02', 'D', 'We don''t really have an institutional approach yet. Individual departments have done their own thing based on local needs', 'Incidental', 1, false, NULL, 'Honest acknowledgement of no institutional approach is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSS-NI-01 :: Professional Services: Strategy :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSS-NI-01', 'maturity-the', 'the-ps-strategy', 'Professional Services: Strategy', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university approved a digital professional services strategy 18 months ago. The strategy has clear objectives and a steering group meets quarterly. However, implementation varies dramatically across faculties. Two faculties are well advanced, three have barely started, and the remaining faculties fall somewhere in between. A new PVC asks how well the strategy is being implemented.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-NI-01', 'A', 'The strategy is fully embedded. All faculties have adopted it and are implementing it consistently with local adaptation', 'Integrated', 3, false, NULL, 'This would indicate Integrated if true, but the scenario contradicts this', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-NI-01', 'B', 'We have strong pockets of implementation and we''re working to bring all faculties up to the standard of our leading areas', 'Intentional', 2, true, '''Strong pockets'' with inconsistent implementation across the institution is characteristic of Intentional. The attractive nuisance is that activity in multiple locations feels like integration', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-NI-01', 'C', 'Implementation is uneven. We have the strategy but we haven''t yet achieved consistent cross-institutional adoption with proper governance and accountability', 'Intentional', 2, false, NULL, 'Honest assessment of uneven implementation describes Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-NI-01', 'D', 'All faculties have operational plans that reference the institutional strategy, with locally adapted targets, and we report on progress to the board termly', 'Integrated', 3, false, NULL, 'Faculty-level plans aligned to institutional strategy with regular reporting indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSS-NI-02 :: Professional Services: Strategy :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSS-NI-02', 'maturity-the', 'the-ps-strategy', 'Professional Services: Strategy', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university has invested in a central digital professional services team of six staff who develop and support digital initiatives. The team runs projects across faculties, but each project requires negotiation with individual faculty leaders for access and cooperation. The team reports to the COO but has no formal authority over faculty-level decisions.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-NI-02', 'A', 'Having a dedicated central team with a clear reporting line shows we have an integrated approach to digital professional services', 'Integrated', 3, true, 'A central team without governance authority over faculties and without faculty-level adoption of institutional strategy operates in an Intentional model. The attractive nuisance is equating central team existence with integration', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-NI-02', 'B', 'We have purposeful investment in digital professional services but the central team operates somewhat independently from faculty planning. We need stronger governance to achieve integration', 'Intentional', 2, false, NULL, 'Purposeful investment without cross-institutional governance is accurately Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-NI-02', 'C', 'The central team''s work is governed by a cross-institutional board with faculty representation, and faculty plans are required to align with the digital strategy', 'Integrated', 3, false, NULL, 'If this were true it would indicate Integrated, but the scenario describes a team negotiating cooperation', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-NI-02', 'D', 'Our digital professional services activity is still at an early stage with no real coordination between the centre and faculties', 'Incidental', 1, false, NULL, 'This understates the scenario, which does describe purposeful investment', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSS-IO-01 :: Professional Services: Strategy :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSS-IO-01', 'maturity-the', 'the-ps-strategy', 'Professional Services: Strategy', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has a well-functioning digital professional services strategy that is implemented consistently across all faculties. Governance is strong, KPIs are reported regularly, and the institution has achieved solid results. The COO wants to know whether the institution can now be considered sector-leading.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"research-intensive","region":"UK"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-IO-01', 'A', 'We regularly benchmark against Russell Group peers and ensure our approach matches best practice. We''ve adopted several approaches we learned from peer institutions', 'Integrated', 3, true, 'Benchmarking by adopting approaches from others is characteristic of Integrated. The attractive nuisance is that benchmarking activity feels like sector leadership, but copying good practice is not the same as setting it', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-IO-01', 'B', 'We review our strategy annually based on data, run a horizon scanning function, publish our approaches, and other institutions regularly visit to learn from us', 'Optimised', 4, false, NULL, 'Evidence-based annual review, horizon scanning, publication, and peer recognition indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-IO-01', 'C', 'Our implementation is strong and consistent but we recognise we''re implementing established good practice rather than innovating ahead of the sector', 'Integrated', 3, false, NULL, 'Honest assessment of implementing good practice without leading innovation is Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-IO-01', 'D', 'We''ve won a national award for one of our digital professional services initiatives and were featured in a trade publication', 'Integrated', 3, true, 'A single award for a specific initiative does not indicate systematic sector leadership. The attractive nuisance is equating one recognition with Optimised maturity', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSS-IO-02 :: Professional Services: Strategy :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSS-IO-02', 'maturity-the', 'the-ps-strategy', 'Professional Services: Strategy', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has strong digital professional services governance and consistent implementation. The sector is now grappling with a new technological development that could significantly impact professional services. Several peer institutions are starting to respond.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-IO-02', 'A', 'We formed an expert panel six months ago to assess the implications and have already piloted approaches. We published a briefing paper that three peer institutions have since adopted', 'Optimised', 4, false, NULL, 'Proactive assessment, early piloting, and sector contribution through publication indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-IO-02', 'B', 'We''re watching what leading institutions are doing and plan to adopt best practice once it becomes clearer what works', 'Integrated', 3, true, 'Waiting to adopt others'' practices is reactive benchmarking, characteristic of Integrated. The attractive nuisance is that this feels prudent and strategic', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-IO-02', 'C', 'We''ve established a task force to develop our response and are developing a pilot programme informed by sector guidance', 'Integrated', 3, false, NULL, 'Developing a response informed by others indicates Integrated responding effectively, not Optimised leading', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSS-IO-02', 'D', 'We''ve been preparing for this for over a year through our horizon scanning process and have an institutional position and implementation plan ready', 'Optimised', 4, false, NULL, 'Anticipating the development through horizon scanning indicates Optimised proactive strategic maturity', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSP-IN-01 :: Professional Services: People & Culture :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSP-IN-01', 'maturity-the', 'the-ps-people', 'Professional Services: People & Culture', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university is assessing digital skills readiness among professional services staff. A survey reveals that 15% of professional services staff are highly digitally capable and actively innovating, while 60% use basic digital tools but avoid anything more advanced. The remaining 25% actively resist using digital tools. No institutional development programme for digital skills in professional services exists.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-IN-01', 'A', 'We clearly have digital capability. 15% of our professional services staff are highly skilled and leading innovation in professional services', 'Incidental', 1, true, 'Concentrated expertise in a small minority without institutional development is the hallmark of Incidental. The attractive nuisance is celebrating individual champions as institutional capability', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-IN-01', 'B', 'We need to start investing in digital skills for professional services staff. We''re planning a development programme targeting the 60% in the middle', 'Intentional', 2, false, NULL, 'Planning purposeful investment in workforce development indicates movement toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-IN-01', 'C', 'We don''t have an institutional programme. Skills are developed through informal peer learning and self-study', 'Incidental', 1, false, NULL, 'Informal, self-directed development without institutional programme is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-IN-01', 'D', 'We''ve launched a targeted digital capabilities programme for professional services staff with modules mapped to role requirements, and we''re tracking participation', 'Intentional', 2, false, NULL, 'A targeted, tracked programme indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSP-IN-02 :: Professional Services: People & Culture :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSP-IN-02', 'maturity-the', 'the-ps-people', 'Professional Services: People & Culture', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university is recruiting for a senior role in professional services. The job description does not mention digital competencies. The hiring manager says: ''Digital skills aren''t really relevant for this role. They just need to be good at the core functions.'' Meanwhile, a departing staff member who was a digital champion leaves a significant gap in the team''s digital capability.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-IN-02', 'A', 'We recognise this is a gap. We''ve started reviewing all professional services role profiles to include digital competency requirements', 'Intentional', 2, false, NULL, 'Purposeful review of role profiles to embed digital requirements indicates movement toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-IN-02', 'B', 'Digital skills are important but they''re something people develop on the job. We don''t need to specify them in recruitment', 'Incidental', 1, true, 'Assuming digital skills will develop organically is the Incidental pattern. The attractive nuisance is that this sounds like a reasonable, flexible approach', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-IN-02', 'C', 'All our professional services role profiles now include digital competencies and we assess them at interview', 'Intentional', 2, false, NULL, 'Embedding digital in recruitment indicates Intentional or above', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-IN-02', 'D', 'We don''t currently include digital skills in role profiles for professional services positions', 'Incidental', 1, false, NULL, 'Absence of digital in role profiles is characteristic of Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSP-NI-01 :: Professional Services: People & Culture :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSP-NI-01', 'maturity-the', 'the-ps-people', 'Professional Services: People & Culture', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university has been running digital skills training for professional services staff for two years. Attendance is good in some departments but poor in others. A few departments have transformed their practice while most continue as before. Performance reviews do not assess digital capability. You are asked whether digital skills development is working.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-NI-01', 'A', 'Absolutely. Training attendance is strong and we''ve seen real transformation in our leading departments', 'Intentional', 2, true, 'Training attendance with patchy uptake and no integration into performance management is Intentional. The attractive nuisance is pointing to leading departments as evidence of institutional maturity', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-NI-01', 'B', 'We''ve made a good start but digital competencies aren''t yet embedded in performance review, career pathways, or promotion criteria across the institution', 'Intentional', 2, false, NULL, 'Honest recognition that development is not yet embedded in HR processes describes Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-NI-01', 'C', 'Digital competencies are integrated into our PDR process. Development pathways exist at every career stage. Innovation is recognised in promotion criteria', 'Integrated', 3, false, NULL, 'Embedded in HR processes and career pathways indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-NI-01', 'D', 'We''ve mandated digital training completion for all professional services staff and track compliance centrally', 'Intentional', 2, true, 'Mandatory compliance training can generate resentment rather than culture change. Mandated attendance without HR embedding is still Intentional', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSP-NI-02 :: Professional Services: People & Culture :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSP-NI-02', 'maturity-the', 'the-ps-people', 'Professional Services: People & Culture', 'intentional-integrated', 'Intentional', 'Integrated', 'A head of department at your university approaches you saying their team needs more digital skills support. They''ve been relying on one team member who is the ''digital person'' for all technology-related work. When that person is on leave, digital projects stall. They ask what the institution offers.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-NI-02', 'A', 'We have a comprehensive programme but capability tends to concentrate in enthusiasts. We haven''t yet built distributed competence across all teams', 'Intentional', 2, false, NULL, 'Champion-dependency describes Intentional even with good programmes', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-NI-02', 'B', 'Our competency framework ensures all staff develop digital skills as part of their role. No team should depend on a single person', 'Integrated', 3, false, NULL, 'Institutional competency framework preventing single-person dependency indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-NI-02', 'C', 'That''s exactly what our community of practice and digital champions network is designed to address. We support knowledge sharing across teams', 'Intentional', 2, true, 'A champions network can address symptoms but not root cause. If capability remains concentrated in designated champions, this is Intentional', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-NI-02', 'D', 'We have some training available. I''d suggest sending a couple of their team on the next session', 'Intentional', 2, false, NULL, 'Ad-hoc training referral without systematic capability building is Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSP-IO-01 :: Professional Services: People & Culture :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSP-IO-01', 'maturity-the', 'the-ps-people', 'Professional Services: People & Culture', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has invested heavily in developing digital capabilities across professional services staff. Competency frameworks are embedded in HR processes, communities of practice are active, and satisfaction is high. You are asked whether the institution should now focus elsewhere or continue investing.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-IO-01', 'A', 'Our development programmes are comprehensive and well-attended. We can now focus investment elsewhere and maintain current provision', 'Integrated', 3, true, 'Good institutional provision maintained centrally is Integrated. Optimised culture is self-sustaining and peer-driven, not dependent on continued central provision', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-IO-01', 'B', 'Development has become largely self-sustaining. Staff drive their own learning and peer development. We should invest in supporting that culture, not controlling it', 'Optimised', 4, false, NULL, 'Self-sustaining, peer-driven development culture indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-IO-01', 'C', 'We''re well-established internally. We should now contribute to sector workforce development by sharing our frameworks and offering training to other institutions', 'Optimised', 4, false, NULL, 'Sector contribution indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-IO-01', 'D', 'We need to maintain our investment. Without continued institutional programmes, capability will erode', 'Integrated', 3, false, NULL, 'Dependency on institutional programmes for capability maintenance indicates Integrated not Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSP-IO-02 :: Professional Services: People & Culture :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSP-IO-02', 'maturity-the', 'the-ps-people', 'Professional Services: People & Culture', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university is known for strong digital skills among professional services staff. A peer institution contacts you asking to learn from your approach. You also notice that your digital innovation often follows trends set by two or three leading institutions rather than originating internally.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-IO-02', 'A', 'We''re happy to share our approach. We''ve developed it by carefully studying and adapting best practice from leading institutions', 'Integrated', 3, true, 'Adapting others'' best practice is Integrated. The attractive nuisance is that sharing your adapted approach feels like leadership', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-IO-02', 'B', 'We generate original approaches that others adopt. Our staff regularly publish and present on novel digital practices they''ve developed', 'Optimised', 4, false, NULL, 'Originating novel approaches adopted by peers indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-IO-02', 'C', 'We''re strong implementers of established good practice. We aren''t really generating new approaches that others follow', 'Integrated', 3, false, NULL, 'Honest recognition of implementing vs leading is Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSP-IO-02', 'D', 'Our approach is distinctive and recognised. We contribute to sector thinking through advisory roles and published research on digital workforce development', 'Optimised', 4, false, NULL, 'Sustained sector contribution through advisory and publication indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PST-IN-01 :: Professional Services: Technology :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PST-IN-01', 'maturity-the', 'the-ps-technology', 'Professional Services: Technology', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university recently discovered that five departments are using three different tools for essentially the same professional services function. No one made a decision to standardise, and each department chose independently. IT maintains all three but has raised concerns about sustainability.', 'What would you most likely do?', '{"institution_size":"medium","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-IN-01', 'A', 'This shows our departments are proactive about adopting technology. We support local autonomy in tool selection', 'Incidental', 1, true, 'Framing fragmentation as autonomy is the Incidental attractive nuisance. Uncoordinated duplication is not empowerment', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-IN-01', 'B', 'We''ve identified this as a problem and are developing a technology roadmap to rationalise and integrate our professional services systems', 'Intentional', 2, false, NULL, 'Developing a purposeful roadmap indicates movement toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-IN-01', 'C', 'This has happened because we don''t have an institutional technology strategy for professional services. Departments filled gaps independently', 'Incidental', 1, false, NULL, 'Honest recognition of uncoordinated adoption is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-IN-01', 'D', 'We have an approved technology roadmap and procurement policy requiring architectural review. New tools must align with our standards', 'Intentional', 2, false, NULL, 'An approved roadmap with procurement governance indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PST-IN-02 :: Professional Services: Technology :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PST-IN-02', 'maturity-the', 'the-ps-technology', 'Professional Services: Technology', 'incidental-intentional', 'Incidental', 'Intentional', 'IT at your university has been asked to prepare a technology investment case for professional services. When they attempt to map the current landscape, they find no documentation of what systems are in use across departments, who owns them, or how they connect.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-IN-02', 'A', 'We know our landscape is complex. We''ve commissioned an audit and will use it to develop a roadmap with integration priorities', 'Intentional', 2, false, NULL, 'Commissioning an audit to inform purposeful planning indicates movement toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-IN-02', 'B', 'We have a comprehensive, documented technology landscape with identified integration points and investment priorities', 'Intentional', 2, false, NULL, 'If true, this indicates Intentional or above', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-IN-02', 'C', 'Our systems work fine individually. We don''t really need a map because each department manages its own technology effectively', 'Incidental', 1, true, 'Defending fragmentation as departmental effectiveness is Incidental', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-IN-02', 'D', 'We recognise we have limited visibility of our technology landscape and it''s never been formally documented', 'Incidental', 1, false, NULL, 'Undocumented, unmanaged technology landscape is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PST-NI-01 :: Professional Services: Technology :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PST-NI-01', 'maturity-the', 'the-ps-technology', 'Professional Services: Technology', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university has invested in modernising its student records and CRM and migrated to cloud hosting. Single sign-on connects the main platforms. However, a review reveals that several key data flows between systems are manual, the architecture has no formal governance, and procurement still happens without architectural review in some faculties.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-NI-01', 'A', 'We''ve achieved integration. Our core systems are in the cloud with SSO and our main platforms are connected', 'Intentional', 2, true, 'Cloud hosting and SSO are positive steps but manual data flows, ungoverned architecture, and uncontrolled procurement indicate Intentional. The attractive nuisance is equating modernisation with integration', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-NI-01', 'B', 'We''ve made good progress on modernisation but we haven''t yet achieved full architectural governance and automated data flows across all systems', 'Intentional', 2, false, NULL, 'Recognising the gap between modernisation and integration accurately describes Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-NI-01', 'C', 'All our professional services systems are governed by enterprise architecture standards, connected through APIs, and procurement requires architectural review institution-wide', 'Integrated', 3, false, NULL, 'Comprehensive architecture governance with API integration indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-NI-01', 'D', 'We have a full integration platform with service level monitoring, planned refresh cycles, and joint IT-domain governance', 'Integrated', 3, false, NULL, 'Integration platform with comprehensive governance indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PST-NI-02 :: Professional Services: Technology :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PST-NI-02', 'maturity-the', 'the-ps-technology', 'Professional Services: Technology', 'intentional-integrated', 'Intentional', 'Integrated', 'All faculties at your university use the same student records and CRM. IT mandated this five years ago. However, faculties configure the system differently, there are no shared standards, and the system was chosen by IT without consulting domain experts. You are asked whether this represents integrated technology.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-NI-02', 'A', 'Absolutely. Everyone uses the same platform, which means our technology is integrated for professional services', 'Intentional', 2, true, 'Mandated shared platform without configuration standards, domain input, or architectural governance is technology standardisation not integration', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-NI-02', 'B', 'We have a shared platform but we recognise it was an IT-led decision without domain governance. We''re now establishing joint governance with configuration standards', 'Intentional', 2, false, NULL, 'Recognising the governance gap and working to address it describes Intentional moving toward Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-NI-02', 'C', 'Our shared platform has institution-wide configuration standards developed jointly by IT and domain stakeholders, with regular review', 'Integrated', 3, false, NULL, 'Jointly governed platform with shared standards indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-NI-02', 'D', 'The platform was chosen without faculty input and each faculty uses it differently. It creates as many problems as it solves', 'Intentional', 2, false, NULL, 'Mandated without governance is not integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PST-IO-01 :: Professional Services: Technology :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PST-IO-01', 'maturity-the', 'the-ps-technology', 'Professional Services: Technology', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has well-governed, integrated technology for professional services. A new technology emerges that could significantly enhance professional services capability. Several peer institutions are evaluating it. You are asked how to respond.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"research-intensive","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-IO-01', 'A', 'We should wait to see how peers implement it and learn from their experience before committing resources', 'Integrated', 3, true, 'Waiting to learn from peers is reactive, characteristic of Integrated. The attractive nuisance is that this feels prudent', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-IO-01', 'B', 'Our sandbox environment is already being used to evaluate this. Our technology futures panel assessed it three months ago and we have a pilot planned', 'Optimised', 4, false, NULL, 'Proactive assessment through established processes indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-IO-01', 'C', 'We should commission a thorough evaluation and develop a business case before proceeding', 'Integrated', 3, false, NULL, 'Thorough evaluation is good practice but reactive evaluation of already-visible technologies is Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-IO-01', 'D', 'We anticipated this development through our horizon scanning. We''ve already published a position paper and are advising sector bodies on implementation approaches', 'Optimised', 4, false, NULL, 'Anticipation through horizon scanning and sector leadership indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PST-IO-02 :: Professional Services: Technology :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PST-IO-02', 'maturity-the', 'the-ps-technology', 'Professional Services: Technology', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university achieves 99.5% uptime on its student records and CRM and has strong user satisfaction scores. A vendor invites you to present your technology approach at their annual conference as a customer success story. You are asked whether this means you are sector-leading.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-IO-02', 'A', 'High availability and vendor recognition confirms we are Optimised in technology for professional services', 'Integrated', 3, true, 'Reliability is Integrated. Vendor marketing invitations are not the same as sector-recognised innovation and leadership', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-IO-02', 'B', 'We''re well-run but reliable operations are table stakes. We need to ask whether our architecture enables innovation and whether we''re advancing practice beyond our own institution', 'Integrated', 3, false, NULL, 'Recognising that reliability alone is not sector leadership accurately assesses Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-IO-02', 'C', 'Our architecture is extensible, we have sandbox environments, we run regular technology innovation cycles, and peer institutions adopt our published architectural patterns', 'Optimised', 4, false, NULL, 'Innovation capacity and adopted patterns indicate Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PST-IO-02', 'D', 'We''re early adopters of new features and always among the first to upgrade to the latest version', 'Integrated', 3, true, 'Early adoption of vendor releases is not the same as architectural innovation and sector contribution', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSD-IN-01 :: Professional Services: Data :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSD-IN-01', 'maturity-the', 'the-ps-data', 'Professional Services: Data', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university needs to produce a report on digital professional services activity for a regulatory body. The data team discovers that relevant metrics are held in spreadsheets by individual departments, each using different definitions and formats. Compiling the report takes three weeks of manual work.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"UK"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-IN-01', 'A', 'We meet all our statutory reporting requirements. Our data management is adequate for external purposes', 'Incidental', 1, true, 'Meeting statutory requirements through manual compilation is Incidental. The attractive nuisance is equating compliance with data maturity', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-IN-01', 'B', 'We''ve identified our core service performance and operational datasets, standardised definitions, and started systematic collection to replace departmental spreadsheets', 'Intentional', 2, false, NULL, 'Standardised definitions and systematic collection indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-IN-01', 'C', 'Our service performance and operational data is fragmented across departments with no standard definitions. We rely on manual compilation for reporting', 'Incidental', 1, false, NULL, 'Fragmented data with manual compilation is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-IN-01', 'D', 'Our service performance and operational data is centrally managed with standardised definitions, automated collection, and dashboards. The report could be produced in hours', 'Intentional', 2, false, NULL, 'This describes at least Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSD-IN-02 :: Professional Services: Data :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSD-IN-02', 'maturity-the', 'the-ps-data', 'Professional Services: Data', 'incidental-intentional', 'Incidental', 'Intentional', 'A dean at your university wants to make a data-informed decision about digital investment in professional services. When they ask for relevant data, they are told it doesn''t exist in any centralised form and would need to be collected manually from multiple sources.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-IN-02', 'A', 'Our service performance and operational data is comprehensive and available through self-service dashboards', 'Intentional', 2, false, NULL, 'This would indicate at least Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-IN-02', 'B', 'We don''t currently have systematic data collection for professional services. We''re planning to implement it', 'Incidental', 1, false, NULL, 'No systematic collection is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-IN-02', 'C', 'We collect good data but it sits in different systems and teams. We''re working on integrating it into a central platform', 'Intentional', 2, false, NULL, 'Purposeful integration of existing data indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-IN-02', 'D', 'Deans should be able to make these decisions based on their professional judgment. They shouldn''t need a dashboard for everything', 'Incidental', 1, true, 'Dismissing the need for data-informed decisions is characteristic of Incidental. The attractive nuisance is framing this as valuing professional expertise', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSD-NI-01 :: Professional Services: Data :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSD-NI-01', 'maturity-the', 'the-ps-data', 'Professional Services: Data', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university has built dashboards for service performance and operational data and drafted a data governance policy. However, a data quality audit reveals significant inconsistencies: 30% of key fields have missing or incorrect data, governance compliance varies by department, and most committees still make decisions without consulting the available data.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-NI-01', 'A', 'We have dashboards and governance in place. Data quality will improve over time as people get used to the new systems', 'Intentional', 2, true, 'Dashboards and policy without quality management and actual use for decision-making is Intentional. The attractive nuisance is expecting passive improvement', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-NI-01', 'B', 'We''ve built the infrastructure but haven''t yet achieved reliable quality, consistent governance, or data-informed decision-making across the institution', 'Intentional', 2, false, NULL, 'Recognising the gap between infrastructure and institutional adoption accurately describes Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-NI-01', 'C', 'Our data quality is actively managed with regular audits, governance is operational with compliance monitoring, and committee papers routinely include data analysis', 'Integrated', 3, false, NULL, 'Active quality management, operational governance, and routine data use indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-NI-01', 'D', 'We need to prioritise data quality remediation and embed data use in committee processes before we can call our data mature', 'Intentional', 2, false, NULL, 'Identifying remediation needs accurately places the institution at Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSD-NI-02 :: Professional Services: Data :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSD-NI-02', 'maturity-the', 'the-ps-data', 'Professional Services: Data', 'intentional-integrated', 'Intentional', 'Integrated', 'The planning department at your university produces excellent service performance and operational reports. However, when you investigate, you find these reports are produced by a small specialist team. Faculty and department leaders do not have self-service access and must request custom reports each time. Data governance relies on the planning team''s expertise rather than institutional processes.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-NI-02', 'A', 'Our reporting is excellent and the planning team ensures data quality. This is an effective model', 'Intentional', 2, true, 'Expert-dependent reporting without self-service or institutional governance is Intentional. The attractive nuisance is that high-quality output feels like maturity', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-NI-02', 'B', 'We produce good reports but data capability is concentrated in one team. We need to democratise access and formalise governance institutionally', 'Intentional', 2, false, NULL, 'Expert dependency without distributed access is Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-NI-02', 'C', 'Self-service analytics are available to authorised users across the institution with institutional data governance ensuring quality', 'Integrated', 3, false, NULL, 'Distributed access with institutional governance indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-NI-02', 'D', 'Leaders can access data through an institutional analytics platform with training support and defined governance roles across all units', 'Integrated', 3, false, NULL, 'Institutional platform with governance indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSD-IO-01 :: Professional Services: Data :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSD-IO-01', 'maturity-the', 'the-ps-data', 'Professional Services: Data', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has comprehensive, well-governed service performance and operational data with institution-wide dashboards. A vendor offers an AI-powered predictive analytics tool. The COO is excited and wants to implement it immediately, claiming it will make the institution Optimised.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-IO-01', 'A', 'Implementing AI analytics on our well-governed data will make us sector-leading immediately', 'Integrated', 3, true, 'Tool adoption does not equal maturity. Optimised requires proven impact, continuous improvement, and sector contribution, not just tool acquisition', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-IO-01', 'B', 'We should pilot the tool, evaluate impact against outcomes, and publish our findings. If it works, we''ll iterate and share our methodology', 'Optimised', 4, false, NULL, 'Evidence-based evaluation with publication and continuous improvement indicates Optimised approach', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-IO-01', 'C', 'We already use predictive models validated against outcomes with documented impact. We''d evaluate this tool against our existing capabilities', 'Optimised', 4, false, NULL, 'Existing validated predictive capabilities with impact evidence indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-IO-01', 'D', 'We should implement it carefully with proper evaluation. AI tools need to be properly governed before deployment', 'Integrated', 3, false, NULL, 'Careful implementation with governance is good Integrated practice, not Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSD-IO-02 :: Professional Services: Data :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSD-IO-02', 'maturity-the', 'the-ps-data', 'Professional Services: Data', 'integrated-optimised', 'Integrated', 'Optimised', 'A sector body asks your university to contribute to developing new data standards for professional services. You currently have strong internal data governance but have not previously engaged with sector-level data practice.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-IO-02', 'A', 'We''d be delighted to contribute. Our data governance for professional services is strong internally and we''re ready to share our approaches', 'Optimised', 4, false, NULL, 'Willingness and readiness to contribute to sector standards indicates movement toward Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-IO-02', 'B', 'We''re confident our internal data practice is good but we should focus on maintaining what we have rather than taking on sector work', 'Integrated', 3, false, NULL, 'Internal focus without sector contribution is Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-IO-02', 'C', 'We''ve been contributing to sector data standards for several years and our governance framework has been adopted by three peer institutions', 'Optimised', 4, false, NULL, 'Sustained contribution with peer adoption indicates established Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSD-IO-02', 'D', 'We benchmark our data governance against the sector body''s existing standards to ensure we meet best practice', 'Integrated', 3, true, 'Benchmarking against others'' standards is consuming not contributing. The attractive nuisance is that benchmarking feels like sector engagement', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSU-IN-01 :: Professional Services: Utilisation :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSU-IN-01', 'maturity-the', 'the-ps-utilization', 'Professional Services: Utilisation', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university invested in self-service portals, automated workflows, and CRM two years ago. Usage data (where available) shows that only 30% of staff and service users regularly use the core features, and fewer than 10% use advanced capabilities. Most staff and service users continue with previous manual or paper-based approaches.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-IN-01', 'A', 'We''ve made the investment and the tools are available. People will adopt at their own pace', 'Incidental', 1, true, 'Availability without promotion of adoption is Incidental. The attractive nuisance is framing passive availability as a strategy', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-IN-01', 'B', 'We''ve launched a training programme with minimum usage expectations and we''re tracking adoption rates institution-wide', 'Intentional', 2, false, NULL, 'Purposeful promotion with training and tracking indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-IN-01', 'C', 'Adoption is low and we don''t have a plan to address it. The tools are there but people haven''t taken to them', 'Incidental', 1, false, NULL, 'Unaddressed low adoption is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-IN-01', 'D', 'Usage is high and consistent. Over 80% of staff and service users regularly use both basic and advanced features', 'Intentional', 2, false, NULL, 'This would indicate at least Intentional but contradicts the scenario', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSU-IN-02 :: Professional Services: Utilisation :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSU-IN-02', 'maturity-the', 'the-ps-utilization', 'Professional Services: Utilisation', 'incidental-intentional', 'Incidental', 'Intentional', 'A few staff and service users at your university have developed impressive workflows using self-service portals, automated workflows, and CRM and have been nominated for an innovation award. Meanwhile, their immediate colleagues continue using older methods. No one has been asked to adopt the innovative approaches more widely.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-IN-02', 'A', 'Our award nominees demonstrate excellent utilisation across the institution', 'Incidental', 1, true, 'Individual excellence is the Incidental pattern. The attractive nuisance is pointing to champions as evidence of institutional utilisation', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-IN-02', 'B', 'We''ve identified these innovators and are using their approaches to develop institutional training and minimum standards for all staff and service users', 'Intentional', 2, false, NULL, 'Converting individual innovation into institutional programme indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-IN-02', 'C', 'These individuals found their own way. We haven''t yet developed an institutional approach to promoting consistent utilisation', 'Incidental', 1, false, NULL, 'Individual adoption without institutional promotion is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-IN-02', 'D', 'We have minimum standards for tool usage and consistent adoption is monitored across the institution', 'Intentional', 2, false, NULL, 'Institutional standards with monitoring indicates at least Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSU-NI-01 :: Professional Services: Utilisation :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSU-NI-01', 'maturity-the', 'the-ps-utilization', 'Professional Services: Utilisation', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university has been actively promoting adoption of self-service portals, automated workflows, and CRM with training programmes and published expectations. Adoption has risen to 65% for basic features, with three faculties at over 80% and two below 40%. Impact on outcomes is not measured.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-NI-01', 'A', 'Adoption is growing strongly. We''re approaching integrated utilisation across the institution', 'Intentional', 2, true, 'Significant variation (40-80%) across faculties and no impact measurement is Intentional. The attractive nuisance is that average growth masks inconsistency', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-NI-01', 'B', 'Adoption is growing but not yet consistent. We need institution-wide standards, measurement of outcomes, and intervention in lagging areas', 'Intentional', 2, false, NULL, 'Inconsistent adoption without impact measurement is Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-NI-01', 'C', 'Adoption is consistently above 80% across all units, we measure impact on outcomes, and user feedback informs tool optimisation', 'Integrated', 3, false, NULL, 'Consistent high adoption with impact measurement indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-NI-01', 'D', 'All core professional services processes run on digital workflows and we can demonstrate improvement in outcomes attributable to tool utilisation', 'Integrated', 3, false, NULL, 'Digital workflows as standard with demonstrated impact indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSU-NI-02 :: Professional Services: Utilisation :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSU-NI-02', 'maturity-the', 'the-ps-utilization', 'Professional Services: Utilisation', 'intentional-integrated', 'Intentional', 'Integrated', 'Several departments at your university report that they have ''gone fully digital'' for professional services processes. However, an audit reveals that while forms are digital, they are printed out for review, approvals happen by email rather than through the system, and data entry is duplicated across platforms.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-NI-02', 'A', 'We''ve digitised our processes. All our forms and records are digital now', 'Intentional', 2, true, 'Digitising the form without digitising the workflow is Intentional. The attractive nuisance is equating digital forms with digital processes', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-NI-02', 'B', 'We''ve digitised inputs but not workflows. True utilisation means end-to-end digital processes without manual intervention', 'Intentional', 2, false, NULL, 'Recognising partial digitisation accurately describes Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-NI-02', 'C', 'Our processes run end-to-end digitally with no paper fallbacks, automated routing, and single data entry', 'Integrated', 3, false, NULL, 'End-to-end digital workflows indicate Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-NI-02', 'D', 'We''ve not digitised our processes yet. Most work is still paper-based', 'Incidental', 1, false, NULL, 'This understates the scenario which describes partial digitisation', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSU-IO-01 :: Professional Services: Utilisation :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSU-IO-01', 'maturity-the', 'the-ps-utilization', 'Professional Services: Utilisation', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has consistent, high utilisation of self-service portals, automated workflows, and CRM across all units. A vendor approaches you asking to feature your institution as a case study for how well you use their product. You are asked whether this means your utilisation is Optimised.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-IO-01', 'A', 'Vendor recognition confirms we are sector-leading in utilisation of professional services tools', 'Integrated', 3, true, 'Vendor case studies are marketing tools, not independent assessment. Consistent usage of a product as intended is Integrated, not innovation', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-IO-01', 'B', 'We use the tools well as designed. But Optimised utilisation means our users innovate new use cases and drive continuous improvement beyond standard deployment', 'Integrated', 3, false, NULL, 'Distinguishing standard effective use from innovation-driven optimisation is accurate', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-IO-01', 'C', 'Our users have developed novel applications of these tools that the vendor has incorporated into their product roadmap. We continuously optimise based on usage analytics', 'Optimised', 4, false, NULL, 'User innovation influencing vendor roadmaps and continuous optimisation indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-IO-01', 'D', 'We regularly present at user conferences sharing innovative workflows our staff and service users have developed, and peer institutions adopt our configurations', 'Optimised', 4, false, NULL, 'Innovative workflows adopted by peers indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PSU-IO-02 :: Professional Services: Utilisation :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PSU-IO-02', 'maturity-the', 'the-ps-utilization', 'Professional Services: Utilisation', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university monitors utilisation of self-service portals, automated workflows, and CRM through monthly reports. Adoption is consistently above 85% for core features. Users report high satisfaction. A review asks whether there is anything more to achieve.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-IO-02', 'A', 'We''ve achieved consistent high utilisation. We should maintain current levels and focus investment elsewhere', 'Integrated', 3, true, 'Maintenance of current utilisation is Integrated. Optimised means continuous improvement and innovation, not steady-state management', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-IO-02', 'B', 'We should move from monitoring adoption to analysing usage patterns to identify optimisation opportunities and measuring impact on outcomes', 'Optimised', 4, false, NULL, 'Moving from adoption monitoring to optimisation and impact measurement indicates Optimised thinking', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-IO-02', 'C', 'Our users already drive innovation. We analyse usage patterns in real-time, users contribute novel workflows, and we quantify impact on outcomes', 'Optimised', 4, false, NULL, 'User-driven innovation with analytics and impact quantification indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PSU-IO-02', 'D', 'We should push adoption of advanced features to increase the 85% further', 'Integrated', 3, false, NULL, 'Pursuing higher adoption of existing features is still an Integrated activity, not innovation-driven optimisation', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGS-IN-01 :: Planning & Governance: Strategy :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGS-IN-01', 'maturity-the', 'the-pg-strategy', 'Planning & Governance: Strategy', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university has been responding to digital demands in planning and governance on a case-by-case basis. Several departments have adopted different tools independently. The senior leadership team has recently discussed the need for a more coordinated approach. A deputy vice-chancellor asks you: ''Where are we on digital strategy for planning and governance?''', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-IN-01', 'A', 'We have a clear digital strategy for planning and governance that was approved last year and is being implemented across all faculties', 'Intentional', 2, false, NULL, 'This would indicate Intentional if true, but the scenario describes no approved strategy', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-IN-01', 'B', 'We recognise the need and are actively developing a digital strategy for planning and governance with identified priorities and a governance proposal', 'Intentional', 2, false, NULL, 'Active development of a purposeful strategy with governance indicates transition toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-IN-01', 'C', 'We''ve been meaning to write a strategy but haven''t found the time. Meanwhile, departments are managing things in their own way', 'Incidental', 1, false, NULL, 'Acknowledged need without action and devolved ad-hoc activity is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-IN-01', 'D', 'Our institutional strategic plan mentions digital transformation and we reference that when departments ask for guidance', 'Incidental', 1, true, 'A passing mention in a broader strategy without specific objectives, owners, or resources for planning and governance is not a purposeful strategy. This is the ''we have a plan'' attractive nuisance', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGS-IN-02 :: Planning & Governance: Strategy :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGS-IN-02', 'maturity-the', 'the-pg-strategy', 'Planning & Governance: Strategy', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university''s board has asked for an update on how digital technology supports planning and governance. You discover that while several successful digital initiatives exist across the institution, they were each initiated by individual champions with no central coordination. The board wants to know what the institutional approach is.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-IN-02', 'A', 'We have a coordinated institutional approach with a strategy document, dedicated budget, and a committee overseeing digital in planning and governance', 'Intentional', 2, false, NULL, 'This describes Intentional with formal strategy, resources, and governance', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-IN-02', 'B', 'We have some excellent initiatives and we''re now developing a formal strategy to bring them together under a coherent plan', 'Intentional', 2, false, NULL, 'Transitioning from ad-hoc to purposeful planning indicates movement toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-IN-02', 'C', 'We have a lot of innovative activity happening organically. Our approach is to let a thousand flowers bloom and learn from what works', 'Incidental', 1, true, 'Framing lack of strategy as deliberate emergent innovation is a common attractive nuisance. Organic activity without coordination is Incidental', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-IN-02', 'D', 'We don''t really have an institutional approach yet. Individual departments have done their own thing based on local needs', 'Incidental', 1, false, NULL, 'Honest acknowledgement of no institutional approach is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGS-NI-01 :: Planning & Governance: Strategy :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGS-NI-01', 'maturity-the', 'the-pg-strategy', 'Planning & Governance: Strategy', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university approved a digital planning and governance strategy 18 months ago. The strategy has clear objectives and a steering group meets quarterly. However, implementation varies dramatically across faculties. Two faculties are well advanced, three have barely started, and the remaining faculties fall somewhere in between. A new PVC asks how well the strategy is being implemented.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-NI-01', 'A', 'The strategy is fully embedded. All faculties have adopted it and are implementing it consistently with local adaptation', 'Integrated', 3, false, NULL, 'This would indicate Integrated if true, but the scenario contradicts this', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-NI-01', 'B', 'We have strong pockets of implementation and we''re working to bring all faculties up to the standard of our leading areas', 'Intentional', 2, true, '''Strong pockets'' with inconsistent implementation across the institution is characteristic of Intentional. The attractive nuisance is that activity in multiple locations feels like integration', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-NI-01', 'C', 'Implementation is uneven. We have the strategy but we haven''t yet achieved consistent cross-institutional adoption with proper governance and accountability', 'Intentional', 2, false, NULL, 'Honest assessment of uneven implementation describes Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-NI-01', 'D', 'All faculties have operational plans that reference the institutional strategy, with locally adapted targets, and we report on progress to the board termly', 'Integrated', 3, false, NULL, 'Faculty-level plans aligned to institutional strategy with regular reporting indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGS-NI-02 :: Planning & Governance: Strategy :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGS-NI-02', 'maturity-the', 'the-pg-strategy', 'Planning & Governance: Strategy', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university has invested in a central digital planning and governance team of six staff who develop and support digital initiatives. The team runs projects across faculties, but each project requires negotiation with individual faculty leaders for access and cooperation. The team reports to the Vice-Chancellor but has no formal authority over faculty-level decisions.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-NI-02', 'A', 'Having a dedicated central team with a clear reporting line shows we have an integrated approach to digital planning and governance', 'Integrated', 3, true, 'A central team without governance authority over faculties and without faculty-level adoption of institutional strategy operates in an Intentional model. The attractive nuisance is equating central team existence with integration', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-NI-02', 'B', 'We have purposeful investment in digital planning and governance but the central team operates somewhat independently from faculty planning. We need stronger governance to achieve integration', 'Intentional', 2, false, NULL, 'Purposeful investment without cross-institutional governance is accurately Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-NI-02', 'C', 'The central team''s work is governed by a cross-institutional board with faculty representation, and faculty plans are required to align with the digital strategy', 'Integrated', 3, false, NULL, 'If this were true it would indicate Integrated, but the scenario describes a team negotiating cooperation', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-NI-02', 'D', 'Our digital planning and governance activity is still at an early stage with no real coordination between the centre and faculties', 'Incidental', 1, false, NULL, 'This understates the scenario, which does describe purposeful investment', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGS-IO-01 :: Planning & Governance: Strategy :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGS-IO-01', 'maturity-the', 'the-pg-strategy', 'Planning & Governance: Strategy', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has a well-functioning digital planning and governance strategy that is implemented consistently across all faculties. Governance is strong, KPIs are reported regularly, and the institution has achieved solid results. The Vice-Chancellor wants to know whether the institution can now be considered sector-leading.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"research-intensive","region":"UK"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-IO-01', 'A', 'We regularly benchmark against Russell Group peers and ensure our approach matches best practice. We''ve adopted several approaches we learned from peer institutions', 'Integrated', 3, true, 'Benchmarking by adopting approaches from others is characteristic of Integrated. The attractive nuisance is that benchmarking activity feels like sector leadership, but copying good practice is not the same as setting it', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-IO-01', 'B', 'We review our strategy annually based on data, run a horizon scanning function, publish our approaches, and other institutions regularly visit to learn from us', 'Optimised', 4, false, NULL, 'Evidence-based annual review, horizon scanning, publication, and peer recognition indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-IO-01', 'C', 'Our implementation is strong and consistent but we recognise we''re implementing established good practice rather than innovating ahead of the sector', 'Integrated', 3, false, NULL, 'Honest assessment of implementing good practice without leading innovation is Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-IO-01', 'D', 'We''ve won a national award for one of our digital planning and governance initiatives and were featured in a trade publication', 'Integrated', 3, true, 'A single award for a specific initiative does not indicate systematic sector leadership. The attractive nuisance is equating one recognition with Optimised maturity', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGS-IO-02 :: Planning & Governance: Strategy :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGS-IO-02', 'maturity-the', 'the-pg-strategy', 'Planning & Governance: Strategy', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has strong digital planning and governance governance and consistent implementation. The sector is now grappling with a new technological development that could significantly impact planning and governance. Several peer institutions are starting to respond.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-IO-02', 'A', 'We formed an expert panel six months ago to assess the implications and have already piloted approaches. We published a briefing paper that three peer institutions have since adopted', 'Optimised', 4, false, NULL, 'Proactive assessment, early piloting, and sector contribution through publication indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-IO-02', 'B', 'We''re watching what leading institutions are doing and plan to adopt best practice once it becomes clearer what works', 'Integrated', 3, true, 'Waiting to adopt others'' practices is reactive benchmarking, characteristic of Integrated. The attractive nuisance is that this feels prudent and strategic', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-IO-02', 'C', 'We''ve established a task force to develop our response and are developing a pilot programme informed by sector guidance', 'Integrated', 3, false, NULL, 'Developing a response informed by others indicates Integrated responding effectively, not Optimised leading', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGS-IO-02', 'D', 'We''ve been preparing for this for over a year through our horizon scanning process and have an institutional position and implementation plan ready', 'Optimised', 4, false, NULL, 'Anticipating the development through horizon scanning indicates Optimised proactive strategic maturity', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGP-IN-01 :: Planning & Governance: People & Culture :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGP-IN-01', 'maturity-the', 'the-pg-people', 'Planning & Governance: People & Culture', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university is assessing digital skills readiness among senior leaders and planning professionals. A survey reveals that 15% of senior leaders and planning professionals are highly digitally capable and actively innovating, while 60% use basic digital tools but avoid anything more advanced. The remaining 25% actively resist using digital tools. No institutional development programme for digital skills in planning and governance exists.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-IN-01', 'A', 'We clearly have digital capability. 15% of our senior leaders and planning professionals are highly skilled and leading innovation in planning and governance', 'Incidental', 1, true, 'Concentrated expertise in a small minority without institutional development is the hallmark of Incidental. The attractive nuisance is celebrating individual champions as institutional capability', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-IN-01', 'B', 'We need to start investing in digital skills for senior leaders and planning professionals. We''re planning a development programme targeting the 60% in the middle', 'Intentional', 2, false, NULL, 'Planning purposeful investment in workforce development indicates movement toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-IN-01', 'C', 'We don''t have an institutional programme. Skills are developed through informal peer learning and self-study', 'Incidental', 1, false, NULL, 'Informal, self-directed development without institutional programme is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-IN-01', 'D', 'We''ve launched a targeted digital capabilities programme for senior leaders and planning professionals with modules mapped to role requirements, and we''re tracking participation', 'Intentional', 2, false, NULL, 'A targeted, tracked programme indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGP-IN-02 :: Planning & Governance: People & Culture :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGP-IN-02', 'maturity-the', 'the-pg-people', 'Planning & Governance: People & Culture', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university is recruiting for a senior role in planning and governance. The job description does not mention digital competencies. The hiring manager says: ''Digital skills aren''t really relevant for this role. They just need to be good at the core functions.'' Meanwhile, a departing staff member who was a digital champion leaves a significant gap in the team''s digital capability.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-IN-02', 'A', 'We recognise this is a gap. We''ve started reviewing all planning and governance role profiles to include digital competency requirements', 'Intentional', 2, false, NULL, 'Purposeful review of role profiles to embed digital requirements indicates movement toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-IN-02', 'B', 'Digital skills are important but they''re something people develop on the job. We don''t need to specify them in recruitment', 'Incidental', 1, true, 'Assuming digital skills will develop organically is the Incidental pattern. The attractive nuisance is that this sounds like a reasonable, flexible approach', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-IN-02', 'C', 'All our planning and governance role profiles now include digital competencies and we assess them at interview', 'Intentional', 2, false, NULL, 'Embedding digital in recruitment indicates Intentional or above', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-IN-02', 'D', 'We don''t currently include digital skills in role profiles for planning and governance positions', 'Incidental', 1, false, NULL, 'Absence of digital in role profiles is characteristic of Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGP-NI-01 :: Planning & Governance: People & Culture :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGP-NI-01', 'maturity-the', 'the-pg-people', 'Planning & Governance: People & Culture', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university has been running digital skills training for senior leaders and planning professionals for two years. Attendance is good in some departments but poor in others. A few departments have transformed their practice while most continue as before. Performance reviews do not assess digital capability. You are asked whether digital skills development is working.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-NI-01', 'A', 'Absolutely. Training attendance is strong and we''ve seen real transformation in our leading departments', 'Intentional', 2, true, 'Training attendance with patchy uptake and no integration into performance management is Intentional. The attractive nuisance is pointing to leading departments as evidence of institutional maturity', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-NI-01', 'B', 'We''ve made a good start but digital competencies aren''t yet embedded in performance review, career pathways, or promotion criteria across the institution', 'Intentional', 2, false, NULL, 'Honest recognition that development is not yet embedded in HR processes describes Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-NI-01', 'C', 'Digital competencies are integrated into our PDR process. Development pathways exist at every career stage. Innovation is recognised in promotion criteria', 'Integrated', 3, false, NULL, 'Embedded in HR processes and career pathways indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-NI-01', 'D', 'We''ve mandated digital training completion for all senior leaders and planning professionals and track compliance centrally', 'Intentional', 2, true, 'Mandatory compliance training can generate resentment rather than culture change. Mandated attendance without HR embedding is still Intentional', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGP-NI-02 :: Planning & Governance: People & Culture :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGP-NI-02', 'maturity-the', 'the-pg-people', 'Planning & Governance: People & Culture', 'intentional-integrated', 'Intentional', 'Integrated', 'A head of department at your university approaches you saying their team needs more digital skills support. They''ve been relying on one team member who is the ''digital person'' for all technology-related work. When that person is on leave, digital projects stall. They ask what the institution offers.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-NI-02', 'A', 'We have a comprehensive programme but capability tends to concentrate in enthusiasts. We haven''t yet built distributed competence across all teams', 'Intentional', 2, false, NULL, 'Champion-dependency describes Intentional even with good programmes', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-NI-02', 'B', 'Our competency framework ensures all staff develop digital skills as part of their role. No team should depend on a single person', 'Integrated', 3, false, NULL, 'Institutional competency framework preventing single-person dependency indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-NI-02', 'C', 'That''s exactly what our community of practice and digital champions network is designed to address. We support knowledge sharing across teams', 'Intentional', 2, true, 'A champions network can address symptoms but not root cause. If capability remains concentrated in designated champions, this is Intentional', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-NI-02', 'D', 'We have some training available. I''d suggest sending a couple of their team on the next session', 'Intentional', 2, false, NULL, 'Ad-hoc training referral without systematic capability building is Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGP-IO-01 :: Planning & Governance: People & Culture :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGP-IO-01', 'maturity-the', 'the-pg-people', 'Planning & Governance: People & Culture', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has invested heavily in developing digital capabilities across senior leaders and planning professionals. Competency frameworks are embedded in HR processes, communities of practice are active, and satisfaction is high. You are asked whether the institution should now focus elsewhere or continue investing.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-IO-01', 'A', 'Our development programmes are comprehensive and well-attended. We can now focus investment elsewhere and maintain current provision', 'Integrated', 3, true, 'Good institutional provision maintained centrally is Integrated. Optimised culture is self-sustaining and peer-driven, not dependent on continued central provision', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-IO-01', 'B', 'Development has become largely self-sustaining. Staff drive their own learning and peer development. We should invest in supporting that culture, not controlling it', 'Optimised', 4, false, NULL, 'Self-sustaining, peer-driven development culture indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-IO-01', 'C', 'We''re well-established internally. We should now contribute to sector workforce development by sharing our frameworks and offering training to other institutions', 'Optimised', 4, false, NULL, 'Sector contribution indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-IO-01', 'D', 'We need to maintain our investment. Without continued institutional programmes, capability will erode', 'Integrated', 3, false, NULL, 'Dependency on institutional programmes for capability maintenance indicates Integrated not Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGP-IO-02 :: Planning & Governance: People & Culture :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGP-IO-02', 'maturity-the', 'the-pg-people', 'Planning & Governance: People & Culture', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university is known for strong digital skills among senior leaders and planning professionals. A peer institution contacts you asking to learn from your approach. You also notice that your digital innovation often follows trends set by two or three leading institutions rather than originating internally.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-IO-02', 'A', 'We''re happy to share our approach. We''ve developed it by carefully studying and adapting best practice from leading institutions', 'Integrated', 3, true, 'Adapting others'' best practice is Integrated. The attractive nuisance is that sharing your adapted approach feels like leadership', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-IO-02', 'B', 'We generate original approaches that others adopt. Our staff regularly publish and present on novel digital practices they''ve developed', 'Optimised', 4, false, NULL, 'Originating novel approaches adopted by peers indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-IO-02', 'C', 'We''re strong implementers of established good practice. We aren''t really generating new approaches that others follow', 'Integrated', 3, false, NULL, 'Honest recognition of implementing vs leading is Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGP-IO-02', 'D', 'Our approach is distinctive and recognised. We contribute to sector thinking through advisory roles and published research on digital workforce development', 'Optimised', 4, false, NULL, 'Sustained sector contribution through advisory and publication indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGT-IN-01 :: Planning & Governance: Technology :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGT-IN-01', 'maturity-the', 'the-pg-technology', 'Planning & Governance: Technology', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university recently discovered that five departments are using three different tools for essentially the same planning and governance function. No one made a decision to standardise, and each department chose independently. IT maintains all three but has raised concerns about sustainability.', 'What would you most likely do?', '{"institution_size":"medium","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-IN-01', 'A', 'This shows our departments are proactive about adopting technology. We support local autonomy in tool selection', 'Incidental', 1, true, 'Framing fragmentation as autonomy is the Incidental attractive nuisance. Uncoordinated duplication is not empowerment', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-IN-01', 'B', 'We''ve identified this as a problem and are developing a technology roadmap to rationalise and integrate our planning and governance systems', 'Intentional', 2, false, NULL, 'Developing a purposeful roadmap indicates movement toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-IN-01', 'C', 'This has happened because we don''t have an institutional technology strategy for planning and governance. Departments filled gaps independently', 'Incidental', 1, false, NULL, 'Honest recognition of uncoordinated adoption is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-IN-01', 'D', 'We have an approved technology roadmap and procurement policy requiring architectural review. New tools must align with our standards', 'Intentional', 2, false, NULL, 'An approved roadmap with procurement governance indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGT-IN-02 :: Planning & Governance: Technology :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGT-IN-02', 'maturity-the', 'the-pg-technology', 'Planning & Governance: Technology', 'incidental-intentional', 'Incidental', 'Intentional', 'IT at your university has been asked to prepare a technology investment case for planning and governance. When they attempt to map the current landscape, they find no documentation of what systems are in use across departments, who owns them, or how they connect.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-IN-02', 'A', 'We know our landscape is complex. We''ve commissioned an audit and will use it to develop a roadmap with integration priorities', 'Intentional', 2, false, NULL, 'Commissioning an audit to inform purposeful planning indicates movement toward Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-IN-02', 'B', 'We have a comprehensive, documented technology landscape with identified integration points and investment priorities', 'Intentional', 2, false, NULL, 'If true, this indicates Intentional or above', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-IN-02', 'C', 'Our systems work fine individually. We don''t really need a map because each department manages its own technology effectively', 'Incidental', 1, true, 'Defending fragmentation as departmental effectiveness is Incidental', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-IN-02', 'D', 'We recognise we have limited visibility of our technology landscape and it''s never been formally documented', 'Incidental', 1, false, NULL, 'Undocumented, unmanaged technology landscape is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGT-NI-01 :: Planning & Governance: Technology :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGT-NI-01', 'maturity-the', 'the-pg-technology', 'Planning & Governance: Technology', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university has invested in modernising its business intelligence platform and migrated to cloud hosting. Single sign-on connects the main platforms. However, a review reveals that several key data flows between systems are manual, the architecture has no formal governance, and procurement still happens without architectural review in some faculties.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-NI-01', 'A', 'We''ve achieved integration. Our core systems are in the cloud with SSO and our main platforms are connected', 'Intentional', 2, true, 'Cloud hosting and SSO are positive steps but manual data flows, ungoverned architecture, and uncontrolled procurement indicate Intentional. The attractive nuisance is equating modernisation with integration', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-NI-01', 'B', 'We''ve made good progress on modernisation but we haven''t yet achieved full architectural governance and automated data flows across all systems', 'Intentional', 2, false, NULL, 'Recognising the gap between modernisation and integration accurately describes Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-NI-01', 'C', 'All our planning and governance systems are governed by enterprise architecture standards, connected through APIs, and procurement requires architectural review institution-wide', 'Integrated', 3, false, NULL, 'Comprehensive architecture governance with API integration indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-NI-01', 'D', 'We have a full integration platform with service level monitoring, planned refresh cycles, and joint IT-domain governance', 'Integrated', 3, false, NULL, 'Integration platform with comprehensive governance indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGT-NI-02 :: Planning & Governance: Technology :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGT-NI-02', 'maturity-the', 'the-pg-technology', 'Planning & Governance: Technology', 'intentional-integrated', 'Intentional', 'Integrated', 'All faculties at your university use the same business intelligence platform. IT mandated this five years ago. However, faculties configure the system differently, there are no shared standards, and the system was chosen by IT without consulting domain experts. You are asked whether this represents integrated technology.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-NI-02', 'A', 'Absolutely. Everyone uses the same platform, which means our technology is integrated for planning and governance', 'Intentional', 2, true, 'Mandated shared platform without configuration standards, domain input, or architectural governance is technology standardisation not integration', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-NI-02', 'B', 'We have a shared platform but we recognise it was an IT-led decision without domain governance. We''re now establishing joint governance with configuration standards', 'Intentional', 2, false, NULL, 'Recognising the governance gap and working to address it describes Intentional moving toward Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-NI-02', 'C', 'Our shared platform has institution-wide configuration standards developed jointly by IT and domain stakeholders, with regular review', 'Integrated', 3, false, NULL, 'Jointly governed platform with shared standards indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-NI-02', 'D', 'The platform was chosen without faculty input and each faculty uses it differently. It creates as many problems as it solves', 'Intentional', 2, false, NULL, 'Mandated without governance is not integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGT-IO-01 :: Planning & Governance: Technology :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGT-IO-01', 'maturity-the', 'the-pg-technology', 'Planning & Governance: Technology', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has well-governed, integrated technology for planning and governance. A new technology emerges that could significantly enhance planning and governance capability. Several peer institutions are evaluating it. You are asked how to respond.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"research-intensive","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-IO-01', 'A', 'We should wait to see how peers implement it and learn from their experience before committing resources', 'Integrated', 3, true, 'Waiting to learn from peers is reactive, characteristic of Integrated. The attractive nuisance is that this feels prudent', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-IO-01', 'B', 'Our sandbox environment is already being used to evaluate this. Our technology futures panel assessed it three months ago and we have a pilot planned', 'Optimised', 4, false, NULL, 'Proactive assessment through established processes indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-IO-01', 'C', 'We should commission a thorough evaluation and develop a business case before proceeding', 'Integrated', 3, false, NULL, 'Thorough evaluation is good practice but reactive evaluation of already-visible technologies is Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-IO-01', 'D', 'We anticipated this development through our horizon scanning. We''ve already published a position paper and are advising sector bodies on implementation approaches', 'Optimised', 4, false, NULL, 'Anticipation through horizon scanning and sector leadership indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGT-IO-02 :: Planning & Governance: Technology :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGT-IO-02', 'maturity-the', 'the-pg-technology', 'Planning & Governance: Technology', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university achieves 99.5% uptime on its business intelligence platform and has strong user satisfaction scores. A vendor invites you to present your technology approach at their annual conference as a customer success story. You are asked whether this means you are sector-leading.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-IO-02', 'A', 'High availability and vendor recognition confirms we are Optimised in technology for planning and governance', 'Integrated', 3, true, 'Reliability is Integrated. Vendor marketing invitations are not the same as sector-recognised innovation and leadership', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-IO-02', 'B', 'We''re well-run but reliable operations are table stakes. We need to ask whether our architecture enables innovation and whether we''re advancing practice beyond our own institution', 'Integrated', 3, false, NULL, 'Recognising that reliability alone is not sector leadership accurately assesses Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-IO-02', 'C', 'Our architecture is extensible, we have sandbox environments, we run regular technology innovation cycles, and peer institutions adopt our published architectural patterns', 'Optimised', 4, false, NULL, 'Innovation capacity and adopted patterns indicate Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGT-IO-02', 'D', 'We''re early adopters of new features and always among the first to upgrade to the latest version', 'Integrated', 3, true, 'Early adoption of vendor releases is not the same as architectural innovation and sector contribution', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGD-IN-01 :: Planning & Governance: Data :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGD-IN-01', 'maturity-the', 'the-pg-data', 'Planning & Governance: Data', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university needs to produce a report on digital planning and governance activity for a regulatory body. The data team discovers that relevant metrics are held in spreadsheets by individual departments, each using different definitions and formats. Compiling the report takes three weeks of manual work.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"UK"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-IN-01', 'A', 'We meet all our statutory reporting requirements. Our data management is adequate for external purposes', 'Incidental', 1, true, 'Meeting statutory requirements through manual compilation is Incidental. The attractive nuisance is equating compliance with data maturity', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-IN-01', 'B', 'We''ve identified our core institutional performance and planning datasets, standardised definitions, and started systematic collection to replace departmental spreadsheets', 'Intentional', 2, false, NULL, 'Standardised definitions and systematic collection indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-IN-01', 'C', 'Our institutional performance and planning data is fragmented across departments with no standard definitions. We rely on manual compilation for reporting', 'Incidental', 1, false, NULL, 'Fragmented data with manual compilation is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-IN-01', 'D', 'Our institutional performance and planning data is centrally managed with standardised definitions, automated collection, and dashboards. The report could be produced in hours', 'Intentional', 2, false, NULL, 'This describes at least Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGD-IN-02 :: Planning & Governance: Data :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGD-IN-02', 'maturity-the', 'the-pg-data', 'Planning & Governance: Data', 'incidental-intentional', 'Incidental', 'Intentional', 'A dean at your university wants to make a data-informed decision about digital investment in planning and governance. When they ask for relevant data, they are told it doesn''t exist in any centralised form and would need to be collected manually from multiple sources.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-IN-02', 'A', 'Our institutional performance and planning data is comprehensive and available through self-service dashboards', 'Intentional', 2, false, NULL, 'This would indicate at least Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-IN-02', 'B', 'We don''t currently have systematic data collection for planning and governance. We''re planning to implement it', 'Incidental', 1, false, NULL, 'No systematic collection is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-IN-02', 'C', 'We collect good data but it sits in different systems and teams. We''re working on integrating it into a central platform', 'Intentional', 2, false, NULL, 'Purposeful integration of existing data indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-IN-02', 'D', 'Deans should be able to make these decisions based on their professional judgment. They shouldn''t need a dashboard for everything', 'Incidental', 1, true, 'Dismissing the need for data-informed decisions is characteristic of Incidental. The attractive nuisance is framing this as valuing professional expertise', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGD-NI-01 :: Planning & Governance: Data :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGD-NI-01', 'maturity-the', 'the-pg-data', 'Planning & Governance: Data', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university has built dashboards for institutional performance and planning data and drafted a data governance policy. However, a data quality audit reveals significant inconsistencies: 30% of key fields have missing or incorrect data, governance compliance varies by department, and most committees still make decisions without consulting the available data.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-NI-01', 'A', 'We have dashboards and governance in place. Data quality will improve over time as people get used to the new systems', 'Intentional', 2, true, 'Dashboards and policy without quality management and actual use for decision-making is Intentional. The attractive nuisance is expecting passive improvement', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-NI-01', 'B', 'We''ve built the infrastructure but haven''t yet achieved reliable quality, consistent governance, or data-informed decision-making across the institution', 'Intentional', 2, false, NULL, 'Recognising the gap between infrastructure and institutional adoption accurately describes Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-NI-01', 'C', 'Our data quality is actively managed with regular audits, governance is operational with compliance monitoring, and committee papers routinely include data analysis', 'Integrated', 3, false, NULL, 'Active quality management, operational governance, and routine data use indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-NI-01', 'D', 'We need to prioritise data quality remediation and embed data use in committee processes before we can call our data mature', 'Intentional', 2, false, NULL, 'Identifying remediation needs accurately places the institution at Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGD-NI-02 :: Planning & Governance: Data :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGD-NI-02', 'maturity-the', 'the-pg-data', 'Planning & Governance: Data', 'intentional-integrated', 'Intentional', 'Integrated', 'The planning department at your university produces excellent institutional performance and planning reports. However, when you investigate, you find these reports are produced by a small specialist team. Faculty and department leaders do not have self-service access and must request custom reports each time. Data governance relies on the planning team''s expertise rather than institutional processes.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-NI-02', 'A', 'Our reporting is excellent and the planning team ensures data quality. This is an effective model', 'Intentional', 2, true, 'Expert-dependent reporting without self-service or institutional governance is Intentional. The attractive nuisance is that high-quality output feels like maturity', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-NI-02', 'B', 'We produce good reports but data capability is concentrated in one team. We need to democratise access and formalise governance institutionally', 'Intentional', 2, false, NULL, 'Expert dependency without distributed access is Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-NI-02', 'C', 'Self-service analytics are available to authorised users across the institution with institutional data governance ensuring quality', 'Integrated', 3, false, NULL, 'Distributed access with institutional governance indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-NI-02', 'D', 'Leaders can access data through an institutional analytics platform with training support and defined governance roles across all units', 'Integrated', 3, false, NULL, 'Institutional platform with governance indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGD-IO-01 :: Planning & Governance: Data :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGD-IO-01', 'maturity-the', 'the-pg-data', 'Planning & Governance: Data', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has comprehensive, well-governed institutional performance and planning data with institution-wide dashboards. A vendor offers an AI-powered predictive analytics tool. The Vice-Chancellor is excited and wants to implement it immediately, claiming it will make the institution Optimised.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-IO-01', 'A', 'Implementing AI analytics on our well-governed data will make us sector-leading immediately', 'Integrated', 3, true, 'Tool adoption does not equal maturity. Optimised requires proven impact, continuous improvement, and sector contribution, not just tool acquisition', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-IO-01', 'B', 'We should pilot the tool, evaluate impact against outcomes, and publish our findings. If it works, we''ll iterate and share our methodology', 'Optimised', 4, false, NULL, 'Evidence-based evaluation with publication and continuous improvement indicates Optimised approach', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-IO-01', 'C', 'We already use predictive models validated against outcomes with documented impact. We''d evaluate this tool against our existing capabilities', 'Optimised', 4, false, NULL, 'Existing validated predictive capabilities with impact evidence indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-IO-01', 'D', 'We should implement it carefully with proper evaluation. AI tools need to be properly governed before deployment', 'Integrated', 3, false, NULL, 'Careful implementation with governance is good Integrated practice, not Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGD-IO-02 :: Planning & Governance: Data :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGD-IO-02', 'maturity-the', 'the-pg-data', 'Planning & Governance: Data', 'integrated-optimised', 'Integrated', 'Optimised', 'A sector body asks your university to contribute to developing new data standards for planning and governance. You currently have strong internal data governance but have not previously engaged with sector-level data practice.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-IO-02', 'A', 'We''d be delighted to contribute. Our data governance for planning and governance is strong internally and we''re ready to share our approaches', 'Optimised', 4, false, NULL, 'Willingness and readiness to contribute to sector standards indicates movement toward Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-IO-02', 'B', 'We''re confident our internal data practice is good but we should focus on maintaining what we have rather than taking on sector work', 'Integrated', 3, false, NULL, 'Internal focus without sector contribution is Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-IO-02', 'C', 'We''ve been contributing to sector data standards for several years and our governance framework has been adopted by three peer institutions', 'Optimised', 4, false, NULL, 'Sustained contribution with peer adoption indicates established Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGD-IO-02', 'D', 'We benchmark our data governance against the sector body''s existing standards to ensure we meet best practice', 'Integrated', 3, true, 'Benchmarking against others'' standards is consuming not contributing. The attractive nuisance is that benchmarking feels like sector engagement', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGU-IN-01 :: Planning & Governance: Utilisation :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGU-IN-01', 'maturity-the', 'the-pg-utilization', 'Planning & Governance: Utilisation', 'incidental-intentional', 'Incidental', 'Intentional', 'Your university invested in BI dashboards, digital board papers, and planning tools two years ago. Usage data (where available) shows that only 30% of senior leaders and governors regularly use the core features, and fewer than 10% use advanced capabilities. Most senior leaders and governors continue with previous manual or paper-based approaches.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-IN-01', 'A', 'We''ve made the investment and the tools are available. People will adopt at their own pace', 'Incidental', 1, true, 'Availability without promotion of adoption is Incidental. The attractive nuisance is framing passive availability as a strategy', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-IN-01', 'B', 'We''ve launched a training programme with minimum usage expectations and we''re tracking adoption rates institution-wide', 'Intentional', 2, false, NULL, 'Purposeful promotion with training and tracking indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-IN-01', 'C', 'Adoption is low and we don''t have a plan to address it. The tools are there but people haven''t taken to them', 'Incidental', 1, false, NULL, 'Unaddressed low adoption is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-IN-01', 'D', 'Usage is high and consistent. Over 80% of senior leaders and governors regularly use both basic and advanced features', 'Intentional', 2, false, NULL, 'This would indicate at least Intentional but contradicts the scenario', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGU-IN-02 :: Planning & Governance: Utilisation :: Incidental-Intentional
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGU-IN-02', 'maturity-the', 'the-pg-utilization', 'Planning & Governance: Utilisation', 'incidental-intentional', 'Incidental', 'Intentional', 'A few senior leaders and governors at your university have developed impressive workflows using BI dashboards, digital board papers, and planning tools and have been nominated for an innovation award. Meanwhile, their immediate colleagues continue using older methods. No one has been asked to adopt the innovative approaches more widely.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-IN-02', 'A', 'Our award nominees demonstrate excellent utilisation across the institution', 'Incidental', 1, true, 'Individual excellence is the Incidental pattern. The attractive nuisance is pointing to champions as evidence of institutional utilisation', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-IN-02', 'B', 'We''ve identified these innovators and are using their approaches to develop institutional training and minimum standards for all senior leaders and governors', 'Intentional', 2, false, NULL, 'Converting individual innovation into institutional programme indicates Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-IN-02', 'C', 'These individuals found their own way. We haven''t yet developed an institutional approach to promoting consistent utilisation', 'Incidental', 1, false, NULL, 'Individual adoption without institutional promotion is Incidental', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-IN-02', 'D', 'We have minimum standards for tool usage and consistent adoption is monitored across the institution', 'Intentional', 2, false, NULL, 'Institutional standards with monitoring indicates at least Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGU-NI-01 :: Planning & Governance: Utilisation :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGU-NI-01', 'maturity-the', 'the-pg-utilization', 'Planning & Governance: Utilisation', 'intentional-integrated', 'Intentional', 'Integrated', 'Your university has been actively promoting adoption of BI dashboards, digital board papers, and planning tools with training programmes and published expectations. Adoption has risen to 65% for basic features, with three faculties at over 80% and two below 40%. Impact on outcomes is not measured.', 'What would you most likely do?', '{"institution_size":"large","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-NI-01', 'A', 'Adoption is growing strongly. We''re approaching integrated utilisation across the institution', 'Intentional', 2, true, 'Significant variation (40-80%) across faculties and no impact measurement is Intentional. The attractive nuisance is that average growth masks inconsistency', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-NI-01', 'B', 'Adoption is growing but not yet consistent. We need institution-wide standards, measurement of outcomes, and intervention in lagging areas', 'Intentional', 2, false, NULL, 'Inconsistent adoption without impact measurement is Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-NI-01', 'C', 'Adoption is consistently above 80% across all units, we measure impact on outcomes, and user feedback informs tool optimisation', 'Integrated', 3, false, NULL, 'Consistent high adoption with impact measurement indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-NI-01', 'D', 'All core planning and governance processes run on digital workflows and we can demonstrate improvement in outcomes attributable to tool utilisation', 'Integrated', 3, false, NULL, 'Digital workflows as standard with demonstrated impact indicates Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGU-NI-02 :: Planning & Governance: Utilisation :: Intentional-Integrated
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGU-NI-02', 'maturity-the', 'the-pg-utilization', 'Planning & Governance: Utilisation', 'intentional-integrated', 'Intentional', 'Integrated', 'Several departments at your university report that they have ''gone fully digital'' for planning and governance processes. However, an audit reveals that while forms are digital, they are printed out for review, approvals happen by email rather than through the system, and data entry is duplicated across platforms.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-NI-02', 'A', 'We''ve digitised our processes. All our forms and records are digital now', 'Intentional', 2, true, 'Digitising the form without digitising the workflow is Intentional. The attractive nuisance is equating digital forms with digital processes', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-NI-02', 'B', 'We''ve digitised inputs but not workflows. True utilisation means end-to-end digital processes without manual intervention', 'Intentional', 2, false, NULL, 'Recognising partial digitisation accurately describes Intentional', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-NI-02', 'C', 'Our processes run end-to-end digitally with no paper fallbacks, automated routing, and single data entry', 'Integrated', 3, false, NULL, 'End-to-end digital workflows indicate Integrated', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-NI-02', 'D', 'We''ve not digitised our processes yet. Most work is still paper-based', 'Incidental', 1, false, NULL, 'This understates the scenario which describes partial digitisation', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGU-IO-01 :: Planning & Governance: Utilisation :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGU-IO-01', 'maturity-the', 'the-pg-utilization', 'Planning & Governance: Utilisation', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university has consistent, high utilisation of BI dashboards, digital board papers, and planning tools across all units. A vendor approaches you asking to feature your institution as a case study for how well you use their product. You are asked whether this means your utilisation is Optimised.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-IO-01', 'A', 'Vendor recognition confirms we are sector-leading in utilisation of planning and governance tools', 'Integrated', 3, true, 'Vendor case studies are marketing tools, not independent assessment. Consistent usage of a product as intended is Integrated, not innovation', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-IO-01', 'B', 'We use the tools well as designed. But Optimised utilisation means our users innovate new use cases and drive continuous improvement beyond standard deployment', 'Integrated', 3, false, NULL, 'Distinguishing standard effective use from innovation-driven optimisation is accurate', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-IO-01', 'C', 'Our users have developed novel applications of these tools that the vendor has incorporated into their product roadmap. We continuously optimise based on usage analytics', 'Optimised', 4, false, NULL, 'User innovation influencing vendor roadmaps and continuous optimisation indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-IO-01', 'D', 'We regularly present at user conferences sharing innovative workflows our senior leaders and governors have developed, and peer institutions adopt our configurations', 'Optimised', 4, false, NULL, 'Innovative workflows adopted by peers indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

-- THE-PGU-IO-02 :: Planning & Governance: Utilisation :: Integrated-Optimised
INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('THE-PGU-IO-02', 'maturity-the', 'the-pg-utilization', 'Planning & Governance: Utilisation', 'integrated-optimised', 'Integrated', 'Optimised', 'Your university monitors utilisation of BI dashboards, digital board papers, and planning tools through monthly reports. Adoption is consistently above 85% for core features. Users report high satisfaction. A review asks whether there is anything more to achieve.', 'What would you most likely do?', '{"institution_size":"universal","institution_type":"universal","region":"universal"}'::jsonb, 'active', '{"source_framework":"THE Digital Maturity Index","content_type":"original","attribution_text":"Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.","share_alike_applies":false}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id,
  dimension_id = EXCLUDED.dimension_id,
  dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary,
  target_lower_level = EXCLUDED.target_lower_level,
  target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem,
  question = EXCLUDED.question,
  context_tags = EXCLUDED.context_tags,
  status = EXCLUDED.status,
  source_attribution = EXCLUDED.source_attribution,
  updated_at = now();

INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-IO-02', 'A', 'We''ve achieved consistent high utilisation. We should maintain current levels and focus investment elsewhere', 'Integrated', 3, true, 'Maintenance of current utilisation is Integrated. Optimised means continuous improvement and innovation, not steady-state management', NULL, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-IO-02', 'B', 'We should move from monitoring adoption to analysing usage patterns to identify optimisation opportunities and measuring impact on outcomes', 'Optimised', 4, false, NULL, 'Moving from adoption monitoring to optimisation and impact measurement indicates Optimised thinking', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-IO-02', 'C', 'Our users already drive innovation. We analyse usage patterns in real-time, users contribute novel workflows, and we quantify impact on outcomes', 'Optimised', 4, false, NULL, 'User-driven innovation with analytics and impact quantification indicates Optimised', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;
INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES ('THE-PGU-IO-02', 'D', 'We should push adoption of advanced features to increase the 85% further', 'Integrated', 3, false, NULL, 'Pursuing higher adoption of existing features is still an Integrated activity, not innovation-driven optimisation', NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;

COMMIT;
