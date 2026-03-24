-- QS AI Capability Framework: 56 scenarios + responses
-- Generated from ~/Downloads/jjj/scenarios.json
-- Idempotent: ON CONFLICT DO UPDATE

BEGIN;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-GOVREG-BD-01', 'ai-capability', 'qs-gov-regulatory', 'Regulatory & Ethical Standards', 'basic-developing', 'Basic', 'Developing', 'Your university has deployed several AI tools across different departments over the past two years. A new government consultation on AI in higher education asks institutions to describe their approach to AI regulatory compliance. The DVC asks you to coordinate the response.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVREG-BD-01', 'A', 'Acknowledge the consultation but explain that each department manages its own AI tools and there is no central record of what is in use or which regulations apply. Suggest the DVC asks individual heads of department to contribute.', 'Basic', 1, true, 'Sounds pragmatic and honest, but it reveals the institution has no systematic approach to regulatory monitoring. The candour masks the absence of capability.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVREG-BD-01', 'B', 'Compile an inventory of AI tools in use across the institution. Identify which regulations apply to each use case. Assign a staff member to draft the consultation response and establish a basic compliance tracking mechanism going forward.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVREG-BD-01', 'C', 'Draw on the institution''s existing AI compliance register and regulatory monitoring function to draft a comprehensive response. Use the consultation as an opportunity to review and update the institution''s regulatory compliance framework.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-GOVREG-BD-02', 'ai-capability', 'qs-gov-regulatory', 'Regulatory & Ethical Standards', 'basic-developing', 'Basic', 'Developing', 'The EU AI Act has come into force and your university''s legal team has flagged that some of your AI applications may fall under the ''high-risk'' category. The Registrar asks what the institution should do.', 'What would you most likely do?', '{"institution_type":"universal","region":["EU","UK"],"size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVREG-BD-02', 'A', 'Ask the legal team to send the guidance to relevant departments and let each one decide how to respond. The institution does not have a central AI function so this is best handled locally.', 'Basic', 1, true, 'Appears to be delegating to experts (legal, departments), but actually reveals no institutional mechanism for AI regulatory response.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVREG-BD-02', 'B', 'Convene a cross-functional group to map institutional AI use cases against the Act''s risk categories. Develop a prioritised action plan for high-risk applications. Assign ongoing monitoring responsibility.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVREG-BD-02', 'C', 'Activate the institution''s existing AI regulatory compliance process. The AI governance team has already mapped use cases to the Act''s categories. Review and update the institutional risk classification, adjust existing governance processes, and report compliance status to the board.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-GOVREG-DA-01', 'ai-capability', 'qs-gov-regulatory', 'Regulatory & Ethical Standards', 'developing-advanced', 'Developing', 'Advanced', 'Your university has an AI governance working group that meets quarterly and has developed AI ethical guidelines. The group now proposes embedding AI regulatory compliance into the institution''s annual governance cycle, with standing committee agenda items and automated compliance monitoring.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":["medium","large"]}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVREG-DA-01', 'A', 'The working group has done excellent work. Continue the quarterly meetings and add regulatory compliance to the existing agenda. The current approach is working well and formalising it further risks bureaucracy.', 'Developing', 2, true, 'Sounds like it values the group''s work and avoids bureaucracy, but resists the systematic embedding that characterises Advanced capability.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVREG-DA-01', 'B', 'Approve the proposal to embed AI compliance in annual governance cycles. Establish standing agenda items on relevant committees, automated regulatory change monitoring, and annual compliance reporting to the governing body.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVREG-DA-01', 'C', 'Approve the proposal but phase it in gradually. Start with one committee and expand based on experience. Commission an external review of the compliance framework after the first year.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-GOVREG-DA-02', 'ai-capability', 'qs-gov-regulatory', 'Regulatory & Ethical Standards', 'developing-advanced', 'Developing', 'Advanced', 'Your university has been monitoring AI regulatory developments for two years. Staff in relevant roles receive quarterly briefings. The AI lead proposes that the institution should now contribute to sector regulatory consultations and join an international AI governance consortium.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVREG-DA-02', 'A', 'The institution should focus on its own compliance before trying to influence sector-level regulation. Contributing to consultations is a distraction from embedding our own practices.', 'Developing', 2, true, 'Sounds prudent and focused, but an Advanced institution both manages its own compliance and contributes externally.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVREG-DA-02', 'B', 'Support the proposal. An institution with mature AI governance should contribute to the sector and learn from international peers. Allocate time for the AI lead to engage in consultations and join the consortium. Integrate external learning into institutional practice.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVREG-DA-02', 'C', 'Agree in principle but suggest the AI lead attend one consortium meeting first to assess the value before committing institutional resource.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-GOVRISK-BD-01', 'ai-capability', 'qs-gov-risk', 'Governance & Risk Management', 'basic-developing', 'Basic', 'Developing', 'Your university is about to deploy an AI-powered plagiarism detection system across all programmes. Several faculty members have raised concerns about potential bias. The PVC Education asks how the institution should proceed.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVRISK-BD-01', 'A', 'Let the department that raised concerns handle it locally. They understand their context. A central risk assessment process would slow down AI adoption when the institution needs to move quickly.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in governance & risk management. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVRISK-BD-01', 'B', 'Pause the deployment until a risk assessment has been completed. Develop an AI risk assessment template covering bias, data protection, accuracy, and reputational risk. Use this case to establish the institution''s approach to AI risk.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVRISK-BD-01', 'C', 'Apply the institution''s established AI risk framework to classify the deployment. Conduct the appropriate risk assessment for its risk tier. Document the decision and add it to the AI risk register. Review the framework''s coverage of this type of tool.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-GOVRISK-BD-02', 'ai-capability', 'qs-gov-risk', 'Governance & Risk Management', 'basic-developing', 'Basic', 'Developing', 'A department has purchased an AI tool for student feedback analysis without consulting IT or governance. The CIO discovers this during a routine software audit. The tool processes student personal data.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVRISK-BD-02', 'A', 'Note the concern but proceed with deployment. The vendor assures the tool has been tested. Requiring a formal risk assessment for every AI tool would create an unworkable bottleneck.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in governance & risk management. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVRISK-BD-02', 'B', 'Establish an AI due diligence process for all tools processing personal data. Require the department to complete a risk assessment and DPIA before the tool can be used. Create a lightweight assessment process for future deployments.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVRISK-BD-02', 'C', 'Activate the existing risk assessment process. The AI governance team already has a tiered framework for this. Conduct the assessment, document findings, implement mitigations, and add to the institution''s AI risk register.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-GOVRISK-DA-01', 'ai-capability', 'qs-gov-risk', 'Governance & Risk Management', 'developing-advanced', 'Developing', 'Advanced', 'Your university has established an AI risk assessment process that covers all new deployments. The AI governance committee proposes moving to a tiered risk framework where low-risk tools have a lighter assessment process and high-risk tools require board-level approval.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVRISK-DA-01', 'A', 'The current risk assessment process is working for high-profile deployments. Extend it incrementally to cover more AI tools but avoid making it too bureaucratic. Continue the quarterly review approach.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in governance & risk management.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVRISK-DA-01', 'B', 'Approve the tiered risk framework. Integrate AI risk into the corporate risk management cycle with annual board-level reporting. This ensures proportionate scrutiny and institutional oversight.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVRISK-DA-01', 'C', 'Pilot the tiered approach with one faculty before rolling out institution-wide. Gather evidence on whether it works before committing to full integration.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-GOVRISK-DA-02', 'ai-capability', 'qs-gov-risk', 'Governance & Risk Management', 'developing-advanced', 'Developing', 'Advanced', 'Your institution''s AI risk register has grown to cover 40 AI deployments. The AI lead proposes an annual institution-wide AI risk review integrated with the corporate risk management cycle.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVRISK-DA-02', 'A', 'Keep the current approach and focus on improving risk assessment quality rather than changing the structure. More training for assessors would be more valuable than a new framework.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in governance & risk management.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVRISK-DA-02', 'B', 'Implement the proposal. Embed AI risk management in institutional governance with defined risk appetite, standing risk register reviews, and formal reporting to the board. Commission an independent review after the first year.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVRISK-DA-02', 'C', 'Focus on improving the existing process before changing the structure. A tiered framework sounds good in theory but the institution is not ready for that level of sophistication.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-GOVCON-BD-01', 'ai-capability', 'qs-gov-conduct', 'Code of Conduct & Privacy', 'basic-developing', 'Basic', 'Developing', 'A student is caught using an AI tool to generate a substantial portion of their dissertation. The academic integrity panel finds that the institutional regulations do not mention AI. Different departments have been applying different standards.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVCON-BD-01', 'A', 'Let each department set its own AI use rules. Academic freedom means departments should decide their own standards for student AI use. A central code would be too rigid for diverse disciplines.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in code of conduct & privacy. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVCON-BD-01', 'B', 'Develop an institution-wide AI code of conduct that sets minimum standards while allowing disciplinary flexibility. Establish a task group with representatives from across the institution to draft it.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVCON-BD-01', 'C', 'Review and update the existing code of conduct to address the specific case. The institution has an established code with clear enforcement mechanisms. This case tests its coverage and triggers a review cycle.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-GOVCON-BD-02', 'ai-capability', 'qs-gov-conduct', 'Code of Conduct & Privacy', 'basic-developing', 'Basic', 'Developing', 'Staff in the marketing department are using generative AI to produce recruitment materials. Some of the generated content contains inaccurate claims about the university. There is no institutional guidance on staff AI use.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVCON-BD-02', 'A', 'Deal with this case under existing academic integrity regulations. AI is just another tool. Creating new AI-specific rules would be premature given how fast the technology is changing.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in code of conduct & privacy. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVCON-BD-02', 'B', 'Commission an AI code of conduct covering academic integrity, staff use, and student expectations. Communicate it through multiple channels and integrate it into student handbooks and staff induction.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVCON-BD-02', 'C', 'Apply the existing code, which was designed to handle evolving AI capabilities. Use this case for a scheduled review, updating the code, communicating changes, and reinforcing enforcement across all faculties.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-GOVCON-DA-01', 'ai-capability', 'qs-gov-conduct', 'Code of Conduct & Privacy', 'developing-advanced', 'Developing', 'Advanced', 'Your university published an AI code of conduct 18 months ago. Student union representatives report that many students are unaware of it and that enforcement is inconsistent across faculties.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVCON-DA-01', 'A', 'The code has been published and that is good progress. Focus on raising awareness through more communications rather than changing the enforcement approach. Culture change takes time.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in code of conduct & privacy.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVCON-DA-01', 'B', 'Commission a systematic review: assess awareness levels across all faculties, identify enforcement gaps, update the code for current AI capabilities, integrate it into induction and assessment processes, and establish annual monitoring.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVCON-DA-01', 'C', 'Ask student union representatives and faculty leads to co-design better communication. The code is fine; the problem is awareness. More targeted messaging will help.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-GOVCON-DA-02', 'ai-capability', 'qs-gov-conduct', 'Code of Conduct & Privacy', 'developing-advanced', 'Developing', 'Advanced', 'The AI code of conduct is due for annual review. Since publication, AI capabilities have advanced significantly. The review committee must decide between incremental updates and a fundamental revision.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVCON-DA-02', 'A', 'Conduct awareness sessions in faculties and add the code to the VLE. Enforcement consistency will improve as awareness grows. An incremental approach is more realistic.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in code of conduct & privacy.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVCON-DA-02', 'B', 'Address this systematically. The code needs to be embedded in processes, not just published. Integrate into programme handbooks, assessment briefs, induction, and staff contracts. Establish monitoring and annual review.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVCON-DA-02', 'C', 'Focus on the faculties with the worst enforcement first. A targeted improvement plan is more achievable than trying to fix everything at once.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-GOVLEAD-BD-01', 'ai-capability', 'qs-gov-leadership', 'Leadership & Capability', 'basic-developing', 'Basic', 'Developing', 'Your university''s strategic plan makes no mention of AI. Several departments have started using AI tools independently. The Vice-Chancellor has asked the senior team to consider whether the institution needs an AI strategy.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVLEAD-BD-01', 'A', 'AI is evolving too fast for a formal strategy. The institution should stay flexible and let departments innovate. A central strategy would become outdated before it was implemented.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in leadership & capability. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVLEAD-BD-01', 'B', 'Assign AI responsibility to a senior leader. Commission an AI strategy that aligns with the institutional plan. Coordinate the existing initiatives and establish a governance structure.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVLEAD-BD-01', 'C', 'Appoint a dedicated senior AI leader with authority across institutional boundaries. Develop a funded AI strategy integrated with the institutional plan. Establish governance, reporting, and progress tracking.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-GOVLEAD-BD-02', 'ai-capability', 'qs-gov-leadership', 'Leadership & Capability', 'basic-developing', 'Basic', 'Developing', 'Three separate AI initiatives are running at the institution: IT is piloting Microsoft Copilot, the business school has created an AI lab, and student services is testing a chatbot. Nobody is coordinating them.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVLEAD-BD-02', 'A', 'Acknowledge the need but prioritise other strategic issues first. AI is important but the institution has more pressing challenges. Revisit this in the next planning cycle.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in leadership & capability. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVLEAD-BD-02', 'B', 'Create a cross-institutional AI working group, appoint a senior sponsor, and develop a roadmap that connects the existing initiatives to institutional priorities.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVLEAD-BD-02', 'C', 'Integrate AI leadership into the permanent senior management structure. Approve a resourced AI strategy with clear KPIs, governance reporting, and board-level accountability.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-GOVLEAD-DA-01', 'ai-capability', 'qs-gov-leadership', 'Leadership & Capability', 'developing-advanced', 'Developing', 'Advanced', 'Your university appointed a senior AI lead 18 months ago and published an AI strategy. The AI lead reports that implementation is progressing but some faculties are resistant and the strategy is not yet reflected in budget allocation.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVLEAD-DA-01', 'A', 'Continue the current approach. The AI lead is making progress and faculty resistance will reduce over time. Dedicated budget can wait until the strategy has proved its value.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in leadership & capability.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVLEAD-DA-01', 'B', 'Make AI leadership a permanent institutional function with dedicated budget. Integrate AI strategy reporting into the board cycle. Give the AI lead authority to convene across faculty boundaries and allocate resource.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVLEAD-DA-01', 'C', 'Extend the AI lead role for another year and review again. Permanent status is premature until the strategy has been fully implemented.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-GOVLEAD-DA-02', 'ai-capability', 'qs-gov-leadership', 'Leadership & Capability', 'developing-advanced', 'Developing', 'Advanced', 'The institution''s AI strategy is two years old and has been largely implemented. The board asks whether AI leadership should become a permanent function with dedicated budget or remain a time-limited transformation programme.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVLEAD-DA-02', 'A', 'Support the AI lead with stronger communications to resistant faculties. The strategy exists; the challenge is adoption. More engagement will help.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in leadership & capability.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVLEAD-DA-02', 'B', 'Approve permanent status with dedicated budget. The strategic value is demonstrated. Integration into governance ensures sustainability beyond individual champions.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-GOVLEAD-DA-02', 'C', 'Address faculty resistance through a dedicated engagement programme before formalising the role. Buy-in is more important than structural changes.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-OUTREC-BD-01', 'ai-capability', 'qs-out-recruitment', 'AI Enhanced Recruitment', 'basic-developing', 'Basic', 'Developing', 'Your university''s admissions team is overwhelmed by application volumes. A vendor demonstrates an AI tool that can triage applications and predict yield. The registrar asks whether the institution should pilot it.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTREC-BD-01', 'A', 'The admissions team has managed fine without AI. Adding AI to recruitment risks depersonalising the applicant experience. Focus on hiring more admissions staff instead.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in ai enhanced recruitment. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTREC-BD-01', 'B', 'Pilot the AI tool in a controlled scope with clear evaluation criteria. Establish ethical guardrails including bias testing and human oversight. Evaluate before scaling.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTREC-BD-01', 'C', 'Integrate the tool into the existing AI-enhanced admissions pipeline. Apply the institution''s standard AI deployment process including risk assessment, bias audit, and effectiveness measurement. Scale based on evidence.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-OUTREC-BD-02', 'ai-capability', 'qs-out-recruitment', 'AI Enhanced Recruitment', 'basic-developing', 'Basic', 'Developing', 'The marketing team proposes using AI to personalise email campaigns to prospective students based on their browsing behaviour on the university website. The DVC asks about the implications.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTREC-BD-02', 'A', 'AI recruitment tools are expensive and unproven. Wait until other institutions have tested them and established best practice before investing.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in ai enhanced recruitment. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTREC-BD-02', 'B', 'Run a structured pilot with defined success metrics, ethical review, and human oversight. Use results to inform whether and how to scale AI in admissions.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTREC-BD-02', 'C', 'Deploy through the established AI procurement and risk assessment process. The institution has mature frameworks for AI in sensitive processes. Conduct bias audit, establish monitoring, and integrate with existing systems.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-OUTREC-DA-01', 'ai-capability', 'qs-out-recruitment', 'AI Enhanced Recruitment', 'developing-advanced', 'Developing', 'Advanced', 'Your university has been using AI in admissions for two years across several programmes. The AI lead proposes scaling to all programmes and integrating AI yield prediction into offer-making strategy.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTREC-DA-01', 'A', 'Scaling too fast risks problems. Continue the pilots and extend gradually. Each programme is different and a one-size-fits-all approach will not work.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in ai enhanced recruitment.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTREC-DA-01', 'B', 'Approve institution-wide scaling with appropriate safeguards. Integrate AI yield prediction into the institutional offer strategy. Establish ongoing bias monitoring, transparent reporting, and continuous improvement cycles.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTREC-DA-01', 'C', 'Extend to three more programmes first, then review. Gradual extension with evaluation at each step is the responsible approach.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-OUTREC-DA-02', 'ai-capability', 'qs-out-recruitment', 'AI Enhanced Recruitment', 'developing-advanced', 'Developing', 'Advanced', 'AI chatbots handle 60% of applicant enquiries but student feedback shows frustration with the bot''s limitations. The admissions director proposes a major upgrade integrating AI across the entire applicant journey.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTREC-DA-02', 'A', 'Keep AI in admissions at its current scope and focus on improving the tools we have. Scaling to all programmes is ambitious and the risks of bias at scale are significant.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in ai enhanced recruitment.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTREC-DA-02', 'B', 'Scale with a comprehensive framework: institution-wide deployment, integrated bias monitoring, transparent reporting to governance, and annual independent audit. Maintain human oversight for final decisions.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTREC-DA-02', 'C', 'Commission an independent review of the pilot results before scaling. External validation will build confidence.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-OUTSUP-BD-01', 'ai-capability', 'qs-out-support', 'Personalised Student Support', 'basic-developing', 'Basic', 'Developing', 'Student satisfaction surveys consistently show that students cannot get support outside office hours. The student services director proposes an AI chatbot for after-hours FAQ handling.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTSUP-BD-01', 'A', 'Students should learn to manage without 24/7 support. After-hours needs can be addressed the next working day. An AI chatbot would give impersonal responses to sensitive student queries.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in personalised student support. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTSUP-BD-01', 'B', 'Pilot an AI chatbot for high-volume, low-complexity queries with clear escalation to human staff. Evaluate student satisfaction and query resolution before considering expansion.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTSUP-BD-01', 'C', 'Deploy a sophisticated AI support system integrated across channels, with intelligent triage, automated handling of routine queries, and seamless escalation to human specialists. Monitor quality and equity of access.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-OUTSUP-BD-02', 'ai-capability', 'qs-out-support', 'Personalised Student Support', 'basic-developing', 'Basic', 'Developing', 'A career services review finds that advisors spend 40% of their time on routine enquiries. The director proposes introducing AI career tools to free advisors for complex guidance.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTSUP-BD-02', 'A', 'A chatbot will frustrate students more than help them. Focus on extending staff hours or better online self-service rather than AI.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in personalised student support. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTSUP-BD-02', 'B', 'Implement a phased approach: start with FAQ chatbot, measure impact, then extend to more complex support queries with proper human escalation.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTSUP-BD-02', 'C', 'Implement institution-wide AI support with 24/7 coverage, multi-channel integration, satisfaction monitoring, and continuous improvement. Ensure the system is accessible and equitable.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-OUTSUP-DA-01', 'ai-capability', 'qs-out-support', 'Personalised Student Support', 'developing-advanced', 'Developing', 'Advanced', 'Your university''s AI chatbot handles FAQs effectively but cannot deal with complex welfare or academic concerns. The student services director proposes a more sophisticated AI triage system that can assess urgency and route to appropriate support.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTSUP-DA-01', 'A', 'The chatbot works well for FAQs. Extending to complex queries is risky. Focus on improving the chatbot''s FAQ coverage rather than expanding into sensitive areas.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in personalised student support.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTSUP-DA-01', 'B', 'Approve the sophisticated triage system with strong safeguards. Design it to assess urgency, route appropriately, and maintain quality across all query types. Monitor equity of access and outcomes.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTSUP-DA-01', 'C', 'Pilot the triage system with one type of complex query before expanding. Sensitive support areas need careful testing.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-OUTSUP-DA-02', 'ai-capability', 'qs-out-support', 'Personalised Student Support', 'developing-advanced', 'Developing', 'Advanced', 'AI career tools have been available for a year but uptake is uneven. Employability data shows no improvement in graduate outcomes. The PVC Education asks whether the AI investment is justified.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTSUP-DA-02', 'A', 'Students value human support for complex issues. Improve the chatbot for routine queries and invest in more human capacity for complex ones.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in personalised student support.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTSUP-DA-02', 'B', 'Implement integrated AI triage with clear escalation protocols, quality monitoring, equity analysis, and continuous improvement. Commission regular student experience evaluation.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTSUP-DA-02', 'C', 'Invest in better training for the existing chatbot rather than a new system. Incremental improvement is less risky.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-OUTEFF-BD-01', 'ai-capability', 'qs-out-efficiency', 'Faculty & Administrative Efficiency', 'basic-developing', 'Basic', 'Developing', 'The finance team spends three weeks each quarter producing management reports by manually extracting data from multiple systems. A colleague suggests AI could automate much of this. The CFO asks you to investigate.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTEFF-BD-01', 'A', 'The finance team has always produced reports manually and they are accurate. AI automation introduces risk of errors that could affect financial reporting. The team is not asking for change.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in faculty & administrative efficiency. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTEFF-BD-01', 'B', 'Investigate the AI automation opportunity with a structured business case. Pilot in a defined scope with clear success criteria. Assess data readiness and integration requirements.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTEFF-BD-01', 'C', 'Deploy through the institution''s established AI administrative automation programme. Apply standard evaluation, data integration, and change management processes. Measure efficiency gains and quality impact.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-OUTEFF-BD-02', 'ai-capability', 'qs-out-efficiency', 'Faculty & Administrative Efficiency', 'basic-developing', 'Basic', 'Developing', 'Timetabling takes four months annually and still produces suboptimal results. An AI timetabling vendor offers a pilot. The registrar is interested but IT has concerns about data integration.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTEFF-BD-02', 'A', 'AI timetabling is promising but data integration issues make it impractical. Fix the data quality first, which is a multi-year project, before attempting AI automation.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in faculty & administrative efficiency. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTEFF-BD-02', 'B', 'Commission a pilot with evaluation criteria. Address data integration as part of the pilot scope. Establish a clear comparison between manual and AI-assisted processes.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTEFF-BD-02', 'C', 'Integrate into the existing AI efficiency programme with standard deployment processes, efficiency measurement, and staff training.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-OUTEFF-DA-01', 'ai-capability', 'qs-out-efficiency', 'Faculty & Administrative Efficiency', 'developing-advanced', 'Developing', 'Advanced', 'AI has been piloted in timetabling, document processing, and HR screening across different departments. The COO proposes a coordinated institutional programme to deploy AI across all administrative functions.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTEFF-DA-01', 'A', 'Each department has different systems and processes. A coordinated programme would be complex. Continue supporting departmental AI initiatives and share good practice.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in faculty & administrative efficiency.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTEFF-DA-01', 'B', 'Approve the coordinated institutional programme. Establish common standards, shared infrastructure, and central measurement. Integrate efficiency gains into planning and resource allocation.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTEFF-DA-01', 'C', 'Create a good practice network for administrative AI but let departments continue leading their own initiatives. Coordination is good; centralisation is not necessary.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-OUTEFF-DA-02', 'ai-capability', 'qs-out-efficiency', 'Faculty & Administrative Efficiency', 'developing-advanced', 'Developing', 'Advanced', 'Administrative AI tools are saving an estimated 2,000 staff hours per year across the institution. The AI lead proposes measuring and reporting these efficiency gains formally as part of the annual planning cycle.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTEFF-DA-02', 'A', 'Focus on getting the existing pilots working well before expanding. Quality over quantity.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in faculty & administrative efficiency.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTEFF-DA-02', 'B', 'Implement the institutional programme with dedicated resource, formal measurement, and reporting. Use efficiency gains to reinvest in further AI-enhanced administrative capability.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTEFF-DA-02', 'C', 'Extend successful pilots to similar departments first. Natural adoption will be more sustainable than a top-down programme.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-OUTENG-BD-01', 'ai-capability', 'qs-out-engagement', 'External Engagement & Partnership', 'basic-developing', 'Basic', 'Developing', 'A local employers'' association asks your university to run AI training for small businesses. No AI community engagement programme exists. The engagement director sees an opportunity but has no budget.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTENG-BD-01', 'A', 'Community engagement is not the university''s primary mission. Limited resources should focus on students and research. AI community training should be left to commercial providers.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in external engagement & partnership. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTENG-BD-01', 'B', 'Explore the opportunity with a defined scope. Develop a pilot community AI programme with clear objectives and evaluation. Use existing partnerships as a foundation.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTENG-BD-01', 'C', 'Approve strategic commitment to AI community engagement with dedicated resource and integration into the civic engagement strategy. Measure impact and community benefit.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-OUTENG-BD-02', 'ai-capability', 'qs-out-engagement', 'External Engagement & Partnership', 'basic-developing', 'Basic', 'Developing', 'A national AI consortium invites your university to join as a founding member. The membership would involve contributing to sector AI policy and sharing practice. The PVC asks whether the commitment is worthwhile.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTENG-BD-02', 'A', 'Joining an AI consortium is an ongoing commitment that will take senior staff time away from core work. The benefits are unclear. Observe from a distance first.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in external engagement & partnership. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTENG-BD-02', 'B', 'Join the consortium with defined objectives. Assign a representative with clear expectations for what the institution aims to gain and contribute.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTENG-BD-02', 'C', 'Join as a strategic priority with senior representation, defined contribution plan, and integration of consortium learning into institutional practice.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-OUTENG-DA-01', 'ai-capability', 'qs-out-engagement', 'External Engagement & Partnership', 'developing-advanced', 'Developing', 'Advanced', 'Your university runs occasional AI workshops for the local community and participates in two sector AI groups. The engagement director proposes making AI community engagement a sustained strategic commitment with dedicated resource.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTENG-DA-01', 'A', 'Continue the occasional workshops and working group participation. Community engagement is valuable but the institution cannot justify dedicated resource for AI-specific engagement.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in external engagement & partnership.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTENG-DA-01', 'B', 'Approve dedicated resource for sustained AI community engagement. Position the university as a regional AI education hub. Integrate community AI engagement into the institutional strategy and measure impact.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTENG-DA-01', 'C', 'Seek external funding for the community programme before committing institutional resource. If there is demand, funders will support it.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-OUTENG-DA-02', 'ai-capability', 'qs-out-engagement', 'External Engagement & Partnership', 'developing-advanced', 'Developing', 'Advanced', 'Industry partners approach the university about a multi-year collaborative AI research programme. It would require dedicated infrastructure and academic time. The PVC Research sees strategic value but the investment is significant.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTENG-DA-02', 'A', 'Increase the frequency of workshops and expand to more community groups. Sustaining what we have is more realistic than a step-change.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in external engagement & partnership.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTENG-DA-02', 'B', 'Commit to the multi-year partnership with dedicated infrastructure and academic time. The strategic value justifies the investment. Establish formal governance and impact measurement.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-OUTENG-DA-02', 'C', 'Pilot a sustained programme for one year with a clear evaluation framework before making a longer commitment.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-TLCUR-BD-01', 'ai-capability', 'qs-tl-curriculum', 'Course Design & Curriculum', 'basic-developing', 'Basic', 'Developing', 'Students in the same programme are receiving contradictory messages about AI use from different lecturers. Some encourage it; others ban it. A student complaint reaches the PVC Education.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLCUR-BD-01', 'A', 'Academic freedom means each lecturer should decide their own approach to AI. A central policy would be too prescriptive for diverse disciplines. Students will learn to navigate different expectations.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in course design & curriculum. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLCUR-BD-01', 'B', 'Develop institutional guidance on AI in teaching through a working group with faculty representation. Cover expectations for both staff and student AI use. Allow disciplinary flexibility within institutional principles.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLCUR-BD-01', 'C', 'Implement comprehensive AI teaching guidance integrated into quality assurance. All programmes must demonstrate how they prepare students for an AI-influenced world. Monitor adoption and update annually.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-TLCUR-BD-02', 'ai-capability', 'qs-tl-curriculum', 'Course Design & Curriculum', 'basic-developing', 'Basic', 'Developing', 'The programme review process does not ask about AI integration. The PVC Education proposes adding AI-related questions to the programme approval and review template.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLCUR-BD-02', 'A', 'AI is changing too fast for the institution to set a fixed approach. Let practice develop organically and codify it once things settle down.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in course design & curriculum. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLCUR-BD-02', 'B', 'Add AI integration questions to programme approval and review templates. This embeds AI consideration in the existing quality assurance cycle without requiring a separate initiative.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLCUR-BD-02', 'C', 'Establish AI literacy as a graduate attribute with programme-level embedding. Integrate into quality assurance, professional development, and curriculum design processes.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-TLCUR-DA-01', 'ai-capability', 'qs-tl-curriculum', 'Course Design & Curriculum', 'developing-advanced', 'Developing', 'Advanced', 'Your university published AI teaching guidance a year ago and most departments reference it. However, a curriculum audit finds that only 30% of programmes have explicitly updated learning outcomes to include AI literacy.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLCUR-DA-01', 'A', '30% is good progress for the first year. Continue encouraging departments to update learning outcomes. A mandate would create resistance. Organic adoption is more sustainable.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in course design & curriculum.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLCUR-DA-01', 'B', 'Approve AI literacy as a graduate attribute. Require all programmes to demonstrate AI integration at the next programme review. Provide development support for programme teams and track progress.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLCUR-DA-01', 'C', 'Focus on the 70% of programmes that have not yet updated. A targeted support programme for these programmes is more achievable than a universal mandate.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-TLCUR-DA-02', 'ai-capability', 'qs-tl-curriculum', 'Course Design & Curriculum', 'developing-advanced', 'Developing', 'Advanced', 'The institution has agreed an approach to AI in teaching and revised some curricula. The PVC Education proposes that AI literacy should become a graduate attribute expected of all students regardless of discipline.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLCUR-DA-02', 'A', 'AI literacy as a graduate attribute is aspirational but premature. Focus on getting the current guidance embedded in more programmes before expanding scope.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in course design & curriculum.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLCUR-DA-02', 'B', 'Mandate AI literacy integration across all programmes with a defined timeline, development support, and quality assurance monitoring. This is a fundamental curriculum requirement for graduates.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLCUR-DA-02', 'C', 'Pilot AI literacy as a graduate attribute in three faculties before requiring it institution-wide. Learn from early adopters.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-TLPER-BD-01', 'ai-capability', 'qs-tl-personalised', 'Personalised Learning & Support', 'basic-developing', 'Basic', 'Developing', 'Retention data shows that 15% of first-year students leave before completing their first year. A learning analytics vendor offers an AI early warning system. The student experience team is interested.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLPER-BD-01', 'A', 'An AI early warning system will label students as at-risk, which is stigmatising. The pastoral system works through human relationships. AI cannot replace the personal tutoring that helps students stay.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in personalised learning & support. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLPER-BD-01', 'B', 'Pilot the early warning system with clear ethical guardrails, student consent, and defined intervention pathways. Evaluate effectiveness and equity before considering expansion.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLPER-BD-01', 'C', 'Integrate the system into student support operations with automated alerts, defined intervention protocols, fairness monitoring, and continuous improvement. Expand to all students with appropriate safeguards.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-TLPER-BD-02', 'ai-capability', 'qs-tl-personalised', 'Personalised Learning & Support', 'basic-developing', 'Basic', 'Developing', 'An academic proposes piloting an AI tutoring system in introductory maths modules where failure rates are high. The system would provide personalised practice and feedback.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLPER-BD-02', 'A', 'Personalised AI tutoring undermines the value of human teaching. Students need more contact hours with academics, not more screen time with machines.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in personalised learning & support. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLPER-BD-02', 'B', 'Run a controlled pilot in the target modules with evaluation of learning outcomes, student experience, and equity. Define how AI tutoring complements rather than replaces human teaching.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLPER-BD-02', 'C', 'Deploy at scale with comprehensive quality assurance, equity monitoring, integration with the VLE, and evidence-based continuous improvement.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-TLPER-DA-01', 'ai-capability', 'qs-tl-personalised', 'Personalised Learning & Support', 'developing-advanced', 'Developing', 'Advanced', 'An AI early warning system for student retention has been running in two faculties for 18 months. It identifies at-risk students but academic tutors report they do not have capacity to follow up on all alerts.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLPER-DA-01', 'A', 'The alert system identifies students but tutors cannot keep up. Add more tutors rather than making the AI system more sophisticated. The human response is the bottleneck, not the AI.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in personalised learning & support.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLPER-DA-01', 'B', 'Address the capacity gap by designing a response system that combines automated low-touch interventions with human support for complex cases. Monitor equity and effectiveness. Scale the AI and the human response together.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLPER-DA-01', 'C', 'Improve the existing system''s accuracy before expanding. False positives waste tutor time and false negatives miss students who need help.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-TLPER-DA-02', 'ai-capability', 'qs-tl-personalised', 'Personalised Learning & Support', 'developing-advanced', 'Developing', 'Advanced', 'AI personalised learning tools are used in 20% of modules. The PVC Education wants to scale to 50% but faculty express concern about AI replacing human teaching.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLPER-DA-02', 'A', 'Scaling from 20% to 50% of modules is too fast. Consolidate the current deployment and improve quality before expanding.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in personalised learning & support.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLPER-DA-02', 'B', 'Develop a scaling plan that addresses faculty concerns, provides pedagogic training, maintains quality, and monitors equity. AI enhances rather than replaces teaching.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLPER-DA-02', 'C', 'Invest in tutor capacity to respond to existing alerts before generating more alerts. The system is only valuable if responses happen.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-TLASS-BD-01', 'ai-capability', 'qs-tl-assessment', 'Assessment, Grading & Feedback', 'basic-developing', 'Basic', 'Developing', 'Students are submitting AI-generated work and current detection tools are unreliable. The assessment board asks whether the institution should invest in AI detection or redesign assessments.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLASS-BD-01', 'A', 'AI detection tools are improving. Invest in better detection rather than redesigning assessments. Students who use AI to cheat should be caught and penalised under academic integrity regulations.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in assessment, grading & feedback. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLASS-BD-01', 'B', 'Develop an institutional assessment strategy that addresses AI holistically. This should cover AI-resilient assessment design, appropriate AI use in assessment, AI-powered feedback, and academic integrity.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLASS-BD-01', 'C', 'Implement an institution-wide assessment strategy that integrates AI-resilient design, AI-powered feedback, adaptive assessment, and bias monitoring. All new assessments should meet AI-aware design standards.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-TLASS-BD-02', 'ai-capability', 'qs-tl-assessment', 'Assessment, Grading & Feedback', 'basic-developing', 'Basic', 'Developing', 'An AI assessment tool that provides automated formative feedback is available for pilot. The education team is keen but academics worry about feedback quality and the impact on student learning.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLASS-BD-02', 'A', 'Let individual academics decide whether to allow AI in their assessments. Disciplinary differences mean a central policy would be inappropriate.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in assessment, grading & feedback. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLASS-BD-02', 'B', 'Pilot the AI feedback tool with clear evaluation criteria: feedback quality, student satisfaction, learning outcomes, and equity of access. Use findings to inform institutional assessment strategy.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLASS-BD-02', 'C', 'Commission a systematic equity analysis of AI assessment tools. Develop targeted interventions to ensure equitable access and benefit. Integrate findings into assessment quality assurance processes.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-TLASS-DA-01', 'ai-capability', 'qs-tl-assessment', 'Assessment, Grading & Feedback', 'developing-advanced', 'Developing', 'Advanced', 'Your university uses AI formative feedback tools in several programmes. A review finds that students from disadvantaged backgrounds use the tools less often and benefit less. The PVC Education asks how to respond.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLASS-DA-01', 'A', 'Equity issues are concerning but complex. Commission further research before making policy changes. The current tools are better than no feedback at all.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in assessment, grading & feedback.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLASS-DA-01', 'B', 'Implement targeted interventions to address the equity gap immediately. Commission independent analysis. Integrate equity monitoring into all AI assessment tool deployments. Report to the education committee.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLASS-DA-01', 'C', 'Address the equity gap in the specific tools identified before developing institution-wide policy. Fix the known problem first.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-TLASS-DA-02', 'ai-capability', 'qs-tl-assessment', 'Assessment, Grading & Feedback', 'developing-advanced', 'Developing', 'Advanced', 'The assessment strategy is being rewritten. The head of quality proposes that all new assessments should be designed to be both AI-resilient (resistant to AI-generated submissions) and AI-inclusive (allowing appropriate AI use where it enhances learning).', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLASS-DA-02', 'A', 'AI-resilient and AI-inclusive assessment design sounds good in theory but academics need practical guidance, not principles. Focus on creating assessment templates and examples.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in assessment, grading & feedback.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLASS-DA-02', 'B', 'Approve the AI-aware assessment design standard for all new assessments. Provide development support, create exemplars, integrate into QA processes, and establish monitoring.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-TLASS-DA-02', 'C', 'Create a working group to develop practical assessment guidance before mandating AI-aware design standards.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-RIPRA-BD-01', 'ai-capability', 'qs-res-practice', 'AI in Research Practice', 'basic-developing', 'Basic', 'Developing', 'Researchers in social sciences are using personal ChatGPT accounts for qualitative data analysis. The research ethics committee raises concerns about data protection. The PVC Research is asked to respond.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIPRA-BD-01', 'A', 'Researchers should use their grant funding to procure the AI tools they need. Institutional provision would mean subsidising some disciplines over others. Research computing should remain grant-funded.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in ai in research practice. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIPRA-BD-01', 'B', 'Develop an institutional AI research tools provision that addresses the data protection concern. Provide institutionally managed AI tools with appropriate data handling. Assess researcher needs across disciplines.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIPRA-BD-01', 'C', 'Expand institutional AI research infrastructure to cover all disciplines equitably. Integrate AI tools into research data management policies. Provide tiered compute resources matched to research needs.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-RIPRA-BD-02', 'ai-capability', 'qs-res-practice', 'AI in Research Practice', 'basic-developing', 'Basic', 'Developing', 'Several research groups request institutional access to GPU computing for AI-assisted analysis. Currently, researchers rely on grant funding for compute. The CIO asks whether this should be an institutional investment.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIPRA-BD-02', 'A', 'The data protection concerns are valid. Ban the use of personal AI accounts for research data processing. This protects the institution without requiring new infrastructure.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in ai in research practice. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIPRA-BD-02', 'B', 'Invest in institutional GPU compute as shared research infrastructure. Develop acceptable use policies for AI in research. Provide training and support for researchers new to AI tools.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIPRA-BD-02', 'C', 'Implement comprehensive, discipline-equitable AI research infrastructure with specialist support, data management integration, and usage monitoring.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-RIPRA-DA-01', 'ai-capability', 'qs-res-practice', 'AI in Research Practice', 'developing-advanced', 'Developing', 'Advanced', 'The institution provides some AI research tools but a researcher survey shows major disciplinary disparities: STEM researchers have good access while humanities and social science researchers report inadequate AI tools for their needs.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIPRA-DA-01', 'A', 'Address the disciplinary disparity gradually. STEM needs are different from humanities needs. Develop discipline-specific AI tool packages rather than trying to provide everything to everyone.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in ai in research practice.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIPRA-DA-01', 'B', 'Commission a needs assessment across all disciplines and develop equitable provision. AI research tools for humanities and social sciences look different from STEM but are equally important.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIPRA-DA-01', 'C', 'Pilot humanities-focused AI research tools in one faculty before investing in institution-wide provision.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-RIPRA-DA-02', 'ai-capability', 'qs-res-practice', 'AI in Research Practice', 'developing-advanced', 'Developing', 'Advanced', 'AI research tools are available institution-wide but usage data shows that only 25% of researchers use them regularly. The PVC Research asks how to increase adoption while maintaining quality.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIPRA-DA-02', 'A', 'Increase adoption through better training and awareness rather than infrastructure investment. Many researchers do not know what is already available.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in ai in research practice.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIPRA-DA-02', 'B', 'Develop a comprehensive researcher engagement programme alongside infrastructure. Usage data, needs assessment, training, and community building should drive continuous improvement.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIPRA-DA-02', 'C', 'Focus on improving awareness and uptake of existing tools before investing in new infrastructure.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-RISCH-BD-01', 'ai-capability', 'qs-res-scholarship', 'Scholarship of AI in Practice', 'basic-developing', 'Basic', 'Developing', 'An education researcher publishes a paper on AI in assessment that gains national attention. The PVC Research realises no one at the institution has been systematically studying AI''s impact on their own practices.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RISCH-BD-01', 'A', 'Individual academics should research what interests them. The institution should not direct research topics. AI scholarship will develop naturally if there is academic interest.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in scholarship of ai in practice. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RISCH-BD-01', 'B', 'Connect the three academics, establish an informal AI scholarship network, and provide seed funding for collaborative projects. Create visibility for AI scholarship to attract others.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RISCH-BD-01', 'C', 'Establish a formal research centre with dedicated funding, administrative support, and explicit links to institutional AI strategy. Integrate AI scholarship into promotion criteria.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-RISCH-BD-02', 'ai-capability', 'qs-res-scholarship', 'Scholarship of AI in Practice', 'basic-developing', 'Basic', 'Developing', 'Three academics in different faculties are independently researching AI in their disciplines. They are unaware of each other''s work. A colleague suggests the institution should coordinate this activity.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RISCH-BD-02', 'A', 'One publication does not make a research strength. Wait to see if sustained interest develops before investing institutional resource.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in scholarship of ai in practice. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RISCH-BD-02', 'B', 'Establish an institutional AI scholarship initiative with a seminar series, seed funding, and visible support from senior leadership.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RISCH-BD-02', 'C', 'Create the centre with a clear mission, governance, and sustainability plan. Integrate AI scholarship into REF/research assessment strategy.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-RISCH-DA-01', 'ai-capability', 'qs-res-scholarship', 'Scholarship of AI in Practice', 'developing-advanced', 'Developing', 'Advanced', 'A small AI scholarship group has been meeting for a year and has produced several publications. The group proposes creating a formal research centre for AI in education and practice with dedicated funding.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RISCH-DA-01', 'A', 'A formal centre is premature. Continue supporting the existing group with seed funding and see if it grows organically. Centres carry overhead costs.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in scholarship of ai in practice.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RISCH-DA-01', 'B', 'Create the centre with a clear mandate to broaden participation. Include incentives for scholars from diverse disciplines. Link to the institutional AI strategy and establish sustainability.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RISCH-DA-01', 'C', 'Fund a fellowship programme to attract more academics to AI scholarship before creating a formal centre.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-RISCH-DA-02', 'ai-capability', 'qs-res-scholarship', 'Scholarship of AI in Practice', 'developing-advanced', 'Developing', 'Advanced', 'The institution''s AI scholarship output has grown but all publications come from three academics. The PVC Research wants AI scholarship to become a broader institutional strength, not dependent on a few individuals.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RISCH-DA-02', 'A', 'Focus on broadening participation rather than creating structures. More academics doing AI scholarship is more valuable than a centre led by three people.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in scholarship of ai in practice.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RISCH-DA-02', 'B', 'Address the dependency risk by investing in AI scholarship capacity-building across faculties. Create structured pathways for academics to develop AI scholarship alongside their disciplinary research.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RISCH-DA-02', 'C', 'Establish a mentoring programme where the three active researchers support colleagues who want to begin AI scholarship.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-RIAIR-BD-01', 'ai-capability', 'qs-res-airesearch', 'AI Research', 'basic-developing', 'Basic', 'Developing', 'The computer science department has a few researchers publishing in AI, but AI research is not identified as an institutional strength. A major national AI research funding call opens and the institution has no coordinated bid capability.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIAIR-BD-01', 'A', 'AI research requires massive investment that this institution cannot afford. Leave AI research to the leading research universities. Focus on using AI rather than researching it.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in ai research. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIAIR-BD-01', 'B', 'Develop a coordinated AI research strategy. Map existing AI research activity, identify strengths, and target specific funding calls. Create a small cross-faculty AI research group.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIAIR-BD-01', 'C', 'Establish a formal AI research centre with dedicated leadership, infrastructure investment, and a sustainable funding model. Position AI research as an institutional strategic priority.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-RIAIR-BD-02', 'ai-capability', 'qs-res-airesearch', 'AI Research', 'basic-developing', 'Basic', 'Developing', 'A leading AI researcher approaches the university about a professorial appointment. The appointment would give the institution an AI research presence but requires investment in lab infrastructure.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIAIR-BD-02', 'A', 'A single professorial appointment will not create a research strength. The infrastructure costs are prohibitive and the appointment could fail.', 'Basic', 1, true, 'This response sounds pragmatic or respectful of autonomy, but it perpetuates the absence of institutional AI capability in ai research. It avoids establishing any systematic approach.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIAIR-BD-02', 'B', 'Support the appointment as a strategic investment. Develop a business case for the lab infrastructure with a clear link to research income potential.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIAIR-BD-02', 'C', 'Invest strategically. The appointment, combined with infrastructure and a research strategy, can create a foundation for competitive AI research with measurable outcomes.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-RIAIR-DA-01', 'ai-capability', 'qs-res-airesearch', 'AI Research', 'developing-advanced', 'Developing', 'Advanced', 'The institution has an emerging AI research group with a growing publication record. A major industry partner offers a five-year funded AI research programme but requires the institution to establish a formal AI research centre.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIAIR-DA-01', 'A', 'The emerging group is doing well. Continue supporting them and let the research grow organically. A formal centre adds bureaucracy.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in ai research.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIAIR-DA-01', 'B', 'Establish the AI research centre. The industry partnership provides funding and a sustainable model. Build interdisciplinary capacity as a strategic priority.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIAIR-DA-01', 'C', 'Negotiate the industry partnership at a smaller initial scale with growth options. Reduce the upfront risk.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES ('QS-RIAIR-DA-02', 'ai-capability', 'qs-res-airesearch', 'AI Research', 'developing-advanced', 'Developing', 'Advanced', 'AI research output is growing in the computer science department but disciplinary AI research (AI in health, AI in law, AI in arts) remains limited. The PVC Research wants to build interdisciplinary AI research capacity.', 'What would you most likely do?', '{"institution_type":"universal","region":"universal","size":"universal"}'::jsonb, 'active', '{"source": "QS", "licence": "CC BY-SA 4.0"}'::jsonb)
ON CONFLICT (scenario_id) DO UPDATE SET
  framework_id = EXCLUDED.framework_id, dimension_id = EXCLUDED.dimension_id, dimension_name = EXCLUDED.dimension_name,
  target_boundary = EXCLUDED.target_boundary, target_lower_level = EXCLUDED.target_lower_level, target_upper_level = EXCLUDED.target_upper_level,
  stem = EXCLUDED.stem, context_tags = EXCLUDED.context_tags, status = EXCLUDED.status;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIAIR-DA-02', 'A', 'Focus on computer science AI research depth before attempting interdisciplinary breadth. Build strength in one area first.', 'Developing', 2, true, 'This response maintains the status quo of active but non-embedded practice. It sounds cautious and reasonable but avoids the systematic embedding that characterises Advanced capability in ai research.')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIAIR-DA-02', 'B', 'Invest in interdisciplinary AI research capacity through cross-faculty appointments, joint PhD programmes, and targeted funding for AI impact research across disciplines.', 'Advanced', 3, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

INSERT INTO scenario_responses (id, scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
VALUES (gen_random_uuid(), 'QS-RIAIR-DA-02', 'C', 'Create a virtual AI research network before establishing a physical centre. Test interdisciplinary appetite first.', 'Developing', 2, false, '')
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text, maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order, is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation;

COMMIT;

-- Verify
SELECT COUNT(*) AS scenario_count FROM scenario_bank WHERE framework_id = 'ai-capability';
SELECT COUNT(*) AS response_count FROM scenario_responses WHERE scenario_id LIKE 'QS-%';