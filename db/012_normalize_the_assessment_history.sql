-- =============================================================================
-- 012: Normalise historical THE assessment rows written before canonical ids
-- =============================================================================

UPDATE assessment_results
SET framework_name = 'THE Digital Maturity Index'
WHERE framework_id = 'maturity-the'
  AND framework_name = 'maturity-the';

UPDATE framework_progress
SET framework_name = 'THE Digital Maturity Index',
    updated_at = now()
WHERE framework_id = 'maturity-the'
  AND framework_name = 'maturity-the';

UPDATE assessment_results
SET dimension = CASE dimension
  WHEN 'Strategy (T&L)' THEN 'the-tl-strategy'
  WHEN 'People & Culture (T&L)' THEN 'the-tl-people'
  WHEN 'Technology (T&L)' THEN 'the-tl-technology'
  WHEN 'Data (T&L)' THEN 'the-tl-data'
  WHEN 'Utilisation (T&L)' THEN 'the-tl-utilization'
  WHEN 'Strategy (Research)' THEN 'the-re-strategy'
  WHEN 'People & Culture (Research)' THEN 'the-re-people'
  WHEN 'Technology (Research)' THEN 'the-re-technology'
  WHEN 'Data (Research)' THEN 'the-re-data'
  WHEN 'Utilisation (Research)' THEN 'the-re-utilization'
  WHEN 'Strategy (Prof Services)' THEN 'the-ps-strategy'
  WHEN 'People & Culture (Prof Services)' THEN 'the-ps-people'
  WHEN 'Technology (Prof Services)' THEN 'the-ps-technology'
  WHEN 'Data (Prof Services)' THEN 'the-ps-data'
  WHEN 'Utilisation (Prof Services)' THEN 'the-ps-utilization'
  WHEN 'Strategy (Planning & Gov)' THEN 'the-pg-strategy'
  WHEN 'People & Culture (Planning & Gov)' THEN 'the-pg-people'
  WHEN 'Technology (Planning & Gov)' THEN 'the-pg-technology'
  WHEN 'Data (Planning & Gov)' THEN 'the-pg-data'
  WHEN 'Utilisation (Planning & Gov)' THEN 'the-pg-utilization'
  WHEN 'Teaching & Learning: Strategy' THEN 'the-tl-strategy'
  WHEN 'Teaching & Learning: People & Culture' THEN 'the-tl-people'
  WHEN 'Teaching & Learning: Technology' THEN 'the-tl-technology'
  WHEN 'Teaching & Learning: Data' THEN 'the-tl-data'
  WHEN 'Teaching & Learning: Utilisation' THEN 'the-tl-utilization'
  WHEN 'Research: Strategy' THEN 'the-re-strategy'
  WHEN 'Research: People & Culture' THEN 'the-re-people'
  WHEN 'Research: Technology' THEN 'the-re-technology'
  WHEN 'Research: Data' THEN 'the-re-data'
  WHEN 'Research: Utilisation' THEN 'the-re-utilization'
  WHEN 'Professional Services: Strategy' THEN 'the-ps-strategy'
  WHEN 'Professional Services: People & Culture' THEN 'the-ps-people'
  WHEN 'Professional Services: Technology' THEN 'the-ps-technology'
  WHEN 'Professional Services: Data' THEN 'the-ps-data'
  WHEN 'Professional Services: Utilisation' THEN 'the-ps-utilization'
  WHEN 'Planning & Governance: Strategy' THEN 'the-pg-strategy'
  WHEN 'Planning & Governance: People & Culture' THEN 'the-pg-people'
  WHEN 'Planning & Governance: Technology' THEN 'the-pg-technology'
  WHEN 'Planning & Governance: Data' THEN 'the-pg-data'
  WHEN 'Planning & Governance: Utilisation' THEN 'the-pg-utilization'
  ELSE dimension
END
WHERE framework_id = 'maturity-the';
