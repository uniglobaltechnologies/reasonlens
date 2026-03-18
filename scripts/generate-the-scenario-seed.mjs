import fs from "fs";
import path from "path";

const repoRoot = process.cwd();
const bundlePath = path.join(repoRoot, "data", "the-dmi", "scenarios.json");
const outputPath = path.join(repoRoot, "db", "009_seed_the_scenarios.sql");

const DIMENSION_ID_MAP = {
  "the-tl-str": "the-tl-strategy",
  "the-tl-ppl": "the-tl-people",
  "the-tl-tec": "the-tl-technology",
  "the-tl-dat": "the-tl-data",
  "the-tl-uti": "the-tl-utilization",
  "the-re-str": "the-re-strategy",
  "the-re-ppl": "the-re-people",
  "the-re-tec": "the-re-technology",
  "the-re-dat": "the-re-data",
  "the-re-uti": "the-re-utilization",
  "the-ps-str": "the-ps-strategy",
  "the-ps-ppl": "the-ps-people",
  "the-ps-tec": "the-ps-technology",
  "the-ps-dat": "the-ps-data",
  "the-ps-uti": "the-ps-utilization",
  "the-pg-str": "the-pg-strategy",
  "the-pg-ppl": "the-pg-people",
  "the-pg-tec": "the-pg-technology",
  "the-pg-dat": "the-pg-data",
  "the-pg-uti": "the-pg-utilization",
};

const DIMENSION_NAME_MAP = {
  "the-tl-strategy": "Teaching & Learning: Strategy",
  "the-tl-people": "Teaching & Learning: People & Culture",
  "the-tl-technology": "Teaching & Learning: Technology",
  "the-tl-data": "Teaching & Learning: Data",
  "the-tl-utilization": "Teaching & Learning: Utilisation",
  "the-re-strategy": "Research: Strategy",
  "the-re-people": "Research: People & Culture",
  "the-re-technology": "Research: Technology",
  "the-re-data": "Research: Data",
  "the-re-utilization": "Research: Utilisation",
  "the-ps-strategy": "Professional Services: Strategy",
  "the-ps-people": "Professional Services: People & Culture",
  "the-ps-technology": "Professional Services: Technology",
  "the-ps-data": "Professional Services: Data",
  "the-ps-utilization": "Professional Services: Utilisation",
  "the-pg-strategy": "Planning & Governance: Strategy",
  "the-pg-people": "Planning & Governance: People & Culture",
  "the-pg-technology": "Planning & Governance: Technology",
  "the-pg-data": "Planning & Governance: Data",
  "the-pg-utilization": "Planning & Governance: Utilisation",
};

const LEVEL_NAME_BY_ORDER = {
  1: "Incidental",
  2: "Intentional",
  3: "Integrated",
  4: "Optimised",
};

function sqlString(value) {
  if (value === null || value === undefined) return "NULL";
  return `'${String(value).replace(/'/g, "''")}'`;
}

function sqlJson(value) {
  return `${sqlString(JSON.stringify(value))}::jsonb`;
}

function normalizeBoundary(boundary) {
  return boundary.toLowerCase().trim().replace(/\s+/g, "-");
}

function normalizeDimensionId(id) {
  if (!DIMENSION_ID_MAP[id]) {
    throw new Error(`Unknown THE dimension id: ${id}`);
  }
  return DIMENSION_ID_MAP[id];
}

function normalizeResponseLevel(scenario, response) {
  if (response.level !== "Below boundary") {
    return {
      name: response.level,
      order: Object.entries(LEVEL_NAME_BY_ORDER).find(([, name]) => name === response.level)?.[0],
    };
  }

  const order = Math.max(1, scenario.lower_level.order - 1);
  return {
    name: LEVEL_NAME_BY_ORDER[order],
    order: String(order),
  };
}

function buildScenarioInsert(scenario) {
  const dimensionId = normalizeDimensionId(scenario.dimension_id);
  const dimensionName = DIMENSION_NAME_MAP[dimensionId] ?? scenario.dimension_name;
  const targetBoundary = normalizeBoundary(scenario.boundary_tested);
  const sourceAttribution = {
    source_framework: "THE Digital Maturity Index",
    content_type: "original",
    attribution_text:
      "Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.",
    share_alike_applies: false,
  };

  return `INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name, target_boundary, target_lower_level, target_upper_level, stem, question, context_tags, status, source_attribution)
VALUES (${sqlString(scenario.scenario_id)}, 'maturity-the', ${sqlString(dimensionId)}, ${sqlString(dimensionName)}, ${sqlString(targetBoundary)}, ${sqlString(scenario.lower_level.name)}, ${sqlString(scenario.upper_level.name)}, ${sqlString(scenario.scenario_text)}, 'What would you most likely do?', ${sqlJson(scenario.context_tags ?? {})}, 'active', ${sqlJson(sourceAttribution)})
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
  updated_at = now();`;
}

function buildResponseInsert(scenario, response, index) {
  const normalised = normalizeResponseLevel(scenario, response);
  const responseKey = String.fromCharCode(65 + index);
  const nuisanceExplanation = response.is_attractive_nuisance
    ? response.explanation
    : null;
  const discriminatingPresence = response.is_attractive_nuisance
    ? null
    : response.explanation;

  return `INSERT INTO scenario_responses (scenario_id, response_key, response_text, maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation, discriminating_presence, discriminating_absence)
VALUES (${sqlString(scenario.scenario_id)}, ${sqlString(responseKey)}, ${sqlString(response.text)}, ${sqlString(normalised.name)}, ${normalised.order}, ${response.is_attractive_nuisance ? "true" : "false"}, ${sqlString(nuisanceExplanation)}, ${sqlString(discriminatingPresence)}, NULL)
ON CONFLICT (scenario_id, response_key) DO UPDATE SET
  response_text = EXCLUDED.response_text,
  maps_to_level_name = EXCLUDED.maps_to_level_name,
  maps_to_level_order = EXCLUDED.maps_to_level_order,
  is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
  nuisance_explanation = EXCLUDED.nuisance_explanation,
  discriminating_presence = EXCLUDED.discriminating_presence,
  discriminating_absence = EXCLUDED.discriminating_absence;`;
}

const scenarios = JSON.parse(fs.readFileSync(bundlePath, "utf8"));
const scenarioIds = scenarios.map((scenario) => sqlString(scenario.scenario_id)).join(", ");

const lines = [
  "-- =============================================================================",
  "-- 009: Seed THE Digital Maturity Index scenarios",
  "-- Generated from data/the-dmi/scenarios.json",
  "-- 20 child dimensions x 3 boundaries x 2 scenarios = 120 scenarios",
  "-- =============================================================================",
  "",
  "BEGIN;",
  "",
  "-- Retire legacy THE scenarios that are no longer in the active production bank.",
  `UPDATE scenario_bank
SET status = 'retired', updated_at = now()
WHERE framework_id = 'maturity-the'
  AND scenario_id NOT IN (${scenarioIds});`,
  "",
];

for (const scenario of scenarios) {
  lines.push(`-- ${scenario.scenario_id} :: ${scenario.dimension_name} :: ${scenario.boundary_tested}`);
  lines.push(buildScenarioInsert(scenario));
  lines.push("");

  scenario.response_options.forEach((response, index) => {
    lines.push(buildResponseInsert(scenario, response, index));
  });

  lines.push("");
}

lines.push("COMMIT;");
lines.push("");

fs.writeFileSync(outputPath, `${lines.join("\n")}`);
console.log(`Wrote ${outputPath} from ${bundlePath}`);
