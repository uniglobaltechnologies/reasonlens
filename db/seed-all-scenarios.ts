#!/usr/bin/env npx tsx
/**
 * Universal scenario seeder for all frameworks.
 * Reads scenario JSON files from api/src/data/{framework}/ and
 * inserts into scenario_bank + scenario_responses tables.
 *
 * Usage: DATABASE_URL="postgresql://..." npx tsx db/seed-all-scenarios.ts [framework-id]
 * If no framework-id is specified, seeds ALL frameworks that have scenario files.
 */

import { Pool } from "pg";
import * as fs from "fs";
import * as path from "path";

const DB_URL = process.env.DATABASE_URL || "postgresql://rladmin:plOYhiKsL2P4HS5e6WVs10HU@reasonlens-db.postgres.database.azure.com/reasonlens?sslmode=require";

// ── Framework ID mapping (JSON framework_id → DB framework_id) ─────
const FRAMEWORK_ID_MAP: Record<string, string> = {
  "unesco-guidance-ai-education": "guidance-policy",
  "unesco-student-ai-competency": "student-competency",
  "digcomp-3.0": "digcomp",
  "digcomp": "digcomp",
  "ailit": "ailit",
  "dec-ai-literacy": "dec-ai-literacy",
  "jisc-ai-maturity": "jisc-ai-maturity",
  "jisc-digital-maturity": "jisc-digital-maturity",
  // BDC profiles use their own IDs
  "bdc-individual": "bdc-individual",
  "bdc-teacher-he": "bdc-teacher-he",
  "bdc-researcher": "bdc-researcher",
  "bdc-professional-services": "bdc-professional-services",
  "bdc-learning-technology": "bdc-learning-technology",
  "bdc-digital-leader": "bdc-digital-leader",
  "bdc-educational-developer": "bdc-educational-developer",
  "jisc-bdc-role-profiles": "bdc-individual", // shared BDC scenarios default to individual
  // ISTE
  "iste-students": "iste-students",
  "iste-educators": "iste-educators",
  "iste-coaches": "iste-coaches",
  "iste-leaders": "iste-leaders",
};

// ── Boundary code → level name mapping per framework ───────────────
const BOUNDARY_LEVEL_MAP: Record<string, Record<string, { lower: string; upper: string }>> = {
  "guidance-policy": {
    ED: { lower: "Emerging", upper: "Developing" },
    DS: { lower: "Developing", upper: "Established" },
  },
  "student-competency": {
    FI: { lower: "Foundational", upper: "Intermediate" },
    IA: { lower: "Intermediate", upper: "Advanced" },
  },
  "digcomp": {
    BI: { lower: "Basic", upper: "Intermediate" },
    IA: { lower: "Intermediate", upper: "Advanced" },
    AH: { lower: "Advanced", upper: "Highly Advanced" },
  },
  "ailit": {
    NI: { lower: "Novice", upper: "Intermediate" },
    IA: { lower: "Intermediate", upper: "Advanced" },
    AE: { lower: "Advanced", upper: "Expert" },
  },
  "dec-ai-literacy": {
    AE: { lower: "Awareness", upper: "Exploration" },
    EP: { lower: "Exploration", upper: "Practice" },
    PM: { lower: "Practice", upper: "Mastery" },
  },
  "jisc-ai-maturity": {
    ED: { lower: "Emerging", upper: "Developing" },
    DM: { lower: "Developing", upper: "Mature" },
  },
  "jisc-digital-maturity": {
    EE: { lower: "Emerging to Established", upper: "Established to Enhanced" },
    EM: { lower: "Established to Enhanced", upper: "Enhanced to Mature" },
  },
};

// BDC profiles all use the same level names
for (const profile of ["bdc-individual", "bdc-teacher-he", "bdc-researcher", "bdc-professional-services", "bdc-learning-technology", "bdc-digital-leader", "bdc-educational-developer"]) {
  BOUNDARY_LEVEL_MAP[profile] = {
    DC: { lower: "Developing", upper: "Capable" },
    CP: { lower: "Capable", upper: "Proficient" },
  };
}

// ISTE uses met/not-met - boundaries represent standard thresholds
for (const fw of ["iste-students", "iste-educators", "iste-coaches", "iste-leaders"]) {
  BOUNDARY_LEVEL_MAP[fw] = {
    NM: { lower: "Not Met", upper: "Met" },
  };
}

interface ScenarioFile {
  framework_id: string;
  data?: any[];
  scenarios?: any[];
}

interface ResponseOption {
  id: string;
  text: string;
  level?: string;
  level_mapping?: string;
  maps_to_level?: string;
  is_nuisance?: boolean;
  nuisance_explanation?: string | null;
  nuisance_type?: string;
}

function resolveFrameworkId(jsonFwId: string): string {
  return FRAMEWORK_ID_MAP[jsonFwId] || jsonFwId;
}

function getLevelOrder(levelName: string, frameworkId: string): number {
  const orderMaps: Record<string, Record<string, number>> = {
    "guidance-policy": { emerging: 1, developing: 2, established: 3 },
    "student-competency": { foundational: 1, intermediate: 2, advanced: 3 },
    "digcomp": { basic: 1, intermediate: 2, advanced: 3, "highly advanced": 4 },
    "ailit": { novice: 1, intermediate: 2, advanced: 3, expert: 4 },
    "dec-ai-literacy": { awareness: 1, exploration: 2, practice: 3, mastery: 4 },
    "jisc-ai-maturity": { emerging: 1, developing: 2, mature: 3 },
    "jisc-digital-maturity": { "emerging to established": 1, "established to enhanced": 2, "enhanced to mature": 3 },
  };
  // BDC
  for (const p of ["bdc-individual", "bdc-teacher-he", "bdc-researcher", "bdc-professional-services", "bdc-learning-technology", "bdc-digital-leader", "bdc-educational-developer"]) {
    orderMaps[p] = { developing: 1, capable: 2, proficient: 3 };
  }
  // ISTE
  for (const f of ["iste-students", "iste-educators", "iste-coaches", "iste-leaders"]) {
    orderMaps[f] = { "not met": 0, met: 1 };
  }

  const map = orderMaps[frameworkId];
  if (!map) return 0;
  return map[levelName.toLowerCase()] ?? 0;
}

async function seedFramework(pool: Pool, dataDir: string, frameworkFilter?: string) {
  const scenariosPath = path.join(dataDir, "scenarios.json");
  if (!fs.existsSync(scenariosPath)) {
    console.log(`  No scenarios.json found in ${dataDir}, skipping`);
    return 0;
  }

  const raw = JSON.parse(fs.readFileSync(scenariosPath, "utf-8"));

  // Derive framework ID — handle plain arrays (DigComp), various dict keys
  const dirName = path.basename(dataDir);
  let jsonFwId: string;
  if (Array.isArray(raw)) {
    jsonFwId = dirName; // plain array (DigComp)
  } else {
    jsonFwId = raw.framework_id || dirName;
  }
  const dbFrameworkId = resolveFrameworkId(jsonFwId);

  if (frameworkFilter && dbFrameworkId !== frameworkFilter) {
    return 0;
  }

  // Extract scenarios array — handle: data, scenarios, objects, or plain array
  let scenarios: any[];
  if (Array.isArray(raw)) {
    scenarios = raw;
  } else {
    scenarios = raw.data || raw.scenarios || raw.objects || [];
  }
  console.log(`  Seeding ${scenarios.length} scenarios for ${dbFrameworkId}...`);

  let inserted = 0;
  const boundaryMap = BOUNDARY_LEVEL_MAP[dbFrameworkId] || {};

  for (const s of scenarios) {
    const scenarioId = s.id || s.scenario_id;
    // Dimension: dimension_id, category_id, competency_id, standard_id, or element_code
    const dimensionId = s.dimension_id || s.category_id || s.competency_id || s.standard_id || s.element_code || "";
    const dimensionName = s.dimension_name || s.category_name || s.competency_name || s.standard_name || s.element_name || dimensionId;
    // Boundary code
    const boundary = s.boundary || s.boundary_code || s.target_boundary || "";
    // Try from_level/to_level first (Batch C+D), fall back to boundary map
    let boundaryLevels: { lower: string; upper: string };
    if (s.from_level_id && s.to_level_id) {
      // Capitalize first letter
      const cap = (str: string) => str.charAt(0).toUpperCase() + str.slice(1).replace(/_/g, " ");
      boundaryLevels = { lower: cap(s.from_level_id), upper: cap(s.to_level_id) };
    } else {
      boundaryLevels = boundaryMap[boundary] || { lower: "", upper: "" };
    }
    const stem = s.stem || s.scenario_stem || "";
    const question = s.question || "What would you most likely do?";
    const contextTags = { ...(s.context_tags || {}) };

    // For BDC, include applicable_profiles in context_tags for role-based filtering
    const applicableProfiles = s.applicable_profiles || s.profile_scope;
    if (applicableProfiles) {
      contextTags.applicable_profiles = applicableProfiles;
    }

    // For BDC, derive per-profile framework_id if profile_id present
    let effectiveFwId = dbFrameworkId;
    if (s.profile_id && FRAMEWORK_ID_MAP[s.profile_id]) {
      effectiveFwId = s.profile_id;
    }
    // For BDC shared scenarios with profile_scope, use the first applicable profile or "bdc-individual"
    if (dbFrameworkId === "jisc-bdc-role-profiles" || jsonFwId === "jisc-bdc-role-profiles") {
      effectiveFwId = s.profile_id || "bdc-individual";
    }

    // Insert scenario
    await pool.query(`
      INSERT INTO scenario_bank (scenario_id, framework_id, dimension_id, dimension_name,
        target_boundary, target_lower_level, target_upper_level,
        stem, question, context_tags, status)
      VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 'active')
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
        status = 'active'
    `, [scenarioId, effectiveFwId, dimensionId, dimensionName,
        `${boundaryLevels.lower}-${boundaryLevels.upper}`.toLowerCase().replace(/ /g, "_"),
        boundaryLevels.lower, boundaryLevels.upper,
        stem, question, JSON.stringify(contextTags)]);

    // Insert responses — handle response_options or responses array
    const responses = s.response_options || s.responses || [];
    const keys = ["A", "B", "C", "D"];

    for (let i = 0; i < responses.length; i++) {
      const r = responses[i];
      // Response key: last char of id/option_id, or positional A/B/C/D
      const rawId = r.id || r.option_id || "";
      const responseKey = rawId ? rawId.slice(-1).toUpperCase() : (keys[i] || String.fromCharCode(65 + i));
      // Level name: level, level_mapping, maps_to_level, maps_to_level_id, mapped_level, maps_to
      const levelName = r.level || r.level_mapping || r.maps_to_level || r.maps_to_level_id || r.mapped_level || r.maps_to || "";
      const levelOrder = getLevelOrder(levelName, effectiveFwId);
      // Nuisance: is_nuisance, nuisance (bool), is_attractive_nuisance
      const isNuisance = r.is_nuisance ?? r.nuisance ?? r.is_attractive_nuisance ?? false;
      // Nuisance explanation: nuisance_explanation, nuisance_reason
      const nuisanceExplanation = r.nuisance_explanation || r.nuisance_reason || null;

      await pool.query(`
        INSERT INTO scenario_responses (scenario_id, response_key, response_text,
          maps_to_level_name, maps_to_level_order, is_attractive_nuisance, nuisance_explanation)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (scenario_id, response_key) DO UPDATE SET
          response_text = EXCLUDED.response_text,
          maps_to_level_name = EXCLUDED.maps_to_level_name,
          maps_to_level_order = EXCLUDED.maps_to_level_order,
          is_attractive_nuisance = EXCLUDED.is_attractive_nuisance,
          nuisance_explanation = EXCLUDED.nuisance_explanation
      `, [scenarioId, responseKey.slice(-1), r.text,
          levelName, levelOrder, isNuisance, nuisanceExplanation]);
    }

    inserted++;
  }

  console.log(`  ✓ ${inserted} scenarios seeded for ${dbFrameworkId}`);
  return inserted;
}

async function main() {
  const filterFw = process.argv[2];
  const pool = new Pool({ connectionString: DB_URL, ssl: { rejectUnauthorized: false } });

  console.log("Universal Scenario Seeder");
  console.log("========================\n");

  const apiDataDir = path.join(__dirname, "..", "api", "src", "data");
  const dirs = fs.readdirSync(apiDataDir).filter(d =>
    fs.statSync(path.join(apiDataDir, d)).isDirectory() && d !== "cross-framework"
  );

  let total = 0;
  for (const dir of dirs.sort()) {
    const fullPath = path.join(apiDataDir, dir);
    console.log(`Processing ${dir}/...`);
    total += await seedFramework(pool, fullPath, filterFw);
  }

  console.log(`\n✓ Total: ${total} scenarios seeded across all frameworks`);
  await pool.end();
}

main().catch(e => { console.error(e); process.exit(1); });
