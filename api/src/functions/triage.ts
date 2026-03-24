import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { queryOne, execute } from "../shared/db";
import { requireAuth, AuthError } from "../shared/auth";
import { corsHeaders, handleCors } from "../middleware/cors";

type Signal = string; // Framework-dependent: THE uses incidental/intentional/integrated/optimised; QS uses basic/developing/advanced

// ─── THE DMI signals ───
const THE_SIGNAL_ORDER: Record<string, number> = {
  incidental: 1, intentional: 2, integrated: 3, optimised: 4,
};
const THE_SIGNAL_CATEGORY: Record<string, string> = {
  incidental: "needs_attention", intentional: "progress_underway",
  integrated: "functioning_well", optimised: "sector_leading",
};
const THE_PILLAR_KEYS = ["teaching_learning", "research", "professional_services", "planning_governance"];
const THE_PILLAR_NAMES: Record<string, string> = {
  teaching_learning: "Teaching & Learning", research: "Research",
  professional_services: "Professional Services", planning_governance: "Planning & Governance",
};

// ─── QS AI Capability signals ───
const QS_SIGNAL_ORDER: Record<string, number> = {
  basic: 1, developing: 2, advanced: 3,
};
const QS_SIGNAL_CATEGORY: Record<string, string> = {
  basic: "needs_attention", developing: "progress_underway", advanced: "functioning_well",
};
const QS_PILLAR_KEYS = ["governance", "outreach", "teaching", "research"];
const QS_PILLAR_NAMES: Record<string, string> = {
  governance: "Governance & Human Commitment", outreach: "Outreach & Operational Efficiency",
  teaching: "Teaching, Learning & Assessment", research: "Research & Scholarship",
};

// Framework-aware getters
function getPillarKeys(frameworkId: string): string[] {
  return frameworkId === "ai-capability" ? QS_PILLAR_KEYS : THE_PILLAR_KEYS;
}
function getPillarNames(frameworkId: string): Record<string, string> {
  return frameworkId === "ai-capability" ? QS_PILLAR_NAMES : THE_PILLAR_NAMES;
}
function getSignalOrder(frameworkId: string): Record<string, number> {
  return frameworkId === "ai-capability" ? QS_SIGNAL_ORDER : THE_SIGNAL_ORDER;
}
function getSignalCategory(frameworkId: string): Record<string, string> {
  return frameworkId === "ai-capability" ? QS_SIGNAL_CATEGORY : THE_SIGNAL_CATEGORY;
}
function getScenariosPerLowPillar(frameworkId: string): number {
  // QS: ~14 categories × 2 boundaries / 4 pillars = 7 per pillar → ~14 scenarios
  // THE: 5 dimensions × 2 scenarios = 10 per pillar
  return frameworkId === "ai-capability" ? 14 : 10;
}

// Which pillars are most visible to each role
const ROLE_PILLAR_AFFINITY: Record<string, string[]> = {
  // THE roles
  senior_leadership: ["planning_governance", "teaching_learning", "research", "professional_services"],
  faculty_leader: ["teaching_learning", "research"],
  ps_director: ["professional_services", "planning_governance"],
  department_head: ["teaching_learning", "research"],
  academic_staff: ["teaching_learning", "research"],
  ps_staff: ["professional_services"],
  // QS roles
  CIO: ["governance", "outreach"],
  academic_leadership: ["teaching", "research"],
  research_leadership: ["research", "governance"],
  professional_services: ["outreach", "governance"],
};

// Which dimension maps to which pillars for tiebreaking
const DIMENSION_PILLAR_MAP: Record<string, string[]> = {
  // THE dimensions
  strategy: ["planning_governance", "teaching_learning", "research", "professional_services"],
  people_culture: ["teaching_learning", "professional_services", "research", "planning_governance"],
  technology: ["professional_services", "teaching_learning", "research", "planning_governance"],
  data: ["professional_services", "research", "planning_governance", "teaching_learning"],
  utilisation: ["teaching_learning", "research", "professional_services", "planning_governance"],
  // QS dimensions (map to QS pillar keys)
  regulatory: ["governance", "outreach", "teaching", "research"],
  curriculum: ["teaching", "research", "outreach", "governance"],
  ai_research: ["research", "teaching", "governance", "outreach"],
  recruitment: ["outreach", "governance", "teaching", "research"],
};

interface TriageBody {
  framework_id: string;
  respondent_role: string;
  respondent_visibility: string;
  pillar_responses: Record<string, Signal>;
  perceived_priority_dimension?: string;
}

function computeRecommendation(
  pillarResponses: Record<string, Signal>,
  perceivedPriority: string | undefined,
  role: string,
  visibility: string,
  frameworkId: string = "maturity-the"
): { pillar: string; reason: string } {
  const pillarKeys = getPillarKeys(frameworkId);
  const signalOrder = getSignalOrder(frameworkId);

  // Find lowest-signal pillar(s)
  let minOrder = Infinity;
  for (const key of pillarKeys) {
    const signal = pillarResponses[key];
    if (signal && signalOrder[signal] < minOrder) {
      minOrder = signalOrder[signal];
    }
  }

  const lowestPillars = pillarKeys.filter(
    (key) => pillarResponses[key] && signalOrder[pillarResponses[key]] === minOrder
  );

  if (lowestPillars.length === 1) {
    return { pillar: lowestPillars[0], reason: "Lowest signal level among pillars" };
  }

  // Tiebreak 1: use perceived priority dimension
  if (perceivedPriority && DIMENSION_PILLAR_MAP[perceivedPriority]) {
    const affinePillars = DIMENSION_PILLAR_MAP[perceivedPriority];
    for (const p of affinePillars) {
      if ((lowestPillars as readonly string[]).includes(p)) {
        return { pillar: p, reason: `Lowest signal, aligned with your identified priority (${perceivedPriority.replace(/_/g, " ")})` };
      }
    }
  }

  // Tiebreak 2: use role affinity (recommend pillar where respondent has best visibility)
  const roleAffinity = ROLE_PILLAR_AFFINITY[role] || [];
  for (const p of roleAffinity) {
    if ((lowestPillars as readonly string[]).includes(p)) {
      return { pillar: p, reason: "Lowest signal in an area you have strong visibility into" };
    }
  }

  // Fallback: first lowest pillar
  return { pillar: lowestPillars[0], reason: "Lowest signal level among pillars" };
}

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  try {
    const user = await requireAuth(req);

    if (req.method === "GET") {
      // Return most recent triage for this user + framework
      const frameworkId = req.query.get("framework_id") || "maturity-the";
      const row = await queryOne(
        "SELECT * FROM triage_results WHERE user_id = $1 AND framework_id = $2 ORDER BY created_at DESC LIMIT 1",
        [user.userId, frameworkId]
      );
      return {
        status: row ? 200 : 404,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify(row ?? { error: "No triage results found" }),
      };
    }

    if (req.method !== "POST") {
      return { status: 405, headers: corsHeaders(req), body: "Method not allowed" };
    }

    const body = (await req.json()) as TriageBody;

    if (!body.respondent_role || !body.respondent_visibility || !body.pillar_responses) {
      return {
        status: 400,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ error: "respondent_role, respondent_visibility, and pillar_responses are required" }),
      };
    }

    // Validate pillar responses (framework-aware)
    const fwId = body.framework_id || "maturity-the";
    const pillarKeys = getPillarKeys(fwId);
    const signalOrder = getSignalOrder(fwId);

    for (const key of pillarKeys) {
      const signal = body.pillar_responses[key];
      if (!signal || !signalOrder[signal]) {
        return {
          status: 400,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: `Invalid or missing signal for pillar "${key}"` }),
        };
      }
    }

    // Compute recommendation
    const recommendation = computeRecommendation(
      body.pillar_responses,
      body.perceived_priority_dimension,
      body.respondent_role,
      body.respondent_visibility,
      fwId
    );

    const lowThreshold = fwId === "ai-capability" ? 1 : 2; // basic or incidental/intentional
    const scenariosPerPillar = getScenariosPerLowPillar(fwId);
    const scenarioCount = pillarKeys.filter(
      (k) => signalOrder[body.pillar_responses[k]] <= lowThreshold
    ).length * scenariosPerPillar;

    // Store triage result
    const row = await queryOne<{ id: string }>(
      `INSERT INTO triage_results (user_id, framework_id, respondent_role, respondent_visibility, pillar_signals, perceived_priority_dimension, recommended_pillar)
       VALUES ($1, $2, $3, $4, $5, $6, $7)
       RETURNING id`,
      [
        user.userId,
        body.framework_id || "maturity-the",
        body.respondent_role,
        body.respondent_visibility,
        JSON.stringify(body.pillar_responses),
        body.perceived_priority_dimension || null,
        recommendation.pillar,
      ]
    );

    // Update user_assessment_context with role + visibility
    await execute(
      `INSERT INTO user_assessment_context (user_id, respondent_role, respondent_institutional_visibility)
       VALUES ($1, $2, $3)
       ON CONFLICT (user_id) DO UPDATE SET
         respondent_role = COALESCE($2, user_assessment_context.respondent_role),
         respondent_institutional_visibility = COALESCE($3, user_assessment_context.respondent_institutional_visibility),
         updated_at = now()`,
      [user.userId, body.respondent_role, body.respondent_visibility]
    );

    // Build response
    const signalCategory = getSignalCategory(fwId);
    const pillarNames = getPillarNames(fwId);
    const pillarSignals: Record<string, { signal: Signal; category: string; name: string }> = {};
    for (const key of pillarKeys) {
      const signal = body.pillar_responses[key];
      pillarSignals[key] = {
        signal,
        category: signalCategory[signal] || "unknown",
        name: pillarNames[key] || key,
      };
    }

    const roleLabel = body.respondent_role.replace(/_/g, " ");
    const visibilityLabel = body.respondent_visibility.replace(/_/g, " ");

    return {
      status: 200,
      headers: { ...corsHeaders(req), "Content-Type": "application/json" },
      body: JSON.stringify({
        triage_id: row?.id,
        pillar_signals: pillarSignals,
        perceived_priority: body.perceived_priority_dimension || null,
        recommendation: {
          pillar: recommendation.pillar,
          pillar_name: pillarNames[recommendation.pillar] || recommendation.pillar,
          reason: recommendation.reason,
          scenario_count: Math.max(scenarioCount, 10),
          estimated_time_minutes: Math.max(Math.ceil(scenarioCount / 2.5), 4),
        },
        visibility_note: `Results reflect a ${roleLabel} perspective with ${visibilityLabel} visibility`,
      }),
    };
  } catch (err) {
    if (err instanceof AuthError) {
      return { status: err.statusCode, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: err.message }) };
    }
    context.error("triage error:", err);
    return { status: 500, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: "Internal server error" }) };
  }
}

app.http("triage", {
  methods: ["GET", "POST", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
