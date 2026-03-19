import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { queryOne, execute } from "../shared/db";
import { requireAuth, AuthError } from "../shared/auth";
import { corsHeaders, handleCors } from "../middleware/cors";

type Signal = "incidental" | "intentional" | "integrated" | "optimised";

const SIGNAL_ORDER: Record<Signal, number> = {
  incidental: 1,
  intentional: 2,
  integrated: 3,
  optimised: 4,
};

const SIGNAL_CATEGORY: Record<Signal, string> = {
  incidental: "needs_attention",
  intentional: "progress_underway",
  integrated: "functioning_well",
  optimised: "sector_leading",
};

const PILLAR_KEYS = [
  "teaching_learning",
  "research",
  "professional_services",
  "planning_governance",
] as const;

const PILLAR_NAMES: Record<string, string> = {
  teaching_learning: "Teaching & Learning",
  research: "Research",
  professional_services: "Professional Services",
  planning_governance: "Planning & Governance",
};

// Which pillars are most visible to each role
const ROLE_PILLAR_AFFINITY: Record<string, string[]> = {
  senior_leadership: ["planning_governance", "teaching_learning", "research", "professional_services"],
  faculty_leader: ["teaching_learning", "research"],
  ps_director: ["professional_services", "planning_governance"],
  department_head: ["teaching_learning", "research"],
  academic_staff: ["teaching_learning", "research"],
  ps_staff: ["professional_services"],
};

// Which dimension maps to which pillars for tiebreaking
const DIMENSION_PILLAR_MAP: Record<string, string[]> = {
  strategy: ["planning_governance", "teaching_learning", "research", "professional_services"],
  people_culture: ["teaching_learning", "professional_services", "research", "planning_governance"],
  technology: ["professional_services", "teaching_learning", "research", "planning_governance"],
  data: ["professional_services", "research", "planning_governance", "teaching_learning"],
  utilisation: ["teaching_learning", "research", "professional_services", "planning_governance"],
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
  visibility: string
): { pillar: string; reason: string } {
  // Find lowest-signal pillar(s)
  let minOrder = Infinity;
  for (const key of PILLAR_KEYS) {
    const signal = pillarResponses[key];
    if (signal && SIGNAL_ORDER[signal] < minOrder) {
      minOrder = SIGNAL_ORDER[signal];
    }
  }

  const lowestPillars = PILLAR_KEYS.filter(
    (key) => pillarResponses[key] && SIGNAL_ORDER[pillarResponses[key]] === minOrder
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

    // Validate pillar responses
    for (const key of PILLAR_KEYS) {
      const signal = body.pillar_responses[key];
      if (!signal || !SIGNAL_ORDER[signal]) {
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
      body.respondent_visibility
    );

    const scenarioCount = PILLAR_KEYS.filter(
      (k) => SIGNAL_ORDER[body.pillar_responses[k]] <= 2 // Incidental or Intentional
    ).length * 10; // 5 dimensions * 2 scenarios

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
    const pillarSignals: Record<string, { signal: Signal; category: string; name: string }> = {};
    for (const key of PILLAR_KEYS) {
      const signal = body.pillar_responses[key];
      pillarSignals[key] = {
        signal,
        category: SIGNAL_CATEGORY[signal],
        name: PILLAR_NAMES[key],
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
          pillar_name: PILLAR_NAMES[recommendation.pillar],
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
