import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { query, queryOne, execute } from "../shared/db";
import { requireAuth, guestAuth, AuthError } from "../shared/auth";
import { corsHeaders, handleCors } from "../middleware/cors";
import {
  getTheScenarioContextScore,
  hasCompleteTheContext,
  listMissingTheContextFields,
  normalizeTheBoundary,
} from "../shared/maturity-the";

// Fisher-Yates shuffle
function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

interface AssessmentContextRow {
  subject_area?: string | null;
  institution_size?: string | null;
  institution_type?: string | null;
  institution_level?: string | null;
  region?: string | null;
  funding_model?: string | null;
  respondent_role?: string | null;
  respondent_institutional_visibility?: string | null;
  digital_infrastructure_baseline?: string | null;
  current_ai_tools?: string[] | null;
  primary_frustration?: string | null;
  years_of_experience?: string | null;
  management_responsibility?: string | null;
}

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  try {
    if (req.method === "GET") {
      const user = await requireAuth(req);
      const frameworkId = req.query.get("framework_id");
      const rows = frameworkId
        ? await query(
            `SELECT id, framework_id, status, started_at, completed_at,
                    (SELECT COUNT(*) FROM scenario_answers WHERE session_id = s.id) AS answered_count,
                    array_length(scenario_ids, 1) AS total_count
             FROM scenario_sessions s
             WHERE user_id = $1 AND framework_id = $2
             ORDER BY started_at DESC`,
            [user.userId, frameworkId]
          )
        : await query(
            `SELECT id, framework_id, status, started_at, completed_at,
                    (SELECT COUNT(*) FROM scenario_answers WHERE session_id = s.id) AS answered_count,
                    array_length(scenario_ids, 1) AS total_count
             FROM scenario_sessions s
             WHERE user_id = $1
             ORDER BY started_at DESC LIMIT 50`,
            [user.userId]
          );
      return {
        status: 200,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ sessions: rows }),
      };
    }

    if (req.method === "POST") {
      const body = (await req.json()) as { framework_id: string; pillar_filter?: string };
      if (!body.framework_id) {
        return {
          status: 400,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "framework_id required" }),
        };
      }

      // DMI assessment allows guest (anonymous) access
      const user = body.framework_id === "maturity-the"
        ? guestAuth(req)
        : await requireAuth(req);

      // Fetch active scenarios for this framework
      const scenarios = await query<{
        scenario_id: string;
        dimension_id: string;
        dimension_name: string;
        target_boundary: string;
        stem: string;
        question: string;
        context_tags: Record<string, unknown>;
      }>(
        "SELECT scenario_id, dimension_id, dimension_name, target_boundary, stem, question, context_tags FROM scenario_bank WHERE framework_id = $1 AND status = 'active' ORDER BY scenario_id",
        [body.framework_id]
      );

      if (scenarios.length === 0) {
        return {
          status: 404,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "No active scenarios found for this framework" }),
        };
      }

      // Fetch user context for snapshot
      const userContext = await queryOne<AssessmentContextRow>(
        "SELECT subject_area, institution_size, institution_type, institution_level, region, funding_model, respondent_role, respondent_institutional_visibility, digital_infrastructure_baseline, current_ai_tools, primary_frustration, years_of_experience, management_responsibility FROM user_assessment_context WHERE user_id = $1",
        [user.userId]
      );

      if (body.framework_id === "maturity-the" && !hasCompleteTheContext(userContext ?? {})) {
        return {
          status: 400,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({
            error: "Institutional assessment context is incomplete",
            missing_fields: listMissingTheContextFields(userContext ?? {}),
          }),
        };
      }

      // Fetch response options for each scenario (strip nuisance metadata)
      const scenarioIds = scenarios.map((s) => s.scenario_id);
      const responses = await query<{
        id: string;
        scenario_id: string;
        response_key: string;
        response_text: string;
        maps_to_level_name: string;
        maps_to_level_order: number;
      }>(
        "SELECT id, scenario_id, response_key, response_text FROM scenario_responses WHERE scenario_id = ANY($1)",
        [scenarioIds]
      );

      // Group and shuffle responses per scenario
      const responsesByScenario = new Map<string, typeof responses>();
      for (const r of responses) {
        if (!responsesByScenario.has(r.scenario_id))
          responsesByScenario.set(r.scenario_id, []);
        responsesByScenario.get(r.scenario_id)!.push(r);
      }

      // Apply pillar filter if provided (e.g., from triage recommendation)
      const PILLAR_PREFIX: Record<string, Record<string, string>> = {
        "maturity-the": {
          teaching_learning: "the-tl-", research: "the-re-",
          professional_services: "the-ps-", planning_governance: "the-pg-",
        },
        "ai-capability": {
          governance: "qs-gov-", outreach: "qs-out-",
          teaching: "qs-tl-", research: "qs-res-",
        },
        // DigComp: 5 competence areas (Area 1–5 map to dc-1, dc-2, etc.)
        "digcomp": {
          information_data: "dc-1-", communication: "dc-2-",
          content_creation: "dc-3-", safety: "dc-4-", problem_solving: "dc-5-",
        },
      };
      const fwPrefixes = PILLAR_PREFIX[body.framework_id] || {};
      const pillarPrefix = body.pillar_filter ? fwPrefixes[body.pillar_filter] : null;
      const filteredScenarios = pillarPrefix
        ? scenarios.filter((s) => s.dimension_id.startsWith(pillarPrefix))
        : scenarios;

      const selectedScenarios = body.framework_id === "maturity-the"
        ? selectTheScenarios(filteredScenarios, userContext ?? {})
        : filteredScenarios;

      // Build scenario list with shuffled responses
      const shuffledScenarios = shuffle(selectedScenarios).map((s) => ({
        scenario_id: s.scenario_id,
        dimension_name: s.dimension_name,
        stem: s.stem,
        question: s.question,
        responses: shuffle(responsesByScenario.get(s.scenario_id) ?? []).map((r) => ({
          id: r.id,
          text: r.response_text,
        })),
      }));

      // Create session
      const orderedIds = shuffledScenarios.map((s) => s.scenario_id);
      const session = await queryOne<{ id: string }>(
        `INSERT INTO scenario_sessions (user_id, framework_id, context_snapshot, scenario_ids)
         VALUES ($1, $2, $3::jsonb, $4)
         RETURNING id`,
        [user.userId, body.framework_id, JSON.stringify(userContext ?? {}), orderedIds]
      );

      return {
        status: 201,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: session!.id,
          framework_id: body.framework_id,
          estimated_time_minutes: (() => {
            const n = shuffledScenarios.length;
            if (body.framework_id === "maturity-the") return pillarPrefix ? Math.ceil(n * 1) : 40;
            if (body.framework_id === "ai-capability") return pillarPrefix ? Math.ceil(n * 0.4) : 20;
            if (body.framework_id === "digcomp") return pillarPrefix ? Math.ceil(n * 0.5) : Math.ceil(n * 0.5); // ~63 mins full, ~13 per area
            // Small frameworks: ~2 min per scenario
            return Math.max(5, Math.ceil(n * 2));
          })(),
          total_scenarios: shuffledScenarios.length,
          scenarios: shuffledScenarios,
        }),
      };
    }

    return { status: 405, headers: corsHeaders(req), body: "Method not allowed" };
  } catch (err) {
    if (err instanceof AuthError) {
      return { status: err.statusCode, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: err.message }) };
    }
    context.error("scenario-sessions error:", err);
    return { status: 500, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: "Internal server error" }) };
  }
}

function selectTheScenarios<
  T extends {
    scenario_id: string;
    dimension_id: string;
    target_boundary: string;
    context_tags: Record<string, unknown>;
  },
>(scenarios: T[], context: AssessmentContextRow): T[] {
  const grouped = new Map<string, T[]>();

  for (const scenario of scenarios) {
    const key = `${scenario.dimension_id}|${normalizeTheBoundary(scenario.target_boundary)}`;
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key)!.push(scenario);
  }

  const selected: T[] = [];

  for (const group of grouped.values()) {
    const ranked = [...group].sort((a, b) => {
      const scoreDiff =
        getTheScenarioContextScore(b.context_tags, context) -
        getTheScenarioContextScore(a.context_tags, context);
      if (scoreDiff !== 0) return scoreDiff;
      return a.scenario_id.localeCompare(b.scenario_id);
    });

    selected.push(...ranked.slice(0, Math.min(2, ranked.length)));
  }

  return selected;
}

app.http("scenario-sessions", {
  methods: ["GET", "POST", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
