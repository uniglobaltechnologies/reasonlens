import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { query, queryOne, execute } from "../shared/db";
import { requireAuth, AuthError } from "../shared/auth";
import { corsHeaders, handleCors } from "../middleware/cors";

// Fisher-Yates shuffle
function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
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
      const body = (await req.json()) as { framework_id: string };
      if (!body.framework_id) {
        return {
          status: 400,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "framework_id required" }),
        };
      }

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
      const userContext = await queryOne(
        "SELECT subject_area, institution_type, institution_level, region, current_ai_tools, primary_frustration, years_of_experience, management_responsibility FROM user_assessment_context WHERE user_id = $1",
        [user.userId]
      );

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

      // Build scenario list with shuffled responses
      const shuffledScenarios = shuffle(scenarios).map((s) => ({
        scenario_id: s.scenario_id,
        dimension_name: s.dimension_name,
        stem: s.stem,
        question: s.question,
        responses: shuffle(responsesByScenario.get(s.scenario_id) ?? []).map(
          (r) => ({
            id: r.id,
            response_key: r.response_key,
            text: r.response_text,
          })
        ),
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

app.http("scenario-sessions", {
  methods: ["GET", "POST", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
