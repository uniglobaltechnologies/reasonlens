import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { query, queryOne, execute } from "../shared/db";
import { requireAuth, AuthError } from "../shared/auth";
import { corsHeaders, handleCors } from "../middleware/cors";
import { scoreSession, ScenarioAnswer } from "../shared/scenario-scoring";

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  try {
    const user = await requireAuth(req);

    if (req.method !== "POST") {
      return { status: 405, headers: corsHeaders(req), body: "Method not allowed" };
    }

    const body = (await req.json()) as { session_id: string };
    if (!body.session_id) {
      return {
        status: 400,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ error: "session_id required" }),
      };
    }

    // Validate session
    const session = await queryOne<{
      id: string;
      status: string;
      framework_id: string;
      scenario_ids: string[];
    }>(
      "SELECT id, status, framework_id, scenario_ids FROM scenario_sessions WHERE id = $1 AND user_id = $2",
      [body.session_id, user.userId]
    );

    if (!session) {
      return {
        status: 404,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ error: "Session not found" }),
      };
    }

    if (session.status !== "in_progress") {
      return {
        status: 409,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ error: "Session is not in progress" }),
      };
    }

    // Fetch all answers with scenario metadata
    const answers = await query<{
      scenario_id: string;
      mapped_level: string;
      dimension_id: string;
      dimension_name: string;
      maps_to_level_order: number;
    }>(
      `SELECT
         sa.scenario_id,
         sa.mapped_level,
         sb.dimension_id,
         sb.dimension_name,
         sr.maps_to_level_order
       FROM scenario_answers sa
       JOIN scenario_bank sb ON sb.scenario_id = sa.scenario_id
       JOIN scenario_responses sr ON sr.id = sa.response_id
       WHERE sa.session_id = $1`,
      [body.session_id]
    );

    if (answers.length === 0) {
      return {
        status: 400,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ error: "No answers recorded for this session" }),
      };
    }

    // Score the session
    const scoringInput: ScenarioAnswer[] = answers.map((a) => ({
      scenario_id: a.scenario_id,
      dimension_id: a.dimension_id,
      dimension_name: a.dimension_name,
      mapped_level: a.mapped_level,
      level_order: a.maps_to_level_order,
    }));

    const results = scoreSession(scoringInput);

    // Look up framework name for assessment_results
    const frameworkName = session.framework_id; // Will use framework_id as name fallback

    // Write results to assessment_results with assessment_method = 'scenario'
    for (const r of results) {
      await execute(
        `INSERT INTO assessment_results (user_id, framework_id, framework_name, question_id, dimension, selected_level, assessment_method)
         VALUES ($1, $2, $3, $4, $5, $6, 'scenario')`,
        [
          user.userId,
          session.framework_id,
          frameworkName,
          `scenario-session-${body.session_id}`,
          r.dimension_name,
          r.assigned_level,
        ]
      );
    }

    // Update framework progress
    await execute(
      `INSERT INTO framework_progress (user_id, framework_id, framework_name, progress, completed_items, total_items, last_activity)
       VALUES ($1, $2, $3, 100, $4, $4, now())
       ON CONFLICT (user_id, framework_id) DO UPDATE SET
         progress = 100, completed_items = $4, total_items = $4, last_activity = now(), updated_at = now()`,
      [user.userId, session.framework_id, frameworkName, results.length]
    );

    // Mark session as completed
    await execute(
      "UPDATE scenario_sessions SET status = 'completed', completed_at = now() WHERE id = $1",
      [body.session_id]
    );

    return {
      status: 200,
      headers: { ...corsHeaders(req), "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: body.session_id,
        framework_id: session.framework_id,
        results,
      }),
    };
  } catch (err) {
    if (err instanceof AuthError) {
      return { status: err.statusCode, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: err.message }) };
    }
    context.error("scenario-session-complete error:", err);
    return { status: 500, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: "Internal server error" }) };
  }
}

app.http("scenario-session-complete", {
  methods: ["POST", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
