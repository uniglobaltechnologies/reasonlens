import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { transaction } from "../shared/db";
import { requireAuth, AuthError } from "../shared/auth";
import { corsHeaders, handleCors } from "../middleware/cors";
import { scoreSession, ScenarioAnswer, DimensionResult } from "../shared/scenario-scoring";

class BusinessError extends Error {
  constructor(public statusCode: number, message: string) {
    super(message);
    this.name = "BusinessError";
  }
}

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

    const { results, frameworkId, answeredCount, totalCount } = await transaction(async (client) => {
      // Lock session row to prevent concurrent completion
      const sessionRes = await client.query(
        "SELECT id, status, framework_id, scenario_ids FROM scenario_sessions WHERE id = $1 AND user_id = $2 FOR UPDATE",
        [body.session_id, user.userId]
      );
      const session = sessionRes.rows[0];

      if (!session) {
        throw new BusinessError(404, "Session not found");
      }

      if (session.status !== "in_progress") {
        throw new BusinessError(409, "Session is not in progress");
      }

      // Fetch all answers with scenario metadata
      const answersRes = await client.query(
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
      const answers = answersRes.rows;
      const totalScenarios = session.scenario_ids?.length ?? 0;

      if (answers.length === 0) {
        throw new BusinessError(400, "No answers recorded for this session");
      }

      if (answers.length < totalScenarios) {
        throw new BusinessError(
          400,
          `Only ${answers.length} of ${totalScenarios} scenarios answered. Complete all scenarios before submitting.`
        );
      }

      // Score the session
      const scoringInput: ScenarioAnswer[] = answers.map((a: any) => ({
        scenario_id: a.scenario_id,
        dimension_id: a.dimension_id,
        dimension_name: a.dimension_name,
        mapped_level: a.mapped_level,
        level_order: a.maps_to_level_order,
      }));

      const scored = scoreSession(scoringInput);
      const frameworkName = session.framework_id;

      // Write results to assessment_results
      for (const r of scored) {
        await client.query(
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

      // Update framework progress with correct counts
      await client.query(
        `INSERT INTO framework_progress (user_id, framework_id, framework_name, progress, completed_items, total_items, last_activity)
         VALUES ($1, $2, $3, 100, $4, $5, now())
         ON CONFLICT (user_id, framework_id) DO UPDATE SET
           progress = 100, completed_items = $4, total_items = $5, last_activity = now(), updated_at = now()`,
        [user.userId, session.framework_id, frameworkName, answers.length, totalScenarios]
      );

      // Mark session as completed
      await client.query(
        "UPDATE scenario_sessions SET status = 'completed', completed_at = now() WHERE id = $1",
        [body.session_id]
      );

      return {
        results: scored,
        frameworkId: session.framework_id,
        answeredCount: answers.length,
        totalCount: totalScenarios,
      };
    });

    return {
      status: 200,
      headers: { ...corsHeaders(req), "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: body.session_id,
        framework_id: frameworkId,
        results,
      }),
    };
  } catch (err) {
    if (err instanceof AuthError) {
      return { status: err.statusCode, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: err.message }) };
    }
    if (err instanceof BusinessError) {
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
