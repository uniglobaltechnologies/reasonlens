import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { query, queryOne, execute } from "../shared/db";
import { requireAuth, guestAuth, AuthError } from "../shared/auth";
import { corsHeaders, handleCors } from "../middleware/cors";

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  try {
    if (req.method !== "POST") {
      return { status: 405, headers: corsHeaders(req), body: "Method not allowed" };
    }

    const body = (await req.json()) as {
      session_id: string;
      scenario_id: string;
      response_id: string;
      time_to_respond_seconds?: number;
    };

    if (!body.session_id || !body.scenario_id || !body.response_id) {
      return {
        status: 400,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ error: "session_id, scenario_id, and response_id required" }),
      };
    }

    // Check if this session is a DMI guest session; if so allow guest auth
    const sessionCheck = await queryOne<{ framework_id: string }>(
      "SELECT framework_id FROM scenario_sessions WHERE id = $1",
      [body.session_id]
    );
    const user = sessionCheck?.framework_id === "maturity-the"
      ? guestAuth(req)
      : await requireAuth(req);

    // Validate session belongs to user and is in_progress
    const session = await queryOne<{ id: string; status: string; scenario_ids: string[] | null }>(
      "SELECT id, status, scenario_ids FROM scenario_sessions WHERE id = $1 AND user_id = $2",
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

    if (!session.scenario_ids?.includes(body.scenario_id)) {
      return {
        status: 400,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ error: "scenario_id is not part of this session" }),
      };
    }

    // Look up the response to get mapped level
    const response = await queryOne<{
      maps_to_level_name: string;
      maps_to_level_order: number;
      scenario_id: string;
    }>(
      "SELECT maps_to_level_name, maps_to_level_order, scenario_id FROM scenario_responses WHERE id = $1",
      [body.response_id]
    );

    if (!response) {
      return {
        status: 400,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ error: "Invalid response_id" }),
      };
    }

    if (response.scenario_id !== body.scenario_id) {
      return {
        status: 400,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ error: "response_id does not match scenario_id" }),
      };
    }

    // Insert answer (upsert to handle re-answers)
    await execute(
      `INSERT INTO scenario_answers (session_id, scenario_id, response_id, mapped_level, time_to_respond_seconds)
       VALUES ($1, $2, $3, $4, $5)
       ON CONFLICT (session_id, scenario_id) DO UPDATE SET
         response_id = $3, mapped_level = $4, time_to_respond_seconds = $5, answered_at = now()`,
      [
        body.session_id,
        body.scenario_id,
        body.response_id,
        response.maps_to_level_name,
        body.time_to_respond_seconds ?? null,
      ]
    );

    // Get progress
    const progress = await queryOne<{ answered: string; total: string }>(
      `SELECT
         (SELECT COUNT(*) FROM scenario_answers WHERE session_id = $1) AS answered,
         array_length(scenario_ids, 1) AS total
       FROM scenario_sessions WHERE id = $1`,
      [body.session_id]
    );

    return {
      status: 200,
      headers: { ...corsHeaders(req), "Content-Type": "application/json" },
      body: JSON.stringify({
        answered_count: parseInt(progress?.answered ?? "0", 10),
        total_count: parseInt(progress?.total ?? "0", 10),
      }),
    };
  } catch (err) {
    if (err instanceof AuthError) {
      return { status: err.statusCode, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: err.message }) };
    }
    context.error("scenario-answers error:", err);
    return { status: 500, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: "Internal server error" }) };
  }
}

app.http("scenario-answers", {
  methods: ["POST", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
