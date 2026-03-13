import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { query } from "../shared/db";
import { requireRole, AuthError } from "../shared/auth";
import { corsHeaders, handleCors } from "../middleware/cors";

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  try {
    await requireRole(req, "admin");

    const frameworkId = req.query.get("framework_id") || "teacher-competency";

    // Option selection distributions per scenario
    const distributions = await query(
      `SELECT
         sb.scenario_id,
         sb.dimension_name,
         sr.response_key,
         sr.maps_to_level_name,
         sr.is_attractive_nuisance,
         COUNT(sa.id) AS selection_count,
         ROUND(COUNT(sa.id)::numeric / NULLIF(SUM(COUNT(sa.id)) OVER (PARTITION BY sb.scenario_id), 0) * 100, 1) AS selection_pct
       FROM scenario_bank sb
       JOIN scenario_responses sr ON sr.scenario_id = sb.scenario_id
       LEFT JOIN scenario_answers sa ON sa.response_id = sr.id
       WHERE sb.framework_id = $1
       GROUP BY sb.scenario_id, sb.dimension_name, sr.response_key, sr.maps_to_level_name, sr.is_attractive_nuisance
       ORDER BY sb.scenario_id, sr.response_key`,
      [frameworkId]
    );

    // Average time-to-respond per scenario
    const timing = await query(
      `SELECT
         sb.scenario_id,
         ROUND(AVG(sa.time_to_respond_seconds)::numeric, 1) AS avg_seconds,
         ROUND(MIN(sa.time_to_respond_seconds)::numeric, 1) AS min_seconds,
         ROUND(MAX(sa.time_to_respond_seconds)::numeric, 1) AS max_seconds,
         COUNT(sa.id) AS response_count,
         COUNT(CASE WHEN sa.time_to_respond_seconds < 10 THEN 1 END) AS fast_responses,
         COUNT(CASE WHEN sa.time_to_respond_seconds > 180 THEN 1 END) AS slow_responses
       FROM scenario_bank sb
       LEFT JOIN scenario_answers sa ON sa.scenario_id = sb.scenario_id
       WHERE sb.framework_id = $1
       GROUP BY sb.scenario_id
       ORDER BY sb.scenario_id`,
      [frameworkId]
    );

    // Session completion rates
    const sessions = await query(
      `SELECT
         status,
         COUNT(*) AS count
       FROM scenario_sessions
       WHERE framework_id = $1
       GROUP BY status`,
      [frameworkId]
    );

    // Same-boundary agreement rates (per dimension, scenarios testing same boundary)
    const agreement = await query(
      `SELECT
         sb.dimension_id,
         sb.dimension_name,
         sb.target_boundary,
         COUNT(DISTINCT sa.session_id) AS sessions_with_both,
         COUNT(CASE WHEN consistent THEN 1 END) AS consistent_sessions
       FROM (
         SELECT
           sa1.session_id,
           sb1.dimension_id,
           sb1.target_boundary,
           sa1.mapped_level = sa2.mapped_level AS consistent
         FROM scenario_answers sa1
         JOIN scenario_bank sb1 ON sb1.scenario_id = sa1.scenario_id
         JOIN scenario_answers sa2 ON sa2.session_id = sa1.session_id AND sa2.scenario_id != sa1.scenario_id
         JOIN scenario_bank sb2 ON sb2.scenario_id = sa2.scenario_id
           AND sb2.dimension_id = sb1.dimension_id
           AND sb2.target_boundary = sb1.target_boundary
         WHERE sb1.framework_id = $1
           AND sb1.scenario_id < sb2.scenario_id
       ) sub
       JOIN scenario_bank sb ON sb.dimension_id = sub.dimension_id AND sb.target_boundary = sub.target_boundary AND sb.framework_id = $1
       GROUP BY sb.dimension_id, sb.dimension_name, sb.target_boundary`,
      [frameworkId]
    );

    return {
      status: 200,
      headers: { ...corsHeaders(req), "Content-Type": "application/json" },
      body: JSON.stringify({
        framework_id: frameworkId,
        distributions,
        timing,
        sessions,
        boundary_agreement: agreement,
      }),
    };
  } catch (err) {
    if (err instanceof AuthError) {
      return { status: err.statusCode, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: err.message }) };
    }
    context.error("scenario-monitoring error:", err);
    return { status: 500, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: "Internal server error" }) };
  }
}

app.http("scenario-monitoring", {
  methods: ["GET", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
