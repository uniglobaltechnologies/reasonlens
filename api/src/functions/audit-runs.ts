import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { query, queryOne } from "../shared/db";
import { requireAuth, AuthError } from "../shared/auth";
import { corsHeaders, handleCors } from "../middleware/cors";

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  try {
    const user = await requireAuth(req);

    // GET /api/audit-runs?id=xxx → single run with transcripts
    // GET /api/audit-runs → list user's runs
    const runId = req.query.get("id");

    if (runId) {
      const run = await queryOne(
        "SELECT * FROM audit_runs WHERE id = $1 AND created_by = $2",
        [runId, user.userId]
      );
      if (!run) {
        return { status: 404, headers: { ...corsHeaders(), "Content-Type": "application/json" }, body: JSON.stringify({ error: "Run not found" }) };
      }

      const transcripts = await query(
        "SELECT * FROM audit_transcripts WHERE run_id = $1 ORDER BY created_at",
        [runId]
      );
      const posthoc = await query(
        "SELECT * FROM posthoc_pack_results WHERE run_id = $1",
        [runId]
      );
      const benchmarks = await query(
        "SELECT * FROM benchmark_runs WHERE petri_run_id = $1",
        [runId]
      );
      const report = await queryOne(
        "SELECT * FROM audit_reports WHERE run_id = $1",
        [runId]
      );

      return {
        status: 200,
        headers: { ...corsHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({ run, transcripts, posthoc, benchmarks, report }),
      };
    }

    // List runs
    const runs = await query(
      `SELECT id, scenario_pack, auditor_model, target_model, judge_model,
              status, cost_tokens, cost_currency, error_message,
              created_at, started_at, completed_at, mode
       FROM audit_runs WHERE created_by = $1 ORDER BY created_at DESC LIMIT 50`,
      [user.userId]
    );

    return {
      status: 200,
      headers: { ...corsHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ runs }),
    };
  } catch (err) {
    if (err instanceof AuthError) {
      return { status: err.statusCode, headers: { ...corsHeaders(), "Content-Type": "application/json" }, body: JSON.stringify({ error: err.message }) };
    }
    context.error("audit-runs error:", err);
    return { status: 500, headers: { ...corsHeaders(), "Content-Type": "application/json" }, body: JSON.stringify({ error: "Internal server error" }) };
  }
}

app.http("audit-runs", {
  methods: ["GET", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
