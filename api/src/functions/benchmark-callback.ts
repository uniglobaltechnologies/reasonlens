import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { queryOne, execute } from "../shared/db";
import { validateHmac } from "../middleware/hmac";
import { corsHeaders, handleCors } from "../middleware/cors";

const TERMINAL_STATES = new Set(["completed", "failed", "stopped"]);

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  try {
    const bodyText = await req.text();
    const secret = process.env.PETRI_CALLBACK_SECRET ?? "";

    if (!validateHmac(req, bodyText, secret)) {
      context.warn("benchmark-callback: Invalid HMAC signature");
      return {
        status: 401,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ error: "Invalid signature" }),
      };
    }

    const body = JSON.parse(bodyText) as {
      run_id: string;
      status: string;
      benchmark_type?: string;
      metrics?: Record<string, any>;
      error_message?: string;
    };

    if (!body.run_id || !body.status) {
      return {
        status: 400,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ error: "run_id and status required" }),
      };
    }

    // Fetch existing run
    const existing = await queryOne<{ id: string; status: string }>(
      "SELECT id, status FROM benchmark_runs WHERE id = $1",
      [body.run_id]
    );

    if (!existing) {
      return {
        status: 404,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ error: "Benchmark run not found" }),
      };
    }

    // Idempotent: don't regress from terminal state
    if (TERMINAL_STATES.has(existing.status)) {
      context.log(`benchmark-callback: Run ${body.run_id} already in terminal state ${existing.status}, ignoring`);
      return {
        status: 200,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ success: true, message: "Already terminal" }),
      };
    }

    // Update benchmark run
    await execute(
      `UPDATE benchmark_runs SET
        status = $2,
        results_json = $3,
        error_message = $4,
        completed_at = CASE WHEN $2 IN ('completed', 'failed', 'stopped') THEN now() ELSE completed_at END
       WHERE id = $1`,
      [
        body.run_id,
        body.status,
        body.metrics ? JSON.stringify(body.metrics) : null,
        body.error_message ?? null,
      ]
    );

    // Audit log
    await execute(
      "INSERT INTO audit_log (run_id, action, details) VALUES ($1, $2, $3)",
      [
        body.run_id,
        "benchmark_callback_received",
        JSON.stringify({ status: body.status, benchmark_type: body.benchmark_type }),
      ]
    );

    context.log(`benchmark-callback: Updated run ${body.run_id} to ${body.status}`);

    return {
      status: 200,
      headers: { ...corsHeaders(req), "Content-Type": "application/json" },
      body: JSON.stringify({ success: true }),
    };
  } catch (err) {
    context.error("benchmark-callback error:", err);
    return {
      status: 500,
      headers: { ...corsHeaders(req), "Content-Type": "application/json" },
      body: JSON.stringify({ error: "Internal server error" }),
    };
  }
}

app.http("benchmark-callback", {
  methods: ["POST", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
