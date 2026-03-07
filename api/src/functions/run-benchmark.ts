import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { query, queryOne, execute } from "../shared/db";
import { requireRole, AuthError } from "../shared/auth";
import { decryptValue } from "../shared/crypto";
import { corsHeaders, handleCors } from "../middleware/cors";

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  try {
    const user = await requireRole(req, "admin");

    const body = (await req.json()) as {
      benchmark_type: string;
      target_model: string;
      max_samples?: number;
    };

    if (!body.benchmark_type || !body.target_model) {
      return {
        status: 400,
        headers: { ...corsHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({ error: "benchmark_type and target_model required" }),
      };
    }

    if (!["crows_pairs", "truthfulqa"].includes(body.benchmark_type)) {
      return {
        status: 400,
        headers: { ...corsHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({ error: "Unsupported benchmark_type" }),
      };
    }

    // Check model and BYOK
    const model = await queryOne<{ model_id: string; provider_slug: string; is_free_tier: boolean }>(
      "SELECT model_id, provider_slug, is_free_tier FROM models WHERE model_id = $1",
      [body.target_model]
    );

    let apiKeys: Record<string, string> = {};
    if (model && !model.is_free_tier) {
      const secret = process.env.BYOK_ENC_SECRET;
      if (!secret) throw new Error("BYOK_ENC_SECRET not configured");

      const keyRow = await queryOne<{ encrypted_key: string }>(
        "SELECT encrypted_key FROM user_api_keys WHERE user_id = $1 AND provider = $2",
        [user.userId, model.provider_slug]
      );

      if (!keyRow) {
        return {
          status: 403,
          headers: { ...corsHeaders(), "Content-Type": "application/json" },
          body: JSON.stringify({ error: `API key required for ${model.provider_slug}` }),
        };
      }
      apiKeys[model.provider_slug] = await decryptValue(keyRow.encrypted_key, secret);
    }

    // Create benchmark run
    const run = await queryOne<{ id: string }>(
      `INSERT INTO benchmark_runs (created_by, benchmark_type, target_model, status, started_at)
       VALUES ($1, $2, $3, 'running', now()) RETURNING id`,
      [user.userId, body.benchmark_type, body.target_model]
    );

    if (!run) throw new Error("Failed to create benchmark run");

    await execute(
      "INSERT INTO audit_log (user_id, run_id, action, details) VALUES ($1, $2, $3, $4)",
      [user.userId, run.id, "benchmark_started", JSON.stringify({ type: body.benchmark_type, model: body.target_model })]
    );

    // Fire and forget to benchmark service
    const benchmarkUrl = process.env.BENCHMARK_SERVICE_URL;
    if (!benchmarkUrl) throw new Error("BENCHMARK_SERVICE_URL not configured");

    const callbackUrl = `https://${process.env.WEBSITE_HOSTNAME || "reasonlens-api.azurewebsites.net"}/api/benchmark-callback`;

    fetch(benchmarkUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        run_id: run.id,
        benchmark_type: body.benchmark_type,
        target_model: body.target_model,
        max_samples: body.max_samples || 200,
        callback_url: callbackUrl,
        api_keys: apiKeys,
      }),
    }).catch((err) => context.error("Benchmark service call failed:", err));

    return {
      status: 200,
      headers: { ...corsHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ success: true, run_id: run.id, status: "running" }),
    };
  } catch (err) {
    if (err instanceof AuthError) {
      return {
        status: err.statusCode,
        headers: { ...corsHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({ error: err.message }),
      };
    }
    context.error("run-benchmark error:", err);
    return {
      status: 500,
      headers: { ...corsHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ error: "Internal server error" }),
    };
  }
}

app.http("run-benchmark", {
  methods: ["POST", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
