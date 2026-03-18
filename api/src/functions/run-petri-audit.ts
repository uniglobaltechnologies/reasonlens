import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { query, queryOne, execute } from "../shared/db";
import { requireRole, hasRole, AuthError } from "../shared/auth";
import { decryptValue } from "../shared/crypto";
import { corsHeaders, handleCors } from "../middleware/cors";

function normalizeModelId(modelId: string): string {
  if (!modelId) return modelId;
  // Ensure provider prefix for PETRI compatibility
  if (!modelId.includes("/")) {
    const lower = modelId.toLowerCase();
    if (lower.includes("gemini")) return `google/${modelId}`;
    if (lower.includes("gpt")) return `openai/${modelId}`;
    if (lower.includes("claude")) return `anthropic/${modelId}`;
  }
  // Normalize gemini/ → google/ for PETRI
  if (modelId.startsWith("gemini/")) {
    return modelId.replace("gemini/", "google/");
  }
  return modelId;
}

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  try {
    const user = await requireRole(req, "runner", "admin");

    const body = (await req.json()) as {
      scenario_ids?: string[];
      scenario_pack?: string;
      auditor_model: string;
      target_model: string;
      judge_model: string;
      max_turns?: number;
      samples_per_scenario?: number;
      epochs_reducer?: string;
      cap_tokens?: number;
      cap_cost?: number;
      posthoc_packs?: string[];
      benchmark_packs?: string[];
    };

    if ((!body.scenario_ids?.length && !body.scenario_pack) || !body.auditor_model || !body.target_model || !body.judge_model) {
      return {
        status: 400,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ error: "scenario_ids or scenario_pack, auditor_model, target_model, and judge_model are required" }),
      };
    }

    // Fetch scenarios
    const scenarios = body.scenario_ids?.length
      ? await query(
          `SELECT id, pack_id, seed_instruction
           FROM scenarios
           WHERE id = ANY($1::uuid[]) AND (is_default = true OR owner_id = $2::uuid)`,
          [body.scenario_ids, user.userId]
        )
      : await query(
          `SELECT id, pack_id, seed_instruction
           FROM scenarios
           WHERE pack_id = $1 AND (is_default = true OR owner_id = $2::uuid)
           ORDER BY created_at ASC`,
          [body.scenario_pack, user.userId]
        );

    if (scenarios.length === 0) {
      return {
        status: 400,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ error: "No valid scenarios found" }),
      };
    }

    // Normalize model IDs
    const auditorModel = normalizeModelId(body.auditor_model);
    const targetModel = normalizeModelId(body.target_model);
    // Force judge to openai/azure/gpt-5.2 for PETRI compatibility (Inspect requires openai/ prefix)
    let judgeModel = normalizeModelId(body.judge_model);
    if (!judgeModel.includes("azure/")) {
      judgeModel = "openai/azure/gpt-5.2";
    } else if (judgeModel === "azure/gpt-5.2") {
      judgeModel = "openai/azure/gpt-5.2";
    }

    // Check BYOK keys for non-free-tier models (admins bypass)
    const isAdmin = await hasRole(user.userId, "admin");
    const modelIds = [auditorModel, targetModel, judgeModel];
    const models = await query(
      `SELECT model_id, provider_slug, is_free_tier FROM models WHERE model_id = ANY($1)`,
      [modelIds]
    );
    const nonFreeProviders = new Set(
      models.filter((m: any) => !m.is_free_tier).map((m: any) => m.provider_slug)
    );

    let apiKeys: Record<string, string> = {};
    if (nonFreeProviders.size > 0 && !isAdmin) {
      const secret = process.env.BYOK_ENC_SECRET;
      if (!secret) throw new Error("BYOK_ENC_SECRET not configured");

      const userKeys = await query(
        "SELECT provider, encrypted_key FROM user_api_keys WHERE user_id = $1",
        [user.userId]
      );

      for (const provider of nonFreeProviders) {
        const keyRow = userKeys.find((k: any) => k.provider === provider);
        if (!keyRow) {
          return {
            status: 403,
            headers: { ...corsHeaders(req), "Content-Type": "application/json" },
            body: JSON.stringify({ error: `API key required for ${provider}. Add it in Settings → API Keys.` }),
          };
        }
        apiKeys[provider] = await decryptValue(keyRow.encrypted_key, secret);
      }
    }

    // Create run record
    const run = await queryOne<{ id: string }>(
      `INSERT INTO audit_runs (created_by, scenario_pack, auditor_model, target_model, judge_model, max_turns, samples_per_scenario, cap_tokens, cap_cost, posthoc_packs, benchmark_packs, status, started_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, 'running', now())
       RETURNING id`,
      [
        user.userId,
        scenarios.map((s: any) => s.pack_id).join(","),
        auditorModel,
        targetModel,
        judgeModel,
        body.max_turns || 10,
        body.samples_per_scenario || 1,
        body.cap_tokens ?? null,
        body.cap_cost ?? null,
        body.posthoc_packs ?? null,
        body.benchmark_packs ?? null,
      ]
    );

    if (!run) throw new Error("Failed to create run");

    // Audit log
    await execute(
      "INSERT INTO audit_log (user_id, run_id, action, details) VALUES ($1, $2, $3, $4)",
      [user.userId, run.id, "run_started", JSON.stringify({ target: targetModel, scenarios: scenarios.length })]
    );

    // Fire-and-forget call to Modal PETRI service
    const petriUrl = process.env.PETRI_SERVICE_URL;
    if (!petriUrl) throw new Error("PETRI_SERVICE_URL not configured");

    if (!process.env.WEBSITE_HOSTNAME) throw new Error("WEBSITE_HOSTNAME not configured");
    const callbackUrl = `https://${process.env.WEBSITE_HOSTNAME}/api/petri-audit-callback`;

    const petriPayload = {
      run_id: run.id,
      scenarios: scenarios.map((s: any) => ({
        pack_id: s.pack_id,
        seed_instruction: s.seed_instruction,
      })),
      auditor_model: auditorModel,
      target_model: targetModel,
      judge_model: judgeModel,
      max_turns: body.max_turns || 10,
      samples_per_scenario: body.samples_per_scenario || 1,
      epochs_reducer: body.epochs_reducer || "mean",
      cap_tokens: body.cap_tokens,
      cap_cost: body.cap_cost,
      callback_url: callbackUrl,
      api_keys: apiKeys,
    };

    // Fire and forget — but mark run as failed if PETRI is unreachable
    const petriAbort = new AbortController();
    setTimeout(() => petriAbort.abort(), 30000);
    fetch(petriUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(petriPayload),
      signal: petriAbort.signal,
    }).catch(async (err) => {
      context.error("PETRI service call failed:", err);
      try {
        await execute(
          "UPDATE audit_runs SET status = 'failed', error_message = $1, completed_at = now() WHERE id = $2",
          [`PETRI service unreachable: ${err.message}`, run.id]
        );
      } catch (dbErr) {
        context.error("Failed to update run status after PETRI error:", dbErr);
      }
    });

    return {
      status: 200,
      headers: { ...corsHeaders(req), "Content-Type": "application/json" },
      body: JSON.stringify({
        success: true,
        run_id: run.id,
        status: "running",
        message: "Audit started. Results will arrive via callback.",
      }),
    };
  } catch (err) {
    if (err instanceof AuthError) {
      return {
        status: err.statusCode,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ error: err.message }),
      };
    }
    context.error("run-petri-audit error:", err);
    return {
      status: 500,
      headers: { ...corsHeaders(req), "Content-Type": "application/json" },
      body: JSON.stringify({ error: "Internal server error" }),
    };
  }
}

app.http("run-petri-audit", {
  methods: ["POST", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
