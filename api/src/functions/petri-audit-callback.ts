import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { query, queryOne, execute } from "../shared/db";
import { decryptValue } from "../shared/crypto";
import { validateHmac } from "../middleware/hmac";
import { corsHeaders, handleCors } from "../middleware/cors";
import {
  parseScores,
  extractAssistantResponses,
  summarizeScores,
  analyzeTranscriptsForErrors,
  TOXICITY_FLAG_THRESHOLD,
  TOXICITY_CRITICAL_THRESHOLD,
} from "../shared/petri-helpers";

const TERMINAL_STATES = new Set(["completed", "failed", "stopped"]);
const POSTHOC_PACKS = new Set(["jt", "rtp"]);
const BENCHMARK_PACKS = new Set(["crows_pairs", "truthfulqa"]);
const MAX_POSTHOC_TEXTS = 500;
const MAX_EVIDENCE_ITEMS = 5;
const DEFAULT_BENCHMARK_SAMPLES = 200;

// --- Posthoc toxicity ---
async function runPosthocToxicity(
  runId: string,
  packs: string[],
  transcripts: Array<{ content?: string | null }>,
  context: InvocationContext
) {
  const requested = packs.filter(p => POSTHOC_PACKS.has(p.toLowerCase()));
  if (!requested.length) return;

  const existing = await query<{ pack_id: string; status: string }>(
    "SELECT pack_id, status FROM posthoc_pack_results WHERE run_id = $1 AND pack_id = ANY($2)",
    [runId, requested]
  );
  const pending = requested.filter(p => {
    const match = existing.find(r => r.pack_id === p);
    return !match || match.status !== "completed";
  });
  if (!pending.length) return;

  // Mark as running
  for (const packId of pending) {
    await execute(
      `INSERT INTO posthoc_pack_results (run_id, pack_id, status) VALUES ($1, $2, 'running')
       ON CONFLICT (run_id, pack_id) DO UPDATE SET status = 'running'`,
      [runId, packId]
    );
  }

  const assistantTexts = extractAssistantResponses(transcripts).slice(0, MAX_POSTHOC_TEXTS);
  if (!assistantTexts.length) {
    for (const packId of pending) {
      await execute(
        `UPDATE posthoc_pack_results SET status = 'failed', error_message = 'No assistant responses found' WHERE run_id = $1 AND pack_id = $2`,
        [runId, packId]
      );
    }
    return;
  }

  const toxicityUrl = process.env.TOXICITY_SERVICE_URL;
  if (!toxicityUrl) {
    for (const packId of pending) {
      await execute(
        `UPDATE posthoc_pack_results SET status = 'failed', error_message = 'TOXICITY_SERVICE_URL not configured' WHERE run_id = $1 AND pack_id = $2`,
        [runId, packId]
      );
    }
    return;
  }

  try {
    const response = await fetch(toxicityUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts: assistantTexts }),
    });
    if (!response.ok) throw new Error(`Toxicity service error (${response.status})`);

    const result = await response.json() as { scores: Array<Record<string, number>> };
    const scores = result.scores?.slice(0, assistantTexts.length) || [];
    if (!scores.length) throw new Error("Toxicity service returned no scores");

    const { labelStats, summary } = summarizeScores(scores);
    const summaryStats = summary || { label: "toxicity", avg: 0, max: 0, p95: 0, flagged: 0, critical: 0 };

    const evidence = scores
      .map((entry, idx) => ({ text: assistantTexts[idx], score: Number(entry[summaryStats.label] ?? 0) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, MAX_EVIDENCE_ITEMS);

    for (const packId of pending) {
      const metrics = {
        version: "1",
        thresholds: { flagged: TOXICITY_FLAG_THRESHOLD, critical: TOXICITY_CRITICAL_THRESHOLD },
        sample_count: scores.length,
        ...(packId === "jt" ? { summary: summaryStats, labels: labelStats } : { toxicity: summaryStats }),
      };
      await execute(
        `UPDATE posthoc_pack_results SET status = 'completed', metrics_json = $3, evidence_json = $4, error_message = NULL WHERE run_id = $1 AND pack_id = $2`,
        [runId, packId, JSON.stringify(metrics), JSON.stringify(evidence)]
      );
    }
  } catch (err: any) {
    for (const packId of pending) {
      await execute(
        `UPDATE posthoc_pack_results SET status = 'failed', error_message = $3 WHERE run_id = $1 AND pack_id = $2`,
        [runId, packId, err?.message || "Posthoc toxicity scoring failed"]
      );
    }
  }
}

// --- Posthoc benchmarks ---
async function runPosthocBenchmarks(run: any, context: InvocationContext) {
  const packs = Array.isArray(run.benchmark_packs) ? run.benchmark_packs : [];
  const requested = packs.filter((p: string) => BENCHMARK_PACKS.has(p.toLowerCase()));
  if (!requested.length) return;

  const benchmarkUrl = process.env.BENCHMARK_SERVICE_URL;
  if (!benchmarkUrl) return;

  const model = await queryOne<{ model_id: string; provider_slug: string; is_free_tier: boolean }>(
    "SELECT model_id, provider_slug, is_free_tier FROM models WHERE model_id = $1",
    [run.target_model]
  );
  if (!model) return;

  let providerKeys: Record<string, string> = {};
  if (!model.is_free_tier) {
    const secret = process.env.BYOK_ENC_SECRET;
    if (!secret) return;
    const keys = await query<{ provider: string; encrypted_key: string }>(
      "SELECT provider, encrypted_key FROM user_api_keys WHERE user_id = $1",
      [run.created_by]
    );
    for (const k of keys) {
      try { providerKeys[k.provider] = await decryptValue(k.encrypted_key, secret); } catch {}
    }
    if (!providerKeys[model.provider_slug]) return;
  }

  if (!process.env.WEBSITE_HOSTNAME) return; // Can't construct callback URL without hostname
  const callbackUrl = `https://${process.env.WEBSITE_HOSTNAME}/api/benchmark-callback`;

  for (const benchType of requested) {
    const benchRun = await queryOne<{ id: string }>(
      `INSERT INTO benchmark_runs (created_by, benchmark_type, target_model, petri_run_id, status, started_at)
       VALUES ($1, $2, $3, $4, 'running', now())
       ON CONFLICT (petri_run_id, benchmark_type) DO UPDATE SET status = 'running', started_at = now()
       RETURNING id`,
      [run.created_by, benchType, run.target_model, run.id]
    );
    if (!benchRun) continue;

    fetch(benchmarkUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        run_id: benchRun.id,
        benchmark_type: benchType,
        target_model: `${model.provider_slug}/${model.model_id}`,
        max_samples: DEFAULT_BENCHMARK_SAMPLES,
        callback_url: callbackUrl,
        api_keys: Object.keys(providerKeys).length ? providerKeys : undefined,
      }),
    }).catch(err => context.error(`Benchmark call failed for ${benchType}:`, err));
  }
}

// --- Main handler ---
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
      context.warn("petri-audit-callback: Invalid HMAC signature");
      return { status: 401, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: "Invalid signature" }) };
    }

    const body = JSON.parse(bodyText) as {
      run_id: string;
      status?: string;
      cost_tokens?: number;
      cost_currency?: number;
      transcripts?: Array<{ content?: string | null; judge_scores?: any; flags?: string[]; language?: string; path?: string; scenario_id?: string; epoch_number?: number }>;
      error_message?: string;
      debug_info?: any;
    };

    const incomingStatus = body.status || "completed";
    context.log(`Callback for run ${body.run_id}, status: ${incomingStatus}`);

    // Fetch run
    const run = await queryOne<any>("SELECT * FROM audit_runs WHERE id = $1", [body.run_id]);
    if (!run) {
      return { status: 404, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: "Run not found" }) };
    }

    const currentIsTerminal = TERMINAL_STATES.has(run.status);
    const incomingIsTerminal = TERMINAL_STATES.has(incomingStatus);

    // Idempotent: ignore "running" on terminal runs
    if (currentIsTerminal && incomingStatus === "running") {
      return { status: 200, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ success: true, ignored: true }) };
    }

    let newStatus = incomingStatus;
    let detectedError = body.error_message || null;

    if (currentIsTerminal && !incomingIsTerminal) newStatus = run.status;

    // Analyze transcripts for errors
    if (incomingStatus === "completed" && body.transcripts?.length) {
      const analysis = analyzeTranscriptsForErrors(body.transcripts);
      if (!analysis.hasValidResults) {
        const isJudgeError = analysis.errorMessages.some(m => m.includes("judge_output") || m.includes("transcript_hook.py"));
        if (isJudgeError) {
          newStatus = "failed";
          detectedError = "[FAILED] Judge produced no output (judge_output=None).";
        } else {
          detectedError = analysis.errorMessages.length
            ? `[WARNING] ${analysis.errorMessages.slice(0, 3).join("; ").slice(0, 500)}`
            : "[WARNING] Audit completed but produced no valid judge scores";
        }
      }
    }

    // Update run
    const updateParts: string[] = ["status = $2"];
    const updateParams: any[] = [body.run_id, newStatus];
    let paramIdx = 3;

    if (body.cost_tokens != null) { updateParts.push(`cost_tokens = GREATEST($${paramIdx}, COALESCE(cost_tokens, 0))`); updateParams.push(body.cost_tokens); paramIdx++; }
    if (body.cost_currency != null) { updateParts.push(`cost_currency = GREATEST($${paramIdx}, COALESCE(cost_currency, 0))`); updateParams.push(body.cost_currency); paramIdx++; }
    if (detectedError) { updateParts.push(`error_message = $${paramIdx}`); updateParams.push(detectedError); paramIdx++; }
    if (run.status === "queued" && !run.started_at) { updateParts.push("started_at = now()"); }
    if (incomingIsTerminal && !run.completed_at) { updateParts.push("completed_at = now()"); }

    await execute(`UPDATE audit_runs SET ${updateParts.join(", ")} WHERE id = $1`, updateParams);

    // Process transcripts (batch upsert)
    if (body.transcripts?.length) {
      const values: any[] = [];
      const placeholders: string[] = [];
      let idx = 1;

      for (let i = 0; i < body.transcripts.length; i++) {
        const t = body.transcripts[i];
        const path = t.scenario_id && t.epoch_number != null
          ? `${body.run_id}/${t.scenario_id}-epoch${t.epoch_number}.json`
          : t.scenario_id
            ? `${body.run_id}/${t.scenario_id}-${i + 1}.json`
            : `${body.run_id}/transcript-${i + 1}.json`;

        let judgeScores = t.judge_scores || {};
        if (Object.keys(judgeScores).length === 0 && t.content) {
          judgeScores = parseScores(t.content);
        }

        placeholders.push(`($${idx}, $${idx + 1}, $${idx + 2}, $${idx + 3}, $${idx + 4}, $${idx + 5}, $${idx + 6}, $${idx + 7})`);
        values.push(body.run_id, path, t.content || null, JSON.stringify(judgeScores), t.flags || [], t.language || "en", t.scenario_id || null, t.epoch_number ?? null);
        idx += 8;
      }

      await execute(
        `INSERT INTO audit_transcripts (run_id, path, content, judge_scores_json, flags, language, scenario_id, epoch_number)
         VALUES ${placeholders.join(", ")}
         ON CONFLICT (run_id, path) DO UPDATE SET
           content = EXCLUDED.content,
           judge_scores_json = EXCLUDED.judge_scores_json,
           flags = EXCLUDED.flags,
           language = EXCLUDED.language,
           scenario_id = EXCLUDED.scenario_id,
           epoch_number = EXCLUDED.epoch_number`,
        values
      );
    }

    // Posthoc processing on completion
    if (incomingStatus === "completed") {
      const posthocPacks = Array.isArray(run.posthoc_packs) ? run.posthoc_packs : [];
      if (posthocPacks.length && body.transcripts?.length) {
        try {
          await runPosthocToxicity(body.run_id, posthocPacks, body.transcripts, context);
        } catch (posthocErr) {
          context.error("Posthoc toxicity failed:", posthocErr);
        }
      }
      await runPosthocBenchmarks(run, context);
    }

    // Audit log
    await execute(
      "INSERT INTO audit_log (user_id, run_id, action, details) VALUES ($1, $2, $3, $4)",
      [run.created_by, body.run_id, incomingIsTerminal ? "run_completed" : "run_status_update",
       JSON.stringify({ previous: run.status, new: newStatus, transcripts: body.transcripts?.length || 0 })]
    );

    return { status: 200, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ success: true }) };
  } catch (err) {
    context.error("petri-audit-callback error:", err);
    return { status: 500, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: "Internal server error" }) };
  }
}

app.http("petri-audit-callback", {
  methods: ["POST", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
