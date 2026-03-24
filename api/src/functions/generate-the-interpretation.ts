import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { query, queryOne, execute } from "../shared/db";
import { requireAuth, AuthError } from "../shared/auth";
import { corsHeaders, handleCors } from "../middleware/cors";
import { scoreSession, ScenarioAnswer, DimensionResult } from "../shared/scenario-scoring";
import { generateContent } from "../shared/ai";
import {
  buildExecutiveSummaryMethodology,
  buildPillarAnalysisMethodology,
  buildRecommendationsMethodology,
} from "../shared/the-interpretive-methodology";

// ── Pillar mapping (dimension_id prefix → pillar name) ──────────────

const PILLAR_MAP: Record<string, string> = {
  "the-tl": "Teaching & Learning",
  "the-re": "Research",
  "the-ps": "Professional Services",
  "the-pg": "Planning & Governance",
};

function derivePillar(dimensionId: string): string {
  const prefix = dimensionId.split("-").slice(0, 2).join("-");
  return PILLAR_MAP[prefix] ?? "Unknown";
}

// ── Data types ──────────────────────────────────────────────────────

interface InstitutionalContext {
  institution_type?: string;
  institution_size?: string;
  region?: string;
  funding_model?: string;
  respondent_role?: string;
  respondent_institutional_visibility?: string;
  digital_infrastructure_baseline?: string;
}

interface ResponseDetail {
  scenario_id: string;
  pillar: string;
  dimension: string;
  boundary: string;
  mapped_level: string;
  response_text: string;
  is_attractive_nuisance: boolean;
  nuisance_explanation: string | null;
  time_to_respond_seconds: number | null;
}

interface OpenEndedContext {
  trigger_response?: string;
  previous_attempts?: string;
  constraints?: string[];
  constraints_detail?: string;
  success_definition?: string;
  additional_context?: string;
}

// ── Formatting helpers ──────────────────────────────────────────────

function formatContext(ctx: InstitutionalContext): string {
  return `INSTITUTIONAL CONTEXT:
- Institution type: ${ctx.institution_type || "Not specified"}
- Institution size: ${ctx.institution_size || "Not specified"}
- Region: ${ctx.region || "Not specified"}
- Funding model: ${ctx.funding_model || "Not specified"}
- Respondent role: ${ctx.respondent_role || "Not specified"}
- Respondent visibility: ${ctx.respondent_institutional_visibility || "Not specified"}
- Digital infrastructure baseline: ${ctx.digital_infrastructure_baseline || "Not specified"}`;
}

function formatDimensionTable(dimensions: Array<DimensionResult & { pillar: string }>): string {
  let table = "SCORED MATURITY PROFILE:\nPillar | Dimension | Level | Confidence\n---|---|---|---\n";
  for (const d of dimensions) {
    table += `${d.pillar} | ${d.dimension_name} | ${d.assigned_level} | ${d.confidence}\n`;
  }
  return table;
}

function formatResponseDetail(responses: ResponseDetail[], pillar?: string): string {
  const filtered = pillar ? responses.filter(r => r.pillar === pillar) : responses;
  let detail = "SCENARIO RESPONSE DETAIL:\n";
  for (const r of filtered) {
    const nuisanceFlag = r.is_attractive_nuisance ? " [NUISANCE SELECTED]" : "";
    detail += `\n${r.scenario_id} (${r.dimension}, ${r.boundary}):\n`;
    detail += `  Selected: ${r.mapped_level}${nuisanceFlag}\n`;
    detail += `  Response text: "${r.response_text}"\n`;
    if (r.is_attractive_nuisance && r.nuisance_explanation) {
      detail += `  Nuisance insight: ${r.nuisance_explanation}\n`;
    }
  }
  return detail;
}

function formatNuisanceSummary(responses: ResponseDetail[]): string {
  const nuisances = responses.filter(r => r.is_attractive_nuisance);
  const total = responses.length;
  let summary = `NUISANCE ANALYSIS SUMMARY:\n`;
  summary += `Total scenario responses: ${total}\n`;
  summary += `Nuisance responses selected: ${nuisances.length} (${total > 0 ? Math.round(nuisances.length / total * 100) : 0}%)\n\n`;
  if (nuisances.length > 0) {
    summary += "Nuisance selections:\n";
    for (const n of nuisances) {
      summary += `- ${n.scenario_id} (${n.pillar}, ${n.dimension}, ${n.boundary}): ${n.nuisance_explanation || "No explanation available"}\n`;
    }
  }
  return summary;
}

function formatTriageComparison(
  triageSignals: Record<string, string> | null,
  dimensions: Array<DimensionResult & { pillar: string }>
): string {
  if (!triageSignals) return "TRIAGE COMPARISON: Not available (user did not complete triage before scenario assessment).";

  const pillarMode = new Map<string, number[]>();
  for (const d of dimensions) {
    if (!pillarMode.has(d.pillar)) pillarMode.set(d.pillar, []);
    pillarMode.get(d.pillar)!.push(d.assigned_level_order);
  }

  const SIGNAL_ORDER: Record<string, number> = { incidental: 1, intentional: 2, integrated: 3, optimised: 4 };
  const LEVEL_NAMES: Record<number, string> = { 1: "Incidental", 2: "Intentional", 3: "Integrated", 4: "Optimised" };
  const PILLAR_KEY_MAP: Record<string, string> = {
    "Teaching & Learning": "teaching_learning",
    "Research": "research",
    "Professional Services": "professional_services",
    "Planning & Governance": "planning_governance",
  };

  let comp = "TRIAGE vs SCENARIO COMPARISON:\nPillar | Triage Signal | Scenario Mode | Gap\n---|---|---|---\n";
  for (const [pillar, orders] of pillarMode) {
    const key = PILLAR_KEY_MAP[pillar];
    const triageSignal = key ? triageSignals[key] : null;
    const mode = Math.round(orders.reduce((a, b) => a + b, 0) / orders.length);
    const modeName = LEVEL_NAMES[mode] ?? "Unknown";
    if (triageSignal) {
      const triageOrder = SIGNAL_ORDER[triageSignal] ?? 0;
      const gap = triageOrder - mode;
      const gapDesc = gap > 0 ? `Triage higher by ${gap} level(s)` : gap < 0 ? `Scenario higher by ${Math.abs(gap)} level(s)` : "Aligned";
      comp += `${pillar} | ${triageSignal} | ${modeName} | ${gapDesc}\n`;
    } else {
      comp += `${pillar} | N/A | ${modeName} | N/A\n`;
    }
  }
  return comp;
}

function formatOpenEnded(oe: OpenEndedContext): string {
  let text = "OPEN-ENDED CONTEXT FROM INSTITUTION:\n\n";
  text += `Q1 - What triggered this assessment:\n${oe.trigger_response || "Not provided"}\n\n`;
  text += `Q2 - Previous improvement attempts:\n${oe.previous_attempts || "Not provided"}\n\n`;
  text += `Q3 - Known constraints:\n${oe.constraints?.join(", ") || "None selected"}\n`;
  if (oe.constraints_detail) text += `  Detail: ${oe.constraints_detail}\n`;
  text += `\nQ4 - Definition of success:\n${oe.success_definition || "Not provided"}\n\n`;
  text += `Q5 - Additional context:\n${oe.additional_context || "Not provided"}\n`;
  return text;
}

// ── User prompts per section ────────────────────────────────────────

const EXEC_USER_PROMPT = `Produce the EXECUTIVE SUMMARY section of the interpretive report.

Structure your output as markdown with these subsections:

## Executive Summary

### Headline Finding
The single most important pattern in this institution's maturity profile. Not the average level, not a list of scores, but the one insight a Vice-Chancellor needs to hear first. Name the pattern from the taxonomy if applicable. Cite the specific dimension scores that evidence it.

### Strengths
2-3 genuine strengths, each supported by high-confidence scoring and specific scenario response evidence. A strength is only genuine if the confidence is high.

### Critical Gaps
2-3 areas requiring the most urgent attention. Prioritise gaps where cross-dimension dependencies mean one weakness undermines another. Cite specific scores and dependencies.

### Blind Spots
Analysis of nuisance response selections. What do they reveal about the institution's self-perception versus actual practice? Reference specific scenario IDs and explain what the nuisance selection tells us. If the triage comparison shows gaps between self-perception and scenario results, include that here.

Keep the entire section to 400-600 words. Be direct. Every sentence must earn its place.`;

function buildPillarUserPrompt(pillar: string): string {
  return `Produce the ${pillar.toUpperCase()} pillar analysis section.

Structure your output as markdown:

## ${pillar}

### Pillar Profile
What does the five-dimension profile within this pillar tell us? Are all dimensions at similar levels (uniform maturity) or is there significant variation (uneven development)? State the pattern clearly.

### Dimension Interactions
Analyse how the five dimensions relate to each other within this pillar using the dependency model. If Technology is ahead of People, what does that mean? If Strategy is behind everything else, what does that imply? Cite specific scores.

### Contextual Position
Given this institution's type, size, and region, is this pillar profile typical, concerning, or notable? Use the calibration norms. Be specific: "For a [type] institution in [region], scoring [level] on [dimension] is [assessment]."

### Scenario Insights
Reference 1-2 specific scenario responses that reveal something important about how this institution operates within this pillar. Quote the response they selected and explain what it tells us. If they selected a nuisance, explain the blind spot it reveals.

Keep the entire section to 300-500 words. Focus on what is distinctive about this pillar's results, not generic observations.`;
}

const RECS_USER_PROMPT = `Produce the STRATEGIC RECOMMENDATIONS section.

Structure your output as markdown:

## Strategic Recommendations

### Priority Framework
Before listing recommendations, state your prioritisation logic in 2-3 sentences. Why are you recommending these actions in this order? What is the organising principle?

### Recommendations
Produce 5-7 numbered recommendations. Each must include:
1. **What to do**: A specific, concrete action
2. **Why this is the priority**: Cite the specific assessment evidence (dimension scores, scenario responses, nuisance patterns, cross-dimension dependencies)
3. **What success looks like**: A measurable outcome
4. **Timeframe**: Realistic, calibrated to the institution's current level and stated constraints
5. **Constraint acknowledgment**: If the institution flagged relevant constraints in Q3, state how this recommendation accounts for them

### What NOT to Do
1-2 specific anti-recommendations: common actions that this assessment evidence suggests would be counterproductive for this institution.

### Confidence and Limitations
Where confidence is low on key dimensions, what should the institution do to validate before acting? Where respondent visibility is limited, which recommendations should be tested with additional respondents?

Keep the entire section to 500-700 words. Every recommendation must be traceable to assessment evidence. No generic best practice.`;

// ── Scored data loader (reused by GET and POST) ─────────────────────

async function loadScoredData(sessionId: string, userId: string) {
  const rawAnswers = await query<{
    scenario_id: string; mapped_level: string; dimension_id: string;
    dimension_name: string; target_boundary: string; maps_to_level_order: number;
  }>(
    `SELECT sa.scenario_id, sa.mapped_level, sb.dimension_id, sb.dimension_name,
            sb.target_boundary, sr.maps_to_level_order
     FROM scenario_answers sa
     JOIN scenario_bank sb ON sb.scenario_id = sa.scenario_id
     JOIN scenario_responses sr ON sr.id = sa.response_id
     WHERE sa.session_id = $1`,
    [sessionId]
  );
  const scoringInput: ScenarioAnswer[] = rawAnswers.map(a => ({
    scenario_id: a.scenario_id, dimension_id: a.dimension_id,
    dimension_name: a.dimension_name, mapped_level: a.mapped_level,
    level_order: a.maps_to_level_order, target_boundary: a.target_boundary,
  }));
  const scoredResults = scoreSession(scoringInput, { frameworkId: "maturity-the" });
  const institutionalContext = await queryOne<InstitutionalContext>(
    `SELECT institution_type, institution_size, region, funding_model,
            respondent_role, respondent_institutional_visibility,
            digital_infrastructure_baseline
     FROM user_assessment_context WHERE user_id = $1`,
    [userId]
  ) ?? {};
  return { scoredResults, institutionalContext };
}

// ── Handler ─────────────────────────────────────────────────────────

const CALL_TIMEOUT = 45_000; // 45s per LLM call

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  try {
    const user = await requireAuth(req);

    // GET: check if report already exists
    if (req.method === "GET") {
      const sessionId = req.query.get("session_id");
      if (!sessionId) {
        return { status: 400, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: "session_id required" }) };
      }
      const report = await queryOne<{
        id: string;
        executive_summary: string;
        pillar_tl: string;
        pillar_re: string;
        pillar_ps: string;
        pillar_pg: string;
        recommendations: string;
        generation_time_ms: number;
        methodology_version: string;
        model_used: string;
        created_at: string;
      }>(
        "SELECT * FROM interpretive_reports WHERE session_id = $1",
        [sessionId]
      );
      if (!report) {
        return { status: 404, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: "No report found" }) };
      }
      // Also load scored results + context for docx generation
      const { scoredResults, institutionalContext } = await loadScoredData(sessionId, user.userId);
      return {
        status: 200,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({
          interpretation_id: report.id,
          sections: {
            executive_summary: report.executive_summary,
            pillar_teaching_learning: report.pillar_tl,
            pillar_research: report.pillar_re,
            pillar_professional_services: report.pillar_ps,
            pillar_planning_governance: report.pillar_pg,
            recommendations: report.recommendations,
          },
          metadata: {
            generated_at: report.created_at,
            model_used: report.model_used,
            methodology_version: report.methodology_version,
            total_generation_time_ms: report.generation_time_ms,
          },
          scored_results: scoredResults,
          context: institutionalContext,
        }),
      };
    }

    if (req.method !== "POST") {
      return { status: 405, headers: corsHeaders(req), body: "Method not allowed" };
    }

    const body = (await req.json()) as { session_id: string; regenerate?: boolean };
    if (!body.session_id) {
      return { status: 400, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: "session_id required" }) };
    }

    // Check for existing report (cache hit unless regenerating)
    if (!body.regenerate) {
      const existing = await queryOne<{
        id: string; executive_summary: string; pillar_tl: string; pillar_re: string;
        pillar_ps: string; pillar_pg: string; recommendations: string;
        generation_time_ms: number; methodology_version: string; model_used: string; created_at: string;
      }>(
        "SELECT * FROM interpretive_reports WHERE session_id = $1",
        [body.session_id]
      );
      if (existing) {
        const { scoredResults, institutionalContext } = await loadScoredData(body.session_id, user.userId);
        return {
          status: 200,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({
            interpretation_id: existing.id,
            sections: {
              executive_summary: existing.executive_summary,
              pillar_teaching_learning: existing.pillar_tl,
              pillar_research: existing.pillar_re,
              pillar_professional_services: existing.pillar_ps,
              pillar_planning_governance: existing.pillar_pg,
              recommendations: existing.recommendations,
            },
            metadata: {
              generated_at: existing.created_at,
              model_used: existing.model_used,
              methodology_version: existing.methodology_version,
              total_generation_time_ms: existing.generation_time_ms,
            },
            scored_results: scoredResults,
            context: institutionalContext,
          }),
        };
      }
    }

    // Verify session
    const session = await queryOne<{ user_id: string; framework_id: string; status: string }>(
      "SELECT user_id, framework_id, status FROM scenario_sessions WHERE id = $1",
      [body.session_id]
    );
    if (!session || session.status !== "completed" || session.framework_id !== "maturity-the") {
      return { status: 404, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: "Completed THE DMI session not found" }) };
    }
    if (session.user_id !== user.userId) {
      return { status: 403, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: "Session does not belong to you" }) };
    }

    // ── Load all data ───────────────────────────────────────────────

    // 1. Institutional context
    const ctx = await queryOne<InstitutionalContext>(
      `SELECT institution_type, institution_size, region, funding_model,
              respondent_role, respondent_institutional_visibility,
              digital_infrastructure_baseline
       FROM user_assessment_context WHERE user_id = $1`,
      [user.userId]
    ) ?? {};

    // 2. Re-compute scored results from raw answers
    const rawAnswers = await query<{
      scenario_id: string;
      mapped_level: string;
      dimension_id: string;
      dimension_name: string;
      target_boundary: string;
      maps_to_level_order: number;
    }>(
      `SELECT sa.scenario_id, sa.mapped_level,
              sb.dimension_id, sb.dimension_name, sb.target_boundary,
              sr.maps_to_level_order
       FROM scenario_answers sa
       JOIN scenario_bank sb ON sb.scenario_id = sa.scenario_id
       JOIN scenario_responses sr ON sr.id = sa.response_id
       WHERE sa.session_id = $1`,
      [body.session_id]
    );

    const scoringInput: ScenarioAnswer[] = rawAnswers.map(a => ({
      scenario_id: a.scenario_id,
      dimension_id: a.dimension_id,
      dimension_name: a.dimension_name,
      mapped_level: a.mapped_level,
      level_order: a.maps_to_level_order,
      target_boundary: a.target_boundary,
    }));

    const scored = scoreSession(scoringInput, { frameworkId: "maturity-the" });
    const dimensionsWithPillar = scored.map(d => ({
      ...d,
      pillar: derivePillar(d.dimension_id),
    }));

    // 3. Per-scenario response detail with nuisance flags
    const responseDetails = await query<{
      scenario_id: string;
      dimension_id: string;
      dimension_name: string;
      target_boundary: string;
      mapped_level: string;
      response_text: string;
      is_attractive_nuisance: boolean;
      nuisance_explanation: string | null;
      time_to_respond_seconds: number | null;
    }>(
      `SELECT sa.scenario_id, sb.dimension_id, sb.dimension_name, sb.target_boundary,
              sa.mapped_level, sr.response_text,
              sr.is_attractive_nuisance, sr.nuisance_explanation,
              sa.time_to_respond_seconds
       FROM scenario_answers sa
       JOIN scenario_bank sb ON sb.scenario_id = sa.scenario_id
       JOIN scenario_responses sr ON sr.id = sa.response_id
       WHERE sa.session_id = $1
       ORDER BY sb.dimension_id`,
      [body.session_id]
    );

    const responses: ResponseDetail[] = responseDetails.map(r => ({
      scenario_id: r.scenario_id,
      pillar: derivePillar(r.dimension_id),
      dimension: r.dimension_name,
      boundary: r.target_boundary,
      mapped_level: r.mapped_level,
      response_text: r.response_text,
      is_attractive_nuisance: r.is_attractive_nuisance,
      nuisance_explanation: r.nuisance_explanation,
      time_to_respond_seconds: r.time_to_respond_seconds,
    }));

    // 4. Triage comparison
    const triage = await queryOne<{ pillar_signals: Record<string, string> }>(
      `SELECT pillar_signals FROM triage_results
       WHERE user_id = $1 AND framework_id = 'maturity-the'
       ORDER BY created_at DESC LIMIT 1`,
      [user.userId]
    );

    // 5. Open-ended responses
    const openEnded = await queryOne<OpenEndedContext>(
      `SELECT trigger_response, previous_attempts, constraints,
              constraints_detail, success_definition, additional_context
       FROM interpretation_context WHERE session_id = $1`,
      [body.session_id]
    ) ?? {};

    // ── Build data blocks ───────────────────────────────────────────

    const contextBlock = formatContext(ctx);
    const dimTable = formatDimensionTable(dimensionsWithPillar);
    const nuisanceSummary = formatNuisanceSummary(responses);
    const triageBlock = formatTriageComparison(triage?.pillar_signals ?? null, dimensionsWithPillar);
    const openEndedBlock = formatOpenEnded(openEnded);

    // ── 3-stage LLM pipeline ────────────────────────────────────────

    const startTime = Date.now();

    // Stage 1: Executive summary
    const execSystemPrompt = [
      buildExecutiveSummaryMethodology(),
      contextBlock,
      dimTable,
      nuisanceSummary,
      triageBlock,
    ].join("\n\n");

    const execSummary = await generateContent(
      execSystemPrompt,
      [{ role: "user", content: EXEC_USER_PROMPT }],
      { timeoutMs: CALL_TIMEOUT }
    );

    // Stage 2: Four pillar analyses in parallel
    const pillarCodes = ["Teaching & Learning", "Research", "Professional Services", "Planning & Governance"] as const;

    const [tl, re, ps, pg] = await Promise.all(
      pillarCodes.map(pillar => {
        const pillarDims = dimensionsWithPillar.filter(d => d.pillar === pillar);
        const systemPrompt = [
          buildPillarAnalysisMethodology(),
          contextBlock,
          `PILLAR BEING ANALYSED: ${pillar}\n\n` + formatDimensionTable(pillarDims),
          formatResponseDetail(responses, pillar),
        ].join("\n\n");

        return generateContent(
          systemPrompt,
          [{ role: "user", content: buildPillarUserPrompt(pillar) }],
          { timeoutMs: CALL_TIMEOUT }
        );
      })
    );

    // Stage 3: Strategic recommendations
    const recsSystemPrompt = [
      buildRecommendationsMethodology(),
      contextBlock,
      dimTable,
      nuisanceSummary,
      openEndedBlock,
    ].join("\n\n");

    const recommendations = await generateContent(
      recsSystemPrompt,
      [{ role: "user", content: RECS_USER_PROMPT }],
      { timeoutMs: CALL_TIMEOUT }
    );

    const totalTime = Date.now() - startTime;
    const modelUsed = process.env.AZURE_OPENAI_DEPLOYMENT || "gpt-5.2";

    // ── Store results ───────────────────────────────────────────────

    const report = await queryOne<{ id: string }>(
      `INSERT INTO interpretive_reports
         (session_id, user_id, executive_summary, pillar_tl, pillar_re,
          pillar_ps, pillar_pg, recommendations, generation_time_ms,
          methodology_version, model_used)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, '1.0', $10)
       ON CONFLICT (session_id) DO UPDATE SET
         executive_summary = $3, pillar_tl = $4, pillar_re = $5,
         pillar_ps = $6, pillar_pg = $7, recommendations = $8,
         generation_time_ms = $9, model_used = $10, created_at = now()
       RETURNING id`,
      [body.session_id, user.userId, execSummary, tl, re, ps, pg,
       recommendations, totalTime, modelUsed]
    );

    return {
      status: 200,
      headers: { ...corsHeaders(req), "Content-Type": "application/json" },
      body: JSON.stringify({
        interpretation_id: report?.id,
        sections: {
          executive_summary: execSummary,
          pillar_teaching_learning: tl,
          pillar_research: re,
          pillar_professional_services: ps,
          pillar_planning_governance: pg,
          recommendations,
        },
        metadata: {
          generated_at: new Date().toISOString(),
          model_used: modelUsed,
          methodology_version: "1.0",
          total_generation_time_ms: totalTime,
        },
        scored_results: dimensionsWithPillar,
        context: ctx,
      }),
    };
  } catch (err) {
    if (err instanceof AuthError) {
      return { status: err.statusCode, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: err.message }) };
    }
    context.error("generate-the-interpretation error:", err);
    return {
      status: 500,
      headers: { ...corsHeaders(req), "Content-Type": "application/json" },
      body: JSON.stringify({ error: "Report generation failed. Please try again." }),
    };
  }
}

app.http("generate-the-interpretation", {
  methods: ["GET", "POST", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
