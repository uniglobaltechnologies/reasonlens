import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { query } from "../shared/db";
import { requireAuth, AuthError } from "../shared/auth";
import { generateContentStream, createSSEResponse } from "../shared/ai";
import { getFrameworkContextById } from "../shared/framework-context";
import { PLATFORM_PREAMBLE } from "../shared/prompt-preamble";
import { corsHeaders, handleCors } from "../middleware/cors";
import { levelToScore, scoreToLabel } from "../shared/level-mapping";
import { loadRegulatoryContext } from "../shared/regulatory-context";

const SYSTEM_PROMPT = `${PLATFORM_PREAMBLE}

You are a policy drafting specialist. You generate AI policy documents for educational institutions.

EVIDENCE HIERARCHY (in order of authority):
1. Framework indicators — cite specific dimension, level, and indicator description from the GROUNDING FRAMEWORK section.
2. Regulatory provisions — cite article number and title from the REGULATORY CONTEXT section.
3. Sector practice — cite widely adopted institutional practice relevant to the sector/region.
4. [UNGROUNDED] — if a clause cannot be traced to any of the above, prefix it with "[UNGROUNDED]" so the institution knows it requires local justification.

GROUNDING RULES:
1. Every policy clause MUST trace to at least one evidence source. Use this reference format:
   "Clause 3.1: [Framework], [Dimension], [Indicator] — "[description]""
   Example: "Clause 3.1: JISC AI Maturity Model, Governance & Ethics, Level 3 — "Formal AI governance committee with defined ToR and regular reporting cycle""
2. Flag gaps with [NEEDS INSTITUTIONAL INPUT] — e.g., specific tool names, department structures, budget figures, named roles.
3. Match policy ambition to the institution's assessed maturity level:
   - Emerging/Incidental (score 1): foundational policies — define roles, establish basic processes
   - Developing/Intentional (score 2): developing policies — formalise processes, begin monitoring
   - Established/Integrated (score 3): operational policies — embed in governance, measure outcomes
   - Advanced/Embedded (score 4): advanced policies — optimise, cross-institutional alignment
   - Optimised/Transformed (score 5): innovation policies — sector leadership, continuous improvement
   If no assessment data is available, default to "Developing" and flag: "NOTE: No assessment data available. This policy is calibrated to a Developing maturity level. Re-generate after completing an institutional assessment for better calibration."
4. Include numbered clause references for auditability (e.g., 3.1, 3.2).
5. Write in professional policy language, not academic prose. Active voice. Clear obligations ("The institution shall..." not "It is recommended that...").
6. For the user's low-scoring dimensions, include aspirational clauses that describe the pathway to the next maturity level.
7. Use UK English for UK users, US English for US users, International English otherwise. Default to UK English if region is unknown.
8. Target word count: 1,500–2,500 words for a single policy type. Do not pad with generic content to reach the target.

OUTPUT FORMAT:
- Start with: "DRAFT — For review by institutional governance and legal teams before adoption."
- Use markdown headings for sections
- Include a "Definitions" section at the start defining key terms used in the policy
- Number all policy clauses (e.g., 3.1, 3.2)
- End each section with a "References" sub-section citing the framework dimensions and regulatory articles used
- End the document with a "Document Control" section: version, date, review cycle, owner [NEEDS INSTITUTIONAL INPUT]`;

function buildUserPrompt(ctx: {
  policy_type: string;
  institution_name: string;
  region: string;
  sector: string;
  framework_id?: string;
  frameworkDetail?: string | null;
  template?: any;
  regulatory_provisions?: any[];
  assessment_summary?: string;
}): string {
  let prompt = `Generate a draft "${ctx.policy_type}" for ${ctx.institution_name}.\n\n`;

  prompt += `INSTITUTION CONTEXT:\n`;
  prompt += `- Region: ${ctx.region}\n`;
  prompt += `- Sector: ${ctx.sector}\n`;
  if (ctx.assessment_summary) {
    prompt += `- Current assessment results (use these to calibrate policy ambition):\n${ctx.assessment_summary}\n`;
  }
  prompt += `\n`;

  if (ctx.frameworkDetail) {
    prompt += `GROUNDING FRAMEWORK (use this framework's dimensions and indicators as the primary evidence base for policy clauses):\n${ctx.frameworkDetail}\n\n`;
  }

  if (ctx.template) {
    prompt += `TEMPLATE STRUCTURE:\nGenerate the policy following these sections:\n`;
    for (const section of ctx.template.sections || []) {
      prompt += `\n## ${section.title}${section.required ? " (Required)" : " (Optional)"}\n`;
      prompt += `Instructions: ${section.copilot_instruction}\n`;
    }
    prompt += `\n`;
  }

  if (ctx.regulatory_provisions?.length) {
    prompt += `REGULATORY CONTEXT:\n`;
    for (const prov of ctx.regulatory_provisions) {
      prompt += `\n### ${prov.title} (${prov.article || prov.id})\n`;
      if (prov.full_text) {
        const text =
          prov.full_text.length > 500
            ? prov.full_text.substring(0, 500) + "..."
            : prov.full_text;
        prompt += text + "\n";
      }
      if (prov.education_relevance) {
        prompt += `Education relevance: ${prov.education_relevance}\n`;
      }
    }
    prompt += `\n`;
  }

  prompt += `Generate the complete draft policy. Every clause must reference its evidence source (framework dimension or regulation article). Use professional policy language appropriate for a ${ctx.sector} institution in ${ctx.region}. Include the disclaimer header.`;
  return prompt;
}

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  try {
    const user = await requireAuth(req);

    const body = (await req.json()) as {
      policy_type: string;
      institution_name?: string;
      region?: string;
      sector?: string;
      framework_id?: string;
      template?: any;
      regulatory_provisions?: any[];
      assessment_summary?: string;
    };

    if (!body.policy_type) {
      return {
        status: 400,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ error: "policy_type is required" }),
      };
    }

    const frameworkDetail = body.framework_id
      ? getFrameworkContextById(body.framework_id)
      : null;

    // Server-side assessment enrichment: fetch from DB if not provided by frontend
    let assessmentSummary = body.assessment_summary || "";
    if (!assessmentSummary) {
      const results = await query<{
        framework_id: string;
        dimension: string;
        selected_level: string;
      }>(
        "SELECT DISTINCT ON (framework_id, dimension) framework_id, dimension, selected_level FROM assessment_results WHERE user_id = $1 ORDER BY framework_id, dimension, completed_at DESC",
        [user.userId]
      );
      if (results.length > 0) {
        const byFramework = new Map<string, { dimension: string; level: string; score: number }[]>();
        for (const r of results) {
          const list = byFramework.get(r.framework_id) || [];
          list.push({ dimension: r.dimension, level: r.selected_level, score: levelToScore(r.selected_level) });
          byFramework.set(r.framework_id, list);
        }
        const lines: string[] = [];
        for (const [fwId, dims] of byFramework) {
          const avgScore = dims.reduce((a, d) => a + d.score, 0) / dims.length;
          lines.push(`  ${fwId} (avg: ${scoreToLabel(Math.round(avgScore))}):`);
          for (const d of dims) {
            lines.push(`    - ${d.dimension}: ${d.level} (${d.score}/5)`);
          }
        }
        assessmentSummary = lines.join("\n");
      }
    }

    // Server-side regulatory context: load from DB if not provided by frontend
    const region = body.region || "international";
    let regulatoryProvisions = body.regulatory_provisions;
    if (!regulatoryProvisions || regulatoryProvisions.length === 0) {
      regulatoryProvisions = loadRegulatoryContext(region);
    }

    const userPrompt = buildUserPrompt({
      policy_type: body.policy_type,
      institution_name: body.institution_name || "[Institution Name]",
      region,
      sector: body.sector || "higher education",
      framework_id: body.framework_id,
      frameworkDetail,
      template: body.template,
      regulatory_provisions: regulatoryProvisions,
      assessment_summary: assessmentSummary,
    });

    const stream = generateContentStream(SYSTEM_PROMPT, [
      { role: "user", content: userPrompt },
    ]);

    return {
      status: 200,
      headers: {
        ...corsHeaders(req),
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
      body: createSSEResponse(stream),
    };
  } catch (err) {
    if (err instanceof AuthError) {
      return {
        status: err.statusCode,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ error: err.message }),
      };
    }
    context.error("policy-generator error:", err);
    return {
      status: 500,
      headers: { ...corsHeaders(req), "Content-Type": "application/json" },
      body: JSON.stringify({
        error: err instanceof Error ? err.message : "Unknown error",
      }),
    };
  }
}

app.http("policy-generator", {
  methods: ["POST", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
