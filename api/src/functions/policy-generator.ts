import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { requireAuth, AuthError } from "../shared/auth";
import { generateContentStream, createSSEResponse } from "../shared/ai";
import { getFrameworkContextById } from "../shared/framework-context";
import { PLATFORM_PREAMBLE } from "../shared/prompt-preamble";
import { corsHeaders, handleCors } from "../middleware/cors";

const SYSTEM_PROMPT = `${PLATFORM_PREAMBLE}

You are a policy drafting specialist. You generate AI policy documents for educational institutions.

GROUNDING RULES:
1. Every policy clause MUST trace to either: (a) a framework indicator/level description, or (b) a regulatory provision provided in context. Never invent compliance requirements.
2. Flag gaps with [NEEDS INSTITUTIONAL INPUT] — e.g., specific tool names, department structures, budget figures, named roles.
3. Match policy ambition to the institution's assessed maturity level. An "Emerging"/"Incidental" institution needs foundational policies; an "Advanced"/"Optimised" institution needs optimisation and innovation policies.
4. Include numbered clause references for auditability (e.g., 3.1, 3.2).
5. Write in professional policy language, not academic prose. Active voice. Clear obligations ("The institution shall..." not "It is recommended that...").
6. For the user's low-scoring dimensions, include aspirational clauses that describe the pathway to the next maturity level.
7. Use UK English spelling conventions.

OUTPUT FORMAT:
- Start with: "DRAFT — For review by institutional governance and legal teams before adoption."
- Use markdown headings for sections
- Number all policy clauses (e.g., 3.1, 3.2)
- End each section with a "References" sub-section citing the framework dimensions and regulatory articles used
- Include a "Definitions" section at the start defining key terms used in the policy`;

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
    await requireAuth(req);

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
        headers: { ...corsHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({ error: "policy_type is required" }),
      };
    }

    const frameworkDetail = body.framework_id
      ? getFrameworkContextById(body.framework_id)
      : null;

    const userPrompt = buildUserPrompt({
      policy_type: body.policy_type,
      institution_name: body.institution_name || "[Institution Name]",
      region: body.region || "international",
      sector: body.sector || "higher education",
      framework_id: body.framework_id,
      frameworkDetail,
      template: body.template,
      regulatory_provisions: body.regulatory_provisions,
      assessment_summary: body.assessment_summary,
    });

    const stream = generateContentStream("gemini-2.5-pro", SYSTEM_PROMPT, [
      { role: "user", content: userPrompt },
    ]);

    return {
      status: 200,
      headers: {
        ...corsHeaders(),
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
        headers: { ...corsHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({ error: err.message }),
      };
    }
    context.error("policy-generator error:", err);
    return {
      status: 500,
      headers: { ...corsHeaders(), "Content-Type": "application/json" },
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
