import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { query } from "../shared/db";
import { requireAuth, AuthError } from "../shared/auth";
import { corsHeaders, handleCors } from "../middleware/cors";

const dimensionMappings = [
  { framework_id: "maturity-jisc-ai", dimension_id: "governance-ethics", low_score_triggers: ["ai-acceptable-use", "ai-governance"] },
  { framework_id: "maturity-jisc-ai", dimension_id: "pedagogy-practice", low_score_triggers: ["ai-assessment-integrity"] },
  { framework_id: "maturity-jisc-ai", dimension_id: "workforce-development", low_score_triggers: ["staff-ai-development"] },
  { framework_id: "maturity-jisc-ai", dimension_id: "infrastructure-data", low_score_triggers: ["ai-data-governance"] },
  { framework_id: "maturity-jisc-ai", dimension_id: "student-experience", low_score_triggers: ["student-ai-guidance"] },
  { framework_id: "maturity-the", dimension_id: "culture-strategy", low_score_triggers: ["ai-governance"] },
  { framework_id: "maturity-the", dimension_id: "content-curriculum", low_score_triggers: ["ai-assessment-integrity"] },
  { framework_id: "maturity-the", dimension_id: "data-analytics", low_score_triggers: ["ai-data-governance"] },
  { framework_id: "maturity-the", dimension_id: "teaching-learning", low_score_triggers: ["ai-assessment-integrity", "staff-ai-development"] },
  { framework_id: "maturity-the", dimension_id: "operations-support", low_score_triggers: ["ai-governance"] },
  { framework_id: "ai-capability", dimension_id: "ai-vision-strategy", low_score_triggers: ["ai-governance"] },
  { framework_id: "ai-capability", dimension_id: "ai-governance-ethics", low_score_triggers: ["ai-acceptable-use", "ai-governance"] },
  { framework_id: "ai-capability", dimension_id: "ai-skills-workforce", low_score_triggers: ["staff-ai-development"] },
  { framework_id: "ai-capability", dimension_id: "ai-infrastructure", low_score_triggers: ["ai-data-governance"] },
  { framework_id: "ai-capability", dimension_id: "ai-teaching-learning", low_score_triggers: ["ai-assessment-integrity"] },
];

const policyTypeInfo: Record<string, { name: string; description: string }> = {
  "ai-acceptable-use": { name: "AI Acceptable Use Policy", description: "Defines permitted and prohibited uses of AI tools" },
  "ai-governance": { name: "AI Governance Policy", description: "Establishes strategic AI governance structures and accountability" },
  "ai-assessment-integrity": { name: "AI Assessment Integrity Policy", description: "Governs AI use in educational assessment" },
  "staff-ai-development": { name: "Staff AI Development Policy", description: "Defines AI literacy requirements and training programmes" },
  "ai-data-governance": { name: "AI Data Governance Policy", description: "Governs data processing and privacy in AI systems" },
  "student-ai-guidance": { name: "Student AI Guidance", description: "Student-facing guidance on responsible AI use" },
};

function levelToScore(level: string): number {
  const l = level.toLowerCase();
  if (l.includes("emerging") || l.includes("initial") || l.includes("level-1") || l === "1") return 1;
  if (l.includes("developing") || l.includes("level-2") || l === "2") return 2;
  if (l.includes("established") || l.includes("defined") || l.includes("level-3") || l === "3") return 3;
  if (l.includes("advanced") || l.includes("embedded") || l.includes("level-4") || l === "4") return 4;
  if (l.includes("leading") || l.includes("optimising") || l.includes("level-5") || l === "5") return 5;
  return 2;
}

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  try {
    const user = await requireAuth(req);

    const results = await query<{
      framework_id: string;
      dimension: string;
      selected_level: string;
    }>(
      "SELECT framework_id, dimension, selected_level FROM assessment_results WHERE user_id = $1",
      [user.userId]
    );

    if (results.length === 0) {
      return {
        status: 200,
        headers: { ...corsHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({
          recommendations: [],
          message: "Complete an institutional framework assessment first to get policy recommendations.",
        }),
      };
    }

    const frameworkDimensions: Record<string, Record<string, number[]>> = {};
    for (const r of results) {
      if (!frameworkDimensions[r.framework_id]) frameworkDimensions[r.framework_id] = {};
      if (!frameworkDimensions[r.framework_id][r.dimension]) frameworkDimensions[r.framework_id][r.dimension] = [];
      frameworkDimensions[r.framework_id][r.dimension].push(levelToScore(r.selected_level));
    }

    const triggeredPolicies: Record<string, { count: number; rationales: string[] }> = {};

    for (const [fwId, dims] of Object.entries(frameworkDimensions)) {
      const dimScores = Object.entries(dims).map(([dimId, scores]) => ({
        dimId,
        avg: scores.reduce((a, b) => a + b, 0) / scores.length,
      }));
      dimScores.sort((a, b) => a.avg - b.avg);
      const lowDims = dimScores.slice(0, 3);

      for (const dim of lowDims) {
        const mapping = dimensionMappings.find(
          (m) => m.framework_id === fwId && m.dimension_id === dim.dimId
        );
        if (!mapping) continue;

        for (const policyType of mapping.low_score_triggers) {
          if (!triggeredPolicies[policyType])
            triggeredPolicies[policyType] = { count: 0, rationales: [] };
          triggeredPolicies[policyType].count++;
          triggeredPolicies[policyType].rationales.push(
            `Low score on "${dim.dimId}" in ${fwId} (avg: ${dim.avg.toFixed(1)}/5)`
          );
        }
      }
    }

    const recommendations = Object.entries(triggeredPolicies)
      .sort((a, b) => b[1].count - a[1].count)
      .map(([policyId, data]) => ({
        policy_type: policyId,
        ...policyTypeInfo[policyId],
        priority: data.count >= 3 ? "high" : data.count >= 2 ? "medium" : "recommended",
        rationale: data.rationales.slice(0, 3).join("; "),
        trigger_count: data.count,
      }));

    if (!triggeredPolicies["student-ai-guidance"]) {
      recommendations.push({
        policy_type: "student-ai-guidance",
        ...policyTypeInfo["student-ai-guidance"],
        priority: "recommended",
        rationale: "Recommended for all institutions deploying AI tools",
        trigger_count: 0,
      });
    }

    return {
      status: 200,
      headers: { ...corsHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ recommendations }),
    };
  } catch (err) {
    if (err instanceof AuthError) {
      return {
        status: err.statusCode,
        headers: { ...corsHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({ error: err.message }),
      };
    }
    context.error("policy-recommender error:", err);
    return {
      status: 500,
      headers: { ...corsHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ error: "Internal server error" }),
    };
  }
}

app.http("policy-recommender", {
  methods: ["POST", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
