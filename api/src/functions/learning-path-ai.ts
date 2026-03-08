import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { query, queryOne, execute } from "../shared/db";
import { requireAuth, AuthError } from "../shared/auth";
import { generateContent } from "../shared/ai";
import { getFrameworkContextById, getFrameworkNameById } from "../shared/framework-context";
import { PLATFORM_PREAMBLE } from "../shared/prompt-preamble";
import { corsHeaders, handleCors } from "../middleware/cors";

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  try {
    const user = await requireAuth(req);
    const { frameworkId } = (await req.json()) as { frameworkId: string };

    if (!frameworkId) {
      return {
        status: 400,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ error: "frameworkId is required" }),
      };
    }

    // Fetch user's assessment results for this framework
    const results = await query<{
      dimension: string;
      selected_level: string;
    }>(
      "SELECT DISTINCT ON (dimension) dimension, selected_level FROM assessment_results WHERE user_id = $1 AND framework_id = $2 ORDER BY dimension, completed_at DESC",
      [user.userId, frameworkId]
    );

    if (results.length === 0) {
      return {
        status: 200,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({
          gaps: [],
          recommendations: [],
          message: "Complete an assessment for this framework first.",
        }),
      };
    }

    // Fetch user profile for context
    const profile = await queryOne<{
      institution: string;
      region: string;
      sector: string;
      comfort_level: number;
    }>(
      "SELECT institution, region, sector, comfort_level FROM profiles WHERE id = $1",
      [user.userId]
    );

    // Check evidence status
    const evidence = await query<{ dimension: string }>(
      "SELECT DISTINCT dimension FROM assessment_evidence WHERE user_id = $1 AND framework_id = $2",
      [user.userId, frameworkId]
    );
    const evidencedDimensions = new Set(evidence.map(e => e.dimension));

    // Get framework context for AI
    const frameworkDetail = getFrameworkContextById(frameworkId);

    // Build gap analysis summary
    const gapSummary = results.map(r => ({
      dimension: r.dimension,
      currentLevel: r.selected_level,
      hasEvidence: evidencedDimensions.has(r.dimension),
    }));

    const systemPrompt = `${PLATFORM_PREAMBLE}

You are a learning path advisor. Generate personalised, actionable learning recommendations based on the user's framework assessment gaps.

FRAMEWORK DETAIL:
${frameworkDetail || "Framework detail not available."}

PRIORITISATION CRITERIA (in order):
1. Largest gap first: dimensions where the user scored lowest get highest priority.
2. Self-reported over evidenced: dimensions without evidence are less reliable and need development attention.
3. Dependency ordering: if dimension B requires skills from dimension A, recommend A first.
4. Comfort-level matching: if user comfort is 1-2, prioritise foundational actions; if 4-5, include stretch activities.

INSTRUCTIONS:
1. For each dimension, identify the user's current level and what the next level requires.
2. Reference specific indicators from the framework data. Quote the indicator description, do not paraphrase.
3. Cap total recommendations at 8 across all dimensions (not 3-5 per dimension).
4. Include estimated time for each action.
5. Consider the user's institution type, region, and comfort level.
6. Use UK English for UK users, US English for US users, International English otherwise.
7. Each action MUST specify a resource_type from: self-study, workshop, peer-activity, tool-exploration, reflection, institutional-action.
8. At least 50% of actions should include a portfolio_evidence suggestion — a concrete artefact the user can create to evidence their learning (e.g., "Write a 500-word reflection on...", "Create a rubric for...", "Document a case study of...").
9. For frameworks without clear level progression (e.g., ISTE Standards), focus on competency indicators rather than "next level".

OUTPUT FORMAT:
Return a JSON object with this structure:
{
  "recommendations": [
    {
      "dimension": "dimension name",
      "priority": "high" | "medium" | "low",
      "currentLevel": "level name",
      "nextLevel": "target level name",
      "actions": [
        {
          "title": "short title",
          "description": "detailed action",
          "estimatedTime": "2-3 hours",
          "resource_type": "self-study",
          "portfolio_evidence": "optional: concrete artefact to create"
        }
      ],
      "frameworkIndicators": ["specific indicators to target"]
    }
  ]
}`;

    const userPrompt = `Generate learning path recommendations for this user:

Assessment Results (framework: ${frameworkId}):
${gapSummary.map(g => `- ${g.dimension}: ${g.currentLevel}${g.hasEvidence ? " (evidenced)" : " (self-reported)"}`).join("\n")}

User Context:
- Institution: ${profile?.institution || "Not specified"}
- Region: ${profile?.region || "Not specified"}
- Sector: ${profile?.sector || "Not specified"}
- Comfort level: ${profile?.comfort_level || 1}/5

Return ONLY the JSON object, no markdown fences.`;

    const aiResponse = await generateContent(
      systemPrompt,
      [{ role: "user", content: userPrompt }]
    );

    // Parse AI response
    let recommendations: any[] = [];
    try {
      const cleaned = aiResponse.replace(/```json\n?/g, "").replace(/```\n?/g, "").trim();
      const parsed = JSON.parse(cleaned);
      recommendations = parsed.recommendations || [];
    } catch {
      context.warn("Failed to parse AI recommendations, returning raw");
      recommendations = [{ dimension: "General", priority: "medium", actions: [{ title: "Review", description: aiResponse, estimatedTime: "Variable" }] }];
    }

    // Persist to learning_paths table
    const frameworkName = getFrameworkNameById(frameworkId);
    await execute(
      `INSERT INTO learning_paths (user_id, framework_id, framework_name, recommendations, overall_progress, dimension_gaps, ai_recommendations, generated_at)
       VALUES ($1, $2, $3, $4, 0, $5, $6, now())
       ON CONFLICT (user_id, framework_id) DO UPDATE SET
         recommendations = $4, dimension_gaps = $5, ai_recommendations = $6, generated_at = now(), updated_at = now()`,
      [user.userId, frameworkId, frameworkName, JSON.stringify(recommendations), JSON.stringify(gapSummary), JSON.stringify(recommendations)]
    );

    return {
      status: 200,
      headers: { ...corsHeaders(req), "Content-Type": "application/json" },
      body: JSON.stringify({ gaps: gapSummary, recommendations }),
    };
  } catch (err) {
    if (err instanceof AuthError) {
      return { status: err.statusCode, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: err.message }) };
    }
    context.error("learning-path-ai error:", err);
    return { status: 500, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: "Internal server error" }) };
  }
}

app.http("learning-path-ai", {
  methods: ["POST", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
