import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { query, queryOne } from "../shared/db";
import { validateToken } from "../shared/auth";
import { generateContentStream, createSSEResponse } from "../shared/ai";
import {
  getFrameworkContext,
  getFrameworkContextById,
} from "../shared/framework-context";
import { PLATFORM_PREAMBLE } from "../shared/prompt-preamble";
import { corsHeaders, handleCors } from "../middleware/cors";

interface UserContext {
  profile: any;
  assessmentsByFramework: Map<string, { dimension: string; level: string }[]>;
  portfolioCount: number;
  portfolioTypes: string[];
  badgeCount: number;
  learningPaths: any[];
  policyDrafts: any[];
  goals: string[];
}

async function fetchUserContext(
  userId: string
): Promise<UserContext | null> {
  try {
    const [profile, assessments, portfolioItems, userBadges, learningPaths, policyDrafts, goals] =
      await Promise.all([
        queryOne(
          "SELECT full_name, institution, region, sector, institution_type, comfort_level FROM profiles WHERE id = $1",
          [userId]
        ),
        query(
          "SELECT framework_name, framework_id, dimension, selected_level, completed_at FROM assessment_results WHERE user_id = $1 ORDER BY completed_at DESC LIMIT 100",
          [userId]
        ),
        query(
          "SELECT id, artifact_type FROM portfolio_items WHERE user_id = $1",
          [userId]
        ),
        query(
          "SELECT badge_id FROM user_badges WHERE user_id = $1",
          [userId]
        ),
        query(
          "SELECT framework_name, overall_progress FROM learning_paths WHERE user_id = $1",
          [userId]
        ),
        query(
          "SELECT policy_type, status FROM policy_drafts WHERE user_id = $1",
          [userId]
        ),
        query("SELECT goal FROM user_goals WHERE user_id = $1", [userId]),
      ]);

    // Deduplicate assessments: keep latest per framework+dimension
    const deduped = new Map<string, any>();
    for (const a of assessments) {
      const key = `${a.framework_name}::${a.dimension}`;
      if (!deduped.has(key)) deduped.set(key, a);
    }

    const assessmentsByFramework = new Map<
      string,
      { dimension: string; level: string }[]
    >();
    for (const a of deduped.values()) {
      const list = assessmentsByFramework.get(a.framework_name) || [];
      list.push({ dimension: a.dimension, level: a.selected_level });
      assessmentsByFramework.set(a.framework_name, list);
    }

    return {
      profile,
      assessmentsByFramework,
      portfolioCount: portfolioItems.length,
      portfolioTypes: [
        ...new Set(portfolioItems.map((p: any) => p.artifact_type)),
      ],
      badgeCount: userBadges.length,
      learningPaths,
      policyDrafts,
      goals: goals.map((g: any) => g.goal),
    };
  } catch (error) {
    console.error("Error fetching user context:", error);
    return null;
  }
}

function buildSystemPrompt(
  context: any,
  userContext: UserContext | null
): string {
  const sections: string[] = [PLATFORM_PREAMBLE];

  if (userContext) {
    const p = userContext.profile;
    sections.push(`PERSONALISATION CONTEXT:
- Name: ${p?.full_name || "Not set"}
- Institution: ${p?.institution || "Not specified"}${p?.institution_type ? ` (${p.institution_type})` : ""}
- Region: ${p?.region || "Not specified"}
- Sector: ${p?.sector || "Not specified"}
- Self-reported comfort level: ${p?.comfort_level || 1}/5
- Role: ${context?.userRole || "Not specified"}
- Goals: ${(context?.goals?.length ? context.goals : userContext.goals)?.join(", ") || "Not specified"}`);

    if (userContext.assessmentsByFramework.size > 0) {
      let assessmentBlock =
        "ASSESSMENT PROFILE (latest results, grouped by framework):";
      for (const [fw, dims] of userContext.assessmentsByFramework) {
        assessmentBlock += `\n  ${fw}:`;
        for (const d of dims) {
          assessmentBlock += `\n    - ${d.dimension}: ${d.level}`;
        }
      }
      sections.push(assessmentBlock);
    } else {
      sections.push(
        "ASSESSMENT PROFILE: No assessments completed yet. Encourage the user to start with a self-assessment."
      );
    }

    const progressLines: string[] = [];
    if (userContext.learningPaths.length > 0) {
      for (const lp of userContext.learningPaths) {
        progressLines.push(
          `- ${lp.framework_name}: ${lp.overall_progress || 0}% complete`
        );
      }
    }
    progressLines.push(
      `- Portfolio: ${userContext.portfolioCount} items${userContext.portfolioTypes.length ? ` (${userContext.portfolioTypes.join(", ")})` : ""}`
    );
    progressLines.push(`- Badges earned: ${userContext.badgeCount}`);
    if (userContext.policyDrafts.length > 0) {
      const draftTypes = [
        ...new Set(userContext.policyDrafts.map((d: any) => d.policy_type)),
      ];
      progressLines.push(
        `- Policy drafts: ${userContext.policyDrafts.length} (${draftTypes.join(", ")})`
      );
    }
    sections.push(`LEARNING PROGRESS:\n${progressLines.join("\n")}`);
  }

  const currentPage = context?.page || "Dashboard";
  sections.push(`CURRENT PAGE: ${currentPage}`);

  if (context?.frameworkId) {
    const fwDetail = getFrameworkContextById(context.frameworkId);
    if (fwDetail) {
      sections.push(
        `CURRENT FRAMEWORK (full detail for the page the user is viewing):\n${fwDetail}`
      );
    }
  }

  sections.push(`ESCO SKILLS INTEGRATION (DigComp 3.0 only):
DigComp 3.0 is linked to the EU's ESCO taxonomy via an official JRC mapping of 732 skills from European job advertisements (2018-2022). Key facts:
- Area 1 (Information & data literacy): highest coverage — skills like "manage time" (25M+ mentions), "statistics" (4.5M)
- Area 2 (Communication): "liaise with managers" (8.9M), "coordinate communication" (5.2M)
- Area 3 (Content creation): "report analysis results" (8.3M), "3D modelling" (128K)
- Areas 4-5 (Safety, Problem solving): low ESCO coverage — fewer digital-specific skills in taxonomy
When discussing DigComp competences, reference ESCO data to show labour market relevance.`);

  sections.push(
    `AVAILABLE FRAMEWORKS (all 22 — use for cross-referencing):\n${getFrameworkContext()}`
  );

  sections.push(`INSTRUCTIONS — HOW TO REASON:
1. Ground every recommendation in the user's actual assessment levels. If they scored "Acquire" on Ethics, don't recommend advanced ethics activities — suggest what "Deepen" requires.
2. When suggesting next steps, name the specific dimension and the next level up with its indicators.
3. Cross-reference frameworks when relevant.
4. For portfolio advice, suggest evidence types that map to the user's weakest dimensions.
5. If the user has no assessments, guide them to start one relevant to their role.
6. Consider the user's region when discussing regulatory context (UK → DfE, EU → AI Act, International → UNESCO Guidance).
7. Consider the user's institution type and sector when suggesting implementation approaches.
8. Prioritise actionable, concrete suggestions over general encouragement.
9. If asked about a framework, cite specific indicators from the framework data provided.
10. Never suggest tools or approaches beyond the user's current comfort level without acknowledging the stretch.`);

  return sections.join("\n\n");
}

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  try {
    const { messages, context: chatContext } = (await req.json()) as {
      messages: Array<{ role: string; content: string }>;
      context?: any;
    };

    // Input validation
    if (!Array.isArray(messages) || messages.length === 0 || messages.length > 50) {
      return {
        status: 400,
        headers: { ...corsHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({ error: "Messages must be an array of 1-50 items" }),
      };
    }
    for (const m of messages) {
      if (typeof m.content !== "string" || m.content.length > 10000) {
        return {
          status: 400,
          headers: { ...corsHeaders(), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "Each message must be under 10,000 characters" }),
        };
      }
    }

    const user = await validateToken(req);
    const userContext = user
      ? await fetchUserContext(user.userId)
      : null;

    const systemPrompt = buildSystemPrompt(chatContext, userContext);
    const stream = generateContentStream(systemPrompt, messages);

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
    context.error("copilot-chat error:", err);
    return {
      status: 500,
      headers: { ...corsHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({
        error: err instanceof Error ? err.message : "Unknown error",
      }),
    };
  }
}

app.http("copilot-chat", {
  methods: ["POST", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
