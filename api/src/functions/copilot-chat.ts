import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { query, queryOne } from "../shared/db";
import { requireAuth, AuthError } from "../shared/auth";
import { generateContentStream, createSSEResponse } from "../shared/ai";
import {
  getFrameworkContextById,
  getFrameworkContextByIds,
  getFrameworkIndex,
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
          "SELECT framework_name, framework_id, dimension, selected_level, completed_at, COALESCE(assessment_method, 'self_report') AS assessment_method FROM assessment_results WHERE user_id = $1 ORDER BY completed_at DESC LIMIT 100",
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

  // --- User profile ---
  if (userContext) {
    const p = userContext.profile;
    sections.push(`USER PROFILE:
- Name: ${p?.full_name || "Anonymous"}
- Institution: ${p?.institution || "Not specified"}${p?.institution_type ? ` (${p.institution_type})` : ""}
- Region: ${p?.region || "Not specified"}
- Sector: ${p?.sector || "Not specified"}
- Self-reported comfort level: ${p?.comfort_level || "Not set"}/5
- Goals: ${userContext.goals?.join(", ") || "Not specified"}`);

    // --- Assessment profile with consistency check ---
    if (userContext.assessmentsByFramework.size > 0) {
      let block = "ASSESSMENT PROFILE (latest self-reported results):";
      for (const [fw, dims] of userContext.assessmentsByFramework) {
        block += `\n  ${fw}:`;
        for (const d of dims) {
          block += `\n    - ${d.dimension}: ${d.level}`;
        }
      }
      // Flag potential inconsistencies
      if (p?.comfort_level && p.comfort_level <= 2) {
        const highLevels = [...userContext.assessmentsByFramework.values()]
          .flat()
          .filter((d) =>
            ["create", "advanced", "optimised", "mastery", "expert", "highly advanced"]
              .some((l) => d.level.toLowerCase().includes(l))
          );
        if (highLevels.length > 0) {
          block += `\n  NOTE: User's comfort level is ${p.comfort_level}/5 but ${highLevels.length} dimensions are self-assessed at advanced levels. Treat with sensitivity — may reflect aspiration rather than current practice.`;
        }
      }
      sections.push(block);
    } else {
      sections.push(
        "ASSESSMENT PROFILE: No assessments completed. Guide them toward an assessment relevant to their role before making specific recommendations."
      );
    }

    // --- Progress (condensed) ---
    const progress: string[] = [];
    if (userContext.learningPaths.length > 0) {
      for (const lp of userContext.learningPaths) {
        progress.push(
          `${lp.framework_name}: ${lp.overall_progress || 0}%`
        );
      }
    }
    progress.push(`Portfolio: ${userContext.portfolioCount} items`);
    progress.push(`Badges: ${userContext.badgeCount}`);
    if (userContext.policyDrafts.length > 0) {
      progress.push(`Policy drafts: ${userContext.policyDrafts.length}`);
    }
    sections.push(`PROGRESS: ${progress.join(" | ")}`);
  }

  // --- Page context + targeted framework injection ---
  const currentPage = context?.page || "Hub";
  sections.push(`CURRENT PAGE: ${currentPage}`);

  // Inject full detail only for the active framework (if viewing one)
  if (context?.frameworkId) {
    const fwDetail = getFrameworkContextById(context.frameworkId);
    if (fwDetail) {
      sections.push(
        `ACTIVE FRAMEWORK (full detail — user is viewing this framework's page):\n${fwDetail}`
      );
    }
  }

  // Inject full detail for user's assessed frameworks (max 3, excluding active)
  if (userContext?.assessmentsByFramework && userContext.assessmentsByFramework.size > 0) {
    const assessedIds: string[] = [];
    // We need framework IDs — fetch from assessments query
    // The assessmentsByFramework map is keyed by name, but we need IDs
    // For now, inject the index which covers all frameworks concisely
  }

  // Lightweight index instead of full 22-framework dump (~500 tokens vs ~23K)
  sections.push(
    `FRAMEWORK INDEX (all 22 — for cross-referencing; ask user to navigate to a framework's page for full detail):\n${getFrameworkIndex()}`
  );

  // --- Reasoning instructions (prioritised) ---
  sections.push(`REASONING PRIORITIES (in order of importance):
1. GROUND IN DATA: Every recommendation must reference the user's actual assessment levels. If they scored "Acquire" on Ethics, do not recommend advanced ethics work — suggest what "Deepen" requires.
2. BE SPECIFIC: Name the exact dimension, current level, next level, and at least one indicator. "Improve your ethics skills" is not acceptable; cite the specific indicator from framework data.
3. RESPECT SCOPE: Individual competency frameworks assess people; institutional maturity frameworks assess organisations. Do not conflate them.
4. CONTEXTUALISE TO REGION: UK → DfE guidance, UK GDPR, DPA 2018. EU → EU AI Act risk categories, GDPR. US → FERPA, NIST AI RMF. International → UNESCO Guidance.
5. MATCH COMFORT LEVEL: Comfort 1-2 → foundational activities. Comfort 4-5 → stretch activities. Flag the gap if recommending above comfort level.
6. CROSS-REFERENCE SPARINGLY: Only cite another framework when it adds genuine value. If you need full detail for a framework not loaded, tell the user to navigate to that framework's page.
7. PRIORITISE ACTIONABLE OVER ASPIRATIONAL: Concrete next steps, not general encouragement.

OUT OF SCOPE:
- Do not help with tasks unrelated to AI in education.
- Do not generate full policy documents — direct users to the Policy Generator page.
- Do not run or interpret safety audit results — direct users to the Audit page.`);

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
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ error: "Messages must be an array of 1-50 items" }),
      };
    }
    for (const m of messages) {
      if (typeof m.content !== "string" || m.content.length > 10000) {
        return {
          status: 400,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "Each message must be under 10,000 characters" }),
        };
      }
    }

    const user = await requireAuth(req);
    const userContext = await fetchUserContext(user.userId);

    const systemPrompt = buildSystemPrompt(chatContext, userContext);
    const stream = generateContentStream(systemPrompt, messages);

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
    context.error("copilot-chat error:", err);
    return {
      status: 500,
      headers: { ...corsHeaders(req), "Content-Type": "application/json" },
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
