import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { generateWithTools } from "../shared/ai";
import { requireAuth, AuthError } from "../shared/auth";
import {
  getFrameworkIndex,
  getFrameworkPaths,
} from "../shared/framework-context";
import {
  PLATFORM_PREAMBLE,
  FRAMEWORK_NAMES_ENUM,
} from "../shared/prompt-preamble";
import { corsHeaders, handleCors } from "../middleware/cors";

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  try {
    await requireAuth(req);

    const { answers } = (await req.json()) as {
      answers: {
        role: string;
        primaryGoal: string;
        institutionLevel: string;
        aiExperience: string;
        focusArea: string;
      };
    };

    const frameworkIndex = getFrameworkIndex();
    const frameworkPaths = getFrameworkPaths();

    const systemPrompt = `${PLATFORM_PREAMBLE}

You recommend frameworks based on the user's professional profile. You MUST only recommend from the exact framework names listed in the FRAMEWORK_NAMES_ENUM.

FRAMEWORK INDEX:
${frameworkIndex}

STEP 1 — DETERMINE SCOPE:
- Is the user asking about their own skills/competencies? → Individual competency framework
- Is the user asking about their institution's readiness/maturity? → Institutional maturity framework
- Unclear? → Default to individual competency for educators/students, institutional for leaders/administrators

STEP 2 — MATCH BY ROLE:
- Educators/Lecturers → UNESCO Teacher AI Competency Framework (primary), DigComp 3.0 or ISTE Standards for Educators (secondary)
- Leaders/Administrators → QS AI Capability Framework or JISC AI Maturity Model (primary), UNESCO Guidance as secondary
- Students → UNESCO Student AI Competency Framework (primary), ISTE Standards for Students (secondary)
- Researchers → JISC BDC Researcher Profile (primary), DigComp 3.0 (secondary)
- Learning technologists → JISC BDC Learning Technology Profile (primary)
- Educational developers → JISC BDC Educational Developer Profile (primary)
- Digital leaders → JISC BDC Digital Leader Profile (primary)
- Coaches/Mentors → ISTE Standards for Coaches (primary)
- Professional services → JISC BDC Professional Services Profile (primary)

STEP 3 — ADJUST BY CONTEXT:
- If goal mentions "policy" → add UNESCO Guidance for AI in Education & Research as secondary
- If goal mentions "institutional assessment" or "maturity" → JISC AI Maturity Model or THE Digital Maturity Index
- Regional preferences: EU → DigComp 3.0 should be considered; US → ISTE; UK → JISC/BDC frameworks
- Institution level: K-12 → prefer ISTE; HE → prefer UNESCO/QS/JISC
- AI experience: beginners → individual competency first; experienced → can tackle institutional maturity

STEP 4 — PROVIDE ACTIONABLE START:
The start_with field must be a concrete, immediate action. NOT "explore the framework" or "familiarise yourself with the dimensions".
Good: "Complete the self-assessment for the Teaching & Learning dimension to identify your current level."
Bad: "Look through the framework to understand its structure."

OUTPUT: 1 primary + 1-2 secondary frameworks maximum.`;

    const userPrompt = `User Profile:
- Role: ${answers.role}
- Primary Goal: ${answers.primaryGoal}
- Institution Level: ${answers.institutionLevel}
- AI Experience: ${answers.aiExperience}
- Focus Area: ${answers.focusArea}

Use the recommend_frameworks function to return your recommendation.`;

    const result = await generateWithTools(
      systemPrompt,
      [{ role: "user", content: userPrompt }],
      [
        {
          type: "function" as const,
          function: {
            name: "recommend_frameworks",
            description: "Return framework recommendations for the user",
            parameters: {
              type: "object",
              properties: {
                primary_name: {
                  type: "string",
                  enum: [...FRAMEWORK_NAMES_ENUM],
                  description: "The exact name of the primary recommended framework",
                },
                primary_reason: {
                  type: "string",
                  description: "One concise sentence (max 20 words) explaining why this is the best match",
                },
                secondary: {
                  type: "array",
                  items: {
                    type: "object",
                    properties: {
                      name: { type: "string", enum: [...FRAMEWORK_NAMES_ENUM], description: "Exact framework name" },
                      reason: { type: "string", description: "One concise sentence (max 15 words)" },
                    },
                    required: ["name", "reason"],
                  },
                  description: "1-2 secondary framework recommendations",
                },
                start_with: {
                  type: "string",
                  description: "One sentence (max 25 words) describing the immediate next step",
                },
              },
              required: ["primary_name", "primary_reason", "secondary", "start_with"],
            },
          },
        },
      ]
    );

    if (!result) {
      throw new Error("No tool call in AI response");
    }

    // Output validation: verify framework names exist in paths lookup
    const validNames = new Set(Object.keys(frameworkPaths));
    if (!validNames.has(result.primary_name)) {
      context.warn(`Invalid primary framework name from LLM: "${result.primary_name}". Falling back.`);
      // Try to find a close match
      const match = [...validNames].find(n => n.toLowerCase().includes(result.primary_name?.toLowerCase()?.split(" ")[0] || ""));
      if (match) result.primary_name = match;
    }
    const validatedSecondary = (result.secondary || []).filter((fw: any) => {
      if (validNames.has(fw.name)) return true;
      context.warn(`Invalid secondary framework name from LLM: "${fw.name}". Removed.`);
      return false;
    });

    const recommendation = {
      primary: {
        name: result.primary_name,
        reason: result.primary_reason,
        path: frameworkPaths[result.primary_name] || "/dashboard",
      },
      secondary: validatedSecondary.map((fw: any) => ({
        name: fw.name,
        reason: fw.reason,
        path: frameworkPaths[fw.name] || "/dashboard",
      })),
      startWith: result.start_with,
    };

    return {
      status: 200,
      headers: { ...corsHeaders(req), "Content-Type": "application/json" },
      body: JSON.stringify({ recommendation }),
    };
  } catch (err) {
    if (err instanceof AuthError) {
      return { status: err.statusCode, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: err.message }) };
    }
    context.error("framework-recommender error:", err);
    return {
      status: 500,
      headers: { ...corsHeaders(req), "Content-Type": "application/json" },
      body: JSON.stringify({ error: "Internal server error" }),
    };
  }
}

app.http("framework-recommender", {
  methods: ["POST", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
