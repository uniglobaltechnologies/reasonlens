import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { generateWithTools } from "../shared/ai";
import { requireAuth, AuthError } from "../shared/auth";
import {
  getFrameworkContext,
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

    const frameworkContext = getFrameworkContext();
    const frameworkPaths = getFrameworkPaths();

    const systemPrompt = `${PLATFORM_PREAMBLE}

You recommend frameworks based on the user's professional profile. You MUST only recommend from the exact framework names listed below.

Available Frameworks (detailed):
${frameworkContext}

MATCHING RULES:
- Educators/Lecturers → UNESCO Teacher AI Competency Framework (primary), DigComp 3.0 or ISTE Standards for Educators (secondary)
- Leaders/Administrators → QS AI Capability Framework or JISC AI Maturity Model (primary), UNESCO Guidance as secondary
- Students → UNESCO Student AI Competency Framework (primary), ISTE Standards for Students (secondary)
- Researchers → BDC Researcher Profile (primary), DigComp 3.0 (secondary)
- Learning technologists → BDC Learning Technologist Profile (primary)
- Educational developers → BDC Educational Developer Profile (primary)
- If goal mentions "policy" → add UNESCO Guidance for AI in Education & Research as secondary
- If goal mentions "institutional assessment" or "maturity" → JISC AI Maturity Model or THE Digital Maturity Index
- Consider institution level: K-12 may prefer ISTE; HE should prefer UNESCO/QS/JISC
- Consider AI experience: beginners need individual competency frameworks first; experienced users can tackle institutional maturity
- Recommend 1 primary and 1-2 secondary frameworks maximum`;

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

    const recommendation = {
      primary: {
        name: result.primary_name,
        reason: result.primary_reason,
        path: frameworkPaths[result.primary_name] || "/dashboard",
      },
      secondary: (result.secondary || []).map((fw: any) => ({
        name: fw.name,
        reason: fw.reason,
        path: frameworkPaths[fw.name] || "/dashboard",
      })),
      startWith: result.start_with,
    };

    return {
      status: 200,
      headers: { ...corsHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ recommendation }),
    };
  } catch (err) {
    if (err instanceof AuthError) {
      return { status: err.statusCode, headers: { ...corsHeaders(), "Content-Type": "application/json" }, body: JSON.stringify({ error: err.message }) };
    }
    context.error("framework-recommender error:", err);
    return {
      status: 500,
      headers: { ...corsHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ error: "Internal server error" }),
    };
  }
}

app.http("framework-recommender", {
  methods: ["POST", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
