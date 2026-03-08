import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { generateWithTools } from "../shared/ai";
import { requireAuth, AuthError } from "../shared/auth";
import { getFrameworkIndex } from "../shared/framework-context";
import { PLATFORM_PREAMBLE } from "../shared/prompt-preamble";
import { corsHeaders, handleCors } from "../middleware/cors";

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  try {
    await requireAuth(req);

    const { taskDescription, userContext } = (await req.json()) as {
      taskDescription: string;
      userContext?: {
        role?: string;
        region?: string;
        sector?: string;
        institutionType?: string;
        comfortLevel?: number;
        assessmentSummary?: string;
      };
    };

    const frameworkIndex = getFrameworkIndex();

    let userContextBlock = "";
    if (userContext) {
      userContextBlock = `\nUSER CONTEXT:
- Role: ${userContext.role || "Not specified"}
- Region: ${userContext.region || "Not specified"}
- Sector: ${userContext.sector || "Not specified"}
- Institution type: ${userContext.institutionType || "Not specified"}
- AI comfort level: ${userContext.comfortLevel || "Not specified"}/5`;
      if (userContext.assessmentSummary) {
        userContextBlock += `\n- Relevant assessment levels:\n${userContext.assessmentSummary}`;
      }
    }

    const systemPrompt = `${PLATFORM_PREAMBLE}
${userContextBlock}

FRAMEWORK INDEX (for cross-referencing):
${frameworkIndex}

EVALUATION METHOD:
Step 1 — CLASSIFY THE TASK using OECD categories:
- Routine Cognitive: rule-based, predictable (e.g., marking multiple choice, generating rubrics from criteria)
- Non-Routine Cognitive: judgment, creativity, ambiguity (e.g., providing pastoral feedback, designing novel assessments)
- Interpersonal: relationship-dependent, emotional (e.g., student welfare conversations, conflict resolution)
- Manual: physical presence required (e.g., lab supervision, physical demonstrations)

Step 2 — ASSESS FEASIBILITY on a 1–5 scale:
1 = Not Feasible: AI cannot meaningfully perform this task (e.g., in-person student mentoring, physical lab safety)
2 = Low Feasibility: AI can assist minimally but human judgment dominates (e.g., complex pastoral care, novel research supervision)
3 = Moderate: AI can handle structured components but needs human oversight (e.g., initial essay feedback with educator review)
4 = High Feasibility: AI can perform most of the task with light human oversight (e.g., generating quiz questions from syllabus)
5 = Highly Suitable: AI can handle this reliably with standard guardrails (e.g., summarising meeting notes, translating materials)

Step 3 — RECOMMEND one of:
- "augment": AI assists a human who retains control and final decision
- "automate": AI performs the task end-to-end with periodic human review
- "avoid": AI should not be used for this task

Step 4 — IDENTIFY SAFEGUARDS tailored to the user's region:
- UK → UK GDPR, DPA 2018, DfE guidance on AI in education
- EU → EU AI Act risk categories, GDPR
- US → FERPA, NIST AI RMF, state-level regulations
- International → UNESCO Guidance principles

CALIBRATION EXAMPLES:
- "Grade 200 multiple-choice exams" → Routine Cognitive, feasibility 5, automate. Safeguards: verify answer key, random audit 5%.
- "Write personalised feedback on student essays" → Non-Routine Cognitive, feasibility 3, augment. Safeguards: educator reviews all AI feedback before release, student informed AI was used.
- "Conduct a student disciplinary hearing" → Interpersonal, feasibility 1, avoid. Reasoning: requires procedural fairness, emotional sensitivity, legal accountability that AI cannot provide.

RULES:
1. Be realistic about AI capabilities. Distinguish between what AI *can* do technically and what it *should* do in an educational context.
2. Always consider student welfare, academic integrity, and data protection.
3. For the "implementation" field, provide concrete, step-by-step guidance — not vague suggestions.
4. If the task involves student-facing AI, always include safeguards around transparency and appeal mechanisms.
5. Reference the OECD category in your reasoning.`;

    const result = await generateWithTools(
      systemPrompt,
      [{ role: "user", content: `Evaluate this task: ${taskDescription}` }],
      [
        {
          type: "function" as const,
          function: {
            name: "evaluate_task",
            description: "Evaluate if AI can handle a given educational task",
            parameters: {
              type: "object",
              properties: {
                feasibility: {
                  type: "number",
                  description: "Score from 1-5 indicating how well AI can handle this task",
                },
                recommendation: {
                  type: "string",
                  enum: ["augment", "automate", "avoid"],
                  description: "Whether to augment (AI assists), automate (AI leads), or avoid using AI",
                },
                reasoning: {
                  type: "string",
                  description: "Brief explanation of the recommendation, referencing relevant framework indicators",
                },
                safeguards: {
                  type: "array",
                  items: { type: "string" },
                  description: "List of safeguards needed, tailored to the user's region and sector",
                },
                risks: {
                  type: "array",
                  items: { type: "string" },
                  description: "Potential risks to consider",
                },
                implementation: {
                  type: "string",
                  description: "Concrete, step-by-step implementation guidance",
                },
              },
              required: ["feasibility", "recommendation", "reasoning", "safeguards", "risks", "implementation"],
            },
          },
        },
      ]
    );

    if (!result) {
      throw new Error("No tool call in AI response");
    }

    // Output validation: clamp feasibility, validate recommendation
    if (typeof result.feasibility === "number") {
      result.feasibility = Math.max(1, Math.min(5, Math.round(result.feasibility)));
    } else {
      result.feasibility = 3; // Default to moderate
    }
    const validRecommendations = ["augment", "automate", "avoid"];
    if (!validRecommendations.includes(result.recommendation)) {
      result.recommendation = "augment"; // Safe default
    }

    return {
      status: 200,
      headers: { ...corsHeaders(req), "Content-Type": "application/json" },
      body: JSON.stringify(result),
    };
  } catch (err) {
    if (err instanceof AuthError) {
      return { status: err.statusCode, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: err.message }) };
    }
    context.error("task-evaluator error:", err);
    return {
      status: 500,
      headers: { ...corsHeaders(req), "Content-Type": "application/json" },
      body: JSON.stringify({ error: "Internal server error" }),
    };
  }
}

app.http("task-evaluator", {
  methods: ["POST", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
