import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { generateWithTools } from "../shared/ai";
import { getFrameworkContext } from "../shared/framework-context";
import { PLATFORM_PREAMBLE } from "../shared/prompt-preamble";
import { corsHeaders, handleCors } from "../middleware/cors";

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  try {
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

    const frameworkContext = getFrameworkContext();

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

FRAMEWORK CONTEXT (use to inform your evaluation):
${frameworkContext}

EVALUATION INSTRUCTIONS:
1. Consider the user's institutional context — region affects regulatory requirements (UK → DfE guidance, EU → EU AI Act risk categories, International → UNESCO principles), sector affects risk tolerance.
2. Map feasibility to the user's current maturity level if assessment data is available. An "Emerging" institution needs more safeguards than an "Advanced" one.
3. Reference specific framework indicators that relate to the task.
4. Be realistic about AI capabilities. Distinguish between what AI *can* do technically and what it *should* do in an educational context.
5. Always consider student welfare, academic integrity, and data protection.
6. For the "implementation" field, provide concrete, step-by-step guidance — not vague suggestions.
7. Tailor safeguards to the user's region: cite GDPR for EU, UK GDPR + DPA 2018 for UK, FERPA for US institutions.
8. If the task involves student-facing AI, always include safeguards around transparency and appeal mechanisms.`;

    const result = await generateWithTools(
      "gemini-2.5-flash",
      systemPrompt,
      [{ role: "user", content: `Evaluate this task: ${taskDescription}` }],
      [
        {
          name: "evaluate_task",
          description:
            "Evaluate if AI can handle a given educational task",
          parameters: {
            type: "OBJECT" as any,
            properties: {
              feasibility: {
                type: "NUMBER" as any,
                description:
                  "Score from 1-5 indicating how well AI can handle this task",
              },
              recommendation: {
                type: "STRING" as any,
                enum: ["augment", "automate", "avoid"],
                description:
                  "Whether to augment (AI assists), automate (AI leads), or avoid using AI",
              },
              reasoning: {
                type: "STRING" as any,
                description:
                  "Brief explanation of the recommendation, referencing relevant framework indicators",
              },
              safeguards: {
                type: "ARRAY" as any,
                items: { type: "STRING" as any },
                description:
                  "List of safeguards needed, tailored to the user's region and sector",
              },
              risks: {
                type: "ARRAY" as any,
                items: { type: "STRING" as any },
                description: "Potential risks to consider",
              },
              implementation: {
                type: "STRING" as any,
                description:
                  "Concrete, step-by-step implementation guidance",
              },
            },
            required: [
              "feasibility",
              "recommendation",
              "reasoning",
              "safeguards",
              "risks",
              "implementation",
            ],
          },
        },
      ]
    );

    if (!result) {
      throw new Error("No tool call in AI response");
    }

    return {
      status: 200,
      headers: { ...corsHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify(result),
    };
  } catch (err) {
    context.error("task-evaluator error:", err);
    return {
      status: 500,
      headers: { ...corsHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({
        error: err instanceof Error ? err.message : "Unknown error",
      }),
    };
  }
}

app.http("task-evaluator", {
  methods: ["POST", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
