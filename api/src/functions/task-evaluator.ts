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
import * as fs from "fs";
import * as path from "path";

// ── Load OECD rubric and calibration data ──────────────────────────
function loadOecdData() {
  // At runtime __dirname is dist/src/functions/ or src/functions/
  const candidates = [
    path.join(__dirname, "..", "data", "oecd"),           // dist/src/data/oecd or src/data/oecd
    path.join(__dirname, "..", "..", "src", "data", "oecd"), // from dist/ up to src/data/oecd
    path.join(__dirname, "..", "..", "data", "oecd"),      // fallback
  ];
  const dir = candidates.find(d => fs.existsSync(d)) || candidates[0];

  let rubric: any = null;
  let calibration: any = null;
  try {
    rubric = JSON.parse(fs.readFileSync(path.join(dir, "task-classification-rubric.json"), "utf-8"));
  } catch { /* fall back to inline */ }
  try {
    calibration = JSON.parse(fs.readFileSync(path.join(dir, "task-evaluation-calibration.json"), "utf-8"));
  } catch { /* fall back to inline */ }
  return { rubric, calibration };
}

function buildTaskTypeBlock(rubric: any): string {
  if (!rubric?.task_types) return "";
  return rubric.task_types.map((t: any) =>
    `### ${t.label} (${t.code}) — AI readiness: ${t.baseline_ai_readiness} (${t.ai_readiness_range[0]}-${t.ai_readiness_range[1]})
${t.definition}
Core characteristics: ${t.core_characteristics.join("; ")}
Discriminators: ${t.primary_discriminators.join(" | ")}
Examples: ${t.typical_domain_examples.slice(0, 4).join(", ")}`
  ).join("\n\n");
}

function buildBoundaryBlock(rubric: any): string {
  if (!rubric?.boundary_definitions) return "";
  return Object.values(rubric.boundary_definitions).map((b: any) =>
    `**${b.boundary_id}**: ${b.one_sentence}
Resolution: ${b.resolution_principle}`
  ).join("\n\n");
}

function buildCalibrationBlock(calibration: any): string {
  if (!calibration?.reference_tasks) return "";
  return calibration.reference_tasks.map((t: any) =>
    `- "${t.task_description}" → ${t.oecd_type}, feasibility ${t.feasibility_score}, ${t.recommendation}. ${t.reasoning}`
  ).join("\n");
}

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
    const { rubric, calibration } = loadOecdData();

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

    const taskTypeSection = rubric
      ? `OECD TASK CLASSIFICATION RUBRIC:\n${buildTaskTypeBlock(rubric)}\n\nBOUNDARY DEFINITIONS:\n${buildBoundaryBlock(rubric)}`
      : `OECD TASK CATEGORIES:
- Routine Cognitive (RC): rule-based, predictable, algorithm-decomposable, verifiable output
- Non-Routine Cognitive (NRC): judgment-dependent, context-sensitive, expert-evaluated
- Interpersonal (IP): relationship-dependent, emotional authenticity, ethical accountability
- Manual/Physical (MP): physical presence, embodied skill, sensorimotor integration`;

    const feasibilitySection = calibration?.feasibility_scale
      ? Object.entries(calibration.feasibility_scale).map(([k, v]) => `${k} = ${v}`).join("\n")
      : `1 = Not Feasible\n2 = Low Feasibility\n3 = Moderate\n4 = High Feasibility\n5 = Highly Suitable`;

    const calibrationSection = calibration
      ? `CALIBRATION EXAMPLES (${calibration.reference_tasks?.length || 0} reference tasks):\n${buildCalibrationBlock(calibration)}`
      : `CALIBRATION EXAMPLES:
- "Grade 200 multiple-choice exams" → RC, feasibility 5, automate.
- "Write personalised feedback on student essays" → NRC, feasibility 3, augment.
- "Conduct a student disciplinary hearing" → IP, feasibility 1, avoid.`;

    const systemPrompt = `${PLATFORM_PREAMBLE}
${userContextBlock}

FRAMEWORK INDEX (for cross-referencing):
${frameworkIndex}

${taskTypeSection}

EVALUATION METHOD:
Step 1 — CLASSIFY THE TASK using the OECD rubric above. Apply boundary definitions to resolve ambiguous cases.
Step 2 — ASSESS FEASIBILITY:
${feasibilitySection}
Step 3 — RECOMMEND: "augment" (AI assists human), "automate" (AI leads with oversight), or "avoid" (AI should not be used).
Step 4 — IDENTIFY SAFEGUARDS tailored to the user's region:
- UK → UK GDPR, DPA 2018, DfE guidance on AI in education
- EU → EU AI Act risk categories, GDPR
- US → FERPA, NIST AI RMF, state-level regulations
- International → UNESCO Guidance principles

${calibrationSection}

RULES:
1. Be realistic about AI capabilities. Distinguish between what AI *can* do technically and what it *should* do in an educational context.
2. Always consider student welfare, academic integrity, and data protection.
3. For the "implementation" field, provide concrete, step-by-step guidance — not vague suggestions.
4. If the task involves student-facing AI, always include safeguards around transparency and appeal mechanisms.
5. Reference the OECD category and boundary definitions in your reasoning.
6. When a task spans multiple categories, classify by the primary objective and note the secondary component.`;

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
