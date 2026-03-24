import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { queryOne, execute } from "../shared/db";
import { requireAuth, AuthError } from "../shared/auth";
import { corsHeaders, handleCors } from "../middleware/cors";

const GUEST_UUID = "00000000-0000-4000-a000-000000000000";

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  try {
    const user = await requireAuth(req);

    if (req.method === "GET") {
      const sessionId = req.query.get("session_id");
      if (!sessionId) {
        return {
          status: 400,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "session_id required" }),
        };
      }
      const row = await queryOne(
        "SELECT * FROM interpretation_context WHERE session_id = $1",
        [sessionId]
      );
      return {
        status: row ? 200 : 404,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify(row ?? { error: "No context found" }),
      };
    }

    if (req.method === "POST") {
      const body = (await req.json()) as {
        session_id: string;
        trigger_response?: string;
        previous_attempts?: string;
        constraints?: string[];
        constraints_detail?: string;
        success_definition?: string;
        additional_context?: string;
      };

      if (!body.session_id) {
        return {
          status: 400,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "session_id required" }),
        };
      }

      // Verify session exists and is completed
      const session = await queryOne<{ user_id: string; framework_id: string; status: string }>(
        "SELECT user_id, framework_id, status FROM scenario_sessions WHERE id = $1",
        [body.session_id]
      );

      if (!session) {
        return {
          status: 404,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "Session not found" }),
        };
      }

      if (session.framework_id !== "maturity-the") {
        return {
          status: 400,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "Interpretive reports are only available for THE DMI assessments" }),
        };
      }

      if (session.status !== "completed") {
        return {
          status: 400,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "Assessment must be completed before adding context" }),
        };
      }

      // Guest-to-auth session transfer: if session belongs to guest, claim it
      if (session.user_id === GUEST_UUID && user.userId !== GUEST_UUID) {
        await execute(
          "UPDATE scenario_sessions SET user_id = $1 WHERE id = $2 AND user_id = $3",
          [user.userId, body.session_id, GUEST_UUID]
        );
        // Transfer assessment context too
        const guestCtx = await queryOne(
          "SELECT id FROM user_assessment_context WHERE user_id = $1",
          [GUEST_UUID]
        );
        if (guestCtx) {
          await execute(
            `INSERT INTO user_assessment_context (user_id, institution_size, institution_type, region, funding_model, respondent_role, respondent_institutional_visibility, digital_infrastructure_baseline)
             SELECT $1, institution_size, institution_type, region, funding_model, respondent_role, respondent_institutional_visibility, digital_infrastructure_baseline
             FROM user_assessment_context WHERE user_id = $2
             ON CONFLICT (user_id) DO UPDATE SET
               institution_size = COALESCE(EXCLUDED.institution_size, user_assessment_context.institution_size),
               institution_type = COALESCE(EXCLUDED.institution_type, user_assessment_context.institution_type),
               region = COALESCE(EXCLUDED.region, user_assessment_context.region),
               funding_model = COALESCE(EXCLUDED.funding_model, user_assessment_context.funding_model),
               respondent_role = COALESCE(EXCLUDED.respondent_role, user_assessment_context.respondent_role),
               respondent_institutional_visibility = COALESCE(EXCLUDED.respondent_institutional_visibility, user_assessment_context.respondent_institutional_visibility),
               digital_infrastructure_baseline = COALESCE(EXCLUDED.digital_infrastructure_baseline, user_assessment_context.digital_infrastructure_baseline),
               updated_at = now()`,
            [user.userId, GUEST_UUID]
          );
        }
      } else if (session.user_id !== user.userId) {
        return {
          status: 403,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "Session does not belong to you" }),
        };
      }

      // Upsert open-ended responses
      await execute(
        `INSERT INTO interpretation_context
           (session_id, user_id, trigger_response, previous_attempts, constraints, constraints_detail, success_definition, additional_context)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
         ON CONFLICT (session_id) DO UPDATE SET
           trigger_response = $3,
           previous_attempts = $4,
           constraints = $5,
           constraints_detail = $6,
           success_definition = $7,
           additional_context = $8,
           created_at = now()`,
        [
          body.session_id,
          user.userId,
          body.trigger_response ?? null,
          body.previous_attempts ?? null,
          body.constraints ?? [],
          body.constraints_detail ?? null,
          body.success_definition ?? null,
          body.additional_context ?? null,
        ]
      );

      return {
        status: 200,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ success: true }),
      };
    }

    return { status: 405, headers: corsHeaders(req), body: "Method not allowed" };
  } catch (err) {
    if (err instanceof AuthError) {
      return { status: err.statusCode, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: err.message }) };
    }
    context.error("interpretation-context error:", err);
    return { status: 500, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: "Internal server error" }) };
  }
}

app.http("interpretation-context", {
  methods: ["GET", "POST", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
