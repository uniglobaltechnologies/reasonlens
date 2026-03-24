import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { query, queryOne, execute } from "../shared/db";
import { requireAuth, guestAuth, AuthError } from "../shared/auth";
import { corsHeaders, handleCors } from "../middleware/cors";

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  try {
    // Allow guest access for DMI assessment (frontend sends ?guest=true)
    const isGuest = req.query.get("guest") === "true";
    const user = isGuest ? guestAuth(req) : await requireAuth(req);

    if (req.method === "GET") {
      const row = await queryOne(
        "SELECT * FROM user_assessment_context WHERE user_id = $1",
        [user.userId]
      );
      return {
        status: row ? 200 : 404,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify(row ?? { error: "No context found" }),
      };
    }

    if (req.method === "POST") {
      const body = (await req.json()) as {
        subject_area?: string;
        institution_size?: string;
        institution_type?: string;
        institution_level?: string;
        region?: string;
        funding_model?: string;
        respondent_role?: string;
        respondent_institutional_visibility?: string;
        digital_infrastructure_baseline?: string;
        current_ai_tools?: string[];
        primary_frustration?: string;
        years_of_experience?: string;
        management_responsibility?: string;
        ai_maturity_baseline?: string;
        sector_focus?: string;
        respondent_ai_familiarity?: string;
      };

      await execute(
        `INSERT INTO user_assessment_context
           (user_id, subject_area, institution_size, institution_type, institution_level, region, funding_model, respondent_role, respondent_institutional_visibility, digital_infrastructure_baseline, current_ai_tools, primary_frustration, years_of_experience, management_responsibility, ai_maturity_baseline, sector_focus, respondent_ai_familiarity)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
         ON CONFLICT (user_id) DO UPDATE SET
           subject_area = COALESCE($2, user_assessment_context.subject_area),
           institution_size = COALESCE($3, user_assessment_context.institution_size),
           institution_type = COALESCE($4, user_assessment_context.institution_type),
           institution_level = COALESCE($5, user_assessment_context.institution_level),
           region = COALESCE($6, user_assessment_context.region),
           funding_model = COALESCE($7, user_assessment_context.funding_model),
           respondent_role = COALESCE($8, user_assessment_context.respondent_role),
           respondent_institutional_visibility = COALESCE($9, user_assessment_context.respondent_institutional_visibility),
           digital_infrastructure_baseline = COALESCE($10, user_assessment_context.digital_infrastructure_baseline),
           current_ai_tools = COALESCE($11, user_assessment_context.current_ai_tools),
           primary_frustration = COALESCE($12, user_assessment_context.primary_frustration),
           years_of_experience = COALESCE($13, user_assessment_context.years_of_experience),
           management_responsibility = COALESCE($14, user_assessment_context.management_responsibility),
           ai_maturity_baseline = COALESCE($15, user_assessment_context.ai_maturity_baseline),
           sector_focus = COALESCE($16, user_assessment_context.sector_focus),
           respondent_ai_familiarity = COALESCE($17, user_assessment_context.respondent_ai_familiarity),
           updated_at = now()`,
        [
          user.userId,
          body.subject_area ?? null,
          body.institution_size ?? null,
          body.institution_type ?? null,
          body.institution_level ?? null,
          body.region ?? null,
          body.funding_model ?? null,
          body.respondent_role ?? null,
          body.respondent_institutional_visibility ?? null,
          body.digital_infrastructure_baseline ?? null,
          body.current_ai_tools ?? null,
          body.primary_frustration ?? null,
          body.years_of_experience ?? null,
          body.management_responsibility ?? null,
          body.ai_maturity_baseline ?? null,
          body.sector_focus ?? null,
          body.respondent_ai_familiarity ?? null,
        ]
      );

      const updated = await queryOne(
        "SELECT * FROM user_assessment_context WHERE user_id = $1",
        [user.userId]
      );
      return {
        status: 200,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify(updated),
      };
    }

    return { status: 405, headers: corsHeaders(req), body: "Method not allowed" };
  } catch (err) {
    if (err instanceof AuthError) {
      return { status: err.statusCode, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: err.message }) };
    }
    context.error("user-assessment-context error:", err);
    return { status: 500, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: "Internal server error" }) };
  }
}

app.http("user-assessment-context", {
  methods: ["GET", "POST", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
