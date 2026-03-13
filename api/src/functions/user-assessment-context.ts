import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { query, queryOne, execute } from "../shared/db";
import { requireAuth, AuthError } from "../shared/auth";
import { corsHeaders, handleCors } from "../middleware/cors";

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  try {
    const user = await requireAuth(req);

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
        institution_type?: string;
        institution_level?: string;
        region?: string;
        current_ai_tools?: string[];
        primary_frustration?: string;
        years_of_experience?: string;
        management_responsibility?: string;
      };

      await execute(
        `INSERT INTO user_assessment_context
           (user_id, subject_area, institution_type, institution_level, region, current_ai_tools, primary_frustration, years_of_experience, management_responsibility)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
         ON CONFLICT (user_id) DO UPDATE SET
           subject_area = COALESCE($2, user_assessment_context.subject_area),
           institution_type = COALESCE($3, user_assessment_context.institution_type),
           institution_level = COALESCE($4, user_assessment_context.institution_level),
           region = COALESCE($5, user_assessment_context.region),
           current_ai_tools = COALESCE($6, user_assessment_context.current_ai_tools),
           primary_frustration = COALESCE($7, user_assessment_context.primary_frustration),
           years_of_experience = COALESCE($8, user_assessment_context.years_of_experience),
           management_responsibility = COALESCE($9, user_assessment_context.management_responsibility),
           updated_at = now()`,
        [
          user.userId,
          body.subject_area ?? null,
          body.institution_type ?? null,
          body.institution_level ?? null,
          body.region ?? null,
          body.current_ai_tools ?? null,
          body.primary_frustration ?? null,
          body.years_of_experience ?? null,
          body.management_responsibility ?? null,
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
