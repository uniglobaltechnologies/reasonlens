import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { query, queryOne } from "../shared/db";
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

    const [frameworks, assessmentCount, portfolioCount, badgeCount, recentAudits, policyCount] =
      await Promise.all([
        query(
          "SELECT * FROM framework_progress WHERE user_id = $1 ORDER BY last_activity DESC",
          [user.userId]
        ),
        queryOne<{ count: string }>(
          "SELECT COUNT(DISTINCT framework_id) as count FROM assessment_results WHERE user_id = $1",
          [user.userId]
        ),
        queryOne<{ count: string }>(
          "SELECT COUNT(*) as count FROM portfolio_items WHERE user_id = $1",
          [user.userId]
        ),
        queryOne<{ count: string }>(
          "SELECT COUNT(*) as count FROM user_badges WHERE user_id = $1",
          [user.userId]
        ),
        query(
          `SELECT id, scenario_pack, target_model, status, created_at
           FROM audit_runs WHERE created_by = $1 ORDER BY created_at DESC LIMIT 3`,
          [user.userId]
        ),
        queryOne<{ count: string }>(
          "SELECT COUNT(*) as count FROM policy_drafts WHERE user_id = $1",
          [user.userId]
        ),
      ]);

    return {
      status: 200,
      headers: { ...corsHeaders(req), "Content-Type": "application/json" },
      body: JSON.stringify({
        frameworks,
        assessmentCount: parseInt(assessmentCount?.count ?? "0"),
        portfolioCount: parseInt(portfolioCount?.count ?? "0"),
        badgeCount: parseInt(badgeCount?.count ?? "0"),
        policyCount: parseInt(policyCount?.count ?? "0"),
        recentAudits,
      }),
    };
  } catch (err) {
    if (err instanceof AuthError) {
      return { status: err.statusCode, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: err.message }) };
    }
    context.error("user-progress error:", err);
    return { status: 500, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: "Internal server error" }) };
  }
}

app.http("user-progress", {
  methods: ["GET", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
