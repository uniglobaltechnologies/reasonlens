import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { query, execute } from "../shared/db";
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
      const frameworkId = req.query.get("framework_id");
      let rows;
      if (frameworkId) {
        rows = await query(
          "SELECT * FROM assessment_results WHERE user_id = $1 AND framework_id = $2 ORDER BY completed_at DESC",
          [user.userId, frameworkId]
        );
      } else {
        rows = await query(
          "SELECT * FROM assessment_results WHERE user_id = $1 ORDER BY completed_at DESC LIMIT 200",
          [user.userId]
        );
      }
      return {
        status: 200,
        headers: { ...corsHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({ results: rows }),
      };
    }

    if (req.method === "POST") {
      const body = (await req.json()) as {
        results: Array<{
          framework_id: string;
          framework_name: string;
          question_id: string;
          dimension: string;
          selected_level: string;
        }>;
      };

      if (!body.results?.length) {
        return {
          status: 400,
          headers: { ...corsHeaders(), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "results array required" }),
        };
      }

      for (const r of body.results) {
        await execute(
          `INSERT INTO assessment_results (user_id, framework_id, framework_name, question_id, dimension, selected_level)
           VALUES ($1, $2, $3, $4, $5, $6)`,
          [user.userId, r.framework_id, r.framework_name, r.question_id, r.dimension, r.selected_level]
        );
      }

      // Update framework progress
      const fw = body.results[0];
      const totalQuestions = body.results.length;
      const progress = 100;
      await execute(
        `INSERT INTO framework_progress (user_id, framework_id, framework_name, progress, completed_items, total_items, last_activity)
         VALUES ($1, $2, $3, $4, $5, $6, now())
         ON CONFLICT (user_id, framework_id) DO UPDATE SET
           progress = $4, completed_items = $5, total_items = $6, last_activity = now(), updated_at = now()`,
        [user.userId, fw.framework_id, fw.framework_name, progress, totalQuestions, totalQuestions]
      );

      return {
        status: 200,
        headers: { ...corsHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({ saved: body.results.length }),
      };
    }

    return { status: 405, headers: corsHeaders(), body: "Method not allowed" };
  } catch (err) {
    if (err instanceof AuthError) {
      return { status: err.statusCode, headers: { ...corsHeaders(), "Content-Type": "application/json" }, body: JSON.stringify({ error: err.message }) };
    }
    context.error("assessments error:", err);
    return { status: 500, headers: { ...corsHeaders(), "Content-Type": "application/json" }, body: JSON.stringify({ error: "Internal server error" }) };
  }
}

app.http("assessments", {
  methods: ["GET", "POST", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
