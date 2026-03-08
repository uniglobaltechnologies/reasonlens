import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { query, queryOne, execute } from "../shared/db";
import { requireAuth, AuthError } from "../shared/auth";
import { corsHeaders, handleCors } from "../middleware/cors";

const VALID_TYPES = ["document", "link", "reflection", "video"];
const VALID_VISIBILITY = ["public", "private"];

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  try {
    const user = await requireAuth(req);

    if (req.method === "GET") {
      const items = await query(
        "SELECT * FROM portfolio_items WHERE user_id = $1 ORDER BY created_at DESC LIMIT 200",
        [user.userId]
      );
      return {
        status: 200,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ items }),
      };
    }

    if (req.method === "POST") {
      const body = (await req.json()) as {
        title: string;
        description?: string;
        artifact_type: string;
        file_url?: string;
        visibility?: string;
      };

      if (!body.title?.trim()) {
        return {
          status: 400,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "title is required" }),
        };
      }

      if (!body.artifact_type || !VALID_TYPES.includes(body.artifact_type)) {
        return {
          status: 400,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: `artifact_type must be one of: ${VALID_TYPES.join(", ")}` }),
        };
      }

      const visibility = body.visibility && VALID_VISIBILITY.includes(body.visibility) ? body.visibility : "private";

      const item = await queryOne<{ id: string }>(
        `INSERT INTO portfolio_items (user_id, title, description, artifact_type, file_url, visibility)
         VALUES ($1, $2, $3, $4, $5, $6)
         RETURNING id`,
        [user.userId, body.title.trim(), body.description?.trim() || null, body.artifact_type, body.file_url?.trim() || null, visibility]
      );

      return {
        status: 201,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ id: item?.id }),
      };
    }

    if (req.method === "DELETE") {
      const body = (await req.json()) as { id: string };
      if (!body.id) {
        return {
          status: 400,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "id is required" }),
        };
      }

      await execute(
        "DELETE FROM portfolio_items WHERE id = $1 AND user_id = $2",
        [body.id, user.userId]
      );

      return {
        status: 200,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ deleted: true }),
      };
    }

    return { status: 405, headers: corsHeaders(req), body: "Method not allowed" };
  } catch (err) {
    if (err instanceof AuthError) {
      return { status: err.statusCode, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: err.message }) };
    }
    context.error("portfolio error:", err);
    return { status: 500, headers: { ...corsHeaders(req), "Content-Type": "application/json" }, body: JSON.stringify({ error: "Internal server error" }) };
  }
}

app.http("portfolio", {
  methods: ["GET", "POST", "DELETE", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
