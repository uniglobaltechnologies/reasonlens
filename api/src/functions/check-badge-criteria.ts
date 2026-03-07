import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { query, queryOne, execute } from "../shared/db";
import { requireAuth, AuthError } from "../shared/auth";
import { corsHeaders, handleCors } from "../middleware/cors";

interface Badge {
  id: string;
  name: string;
  description: string;
  icon: string;
  category: string;
  criteria: { type: string; count?: number; days?: number };
  points: number;
}

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  try {
    const user = await requireAuth(req);

    const allBadges = await query<Badge>("SELECT * FROM badges");
    const earnedRows = await query<{ badge_id: string }>(
      "SELECT badge_id FROM user_badges WHERE user_id = $1",
      [user.userId]
    );
    const earnedBadgeIds = new Set(earnedRows.map((b) => b.badge_id));
    const newlyEarned: Badge[] = [];

    for (const badge of allBadges) {
      if (earnedBadgeIds.has(badge.id)) continue;

      const criteria = badge.criteria;
      let earned = false;

      switch (criteria.type) {
        case "assessments_completed": {
          const row = await queryOne<{ count: string }>(
            "SELECT COUNT(*) as count FROM assessment_results WHERE user_id = $1",
            [user.userId]
          );
          earned = parseInt(row?.count ?? "0") >= (criteria.count ?? 0);
          break;
        }

        case "unique_frameworks": {
          const row = await queryOne<{ count: string }>(
            "SELECT COUNT(DISTINCT framework_id) as count FROM framework_progress WHERE user_id = $1",
            [user.userId]
          );
          earned = parseInt(row?.count ?? "0") >= (criteria.count ?? 0);
          break;
        }

        case "create_level": {
          const row = await queryOne<{ count: string }>(
            "SELECT COUNT(*) as count FROM assessment_results WHERE user_id = $1 AND selected_level = 'create'",
            [user.userId]
          );
          earned = parseInt(row?.count ?? "0") >= (criteria.count ?? 0);
          break;
        }

        case "labs_completed": {
          const row = await queryOne<{ total: string }>(
            "SELECT COALESCE(SUM(completed_items), 0) as total FROM framework_progress WHERE user_id = $1",
            [user.userId]
          );
          earned = parseInt(row?.total ?? "0") >= (criteria.count ?? 0);
          break;
        }

        case "portfolio_items": {
          const row = await queryOne<{ count: string }>(
            "SELECT COUNT(*) as count FROM portfolio_items WHERE user_id = $1",
            [user.userId]
          );
          earned = parseInt(row?.count ?? "0") >= (criteria.count ?? 0);
          break;
        }

        case "shares_created": {
          const row = await queryOne<{ count: string }>(
            `SELECT COUNT(DISTINCT ps.portfolio_item_id) as count
             FROM portfolio_shares ps
             JOIN portfolio_items pi ON ps.portfolio_item_id = pi.id
             WHERE pi.user_id = $1`,
            [user.userId]
          );
          earned = parseInt(row?.count ?? "0") >= (criteria.count ?? 0);
          break;
        }

        case "public_portfolio_items": {
          const row = await queryOne<{ count: string }>(
            "SELECT COUNT(*) as count FROM portfolio_items WHERE user_id = $1 AND visibility = 'public'",
            [user.userId]
          );
          earned = parseInt(row?.count ?? "0") >= (criteria.count ?? 0);
          break;
        }

        case "login_streak":
          // Streak tracking not yet implemented
          break;
      }

      if (earned) {
        await execute(
          "INSERT INTO user_badges (user_id, badge_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
          [user.userId, badge.id]
        );
        newlyEarned.push(badge);
      }
    }

    return {
      status: 200,
      headers: { ...corsHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ newlyEarned }),
    };
  } catch (err) {
    if (err instanceof AuthError) {
      return {
        status: err.statusCode,
        headers: { ...corsHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({ error: err.message }),
      };
    }
    context.error("check-badge-criteria error:", err);
    return {
      status: 500,
      headers: { ...corsHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ error: "Internal server error" }),
    };
  }
}

app.http("check-badge-criteria", {
  methods: ["POST", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
