import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { query, execute } from "../shared/db";
import { requireAuth, AuthError } from "../shared/auth";
import { encryptValue, decryptValue } from "../shared/crypto";
import { corsHeaders, handleCors } from "../middleware/cors";

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  try {
    const user = await requireAuth(req);
    const secret = process.env.BYOK_ENC_SECRET;
    if (!secret) throw new Error("BYOK_ENC_SECRET not configured");

    if (req.method === "GET") {
      const keys = await query<{
        provider: string;
        key_last4: string;
        updated_at: string;
      }>(
        "SELECT provider, key_last4, updated_at FROM user_api_keys WHERE user_id = $1",
        [user.userId]
      );
      return {
        status: 200,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ keys }),
      };
    }

    if (req.method === "POST") {
      const body = (await req.json()) as {
        provider: string;
        api_key: string;
      };
      if (!body.provider || !body.api_key) {
        return {
          status: 400,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "provider and api_key required" }),
        };
      }

      const encrypted = await encryptValue(body.api_key, secret);
      const last4 = body.api_key.slice(-4);

      await execute(
        `INSERT INTO user_api_keys (user_id, provider, encrypted_key, key_last4)
         VALUES ($1, $2, $3, $4)
         ON CONFLICT (user_id, provider)
         DO UPDATE SET encrypted_key = $3, key_last4 = $4, updated_at = now()`,
        [user.userId, body.provider, encrypted, last4]
      );

      return {
        status: 200,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ success: true, provider: body.provider }),
      };
    }

    if (req.method === "DELETE") {
      const body = (await req.json()) as { provider: string };
      if (!body.provider) {
        return {
          status: 400,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "provider required" }),
        };
      }

      await execute(
        "DELETE FROM user_api_keys WHERE user_id = $1 AND provider = $2",
        [user.userId, body.provider]
      );

      return {
        status: 200,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ success: true }),
      };
    }

    return {
      status: 405,
      headers: corsHeaders(req),
      body: "Method not allowed",
    };
  } catch (err) {
    if (err instanceof AuthError) {
      return {
        status: err.statusCode,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ error: err.message }),
      };
    }
    context.error("user-api-keys error:", err);
    return {
      status: 500,
      headers: { ...corsHeaders(req), "Content-Type": "application/json" },
      body: JSON.stringify({ error: "Internal server error" }),
    };
  }
}

app.http("user-api-keys", {
  methods: ["GET", "POST", "DELETE", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
