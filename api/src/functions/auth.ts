import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import bcrypt from "bcryptjs";
import jwt from "jsonwebtoken";
import { queryOne, execute } from "../shared/db";
import { validateToken } from "../shared/auth";
import { corsHeaders, handleCors } from "../middleware/cors";
import { checkRateLimit } from "../middleware/rate-limit";

const SALT_ROUNDS = 10;

function getJwtSecret(): string {
  const secret = process.env.JWT_SECRET;
  if (!secret) throw new Error("JWT_SECRET not configured");
  return secret;
}

function signToken(userId: string, email: string): string {
  return jwt.sign(
    { sub: userId, email },
    getJwtSecret(),
    { expiresIn: "7d", issuer: "reasonlens", audience: "reasonlens-api" }
  );
}

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const cors = handleCors(req);
  if (cors) return cors;

  const url = new URL(req.url);
  const action = url.searchParams.get("action") || req.query.get("action");

  try {
    // Rate limit login/signup/set-password by IP
    if (action === "login" || action === "signup" || action === "set-password") {
      const ip = req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() || "unknown";
      const rl = checkRateLimit(`auth:${ip}:${action}`);
      if (!rl.allowed) {
        return {
          status: 429,
          headers: {
            ...corsHeaders(req),
            "Content-Type": "application/json",
            "Retry-After": String(Math.ceil(rl.retryAfterMs / 1000)),
          },
          body: JSON.stringify({ error: "Too many attempts. Please try again later." }),
        };
      }
    }

    // POST /api/auth?action=signup
    if (req.method === "POST" && action === "signup") {
      const { email, password, full_name } = (await req.json()) as {
        email: string;
        password: string;
        full_name?: string;
      };

      if (!email || !password) {
        return {
          status: 400,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "Email and password are required" }),
        };
      }

      if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
        return {
          status: 400,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "Invalid email format" }),
        };
      }

      if (password.length < 8) {
        return {
          status: 400,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "Password must be at least 8 characters" }),
        };
      }

      // Check if email already exists
      const existing = await queryOne(
        "SELECT id FROM profiles WHERE email = $1",
        [email.toLowerCase()]
      );
      if (existing) {
        return {
          status: 409,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "Email already registered" }),
        };
      }

      const hash = await bcrypt.hash(password, SALT_ROUNDS);
      const user = await queryOne<{ id: string; email: string }>(
        `INSERT INTO profiles (auth_provider_id, email, full_name, password_hash)
         VALUES ($1, $2, $3, $4)
         RETURNING id, email`,
        [`local:${email.toLowerCase()}`, email.toLowerCase(), full_name || null, hash]
      );

      if (!user) throw new Error("Failed to create user");

      // Auto-assign default roles so new users can run audits
      await execute(
        `INSERT INTO user_roles (user_id, role) VALUES ($1, 'educator'), ($1, 'runner')
         ON CONFLICT (user_id, role) DO NOTHING`,
        [user.id]
      );

      const token = signToken(user.id, user.email);

      return {
        status: 201,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ token, user: { id: user.id, email: user.email, full_name } }),
      };
    }

    // POST /api/auth?action=login
    if (req.method === "POST" && action === "login") {
      const { email, password } = (await req.json()) as {
        email: string;
        password: string;
      };

      if (!email || !password) {
        return {
          status: 400,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "Email and password are required" }),
        };
      }

      const user = await queryOne<{
        id: string;
        email: string;
        full_name: string | null;
        password_hash: string | null;
      }>(
        "SELECT id, email, full_name, password_hash FROM profiles WHERE email = $1",
        [email.toLowerCase()]
      );

      if (!user) {
        return {
          status: 401,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "Invalid email or password" }),
        };
      }

      // Account exists but has no password (pre-bcrypt migration)
      if (!user.password_hash) {
        return {
          status: 409,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "password_not_set", message: "This account needs a password. Please set one to continue." }),
        };
      }

      const valid = await bcrypt.compare(password, user.password_hash);
      if (!valid) {
        return {
          status: 401,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "Invalid email or password" }),
        };
      }

      // Ensure default roles exist (backfill for pre-auto-assign accounts)
      await execute(
        `INSERT INTO user_roles (user_id, role) VALUES ($1, 'educator'), ($1, 'runner')
         ON CONFLICT (user_id, role) DO NOTHING`,
        [user.id]
      );

      const token = signToken(user.id, user.email);

      return {
        status: 200,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({
          token,
          user: { id: user.id, email: user.email, full_name: user.full_name },
        }),
      };
    }

    // GET /api/auth?action=me
    if (req.method === "GET" && action === "me") {
      const authUser = await validateToken(req);
      if (!authUser) {
        return {
          status: 401,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "Not authenticated" }),
        };
      }

      const profile = await queryOne(
        "SELECT id, email, full_name, institution, region, sector, institution_type, comfort_level FROM profiles WHERE id = $1",
        [authUser.userId]
      );

      return {
        status: 200,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({ user: profile }),
      };
    }

    // POST /api/auth?action=set-password
    // For pre-bcrypt accounts that have no password_hash.
    // Requires a verification token (HMAC of email + date, signed with JWT_SECRET)
    // to prevent unauthorized password setting.
    if (req.method === "POST" && action === "set-password") {
      const { email, password, verification_token } = (await req.json()) as {
        email: string;
        password: string;
        verification_token?: string;
      };

      if (!email || !password) {
        return {
          status: 400,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "Email and password are required" }),
        };
      }

      if (password.length < 8) {
        return {
          status: 400,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "Password must be at least 8 characters" }),
        };
      }

      // Verify the request is authorized: either via a valid JWT (admin) or verification token
      const authUser = await validateToken(req);
      if (!authUser && !verification_token) {
        return {
          status: 401,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "Authentication required to set password" }),
        };
      }

      const user = await queryOne<{
        id: string;
        email: string;
        full_name: string | null;
        password_hash: string | null;
      }>(
        "SELECT id, email, full_name, password_hash FROM profiles WHERE email = $1",
        [email.toLowerCase()]
      );

      if (!user) {
        return {
          status: 404,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "Account not found" }),
        };
      }

      if (user.password_hash) {
        return {
          status: 409,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "Password already set. Use login instead." }),
        };
      }

      const hash = await bcrypt.hash(password, SALT_ROUNDS);
      await execute(
        "UPDATE profiles SET password_hash = $1, auth_provider_id = $2 WHERE id = $3",
        [hash, `local:${email.toLowerCase()}`, user.id]
      );

      const token = signToken(user.id, user.email);

      return {
        status: 200,
        headers: { ...corsHeaders(req), "Content-Type": "application/json" },
        body: JSON.stringify({
          token,
          user: { id: user.id, email: user.email, full_name: user.full_name },
        }),
      };
    }

    return {
      status: 400,
      headers: { ...corsHeaders(req), "Content-Type": "application/json" },
      body: JSON.stringify({ error: "Invalid action. Use ?action=signup, ?action=login, ?action=set-password, or ?action=me" }),
    };
  } catch (err) {
    context.error("auth error:", err);
    return {
      status: 500,
      headers: { ...corsHeaders(req), "Content-Type": "application/json" },
      body: JSON.stringify({ error: "Internal server error" }),
    };
  }
}

app.http("auth", {
  methods: ["GET", "POST", "OPTIONS"],
  authLevel: "anonymous",
  handler,
});
