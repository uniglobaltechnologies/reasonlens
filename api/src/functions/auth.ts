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

const SALT_ROUNDS = 10;

function getJwtSecret(): string {
  const secret = process.env.JWT_SECRET;
  if (!secret) throw new Error("JWT_SECRET not configured");
  return secret;
}

function signToken(userId: string, email: string): string {
  return jwt.sign({ sub: userId, email }, getJwtSecret(), { expiresIn: "7d" });
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

      if (!user || !user.password_hash) {
        return {
          status: 401,
          headers: { ...corsHeaders(req), "Content-Type": "application/json" },
          body: JSON.stringify({ error: "Invalid email or password" }),
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

    return {
      status: 400,
      headers: { ...corsHeaders(req), "Content-Type": "application/json" },
      body: JSON.stringify({ error: "Invalid action. Use ?action=signup, ?action=login, or ?action=me" }),
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
