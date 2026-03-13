import { HttpRequest } from "@azure/functions";
import jwt from "jsonwebtoken";
import { queryOne } from "./db";

export interface AuthUser {
  userId: string;
  email: string;
}

export async function validateToken(
  req: HttpRequest
): Promise<AuthUser | null> {
  const authHeader = req.headers.get("authorization");
  if (!authHeader?.startsWith("Bearer ")) return null;

  const token = authHeader.slice(7);
  const secret = process.env.JWT_SECRET;
  if (!secret) throw new Error("JWT_SECRET not configured");

  try {
    const decoded = jwt.verify(token, secret, {
      issuer: "reasonlens",
      audience: "reasonlens-api",
    }) as {
      sub: string;
      email: string;
    };
    return { userId: decoded.sub, email: decoded.email };
  } catch {
    return null;
  }
}

export async function requireAuth(req: HttpRequest): Promise<AuthUser> {
  const user = await validateToken(req);
  if (!user) throw new AuthError("Unauthorized", 401);
  return user;
}

export async function requireRole(
  req: HttpRequest,
  ...roles: string[]
): Promise<AuthUser> {
  const user = await requireAuth(req);
  const result = await queryOne<{ has_role: boolean }>(
    `SELECT EXISTS(
      SELECT 1 FROM user_roles WHERE user_id = $1::uuid AND role = ANY($2::app_role[])
    ) AS has_role`,
    [user.userId, roles]
  );
  if (result?.has_role) return user;

  throw new AuthError("You don't have the required role for this action", 403);
}

export async function hasRole(
  userId: string,
  ...roles: string[]
): Promise<boolean> {
  const result = await queryOne<{ has_role: boolean }>(
    `SELECT EXISTS(
      SELECT 1 FROM user_roles WHERE user_id = $1::uuid AND role = ANY($2::app_role[])
    ) AS has_role`,
    [userId, roles]
  );
  return result?.has_role ?? false;
}

export class AuthError extends Error {
  constructor(
    message: string,
    public statusCode: number
  ) {
    super(message);
    this.name = "AuthError";
  }
}
