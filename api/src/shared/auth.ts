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
    const decoded = jwt.verify(token, secret) as {
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
  const hasRole = await queryOne<{ has_role: boolean }>(
    `SELECT has_role($1::uuid, $2::app_role) as has_role`,
    [user.userId, roles[0]]
  );

  for (const role of roles) {
    const result = await queryOne<{ has_role: boolean }>(
      `SELECT has_role($1::uuid, $2::app_role) as has_role`,
      [user.userId, role]
    );
    if (result?.has_role) return user;
  }

  throw new AuthError("Forbidden", 403);
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
