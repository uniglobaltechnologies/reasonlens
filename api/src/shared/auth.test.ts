import { describe, it, expect, vi, beforeEach } from "vitest";
import jwt from "jsonwebtoken";
import { validateToken, AuthError } from "./auth";

const TEST_SECRET = "test-jwt-secret-for-unit-tests";

function mockReq(authHeader?: string) {
  return {
    headers: {
      get: (name: string) => {
        if (name.toLowerCase() === "authorization") return authHeader ?? null;
        return null;
      },
    },
  } as any;
}

describe("validateToken", () => {
  beforeEach(() => {
    vi.stubEnv("JWT_SECRET", TEST_SECRET);
  });

  it("returns AuthUser for valid non-expired JWT", async () => {
    const token = jwt.sign({ sub: "user-123", email: "test@example.com" }, TEST_SECRET, { expiresIn: "1h" });
    const result = await validateToken(mockReq(`Bearer ${token}`));
    expect(result).toEqual({ userId: "user-123", email: "test@example.com" });
  });

  it("returns null for expired JWT", async () => {
    const token = jwt.sign({ sub: "user-123", email: "test@example.com" }, TEST_SECRET, { expiresIn: "-1s" });
    const result = await validateToken(mockReq(`Bearer ${token}`));
    expect(result).toBeNull();
  });

  it("returns null for JWT signed with wrong secret", async () => {
    const token = jwt.sign({ sub: "user-123", email: "test@example.com" }, "wrong-secret");
    const result = await validateToken(mockReq(`Bearer ${token}`));
    expect(result).toBeNull();
  });

  it("returns null for missing Authorization header", async () => {
    const result = await validateToken(mockReq());
    expect(result).toBeNull();
  });

  it("returns null for malformed Bearer header", async () => {
    const result = await validateToken(mockReq("Basic abc123"));
    expect(result).toBeNull();
  });

  it("throws when JWT_SECRET env var is missing", async () => {
    vi.stubEnv("JWT_SECRET", "");
    const token = jwt.sign({ sub: "user-123", email: "test@example.com" }, "any");
    await expect(validateToken(mockReq(`Bearer ${token}`))).rejects.toThrow("JWT_SECRET not configured");
  });
});

describe("AuthError", () => {
  it("has correct statusCode property", () => {
    const err = new AuthError("Unauthorized", 401);
    expect(err.statusCode).toBe(401);
    expect(err.message).toBe("Unauthorized");
    expect(err.name).toBe("AuthError");
  });
});
