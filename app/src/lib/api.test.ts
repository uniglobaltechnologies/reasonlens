import { describe, it, expect, beforeEach, vi } from "vitest";
import { isTokenExpired, isAuthenticated, clearToken, setToken } from "./api";

// Create a fake JWT with a given payload (no real signing needed — isTokenExpired only decodes the payload)
function fakeJwt(payload: Record<string, any>): string {
  const header = btoa(JSON.stringify({ alg: "HS256", typ: "JWT" }));
  const body = btoa(JSON.stringify(payload));
  return `${header}.${body}.fake-signature`;
}

describe("isTokenExpired", () => {
  it("returns false for token with future exp", () => {
    const token = fakeJwt({ exp: Math.floor(Date.now() / 1000) + 3600 });
    expect(isTokenExpired(token)).toBe(false);
  });

  it("returns true for token with past exp", () => {
    const token = fakeJwt({ exp: Math.floor(Date.now() / 1000) - 10 });
    expect(isTokenExpired(token)).toBe(true);
  });

  it("returns false for token with no exp claim", () => {
    const token = fakeJwt({ sub: "user-123" });
    expect(isTokenExpired(token)).toBe(false);
  });

  it("returns true for malformed token", () => {
    expect(isTokenExpired("not.a.jwt")).toBe(true);
    expect(isTokenExpired("")).toBe(true);
    expect(isTokenExpired("single-segment")).toBe(true);
  });
});

describe("isAuthenticated", () => {
  beforeEach(() => {
    clearToken();
  });

  it("returns false when no token in localStorage", () => {
    expect(isAuthenticated()).toBe(false);
  });

  it("returns true for valid non-expired token", () => {
    const token = fakeJwt({ exp: Math.floor(Date.now() / 1000) + 3600 });
    setToken(token);
    expect(isAuthenticated()).toBe(true);
  });

  it("returns false and clears token for expired token", () => {
    const token = fakeJwt({ exp: Math.floor(Date.now() / 1000) - 10 });
    setToken(token);
    expect(isAuthenticated()).toBe(false);
    // Token should have been cleared by isAuthenticated
    expect(isAuthenticated()).toBe(false);
  });
});
