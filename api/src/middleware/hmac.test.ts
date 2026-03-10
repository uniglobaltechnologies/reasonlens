import { describe, it, expect } from "vitest";
import { createHmac } from "crypto";
import { validateHmac } from "./hmac";

function mockReq(headers: Record<string, string>) {
  return {
    headers: {
      get: (name: string) => headers[name.toLowerCase()] ?? null,
    },
  } as any;
}

function sign(body: string, secret: string, timestamp: number): string {
  return createHmac("sha256", secret)
    .update(`${timestamp}.${body}`)
    .digest("hex");
}

const SECRET = "test-secret-key-1234567890";

describe("validateHmac", () => {
  it("returns true for valid signature and timestamp", () => {
    const body = '{"run_id":"abc"}';
    const ts = Math.floor(Date.now() / 1000);
    const sig = sign(body, SECRET, ts);
    const req = mockReq({ "x-signature": sig, "x-timestamp": String(ts) });
    expect(validateHmac(req, body, SECRET)).toBe(true);
  });

  it("returns false when x-signature header is missing", () => {
    const ts = Math.floor(Date.now() / 1000);
    const req = mockReq({ "x-timestamp": String(ts) });
    expect(validateHmac(req, "body", SECRET)).toBe(false);
  });

  it("returns false when x-timestamp header is missing", () => {
    const body = "body";
    const ts = Math.floor(Date.now() / 1000);
    const sig = sign(body, SECRET, ts);
    const req = mockReq({ "x-signature": sig });
    expect(validateHmac(req, body, SECRET)).toBe(false);
  });

  it("returns false for expired timestamp (>300s drift)", () => {
    const body = "body";
    const ts = Math.floor(Date.now() / 1000) - 301;
    const sig = sign(body, SECRET, ts);
    const req = mockReq({ "x-signature": sig, "x-timestamp": String(ts) });
    expect(validateHmac(req, body, SECRET)).toBe(false);
  });

  it("returns true for boundary timestamp (exactly 300s)", () => {
    const body = "body";
    const ts = Math.floor(Date.now() / 1000) - 300;
    const sig = sign(body, SECRET, ts);
    const req = mockReq({ "x-signature": sig, "x-timestamp": String(ts) });
    expect(validateHmac(req, body, SECRET)).toBe(true);
  });

  it("returns false for tampered body", () => {
    const body = "original";
    const ts = Math.floor(Date.now() / 1000);
    const sig = sign(body, SECRET, ts);
    const req = mockReq({ "x-signature": sig, "x-timestamp": String(ts) });
    expect(validateHmac(req, "tampered", SECRET)).toBe(false);
  });

  it("returns false for tampered signature", () => {
    const body = "body";
    const ts = Math.floor(Date.now() / 1000);
    const req = mockReq({ "x-signature": "deadbeef".repeat(8), "x-timestamp": String(ts) });
    expect(validateHmac(req, body, SECRET)).toBe(false);
  });

  it("throws when secret is empty", () => {
    const body = "body";
    const ts = Math.floor(Date.now() / 1000);
    const req = mockReq({ "x-signature": "abc", "x-timestamp": String(ts) });
    expect(() => validateHmac(req, body, "")).toThrow("PETRI_CALLBACK_SECRET not configured");
  });

  it("returns false for non-hex signature", () => {
    const body = "body";
    const ts = Math.floor(Date.now() / 1000);
    const req = mockReq({ "x-signature": "not-hex-!@#$", "x-timestamp": String(ts) });
    expect(validateHmac(req, body, SECRET)).toBe(false);
  });
});
