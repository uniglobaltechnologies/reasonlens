import { describe, it, expect, vi, beforeEach } from "vitest";

// We need to test with controlled ALLOWED_ORIGINS, so we reset modules between tests
describe("cors", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  function mockReq(origin?: string) {
    return {
      method: "GET",
      headers: {
        get: (name: string) => name.toLowerCase() === "origin" ? (origin ?? null) : null,
      },
    } as any;
  }

  function mockOptionsReq(origin?: string) {
    return {
      method: "OPTIONS",
      headers: {
        get: (name: string) => name.toLowerCase() === "origin" ? (origin ?? null) : null,
      },
    } as any;
  }

  describe("with default origins", () => {
    it("getAllowedOrigin returns matching origin", async () => {
      const { getAllowedOrigin } = await import("./cors");
      const req = mockReq("https://reasonlens.com");
      expect(getAllowedOrigin(req)).toBe("https://reasonlens.com");
    });

    it("getAllowedOrigin returns empty string for unrecognized origin (regression)", async () => {
      const { getAllowedOrigin } = await import("./cors");
      const req = mockReq("https://evil.com");
      expect(getAllowedOrigin(req)).toBe("");
    });

    it("corsHeaders includes Vary header", async () => {
      const { corsHeaders } = await import("./cors");
      const headers = corsHeaders(mockReq("https://reasonlens.com"));
      expect(headers["Vary"]).toBe("Origin");
    });

    it("corsHeaders without request returns first allowed origin", async () => {
      const { corsHeaders } = await import("./cors");
      const headers = corsHeaders();
      expect(headers["Access-Control-Allow-Origin"]).toBe("https://purple-hill-0a1de9703.1.azurestaticapps.net");
    });

    it("handleCors returns 200 for OPTIONS", async () => {
      const { handleCors } = await import("./cors");
      const result = handleCors(mockOptionsReq("https://reasonlens.com"));
      expect(result).not.toBeNull();
      expect(result!.status).toBe(200);
    });

    it("handleCors returns null for non-OPTIONS", async () => {
      const { handleCors } = await import("./cors");
      const result = handleCors(mockReq("https://reasonlens.com"));
      expect(result).toBeNull();
    });
  });

  describe("with wildcard origin", () => {
    it("getAllowedOrigin returns * when wildcard configured", async () => {
      vi.stubEnv("ALLOWED_ORIGINS", "*");
      const { getAllowedOrigin } = await import("./cors");
      const req = mockReq("https://anything.com");
      expect(getAllowedOrigin(req)).toBe("*");
      vi.unstubAllEnvs();
    });
  });
});
