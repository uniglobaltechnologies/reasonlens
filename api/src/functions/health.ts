import {
  app,
  HttpRequest,
  HttpResponseInit,
  InvocationContext,
} from "@azure/functions";
import { queryOne } from "../shared/db";

async function handler(
  req: HttpRequest,
  context: InvocationContext
): Promise<HttpResponseInit> {
  const checks: Record<string, { ok: boolean; ms?: number; error?: string }> = {};

  // PostgreSQL
  const dbStart = Date.now();
  try {
    await queryOne("SELECT 1 AS ok");
    checks.database = { ok: true, ms: Date.now() - dbStart };
  } catch (err: any) {
    checks.database = { ok: false, ms: Date.now() - dbStart, error: err.message };
  }

  // PETRI service
  const petriUrl = process.env.PETRI_SERVICE_URL;
  if (petriUrl) {
    const petriStart = Date.now();
    try {
      const baseUrl = new URL(petriUrl).origin;
      const controller = new AbortController();
      setTimeout(() => controller.abort(), 5000);
      const res = await fetch(`${baseUrl}/health`, { signal: controller.signal });
      checks.petri = { ok: res.ok, ms: Date.now() - petriStart };
    } catch (err: any) {
      checks.petri = { ok: false, ms: Date.now() - petriStart, error: err.message };
    }
  }

  const allOk = Object.values(checks).every((c) => c.ok);

  return {
    status: allOk ? 200 : 503,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      status: allOk ? "healthy" : "degraded",
      timestamp: new Date().toISOString(),
      checks,
    }),
  };
}

app.http("health", {
  methods: ["GET"],
  authLevel: "anonymous",
  handler,
});
