import { HttpRequest, HttpResponseInit } from "@azure/functions";

const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || "https://purple-hill-0a1de9703.1.azurestaticapps.net,http://localhost:5173").split(",");

function getAllowedOrigin(req: HttpRequest): string {
  const origin = req.headers.get("origin") || "";
  if (ALLOWED_ORIGINS.includes(origin)) return origin;
  return ALLOWED_ORIGINS[0];
}

export function corsHeaders(req?: HttpRequest): Record<string, string> {
  return {
    "Access-Control-Allow-Origin": req ? getAllowedOrigin(req) : ALLOWED_ORIGINS[0],
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Signature, X-Timestamp",
    "Vary": "Origin",
  };
}

export function handleCors(req: HttpRequest): HttpResponseInit | null {
  if (req.method === "OPTIONS") {
    return { status: 204, headers: corsHeaders(req) };
  }
  return null;
}
