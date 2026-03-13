import { HttpRequest, HttpResponseInit } from "@azure/functions";

const IS_LOCAL = process.env.FUNCTIONS_WORKER_RUNTIME && !process.env.WEBSITE_HOSTNAME;

const DEFAULT_ALLOWED_ORIGINS = [
  "https://purple-hill-0a1de9703.1.azurestaticapps.net",
  "https://reasonlens.com",
  "https://www.reasonlens.com",
  ...(IS_LOCAL ? ["http://localhost:5173"] : []),
];

const ALLOWED_ORIGINS = (
  process.env.ALLOWED_ORIGINS || DEFAULT_ALLOWED_ORIGINS.join(",")
)
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

export function getAllowedOrigin(req: HttpRequest): string {
  const origin = req.headers.get("origin") || "";
  if (ALLOWED_ORIGINS.includes("*")) return "*";
  if (ALLOWED_ORIGINS.includes(origin)) return origin;
  return "";
}

export function corsHeaders(req?: HttpRequest): Record<string, string> {
  const fallbackOrigin = ALLOWED_ORIGINS.includes("*")
    ? "*"
    : ALLOWED_ORIGINS[0];

  return {
    "Access-Control-Allow-Origin": req ? getAllowedOrigin(req) : fallbackOrigin,
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Signature, X-Timestamp",
    "Vary": "Origin",
  };
}

export function handleCors(req: HttpRequest): HttpResponseInit | null {
  if (req.method === "OPTIONS") {
    return { status: 200, headers: corsHeaders(req), body: "" };
  }
  return null;
}
