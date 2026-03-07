import { HttpRequest, HttpResponseInit } from "@azure/functions";

export function corsHeaders(): Record<string, string> {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Allow-Headers":
      "Content-Type, Authorization, X-Signature, X-Timestamp",
  };
}

export function handleCors(req: HttpRequest): HttpResponseInit | null {
  if (req.method === "OPTIONS") {
    return { status: 204, headers: corsHeaders() };
  }
  return null;
}
