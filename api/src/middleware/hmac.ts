import { HttpRequest } from "@azure/functions";
import { createHmac, timingSafeEqual } from "crypto";

const MAX_TIMESTAMP_DRIFT_SECONDS = 300; // 5 minutes

export function validateHmac(
  req: HttpRequest,
  bodyText: string,
  secret: string
): boolean {
  const signature = req.headers.get("x-signature");
  const timestamp = req.headers.get("x-timestamp");

  if (!secret) throw new Error("PETRI_CALLBACK_SECRET not configured");
  if (!signature || !timestamp) return false;

  // Replay protection
  const ts = parseInt(timestamp, 10);
  if (isNaN(ts)) return false;
  const now = Math.floor(Date.now() / 1000);
  if (Math.abs(now - ts) > MAX_TIMESTAMP_DRIFT_SECONDS) return false;

  // Compute expected signature
  const signaturePayload = `${timestamp}.${bodyText}`;
  const expected = createHmac("sha256", secret)
    .update(signaturePayload)
    .digest("hex");

  // Timing-safe comparison
  try {
    const sigBuf = Buffer.from(signature, "hex");
    const expBuf = Buffer.from(expected, "hex");
    if (sigBuf.length !== expBuf.length) return false;
    return timingSafeEqual(sigBuf, expBuf);
  } catch {
    return false;
  }
}
