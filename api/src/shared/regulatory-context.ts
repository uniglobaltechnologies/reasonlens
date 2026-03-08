// Server-side regulatory context loader.
// Loads applicable regulatory provisions based on user region.

import { readFileSync } from "fs";
import { join } from "path";

interface Provision {
  id: string;
  article?: string;
  title: string;
  status?: string;
  full_text?: string;
  education_relevance?: string;
  policy_types?: string[];
}

interface RegionalContext {
  framework_name: string;
  status: string;
  provisions: Provision[];
}

let cachedData: Record<string, RegionalContext> | null = null;

function loadData(): Record<string, RegionalContext> {
  if (cachedData) return cachedData;
  const raw = readFileSync(
    join(__dirname, "regulatory-context.json"),
    "utf-8"
  );
  const parsed = JSON.parse(raw);
  cachedData = {
    eu: parsed.eu,
    uk: parsed.uk,
    us: parsed.us,
    international: parsed.international,
  };
  return cachedData;
}

/**
 * Maps a user's region string to the applicable regulatory regions.
 * Returns an array of region keys to look up.
 */
function resolveRegions(region: string): string[] {
  const r = (region || "").toLowerCase();
  if (r.includes("uk") || r.includes("united kingdom") || r.includes("britain") || r.includes("england") || r.includes("scotland") || r.includes("wales")) {
    return ["uk"];
  }
  if (r.includes("eu") || r.includes("europe") || r.includes("germany") || r.includes("france") || r.includes("spain") || r.includes("italy") || r.includes("netherlands")) {
    return ["eu"];
  }
  if (r.includes("us") || r.includes("united states") || r.includes("america")) {
    return ["us"];
  }
  // For unspecified or international, include international context
  return ["international"];
}

/**
 * Loads applicable regulatory provisions for a given region.
 * Returns formatted provisions suitable for LLM prompt injection.
 */
export function loadRegulatoryContext(region: string): Provision[] {
  const data = loadData();
  const regions = resolveRegions(region);
  const provisions: Provision[] = [];

  for (const key of regions) {
    const ctx = data[key];
    if (!ctx?.provisions) continue;
    for (const p of ctx.provisions) {
      provisions.push(p);
    }
  }

  return provisions;
}

/**
 * Formats regulatory provisions for prompt injection.
 * Limits text length per provision to keep prompt size manageable.
 */
export function formatRegulatoryContext(provisions: Provision[]): string {
  if (provisions.length === 0) return "No specific regulatory provisions loaded for this region.";

  return provisions
    .map((p) => {
      const text = p.full_text
        ? p.full_text.length > 400
          ? p.full_text.substring(0, 400) + "..."
          : p.full_text
        : "";
      return `### ${p.title} (${p.article || p.id})
${text}
Education relevance: ${p.education_relevance || "Not specified"}`;
    })
    .join("\n\n");
}
