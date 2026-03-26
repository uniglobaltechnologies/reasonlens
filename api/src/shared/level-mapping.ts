// Comprehensive level-to-score mapping covering all 22 framework level vocabularies.
// Normalises diverse level names to a 1–5 numeric scale for cross-framework comparison.

const LEVEL_PATTERNS: Array<{ score: number; patterns: string[] }> = [
  {
    score: 1,
    patterns: [
      "emerging", "initial", "exploring", "foundational", "basic", "novice",
      "awareness", "discovery", "acquire", "approaching", "incidental",
      "low ai readiness", "level-1", "level 1",
      // Composite levels that span low range
      "emerging to established",
    ],
  },
  {
    score: 2,
    patterns: [
      "developing", "experimenting", "intermediate", "deepen",
      "understanding", "intentional", "exploration",
      "medium ai readiness", "variable ai readiness",
      "level-2", "level 2",
      // BDC composite levels
      "approaching and understanding", "experimenting and exploring",
      // THE composite
      "established to enhanced",
    ],
  },
  {
    score: 3,
    patterns: [
      "established", "defined", "operational", "practice", "capable",
      "create", "integrated", "mature",
      "level-3", "level 3",
      // THE composite
      "enhanced to mature",
    ],
  },
  {
    score: 4,
    patterns: [
      "advanced", "embedded", "highly advanced", "leading", "mastery",
      "expert", "managed", "strategic",
      "high ai readiness",
      "level-4", "level 4",
    ],
  },
  {
    score: 5,
    patterns: [
      "optimising", "optimised", "optimised/transformed", "transformed",
      "proficient",
      "level-5", "level 5",
    ],
  },
];

/**
 * Convert a framework level name/ID to a numeric score (1–5).
 * Handles all 22 framework vocabularies:
 * - UNESCO: Acquire/Deepen/Create (1/2/3)
 * - BDC: Developing/Capable/Proficient (2/3/5) — corrected 3-level individual model
 * - THE: Incidental/Intentional/Integrated/Optimised (1/2/3/5)
 * - JISC AI: Exploring/Developing/Defined/Managed/Optimising (1/2/3/4/5)
 * - DigComp: Basic/Intermediate/Advanced/Highly Advanced (1/2/4/4)
 * - DEC/AILit: Novice→Expert, Awareness→Mastery
 * - OECD: Low/Medium/High/Variable AI Readiness (1/2/4/2)
 * - ISTE: Foundational/Intermediate/Proficient/Advanced (1/2/3/4)
 *
 * Returns 2 (Developing) as default for unrecognised values.
 */
// ISTE binary levels — exact match only to avoid substring collisions (e.g. "implemented" contains "met")
const EXACT_LEVEL_MAP: Record<string, number> = {
  "met": 5,
  "not met": 1,
  "not_met": 1,
};

export function levelToScore(level: string): number {
  if (!level) return 2;
  const normalised = level.toLowerCase().trim();

  // Exact match first (ISTE binary, etc.)
  if (normalised in EXACT_LEVEL_MAP) return EXACT_LEVEL_MAP[normalised];

  // Try pure numeric
  const num = parseInt(normalised, 10);
  if (num >= 1 && num <= 5) return num;

  // Match against known patterns
  for (const { score, patterns } of LEVEL_PATTERNS) {
    for (const pattern of patterns) {
      if (normalised === pattern || normalised.includes(pattern)) {
        return score;
      }
    }
  }

  return 2; // Default: treat unknown as "Developing"
}

/**
 * Convert a numeric score (1–5) to a human-readable label.
 */
export function scoreToLabel(score: number): string {
  switch (score) {
    case 1: return "Emerging";
    case 2: return "Developing";
    case 3: return "Established";
    case 4: return "Advanced";
    case 5: return "Optimised";
    default: return "Unknown";
  }
}
