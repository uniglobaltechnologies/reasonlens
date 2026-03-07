// ============================================================
// Framework Compatibility Map
// Encodes complementary/overlapping/redundant pairs from spec §4.2
// ============================================================

import type { CompatibilityCategory } from "./framework-types";

export interface CompatibilityPair {
  framework1: string;
  framework2: string;
  category: CompatibilityCategory;
  overlapSeverity?: "low" | "low-medium" | "medium" | "medium-high" | "high";
  overlapAreas?: string[];
  warningText?: string;
}

// ── Complementary pairs (always safe) ──────────

const complementaryPairs: [string, string][] = [
  ["teacher-competency", "bdc-teacher-he"],
  ["teacher-competency", "maturity-jisc-ai"],
  ["teacher-competency", "maturity-the"],
  ["teacher-competency", "maturity-jisc"],
  ["teacher-competency", "ai-capability"],
  ["teacher-competency", "digcomp-3"],
  ["teacher-competency", "iste-educators"],
  ["student-competency", "bdc-individual"],
  ["student-competency", "digcomp-3"],
  ["student-competency", "iste-students"],
  ["ailit-framework", "bdc-individual"],
  ["ailit-framework", "bdc-teacher-he"],
  ["ailit-framework", "bdc-researcher"],
  ["ailit-framework", "bdc-professional-services"],
  ["ailit-framework", "bdc-learning-technology"],
  ["ailit-framework", "bdc-digital-leader"],
  ["ailit-framework", "bdc-educational-developer"],
  ["maturity-jisc-ai", "maturity-the"],
  ["maturity-jisc-ai", "maturity-jisc"],
  ["ai-capability", "maturity-jisc"],
  ["ai-capability", "maturity-the"],
  ["dec-ai-literacy", "bdc-individual"],
  ["dec-ai-literacy", "bdc-teacher-he"],
  ["dec-ai-literacy", "bdc-researcher"],
  ["dec-ai-literacy", "digcomp-3"],
  ["dec-ai-literacy", "iste-students"],
  ["dec-ai-literacy", "iste-educators"],
  ["iste-leaders", "maturity-jisc-ai"],
  ["digcomp-3", "bdc-individual"],
  ["digcomp-3", "bdc-teacher-he"],
  ["digcomp-3", "bdc-researcher"],
];

// ── Overlapping pairs (allow with warning) ─────

const overlappingPairs: CompatibilityPair[] = [
  {
    framework1: "student-competency", framework2: "ailit-framework",
    category: "overlapping", overlapSeverity: "medium",
    overlapAreas: ["AI ethics", "AI understanding", "responsible use"],
    warningText: "These frameworks both address AI ethics and understanding. You may see similar topics in both assessments. We'll highlight where they overlap.",
  },
  {
    framework1: "maturity-jisc-ai", framework2: "ai-capability",
    category: "overlapping", overlapSeverity: "medium-high",
    overlapAreas: ["AI governance", "AI in teaching", "AI research"],
    warningText: "JISC AI focuses on operational readiness; QS emphasizes benchmarking. Using both gives a comprehensive view but some areas will overlap.",
  },
  {
    framework1: "teacher-competency", framework2: "ailit-framework",
    category: "overlapping", overlapSeverity: "low-medium",
    overlapAreas: ["AI ethics", "AI foundations"],
    warningText: "These share some common ground on AI ethics and foundations, but UNESCO Teacher goes deeper into pedagogy while AILit covers AI design and creation.",
  },
  {
    framework1: "maturity-the", framework2: "maturity-jisc",
    category: "overlapping", overlapSeverity: "medium",
    overlapAreas: ["digital strategy", "people & culture", "technology"],
    warningText: "Both frameworks cover digital transformation. THE is more comprehensive (80 indicators vs 30). Consider whether you need both or whether THE alone covers your needs.",
  },
  {
    framework1: "teacher-competency", framework2: "dec-ai-literacy",
    category: "overlapping", overlapSeverity: "medium",
    overlapAreas: ["AI ethics", "AI understanding"],
    warningText: "UNESCO goes deeper into AI pedagogy and professional development; DEC adds critical thinking, domain expertise, and human-centricity dimensions.",
  },
  {
    framework1: "student-competency", framework2: "dec-ai-literacy",
    category: "overlapping", overlapSeverity: "medium",
    overlapAreas: ["AI ethics", "AI understanding"],
    warningText: "UNESCO covers AI use, understanding, ethics, and design; DEC adds critical thinking and domain expertise. Using both gives comprehensive coverage with some overlap.",
  },
  {
    framework1: "dec-ai-literacy", framework2: "ailit-framework",
    category: "overlapping", overlapSeverity: "medium",
    overlapAreas: ["AI literacy fundamentals", "ethics", "responsible use"],
    warningText: "Both frameworks address AI literacy fundamentals. DEC is HE-specific with faculty/student tracks; AILit has broader audience and adds a design/creation dimension.",
  },
  {
    framework1: "iste-educators", framework2: "digcomp-3",
    category: "overlapping", overlapSeverity: "medium",
    overlapAreas: ["digital citizenship", "content creation", "problem solving"],
    warningText: "Both are comprehensive digital competence frameworks. ISTE is standards-based (single proficiency target); DigComp has 4 progression levels across 21 competences.",
  },
  {
    framework1: "iste-students", framework2: "digcomp-3",
    category: "overlapping", overlapSeverity: "medium",
    overlapAreas: ["digital citizenship", "creative communication", "information literacy"],
    warningText: "Both address foundational digital competence for learners. ISTE uses 7 standards with indicators; DigComp provides 21 competences across 4 levels.",
  },
  {
    framework1: "iste-leaders", framework2: "maturity-the",
    category: "overlapping", overlapSeverity: "low-medium",
    overlapAreas: ["digital leadership", "vision", "systems"],
    warningText: "ISTE Leaders is a concise leadership standard (5 indicators). THE is a comprehensive maturity model (80 blocks). ISTE Leaders adds value as a quick self-check alongside THE's deeper assessment.",
  },
  {
    framework1: "iste-leaders", framework2: "maturity-jisc",
    category: "overlapping", overlapSeverity: "medium",
    overlapAreas: ["digital strategy", "leadership"],
    warningText: "Both cover digital transformation leadership. JISC DT is more detailed (30 blocks with 5 maturity levels). ISTE Leaders provides a complementary US-oriented perspective.",
  },
];

// ── Redundant pairs (block or strongly discourage) ──

const redundantPairs: CompatibilityPair[] = [
  // BDC role profile pairs
  ...["bdc-individual", "bdc-teacher-he", "bdc-researcher", "bdc-professional-services", "bdc-learning-technology", "bdc-digital-leader", "bdc-educational-developer"]
    .flatMap((a, i, arr) => arr.slice(i + 1).map((b) => ({
      framework1: a,
      framework2: b,
      category: "redundant" as const,
      overlapSeverity: "high" as const,
      overlapAreas: ["all 6 capability areas"],
      warningText: "The JISC BDC framework has role-specific versions. Each covers the same six capability areas but tailored to your role. We recommend picking the one that best fits your primary role.",
    }))),
  // ISTE student + educator
  {
    framework1: "iste-students", framework2: "iste-educators",
    category: "redundant", overlapSeverity: "high" as const,
    overlapAreas: ["digital competence"],
    warningText: "These are designed for different audiences. Choose the one matching your role.",
  },
  // DigComp 3 + BDC (semi-redundant)
  {
    framework1: "digcomp-3", framework2: "bdc-individual",
    category: "redundant", overlapSeverity: "medium" as const,
    overlapAreas: ["general digital competence"],
    warningText: "Both cover general digital competence. DigComp 3 is more comprehensive (84 blocks vs 30). Consider whether you need both.",
  },
];

// ── Compiled map ───────────────────────────────

export const COMPATIBILITY_MAP: CompatibilityPair[] = [
  ...complementaryPairs.map(([f1, f2]) => ({
    framework1: f1,
    framework2: f2,
    category: "complementary" as const,
  })),
  ...overlappingPairs,
  ...redundantPairs,
];

/** Look up compatibility between two frameworks */
export function getCompatibility(id1: string, id2: string): CompatibilityPair | undefined {
  return COMPATIBILITY_MAP.find(
    (p) =>
      (p.framework1 === id1 && p.framework2 === id2) ||
      (p.framework1 === id2 && p.framework2 === id1)
  );
}

/** Get all compatibility entries for a framework */
export function getFrameworkCompatibilities(frameworkId: string): CompatibilityPair[] {
  return COMPATIBILITY_MAP.filter(
    (p) => p.framework1 === frameworkId || p.framework2 === frameworkId
  );
}

/** OECD is always complementary with everything */
export function isOecdCompatible(): true {
  return true;
}
