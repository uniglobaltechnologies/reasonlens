export interface TheAssessmentContext {
  institution_size?: string | null;
  institution_type?: string | null;
  region?: string | null;
  funding_model?: string | null;
  respondent_role?: string | null;
  respondent_institutional_visibility?: string | null;
  digital_infrastructure_baseline?: string | null;
}

export const THE_REQUIRED_CONTEXT_FIELDS = [
  "institution_size",
  "institution_type",
  "region",
  "funding_model",
  "respondent_role",
  "respondent_institutional_visibility",
  "digital_infrastructure_baseline",
] as const;

export const THE_CONTEXT_FIELD_LABELS: Record<
  (typeof THE_REQUIRED_CONTEXT_FIELDS)[number],
  string
> = {
  institution_size: "institution size",
  institution_type: "institution type",
  region: "region",
  funding_model: "funding model",
  respondent_role: "respondent role",
  respondent_institutional_visibility: "institutional visibility",
  digital_infrastructure_baseline: "digital infrastructure baseline",
};

export const THE_DIMENSION_ID_MAP: Record<string, string> = {
  "the-tl-str": "the-tl-strategy",
  "the-tl-ppl": "the-tl-people",
  "the-tl-tec": "the-tl-technology",
  "the-tl-dat": "the-tl-data",
  "the-tl-uti": "the-tl-utilization",
  "the-re-str": "the-re-strategy",
  "the-re-ppl": "the-re-people",
  "the-re-tec": "the-re-technology",
  "the-re-dat": "the-re-data",
  "the-re-uti": "the-re-utilization",
  "the-ps-str": "the-ps-strategy",
  "the-ps-ppl": "the-ps-people",
  "the-ps-tec": "the-ps-technology",
  "the-ps-dat": "the-ps-data",
  "the-ps-uti": "the-ps-utilization",
  "the-pg-str": "the-pg-strategy",
  "the-pg-ppl": "the-pg-people",
  "the-pg-tec": "the-pg-technology",
  "the-pg-dat": "the-pg-data",
  "the-pg-uti": "the-pg-utilization",
};

export const THE_CANONICAL_DIMENSION_NAMES: Record<string, string> = {
  "the-tl-strategy": "Teaching & Learning: Strategy",
  "the-tl-people": "Teaching & Learning: People & Culture",
  "the-tl-technology": "Teaching & Learning: Technology",
  "the-tl-data": "Teaching & Learning: Data",
  "the-tl-utilization": "Teaching & Learning: Utilisation",
  "the-re-strategy": "Research: Strategy",
  "the-re-people": "Research: People & Culture",
  "the-re-technology": "Research: Technology",
  "the-re-data": "Research: Data",
  "the-re-utilization": "Research: Utilisation",
  "the-ps-strategy": "Professional Services: Strategy",
  "the-ps-people": "Professional Services: People & Culture",
  "the-ps-technology": "Professional Services: Technology",
  "the-ps-data": "Professional Services: Data",
  "the-ps-utilization": "Professional Services: Utilisation",
  "the-pg-strategy": "Planning & Governance: Strategy",
  "the-pg-people": "Planning & Governance: People & Culture",
  "the-pg-technology": "Planning & Governance: Technology",
  "the-pg-data": "Planning & Governance: Data",
  "the-pg-utilization": "Planning & Governance: Utilisation",
};

export function normalizeTheDimensionId(dimensionId: string): string {
  return THE_DIMENSION_ID_MAP[dimensionId] ?? dimensionId;
}

export function getTheDimensionName(dimensionId: string, fallback?: string): string {
  return THE_CANONICAL_DIMENSION_NAMES[normalizeTheDimensionId(dimensionId)] ?? fallback ?? dimensionId;
}

export function normalizeTheBoundary(boundary: string): string {
  return boundary.toLowerCase().trim().replace(/\s+/g, "-");
}

export function listMissingTheContextFields(
  context: Partial<TheAssessmentContext>
): string[] {
  return THE_REQUIRED_CONTEXT_FIELDS.filter((field) => {
    const value = context[field];
    return value === null || value === undefined || value === "";
  }).map((field) => THE_CONTEXT_FIELD_LABELS[field]);
}

export function hasCompleteTheContext(
  context: Partial<TheAssessmentContext>
): boolean {
  return listMissingTheContextFields(context).length === 0;
}

export function getTheScenarioContextScore(
  tags: Record<string, unknown>,
  context: Partial<TheAssessmentContext>
): number {
  let score = 0;

  for (const field of ["institution_size", "institution_type", "region"] as const) {
    const tagValue = typeof tags[field] === "string" ? tags[field] : null;
    const contextValue = context[field];

    if (!tagValue || tagValue === "universal") {
      score += 1;
      continue;
    }

    if (contextValue && contextValue === tagValue) {
      score += 2;
    }
  }

  return score;
}
