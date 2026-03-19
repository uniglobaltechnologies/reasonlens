// Framework schema validation utilities.
// Extracts valid level IDs and dimension IDs from framework-context.ts
// for server-side validation of assessment submissions.

import { getFrameworkData } from "./framework-context";

/**
 * Returns the set of valid level IDs for a given framework and dimension.
 * Returns null if the framework or dimension is not found.
 */
export function getValidLevels(
  frameworkId: string,
  dimensionId: string
): Set<string> | null {
  const frameworks = getFrameworkData();
  const fw = frameworks.find((f) => f.id === frameworkId);
  if (!fw) return null;

  const dim = fw.dimensions.find((d) => d.id === dimensionId);
  if (!dim) return null;

  return new Set(dim.levels.map((l) => l.id));
}

/**
 * Returns all valid level identifiers across all dimensions for a framework.
 * Includes both level IDs (e.g., "the-tl-strategy-incidental") and
 * lowercase level names (e.g., "incidental") since the frontend self-assessment
 * sends bare level names while scenario assessment uses full IDs.
 */
export function getValidLevelsForFramework(
  frameworkId: string
): Map<string, Set<string>> | null {
  const frameworks = getFrameworkData();
  const fw = frameworks.find((f) => f.id === frameworkId);
  if (!fw) return null;

  const result = new Map<string, Set<string>>();
  for (const dim of fw.dimensions) {
    const validSet = new Set<string>();
    for (const l of dim.levels) {
      validSet.add(l.id);                    // full ID: "the-tl-strategy-incidental"
      validSet.add(l.name.toLowerCase());     // bare name: "incidental"
    }
    result.set(dim.id, validSet);
  }
  return result;
}

/**
 * Returns all valid dimension IDs for a framework.
 */
export function getValidDimensions(frameworkId: string): Set<string> | null {
  const frameworks = getFrameworkData();
  const fw = frameworks.find((f) => f.id === frameworkId);
  if (!fw) return null;
  return new Set(fw.dimensions.map((d) => d.id));
}

/**
 * Validates an array of assessment results against framework schemas.
 * Returns an array of error messages (empty if all valid).
 */
export function validateAssessmentResults(
  results: Array<{
    framework_id: string;
    dimension: string;
    selected_level: string;
  }>
): string[] {
  const errors: string[] = [];
  const frameworkCache = new Map<string, Map<string, Set<string>> | null>();

  for (const r of results) {
    if (!frameworkCache.has(r.framework_id)) {
      frameworkCache.set(r.framework_id, getValidLevelsForFramework(r.framework_id));
    }
    const fwLevels = frameworkCache.get(r.framework_id);

    if (!fwLevels) {
      // Unknown framework — allow but don't validate levels
      continue;
    }

    const dimLevels = fwLevels.get(r.dimension);
    if (!dimLevels) {
      // Unknown dimension — allow but warn
      continue;
    }

    if (!dimLevels.has(r.selected_level)) {
      errors.push(
        `Invalid level "${r.selected_level}" for ${r.framework_id}/${r.dimension}. ` +
        `Valid levels: ${[...dimLevels].join(", ")}`
      );
    }
  }

  return errors;
}
