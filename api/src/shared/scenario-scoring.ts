// Phase 1 deterministic scoring for scenario-based (SJT) assessments.
// Supports the original conservative minimum rule and a boundary-aware
// institutional scorer for the THE Digital Maturity Index.

import { getFrameworkData } from "./framework-context";
import { normalizeTheBoundary } from "./maturity-the";

export interface ScenarioAnswer {
  scenario_id: string;
  dimension_id: string;
  dimension_name: string;
  mapped_level: string;
  level_order: number;
  target_boundary?: string;
}

export interface DimensionResult {
  dimension_id: string;
  dimension_name: string;
  assigned_level: string;
  assigned_level_order: number;
  confidence: "high" | "medium" | "low";
  answer_count: number;
  answer_distribution: Record<string, number>;
}

export interface ScoreSessionOptions {
  frameworkId?: string;
  calibration?: Map<string, unknown>;
}

type Confidence = DimensionResult["confidence"];
type BoundaryStatus = "pass" | "partial" | "fail" | "below" | "missing";

const THE_LEVELS: Record<number, string> = {
  1: "Incidental",
  2: "Intentional",
  3: "Integrated",
  4: "Optimised",
};

const THE_BOUNDARY_SEQUENCE = [
  { id: "incidental-intentional", lowerOrder: 1, upperOrder: 2 },
  { id: "intentional-integrated", lowerOrder: 2, upperOrder: 3 },
  { id: "integrated-optimised", lowerOrder: 3, upperOrder: 4 },
] as const;

/**
 * Score a completed scenario session.
 *
 * Default behaviour:
 * - All answers agree on same level -> that level, confidence = high
 * - Answers disagree by 1 adjacent level -> lower level, confidence = medium
 * - Answers disagree by 2+ levels -> lower level, confidence = low
 *
 * THE Digital Maturity Index behaviour:
 * - Score each adjacent maturity boundary separately
 * - Advance only across boundaries that are clearly passed
 * - Stop at the first partial or failed boundary
 * - Downgrade confidence for mixed evidence, below-boundary responses, or
 *   contradictions where a later boundary is "passed" before an earlier one
 */
export function scoreSession(
  answers: ScenarioAnswer[],
  optionsOrCalibration?: ScoreSessionOptions | Map<string, unknown>
): DimensionResult[] {
  const frameworkId =
    optionsOrCalibration instanceof Map
      ? undefined
      : optionsOrCalibration?.frameworkId;

  const byDimension = new Map<string, ScenarioAnswer[]>();
  for (const answer of answers) {
    const key = answer.dimension_id;
    if (!byDimension.has(key)) byDimension.set(key, []);
    byDimension.get(key)!.push(answer);
  }

  const results: DimensionResult[] = [];

  for (const [dimensionId, dimAnswers] of byDimension) {
    results.push(
      frameworkId === "maturity-the"
        ? scoreTheDimension(dimensionId, dimAnswers)
        : scoreByMinimumAgreement(dimensionId, dimAnswers)
    );
  }

  sortResultsByFrameworkOrder(results, frameworkId);
  return results;
}

function scoreByMinimumAgreement(
  dimensionId: string,
  dimAnswers: ScenarioAnswer[]
): DimensionResult {
  const distribution = buildDistribution(dimAnswers);
  const orders = dimAnswers.map((a) => a.level_order);
  const minOrder = Math.min(...orders);
  const maxOrder = Math.max(...orders);
  const spread = maxOrder - minOrder;
  const lowestAnswer = dimAnswers.find((a) => a.level_order === minOrder)!;

  let confidence: Confidence;
  if (spread === 0) {
    confidence = "high";
  } else if (spread === 1) {
    confidence = "medium";
  } else {
    confidence = "low";
  }

  return {
    dimension_id: dimensionId,
    dimension_name: dimAnswers[0].dimension_name,
    assigned_level: lowestAnswer.mapped_level,
    assigned_level_order: minOrder,
    confidence,
    answer_count: dimAnswers.length,
    answer_distribution: distribution,
  };
}

function scoreTheDimension(
  dimensionId: string,
  dimAnswers: ScenarioAnswer[]
): DimensionResult {
  const distribution = buildDistribution(dimAnswers);
  const byBoundary = new Map<string, ScenarioAnswer[]>();

  for (const answer of dimAnswers) {
    const boundary = normalizeTheBoundary(answer.target_boundary ?? "");
    if (!byBoundary.has(boundary)) byBoundary.set(boundary, []);
    byBoundary.get(boundary)!.push(answer);
  }

  const statuses = THE_BOUNDARY_SEQUENCE.map((boundary) => {
    const answers = byBoundary.get(boundary.id) ?? [];
    return {
      boundary,
      status: evaluateBoundary(answers, boundary.lowerOrder, boundary.upperOrder),
    };
  });

  let assignedOrder = 1;
  let confidence: Confidence = "high";
  let blocked = false;

  for (const { boundary, status } of statuses) {
    if (blocked) break;

    switch (status) {
      case "pass":
        assignedOrder = boundary.upperOrder;
        break;
      case "partial":
        confidence = degradeConfidence(confidence);
        blocked = true;
        break;
      case "fail":
        blocked = true;
        break;
      case "below":
      case "missing":
        confidence = "low";
        blocked = true;
        break;
    }
  }

  if (hasBoundaryContradiction(statuses)) {
    confidence = "low";
  }

  if (statuses.some(({ status }) => status === "missing")) {
    confidence = "low";
  }

  return {
    dimension_id: dimensionId,
    dimension_name: dimAnswers[0].dimension_name,
    assigned_level: THE_LEVELS[assignedOrder] ?? dimAnswers[0].mapped_level,
    assigned_level_order: assignedOrder,
    confidence,
    answer_count: dimAnswers.length,
    answer_distribution: distribution,
  };
}

function buildDistribution(dimAnswers: ScenarioAnswer[]): Record<string, number> {
  const distribution: Record<string, number> = {};
  for (const answer of dimAnswers) {
    distribution[answer.mapped_level] = (distribution[answer.mapped_level] || 0) + 1;
  }
  return distribution;
}

function evaluateBoundary(
  answers: ScenarioAnswer[],
  lowerOrder: number,
  upperOrder: number
): BoundaryStatus {
  if (answers.length === 0) return "missing";

  const passed = answers.filter((a) => a.level_order >= upperOrder).length;
  const below = answers.filter((a) => a.level_order < lowerOrder).length;

  if (passed === answers.length) return "pass";
  if (below > 0) return "below";
  if (passed > 0) return "partial";
  return "fail";
}

function hasBoundaryContradiction(
  statuses: Array<{ status: BoundaryStatus }>
): boolean {
  for (let i = 0; i < statuses.length; i += 1) {
    if (statuses[i].status === "pass") continue;

    for (let j = i + 1; j < statuses.length; j += 1) {
      if (statuses[j].status === "pass") {
        return true;
      }
    }
  }

  return false;
}

function degradeConfidence(confidence: Confidence): Confidence {
  if (confidence === "high") return "medium";
  return "low";
}

function sortResultsByFrameworkOrder(
  results: DimensionResult[],
  frameworkId?: string
): void {
  if (!frameworkId) {
    results.sort((a, b) => a.dimension_id.localeCompare(b.dimension_id));
    return;
  }

  const framework = getFrameworkData().find((item) => item.id === frameworkId);
  if (!framework) {
    results.sort((a, b) => a.dimension_id.localeCompare(b.dimension_id));
    return;
  }

  const orderMap = new Map(
    framework.dimensions
      .filter((dimension) => !dimension.parentDimensionId)
      .map((dimension) => [dimension.id, dimension.order])
  );

  results.sort((a, b) => {
    const orderA = orderMap.get(a.dimension_id) ?? Number.MAX_SAFE_INTEGER;
    const orderB = orderMap.get(b.dimension_id) ?? Number.MAX_SAFE_INTEGER;
    if (orderA !== orderB) return orderA - orderB;
    return a.dimension_id.localeCompare(b.dimension_id);
  });
}
