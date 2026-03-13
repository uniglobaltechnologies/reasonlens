// Phase 1 deterministic scoring for scenario-based (SJT) assessments.
// Groups answers by dimension, determines level by agreement, assigns confidence.

export interface ScenarioAnswer {
  scenario_id: string;
  dimension_id: string;
  dimension_name: string;
  mapped_level: string;
  level_order: number;
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

/**
 * Score a completed scenario session using Phase 1 deterministic algorithm.
 *
 * Per dimension:
 * - All answers agree on same level → that level, confidence = "high"
 * - Answers disagree by 1 adjacent level → lower level, confidence = "medium"
 * - Answers disagree by 2+ levels → lower level, confidence = "low"
 *
 * The function signature accepts an optional calibration parameter to support
 * future Phase 2 IRT-based scoring without changing the interface.
 */
export function scoreSession(
  answers: ScenarioAnswer[],
  _calibration?: Map<string, unknown> // Phase 2: IRT parameters per scenario
): DimensionResult[] {
  // Group answers by dimension
  const byDimension = new Map<string, ScenarioAnswer[]>();
  for (const answer of answers) {
    const key = answer.dimension_id;
    if (!byDimension.has(key)) byDimension.set(key, []);
    byDimension.get(key)!.push(answer);
  }

  const results: DimensionResult[] = [];

  for (const [dimensionId, dimAnswers] of byDimension) {
    // Build distribution
    const distribution: Record<string, number> = {};
    for (const a of dimAnswers) {
      distribution[a.mapped_level] = (distribution[a.mapped_level] || 0) + 1;
    }

    const orders = dimAnswers.map((a) => a.level_order);
    const minOrder = Math.min(...orders);
    const maxOrder = Math.max(...orders);
    const spread = maxOrder - minOrder;

    // Find the level name corresponding to the minimum order
    const lowestAnswer = dimAnswers.find((a) => a.level_order === minOrder)!;

    let confidence: "high" | "medium" | "low";
    if (spread === 0) {
      confidence = "high";
    } else if (spread === 1) {
      confidence = "medium";
    } else {
      confidence = "low";
    }

    results.push({
      dimension_id: dimensionId,
      dimension_name: dimAnswers[0].dimension_name,
      assigned_level: lowestAnswer.mapped_level,
      assigned_level_order: minOrder,
      confidence,
      answer_count: dimAnswers.length,
      answer_distribution: distribution,
    });
  }

  // Sort by dimension_id for consistent output
  results.sort((a, b) => a.dimension_id.localeCompare(b.dimension_id));
  return results;
}
