/**
 * Generic Dimension × Level chart for flat frameworks.
 *
 * Shows each dimension as a row with a segmented progress bar
 * colored by the assigned maturity/competency level.
 * Works with any level count (3, 4, 5+).
 */

interface DimensionResult {
  dimension_id: string;
  dimension_name: string;
  assigned_level: string;
  assigned_level_order: number;
  confidence: "high" | "medium" | "low";
  answer_count: number;
  answer_distribution: Record<string, number>;
}

interface LevelDef {
  name: string;
  order: number;
}

interface DimensionLevelChartProps {
  results: DimensionResult[];
  levels: LevelDef[];
  showConfidence?: boolean;
}

// Color ramp that works for 3–5+ levels
// Index 0 = lowest level, last = highest
const COLOR_RAMP = [
  { fill: "bg-red-400 dark:bg-red-500", bg: "bg-red-100 dark:bg-red-950/40", text: "text-red-700 dark:text-red-300" },
  { fill: "bg-amber-400 dark:bg-amber-500", bg: "bg-amber-100 dark:bg-amber-950/40", text: "text-amber-700 dark:text-amber-300" },
  { fill: "bg-yellow-400 dark:bg-yellow-500", bg: "bg-yellow-100 dark:bg-yellow-950/40", text: "text-yellow-700 dark:text-yellow-300" },
  { fill: "bg-emerald-400 dark:bg-emerald-500", bg: "bg-emerald-100 dark:bg-emerald-950/40", text: "text-emerald-700 dark:text-emerald-300" },
  { fill: "bg-blue-400 dark:bg-blue-500", bg: "bg-blue-100 dark:bg-blue-950/40", text: "text-blue-700 dark:text-blue-300" },
];

function pickColors(levelCount: number) {
  // Distribute colors evenly across the ramp
  if (levelCount <= 1) return [COLOR_RAMP[3]];
  if (levelCount === 2) return [COLOR_RAMP[0], COLOR_RAMP[4]];
  if (levelCount === 3) return [COLOR_RAMP[0], COLOR_RAMP[3], COLOR_RAMP[4]];
  if (levelCount === 4) return [COLOR_RAMP[0], COLOR_RAMP[1], COLOR_RAMP[3], COLOR_RAMP[4]];
  // 5+: use all
  return COLOR_RAMP.slice(0, levelCount);
}

function confidenceDots(confidence: "high" | "medium" | "low"): string {
  switch (confidence) {
    case "high": return "●●●";
    case "medium": return "●●○";
    case "low": return "●○○";
  }
}

function confidenceLabel(confidence: "high" | "medium" | "low"): string {
  return confidence.charAt(0).toUpperCase() + confidence.slice(1);
}

export default function DimensionLevelChart({
  results,
  levels,
  showConfidence = false,
}: DimensionLevelChartProps) {
  const sortedLevels = [...levels].sort((a, b) => a.order - b.order);
  const colors = pickColors(sortedLevels.length);
  const maxOrder = sortedLevels.length > 0 ? sortedLevels[sortedLevels.length - 1].order : 1;
  const minOrder = sortedLevels.length > 0 ? sortedLevels[0].order : 1;

  return (
    <div className="mb-8">
      <h3 className="text-lg font-semibold text-foreground mb-3">Assessment Results by Dimension</h3>

      {/* Level scale header */}
      <div className="flex items-center gap-1 mb-4 ml-[140px] sm:ml-[200px]">
        {sortedLevels.map((level, i) => (
          <div
            key={level.order}
            className="flex-1 text-center text-[10px] text-muted-foreground font-medium truncate px-0.5"
          >
            {level.name}
          </div>
        ))}
      </div>

      {/* Dimension rows */}
      <div role="list" aria-label="Assessment results by dimension" className="space-y-2">
        {results.map((r) => {
          const levelIdx = r.assigned_level_order - minOrder;
          const color = colors[Math.min(levelIdx, colors.length - 1)] || colors[colors.length - 1];

          return (
            <div
              key={r.dimension_id}
              role="listitem"
              aria-label={`${r.dimension_name}: ${r.assigned_level}${showConfidence ? `, ${r.confidence} confidence` : ""}`}
              className="flex items-center gap-2"
            >
              {/* Dimension name */}
              <div className="w-[140px] sm:w-[200px] shrink-0 text-xs sm:text-sm font-medium text-foreground truncate pr-2" title={r.dimension_name}>
                {r.dimension_name}
              </div>

              {/* Segmented bar */}
              <div className="flex-1 flex gap-0.5 h-8 items-center">
                {sortedLevels.map((level, i) => {
                  const isFilled = r.assigned_level_order >= level.order;
                  const segColor = colors[i] || colors[colors.length - 1];

                  return (
                    <div
                      key={level.order}
                      className={`flex-1 h-full rounded-sm transition-all ${
                        isFilled ? segColor.fill : "bg-muted/40"
                      }`}
                      title={`${level.name}${isFilled ? " (achieved)" : ""}`}
                    />
                  );
                })}
              </div>

              {/* Level label + confidence */}
              <div className="w-[80px] sm:w-[100px] shrink-0 text-right">
                <div className={`text-xs font-semibold ${color.text}`}>
                  {r.assigned_level}
                </div>
                {showConfidence && (
                  <div className="text-[10px] text-muted-foreground" title={`${confidenceLabel(r.confidence)} confidence`}>
                    {confidenceDots(r.confidence)}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-3 mt-4 text-[10px] text-muted-foreground">
        {sortedLevels.map((level, i) => {
          const segColor = colors[i] || colors[colors.length - 1];
          return (
            <div key={level.order} className="flex items-center gap-1">
              <div className={`w-3 h-3 rounded-sm ${segColor.fill}`} />
              <span>{level.name}</span>
            </div>
          );
        })}
        {showConfidence && (
          <>
            <span className="mx-1">|</span>
            <span>●●● High</span>
            <span>●●○ Medium</span>
            <span>●○○ Low confidence</span>
          </>
        )}
      </div>
    </div>
  );
}
