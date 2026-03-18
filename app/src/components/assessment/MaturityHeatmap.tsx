/**
 * THE Digital Maturity Index — Maturity Heatmap
 *
 * 4×5 grid: pillars (rows) × cross-cutting dimensions (columns).
 * Cell color encodes maturity level, dots encode confidence.
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

interface MaturityHeatmapProps {
  results: DimensionResult[];
}

const PILLARS = [
  { prefix: "the-tl-", name: "Teaching & Learning", short: "T&L" },
  { prefix: "the-re-", name: "Research", short: "Research" },
  { prefix: "the-ps-", name: "Professional Services", short: "Prof Svcs" },
  { prefix: "the-pg-", name: "Planning & Governance", short: "Plan & Gov" },
];

const DIMENSIONS = [
  { key: "strategy", label: "Strategy", abbr: "S" },
  { key: "people", label: "People & Culture", abbr: "P" },
  { key: "technology", label: "Technology", abbr: "T" },
  { key: "data", label: "Data", abbr: "D" },
  { key: "utilization", label: "Utilisation", abbr: "U" },
];

const LEVEL_ABBR: Record<number, string> = {
  1: "Incid.",
  2: "Intent.",
  3: "Integ.",
  4: "Optim.",
};

const LEVEL_FULL: Record<number, string> = {
  1: "Incidental",
  2: "Intentional",
  3: "Integrated",
  4: "Optimised",
};

const CELL_COLORS: Record<number, string> = {
  1: "bg-red-100 dark:bg-red-950/40 border-red-200 dark:border-red-900/40",
  2: "bg-amber-100 dark:bg-amber-950/40 border-amber-200 dark:border-amber-900/40",
  3: "bg-emerald-100 dark:bg-emerald-950/40 border-emerald-200 dark:border-emerald-900/40",
  4: "bg-blue-100 dark:bg-blue-950/40 border-blue-200 dark:border-blue-900/40",
};

const LEVEL_TEXT: Record<number, string> = {
  1: "text-red-800 dark:text-red-300",
  2: "text-amber-800 dark:text-amber-300",
  3: "text-emerald-800 dark:text-emerald-300",
  4: "text-blue-800 dark:text-blue-300",
};

function confidenceDots(confidence: "high" | "medium" | "low"): string {
  switch (confidence) {
    case "high": return "●●●";
    case "medium": return "●●○";
    case "low": return "●○○";
  }
}

export default function MaturityHeatmap({ results }: MaturityHeatmapProps) {
  const lookup = new Map(results.map((r) => [r.dimension_id, r]));

  return (
    <div className="mb-8">
      <h3 className="text-lg font-semibold text-foreground mb-3">Maturity Heat Map</h3>

      <div className="overflow-x-auto">
        <div role="grid" aria-label="THE DMI maturity heat map" className="min-w-[500px]">
          {/* Column headers */}
          <div className="grid grid-cols-[140px_repeat(5,1fr)] gap-1 mb-1" role="row">
            <div role="columnheader" className="text-xs font-medium text-muted-foreground p-1" />
            {DIMENSIONS.map((dim) => (
              <div
                key={dim.key}
                role="columnheader"
                className="text-xs font-medium text-muted-foreground text-center p-1"
              >
                <span className="hidden sm:inline">{dim.label}</span>
                <span className="sm:hidden">{dim.abbr}</span>
              </div>
            ))}
          </div>

          {/* Data rows */}
          {PILLARS.map((pillar) => (
            <div
              key={pillar.prefix}
              className="grid grid-cols-[140px_repeat(5,1fr)] gap-1 mb-1"
              role="row"
            >
              {/* Row header */}
              <div
                role="rowheader"
                className="text-xs font-medium text-foreground flex items-center pr-2"
              >
                <span className="hidden sm:inline">{pillar.name}</span>
                <span className="sm:hidden">{pillar.short}</span>
              </div>

              {/* Cells */}
              {DIMENSIONS.map((dim) => {
                const id = `${pillar.prefix}${dim.key}`;
                const result = lookup.get(id);

                if (!result) {
                  return (
                    <div
                      key={dim.key}
                      role="gridcell"
                      className="rounded-lg border border-border bg-muted/30 p-2 text-center"
                    >
                      <span className="text-xs text-muted-foreground">—</span>
                    </div>
                  );
                }

                const order = result.assigned_level_order;
                const distStr = Object.entries(result.answer_distribution)
                  .map(([level, count]) => `${level}: ${count}`)
                  .join(", ");

                return (
                  <div
                    key={dim.key}
                    role="gridcell"
                    aria-label={`${pillar.name} ${dim.label}: ${LEVEL_FULL[order]}, ${result.confidence} confidence`}
                    title={`${LEVEL_FULL[order]} (${result.confidence} confidence)\n${distStr}`}
                    className={`rounded-lg border p-2 text-center transition-transform hover:scale-105 cursor-default ${CELL_COLORS[order] || ""}`}
                  >
                    <div className={`text-xs font-semibold leading-tight ${LEVEL_TEXT[order] || ""}`}>
                      {LEVEL_ABBR[order]}
                    </div>
                    <div className="text-[10px] text-muted-foreground mt-0.5 tracking-wider">
                      {confidenceDots(result.confidence)}
                    </div>
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-3 mt-3 text-[10px] text-muted-foreground">
        {([1, 2, 3, 4] as const).map((order) => (
          <div key={order} className="flex items-center gap-1">
            <div className={`w-3 h-3 rounded border ${CELL_COLORS[order]}`} />
            <span>{LEVEL_FULL[order]}</span>
          </div>
        ))}
        <span className="mx-1">|</span>
        <span>●●● High</span>
        <span>●●○ Medium</span>
        <span>●○○ Low confidence</span>
      </div>
    </div>
  );
}
