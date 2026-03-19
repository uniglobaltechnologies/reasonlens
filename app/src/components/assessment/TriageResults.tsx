/**
 * THE DMI Triage Results — pillar heatmap + recommendation.
 * No scores, no level labels. Drives toward the full scenario assessment.
 */
import { Link } from "react-router-dom";
import { ArrowRight, AlertTriangle, Info } from "lucide-react";

type Signal = "incidental" | "intentional" | "integrated" | "optimised";

interface PillarSignal {
  signal: Signal;
  category: string;
  name: string;
}

interface TriageResultProps {
  result: {
    triage_id: string;
    pillar_signals: Record<string, PillarSignal>;
    perceived_priority: string | null;
    recommendation: {
      pillar: string;
      pillar_name: string;
      reason: string;
      scenario_count: number;
      estimated_time_minutes: number;
    };
    visibility_note: string;
  };
}

const SIGNAL_COLOR: Record<Signal, string> = {
  incidental: "bg-red-500",
  intentional: "bg-amber-500",
  integrated: "bg-emerald-500",
  optimised: "bg-blue-500",
};

const SIGNAL_BG: Record<Signal, string> = {
  incidental: "bg-red-100 dark:bg-red-950/40 border-red-200 dark:border-red-800",
  intentional: "bg-amber-100 dark:bg-amber-950/40 border-amber-200 dark:border-amber-800",
  integrated: "bg-emerald-100 dark:bg-emerald-950/40 border-emerald-200 dark:border-emerald-800",
  optimised: "bg-blue-100 dark:bg-blue-950/40 border-blue-200 dark:border-blue-800",
};

const CATEGORY_LABEL: Record<string, string> = {
  needs_attention: "Needs coordinated attention",
  progress_underway: "Progress underway but not yet embedded",
  functioning_well: "Functioning well across the institution",
  sector_leading: "Sector-leading practice",
};

const PRIORITY_LABELS: Record<string, string> = {
  strategy: "strategy",
  people_culture: "people and culture",
  technology: "technology infrastructure",
  data: "data",
  utilisation: "technology utilisation",
};

const PILLAR_ORDER = ["teaching_learning", "research", "professional_services", "planning_governance"];

export default function TriageResults({ result }: TriageResultProps) {
  const { pillar_signals, recommendation, perceived_priority, visibility_note } = result;

  return (
    <div className="max-w-2xl mx-auto">
      {/* Header */}
      <p className="text-xs text-muted-foreground mb-6 flex items-center gap-1.5">
        <Info className="h-3.5 w-3.5 shrink-0" />
        {visibility_note}
      </p>

      {/* Pillar Heatmap */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-8">
        {PILLAR_ORDER.map((key) => {
          const pillar = pillar_signals[key];
          if (!pillar) return null;
          return (
            <div
              key={key}
              className={`rounded-xl border p-4 text-center ${SIGNAL_BG[pillar.signal]}`}
            >
              <div className={`w-3 h-3 rounded-full mx-auto mb-2 ${SIGNAL_COLOR[pillar.signal]}`} />
              <p className="text-sm font-semibold text-foreground mb-1">{pillar.name}</p>
              <p className="text-[11px] text-muted-foreground leading-tight">
                {CATEGORY_LABEL[pillar.category] || pillar.category}
              </p>
            </div>
          );
        })}
      </div>

      {/* Recommendation */}
      <div className="rounded-xl border border-primary/30 bg-primary/5 p-5 mb-6">
        <div className="flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-primary shrink-0 mt-0.5" />
          <div>
            <p className="text-foreground leading-relaxed">
              Your responses suggest <strong>{recommendation.pillar_name}</strong> would benefit most from a detailed assessment.
            </p>
            <p className="text-sm text-muted-foreground mt-2">
              The full scenario assessment for {recommendation.pillar_name} takes about {recommendation.estimated_time_minutes} minutes ({recommendation.scenario_count} scenarios) and will give you a per-dimension breakdown with confidence scores across Strategy, People & Culture, Technology, Data, and Utilisation.
            </p>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row gap-3 mt-4">
          <Link
            to={`/assess/scenario/maturity-the?pillar=${recommendation.pillar}`}
            className="flex-1 py-3 rounded-lg bg-primary text-primary-foreground font-medium text-center hover:bg-primary/90 transition-colors inline-flex items-center justify-center gap-2"
          >
            Start {recommendation.pillar_name} Assessment
            <ArrowRight className="h-4 w-4" />
          </Link>
          <Link
            to="/assess/scenario/maturity-the"
            className="flex-1 py-3 rounded-lg border border-border text-foreground font-medium text-center hover:bg-accent transition-colors"
          >
            Assess all pillars (15 min)
          </Link>
        </div>
      </div>

      {/* Perceived priority callout */}
      {perceived_priority && PRIORITY_LABELS[perceived_priority] && (
        <div className="rounded-xl border border-border bg-card p-4 mb-6">
          <p className="text-sm text-foreground">
            You identified <strong>{PRIORITY_LABELS[perceived_priority]}</strong> as your institution's most urgent cross-cutting challenge. The full assessment will measure this across all four pillars, which may help you build the case for investment.
          </p>
        </div>
      )}

      {/* Visibility disclaimer */}
      <p className="text-xs text-muted-foreground leading-relaxed">
        These results reflect one respondent's perspective. For a robust institutional assessment, we recommend multiple respondents from different roles and levels complete the full scenario assessment. Results can be aggregated at the institutional level.
      </p>
    </div>
  );
}
