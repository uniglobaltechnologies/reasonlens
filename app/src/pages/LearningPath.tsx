import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, Loader2 } from "lucide-react";
import Header from "@/components/Header";
import { apiPost } from "@/lib/api";
import { getFrameworkById } from "@/data/frameworks";

interface Gap {
  dimension: string;
  currentLevel: string;
  hasEvidence: boolean;
}

interface RecommendationAction {
  title: string;
  description: string;
  estimatedTime: string;
}

interface Recommendation {
  dimension: string;
  priority: "high" | "medium" | "low";
  currentLevel: string;
  nextLevel: string;
  actions: RecommendationAction[];
  frameworkIndicators?: string[];
}

interface LearningPathResponse {
  gaps: Gap[];
  recommendations: Recommendation[];
  message?: string;
}

const priorityStyles: Record<string, string> = {
  high: "bg-red-500/10 text-red-600",
  medium: "bg-amber-500/10 text-amber-600",
  low: "bg-green-500/10 text-green-600",
};

export default function LearningPath() {
  const { frameworkId } = useParams<{ frameworkId: string }>();
  const framework = frameworkId ? getFrameworkById(frameworkId) : undefined;

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<LearningPathResponse | null>(null);

  useEffect(() => {
    if (!frameworkId) {
      setLoading(false);
      setError("Missing framework ID.");
      return;
    }

    let cancelled = false;

    async function run() {
      setLoading(true);
      setError(null);
      try {
        const res = await apiPost<LearningPathResponse>("/learning-path-ai", {
          frameworkId,
        });
        if (!cancelled) setData(res);
      } catch (err: any) {
        if (!cancelled) setError(err?.message || "Failed to generate learning path.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void run();
    return () => {
      cancelled = true;
    };
  }, [frameworkId]);

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto px-4 sm:px-6 py-8 max-w-4xl">
        <Link
          to={`/assess/${frameworkId || ""}`}
          className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-8"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Assessment
        </Link>

        <h2 className="text-2xl sm:text-3xl font-bold text-foreground mb-2">
          Learning Path
        </h2>
        <p className="text-muted-foreground mb-8">
          {framework?.name || frameworkId}
        </p>

        {loading && (
          <div className="p-6 rounded-xl border border-border bg-card flex items-center gap-3">
            <Loader2 className="h-5 w-5 animate-spin text-primary" />
            <p className="text-sm text-muted-foreground">
              Generating recommendations...
            </p>
          </div>
        )}

        {!loading && error && (
          <div className="p-6 rounded-xl border border-red-500/30 bg-red-500/5">
            <p className="text-sm text-red-600 mb-4">{error}</p>
            <Link
              to="/auth"
              className="inline-flex px-4 py-2 bg-primary text-primary-foreground text-sm font-medium rounded-lg hover:bg-primary/90"
            >
              Sign In
            </Link>
          </div>
        )}

        {!loading && !error && data?.message && (
          <div className="p-6 rounded-xl border border-border bg-card">
            <p className="text-sm text-muted-foreground">{data.message}</p>
          </div>
        )}

        {!loading && !error && data && !data.message && !data.recommendations?.length && (
          <div className="p-6 rounded-xl border-2 border-dashed border-border text-center">
            <p className="text-muted-foreground mb-4">No recommendations yet. Complete an assessment for this framework first.</p>
            <Link
              to={`/assess/${frameworkId || ""}`}
              className="inline-flex px-4 py-2 bg-primary text-primary-foreground text-sm font-medium rounded-lg hover:bg-primary/90"
            >
              Start Assessment
            </Link>
          </div>
        )}

        {!loading && !error && data?.recommendations?.length ? (
          <div className="space-y-4">
            {data.recommendations.map((rec, i) => (
              <div key={`${rec.dimension}-${i}`} className="p-5 rounded-xl border border-border bg-card">
                <div className="flex items-center justify-between gap-3 mb-3">
                  <h3 className="text-lg font-semibold text-foreground">{rec.dimension}</h3>
                  <span className={`text-xs font-medium px-2 py-1 rounded-full ${priorityStyles[rec.priority] || priorityStyles.medium}`}>
                    {rec.priority} priority
                  </span>
                </div>
                <p className="text-sm text-muted-foreground mb-4">
                  Move from <span className="capitalize font-medium text-foreground">{rec.currentLevel}</span> to{" "}
                  <span className="capitalize font-medium text-foreground">{rec.nextLevel}</span>
                </p>
                <div className="space-y-3">
                  {rec.actions?.map((action, idx) => (
                    <div key={idx} className="p-3 rounded-lg bg-muted/40">
                      <p className="text-sm font-medium text-foreground">{action.title}</p>
                      <p className="text-sm text-muted-foreground mt-1">{action.description}</p>
                      <p className="text-xs text-muted-foreground mt-2">Estimated time: {action.estimatedTime}</p>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}
