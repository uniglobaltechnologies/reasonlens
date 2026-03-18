import { useState, useEffect } from "react";
import { useParams, Link } from "react-router-dom";
import {
  ArrowLeft,
  CheckCircle2,
  XCircle,
  Loader2,
  Clock,
  ChevronDown,
  ChevronRight,
  FileText,
  AlertTriangle,
} from "lucide-react";
import Header from "@/components/Header";
import { apiGet } from "@/lib/api";

interface Transcript {
  id: string;
  path: string;
  content: string | null;
  judge_scores_json: any;
  flags: string[] | null;
  scenario_id: string | null;
  epoch_number: number | null;
}

interface RunDetail {
  run: {
    id: string;
    scenario_pack: string;
    auditor_model: string;
    target_model: string;
    judge_model: string;
    status: string;
    cost_tokens: number | null;
    cost_currency: number | null;
    error_message: string | null;
    created_at: string;
    completed_at: string | null;
    max_turns: number;
    samples_per_scenario: number | null;
  };
  transcripts: Transcript[];
  posthoc: any[];
  benchmarks: any[];
  report: any | null;
}

const statusConfig: Record<string, { icon: any; color: string; label: string }> = {
  completed: { icon: CheckCircle2, color: "text-green-500", label: "Completed" },
  failed: { icon: XCircle, color: "text-red-500", label: "Failed" },
  running: { icon: Loader2, color: "text-amber-500", label: "Running" },
  queued: { icon: Clock, color: "text-muted-foreground", label: "Queued" },
  stopped: { icon: XCircle, color: "text-muted-foreground", label: "Stopped" },
};

export default function AuditDetail() {
  const { id } = useParams<{ id: string }>();
  const [data, setData] = useState<RunDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedTranscript, setExpandedTranscript] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;

    let cancelled = false;

    const fetchData = () => {
      apiGet<RunDetail>(`/audit-runs?id=${id}`)
        .then((res) => { if (!cancelled) setData(res); })
        .catch((err) => { if (!cancelled) setError(err.message); })
        .finally(() => { if (!cancelled) setLoading(false); });
    };

    fetchData();

    return () => { cancelled = true; };
  }, [id]);

  // Separate polling effect that uses a ref to avoid stale closures
  useEffect(() => {
    const status = data?.run.status;
    if (status !== "running" && status !== "queued") return;

    const interval = setInterval(() => {
      apiGet<RunDetail>(`/audit-runs?id=${id}`)
        .then(setData)
        .catch(() => {}); // Silently retry on next interval
    }, 8000);

    return () => clearInterval(interval);
  }, [id, data?.run.status]);

  if (loading) {
    return (
      <div className="min-h-screen bg-background">
        <Header />
        <div className="flex items-center justify-center py-20">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="min-h-screen bg-background">
        <Header />
        <div className="container mx-auto px-4 py-12 text-center">
          <p className="text-red-500 mb-4">{error || "Run not found"}</p>
          <Link to="/audit/runs" className="text-primary hover:underline">Back to runs</Link>
        </div>
      </div>
    );
  }

  const { run, transcripts, posthoc, report } = data;
  const status = statusConfig[run.status] || statusConfig.queued;
  const StatusIcon = status.icon;

  // Extract scores from transcripts
  const allScores: Record<string, number[]> = {};
  for (const t of transcripts) {
    const scores = t.judge_scores_json?.scores || t.judge_scores_json || {};
    for (const [key, val] of Object.entries(scores)) {
      if (typeof val === "number") {
        if (!allScores[key]) allScores[key] = [];
        allScores[key].push(val);
      }
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto px-4 sm:px-6 py-8 max-w-4xl">
        <Link to="/audit/runs" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-6">
          <ArrowLeft className="h-4 w-4" />All Runs
        </Link>

        {/* Run summary */}
        <div className="p-6 rounded-xl border border-border bg-card mb-6">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h2 className="text-xl font-bold text-foreground mb-1">{run.scenario_pack}</h2>
              <p className="text-sm text-muted-foreground">
                Target: {run.target_model} · Auditor: {run.auditor_model} · Judge: {run.judge_model}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <StatusIcon className={`h-5 w-5 ${status.color} ${run.status === "running" ? "animate-spin" : ""}`} />
              <span className={`text-sm font-medium ${status.color}`}>{status.label}</span>
            </div>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-muted-foreground">Created</p>
              <p className="font-medium text-foreground">{new Date(run.created_at).toLocaleString()}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Max Turns</p>
              <p className="font-medium text-foreground">{run.max_turns}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Tokens</p>
              <p className="font-medium text-foreground">{run.cost_tokens?.toLocaleString() || "—"}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Cost</p>
              <p className="font-medium text-foreground">{run.cost_currency ? `$${run.cost_currency.toFixed(4)}` : "—"}</p>
            </div>
          </div>
          {run.error_message && (
            <div className="mt-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20">
              <div className="flex items-center gap-2 mb-1">
                <AlertTriangle className="h-4 w-4 text-red-500" />
                <span className="text-sm font-medium text-red-600">Error</span>
              </div>
              <p className="text-xs text-red-600/80">{run.error_message}</p>
            </div>
          )}
        </div>

        {/* Scores summary */}
        {Object.keys(allScores).length > 0 && (
          <div className="p-6 rounded-xl border border-border bg-card mb-6">
            <h3 className="text-lg font-semibold text-foreground mb-4">Judge Scores</h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
              {Object.entries(allScores).map(([label, values]) => {
                const avg = values.reduce((a, b) => a + b, 0) / values.length;
                return (
                  <div key={label} className="p-3 rounded-lg bg-muted/50">
                    <p className="text-xs text-muted-foreground capitalize">{label.replace(/_/g, " ")}</p>
                    <p className="text-xl font-bold text-foreground">{avg.toFixed(1)}</p>
                    <p className="text-xs text-muted-foreground">{values.length} sample{values.length > 1 ? "s" : ""}</p>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Posthoc results */}
        {posthoc.length > 0 && (
          <div className="p-6 rounded-xl border border-border bg-card mb-6">
            <h3 className="text-lg font-semibold text-foreground mb-4">Post-Hoc Analysis</h3>
            <div className="space-y-3">
              {posthoc.map((p: any) => (
                <div key={p.id} className="p-3 rounded-lg bg-muted/50">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-foreground uppercase">{p.pack_id}</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full ${p.status === "completed" ? "bg-green-500/10 text-green-600" : "bg-red-500/10 text-red-600"}`}>
                      {p.status}
                    </span>
                  </div>
                  {p.metrics_json?.summary && (
                    <p className="text-xs text-muted-foreground">
                      Avg: {p.metrics_json.summary.avg?.toFixed(3)} · Max: {p.metrics_json.summary.max?.toFixed(3)} · Flagged: {p.metrics_json.summary.flagged}
                    </p>
                  )}
                  {p.status === "failed" && p.error_message && (
                    <p className="text-xs text-red-600/80 mt-1">{p.error_message}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Report */}
        {report?.content_markdown && (
          <div className="p-6 rounded-xl border border-border bg-card mb-6">
            <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <FileText className="h-5 w-5 text-primary" />Report
            </h3>
            <div className="prose prose-sm max-w-none dark:prose-invert text-sm whitespace-pre-wrap">
              {report.content_markdown.replace(/<script[\s\S]*?<\/script>/gi, "").replace(/on\w+\s*=\s*["'][^"']*["']/gi, "")}
            </div>
          </div>
        )}

        {/* Transcripts */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-foreground mb-4">
            Transcripts ({transcripts.length})
          </h3>
          <div className="space-y-2">
            {transcripts.map((t) => (
              <div key={t.id} className="rounded-xl border border-border bg-card overflow-hidden">
                <button
                  onClick={() => setExpandedTranscript(expandedTranscript === t.id ? null : t.id)}
                  className="w-full p-4 flex items-center justify-between text-left hover:bg-muted/50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <FileText className="h-4 w-4 text-muted-foreground" />
                    <div>
                      <p className="text-sm font-medium text-foreground">
                        {t.scenario_id || "Transcript"}{t.epoch_number != null ? ` (epoch ${t.epoch_number})` : ""}
                      </p>
                      {t.flags?.length ? (
                        <div className="flex gap-1 mt-1">
                          {t.flags.map((f, i) => (
                            <span key={i} className="text-xs px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-600">{f}</span>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  </div>
                  {expandedTranscript === t.id ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                </button>
                {expandedTranscript === t.id && t.content && (
                  <div className="px-4 pb-4 border-t border-border">
                    <pre className="text-xs text-muted-foreground whitespace-pre-wrap mt-3 max-h-96 overflow-y-auto bg-muted/30 p-3 rounded-lg">
                      {typeof t.content === "string" && t.content.length > 5000
                        ? t.content.slice(0, 5000) + "\n\n... (truncated)"
                        : t.content}
                    </pre>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
