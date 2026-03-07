import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import {
  ArrowLeft,
  CheckCircle2,
  XCircle,
  Loader2,
  Clock,
  ChevronRight,
} from "lucide-react";
import Header from "@/components/Header";
import { apiGet } from "@/lib/api";

interface AuditRun {
  id: string;
  scenario_pack: string;
  target_model: string;
  status: string;
  cost_tokens: number | null;
  created_at: string;
  completed_at: string | null;
}

const statusIcons: Record<string, typeof CheckCircle2> = {
  completed: CheckCircle2,
  failed: XCircle,
  running: Loader2,
  queued: Clock,
  stopped: XCircle,
};

const statusColors: Record<string, string> = {
  completed: "text-green-500",
  failed: "text-red-500",
  running: "text-amber-500",
  queued: "text-muted-foreground",
  stopped: "text-muted-foreground",
};

export default function AuditRuns() {
  const [runs, setRuns] = useState<AuditRun[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // TODO: Add a /api/audit-runs CRUD endpoint
    // For now, show empty state
    setLoading(false);
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto px-4 sm:px-6 py-8">
        <div className="flex items-center justify-between mb-8">
          <Link
            to="/audit"
            className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
            New Audit
          </Link>
        </div>

        <h2 className="text-2xl font-bold text-foreground mb-6">
          Your Audit Runs
        </h2>

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : runs.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-muted-foreground mb-4">
              No audit runs yet. Start your first audit to see results here.
            </p>
            <Link
              to="/audit"
              className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors text-sm font-medium"
            >
              Start an Audit
            </Link>
          </div>
        ) : (
          <div className="space-y-3">
            {runs.map((run) => {
              const StatusIcon = statusIcons[run.status] || Clock;
              return (
                <Link
                  key={run.id}
                  to={`/audit/runs/${run.id}`}
                  className="flex items-center justify-between p-4 rounded-xl border border-border bg-card hover:bg-muted/50 transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <StatusIcon
                      className={`h-5 w-5 ${statusColors[run.status]} ${
                        run.status === "running" ? "animate-spin" : ""
                      }`}
                    />
                    <div>
                      <p className="font-medium text-foreground">
                        {run.scenario_pack}
                      </p>
                      <p className="text-sm text-muted-foreground">
                        {run.target_model} ·{" "}
                        {new Date(run.created_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                  <ChevronRight className="h-5 w-5 text-muted-foreground" />
                </Link>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
