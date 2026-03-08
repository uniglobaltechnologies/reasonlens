import { useState } from "react";
import { Link } from "react-router-dom";
import {
  ArrowLeft,
  Send,
  Loader2,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Shield,
} from "lucide-react";
import Header from "@/components/Header";
import { apiPost, isAuthenticated } from "@/lib/api";

interface EvalResult {
  feasibility: number;
  recommendation: "augment" | "automate" | "avoid";
  reasoning: string;
  safeguards: string[];
  risks: string[];
  implementation: string;
}

const recColors = {
  augment: { bg: "bg-green-500/10", text: "text-green-600", label: "Augment — AI assists humans" },
  automate: { bg: "bg-amber-500/10", text: "text-amber-600", label: "Automate — AI can lead with oversight" },
  avoid: { bg: "bg-red-500/10", text: "text-red-600", label: "Avoid — AI not suitable for this task" },
};

export default function Evaluate() {
  const [task, setTask] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EvalResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!task.trim() || loading) return;
    if (!isAuthenticated()) {
      setError("Please sign in to run AI evaluations.");
      return;
    }

    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const res = await apiPost<EvalResult>("/task-evaluator", {
        taskDescription: task,
      });
      setResult(res);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto px-4 sm:px-6 py-8 max-w-3xl">
        <Link
          to="/"
          className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-8"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Hub
        </Link>

        <h2 className="text-2xl sm:text-3xl font-bold text-foreground mb-2">
          Can AI Do This?
        </h2>
        <p className="text-muted-foreground mb-8">
          Describe an educational task and get an evidence-based evaluation of
          whether AI should be used, with safeguards and risks.
        </p>

        {/* Input */}
        <form onSubmit={handleSubmit} className="mb-8">
          <textarea
            value={task}
            onChange={(e) => setTask(e.target.value)}
            placeholder="Describe the educational task you want to evaluate for AI use. For example: 'Using AI to grade GCSE English essays and provide feedback to students'"
            className="w-full h-32 px-4 py-3 text-sm bg-background border border-border rounded-xl focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !task.trim()}
            className="mt-3 inline-flex items-center gap-2 px-6 py-2.5 bg-primary text-primary-foreground text-sm font-medium rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50"
          >
            {loading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
            {loading ? "Evaluating..." : "Evaluate Task"}
          </button>
        </form>

        {/* Error */}
        {error && (
          <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 mb-6">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="space-y-6">
            {/* Recommendation badge */}
            <div
              className={`p-4 rounded-xl ${recColors[result.recommendation].bg}`}
            >
              <div className="flex items-center gap-3 mb-2">
                {result.recommendation === "augment" && (
                  <CheckCircle2 className="h-6 w-6 text-green-600" />
                )}
                {result.recommendation === "automate" && (
                  <AlertTriangle className="h-6 w-6 text-amber-600" />
                )}
                {result.recommendation === "avoid" && (
                  <XCircle className="h-6 w-6 text-red-600" />
                )}
                <div>
                  <p
                    className={`font-semibold ${recColors[result.recommendation].text}`}
                  >
                    {recColors[result.recommendation].label}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Feasibility: {result.feasibility}/5
                  </p>
                </div>
              </div>
            </div>

            {/* Reasoning */}
            <div className="p-4 rounded-xl border border-border bg-card">
              <h4 className="font-semibold text-foreground mb-2">Reasoning</h4>
              <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                {result.reasoning}
              </p>
            </div>

            {/* Safeguards */}
            {result.safeguards?.length > 0 && (
              <div className="p-4 rounded-xl border border-border bg-card">
                <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                  <Shield className="h-4 w-4 text-primary" />
                  Required Safeguards
                </h4>
                <ul className="space-y-1.5">
                  {result.safeguards.map((s, i) => (
                    <li
                      key={i}
                      className="text-sm text-muted-foreground flex items-start gap-2"
                    >
                      <span className="text-primary mt-0.5">•</span>
                      {s}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Risks */}
            {result.risks?.length > 0 && (
              <div className="p-4 rounded-xl border border-border bg-card">
                <h4 className="font-semibold text-foreground mb-2 flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-amber-500" />
                  Potential Risks
                </h4>
                <ul className="space-y-1.5">
                  {result.risks.map((r, i) => (
                    <li
                      key={i}
                      className="text-sm text-muted-foreground flex items-start gap-2"
                    >
                      <span className="text-amber-500 mt-0.5">•</span>
                      {r}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Implementation */}
            {result.implementation && (
              <div className="p-4 rounded-xl border border-border bg-card">
                <h4 className="font-semibold text-foreground mb-2">
                  Implementation Guidance
                </h4>
                <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                  {result.implementation}
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
