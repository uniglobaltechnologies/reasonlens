import { useState, useEffect } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import {
  ArrowLeft,
  Loader2,
  AlertTriangle,
  FileText,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Sparkles,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import Header from "@/components/Header";
import { apiGet, apiPost, ApiError, isAuthenticated } from "@/lib/api";

// ── Types ────────────────────────────────────────────────────────────

interface InterpretiveReport {
  interpretation_id?: string;
  sections: {
    executive_summary: string;
    pillar_teaching_learning: string;
    pillar_research: string;
    pillar_professional_services: string;
    pillar_planning_governance: string;
    recommendations: string;
  };
  metadata: {
    generated_at: string;
    model_used: string;
    methodology_version: string;
    total_generation_time_ms: number;
  };
}

interface OpenEndedAnswers {
  trigger_response: string;
  previous_attempts: string;
  constraints: string[];
  constraints_detail: string;
  success_definition: string;
  additional_context: string;
}

type Phase = "loading" | "questions" | "generating" | "report" | "error";

const TRIGGER_OPTIONS = [
  { value: "new_leadership", label: "New institutional leadership" },
  { value: "strategy_refresh", label: "Strategy refresh or planning cycle" },
  { value: "accreditation", label: "Accreditation or external review" },
  { value: "problem_response", label: "Response to a specific problem" },
  { value: "benchmarking", label: "Benchmarking against peers" },
  { value: "funder_requirement", label: "Funder or regulator requirement" },
  { value: "curiosity", label: "General interest" },
];

const CONSTRAINT_OPTIONS = [
  { value: "budget", label: "Budget / competing investment priorities" },
  { value: "staff_skills", label: "Staff skills and confidence" },
  { value: "leadership_buyin", label: "Leadership engagement or buy-in" },
  { value: "legacy_systems", label: "Legacy systems and technical debt" },
  { value: "governance_speed", label: "Governance and decision-making speed" },
  { value: "culture_resistance", label: "Institutional culture and resistance to change" },
  { value: "regulatory", label: "Regulatory or compliance requirements" },
  { value: "size_complexity", label: "Size and complexity of the institution" },
];

const GENERATION_STEPS = [
  "Analysing your maturity profile",
  "Interpreting Teaching & Learning",
  "Interpreting Research",
  "Interpreting Professional Services",
  "Interpreting Planning & Governance",
  "Preparing strategic recommendations",
];

const PILLAR_SECTIONS = [
  { key: "pillar_teaching_learning" as const, label: "Teaching & Learning" },
  { key: "pillar_research" as const, label: "Research" },
  { key: "pillar_professional_services" as const, label: "Professional Services" },
  { key: "pillar_planning_governance" as const, label: "Planning & Governance" },
];

// ── Component ────────────────────────────────────────────────────────

export default function TheInterpretation() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const [phase, setPhase] = useState<Phase>("loading");
  const [report, setReport] = useState<InterpretiveReport | null>(null);
  const [error, setError] = useState("");
  const [answers, setAnswers] = useState<OpenEndedAnswers>({
    trigger_response: "",
    previous_attempts: "",
    constraints: [],
    constraints_detail: "",
    success_definition: "",
    additional_context: "",
  });
  const [genStep, setGenStep] = useState(0);
  const [expandedPillar, setExpandedPillar] = useState<string | null>(null);

  useEffect(() => {
    if (!isAuthenticated()) {
      navigate(`/auth?return=/the-dmi/interpretation/${sessionId}`);
      return;
    }
    checkExistingReport();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  async function checkExistingReport() {
    try {
      const existing = await apiGet<InterpretiveReport>(
        `/generate-the-interpretation?session_id=${sessionId}`
      );
      if (existing?.sections?.executive_summary) {
        setReport(existing);
        setPhase("report");
        return;
      }
    } catch {
      // No existing report — check for existing context
    }
    try {
      const ctx = await apiGet<OpenEndedAnswers>(
        `/interpretation-context?session_id=${sessionId}`
      );
      if (ctx && !("error" in ctx)) {
        setAnswers({
          trigger_response: ctx.trigger_response || "",
          previous_attempts: ctx.previous_attempts || "",
          constraints: ctx.constraints || [],
          constraints_detail: ctx.constraints_detail || "",
          success_definition: ctx.success_definition || "",
          additional_context: ctx.additional_context || "",
        });
      }
    } catch {
      // No existing context — fine
    }
    setPhase("questions");
  }

  async function submitContext() {
    try {
      await apiPost("/interpretation-context", {
        session_id: sessionId,
        ...answers,
      });
      await generateReport();
    } catch (err: any) {
      setError(err instanceof ApiError ? err.message : "Failed to save context");
      setPhase("error");
    }
  }

  async function generateReport(regenerate = false) {
    setPhase("generating");
    setGenStep(0);

    // Cosmetic progress animation
    const interval = setInterval(() => {
      setGenStep(prev => {
        if (prev < GENERATION_STEPS.length - 1) return prev + 1;
        return prev;
      });
    }, 3000);

    try {
      const result = await apiPost<InterpretiveReport>(
        "/generate-the-interpretation",
        { session_id: sessionId, regenerate }
      );
      clearInterval(interval);
      setGenStep(GENERATION_STEPS.length);
      setReport(result);
      setPhase("report");
    } catch (err: any) {
      clearInterval(interval);
      setError(err instanceof ApiError ? err.message : "Report generation failed. Please try again.");
      setPhase("error");
    }
  }

  function toggleConstraint(value: string) {
    setAnswers(prev => ({
      ...prev,
      constraints: prev.constraints.includes(value)
        ? prev.constraints.filter(c => c !== value)
        : [...prev.constraints, value],
    }));
  }

  // ── Error state ────────────────────────────────────────────────────

  if (phase === "error") {
    return (
      <div className="min-h-screen bg-background">
        <Header />
        <div className="container mx-auto px-4 py-8">
          <div className="max-w-2xl mx-auto text-center">
            <AlertTriangle className="h-12 w-12 text-destructive mx-auto mb-4" />
            <h2 className="text-xl font-semibold mb-2">Something went wrong</h2>
            <p className="text-muted-foreground mb-4">{error}</p>
            <button
              onClick={() => { setError(""); setPhase("questions"); }}
              className="text-primary hover:underline"
            >
              Try again
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto px-4 sm:px-6 py-8">
        <Link
          to={`/assess/scenario/maturity-the`}
          className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-8"
        >
          <ArrowLeft className="h-4 w-4" />Back to assessment
        </Link>

        {/* Loading */}
        {phase === "loading" && (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          </div>
        )}

        {/* Open-ended questions */}
        {phase === "questions" && (
          <div className="max-w-2xl mx-auto">
            <div className="flex items-center gap-3 mb-2">
              <Sparkles className="h-6 w-6 text-primary" />
              <h2 className="text-2xl font-bold text-foreground">
                Help us personalise your report
              </h2>
            </div>
            <p className="text-muted-foreground text-sm mb-8">
              Answer these optional questions so our AI can provide contextually
              calibrated interpretations and recommendations. Takes about 3 minutes.
            </p>

            <div className="space-y-8">
              {/* Q1: Trigger */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  What triggered this assessment?
                </label>
                <select
                  value={answers.trigger_response}
                  onChange={e => setAnswers(prev => ({ ...prev, trigger_response: e.target.value }))}
                  className="w-full rounded-lg border border-border bg-card px-4 py-3 text-foreground"
                >
                  <option value="">Select...</option>
                  {TRIGGER_OPTIONS.map(o => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                  <option value="other">Other</option>
                </select>
              </div>

              {/* Q2: Previous attempts */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  What has your institution already tried?
                </label>
                <textarea
                  value={answers.previous_attempts}
                  onChange={e => setAnswers(prev => ({ ...prev, previous_attempts: e.target.value }))}
                  placeholder="In the areas where you scored lowest, has your institution previously attempted improvements? What happened?"
                  maxLength={1000}
                  rows={3}
                  className="w-full rounded-lg border border-border bg-card px-4 py-3 text-foreground placeholder:text-muted-foreground resize-none"
                />
                <p className="text-xs text-muted-foreground mt-1 text-right">
                  {answers.previous_attempts.length}/1000
                </p>
              </div>

              {/* Q3: Constraints */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  What are your main barriers?
                </label>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mb-3">
                  {CONSTRAINT_OPTIONS.map(o => (
                    <label
                      key={o.value}
                      className={`flex items-center gap-2 p-3 rounded-lg border cursor-pointer transition-colors ${
                        answers.constraints.includes(o.value)
                          ? "border-primary bg-primary/5"
                          : "border-border bg-card hover:border-primary/40"
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={answers.constraints.includes(o.value)}
                        onChange={() => toggleConstraint(o.value)}
                        className="rounded border-border"
                      />
                      <span className="text-sm text-foreground">{o.label}</span>
                    </label>
                  ))}
                </div>
                <textarea
                  value={answers.constraints_detail}
                  onChange={e => setAnswers(prev => ({ ...prev, constraints_detail: e.target.value }))}
                  placeholder="Add any detail about the barriers you selected (optional)"
                  maxLength={500}
                  rows={2}
                  className="w-full rounded-lg border border-border bg-card px-4 py-3 text-foreground placeholder:text-muted-foreground resize-none"
                />
              </div>

              {/* Q4: Success definition */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  What does success look like?
                </label>
                <textarea
                  value={answers.success_definition}
                  onChange={e => setAnswers(prev => ({ ...prev, success_definition: e.target.value }))}
                  placeholder="If this assessment leads to action, what would success look like in 2-3 years?"
                  maxLength={500}
                  rows={2}
                  className="w-full rounded-lg border border-border bg-card px-4 py-3 text-foreground placeholder:text-muted-foreground resize-none"
                />
                <p className="text-xs text-muted-foreground mt-1 text-right">
                  {answers.success_definition.length}/500
                </p>
              </div>

              {/* Q5: Anything else */}
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  Anything else we should know?
                </label>
                <textarea
                  value={answers.additional_context}
                  onChange={e => setAnswers(prev => ({ ...prev, additional_context: e.target.value }))}
                  placeholder="Is there anything else about your institution's context we should consider?"
                  maxLength={500}
                  rows={2}
                  className="w-full rounded-lg border border-border bg-card px-4 py-3 text-foreground placeholder:text-muted-foreground resize-none"
                />
              </div>
            </div>

            <div className="flex flex-col sm:flex-row gap-3 mt-8">
              <button
                onClick={submitContext}
                className="flex-1 py-3 rounded-lg bg-primary text-primary-foreground font-medium text-center hover:bg-primary/90 transition-colors"
              >
                Generate Interpretive Report
              </button>
              <button
                onClick={() => generateReport()}
                className="flex-1 py-3 rounded-lg border border-border text-foreground font-medium text-center hover:bg-accent transition-colors"
              >
                Skip and generate report
              </button>
            </div>
          </div>
        )}

        {/* Generating */}
        {phase === "generating" && (
          <div className="max-w-md mx-auto py-16">
            <div className="text-center mb-8">
              <Loader2 className="h-10 w-10 animate-spin text-primary mx-auto mb-4" />
              <h2 className="text-xl font-semibold text-foreground mb-2">
                Generating your interpretive report
              </h2>
              <p className="text-sm text-muted-foreground">
                This takes about 15-25 seconds...
              </p>
            </div>
            <div className="space-y-3">
              {GENERATION_STEPS.map((step, idx) => (
                <div
                  key={step}
                  className={`flex items-center gap-3 px-4 py-2 rounded-lg transition-all ${
                    idx < genStep
                      ? "bg-emerald-50 dark:bg-emerald-950/30"
                      : idx === genStep
                      ? "bg-primary/10"
                      : "bg-muted/30"
                  }`}
                >
                  {idx < genStep ? (
                    <span className="text-emerald-500 text-sm">✓</span>
                  ) : idx === genStep ? (
                    <Loader2 className="h-4 w-4 animate-spin text-primary" />
                  ) : (
                    <span className="h-4 w-4 rounded-full border border-border" />
                  )}
                  <span
                    className={`text-sm ${
                      idx <= genStep ? "text-foreground" : "text-muted-foreground"
                    }`}
                  >
                    {step}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Report */}
        {phase === "report" && report && (
          <div className="max-w-3xl mx-auto">
            {/* Header */}
            <div className="flex items-center gap-3 mb-6">
              <FileText className="h-8 w-8 text-primary" />
              <div>
                <h2 className="text-2xl font-bold text-foreground">
                  Interpretive Report
                </h2>
                <p className="text-muted-foreground text-sm">
                  Generated {new Date(report.metadata.generated_at).toLocaleDateString()} using{" "}
                  {report.metadata.model_used} ({report.metadata.methodology_version})
                </p>
              </div>
            </div>

            {/* AI disclaimer */}
            <div className="p-4 rounded-xl border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/30 mb-8">
              <p className="text-sm text-amber-800 dark:text-amber-300">
                The following interpretation was generated by ReasonLens AI based on
                your assessment results, institutional context, and the information
                you provided. It uses the ReasonLens Interpretive Methodology v
                {report.metadata.methodology_version}.
              </p>
            </div>

            {/* Executive Summary */}
            <section className="mb-10">
              <div className="prose prose-sm max-w-none dark:prose-invert">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {report.sections.executive_summary}
                </ReactMarkdown>
              </div>
            </section>

            {/* Pillar Analyses (accordion) */}
            <div className="space-y-3 mb-10">
              {PILLAR_SECTIONS.map(({ key, label }) => {
                const isOpen = expandedPillar === key;
                return (
                  <div key={key} className="border border-border rounded-xl overflow-hidden">
                    <button
                      onClick={() => setExpandedPillar(isOpen ? null : key)}
                      className="w-full flex items-center justify-between px-5 py-4 text-left hover:bg-accent/50 transition-colors"
                    >
                      <span className="font-semibold text-foreground">{label}</span>
                      {isOpen ? (
                        <ChevronUp className="h-5 w-5 text-muted-foreground" />
                      ) : (
                        <ChevronDown className="h-5 w-5 text-muted-foreground" />
                      )}
                    </button>
                    {isOpen && (
                      <div className="px-5 pb-5 border-t border-border">
                        <div className="prose prose-sm max-w-none dark:prose-invert pt-4">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {report.sections[key]}
                          </ReactMarkdown>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Recommendations */}
            <section className="mb-10">
              <div className="prose prose-sm max-w-none dark:prose-invert">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {report.sections.recommendations}
                </ReactMarkdown>
              </div>
            </section>

            {/* Actions */}
            <div className="flex flex-col sm:flex-row gap-3 mb-8">
              <button
                onClick={() => {
                  setPhase("questions");
                }}
                className="flex items-center justify-center gap-2 flex-1 py-3 rounded-lg border border-border text-foreground font-medium hover:bg-accent transition-colors"
              >
                <RefreshCw className="h-4 w-4" />
                Update context & regenerate
              </button>
            </div>

            {/* Metadata */}
            <div className="text-xs text-muted-foreground border-t border-border pt-4">
              <p>
                Report generated in {(report.metadata.total_generation_time_ms / 1000).toFixed(1)}s
                using {report.metadata.model_used}.
                Methodology version {report.metadata.methodology_version}.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
