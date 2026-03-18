import { useState, useEffect, useRef } from "react";
import { Link, useParams, useNavigate } from "react-router-dom";
import { ArrowLeft, CheckCircle2, Loader2, AlertTriangle, Shield } from "lucide-react";
import Header from "@/components/Header";
import ContextOnboarding from "@/components/assessment/ContextOnboarding";
import SourceAttribution from "@/components/SourceAttribution";
import { apiGet, apiPost, ApiError, isAuthenticated } from "@/lib/api";

interface ScenarioResponse {
  id: string;
  response_key: string;
  text: string;
}

interface Scenario {
  scenario_id: string;
  dimension_name: string;
  stem: string;
  question: string;
  responses: ScenarioResponse[];
}

interface DimensionResult {
  dimension_id: string;
  dimension_name: string;
  assigned_level: string;
  assigned_level_order: number;
  confidence: "high" | "medium" | "low";
  answer_count: number;
  answer_distribution: Record<string, number>;
}

type Phase = "checking" | "onboarding" | "assessing" | "completing" | "results";

export default function ScenarioAssess() {
  const { framework } = useParams<{ framework: string }>();
  const navigate = useNavigate();
  const [phase, setPhase] = useState<Phase>("checking");
  const [sessionId, setSessionId] = useState<string>("");
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [results, setResults] = useState<DimensionResult[]>([]);
  const [error, setError] = useState<string>("");
  const scenarioStartTime = useRef<number>(Date.now());

  useEffect(() => {
    if (!isAuthenticated()) {
      navigate("/auth");
      return;
    }
    checkContext();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  async function checkContext() {
    try {
      await apiGet("/user-assessment-context");
      await startSession();
    } catch (e: unknown) {
      if (e instanceof ApiError && e.status === 404) {
        setPhase("onboarding");
      } else {
        setError(
          e instanceof ApiError
            ? e.message
            : "Unable to connect to the server. Please try again later."
        );
      }
    }
  }

  async function startSession() {
    try {
      const data = await apiPost<{
        session_id: string;
        scenarios: Scenario[];
      }>("/scenario-sessions", { framework_id: framework });
      setSessionId(data.session_id);
      setScenarios(data.scenarios);
      setCurrentIdx(0);
      scenarioStartTime.current = Date.now();
      setPhase("assessing");
    } catch (err: any) {
      setError(err.message || "Failed to start session");
    }
  }

  async function submitAnswer(responseId: string) {
    const scenario = scenarios[currentIdx];
    const timeSeconds = (Date.now() - scenarioStartTime.current) / 1000;

    try {
      await apiPost("/scenario-answers", {
        session_id: sessionId,
        scenario_id: scenario.scenario_id,
        response_id: responseId,
        time_to_respond_seconds: Math.round(timeSeconds * 10) / 10,
      });

      setAnswers((prev) => ({ ...prev, [scenario.scenario_id]: responseId }));

      if (currentIdx < scenarios.length - 1) {
        setCurrentIdx(currentIdx + 1);
        scenarioStartTime.current = Date.now();
      } else {
        await completeSession();
      }
    } catch (err: any) {
      setError(err.message || "Failed to save answer");
    }
  }

  async function completeSession() {
    setPhase("completing");
    try {
      const data = await apiPost<{ results: DimensionResult[] }>(
        "/scenario-session-complete",
        { session_id: sessionId }
      );
      setResults(data.results);
      setPhase("results");
    } catch (err: any) {
      setError(err.message || "Failed to complete session");
    }
  }

  function goBack() {
    if (currentIdx > 0) {
      setCurrentIdx(currentIdx - 1);
      scenarioStartTime.current = Date.now();
    }
  }

  if (error) {
    return (
      <div className="min-h-screen bg-background">
        <Header />
        <div className="container mx-auto px-4 py-8">
          <div className="max-w-2xl mx-auto text-center">
            <AlertTriangle className="h-12 w-12 text-destructive mx-auto mb-4" />
            <h2 className="text-xl font-semibold mb-2">Something went wrong</h2>
            <p className="text-muted-foreground mb-4">{error}</p>
            <Link to="/assess" className="text-primary hover:underline">Back to assessments</Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto px-4 sm:px-6 py-8">
        <Link to="/assess" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-8">
          <ArrowLeft className="h-4 w-4" />Back to assessments
        </Link>

        {phase === "checking" && (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          </div>
        )}

        {phase === "onboarding" && (
          <ContextOnboarding onComplete={startSession} />
        )}

        {phase === "assessing" && scenarios.length > 0 && (
          <ScenarioCard
            scenario={scenarios[currentIdx]}
            index={currentIdx}
            total={scenarios.length}
            selectedResponseId={answers[scenarios[currentIdx].scenario_id]}
            onSelect={submitAnswer}
            onBack={currentIdx > 0 ? goBack : undefined}
          />
        )}

        {phase === "completing" && (
          <div className="flex flex-col items-center justify-center py-20 gap-4">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
            <p className="text-muted-foreground">Calculating your results...</p>
          </div>
        )}

        {phase === "results" && (
          <ResultsView
            results={results}
            frameworkId={framework!}
          />
        )}
      </div>
    </div>
  );
}

function ScenarioCard({
  scenario,
  index,
  total,
  selectedResponseId,
  onSelect,
  onBack,
}: {
  scenario: Scenario;
  index: number;
  total: number;
  selectedResponseId?: string;
  onSelect: (responseId: string) => void;
  onBack?: () => void;
}) {
  return (
    <div className="max-w-3xl mx-auto">
      {/* Progress */}
      <div className="flex items-center justify-between mb-6">
        <span className="text-sm text-muted-foreground">
          Scenario {index + 1} of {total}
        </span>
        <span className="text-xs px-2 py-1 rounded-full bg-primary/10 text-primary font-medium">
          {scenario.dimension_name}
        </span>
      </div>
      <div
        className="w-full bg-muted rounded-full h-1.5 mb-8"
        role="progressbar"
        aria-valuenow={index + 1}
        aria-valuemin={0}
        aria-valuemax={total}
        aria-label={`Scenario ${index + 1} of ${total}`}
      >
        <div
          className="bg-primary h-1.5 rounded-full transition-all duration-300"
          style={{ width: `${((index + 1) / total) * 100}%` }}
        />
      </div>

      {/* Scenario stem */}
      <div className="p-6 rounded-xl border border-border bg-card mb-6">
        <p className="text-foreground leading-relaxed">{scenario.stem}</p>
      </div>

      {/* Question */}
      <p className="text-sm font-medium text-muted-foreground mb-4">
        {scenario.question}
      </p>

      {/* Response options — NO level labels shown */}
      <div className="space-y-3">
        {scenario.responses.map((r) => (
          <button
            key={r.id}
            onClick={() => onSelect(r.id)}
            disabled={!!selectedResponseId}
            className={`w-full text-left p-4 rounded-xl border transition-all ${
              selectedResponseId === r.id
                ? "border-primary bg-primary/5"
                : "border-border bg-card hover:border-primary/40 hover:bg-accent/50"
            } disabled:opacity-70`}
          >
            <span className="text-foreground leading-relaxed text-sm">
              {r.text}
            </span>
          </button>
        ))}
      </div>

      {/* Navigation */}
      <div className="flex items-center justify-between mt-8">
        {onBack ? (
          <button
            onClick={onBack}
            className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />Previous
          </button>
        ) : (
          <div />
        )}
        <span className="text-xs text-muted-foreground">
          Select a response to continue
        </span>
      </div>
    </div>
  );
}

function ResultsView({
  results,
  frameworkId,
}: {
  results: DimensionResult[];
  frameworkId: string;
}) {
  const confidenceStyles = {
    high: { bg: "bg-emerald-50 dark:bg-emerald-950/30", text: "text-emerald-700 dark:text-emerald-400", label: "High confidence" },
    medium: { bg: "bg-amber-50 dark:bg-amber-950/30", text: "text-amber-700 dark:text-amber-400", label: "Medium confidence" },
    low: { bg: "bg-red-50 dark:bg-red-950/30", text: "text-red-700 dark:text-red-400", label: "Low confidence" },
  };

  return (
    <div className="max-w-3xl mx-auto">
      <div className="flex items-center gap-3 mb-6">
        <CheckCircle2 className="h-8 w-8 text-emerald-500" />
        <div>
          <h2 className="text-2xl font-bold text-foreground">Assessment Complete</h2>
          <p className="text-muted-foreground text-sm">
            Your results are based on how you responded to realistic scenarios, not self-assessment.
          </p>
        </div>
      </div>

      <div className="space-y-4 mb-8">
        {results.map((r) => {
          const style = confidenceStyles[r.confidence];
          return (
            <div
              key={r.dimension_id}
              className={`p-5 rounded-xl border border-border ${style.bg}`}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h4 className="font-semibold text-foreground">{r.dimension_name}</h4>
                  <p className="text-lg font-bold text-foreground mt-1">
                    {r.assigned_level}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <Shield className="h-4 w-4" />
                  <span className={`text-xs font-medium ${style.text}`}>
                    {style.label}
                  </span>
                </div>
              </div>
              <div className="mt-3 flex gap-2 flex-wrap">
                {Object.entries(r.answer_distribution).map(([level, count]) => (
                  <span
                    key={level}
                    className="text-xs px-2 py-0.5 rounded-full bg-background border border-border text-muted-foreground"
                  >
                    {level}: {count}
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      <div className="flex flex-col sm:flex-row gap-3">
        <Link
          to={`/learning-path/${frameworkId}`}
          className="flex-1 py-3 rounded-lg bg-primary text-primary-foreground font-medium text-center hover:bg-primary/90 transition-colors"
        >
          View Learning Path
        </Link>
        <Link
          to="/my-progress"
          className="flex-1 py-3 rounded-lg border border-border text-foreground font-medium text-center hover:bg-accent transition-colors"
        >
          My Progress
        </Link>
      </div>

      <div className="mt-8">
        <SourceAttribution attribution={frameworkId === "teacher-competency" ? {
          source_framework: "UNESCO AI Competency Framework for Teachers",
          source_licence: "CC BY-SA 3.0 IGO",
          content_type: "derivative",
          attribution_text: "Based on the UNESCO AI Competency Framework for Teachers (2024). Scenarios created by ReasonLens.",
        } : frameworkId === "maturity-the" ? {
          source_framework: "THE Digital Maturity Index",
          content_type: "original",
          attribution_text: "Scenarios created by ReasonLens based on the Times Higher Education Digital Maturity Index.",
        } : {
          source_framework: "Assessment Framework",
          content_type: "original",
          attribution_text: "Scenarios created by ReasonLens.",
        }} />
      </div>
    </div>
  );
}
