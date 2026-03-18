import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, ArrowRight, CheckCircle2, Loader2 } from "lucide-react";
import Header from "@/components/Header";
import { getFrameworkById, FRAMEWORKS } from "@/data/frameworks";
import { apiPost } from "@/lib/api";

export default function Assess() {
  const { framework: frameworkId } = useParams<{ framework: string }>();

  // If no framework selected, show picker
  if (!frameworkId) return <FrameworkPicker />;

  const fw = getFrameworkById(frameworkId);
  if (!fw) return <FrameworkPicker />;

  return <AssessmentFlow framework={fw} />;
}

// Frameworks that have scenario-based assessment available
const SCENARIO_FRAMEWORKS = new Set(["teacher-competency", "maturity-the"]);

function FrameworkPicker() {
  const assessable = FRAMEWORKS.filter((f) => f.assessmentQuestions?.length > 0 && f.showInDashboard);

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto px-4 sm:px-6 py-8">
        <Link to="/" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-8">
          <ArrowLeft className="h-4 w-4" />Back to Hub
        </Link>
        <h2 className="text-2xl sm:text-3xl font-bold text-foreground mb-2">Assess Your AI Readiness</h2>
        <p className="text-muted-foreground mb-8">Choose a framework to assess against. Quick self-assessments take 5-10 minutes; scenario assessments can be substantially longer.</p>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {assessable.map((fw) => (
            <div key={fw.id} className="p-5 rounded-xl border border-border bg-card">
              <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-primary/10 text-primary">{fw.source}</span>
              <h4 className="font-semibold text-foreground mt-2 mb-1">{fw.shortName || fw.name}</h4>
              <p className="text-sm text-muted-foreground line-clamp-2">{fw.assessmentDescription || fw.description}</p>
              <div className="mt-3 flex flex-col gap-2">
                <Link
                  to={`/assess/${fw.id}`}
                  className="block py-2 px-3 text-sm text-center rounded-lg border border-border hover:border-primary/50 hover:bg-accent/50 transition-all"
                >
                  Quick Self-Assessment · ~{fw.estimatedAssessmentMinutes || 5} min
                </Link>
                {SCENARIO_FRAMEWORKS.has(fw.id) && (
                  <Link
                    to={`/assess/scenario/${fw.id}`}
                    className="block py-2 px-3 text-sm text-center rounded-lg bg-primary/10 text-primary font-medium hover:bg-primary/20 transition-all"
                  >
                    Scenario Assessment · ~{fw.id === "maturity-the" ? 40 : 15} min
                  </Link>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function AssessmentFlow({ framework: fw }: { framework: any }) {
  const [currentQ, setCurrentQ] = useState(0);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [showResults, setShowResults] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedCount, setSavedCount] = useState<number>(0);

  const questions = fw.assessmentQuestions || [];
  const question = questions[currentQ];
  const progress = ((currentQ + 1) / questions.length) * 100;

  // Resolve dimension IDs to human-readable names from keyDimensions
  const dimNameMap = new Map<string, string>();
  for (const dim of fw.keyDimensions || []) {
    dimNameMap.set(dim.id, dim.name);
  }
  const dimLabel = (dim: string) => dimNameMap.get(dim) || dim;

  const handleSelect = (value: string) => {
    setAnswers((prev) => ({ ...prev, [question.id]: value }));
  };

  const saveAssessment = async () => {
    if (isSaving) return;
    setIsSaving(true);
    setSaveError(null);

    try {
      const results = questions
        .filter((q: any) => !!answers[q.id])
        .map((q: any) => {
          const selectedValue = answers[q.id];
          const option = q.options.find((o: any) => o.value === selectedValue);
          return {
            framework_id: fw.id,
            framework_name: fw.name,
            question_id: q.id,
            dimension: q.dimension,
            selected_level: option?.level || selectedValue,
          };
        });

      const res = await apiPost<{ saved: number }>("/assessments", { results });
      setSavedCount(res.saved || results.length);
    } catch (err: any) {
      setSaveError(err?.message || "Failed to save assessment");
    } finally {
      setIsSaving(false);
      setShowResults(true);
    }
  };

  const handleNext = () => {
    if (currentQ < questions.length - 1) {
      setCurrentQ((prev) => prev + 1);
    } else {
      void saveAssessment();
    }
  };

  const handlePrev = () => {
    if (currentQ > 0 && !isSaving) setCurrentQ((prev) => prev - 1);
  };

  if (showResults) {
    const levelCounts: Record<string, number> = {};
    for (const q of questions) {
      const selected = answers[q.id];
      if (selected) {
        const option = q.options.find((o: any) => o.value === selected);
        if (option) {
          levelCounts[option.level] = (levelCounts[option.level] || 0) + 1;
        }
      }
    }
    const total = Object.values(levelCounts).reduce((a, b) => a + b, 0);

    return (
      <div className="min-h-screen bg-background">
        <Header />
        <div className="container mx-auto px-4 sm:px-6 py-8 max-w-2xl">
          <h2 className="text-2xl font-bold text-foreground mb-2">Assessment Complete</h2>
          <p className="text-muted-foreground mb-6">{fw.shortName || fw.name}</p>
          <div className="p-6 rounded-xl border border-border bg-card mb-6">
            <div className="flex items-center gap-3 mb-4">
              <CheckCircle2 className="h-8 w-8 text-green-500" />
              <div>
                <p className="font-semibold text-foreground">Results</p>
                <p className="text-sm text-muted-foreground">{total} questions answered</p>
              </div>
            </div>
            {saveError ? (
              <p className="text-sm text-red-600 mb-4">{saveError}</p>
            ) : (
              <p className="text-sm text-green-700 mb-4">Saved {savedCount || total} responses.</p>
            )}
            <div className="space-y-3">
              {Object.entries(levelCounts).map(([level, count]) => (
                <div key={level}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-foreground capitalize">{level}</span>
                    <span className="text-muted-foreground">{Math.round((count / total) * 100)}%</span>
                  </div>
                  <div className="w-full bg-muted rounded-full h-2">
                    <div className="bg-primary rounded-full h-2 transition-all" style={{ width: `${(count / total) * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="flex gap-3">
            <Link to={`/learning-path/${fw.id}`} className="px-4 py-2 bg-primary text-primary-foreground text-sm font-medium rounded-lg hover:bg-primary/90 transition-colors">
              View Learning Path
            </Link>
            <Link to="/assess" className="px-4 py-2 border border-border text-sm font-medium rounded-lg hover:bg-muted transition-colors">
              Assess Another Framework
            </Link>
          </div>
        </div>
      </div>
    );
  }

  if (!question) return null;

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto px-4 sm:px-6 py-8 max-w-2xl">
        <Link to="/assess" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-6">
          <ArrowLeft className="h-4 w-4" />{fw.shortName || fw.name}
        </Link>

        {/* Progress bar */}
        <div className="w-full bg-muted rounded-full h-1.5 mb-6">
          <div className="bg-primary rounded-full h-1.5 transition-all" style={{ width: `${progress}%` }} />
        </div>

        <p className="text-xs text-muted-foreground mb-2">Question {currentQ + 1} of {questions.length} · {dimLabel(question.dimension)}</p>
        <h3 className="text-lg font-semibold text-foreground mb-6">{question.question}</h3>

        <div className="space-y-3 mb-8">
          {question.options.map((opt: any) => (
            <button
              key={opt.value}
              onClick={() => handleSelect(opt.value)}
              className={`w-full text-left p-4 rounded-xl border-2 transition-all ${
                answers[question.id] === opt.value
                  ? "border-primary bg-primary/5 shadow-sm"
                  : "border-border hover:border-primary/30"
              }`}
            >
              <p className="text-sm text-foreground">{opt.label}</p>
              <span className="text-xs text-muted-foreground capitalize mt-1 inline-block">{opt.level}</span>
            </button>
          ))}
        </div>

        <div className="flex justify-between">
          <button
            onClick={handlePrev}
            disabled={currentQ === 0 || isSaving}
            className="px-4 py-2 text-sm border border-border rounded-lg hover:bg-muted transition-colors disabled:opacity-30"
          >
            Previous
          </button>
          <button
            onClick={handleNext}
            disabled={!answers[question.id] || isSaving}
            className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground text-sm font-medium rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50"
          >
            {isSaving ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Saving...
              </>
            ) : (
              <>
                {currentQ < questions.length - 1 ? "Next" : "See Results"}
                <ArrowRight className="h-4 w-4" />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
