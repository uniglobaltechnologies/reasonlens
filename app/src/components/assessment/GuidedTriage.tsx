/**
 * THE DMI Guided Triage — 7-screen flow replacing the quick self-assessment.
 *
 * Does NOT produce a maturity score. Produces a pillar-level priority map
 * and a recommendation for where to focus the full scenario assessment.
 */
import { useState, useMemo } from "react";
import { ArrowLeft, ArrowRight, Loader2 } from "lucide-react";
import { apiPost, isAuthenticated } from "@/lib/api";
import TriageResults from "./TriageResults";

type Signal = "incidental" | "intentional" | "integrated" | "optimised";

interface TriageResponse {
  triage_id: string;
  pillar_signals: Record<string, { signal: Signal; category: string; name: string }>;
  perceived_priority: string | null;
  recommendation: {
    pillar: string;
    pillar_name: string;
    reason: string;
    scenario_count: number;
    estimated_time_minutes: number;
  };
  visibility_note: string;
}

// ── Question Data ──────────────────────────────

const ROLE_OPTIONS = [
  { value: "senior_leadership", label: "Senior institutional leadership (VC, DVC, PVC, COO)" },
  { value: "faculty_leader", label: "Faculty or school leader (Dean, Head of School)" },
  { value: "ps_director", label: "Professional services director (IT, Library, HR, Finance)" },
  { value: "department_head", label: "Academic department head or programme leader" },
  { value: "academic_staff", label: "Academic staff (lecturer, researcher)" },
  { value: "ps_staff", label: "Professional services staff" },
];

const VISIBILITY_OPTIONS = [
  { value: "institution_wide", label: "I have a good view across the whole institution" },
  { value: "faculty_level", label: "I know my faculty or division well, and have some visibility of others" },
  { value: "department_level", label: "I mainly know what happens in my department or team" },
];

interface PillarQuestion {
  pillar: string;
  question: string;
  options: { text: string; signal: Signal }[];
}

const PILLAR_QUESTIONS: PillarQuestion[] = [
  {
    pillar: "teaching_learning",
    question: "Which of these best describes how your institution approaches digital technology in teaching and learning?",
    options: [
      { text: "Individual academics choose their own digital tools. There is no institutional strategy and the student experience varies significantly depending on who teaches them.", signal: "incidental" },
      { text: "There is a recognised need to coordinate, and some initiatives are underway: pilot programmes, new platforms being trialled, pockets of good practice. But it is not yet joined up across the institution.", signal: "intentional" },
      { text: "Digital tools and platforms are centrally supported and widely adopted. Learning analytics inform course design. Most staff have had training and there are clear institutional standards, though some departments are further ahead than others.", signal: "integrated" },
      { text: "The institution is a recognised leader in digital education. Adaptive learning, predictive analytics, and continuous curriculum innovation are embedded. Performance is benchmarked internationally and feeds into ongoing improvement cycles.", signal: "optimised" },
    ],
  },
  {
    pillar: "research",
    question: "Which of these best describes how your institution supports digital tools and data in the research lifecycle?",
    options: [
      { text: "Researchers find and use their own digital tools. There is no institutional research data management strategy. Collaboration tools are ad-hoc and vary by research group.", signal: "incidental" },
      { text: "The institution has started investing in shared research platforms and data storage. Training on digital research methods exists but is optional. Some research groups collaborate digitally but it is not the norm.", signal: "intentional" },
      { text: "Research data management policies are in place and followed. Digital collaboration platforms are standard across most research groups. The institution supports open research practices and has integrated systems for grant management, outputs, and impact tracking.", signal: "integrated" },
      { text: "The institution uses AI-enhanced research tools, predictive research analytics, and automated compliance checking. Research infrastructure is benchmarked against global leaders. Data sharing and open science practices are sector-leading.", signal: "optimised" },
    ],
  },
  {
    pillar: "professional_services",
    question: "Which of these best describes how digital technology is used across your institution's professional services (IT, HR, finance, library, student admin, marketing)?",
    options: [
      { text: "Most administrative processes are manual or use disconnected systems. Staff in different departments use different tools for similar tasks. Data is siloed and reporting requires manual compilation.", signal: "incidental" },
      { text: "There are initiatives to digitise key processes and integrate systems, but progress is uneven. Some departments have modern platforms while others still rely on legacy systems or spreadsheets.", signal: "intentional" },
      { text: "Core professional services run on integrated digital platforms. Data flows between systems (e.g. student records, finance, HR). Staff are trained on the tools and self-service portals reduce manual work.", signal: "integrated" },
      { text: "Professional services are fully digitised with AI-assisted workflow automation, predictive resource planning, and real-time operational dashboards. The institution benchmarks service delivery against sector leaders.", signal: "optimised" },
    ],
  },
  {
    pillar: "planning_governance",
    question: "Which of these best describes how your institution governs and plans its digital transformation?",
    options: [
      { text: "There is no formal digital strategy. Technology decisions are made reactively, often driven by individual champions or vendor approaches. IT budget is treated as a cost centre with no strategic investment framework.", signal: "incidental" },
      { text: "A digital strategy is being developed or has recently been approved. There is a governance structure (e.g. a digital transformation committee) but it is new and not yet embedded. Budget for digital initiatives exists but competes with other priorities.", signal: "intentional" },
      { text: "Digital transformation is governed by a mature committee structure with clear accountability. The strategy is funded, monitored against KPIs, and reviewed annually. Change management processes support adoption across the institution.", signal: "integrated" },
      { text: "Digital governance is a strategic differentiator. The institution's board treats digital maturity as a core performance metric. Investment decisions are data-driven. The institution leads sector conversations on digital governance.", signal: "optimised" },
    ],
  },
];

const PRIORITY_OPTIONS = [
  { value: "strategy", label: "We need a clearer strategy for digital transformation" },
  { value: "people_culture", label: "Our staff lack the digital skills and confidence needed" },
  { value: "technology", label: "Our technology infrastructure is holding us back" },
  { value: "data", label: "We do not use data effectively for decisions" },
  { value: "utilisation", label: "We have the tools but people are not using them well" },
];

// ── Component ──────────────────────────────────

export default function GuidedTriage() {
  const [step, setStep] = useState(0); // 0=context, 1-4=pillars, 5=priority, 6=submitting/results
  const [role, setRole] = useState("");
  const [visibility, setVisibility] = useState("");
  const [pillarAnswers, setPillarAnswers] = useState<Record<string, Signal>>({});
  const [priorityDimension, setPriorityDimension] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [triageResult, setTriageResult] = useState<TriageResponse | null>(null);

  // Shuffle options per pillar (stable per render via useMemo)
  const shuffledQuestions = useMemo(() => {
    return PILLAR_QUESTIONS.map((q) => ({
      ...q,
      options: [...q.options].sort(() => Math.random() - 0.5),
    }));
  }, []);

  const totalSteps = 7; // context + 4 pillars + priority + submit
  const canNext = () => {
    if (step === 0) return !!role && !!visibility;
    if (step >= 1 && step <= 4) return !!pillarAnswers[PILLAR_QUESTIONS[step - 1].pillar];
    if (step === 5) return !!priorityDimension;
    return false;
  };

  const handleNext = async () => {
    if (step < 5) {
      setStep(step + 1);
    } else if (step === 5) {
      // Submit
      setSubmitting(true);
      setError("");
      try {
        const res = await apiPost<TriageResponse>("/triage", {
          framework_id: "maturity-the",
          respondent_role: role,
          respondent_visibility: visibility,
          pillar_responses: pillarAnswers,
          perceived_priority_dimension: priorityDimension,
        });
        setTriageResult(res);
        setStep(6);
      } catch (err: any) {
        setError(err.message || "Failed to submit triage");
      } finally {
        setSubmitting(false);
      }
    }
  };

  // Show results
  if (step === 6 && triageResult) {
    return <TriageResults result={triageResult} />;
  }

  return (
    <div className="max-w-2xl mx-auto">
      {/* Progress */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-muted-foreground">
          {step === 0 ? "About you" : step <= 4 ? `Question ${step} of 5` : "One last question"}
        </span>
        <span className="text-xs text-muted-foreground">{Math.round(((step + 1) / totalSteps) * 100)}%</span>
      </div>
      <div
        className="w-full bg-muted rounded-full h-1.5 mb-8"
        role="progressbar"
        aria-valuenow={step + 1}
        aria-valuemin={0}
        aria-valuemax={totalSteps}
      >
        <div className="bg-primary h-1.5 rounded-full transition-all duration-300" style={{ width: `${((step + 1) / totalSteps) * 100}%` }} />
      </div>

      {/* Step 0: Context */}
      {step === 0 && (
        <div>
          <h3 className="text-lg font-semibold text-foreground mb-1">Before we start</h3>
          <p className="text-sm text-muted-foreground mb-6">This helps us frame your results. It takes 30 seconds.</p>

          <div className="space-y-5">
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">What best describes your role?</label>
              <select value={role} onChange={(e) => setRole(e.target.value)} className="w-full p-2 rounded-lg border border-border bg-background text-foreground">
                <option value="">Select...</option>
                {ROLE_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-foreground mb-1.5">How much of your institution's operations do you have direct insight into?</label>
              <select value={visibility} onChange={(e) => setVisibility(e.target.value)} className="w-full p-2 rounded-lg border border-border bg-background text-foreground">
                <option value="">Select...</option>
                {VISIBILITY_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
              </select>
            </div>
          </div>
        </div>
      )}

      {/* Steps 1-4: Pillar questions */}
      {step >= 1 && step <= 4 && (
        <div>
          <p className="text-sm font-medium text-muted-foreground mb-4">{shuffledQuestions[step - 1].question}</p>
          <div className="space-y-3">
            {shuffledQuestions[step - 1].options.map((opt, i) => {
              const pillar = PILLAR_QUESTIONS[step - 1].pillar;
              const isSelected = pillarAnswers[pillar] === opt.signal;
              return (
                <button
                  key={i}
                  onClick={() => setPillarAnswers((prev) => ({ ...prev, [pillar]: opt.signal }))}
                  className={`w-full text-left p-4 rounded-xl border transition-all text-sm leading-relaxed ${
                    isSelected
                      ? "border-primary bg-primary/5 ring-1 ring-primary"
                      : "border-border bg-card hover:border-primary/40"
                  }`}
                >
                  {opt.text}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Step 5: Priority dimension */}
      {step === 5 && (
        <div>
          <p className="text-sm font-medium text-muted-foreground mb-4">Thinking about your institution overall, which single area do you think needs the most urgent attention?</p>
          <div className="space-y-3">
            {PRIORITY_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                onClick={() => setPriorityDimension(opt.value)}
                className={`w-full text-left p-4 rounded-xl border transition-all text-sm ${
                  priorityDimension === opt.value
                    ? "border-primary bg-primary/5 ring-1 ring-primary"
                    : "border-border bg-card hover:border-primary/40"
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Error */}
      {error && <p className="mt-4 text-sm text-destructive text-center">{error}</p>}

      {/* Navigation */}
      <div className="flex items-center justify-between mt-8">
        {step > 0 ? (
          <button onClick={() => setStep(step - 1)} className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors">
            <ArrowLeft className="h-4 w-4" />Previous
          </button>
        ) : <div />}

        <button
          onClick={handleNext}
          disabled={!canNext() || submitting}
          className="inline-flex items-center gap-2 px-6 py-2.5 rounded-lg bg-primary text-primary-foreground font-medium hover:bg-primary/90 transition-colors disabled:opacity-50"
        >
          {submitting ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Submitting...
            </>
          ) : step === 5 ? (
            "See results"
          ) : (
            <>
              Next
              <ArrowRight className="h-4 w-4" />
            </>
          )}
        </button>
      </div>
    </div>
  );
}
