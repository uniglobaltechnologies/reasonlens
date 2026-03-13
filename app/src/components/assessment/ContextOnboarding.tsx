import { useState } from "react";
import { Loader2 } from "lucide-react";
import { apiPost } from "@/lib/api";

interface ContextOnboardingProps {
  onComplete: () => void;
}

const SUBJECT_OPTIONS = [
  "Computing", "English", "Maths", "Sciences", "Languages",
  "Art & Design", "History", "Geography", "PE", "Music",
  "Business Studies", "Social Sciences", "Engineering", "Health & Social Care", "Other",
];

const INSTITUTION_LEVEL_OPTIONS = [
  "Primary", "Secondary", "Further Education", "Higher Education", "Other",
];

const INSTITUTION_TYPE_OPTIONS = [
  "State school", "Academy / Trust", "Independent", "University", "FE College", "Other",
];

const AI_TOOLS_OPTIONS = [
  "ChatGPT", "Microsoft Copilot", "Google Gemini", "Claude", "Midjourney",
  "DALL-E", "Perplexity", "NotebookLM", "Other", "None",
];

const EXPERIENCE_OPTIONS = [
  "0-2 years", "3-5 years", "6-10 years", "11-20 years", "20+ years",
];

const MANAGEMENT_OPTIONS = [
  "No management responsibility", "Middle leader", "Senior leader", "Executive / Governor",
];

export default function ContextOnboarding({ onComplete }: ContextOnboardingProps) {
  const [subjectArea, setSubjectArea] = useState("");
  const [institutionLevel, setInstitutionLevel] = useState("");
  const [institutionType, setInstitutionType] = useState("");
  const [currentAiTools, setCurrentAiTools] = useState<string[]>([]);
  const [primaryFrustration, setPrimaryFrustration] = useState("");
  const [yearsOfExperience, setYearsOfExperience] = useState("");
  const [managementResponsibility, setManagementResponsibility] = useState("");
  const [saving, setSaving] = useState(false);

  const toggleTool = (tool: string) => {
    setCurrentAiTools((prev) =>
      prev.includes(tool) ? prev.filter((t) => t !== tool) : [...prev, tool]
    );
  };

  const handleSubmit = async () => {
    setSaving(true);
    try {
      await apiPost("/user-assessment-context", {
        subject_area: subjectArea || null,
        institution_type: institutionType || null,
        institution_level: institutionLevel || null,
        current_ai_tools: currentAiTools.length > 0 ? currentAiTools : null,
        primary_frustration: primaryFrustration || null,
        years_of_experience: yearsOfExperience || null,
        management_responsibility: managementResponsibility || null,
      });
      onComplete();
    } catch (err) {
      console.error("Failed to save context:", err);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <h3 className="text-xl font-semibold text-foreground mb-2">
        Before we start: a little about you
      </h3>
      <p className="text-sm text-muted-foreground mb-6">
        This helps us personalise your assessment scenarios. You can update these later.
      </p>

      <div className="space-y-5">
        <Field label="Subject area">
          <select
            value={subjectArea}
            onChange={(e) => setSubjectArea(e.target.value)}
            className="w-full p-2 rounded-lg border border-border bg-background text-foreground"
          >
            <option value="">Select...</option>
            {SUBJECT_OPTIONS.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </Field>

        <Field label="Institution level">
          <select
            value={institutionLevel}
            onChange={(e) => setInstitutionLevel(e.target.value)}
            className="w-full p-2 rounded-lg border border-border bg-background text-foreground"
          >
            <option value="">Select...</option>
            {INSTITUTION_LEVEL_OPTIONS.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </Field>

        <Field label="Institution type">
          <select
            value={institutionType}
            onChange={(e) => setInstitutionType(e.target.value)}
            className="w-full p-2 rounded-lg border border-border bg-background text-foreground"
          >
            <option value="">Select...</option>
            {INSTITUTION_TYPE_OPTIONS.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </Field>

        <Field label="AI tools you currently use">
          <div className="flex flex-wrap gap-2">
            {AI_TOOLS_OPTIONS.map((tool) => (
              <button
                key={tool}
                type="button"
                onClick={() => toggleTool(tool)}
                className={`px-3 py-1.5 text-sm rounded-full border transition-colors ${
                  currentAiTools.includes(tool)
                    ? "bg-primary text-primary-foreground border-primary"
                    : "bg-background text-foreground border-border hover:border-primary/50"
                }`}
              >
                {tool}
              </button>
            ))}
          </div>
        </Field>

        <Field label="What frustrates you most about AI in education? (optional)">
          <input
            type="text"
            value={primaryFrustration}
            onChange={(e) => setPrimaryFrustration(e.target.value)}
            placeholder="e.g. marking workload, keeping up with new tools..."
            className="w-full p-2 rounded-lg border border-border bg-background text-foreground"
          />
        </Field>

        <Field label="Years of experience">
          <select
            value={yearsOfExperience}
            onChange={(e) => setYearsOfExperience(e.target.value)}
            className="w-full p-2 rounded-lg border border-border bg-background text-foreground"
          >
            <option value="">Select...</option>
            {EXPERIENCE_OPTIONS.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </Field>

        <Field label="Management responsibility">
          <select
            value={managementResponsibility}
            onChange={(e) => setManagementResponsibility(e.target.value)}
            className="w-full p-2 rounded-lg border border-border bg-background text-foreground"
          >
            <option value="">Select...</option>
            {MANAGEMENT_OPTIONS.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </Field>
      </div>

      <button
        onClick={handleSubmit}
        disabled={saving}
        className="mt-8 w-full py-3 rounded-lg bg-primary text-primary-foreground font-medium hover:bg-primary/90 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
      >
        {saving ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Saving...
          </>
        ) : (
          "Continue to assessment"
        )}
      </button>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-sm font-medium text-foreground mb-1.5">{label}</label>
      {children}
    </div>
  );
}
