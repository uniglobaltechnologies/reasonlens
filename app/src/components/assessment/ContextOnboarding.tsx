import { type ReactNode, useMemo, useState } from "react";
import { Loader2 } from "lucide-react";
import { apiPost } from "@/lib/api";

export interface AssessmentContext {
  subject_area?: string | null;
  institution_size?: string | null;
  institution_type?: string | null;
  institution_level?: string | null;
  region?: string | null;
  funding_model?: string | null;
  respondent_role?: string | null;
  respondent_institutional_visibility?: string | null;
  digital_infrastructure_baseline?: string | null;
  current_ai_tools?: string[] | null;
  primary_frustration?: string | null;
  years_of_experience?: string | null;
  management_responsibility?: string | null;
}

interface ContextOnboardingProps {
  frameworkId: string;
  initialContext?: AssessmentContext | null;
  onComplete: () => void;
  guestMode?: boolean;
}

const SUBJECT_OPTIONS = [
  "Computing",
  "English",
  "Maths",
  "Sciences",
  "Languages",
  "Art & Design",
  "History",
  "Geography",
  "PE",
  "Music",
  "Business Studies",
  "Social Sciences",
  "Engineering",
  "Health & Social Care",
  "Other",
];

const INDIVIDUAL_INSTITUTION_LEVEL_OPTIONS = [
  "Primary",
  "Secondary",
  "Further Education",
  "Higher Education",
  "Other",
];

const INDIVIDUAL_INSTITUTION_TYPE_OPTIONS = [
  "State school",
  "Academy / Trust",
  "Independent",
  "University",
  "FE College",
  "Other",
];

const INDIVIDUAL_REGION_OPTIONS = [
  "UK",
  "EU",
  "US",
  "Asia-Pacific",
  "Africa",
  "LATAM",
  "MENA",
  "Other",
];

const AI_TOOLS_OPTIONS = [
  "ChatGPT",
  "Microsoft Copilot",
  "Google Gemini",
  "Claude",
  "Midjourney",
  "DALL-E",
  "Perplexity",
  "NotebookLM",
  "Other",
  "None",
];

const EXPERIENCE_OPTIONS = [
  "0-2 years",
  "3-5 years",
  "6-10 years",
  "11-20 years",
  "20+ years",
];

const MANAGEMENT_OPTIONS = [
  "No management responsibility",
  "Middle leader",
  "Senior leader",
  "Executive / Governor",
];

const THE_INSTITUTION_SIZE_OPTIONS = [
  "small (<5,000 students)",
  "medium (5,000-15,000 students)",
  "large (15,000+ students)",
];

const THE_INSTITUTION_TYPE_OPTIONS = [
  "research-intensive",
  "teaching-focused",
  "polytechnic",
  "specialist",
  "multi-campus",
];

const THE_REGION_OPTIONS = [
  "UK",
  "EU",
  "US",
  "Asia-Pacific",
  "Africa",
  "LATAM",
  "MENA",
];

const THE_FUNDING_MODEL_OPTIONS = ["public", "private", "mixed"];

const THE_RESPONDENT_ROLE_OPTIONS = [
  "senior_leadership",
  "faculty_dean",
  "department_head",
  "professional_services_director",
  "IT_leadership",
  "academic_staff",
];

const THE_VISIBILITY_OPTIONS = [
  "institution_wide",
  "faculty_level",
  "department_level",
];

const THE_INFRASTRUCTURE_OPTIONS = ["limited", "moderate", "advanced"];

export default function ContextOnboarding({
  frameworkId,
  initialContext,
  onComplete,
  guestMode,
}: ContextOnboardingProps) {
  const isInstitutional = frameworkId === "maturity-the";

  const [subjectArea, setSubjectArea] = useState(initialContext?.subject_area ?? "");
  const [institutionSize, setInstitutionSize] = useState(
    initialContext?.institution_size ?? ""
  );
  const [institutionLevel, setInstitutionLevel] = useState(
    initialContext?.institution_level ?? ""
  );
  const [institutionType, setInstitutionType] = useState(
    initialContext?.institution_type ?? ""
  );
  const [region, setRegion] = useState(initialContext?.region ?? "");
  const [fundingModel, setFundingModel] = useState(
    initialContext?.funding_model ?? ""
  );
  const [respondentRole, setRespondentRole] = useState(
    initialContext?.respondent_role ?? ""
  );
  const [institutionalVisibility, setInstitutionalVisibility] = useState(
    initialContext?.respondent_institutional_visibility ?? ""
  );
  const [digitalInfrastructureBaseline, setDigitalInfrastructureBaseline] =
    useState(initialContext?.digital_infrastructure_baseline ?? "");
  const [currentAiTools, setCurrentAiTools] = useState<string[]>(
    initialContext?.current_ai_tools ?? []
  );
  const [primaryFrustration, setPrimaryFrustration] = useState(
    initialContext?.primary_frustration ?? ""
  );
  const [yearsOfExperience, setYearsOfExperience] = useState(
    initialContext?.years_of_experience ?? ""
  );
  const [managementResponsibility, setManagementResponsibility] = useState(
    initialContext?.management_responsibility ?? ""
  );
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState("");

  const missingInstitutionalFields = useMemo(() => {
    if (!isInstitutional) return [];

    return [
      !institutionSize ? "institution size" : null,
      !institutionType ? "institution type" : null,
      !region ? "region" : null,
      !fundingModel ? "funding model" : null,
      !respondentRole ? "respondent role" : null,
      !institutionalVisibility ? "institutional visibility" : null,
      !digitalInfrastructureBaseline ? "digital infrastructure baseline" : null,
    ].filter(Boolean) as string[];
  }, [
    digitalInfrastructureBaseline,
    fundingModel,
    institutionSize,
    institutionType,
    institutionalVisibility,
    isInstitutional,
    region,
    respondentRole,
  ]);

  const toggleTool = (tool: string) => {
    setCurrentAiTools((prev) =>
      prev.includes(tool) ? prev.filter((value) => value !== tool) : [...prev, tool]
    );
  };

  const handleSubmit = async () => {
    setSaving(true);
    setSaveError("");

    try {
      await apiPost(`/user-assessment-context${guestMode ? "?guest=true" : ""}`, {
        subject_area: isInstitutional ? null : subjectArea || null,
        institution_size: isInstitutional ? institutionSize || null : null,
        institution_type: institutionType || null,
        institution_level: isInstitutional ? null : institutionLevel || null,
        region: region || null,
        funding_model: isInstitutional ? fundingModel || null : null,
        respondent_role: isInstitutional ? respondentRole || null : null,
        respondent_institutional_visibility: isInstitutional
          ? institutionalVisibility || null
          : null,
        digital_infrastructure_baseline: isInstitutional
          ? digitalInfrastructureBaseline || null
          : null,
        current_ai_tools: isInstitutional
          ? null
          : currentAiTools.length > 0
            ? currentAiTools
            : null,
        primary_frustration: isInstitutional ? null : primaryFrustration || null,
        years_of_experience: isInstitutional ? null : yearsOfExperience || null,
        management_responsibility: isInstitutional
          ? null
          : managementResponsibility || null,
      });
      onComplete();
    } catch (err: any) {
      console.error("Failed to save context:", err);
      setSaveError(err.message || "Failed to save your information. Please try again.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <h3 className="text-xl font-semibold text-foreground mb-2">
        {isInstitutional ? "Before we start: your institutional context" : "Before we start: a little about you"}
      </h3>
      <p className="text-sm text-muted-foreground mb-6">
        {isInstitutional
          ? "The THE scenario battery is institution-level. These fields let us validate that context and preserve a defensible session snapshot."
          : "This helps us personalise your assessment scenarios. You can update these later."}
      </p>

      <div className="space-y-5">
        {isInstitutional ? (
          <>
            <Field label="Institution size">
              <select
                value={institutionSize}
                onChange={(e) => setInstitutionSize(e.target.value)}
                className="w-full p-2 rounded-lg border border-border bg-background text-foreground"
              >
                <option value="">Select...</option>
                {THE_INSTITUTION_SIZE_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
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
                {THE_INSTITUTION_TYPE_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </Field>

            <Field label="Region">
              <select
                value={region}
                onChange={(e) => setRegion(e.target.value)}
                className="w-full p-2 rounded-lg border border-border bg-background text-foreground"
              >
                <option value="">Select...</option>
                {THE_REGION_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </Field>

            <Field label="Funding model">
              <select
                value={fundingModel}
                onChange={(e) => setFundingModel(e.target.value)}
                className="w-full p-2 rounded-lg border border-border bg-background text-foreground"
              >
                <option value="">Select...</option>
                {THE_FUNDING_MODEL_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </Field>

            <Field label="Respondent role">
              <select
                value={respondentRole}
                onChange={(e) => setRespondentRole(e.target.value)}
                className="w-full p-2 rounded-lg border border-border bg-background text-foreground"
              >
                <option value="">Select...</option>
                {THE_RESPONDENT_ROLE_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </Field>

            <Field label="Institutional visibility">
              <select
                value={institutionalVisibility}
                onChange={(e) => setInstitutionalVisibility(e.target.value)}
                className="w-full p-2 rounded-lg border border-border bg-background text-foreground"
              >
                <option value="">Select...</option>
                {THE_VISIBILITY_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option.replace(/_/g, " ")}
                  </option>
                ))}
              </select>
            </Field>

            <Field label="Digital infrastructure baseline">
              <select
                value={digitalInfrastructureBaseline}
                onChange={(e) => setDigitalInfrastructureBaseline(e.target.value)}
                className="w-full p-2 rounded-lg border border-border bg-background text-foreground"
              >
                <option value="">Select...</option>
                {THE_INFRASTRUCTURE_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </Field>
          </>
        ) : (
          <>
            <Field label="Subject area">
              <select
                value={subjectArea}
                onChange={(e) => setSubjectArea(e.target.value)}
                className="w-full p-2 rounded-lg border border-border bg-background text-foreground"
              >
                <option value="">Select...</option>
                {SUBJECT_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
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
                {INDIVIDUAL_INSTITUTION_LEVEL_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
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
                {INDIVIDUAL_INSTITUTION_TYPE_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </Field>

            <Field label="Region">
              <select
                value={region}
                onChange={(e) => setRegion(e.target.value)}
                className="w-full p-2 rounded-lg border border-border bg-background text-foreground"
              >
                <option value="">Select...</option>
                {INDIVIDUAL_REGION_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
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
                {EXPERIENCE_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
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
                {MANAGEMENT_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </Field>
          </>
        )}
      </div>

      {isInstitutional && missingInstitutionalFields.length > 0 && (
        <p className="mt-4 text-sm text-muted-foreground">
          Required before continuing: {missingInstitutionalFields.join(", ")}.
        </p>
      )}

      <button
        onClick={handleSubmit}
        disabled={saving || missingInstitutionalFields.length > 0}
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

      {saveError && (
        <p className="mt-3 text-sm text-destructive text-center">{saveError}</p>
      )}
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <label className="block text-sm font-medium text-foreground mb-1.5">
        {label}
      </label>
      {children}
    </div>
  );
}
