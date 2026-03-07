// ============================================================
// Shared Framework Types
// ============================================================

export type CompetencyLevel = "acquire" | "deepen" | "create";

export interface AssessmentQuestion {
  id: string;
  dimension: string;
  question: string;
  options: {
    value: string;
    label: string;
    level: CompetencyLevel;
  }[];
}

export interface Indicator {
  id: string;
  description: string;
  assessmentCriteria?: string;
}

export interface CurricularGoal {
  id: string;
  description: string;
}

export interface ContextualActivity {
  id: string;
  name: string;
  description: string;
}

export interface KeyPrinciple {
  id: string;
  name: string;
  description?: string;
  tenets?: string[];
}

export interface Level {
  id: string;
  name: string;
  description: string;
  order: number;
  target?: string;
  curricularGoals?: CurricularGoal[];
  contextualActivities?: ContextualActivity[];
  learningEnvironments?: string[];
  indicators: Indicator[];
}

export interface FrameworkDimension {
  id: string;
  name: string;
  description: string;
  order: number;
  icon: string;
  color: string;
  parentDimensionId?: string;
  levels: Level[];
}

export type FrameworkScope = "individual" | "institutional" | "cross-cutting";

export type FrameworkSource =
  | "UNESCO"
  | "QS"
  | "THE"
  | "JISC"
  | "OECD"
  | "DEC"
  | "EU"
  | "ISTE"
  | "AILit";

export type FrameworkType =
  | "competency"
  | "capability"
  | "maturity"
  | "policy"
  | "indicators";

export type CompatibilityCategory = "complementary" | "overlapping" | "redundant";

export interface CompatibilityEntry {
  frameworkId: string;
  category: CompatibilityCategory;
  overlapSeverity?: "low" | "low-medium" | "medium" | "medium-high" | "high";
  overlapAreas?: string[];
  warningText?: string;
}

export type SourceFidelity = "official" | "synthesized";

export interface Framework {
  id: string;
  name: string;
  shortName: string;
  description: string;
  type: FrameworkType;
  scope: FrameworkScope;
  source: FrameworkSource;
  path: string;
  icon: string;
  color: string;
  badgeLabel: string;
  targetAudience: string[];
  overview: string;
  keyDimensions: FrameworkDimension[];
  keyPrinciples?: KeyPrinciple[];
  metadata: Record<string, unknown>;
  useCases: string[];
  crossReferences: string[];
  assessmentQuestions: AssessmentQuestion[];
  assessmentTitle: string;
  assessmentDescription: string;
  showInQuiz: boolean;
  showInDashboard: boolean;
  showInLanding: boolean;
  isBackgroundFramework: boolean;
  // New fields
  compatibility: CompatibilityEntry[];
  sourceFidelity: SourceFidelity;
  estimatedAssessmentMinutes: number;
  region?: "uk" | "eu" | "us" | "international";
}

// ── Shared level templates ──────────────────────

export const ACQUIRE_DEEPEN_CREATE: Omit<Level, "indicators">[] = [
  { id: "acquire", name: "Acquire", description: "Build foundational knowledge and awareness", order: 1 },
  { id: "deepen", name: "Deepen", description: "Apply and integrate skills in practice", order: 2 },
  { id: "create", name: "Create", description: "Innovate, lead, and mentor others", order: 3 },
];

export const STUDENT_LEVELS: Omit<Level, "indicators">[] = [
  { id: "foundational", name: "Foundational", description: "Basic awareness and guided exploration", order: 1 },
  { id: "intermediate", name: "Intermediate", description: "Independent application and critical analysis", order: 2 },
  { id: "advanced", name: "Advanced", description: "Creation, innovation, and leadership", order: 3 },
];

export const BDC_LEVELS: Omit<Level, "indicators">[] = [
  { id: "discovery", name: "Discovery", description: "Awareness and initial exploration", order: 1 },
  { id: "development", name: "Development", description: "Building skills with support", order: 2 },
  { id: "established", name: "Established", description: "Confident independent practice", order: 3 },
  { id: "leading", name: "Leading", description: "Guiding and mentoring others", order: 4 },
  { id: "strategic", name: "Strategic", description: "Shaping strategy and culture", order: 5 },
];

export const DIGCOMP_LEVELS: Omit<Level, "indicators">[] = [
  { id: "level-basic", name: "Basic", description: "Remembering and implementing simple tasks with guidance as needed", order: 1 },
  { id: "level-intermediate", name: "Intermediate", description: "Identifying and implementing well-defined tasks and solving well-defined problems autonomously", order: 2 },
  { id: "level-advanced", name: "Advanced", description: "Assessing and applying solutions to complex tasks autonomously, guiding others", order: 3 },
  { id: "level-highly-advanced", name: "Highly Advanced", description: "Assessing and resolving highly complex or specialised problems, creating new solutions, leading others", order: 4 },
];

export const JISC_AI_LEVELS: Omit<Level, "indicators">[] = [
  { id: "exploring", name: "Exploring", description: "Ad-hoc experimentation with AI", order: 1 },
  { id: "developing", name: "Developing", description: "Structured pilots and planning", order: 2 },
  { id: "defined", name: "Defined", description: "Formal AI strategy and governance", order: 3 },
  { id: "managed", name: "Managed", description: "Measured and optimised AI deployment", order: 4 },
  { id: "optimising", name: "Optimising", description: "Continuous innovation and sector leadership", order: 5 },
];
