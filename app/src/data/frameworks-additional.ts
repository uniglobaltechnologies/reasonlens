// ============================================================
// Additional Frameworks (15 new frameworks)
// Populated from official source data where available
// ============================================================

import type { Framework, FrameworkDimension, Level } from "./framework-types";
import { DIGCOMP_LEVELS, JISC_AI_LEVELS } from "./framework-types";

// ── BDC source JSON imports ────────────────────
import bdcIndividualSrc from "./bdc-individual.json";
import bdcTeacherHeSrc from "./bdc-teacher-he.json";
import bdcResearcherSrc from "./bdc-researcher.json";
import bdcProfessionalServicesSrc from "./bdc-professional-services.json";
import bdcLearningTechnologySrc from "./bdc-learning-technology.json";
import bdcDigitalLeaderSrc from "./bdc-digital-leader.json";
import bdcEducationalDeveloperSrc from "./bdc-educational-developer.json";

// ── BDC dimension icon/color mapping ───────────
const BDC_DIM_STYLE: Record<string, { icon: string; color: string }> = {
  "bdc-proficiency-productivity": { icon: "Monitor", color: "text-blue-600" },
  "bdc-creation-innovation": { icon: "Lightbulb", color: "text-amber-600" },
  "bdc-learning-development": { icon: "GraduationCap", color: "text-rose-600" },
  "bdc-literacies": { icon: "Search", color: "text-purple-600" },
  "bdc-communication-collaboration": { icon: "MessageSquare", color: "text-emerald-600" },
  "bdc-identity-wellbeing": { icon: "Shield", color: "text-cyan-600" },
};

// BDC JSON files use 5 AI maturity levels, but the correct JISC BDC individual
// model uses 3 Discovery Tool levels: Developing / Capable / Proficient.
// Mapping: level-1 + level-2 → Developing, level-3 → Capable, level-4 + level-5 → Proficient
const BDC_LEVEL_MERGE: { id: string; name: string; description: string; order: number; sourceIds: string[] }[] = [
  { id: "developing", name: "Developing", description: "Awareness, initial exploration and guided experimentation with AI tools", order: 1, sourceIds: ["level-1", "level-2"] },
  { id: "capable", name: "Capable", description: "Confident, systematic and responsible AI-augmented professional practice", order: 2, sourceIds: ["level-3"] },
  { id: "proficient", name: "Proficient", description: "Leading AI integration, mentoring others and shaping institutional strategy", order: 3, sourceIds: ["level-4", "level-5"] },
];

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function buildBdcDimensions(source: any, frameworkId: string): FrameworkDimension[] {
  const aspects: any[] = source.aspects;
  const blocks: any[] = source.competency_blocks;

  return aspects.map((aspect: any, i: number) => {
    const style = BDC_DIM_STYLE[aspect.id] ?? { icon: "Circle", color: "text-gray-600" };
    return {
      id: `${frameworkId}-${aspect.id}`,
      name: aspect.name,
      description: aspect.description,
      order: i + 1,
      icon: style.icon,
      color: style.color,
      levels: BDC_LEVEL_MERGE.map((merged) => {
        // Collect indicators from all source levels that merge into this level
        const indicators: { id: string; description: string }[] = [];
        const curricularGoals: { id: string; description: string }[] = [];
        const contextualActivities: { id: string; name: string; description: string }[] = [];
        for (const srcId of merged.sourceIds) {
          const block = blocks.find(
            (b: any) => b.aspect_id === aspect.id && b.level_id === srcId
          );
          if (block) {
            (block.curricular_goals ?? []).forEach((g: string, gi: number) => {
              indicators.push({ id: `${frameworkId}-${aspect.id}-${merged.id}-cg${srcId}-${gi}`, description: g });
              curricularGoals.push({
                id: `${frameworkId}-${aspect.id}-${merged.id}-cg${srcId}-${gi}`,
                description: g,
              });
            });
            (block.learning_objectives ?? []).forEach((lo: string, li: number) => {
              indicators.push({ id: `${frameworkId}-${aspect.id}-${merged.id}-lo${srcId}-${li}`, description: lo });
            });
            (block.contextual_activities ?? []).forEach((activity: string, ai: number) =>
              contextualActivities.push({
                id: `${frameworkId}-${aspect.id}-${merged.id}-ca${srcId}-${ai}`,
                name: activity,
                description: activity,
              })
            );
          }
        }
        if (indicators.length === 0) {
          indicators.push({
            id: `${frameworkId}-${aspect.id}-${merged.id}-f`,
            description: `${aspect.name} at ${merged.name} level`,
          });
        }
        return {
          id: `${frameworkId}-${aspect.id}-${merged.id}`,
          name: merged.name,
          description: merged.description,
          order: merged.order,
          indicators,
          curricularGoals: curricularGoals.length > 0 ? curricularGoals : undefined,
          contextualActivities: contextualActivities.length > 0 ? contextualActivities : undefined,
        };
      }),
    };
  });
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function buildBdcAssessmentQuestions(source: any, frameworkId: string) {
  const aspects: any[] = source.aspects;
  return aspects.map((aspect: any, i: number) => ({
    id: `${frameworkId}-q${i + 1}`,
    dimension: aspect.name,
    question: `How would you rate your capability in ${aspect.name.toLowerCase()}?`,
    options: [
      { value: `${frameworkId}-${i}-a`, label: "I'm just starting to explore this area", level: "developing" as const },
      { value: `${frameworkId}-${i}-b`, label: "I'm developing confidence and can work independently", level: "capable" as const },
      { value: `${frameworkId}-${i}-c`, label: "I lead and mentor others in this area", level: "proficient" as const },
    ],
  }));
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function makeBdcFramework(source: any): Framework {
  const fw = source.framework;
  const aspects: any[] = source.aspects;
  const blocks: any[] = source.competency_blocks;

  // Count real indicators
  const totalBlocks = blocks.length;
  const totalIndicators = blocks.reduce(
    (sum: number, b: any) => sum + (b.curricular_goals?.length ?? 0) + (b.learning_objectives?.length ?? 0),
    0,
  );

  return {
    id: fw.id,
    name: fw.name,
    shortName: fw.short_name,
    description: `${fw.short_name} role profile: ${aspects.length} capability areas × 3 capability levels (Developing → Capable → Proficient) with ${totalIndicators} indicators`,
    type: "capability",
    scope: fw.scope,
    source: "JISC",
    path: `/frameworks/${fw.id}`,
    icon: "Users",
    color: "text-teal-600",
    badgeLabel: "JISC BDC",
    targetAudience: fw.target_audience,
    overview: `${fw.name} provides a structured approach to developing digital capabilities. It covers ${aspects.length} capability areas across 3 capability levels (Developing → Capable → Proficient), with ${totalBlocks} competency blocks containing ${totalIndicators} indicators derived from curricular goals and learning objectives. ${fw.key_principles?.[fw.key_principles.length - 1] ?? ""}`,
    keyDimensions: buildBdcDimensions(source, fw.id),
    keyPrinciples: fw.key_principles?.map((p: string, i: number) => ({
      id: `${fw.id}-p${i}`,
      name: p.slice(0, 60),
      description: p,
    })),
    metadata: {
      region: "UK",
      roleProfile: fw.short_name,
      version: fw.version,
      licence: fw.licence,
      sourcePdfs: fw.source_pdfs,
      totalAspects: aspects.length,
      totalLevels: 3,
      totalBlocks,
      totalIndicators,
      crossReferenceFrameworks: fw.cross_reference_frameworks,
      roleSpecificContext: fw.role_specific_context,
    },
    useCases: [
      `Self-assessment for ${fw.target_audience.join(", ")}`,
      "Professional development planning",
      "Identifying digital capability gaps",
      "Benchmarking against peers",
      "Mapping to JISC AI Maturity levels",
    ],
    crossReferences: ["maturity-jisc-ai"],
    assessmentQuestions: buildBdcAssessmentQuestions(source, fw.id),
    assessmentTitle: `${fw.short_name} Digital Capability Assessment`,
    assessmentDescription: `Assess your digital capabilities across ${aspects.length} areas`,
    showInQuiz: false,
    showInDashboard: true,
    showInLanding: false,
    isBackgroundFramework: false,
    compatibility: [],
    sourceFidelity: "official",
    estimatedAssessmentMinutes: 60,
    region: "uk",
  };
}

// ── 7 BDC Role Profiles (built from source JSON) ────────

export const bdcIndividual = makeBdcFramework(bdcIndividualSrc);
export const bdcTeacherHe = makeBdcFramework(bdcTeacherHeSrc);
export const bdcResearcher = makeBdcFramework(bdcResearcherSrc);
export const bdcProfessionalServices = makeBdcFramework(bdcProfessionalServicesSrc);
export const bdcLearningTechnology = makeBdcFramework(bdcLearningTechnologySrc);
export const bdcDigitalLeader = makeBdcFramework(bdcDigitalLeaderSrc);
export const bdcEducationalDeveloper = makeBdcFramework(bdcEducationalDeveloperSrc);

// ── AILit Framework (OECD + European Commission, 2025) ────────────────────────────

// Knowledge, Skills, and Attitudes are cross-cutting — referenced by competences
const AILIT_KNOWLEDGE: { id: string; category: string; description: string }[] = [
  { id: "K1.1", category: "The Nature of AI", description: "AI systems use algorithms combining step-by-step procedures with statistical inferences to process data, detect patterns, and generate probable outputs" },
  { id: "K1.2", category: "The Nature of AI", description: "Machines 'learn' by inferring how to generate outputs such as predictions, content, and recommendations with varying levels of autonomy and adaptiveness" },
  { id: "K1.3", category: "The Nature of AI", description: "Generative AI uses probabilities to generate human-like outputs across modalities but lacks authentic understanding and intent" },
  { id: "K1.4", category: "The Nature of AI", description: "AI systems operate differently depending on their purpose — to create, predict, recommend, or respond" },
  { id: "K2.1", category: "AI Reflects Human Choices", description: "Building AI relies on humans to design algorithms, collect/label data, and moderate content, reflecting choices shaped by unequal global conditions" },
  { id: "K2.2", category: "AI Reflects Human Choices", description: "AI is trained on vast datasets from publicly available info, user-generated content, curated databases, and real-world sensor data" },
  { id: "K2.3", category: "AI Reflects Human Choices", description: "AI systems gather new data from user interactions; outputs may be directly influenced by inputs in real time" },
  { id: "K2.4", category: "AI Reflects Human Choices", description: "AI systems are trained to identify patterns among data elements that humans have selected, categorized, and prioritized" },
  { id: "K2.5", category: "AI Reflects Human Choices", description: "Bias inherently exists in AI systems, reflecting societal biases in training data or algorithm design" },
  { id: "K3.1", category: "AI Reshapes Work", description: "AI automates structured tasks, augments decision-making, and transforms industries, requiring adaptation and reskilling" },
  { id: "K3.2", category: "AI Reshapes Work", description: "AI integration requires determining which tasks suit machines vs human intervention" },
  { id: "K3.3", category: "AI Reshapes Work", description: "Humans must be responsible for decisions reflecting human judgment and ethical considerations" },
  { id: "K4.1", category: "AI Capabilities & Limitations", description: "AI excels at pattern recognition and automation but lacks emotions, ethical reasoning, context, and originality" },
  { id: "K4.2", category: "AI Capabilities & Limitations", description: "AI requires vast computing power and data, consuming energy and increasing carbon emissions" },
  { id: "K4.3", category: "AI Capabilities & Limitations", description: "Generative AI can make it difficult to distinguish fact from fabrication, increasing potential for misinformation and deepfakes" },
  { id: "K5.1", category: "AI's Role in Society", description: "AI plays an increasingly prevalent role in decision-making impacting humans from hiring to healthcare to criminal justice" },
  { id: "K5.2", category: "AI's Role in Society", description: "AI systems must be understood, audited, and regulated to ensure more benefits than harm" },
  { id: "K5.3", category: "AI's Role in Society", description: "Generative AI creates content based on existing materials including copyright-protected work, raising questions about authenticity and ownership" },
  { id: "K5.4", category: "AI's Role in Society", description: "Ethical AI design encompasses fairness, transparency, explainability, accountability, privacy, and legal compliance" },
];

const AILIT_SKILLS = [
  "Critical Thinking", "Creativity", "Computational Thinking",
  "Self and Social Awareness", "Collaboration", "Communication", "Problem Solving",
];

const AILIT_ATTITUDES = ["Responsible", "Curious", "Innovative", "Adaptable", "Empathetic"];

const AILIT_COMPETENCES: {
  id: string; dimId: string; order: number; name: string; description: string;
  knowledgeRefs: string[]; skillRefs: string[]; attitudeRefs: string[];
  primaryScenario: string; secondaryScenario: string;
}[] = [
  { id: "engage-1", dimId: "engaging-with-ai", order: 1, name: "Recognize AI's role and influence in different contexts", description: "Identify AI in everyday tools and reflect on how it influences choices, learning, and perceptions.", knowledgeRefs: ["K1.4", "K5.1"], skillRefs: ["Self and Social Awareness"], attitudeRefs: ["Curious", "Responsible"], primaryScenario: "List familiar digital interactions and discuss if and how each uses AI.", secondaryScenario: "Explore how an online math platform uses real-time data to present content at different difficulty levels." },
  { id: "engage-2", dimId: "engaging-with-ai", order: 2, name: "Evaluate whether AI outputs should be accepted, revised, or rejected", description: "Critically assess accuracy and fairness of AI-generated content, deciding whether to trust, modify, or override.", knowledgeRefs: ["K4.1", "K4.3"], skillRefs: ["Critical Thinking"], attitudeRefs: ["Responsible"], primaryScenario: "Compare an AI tool's step-by-step math solution to a learner's explanation.", secondaryScenario: "Prompt a language model with historical questions and evaluate accuracy by cross-referencing reliable sources." },
  { id: "engage-3", dimId: "engaging-with-ai", order: 3, name: "Examine how predictive AI systems provide recommendations that can inform and limit perspectives", description: "Explore how AI uses data patterns to offer suggestions and how recommendations may reinforce narrow viewpoints.", knowledgeRefs: ["K1.1", "K4.3"], skillRefs: ["Self and Social Awareness"], attitudeRefs: ["Curious"], primaryScenario: "Count by 2s, 5s, and 10s to introduce pattern recognition, then explore how AI generates recommendations.", secondaryScenario: "Examine how social media algorithms contribute to spreading disinformation about public health." },
  { id: "engage-4", dimId: "engaging-with-ai", order: 4, name: "Explain how AI could be used to amplify societal biases", description: "Investigate how AI reflects human decisions and data, identifying how bias can lead to unfair outcomes.", knowledgeRefs: ["K2.1", "K2.5"], skillRefs: ["Critical Thinking", "Self and Social Awareness", "Problem Solving"], attitudeRefs: ["Empathetic", "Responsible"], primaryScenario: "Sort characters from stories into categories and discuss how grouping people can be useful or unfair.", secondaryScenario: "Examine how an AI facial recognition system was trained and evaluate potential sources of bias." },
  { id: "engage-5", dimId: "engaging-with-ai", order: 5, name: "Describe how AI systems consume energy and natural resources", description: "Explore environmental impact of AI including energy and data infrastructure.", knowledgeRefs: ["K4.2"], skillRefs: ["Self and Social Awareness"], attitudeRefs: ["Responsible"], primaryScenario: "Create an infographic illustrating AI's environmental impacts.", secondaryScenario: "Compare AI's environmental costs with reduction efforts, then debate responsible use scenarios." },
  { id: "engage-6", dimId: "engaging-with-ai", order: 6, name: "Analyze how well AI use aligns with ethical principles and human values", description: "Assess whether AI use in given situations supports fairness, transparency, and privacy.", knowledgeRefs: ["K1.4", "K3.3", "K5.4"], skillRefs: ["Self and Social Awareness", "Critical Thinking", "Problem Solving"], attitudeRefs: ["Responsible"], primaryScenario: "Evaluate if AI is used kindly, fairly, and respectfully in multiple scenarios.", secondaryScenario: "Use an AI writing assistant and reflect on whether suggestions supported authentic voice." },
  { id: "engage-7", dimId: "engaging-with-ai", order: 7, name: "Connect AI's social and ethical impacts to its technical capabilities and limitations", description: "Explore how AI strengths and weaknesses affect societal use and real-world impact.", knowledgeRefs: ["K2.1", "K5.2"], skillRefs: ["Self and Social Awareness", "Problem Solving"], attitudeRefs: ["Curious", "Empathetic", "Responsible"], primaryScenario: "Discuss why a voice assistant sometimes doesn't understand commands.", secondaryScenario: "Investigate how predictive AI calculates credit scores, exploring bias and mathematical inequality." },
  { id: "create-1", dimId: "creating-with-ai", order: 1, name: "Use AI to explore new perspectives that build upon original ideas", description: "Experiment with AI to expand thinking and generate new ideas while staying accountable.", knowledgeRefs: ["K4.1"], skillRefs: ["Creativity"], attitudeRefs: ["Innovative", "Adaptable"], primaryScenario: "Evaluate AI-generated images to create story settings, then write stories inspired by unexpected results.", secondaryScenario: "Use AI to develop counterarguments for a class debate." },
  { id: "create-2", dimId: "creating-with-ai", order: 2, name: "Visualize, prototype, and combine ideas using different AI systems", description: "Try AI tools across formats to explore and refine ideas into meaningful products.", knowledgeRefs: ["K1.4"], skillRefs: ["Collaboration", "Creativity"], attitudeRefs: ["Curious", "Adaptable"], primaryScenario: "Use an AI music tool to create a song about a season, experimenting with moods and instruments.", secondaryScenario: "Use AI tools across text, graphics, music for a public awareness campaign." },
  { id: "create-3", dimId: "creating-with-ai", order: 3, name: "Collaborate with generative AI to elicit feedback, refine results, and reflect", description: "Engage iteratively with AI through prompts, refinement, and reflection on thinking.", knowledgeRefs: ["K2.3"], skillRefs: ["Computational Thinking", "Creativity"], attitudeRefs: ["Innovative", "Adaptable"], primaryScenario: "Use an AI writing tool to improve a class story, discussing how ideas changed.", secondaryScenario: "Use an AI coding assistant to fix errors and modify code, reflecting on the process." },
  { id: "create-4", dimId: "creating-with-ai", order: 4, name: "Analyze how AI can safeguard or violate content authenticity and IP", description: "Explore how AI-generated content may replicate existing work and consider fairness of use.", knowledgeRefs: ["K5.3"], skillRefs: ["Problem Solving", "Self and Social Awareness"], attitudeRefs: ["Empathetic", "Responsible"], primaryScenario: "Compare student work to AI-generated poems and discuss originality and credit.", secondaryScenario: "Research how artists' styles appear in AI art and debate whether use is fair." },
  { id: "create-5", dimId: "creating-with-ai", order: 5, name: "Explain how AI performs tasks using precise language that avoids anthropomorphism", description: "Describe AI in realistic, accurate terms without suggesting human feelings or understanding.", knowledgeRefs: ["K1.3", "K1.4"], skillRefs: ["Communication"], attitudeRefs: ["Responsible"], primaryScenario: "Compare human art with AI-generated art and discuss expression vs pattern-based generation.", secondaryScenario: "Describe how generative AI creates a song from prompts and training data without assigning intent." },
  { id: "manage-1", dimId: "managing-ai", order: 1, name: "Decide whether to use AI based on the nature of the task", description: "Assess whether AI is appropriate considering complexity, need for human judgment, and ethics.", knowledgeRefs: ["K4.1", "K5.4"], skillRefs: ["Problem Solving", "Computational Thinking"], attitudeRefs: ["Responsible", "Innovative"], primaryScenario: "Consider everyday tasks and assess when AI use is appropriate vs requiring human judgment.", secondaryScenario: "Determine whether AI should be used for specific tasks based on alignment with learning objectives." },
  { id: "manage-2", dimId: "managing-ai", order: 2, name: "Decompose a problem based on AI and human capabilities and limitations", description: "Break down tasks and distribute between AI and humans based on respective strengths.", knowledgeRefs: ["K4.1"], skillRefs: ["Collaboration", "Computational Thinking", "Problem Solving"], attitudeRefs: ["Innovative", "Adaptable"], primaryScenario: "Use AI to brainstorm science fair ideas while students vote, design experiments, and interpret results.", secondaryScenario: "Use AI to summarize historical sources while students assess context, detect bias, and make interpretations." },
  { id: "manage-3", dimId: "managing-ai", order: 3, name: "Direct generative AI with specific instructions, context, and evaluation criteria", description: "Practice prompt engineering with clear, structured inputs.", knowledgeRefs: ["K1.3", "K2.3"], skillRefs: ["Collaboration", "Computational Thinking"], attitudeRefs: ["Innovative", "Adaptable"], primaryScenario: "Construct a prompt another student could use to draw a poster including topic and quality criteria.", secondaryScenario: "Engineer prompts for an AI chatbot as a debate partner, defining purpose, tone, and task." },
  { id: "manage-4", dimId: "managing-ai", order: 4, name: "Delegate tasks to AI to automate or augment human workflows", description: "Identify opportunities to offload repetitive tasks so people can focus on creativity and judgment.", knowledgeRefs: ["K3.1"], skillRefs: ["Collaboration", "Problem Solving"], attitudeRefs: ["Innovative"], primaryScenario: "Plan writing where AI helps with spelling/synonyms while learners focus on storytelling.", secondaryScenario: "Use AI to generate concept variations while team members evaluate, refine, and present." },
  { id: "manage-5", dimId: "managing-ai", order: 5, name: "Develop and communicate guidelines for responsible AI use", description: "Create guidelines that align with human values, fairness, and transparency.", knowledgeRefs: ["K5.4"], skillRefs: ["Communication", "Critical Thinking", "Self and Social Awareness"], attitudeRefs: ["Responsible", "Empathetic"], primaryScenario: "Create a classroom poster outlining fair AI use guidelines.", secondaryScenario: "Lead a workshop on AI tools sharing guidelines for honesty, IP respect, and critical thinking." },
  { id: "design-1", dimId: "designing-ai", order: 1, name: "Describe how AI can be designed to support community problem solutions", description: "Explore how AI can solve real-world problems by identifying needs and evaluating benefits and risks.", knowledgeRefs: ["K2.3", "K3.2"], skillRefs: ["Collaboration", "Problem Solving", "Self and Social Awareness"], attitudeRefs: ["Curious", "Innovative", "Responsible"], primaryScenario: "Develop a method for sorting healthy vs unhealthy snacks by gathering and labeling images.", secondaryScenario: "Propose how AI could recommend after-school activities exploring needed data and human input." },
  { id: "design-2", dimId: "designing-ai", order: 2, name: "Compare rule-based and data-driven AI systems", description: "Examine differences between fixed-rule systems and ML models to determine when each is useful.", knowledgeRefs: ["K1.2", "K1.4"], skillRefs: ["Computational Thinking", "Problem Solving"], attitudeRefs: ["Curious"], primaryScenario: "Compare organizing animals by physical characteristics vs by habitat/behavior.", secondaryScenario: "Program a simple chatbot with conditional logic and compare to ML-based system for the same task." },
  { id: "design-3", dimId: "designing-ai", order: 3, name: "Collect and curate data for AI training considering representation and impact", description: "Discover how data labeling, selection, and preparation affect model performance and impact on people.", knowledgeRefs: ["K1.2", "K2.2", "K2.4"], skillRefs: ["Computational Thinking", "Self and Social Awareness"], attitudeRefs: ["Innovative", "Responsible"], primaryScenario: "Label and sort building blocks by features, then create a decision tree for categorization.", secondaryScenario: "Train a basic AI model to recognize recyclable materials and describe data impact on performance." },
  { id: "design-4", dimId: "designing-ai", order: 4, name: "Evaluate AI systems using defined criteria, outcomes, and user feedback", description: "Set success criteria, test with various inputs, evaluate performance iteratively.", knowledgeRefs: ["K1.2", "K2.3"], skillRefs: ["Collaboration", "Computational Thinking"], attitudeRefs: ["Innovative", "Adaptable"], primaryScenario: "Use generative AI to create jokes, define quality criteria, rate responses, try new prompts.", secondaryScenario: "Test different AI models with the same datasets for the same task, then propose improvements." },
  { id: "design-5", dimId: "designing-ai", order: 5, name: "Describe an AI model's purpose, intended users, and limitations", description: "Describe model purpose, training data, and capabilities/limitations to help others understand.", knowledgeRefs: ["K1.2", "K2.1"], skillRefs: ["Communication", "Problem Solving", "Self and Social Awareness"], attitudeRefs: ["Curious", "Responsible"], primaryScenario: "Direct a classmate role-playing as a robot to sort items, observing how changing rules creates confusion.", secondaryScenario: "Create a model card summarizing how a ML model works, its data, uses, and limitations." },
];

// Build dimensions from competences
const ailitDimConfigs = [
  { id: "engaging-with-ai", name: "Engaging with AI", description: "Using AI as a tool to access content, information, or recommendations while critically evaluating accuracy and relevance.", icon: "Search", color: "text-blue-600" },
  { id: "creating-with-ai", name: "Creating with AI", description: "Collaborating with AI in creative or problem-solving processes while ensuring fairness, attribution, and responsible use.", icon: "Sparkles", color: "text-purple-600" },
  { id: "managing-ai", name: "Managing AI", description: "Intentionally choosing how AI can support and enhance human work, delegating tasks thoughtfully and maintaining human agency.", icon: "Settings", color: "text-emerald-600" },
  { id: "designing-ai", name: "Designing AI", description: "Understanding how AI works and connecting it to social and ethical impacts by shaping how AI systems function through hands-on exploration.", icon: "Wrench", color: "text-rose-600" },
];

const ailitDimensions: import("./framework-types").FrameworkDimension[] = ailitDimConfigs.map((dim, i) => {
  const dimCompetences = AILIT_COMPETENCES.filter((c) => c.dimId === dim.id);
  return {
    id: `ailit-${dim.id}`,
    name: dim.name,
    description: dim.description,
    order: i + 1,
    icon: dim.icon,
    color: dim.color,
    levels: [
      {
        id: `ailit-${dim.id}-primary`,
        name: "Primary Education",
        description: "Scenarios for primary-age learners (ages 5-11) developing AI literacy",
        order: 1,
        target: "Primary school learners (approximate ages 5-11)",
        indicators: dimCompetences.map((c) => ({
          id: `${c.id}-primary`,
          description: `${c.name}: ${c.primaryScenario}`,
        })),
      },
      {
        id: `ailit-${dim.id}-secondary`,
        name: "Secondary Education",
        description: "Scenarios for secondary-age learners (ages 12-18) developing AI literacy",
        order: 2,
        target: "Secondary school learners (approximate ages 12-18)",
        indicators: dimCompetences.map((c) => ({
          id: `${c.id}-secondary`,
          description: `${c.name}: ${c.secondaryScenario}`,
        })),
      },
    ],
  };
});

export const ailitFramework: Framework = {
  id: "ailit-framework",
  name: "Empowering Learners for the Age of AI: AI Literacy Framework",
  shortName: "AILit Framework",
  description: "AI literacy framework for primary and secondary education defining technical knowledge, durable skills, and future-ready attitudes",
  type: "competency",
  scope: "individual",
  source: "OECD",
  path: "/frameworks/ailit-framework",
  icon: "Brain",
  color: "text-violet-600",
  badgeLabel: "OECD + EU Framework",
  targetAudience: ["educator", "student", "leader", "policymaker", "designer"],
  overview: `The AILit Framework is a joint initiative of the European Commission and the OECD, with support from Code.org and leading international experts. It defines the technical knowledge, durable skills, and future-ready attitudes required to thrive in a world influenced by AI. It enables learners to engage, create with, manage, and design AI, while critically evaluating its benefits, risks, and ethical implications. The framework covers 4 domains with 22 competences, 19 knowledge statements across 5 categories, 7 cross-cutting skills, and 5 attitudes. It contributes to the PISA 2029 Media & Artificial Intelligence Literacy assessment. Unlike maturity or progression frameworks, competences are not arranged on a mastery scale — primary and secondary education scenarios provide age-appropriate implementation guidance.`,
  keyDimensions: ailitDimensions,
  keyPrinciples: [
    { id: "ailit-p1", name: "Foundational", description: "Define a core set of competences needed to demonstrate proficiency in AI literacy" },
    { id: "ailit-p2", name: "Practical", description: "Make AI literacy manageable and attainable in various classroom contexts" },
    { id: "ailit-p3", name: "Interdisciplinary", description: "Integrate AI literacy into a wide range of subjects and educational settings" },
    { id: "ailit-p4", name: "Durable", description: "Identify knowledge and skills that will remain relevant as AI evolves" },
    { id: "ailit-p5", name: "Global", description: "Incorporate insights from educators, researchers, and AI experts worldwide" },
    { id: "ailit-p6", name: "Illustrative", description: "Include scenarios and exemplars that bring AI literacy to life" },
  ],
  metadata: {
    version: "Review Draft (May 2025)",
    year: 2025,
    license: "CC BY-SA 4.0",
    source_url: "https://ailiteracyframework.org",
    coPublishers: ["European Commission"],
    developmentSupport: ["Code.org"],
    totalDomains: 4,
    totalCompetences: 22,
    totalKnowledgeCategories: 5,
    totalKnowledgeStatements: 19,
    totalSkills: 7,
    totalAttitudes: 5,
    competencesPerDomain: { "engaging-with-ai": 7, "creating-with-ai": 5, "managing-ai": 5, "designing-ai": 5 },
    knowledgeStatements: AILIT_KNOWLEDGE,
    skills: AILIT_SKILLS,
    attitudes: AILIT_ATTITUDES,
    policyAlignment: [
      "EU Digital Education Action Plan 2021-2027",
      "2023 EU Council Recommendations on digital education and skills",
      "EU AI Act (Regulation 2024/1689), Article 4 on AI literacy",
      "PISA 2029 Media & Artificial Intelligence Literacy assessment",
    ],
    buildingOn: [
      "European Commission DigComp",
      "UNESCO AI Competencies for Students",
      "UNESCO AI Competencies for Teachers",
      "Digital Promise AI Literacy Framework",
      "AI4K12 5 Big Ideas in AI",
    ],
  },
  useCases: [
    "Building AI literacy across primary and secondary education",
    "Designing AI literacy curricula aligned with PISA 2029",
    "Professional development for educators in AI readiness",
    "Integrating AI literacy into existing subject areas",
    "Evidence tagging against specific competences and knowledge statements",
  ],
  crossReferences: ["teacher-competency", "student-competency", "oecd-indicators"],
  assessmentQuestions: [
    { id: "ailit-q1", dimension: "Engaging with AI", question: "How well can you recognize and critically evaluate AI in everyday contexts?", options: [
      { value: "ailit-a1", label: "I'm just beginning to notice AI around me", level: "acquire" },
      { value: "ailit-a2", label: "I can identify AI and evaluate its outputs", level: "deepen" },
      { value: "ailit-a3", label: "I connect AI's technical design to its societal impact", level: "create" },
    ]},
    { id: "ailit-q2", dimension: "Creating with AI", question: "How effectively do you collaborate with AI in creative processes?", options: [
      { value: "ailit-b1", label: "I use AI outputs without much refinement", level: "acquire" },
      { value: "ailit-b2", label: "I iterate with AI and refine outputs thoughtfully", level: "deepen" },
      { value: "ailit-b3", label: "I use AI to explore new perspectives while considering IP and attribution", level: "create" },
    ]},
    { id: "ailit-q3", dimension: "Managing AI", question: "How intentionally do you decide when and how to use AI?", options: [
      { value: "ailit-c1", label: "I use AI without much planning", level: "acquire" },
      { value: "ailit-c2", label: "I assess tasks and delegate appropriately to AI", level: "deepen" },
      { value: "ailit-c3", label: "I develop and communicate guidelines for responsible AI use", level: "create" },
    ]},
    { id: "ailit-q4", dimension: "Designing AI", question: "How well do you understand how AI systems are built and their implications?", options: [
      { value: "ailit-d1", label: "I have limited understanding of AI design", level: "acquire" },
      { value: "ailit-d2", label: "I can compare AI approaches and understand data's role", level: "deepen" },
      { value: "ailit-d3", label: "I can evaluate AI systems and describe their purpose and limitations", level: "create" },
    ]},
  ],
  assessmentTitle: "AI Literacy Self-Assessment (AILit)",
  assessmentDescription: "Assess your AI literacy across 4 domains with 22 competences",
  showInQuiz: true,
  showInDashboard: true,
  showInLanding: false,
  isBackgroundFramework: false,
  compatibility: [],
  sourceFidelity: "official",
  estimatedAssessmentMinutes: 20,
  region: "international",
};

// ── DEC AI Literacy Framework (2025) ────────────────────────────

const DEC_DIMS: { id: string; name: string; shortName: string; guidingQuestion: string; description: string; icon: string; color: string; category: string;
  blocks: { levelIdx: number; name: string; desc: string; los: string[]; cas: string[] }[]
}[] = [
  {
    id: "dim-1", name: "Understanding AI and Data", shortName: "AI and Data", guidingQuestion: "How does AI work?", icon: "Database", color: "text-blue-600", category: "general",
    description: "Understanding how AI systems work, the principles of data collection, processing, and interpretation, and the implications of AI-generated output.",
    blocks: [
      { levelIdx: 0, name: "AI and Data Awareness", desc: "Develop a basic understanding of AI concepts, how AI systems function, and the role of data in AI decision-making.", los: [
        "Define AI and its key components (e.g. machine learning, automation)",
        "Identify common AI applications in daily life",
        "Understand the basics of how AI processes data to generate output",
      ], cas: [
        "Engage with foundational AI training materials, including introductory online courses or textbooks",
        "Learn basic data concepts, such as structured vs. unstructured data, and how AI systems process information",
        "Explore and experiment how AI systems use training data",
        "Experiment with widely available AI tools (e.g. AI chatbots, translation tools, recommendation systems) to observe how they function",
      ]},
      { levelIdx: 1, name: "AI and Data in Action", desc: "Select AI tools for real-world tasks, understand how AI models work, and assess the role of data in AI performance.", los: [
        "Explain how AI models process data and generate output",
        "Identify factors affecting AI performance, such as data quality",
        "Understand how to apply AI tools to automate or support professional tasks",
      ], cas: [
        "Conduct comparative analysis of different AI models to evaluate their accuracy and limitations",
        "Use AI-driven analytics tools to extract insights from datasets",
        "Learn about data management systems and how AI interacts with structured datasets",
        "Work with datasets in AI applications, focussing on improving data quality for better AI performance",
      ]},
      { levelIdx: 2, name: "AI and Data Optimisation", desc: "Critically engage with AI systems, assess their technical capabilities, and strategically integrate AI into decision-making.", los: [
        "Compare different AI models and their applications for a variety of tasks",
        "Integrate AI into workflows for enhanced efficiency",
        "Communicate AI system capabilities and limitations to others",
      ], cas: [
        "Lead projects involving AI integration, ensuring effective use of data pipelines and model selection",
        "Lead discussions or training sessions on AI integration, ensuring stakeholders understand AI strengths and limitations",
        "Contribute to institutional or policy discussions on AI and data governance",
        "Develop strategies for handling large datasets, and improve AI performance for the institution",
      ]},
    ],
  },
  {
    id: "dim-2", name: "Critical Thinking and Judgement", shortName: "Critical Thinking", guidingQuestion: "How do I evaluate AI output?", icon: "Search", color: "text-purple-600", category: "general",
    description: "The ability to evaluate AI-generated content, discern biases, and apply logical reasoning when using AI in decision-making.",
    blocks: [
      { levelIdx: 0, name: "Question AI Output", desc: "Identify key evaluation criteria for AI output and understand that AI-generated content may contain biases or errors.", los: [
        "Understand the importance of verifying AI-driven insights with human judgement",
        "Understand basic evaluation criteria for AI-generated content, such as accuracy, consistency, and source reliability",
        "Identify inconsistencies or biases in AI-generated content",
      ], cas: [
        "Study introductory materials on AI reliability and accuracy metrics",
        "Compare AI-generated content with verified sources to identify discrepancies",
        "Engage in case studies where AI-generated information led to errors or misinterpretation",
        "Explore AI tools to assess their reliability and accuracy",
      ]},
      { levelIdx: 1, name: "Evaluate AI Output", desc: "Critically assess AI-generated content using established evaluation criteria and identify biases or inconsistencies.", los: [
        "Apply evaluation frameworks to assess the validity of AI-generated insights",
        "Identify and articulate biases or inconsistencies in AI-generated output",
        "Compare AI-generated information against multiple independent sources for verification",
      ], cas: [
        "Develop structured evaluation rubrics for assessing AI-generated output",
        "Conduct comparative studies of different AI models to assess reliability across domains",
        "Engage in interdisciplinary discussions on AI evaluation methodologies",
        "Start applying AI assessment frameworks to real-world scenarios",
      ]},
      { levelIdx: 2, name: "Challenge AI Output", desc: "Expertise in evaluating AI-generated output with rigorous methodologies, interrogating AI's reasoning processes, and assessing AI's impact on human cognition.", los: [
        "Apply logical reasoning to understand how AI generates responses, analyse strengths and weaknesses of different AI models",
        "Effectively leverage AI capability to enhance critical thinking skills",
        "Recognise and manage the nuanced impacts of AI in complex, high-stakes situations",
      ], cas: [
        "Conduct independent evaluation of AI tools, comparing output across multiple sources for consistency and accuracy",
        "Refine evaluation methodologies based on new AI advancements and emerging best practices",
        "Publish assessments or research papers critically examining AI reliability in a specific domain",
        "Apply advanced AI evaluation frameworks to real-world professional, research, or policy contexts",
      ]},
    ],
  },
  {
    id: "dim-3", name: "Ethical and Responsible AI Use", shortName: "Ethics & Responsibility", guidingQuestion: "How do I ensure AI is used ethically and responsibly?", icon: "Shield", color: "text-rose-600", category: "general",
    description: "Ethical considerations and governance frameworks necessary for responsible AI adoption, including fairness, transparency, accountability, and privacy.",
    blocks: [
      { levelIdx: 0, name: "Understand Risks", desc: "Understand fundamental AI ethics principles and recognise potential risks such as bias, misinformation, and discrimination.", los: [
        "Define key AI ethics principles (e.g. fairness, transparency, accountability, privacy)",
        "Recognise how AI systems can perpetuate bias and inequality",
        "Identify ethical concerns in AI-driven decision-making (e.g. hiring, surveillance, law enforcement)",
      ], cas: [
        "Study introductory materials on AI ethics, including case studies of ethical failures",
        "Reflect on personal experiences using AI tools and consider ethical implications",
        "Analyse real-world case studies where AI ethics were challenged",
        "Engage in discussions on ethical dilemmas involving AI decision-making",
      ]},
      { levelIdx: 1, name: "Apply Responsible Practices", desc: "Apply ethical principles and frameworks to evaluate and mitigate risks associated with AI use.", los: [
        "Assess AI systems for compliance with ethical standards and legal frameworks",
        "Identify and mitigate risks related to bias, discrimination, and data privacy in AI applications",
        "Implement strategies to ensure fairness and accountability in AI decision-making",
      ], cas: [
        "Conduct ethical impact assessments for AI applications",
        "Engage in interdisciplinary discussions on responsible AI use across different sectors",
        "Reflect on internal guidelines for ethical AI implementation",
        "Apply ethical AI principles in project development or policy analysis",
      ]},
      { levelIdx: 2, name: "Shape Responsible Practices", desc: "Expertise in evaluating, shaping, and advocating for ethical AI policies, governance frameworks, and best practices.", los: [
        "Critically evaluate ethical implications of AI adoption at institutional or societal level",
        "Contribute to AI governance frameworks and ethical AI policies",
        "Provide guidance on ethical AI adoption in professional, academic, or policy environments",
      ], cas: [
        "Draft or contribute to ethical AI guidelines within an organisation or regulatory body",
        "Publish research or policy papers analysing ethical AI challenges and solutions",
        "Conduct workshops or training sessions on ethical AI adoption",
        "Collaborate with AI ethics advisory groups or contribute to policy discussions",
      ]},
    ],
  },
  {
    id: "dim-4", name: "Human-Centricity, Emotional Intelligence, and Creativity", shortName: "Human-Centricity", guidingQuestion: "How do I ensure humans remain at the core?", icon: "Heart", color: "text-pink-600", category: "general",
    description: "The importance of human skills in an AI-driven world, including empathy, adaptability, communication, lifelong learning, and ensuring AI aligns with societal values.",
    blocks: [
      { levelIdx: 0, name: "Awareness of Human-AI Interaction", desc: "Foundational understanding of how AI affects human decision-making, communication, and emotional intelligence.", los: [
        "Recognise how AI influences human behaviour, decision-making, and interactions",
        "Identify situations where AI may lack human sensitivity (e.g. AI-generated feedback, automated decision-making)",
        "Understand the importance of empathy and adaptability in AI-augmented environments",
      ], cas: [
        "Observe how AI influences human interactions in customer service, education, or workplace settings",
        "Reflect on personal experiences with AI-powered communication tools",
        "Engage in discussions on the limitations of AI in recognising human emotions",
        "Explore literature on psychological and social impact of AI in human interactions",
      ]},
      { levelIdx: 1, name: "AI as Collaborative Tool", desc: "Integrate human-centred skills into AI-assisted environments to promote responsible, ethical, and inclusive AI use.", los: [
        "Apply effective communication and human-in-the-loop strategies when using AI tools",
        "Identify opportunities to enhance human-centred skills and creative thinking with AI",
        "Assess AI tools to ensure inclusivity for different user groups",
      ], cas: [
        "Develop case studies on human-centred AI practices and their impact",
        "Participate in collaborative projects where AI is integrated into human-driven decision-making",
        "Explore frameworks for ensuring AI tools respect social and cultural norms",
        "Analyse AI's impact on workforce skills and creativity, propose strategies for maintaining human abilities",
      ]},
      { levelIdx: 2, name: "Develop Human-Centred AI Practices", desc: "Advocate for human-centred AI approaches, ensuring AI complements rather than replaces human skills.", los: [
        "Develop AI-driven workplace or education policies safeguarding human agency in decision-making",
        "Establish guidelines ensuring AI complements rather than replaces human interaction and creativity",
        "Conduct empirical studies or pilots testing AI's impact in human-centred roles",
      ], cas: [
        "Lead research or policy development on emotional intelligence in AI-driven work environments",
        "Create training programmes balancing AI integration with human-centric skills",
        "Engage with stakeholders to define best practices for human-AI collaboration",
        "Create reports advocating for human-centred AI principles in education, governance, or business",
      ]},
    ],
  },
  {
    id: "dim-5", name: "Domain Expertise", shortName: "Domain Expertise", guidingQuestion: "How do I apply AI in a specific context?", icon: "Target", color: "text-amber-600", category: "specialised",
    description: "Specialised knowledge and skills for understanding, assessing, and managing AI impact within a specific academic or professional context.",
    blocks: [
      { levelIdx: 0, name: "Applied AI Awareness", desc: "Basic understanding of how AI is used in a specific field and identification of relevant AI tools and applications.", los: [
        "Identify key AI applications relevant to a specific domain (e.g. AI in medicine, law, education, finance)",
        "Recognise how AI is transforming professional roles and industry standards",
        "Understand the basic limitations of AI when applied in a particular field",
      ], cas: [
        "Explore and experiment with domain-specific AI tools",
        "Participate in discussions or case studies related to AI applications in the field",
        "Engage in introductory training sessions focussed on AI for a specific sector",
      ]},
      { levelIdx: 1, name: "AI Application in Professional Contexts", desc: "Effectively use AI tools to support tasks, optimise workflows, and improve decision-making within a discipline.", los: [
        "Select and apply AI tools that enhance efficiency and accuracy in a professional or academic setting",
        "Assess strengths and weaknesses of AI applications within specific processes",
        "Integrate AI insights into professional decision-making as a complement to human expertise",
      ], cas: [
        "Implement AI-powered solutions in professional workflows, assessing impact on efficiency and accuracy",
        "Compare multiple AI tools within the field to determine best-fit applications",
        "Conduct small-scale research or pilot projects testing AI solutions in a specific setting",
      ]},
      { levelIdx: 2, name: "Strategic AI Leadership", desc: "Advanced expertise in AI applications within a discipline, ensuring effective integration into strategic decision-making.", los: [
        "Evaluate and refine AI adoption strategies considering regulatory, ethical, and operational constraints",
        "Lead implementation of AI-driven innovations in a professional or academic context",
        "Develop training materials or guidelines to enhance AI literacy among peers and colleagues",
      ], cas: [
        "Conduct industry-level assessments of AI adoption trends and their impact on professional practice",
        "Publish findings on AI applications through research, white papers, or industry reports",
        "Participate in advisory or policy groups to influence AI adoption and governance",
      ]},
    ],
  },
];

const DEC_LEVELS: Omit<Level, "indicators">[] = [
  { id: "level-1", name: "Level 1: Awareness", description: "Foundational awareness and recognition. Basic understanding, identifying key elements, recognising AI's role and implications.", order: 1 },
  { id: "level-2", name: "Level 2: Application", description: "Active application and evaluation. Select, apply, and critically assess AI tools and practices in real-world contexts.", order: 2 },
  { id: "level-3", name: "Level 3: Leadership", description: "Strategic leadership and advocacy. Advanced expertise, leading initiatives, shaping policies, contributing to institutional discourse.", order: 3 },
];

const decDimensions: FrameworkDimension[] = DEC_DIMS.map((dim, i) => ({
  id: `dec-${dim.id}`,
  name: dim.name,
  description: dim.description,
  order: i + 1,
  icon: dim.icon,
  color: dim.color,
  levels: dim.blocks.map((block) => {
    const lvl = DEC_LEVELS[block.levelIdx];
    return {
      id: `dec-${dim.id}-${lvl.id}`,
      name: `${lvl.name}: ${block.name}`,
      description: block.desc,
      order: lvl.order,
      indicators: block.los.map((lo, li) => ({
        id: `dec-${dim.id}-${lvl.id}-lo${li + 1}`,
        description: lo,
      })),
      contextualActivities: block.cas.map((ca, ci) => ({
        id: `dec-${dim.id}-${lvl.id}-ca${ci + 1}`,
        name: ca.substring(0, 60) + (ca.length > 60 ? "..." : ""),
        description: ca,
      })),
    };
  }),
}));

export const decAiLiteracy: Framework = {
  id: "dec-ai-literacy",
  name: "DEC AI Literacy Framework",
  shortName: "DEC AI Literacy",
  description: "A structured, actionable guide to AI literacy for higher education with 5 dimensions, 3 levels, and separate faculty/student tracks",
  type: "competency",
  scope: "individual",
  source: "DEC",
  path: "/frameworks/dec-ai-literacy",
  icon: "GraduationCap",
  color: "text-sky-600",
  badgeLabel: "DEC Framework",
  targetAudience: ["educator", "student"],
  overview: `The DEC AI Literacy Framework is a structured, actionable, and adaptable guide developed in consultation with leading institutions worldwide. It covers 5 dimensions: Understanding AI & Data, Critical Thinking & Judgement, Ethical & Responsible AI Use, Human-Centricity/Emotional Intelligence/Creativity, and Domain Expertise. Each dimension has 3 competency levels progressing from Awareness through Application to Leadership. Dimensions 1-4 provide general AI literacy for all, while Dimension 5 provides specialised AI literacy tailored for specific domains. The framework includes audience-specific profiles with ideal mastery targets and faculty-specific sub-competencies (facilitating critical thinking, promoting literacy, innovating pedagogy, adaptability, and ethical expertise) plus student-specific teaching strategies with classroom applications.`,
  keyDimensions: decDimensions,
  keyPrinciples: [
    { id: "dec-p1", name: "Fundamental Literacy", description: "AI literacy is as fundamental as digital literacy was a generation ago" },
    { id: "dec-p2", name: "Human Amplification", description: "AI amplifies human capability when AI literacy fosters informed, ethical, and strategic decision-making" },
    { id: "dec-p3", name: "Beyond HE", description: "AI literacy should prepare individuals not just for higher education but as a foundation for future productivity and prosperity" },
    { id: "dec-p4", name: "Domain Expertise", description: "Domain expertise is a unique defining feature, building on fundamental literacy for any field of work or study" },
  ],
  metadata: {
    year: 2025,
    publisher: "Digital Education Council",
    license: "Copyright 2025 Digital Education Council. Reproduction with attribution permitted.",
    source_url: "https://www.digitaleducationcouncil.com",
    totalDimensions: 5,
    totalLevels: 3,
    totalCompetencyBlocks: 15,
    totalLearningObjectives: 45,
    totalContextualActivities: 57,
    dimensionCategories: { general: "Dimensions 1-4: general AI literacy for all", specialised: "Dimension 5: specialised domain AI literacy" },
    hasFacultyTrack: true,
    hasStudentTrack: true,
    facultyIdealMastery: { "dim-1": "Level 2", "dim-2": "Level 3", "dim-3": "Level 3", "dim-4": "Level 3", "dim-5": "Level 2" },
    studentIdealMastery: { "dim-1": "Level 2", "dim-2": "Level 2", "dim-3": "Level 2", "dim-4": "Level 2", "dim-5": "Level 1" },
    facultyDomainSubCompetencies: [
      "Facilitating student critical thinking and learning",
      "Promoting AI and digital literacy",
      "Innovating pedagogy",
      "Adaptability and responsiveness to change",
      "Expertise in ethical and responsible AI",
    ],
    objectives: [
      "Guide individuals to acquire knowledge of AI",
      "Build the foundation for appropriate AI use",
      "Enable desirable human-AI collaboration",
    ],
  },
  useCases: [
    "HE faculty AI literacy development",
    "Student AI literacy programmes",
    "Curriculum integration of AI literacy",
    "Institutional AI readiness assessment",
    "Domain-specific AI skill building",
  ],
  crossReferences: ["teacher-competency", "student-competency", "ailit-framework"],
  assessmentQuestions: [
    { id: "dec-q1", dimension: "Understanding AI & Data", question: "How well do you understand AI systems and their data foundations?", options: [
      { value: "dec-a1", label: "I can define AI and identify common applications", level: "acquire" },
      { value: "dec-a2", label: "I can explain how AI models process data and select tools for tasks", level: "deepen" },
      { value: "dec-a3", label: "I strategically integrate AI into workflows and communicate capabilities", level: "create" },
    ]},
    { id: "dec-q2", dimension: "Critical Thinking & Judgement", question: "How critically do you evaluate AI-generated content?", options: [
      { value: "dec-b1", label: "I understand the importance of verifying AI outputs", level: "acquire" },
      { value: "dec-b2", label: "I apply evaluation frameworks and identify biases systematically", level: "deepen" },
      { value: "dec-b3", label: "I interrogate AI reasoning and assess impact on human cognition", level: "create" },
    ]},
    { id: "dec-q3", dimension: "Ethics & Responsibility", question: "How do you engage with AI ethics?", options: [
      { value: "dec-c1", label: "I can recognise ethical risks like bias and misinformation", level: "acquire" },
      { value: "dec-c2", label: "I apply ethical frameworks to evaluate and mitigate AI risks", level: "deepen" },
      { value: "dec-c3", label: "I shape ethical AI policies and governance frameworks", level: "create" },
    ]},
    { id: "dec-q4", dimension: "Human-Centricity", question: "How do you maintain human-centred approaches with AI?", options: [
      { value: "dec-d1", label: "I recognise how AI affects human decision-making and interactions", level: "acquire" },
      { value: "dec-d2", label: "I integrate human-centred skills into AI-assisted environments", level: "deepen" },
      { value: "dec-d3", label: "I advocate for and develop human-centred AI practices", level: "create" },
    ]},
    { id: "dec-q5", dimension: "Domain Expertise", question: "How do you apply AI within your discipline?", options: [
      { value: "dec-e1", label: "I can identify key AI applications in my field", level: "acquire" },
      { value: "dec-e2", label: "I effectively use AI tools to support domain-specific tasks", level: "deepen" },
      { value: "dec-e3", label: "I lead AI-driven innovations and develop AI literacy for peers", level: "create" },
    ]},
  ],
  assessmentTitle: "DEC AI Literacy Assessment",
  assessmentDescription: "Assess your AI literacy across 5 dimensions with faculty and student tracks",
  showInQuiz: true,
  showInDashboard: true,
  showInLanding: false,
  isBackgroundFramework: false,
  compatibility: [],
  sourceFidelity: "official",
  estimatedAssessmentMinutes: 30,
};

// ── DigComp 3.0 ────────────────────────────────
// Programmatic mapping from full DigComp 3.0 source data (5 areas, 21 competences, 4 levels, 362 statements, 523 LOs)

import digcompSource from "./digcomp-3-source.json";

const DC_AREA_META: Record<string, { icon: string; color: string }> = {
  "area-1": { icon: "Search", color: "text-blue-600" },
  "area-2": { icon: "MessageSquare", color: "text-emerald-600" },
  "area-3": { icon: "FileText", color: "text-purple-600" },
  "area-4": { icon: "Shield", color: "text-rose-600" },
  "area-5": { icon: "Lightbulb", color: "text-amber-600" },
};

function buildDigcompDimensions(): FrameworkDimension[] {
  const src = digcompSource as any;
  const competences = src.dimensions.filter((d: any) => d.parent_dimension_id !== null);
  const blocks = src.competency_blocks as any[];
  const dimensions: FrameworkDimension[] = [];

  // Build competence-level dimensions (21 competences, each with 4 levels)
  for (const comp of competences) {
    const meta = DC_AREA_META[comp.parent_dimension_id] || { icon: "Circle", color: "text-gray-600" };
    const compBlocks = blocks.filter((b: any) => b.competence_id === comp.id);

    dimensions.push({
      id: comp.id,
      name: comp.name,
      description: comp.description,
      order: comp.order,
      icon: meta.icon,
      color: meta.color,
      parentDimensionId: comp.parent_dimension_id,
      levels: DIGCOMP_LEVELS.map((lvl) => {
        const block = compBlocks.find((b: any) => b.level_id === lvl.id);
        const statements = block?.competence_statements || [];
        const los = block?.learning_outcomes || [];
        // Combine competence statements as primary indicators, learning outcomes as additional detail
        const indicators = statements.map((cs: any) => ({
          id: cs.id,
          description: cs.description,
          assessmentCriteria: cs.ai_label !== "none" ? `[${cs.ai_label}]` : undefined,
        }));
        return {
          ...lvl,
          id: `${comp.id}-${lvl.id}`,
          indicators,
          curricularGoals: los.map((lo: any) => ({
            id: lo.id,
            description: `[${lo.type || "skill"}] ${lo.description}${lo.ai_label !== "none" ? ` [${lo.ai_label}]` : ""}`,
          })),
        };
      }),
    });
  }

  return dimensions;
}

const digcompDimensions = buildDigcompDimensions();
const totalStatements = (digcompSource as any).competency_blocks.reduce((s: number, b: any) => s + (b.competence_statements?.length || 0), 0);
const totalLOs = (digcompSource as any).competency_blocks.reduce((s: number, b: any) => s + (b.learning_outcomes?.length || 0), 0);

export const digcomp3: Framework = {
  id: "digcomp-3",
  name: "DigComp 3.0: European Digital Competence Framework",
  shortName: "DigComp 3.0",
  description: "EU digital competence: 5 areas, 21 competences, 4 proficiency levels",
  type: "competency",
  scope: "individual",
  source: "EU",
  path: "/frameworks/digcomp-3",
  icon: "Globe",
  color: "text-blue-700",
  badgeLabel: "EU DigComp",
  targetAudience: ["citizen", "learner", "worker", "educator"],
  overview: `DigComp 3.0 (Fifth Edition, 2025) is the European Digital Competence Framework published by the European Commission Joint Research Centre. It defines 5 competence areas containing 21 competences, each assessed across 4 proficiency levels (Basic, Intermediate, Advanced, Highly Advanced). With ${totalStatements} competence statements and ${totalLOs} learning outcomes across 84 competency blocks, it provides the most comprehensive individual digital competence assessment available. This edition systematically integrates AI competence across all 21 competences, aligned with the EU AI Act Article 4. Of all statements, 14% are AI-Explicit and 68% are AI-Implicit.`,
  keyPrinciples: [
    { id: "dc3-p1", name: "People at the centre", description: "Digital technologies should protect people's rights, support democracy, and ensure that all digital players act responsibly and safely." },
    { id: "dc3-p2", name: "Solidarity and inclusion", description: "Technology should unite, not divide, people. Everyone should have access to the internet, to digital skills, to digital public services, and to fair working conditions." },
    { id: "dc3-p3", name: "Freedom of choice", description: "People should benefit from a fair online environment, be safe from illegal and harmful content, and be empowered when they interact with new and evolving technologies like artificial intelligence." },
    { id: "dc3-p4", name: "Participation", description: "Citizens should be able to engage in the democratic process at all levels and have control over their own data." },
    { id: "dc3-p5", name: "Safety and security", description: "The digital environment should be safe and secure. All users, from childhood to old age, should be empowered and protected." },
    { id: "dc3-p6", name: "Sustainability", description: "Digital devices should support sustainability and the green transition. People need to know about the environmental impact and energy consumption of their devices." },
  ],
  keyDimensions: digcompDimensions,
  metadata: {
    totalCompetences: 21,
    totalBlocks: 84,
    totalStatements,
    totalLearningOutcomes: totalLOs,
    version: "3.0 (Fifth Edition)",
    year: 2025,
    publisher: "European Commission Joint Research Centre (JRC)",
    isbn: "978-92-68-32677-0",
    doi: "10.2760/0001149",
    licence: "CC BY 4.0",
    aiIntegration: "14% AI-Explicit, 68% AI-Implicit across all competences",
    region: "EU",
  },
  useCases: [
    "Digital competence assessment for citizens",
    "Curriculum design for digital literacy",
    "Workforce digital skills benchmarking",
    "EU AI Act Article 4 compliance",
    "EU policy alignment and Digital Decade 2030 targets",
  ],
  crossReferences: [],
  assessmentQuestions: [
    { id: "dc3-q1", dimension: "Information Search, Evaluation and Management", question: "How would you rate your ability to search, evaluate, and manage digital information?", options: [
      { value: "dc3-0-a", label: "I need guidance with basic searches and evaluating sources", level: "acquire" as const },
      { value: "dc3-0-b", label: "I can critically assess sources and manage data independently", level: "deepen" as const },
      { value: "dc3-0-c", label: "I solve complex information problems and guide others", level: "create" as const },
    ]},
    { id: "dc3-q2", dimension: "Communication and Collaboration", question: "How effectively do you communicate and collaborate using digital technologies?", options: [
      { value: "dc3-1-a", label: "I use basic communication tools and follow online norms", level: "acquire" as const },
      { value: "dc3-1-b", label: "I select appropriate tools, manage identity, and collaborate effectively", level: "deepen" as const },
      { value: "dc3-1-c", label: "I lead complex digital collaborations and support others", level: "create" as const },
    ]},
    { id: "dc3-q3", dimension: "Content Creation", question: "How well can you create, integrate, and program digital content?", options: [
      { value: "dc3-2-a", label: "I create basic content and understand copyright basics", level: "acquire" as const },
      { value: "dc3-2-b", label: "I create diverse content ethically and understand computational thinking", level: "deepen" as const },
      { value: "dc3-2-c", label: "I lead complex content creation and programming initiatives", level: "create" as const },
    ]},
    { id: "dc3-q4", dimension: "Safety, Wellbeing and Responsible Use", question: "How well do you protect devices, data, wellbeing, and the environment digitally?", options: [
      { value: "dc3-3-a", label: "I apply basic security measures and understand digital wellbeing", level: "acquire" as const },
      { value: "dc3-3-b", label: "I manage privacy, support wellbeing, and reduce environmental impact", level: "deepen" as const },
      { value: "dc3-3-c", label: "I lead cybersecurity and digital sustainability initiatives", level: "create" as const },
    ]},
    { id: "dc3-q5", dimension: "Problem Identification and Solving", question: "How effectively do you identify and solve problems using digital technologies?", options: [
      { value: "dc3-4-a", label: "I follow instructions to solve basic technical problems", level: "acquire" as const },
      { value: "dc3-4-b", label: "I independently troubleshoot and identify creative digital solutions", level: "deepen" as const },
      { value: "dc3-4-c", label: "I lead innovation and support others in complex problem-solving", level: "create" as const },
    ]},
  ],
  assessmentTitle: "DigComp 3.0 Digital Competence Assessment",
  assessmentDescription: "Assess your digital competence across 5 areas and 21 competences (362 statements, 523 learning outcomes)",
  showInQuiz: false,
  showInDashboard: true,
  showInLanding: false,
  isBackgroundFramework: false,
  compatibility: [],
  sourceFidelity: "official",
  estimatedAssessmentMinutes: 168,
  region: "eu",
};

// ── ISTE Frameworks ────────────────────────────
// ISTE uses single-level standards (no progression levels).
// Full indicator data from ISTE v4.02 (2024).

interface IsteStandard {
  name: string;
  description: string;
  icon: string;
  color: string;
  code: string;
  indicators: { code: string; description: string }[];
}

function makeIsteFramework(
  id: string,
  name: string,
  shortName: string,
  audience: string[],
  scope: "individual" | "institutional",
  standards: IsteStandard[],
  opts?: { keyPrinciples?: string[]; description?: string; dimensionGroups?: { name: string; standardIndices: number[] }[] },
): Framework {
  const totalIndicators = standards.reduce((s, st) => s + st.indicators.length, 0);
  return {
    id,
    name,
    shortName,
    description: opts?.description ?? `${standards.length} standards for ${shortName.toLowerCase()}`,
    type: "competency",
    scope,
    source: "ISTE",
    path: `/frameworks/${id}`,
    icon: "Star",
    color: "text-orange-600",
    badgeLabel: "ISTE Standards",
    targetAudience: audience,
    overview: `The ISTE Standards for ${shortName} (v4.02, 2024) define ${standards.length} key competencies with ${totalIndicators} indicators for technology-enhanced education. Unlike progression-based frameworks, ISTE defines a single proficiency target per standard. Widely adopted across all 50 US states and internationally, these standards provide a clear benchmark for technology integration.`,
    keyPrinciples: opts?.keyPrinciples?.map((p, i) => ({ id: `${id}-p${i + 1}`, name: p.split(".")[0].trim(), description: p })),
    keyDimensions: standards.map((s, i) => ({
      id: `${id}-${i + 1}`,
      name: s.name,
      description: s.description,
      order: i + 1,
      icon: s.icon,
      color: s.color,
      levels: [{
        id: `${id}-${i + 1}-proficient`,
        name: "Proficient",
        description: `Meets the ${s.name} standard (${s.code})`,
        order: 1,
        indicators: s.indicators.map((ind, j) => ({
          id: `${id}-${i + 1}-i${j + 1}`,
          code: ind.code,
          description: ind.description,
        })),
      }],
    })),
    metadata: { singleLevelFramework: true, version: "4.02", year: 2024, publisher: "ISTE", totalIndicators, dimensionGroups: opts?.dimensionGroups },
    useCases: [
      "Technology integration benchmarking",
      "Professional development planning",
      "Program accreditation",
      "Self-assessment against national standards",
    ],
    crossReferences: [],
    assessmentQuestions: standards.map((s, i) => ({
      id: `${id}-q${i + 1}`,
      dimension: s.name,
      question: `How well do you meet the ${s.name} standard (${s.code})?`,
      options: [
        { value: `${id}-${i}-a`, label: "Not yet meeting this standard", level: "acquire" as const },
        { value: `${id}-${i}-b`, label: "Developing towards this standard", level: "deepen" as const },
        { value: `${id}-${i}-c`, label: "Consistently meeting this standard", level: "create" as const },
      ],
    })),
    assessmentTitle: `ISTE ${shortName} Standards Assessment`,
    assessmentDescription: `Self-assess against ${standards.length} ISTE ${shortName} standards (${totalIndicators} indicators)`,
    showInQuiz: false,
    showInDashboard: true,
    showInLanding: false,
    isBackgroundFramework: false,
    compatibility: [],
    sourceFidelity: "official",
    estimatedAssessmentMinutes: standards.length * 2,
    region: "us",
  };
}

// ── ISTE Students (1.1–1.7) ─────────────────────
export const isteStudents = makeIsteFramework("iste-students", "ISTE Standards for Students", "Students",
  ["student", "learner"], "individual", [
    { name: "Empowered Learner", code: "1.1", icon: "Sparkles", color: "text-blue-600",
      description: "Students leverage technology to take an active role in choosing, achieving and demonstrating competency in their learning goals, informed by the learning sciences.",
      indicators: [
        { code: "1.1.a", description: "Connect their learning needs, strengths and interests to their goals and use technology to help achieve them and reflect on their progress." },
        { code: "1.1.b", description: "Build networks and customize their learning environments in ways that support the learning process." },
        { code: "1.1.c", description: "Use technology to seek feedback that informs and improves their practice and to demonstrate their learning in a variety of ways." },
        { code: "1.1.d", description: "Understand fundamental concepts of how technology works, demonstrate the ability to choose and use current technologies effectively, and are adept at thoughtfully exploring emerging technologies." },
      ] },
    { name: "Digital Citizen", code: "1.2", icon: "Shield", color: "text-emerald-600",
      description: "Students recognize the responsibilities and opportunities for contributing to their digital communities.",
      indicators: [
        { code: "1.2.a", description: "Manage their digital identity and understand the lasting impact of their online behaviors on themselves and others and make safe, legal and ethical decisions in the digital world." },
        { code: "1.2.b", description: "Demonstrate empathetic, inclusive interactions online and use technology to responsibly contribute to their communities." },
        { code: "1.2.c", description: "Safeguard their well-being by being intentional about what they do online and how much time they spend online." },
        { code: "1.2.d", description: "Take action to protect their digital privacy on devices and manage their personal data and security while online." },
      ] },
    { name: "Knowledge Constructor", code: "1.3", icon: "BookOpen", color: "text-purple-600",
      description: "Students critically curate a variety of resources using digital tools to construct knowledge, produce creative artifacts and make meaningful learning experiences for themselves and others.",
      indicators: [
        { code: "1.3.a", description: "Use effective research strategies to find resources that support their learning needs, personal interests and creative pursuits." },
        { code: "1.3.b", description: "Evaluate the accuracy, validity, bias, origin, and relevance of digital content." },
        { code: "1.3.c", description: "Curate information from digital resources using a variety of tools and methods to create collections of artifacts that demonstrate meaningful connections or conclusions." },
        { code: "1.3.d", description: "Build knowledge by exploring real-world issues and gain experience in applying their learning in authentic settings." },
      ] },
    { name: "Innovative Designer", code: "1.4", icon: "Lightbulb", color: "text-amber-600",
      description: "Students use a variety of technologies within a design process to identify and solve problems by creating new, useful or imaginative solutions.",
      indicators: [
        { code: "1.4.a", description: "Know and use a deliberate design process for generating ideas, testing theories, creating innovative artifacts or solving authentic problems." },
        { code: "1.4.b", description: "Select and use digital tools to plan and manage a design process that considers design constraints and calculated risks." },
        { code: "1.4.c", description: "Develop, test and refine prototypes as part of a cyclical design process." },
        { code: "1.4.d", description: "Exhibit a tolerance for ambiguity, perseverance and the capacity to work with open-ended problems." },
      ] },
    { name: "Computational Thinker", code: "1.5", icon: "Brain", color: "text-rose-600",
      description: "Students develop and employ strategies for understanding and solving problems in ways that leverage the power of technological methods to develop and test solutions.",
      indicators: [
        { code: "1.5.a", description: "Formulate problem definitions suited for technology-assisted methods such as data analysis, abstract models and algorithmic thinking in exploring and finding solutions." },
        { code: "1.5.b", description: "Collect data or identify relevant data sets, use digital tools to analyze them, and represent data in various ways to facilitate problem-solving and decision-making." },
        { code: "1.5.c", description: "Break problems into component parts, extract key information, and develop descriptive models to understand complex systems or facilitate problem-solving." },
        { code: "1.5.d", description: "Understand how automation works and use algorithmic thinking to develop a sequence of steps to create and test automated solutions." },
      ] },
    { name: "Creative Communicator", code: "1.6", icon: "MessageSquare", color: "text-cyan-600",
      description: "Students communicate clearly and express themselves creatively for a variety of purposes using the platforms, tools, styles, formats and digital media appropriate to their goals.",
      indicators: [
        { code: "1.6.a", description: "Choose the appropriate platforms and digital tools for meeting the desired objectives of their creation or communication." },
        { code: "1.6.b", description: "Create original works or responsibly repurpose or remix digital resources into new creations." },
        { code: "1.6.c", description: "Use digital tools to visually communicate complex ideas to others." },
        { code: "1.6.d", description: "Publish or present content that customizes the message and medium for their intended audiences." },
      ] },
    { name: "Global Collaborator", code: "1.7", icon: "Globe", color: "text-teal-600",
      description: "Students use digital tools to broaden their perspectives and enrich their learning by collaborating with others and working effectively in teams locally and globally.",
      indicators: [
        { code: "1.7.a", description: "Use digital tools to connect with peers from a variety of backgrounds recognizing diverse viewpoints and broadening mutual understanding." },
        { code: "1.7.b", description: "Use collaborative technologies to work with others, including peers, experts or community members, to examine issues and problems from multiple viewpoints." },
        { code: "1.7.c", description: "Contribute constructively to project teams, assuming various roles and responsibilities to work effectively toward a common goal." },
        { code: "1.7.d", description: "Explore local and global issues and use collaborative technologies to work with others to investigate solutions." },
      ] },
  ], {
    description: "The ISTE Standards for Students define the skills and dispositions students need to thrive in an increasingly connected world. They emphasize agency, digital citizenship, computational thinking, creative communication, and collaboration using technology.",
    keyPrinciples: [
      "The standards are about learning, not tools.",
      "They emphasize student agency and ways to transform learning.",
      "Technology is leveraged in the service of learning goals.",
      "Students take an active role in choosing, achieving, and demonstrating competency.",
    ],
  }
);

// ── ISTE Educators (2.1–2.7) ────────────────────
export const isteEducators = makeIsteFramework("iste-educators", "ISTE Standards for Educators", "Educators",
  ["educator"], "individual", [
    { name: "Learner", code: "2.1", icon: "GraduationCap", color: "text-blue-600",
      description: "Educators continually improve their practice by learning from and with others and exploring proven and promising practices that leverage technology to improve student learning.",
      indicators: [
        { code: "2.1.a", description: "Set professional learning goals to apply teaching practices made possible by technology, explore promising innovations, and reflect on their effectiveness." },
        { code: "2.1.b", description: "Pursue professional interests by creating and actively participating in local and global learning networks." },
        { code: "2.1.c", description: "Stay current with research that supports improved student learning outcomes, including findings from the learning sciences." },
      ] },
    { name: "Leader", code: "2.2", icon: "Users", color: "text-emerald-600",
      description: "Educators seek opportunities for leadership to support student empowerment and success and to improve teaching and learning.",
      indicators: [
        { code: "2.2.a", description: "Shape, advance and accelerate a shared vision for empowered learning with technology by engaging with education stakeholders." },
        { code: "2.2.b", description: "Advocate for equitable access to technology, high-quality digital content, and learning opportunities to meet the diverse needs of all students." },
        { code: "2.2.c", description: "Model for colleagues the identification, experimentation, evaluation, curation and adoption of new digital resources and tools for learning." },
      ] },
    { name: "Citizen", code: "2.3", icon: "Shield", color: "text-purple-600",
      description: "Educators inspire students to positively contribute and responsibly participate in the digital world.",
      indicators: [
        { code: "2.3.a", description: "Create experiences for learners to make positive, socially responsible contributions and build inclusive communities online." },
        { code: "2.3.b", description: "Foster digital literacy by encouraging curiosity, reflection, and the critical evaluation of digital resources." },
        { code: "2.3.c", description: "Mentor students in safe, legal, and ethical practices with digital tools and content." },
        { code: "2.3.d", description: "Model and promote management of personal data, digital identity, and protection of student data." },
      ] },
    { name: "Collaborator", code: "2.4", icon: "MessageSquare", color: "text-amber-600",
      description: "Educators dedicate time to collaborate with both colleagues and students to improve practice, discover and share resources and ideas, and solve problems.",
      indicators: [
        { code: "2.4.a", description: "Dedicate planning time to collaborate with colleagues to create authentic learning experiences that leverage technology." },
        { code: "2.4.b", description: "Collaborate and co-learn with students to discover and use new digital resources and diagnose and troubleshoot technology issues." },
        { code: "2.4.c", description: "Use collaborative tools to expand students' authentic, real-world learning experiences by engaging virtually with experts, teams and students, locally and globally." },
        { code: "2.4.d", description: "Use technology to convene and empower a broad community including families, school leaders, and mentors to help students achieve their learning goals." },
      ] },
    { name: "Designer", code: "2.5", icon: "Lightbulb", color: "text-rose-600",
      description: "Educators design authentic, learner-driven activities and environments that recognize and accommodate learner variability.",
      indicators: [
        { code: "2.5.a", description: "Use technology to create, adapt and personalize learning experiences that foster independent learning and accommodate learner differences and needs." },
        { code: "2.5.b", description: "Design authentic learning activities that incorporate technology to advance student outcomes and develop opportunities for students to apply their knowledge." },
        { code: "2.5.c", description: "Apply evidence-based instructional design principles to create innovative and equitable digital learning environments that support learning." },
      ] },
    { name: "Facilitator", code: "2.6", icon: "Target", color: "text-cyan-600",
      description: "Educators facilitate learning with technology to support student achievement of the ISTE Standards for Students.",
      indicators: [
        { code: "2.6.a", description: "Foster a culture where students take ownership of their learning goals and outcomes in both independent and group settings." },
        { code: "2.6.b", description: "Manage the use of technology and student learning strategies in digital platforms, virtual environments, hands-on makerspaces or in the field." },
        { code: "2.6.c", description: "Create learning opportunities that challenge students to use a design process and/or computational thinking to innovate and solve problems." },
        { code: "2.6.d", description: "Model and nurture creativity and creative expression to communicate ideas, knowledge or connections." },
      ] },
    { name: "Analyst", code: "2.7", icon: "BarChart", color: "text-teal-600",
      description: "Educators understand and use data to drive their instruction and support students in achieving their learning goals.",
      indicators: [
        { code: "2.7.a", description: "Provide alternative ways for students to demonstrate competency and reflect on their learning using technology." },
        { code: "2.7.b", description: "Use technology to design and implement a variety of formative and summative assessments that accommodate learner needs, provide timely feedback to students and inform instruction." },
        { code: "2.7.c", description: "Use assessment data to guide progress, personalize learning, and communicate feedback to education stakeholders in support of students reaching their learning goals." },
      ] },
  ], {
    description: "The ISTE Standards for Educators define the skills educators need to teach effectively with technology. They are organized into two groupings: Empowered Professional (standards for educator growth) and Learning Catalyst (standards for catalyzing student learning).",
    keyPrinciples: [
      "Educators continually improve their practice through technology-enabled professional learning.",
      "Educators model digital citizenship and inspire responsible participation in the digital world.",
      "Educators design authentic, learner-driven activities that leverage technology.",
      "Educators use data to drive instruction and support student achievement.",
      "Evidence links these standards to improved student learning outcomes.",
    ],
    dimensionGroups: [
      { name: "Empowered Professional", standardIndices: [0, 1, 2] },
      { name: "Learning Catalyst", standardIndices: [3, 4, 5, 6] },
    ],
  }
);

// ── ISTE Coaches (4.1–4.7) ──────────────────────
export const isteCoaches = makeIsteFramework("iste-coaches", "ISTE Standards for Coaches", "Coaches",
  ["coach", "learning_technologists"], "individual", [
    { name: "Change Agent", code: "4.1", icon: "Sparkles", color: "text-blue-600",
      description: "Coaches inspire educators and leaders to use technology to create equitable and ongoing access to high-quality learning.",
      indicators: [
        { code: "4.1.a", description: "Create a shared vision and culture for using technology to learn and accelerate transformation through the coaching process." },
        { code: "4.1.b", description: "Facilitate equitable use of digital learning tools and content that meet the needs of each learner." },
        { code: "4.1.c", description: "Cultivate a supportive coaching culture that encourages educators and leaders to achieve a shared vision and individual goals." },
        { code: "4.1.d", description: "Recognize educators across the organization who use technology effectively to enable high-impact teaching and learning." },
        { code: "4.1.e", description: "Connect leaders, educators, instructional support, technical support, domain experts and solution providers to maximize the potential of technology for learning." },
      ] },
    { name: "Connected Learner", code: "4.2", icon: "Globe", color: "text-emerald-600",
      description: "Coaches model the ISTE Standards for Students and the ISTE Standards for Educators and identify ways to improve their coaching practice.",
      indicators: [
        { code: "4.2.a", description: "Pursue professional learning that deepens expertise in the ISTE Standards in order to serve as a model for educators and leaders." },
        { code: "4.2.b", description: "Actively participate in professional learning networks to enhance coaching practice and keep current with emerging technology and innovations in pedagogy and the learning sciences." },
        { code: "4.2.c", description: "Establish shared goals with educators, reflect on successes and continually improve coaching and teaching practice." },
      ] },
    { name: "Collaborator", code: "4.3", icon: "Users", color: "text-purple-600",
      description: "Coaches establish productive relationships with educators in order to improve instructional practice and learning outcomes.",
      indicators: [
        { code: "4.3.a", description: "Establish trusting and respectful coaching relationships that encourage educators to explore new instructional strategies." },
        { code: "4.3.b", description: "Partner with educators to identify digital learning content that is culturally relevant, developmentally appropriate and considers student interests and aspirations." },
        { code: "4.3.c", description: "Partner with educators to evaluate the efficacy of digital learning content and tools to inform procurement decisions and adoption." },
        { code: "4.3.d", description: "Personalize support for educators by planning and modeling the effective use of technology to improve student learning." },
      ] },
    { name: "Learning Designer", code: "4.4", icon: "Lightbulb", color: "text-amber-600",
      description: "Coaches model and support educators to design learning experiences and environments to meet the needs and interests of all students.",
      indicators: [
        { code: "4.4.a", description: "Collaborate with educators to develop authentic, active learning experiences that foster student agency, deepen content mastery and allow students to demonstrate their competency." },
        { code: "4.4.b", description: "Help educators use digital tools to create effective assessments that provide timely feedback and support personalized learning." },
        { code: "4.4.c", description: "Collaborate with educators to design accessible and active digital learning environments that accommodate learner variability." },
        { code: "4.4.d", description: "Model the use of instructional design principles with educators to create effective digital learning environments." },
      ] },
    { name: "Professional Learning Facilitator", code: "4.5", icon: "GraduationCap", color: "text-rose-600",
      description: "Coaches plan, provide and evaluate the impact of professional learning for educators and leaders to use technology to advance teaching and learning.",
      indicators: [
        { code: "4.5.a", description: "Design professional learning based on needs assessments and frameworks for working with adults to support their cultural, social-emotional and learning needs." },
        { code: "4.5.b", description: "Build the capacity of educators, leaders and instructional teams to put the ISTE Standards into practice by facilitating active learning and providing meaningful feedback." },
        { code: "4.5.c", description: "Evaluate impact of professional learning and continually make improvements in order to meet schoolwide vision for using technology for high-impact teaching and learning." },
      ] },
    { name: "Data-Driven Decision-Maker", code: "4.6", icon: "BarChart", color: "text-cyan-600",
      description: "Coaches model and support the use of qualitative and quantitative data to inform their own instruction and professional learning.",
      indicators: [
        { code: "4.6.a", description: "Model best practices for educators and leaders for securely collecting, protecting and analyzing student data." },
        { code: "4.6.b", description: "Support educators to interpret qualitative and quantitative data to inform their decisions and support individual student learning." },
        { code: "4.6.c", description: "Partner with educators to empower students to use learning data to set their own goals and measure their progress." },
      ] },
    { name: "Digital Citizen Advocate", code: "4.7", icon: "Shield", color: "text-teal-600",
      description: "Coaches model digital citizenship and support educators and students in recognizing the responsibilities and opportunities inherent in living in a digital world.",
      indicators: [
        { code: "4.7.a", description: "Work with educators to create pathways for students to use technology to address community challenges and gain real-world experience." },
        { code: "4.7.b", description: "Collaborate with educators, leaders and students to foster inclusive online spaces and healthy balance in their use of technology." },
        { code: "4.7.c", description: "Support educators and students to critically examine the sources and accuracy of online content and evaluate underlying assumptions, biases, and perspectives." },
        { code: "4.7.d", description: "Empower educators, leaders and students to make informed decisions to protect their personal data and curate the digital profile they intend to reflect." },
      ] },
  ], {
    description: "The ISTE Standards for Coaches define the competencies technology coaches need to support educators and leaders in using technology effectively for teaching and learning. Coaches serve as change agents, learning designers, and professional learning facilitators.",
    keyPrinciples: [
      "Coaches inspire and support educators and leaders in technology integration.",
      "Coaches model the ISTE Standards for Students and Educators.",
      "Coaches build productive, trusting relationships with educators.",
      "Coaches use data to inform instruction and professional learning.",
      "Coaches advocate for digital citizenship and equitable technology access.",
    ],
  }
);

// ── ISTE Leaders (3.1–3.5) ──────────────────────
export const isteLeaders = makeIsteFramework("iste-leaders", "ISTE Standards for Education Leaders", "Leaders",
  ["education_leader", "senior_leaders", "strategic_leaders"], "institutional", [
    { name: "Digital Citizenship Advocate", code: "3.1", icon: "Shield", color: "text-blue-600",
      description: "Leaders ensure all students engage in the active use of technology for learning and build their digital citizenship skills.",
      indicators: [
        { code: "3.1.a", description: "Ensure all students learn from educators who are skilled in using technology to create authentic and engaging learning experiences." },
        { code: "3.1.b", description: "Ensure access to technology, connectivity, inclusive digital content and learning environments that meet the needs of all students." },
        { code: "3.1.c", description: "Model the use of technology in inclusive, healthy ways to solve problems and strengthen community." },
        { code: "3.1.d", description: "Model the safe, ethical, and legal use of technology and the critical examination of digital content." },
        { code: "3.1.e", description: "Ensure all students engage in the active use of technology for learning and build their digital citizenship skills." },
      ] },
    { name: "Visionary Planner", code: "3.2", icon: "Target", color: "text-emerald-600",
      description: "Leaders engage others in establishing a vision, strategic plan and ongoing evaluation cycle for transforming learning with technology.",
      indicators: [
        { code: "3.2.a", description: "Include a wide range of perspectives from the community to develop and sustain a vision for using technology to advance student learning and success." },
        { code: "3.2.b", description: "Build on the shared vision by collaboratively creating a strategic plan that articulates how technology will be used to enhance learning." },
        { code: "3.2.c", description: "Evaluate progress on the strategic plan, make course corrections, measure impact and scale effective approaches for using technology to transform learning." },
        { code: "3.2.d", description: "Communicate effectively with stakeholders to gather input on the plan, celebrate successes and engage in a continuous improvement cycle." },
        { code: "3.2.e", description: "Share lessons learned, best practices, challenges and the impact of learning with technology with other education leaders who want to learn from this work." },
      ] },
    { name: "Empowering Leader", code: "3.3", icon: "Sparkles", color: "text-purple-600",
      description: "Leaders create a culture where teachers and learners are empowered to use technology in innovative ways to enrich teaching and learning.",
      indicators: [
        { code: "3.3.a", description: "Empower educators to exercise professional agency, build teacher leadership skills and pursue personalized professional learning." },
        { code: "3.3.b", description: "Build the confidence and competency of educators to put the ISTE Standards for Students and Educators into practice." },
        { code: "3.3.c", description: "Inspire a culture of innovation, creative problem-solving, and collaboration that allows the time to explore and develop teaching practices using digital tools." },
        { code: "3.3.d", description: "Support educators in using technology to advance learning that meets the diverse learning, cultural, and social-emotional needs of individual students." },
        { code: "3.3.e", description: "Develop learning assessments that provide a personalized, actionable view of student progress in real time." },
      ] },
    { name: "Systems Designer", code: "3.4", icon: "Settings", color: "text-amber-600",
      description: "Leaders build teams and systems to implement, sustain and continually improve the use of technology to support learning.",
      indicators: [
        { code: "3.4.a", description: "Guide teams to establish equitable technology policies that support effective learning." },
        { code: "3.4.b", description: "Ensure that resources and infrastructure for supporting effective use of technology for learning are sufficient and scalable to meet future demand." },
        { code: "3.4.c", description: "Protect privacy and security by ensuring that students and staff observe effective privacy and data management policies." },
        { code: "3.4.d", description: "Establish partnerships to advance strategic plans, achieve learning priorities and develop opportunities for students to gain real-world experience." },
      ] },
    { name: "Connected Learner", code: "3.5", icon: "Globe", color: "text-rose-600",
      description: "Leaders model and promote continuous professional learning for themselves and others.",
      indicators: [
        { code: "3.5.a", description: "Set goals to remain current on emerging technologies for learning, innovations in pedagogy and advancements in the learning sciences." },
        { code: "3.5.b", description: "Participate regularly in online professional learning networks to collaboratively learn with and mentor other professionals." },
        { code: "3.5.c", description: "Use technology to regularly engage in reflective practices that support personal and professional growth." },
        { code: "3.5.d", description: "Develop the skills needed to lead and navigate change, advance systems and promote a mindset of continuous improvement for how technology can improve learning." },
      ] },
  ], {
    description: "The ISTE Standards for Education Leaders define the competencies leaders need to ensure all students engage in active technology-enhanced learning, and to build systems that support technology integration for teaching and learning.",
    keyPrinciples: [
      "Leaders ensure equitable access to technology for all students.",
      "Leaders establish and communicate a shared vision for technology-enhanced learning.",
      "Leaders empower educators to innovate with technology.",
      "Leaders build sustainable systems and infrastructure for technology integration.",
      "Leaders model continuous professional learning.",
    ],
  }
);

// ── JISC AI Maturity ───────────────────────────

export const maturityJiscAi: Framework = {
  id: "maturity-jisc-ai",
  name: "AI Maturity Model (JISC)",
  shortName: "JISC AI Maturity",
  description: "5 dimensions across 5 maturity levels for institutional AI readiness",
  type: "maturity",
  scope: "institutional",
  source: "JISC",
  path: "/frameworks/maturity-jisc-ai",
  icon: "Brain",
  color: "text-violet-600",
  badgeLabel: "JISC AI Maturity",
  targetAudience: ["leader", "admin"],
  overview: `The JISC AI Maturity Model helps UK higher and further education institutions assess their AI readiness across 5 dimensions: AI Strategy & Governance, AI Skills & Culture, AI in Teaching & Learning, AI in Research, and AI Infrastructure & Data. Each dimension has 5 maturity levels (Exploring → Optimising). Unlike the broader JISC Digital Maturity Model, this framework focuses specifically on AI adoption and integration.`,
  keyDimensions: [
    { id: "jiscai-strategy", name: "AI Strategy & Governance", description: "Strategic planning and governance for AI adoption", order: 1, icon: "Target", color: "text-blue-600" },
    { id: "jiscai-skills", name: "AI Skills & Culture", description: "Building AI literacy and a culture of AI adoption", order: 2, icon: "Users", color: "text-purple-600" },
    { id: "jiscai-teaching", name: "AI in Teaching & Learning", description: "Integrating AI into pedagogy and curriculum", order: 3, icon: "GraduationCap", color: "text-emerald-600" },
    { id: "jiscai-research", name: "AI in Research", description: "Using AI to enhance research capabilities", order: 4, icon: "FlaskConical", color: "text-amber-600" },
    { id: "jiscai-infra", name: "AI Infrastructure & Data", description: "Technical infrastructure and data readiness for AI", order: 5, icon: "Database", color: "text-rose-600" },
  ].map((dim) => ({
    ...dim,
    levels: JISC_AI_LEVELS.map((l) => ({
      ...l,
      id: `${dim.id}-${l.id}`,
      indicators: [
        { id: `${dim.id}-${l.id}-i1`, description: `${dim.name} at ${l.name} level: ${l.description}` },
      ],
    })),
  })),
  metadata: { region: "UK", sector: "HE/FE", aiSpecific: true },
  useCases: [
    "Institutional AI readiness assessment",
    "AI strategy development",
    "AI investment planning",
    "Sector benchmarking",
  ],
  crossReferences: ["ai-capability", "maturity-jisc"],
  assessmentQuestions: [
    { id: "jiscai-q1", dimension: "AI Strategy & Governance", question: "How mature is your institution's AI strategy?", options: [
      { value: "jiscai-a1", label: "No formal AI strategy exists", level: "acquire" },
      { value: "jiscai-a2", label: "AI strategy is developing but not institution-wide", level: "deepen" },
      { value: "jiscai-a3", label: "Comprehensive AI strategy with measurable outcomes", level: "create" },
    ]},
    { id: "jiscai-q2", dimension: "AI Skills & Culture", question: "How AI-literate are your staff and students?", options: [
      { value: "jiscai-b1", label: "Limited AI awareness across the institution", level: "acquire" },
      { value: "jiscai-b2", label: "AI training available but uptake varies", level: "deepen" },
      { value: "jiscai-b3", label: "Strong AI literacy culture with ongoing development", level: "create" },
    ]},
    { id: "jiscai-q3", dimension: "AI in Teaching & Learning", question: "How is AI integrated into teaching and learning?", options: [
      { value: "jiscai-c1", label: "Ad-hoc experiments by individual staff", level: "acquire" },
      { value: "jiscai-c2", label: "Structured AI pilots in some programmes", level: "deepen" },
      { value: "jiscai-c3", label: "AI embedded across curriculum with support", level: "create" },
    ]},
    { id: "jiscai-q4", dimension: "AI in Research", question: "How is AI supporting research at your institution?", options: [
      { value: "jiscai-d1", label: "Individual researchers using AI tools ad hoc", level: "acquire" },
      { value: "jiscai-d2", label: "Institutional AI research tools and training", level: "deepen" },
      { value: "jiscai-d3", label: "AI integral to research strategy", level: "create" },
    ]},
    { id: "jiscai-q5", dimension: "AI Infrastructure", question: "How ready is your technical infrastructure for AI?", options: [
      { value: "jiscai-e1", label: "No AI-specific infrastructure", level: "acquire" },
      { value: "jiscai-e2", label: "Some AI tools deployed but not integrated", level: "deepen" },
      { value: "jiscai-e3", label: "Comprehensive AI infrastructure with data governance", level: "create" },
    ]},
  ],
  assessmentTitle: "JISC AI Maturity Assessment",
  assessmentDescription: "Assess your institution's AI maturity across 5 dimensions",
  showInQuiz: true,
  showInDashboard: true,
  showInLanding: false,
  isBackgroundFramework: false,
  compatibility: [],
  sourceFidelity: "official",
  estimatedAssessmentMinutes: 50,
  region: "uk",
};

// ── Export all new frameworks ──────────────────

export const ADDITIONAL_FRAMEWORKS: Framework[] = [
  ailitFramework,
  bdcIndividual,
  bdcTeacherHe,
  bdcResearcher,
  bdcProfessionalServices,
  bdcLearningTechnology,
  bdcDigitalLeader,
  bdcEducationalDeveloper,
  maturityJiscAi,
  decAiLiteracy,
  digcomp3,
  isteStudents,
  isteEducators,
  isteCoaches,
  isteLeaders,
];
