// ============================================================
// Central Framework Configuration — Single Source of Truth
// Machine-readable hierarchical schema:
//   Framework → Dimension → Level → Indicator
// ============================================================

// Re-export all types from the shared types file
export type {
  CompetencyLevel,
  AssessmentQuestion,
  Indicator,
  CurricularGoal,
  ContextualActivity,
  KeyPrinciple,
  Level,
  FrameworkDimension,
  FrameworkScope,
  FrameworkSource,
  FrameworkType,
  CompatibilityCategory,
  CompatibilityEntry,
  SourceFidelity,
  Framework,
} from "./framework-types";

import type { Framework } from "./framework-types";
import { ACQUIRE_DEEPEN_CREATE } from "./framework-types";
import { ADDITIONAL_FRAMEWORKS } from "./frameworks-additional";

// ────────────────────────────────────────────────
// 1. UNESCO Guidance & Policy
// ────────────────────────────────────────────────
const guidancePolicy: Framework = {
  id: "guidance-policy",
  name: "Guidance for AI in Education & Research",
  shortName: "Guidance & Policy",
  description: "Human-centered AI adoption with safe, equitable practices",
  type: "policy",
  scope: "institutional",
  source: "UNESCO",
  path: "/frameworks/guidance-policy",
  icon: "FileText",
  color: "text-blue-600",
  badgeLabel: "UNESCO Framework",
  targetAudience: ["leader", "admin", "educator"],
  overview: `UNESCO's "Guidance for generative AI in education and research" (2023) is the first global policy framework addressing GenAI in education. It provides actionable steps for governments and institutions to regulate AI ethically. The guidance covers human-centered design, data governance, content validation, and inclusive access. It stresses that AI must never replace the human relationship at the heart of education and calls for age-appropriate deployment, robust privacy safeguards, and transparent accountability chains. Institutions can use this framework to draft policies, conduct Data Protection Impact Assessments (DPIAs), and build risk registers.`,
  keyDimensions: [
    {
      id: "gp-human-centered", name: "Human-Centered AI", description: "Prioritize human agency, dignity, and wellbeing in all AI applications",
      order: 1, icon: "Users", color: "text-blue-600",
      levels: [
        { id: "gp-hc-emerging", name: "Emerging", description: "Initial awareness of human-centered AI principles", order: 1, indicators: [
          { id: "gp-hc-e1", description: "Recognise that AI should serve human needs, not replace human judgment" },
          { id: "gp-hc-e2", description: "Identify stakeholders affected by AI decisions" },
        ]},
        { id: "gp-hc-developing", name: "Developing", description: "Actively applying human-centered principles", order: 2, indicators: [
          { id: "gp-hc-d1", description: "Conduct stakeholder impact assessments before AI deployment" },
          { id: "gp-hc-d2", description: "Establish feedback mechanisms for AI-affected communities" },
        ]},
        { id: "gp-hc-established", name: "Established", description: "Embedded human-centered AI governance", order: 3, indicators: [
          { id: "gp-hc-s1", description: "Maintain ongoing human oversight of all AI systems", assessmentCriteria: "Documented oversight procedures with named responsible officers" },
          { id: "gp-hc-s2", description: "Regularly audit AI systems for alignment with human values" },
        ]},
      ],
    },
    {
      id: "gp-safe-equitable", name: "Safe & Equitable", description: "Ensure safe, age-appropriate use with equitable access for all learners",
      order: 2, icon: "Shield", color: "text-emerald-600",
      levels: [
        { id: "gp-se-emerging", name: "Emerging", description: "Basic safety awareness", order: 1, indicators: [
          { id: "gp-se-e1", description: "Awareness of age-appropriateness concerns" },
          { id: "gp-se-e2", description: "Recognition of access inequities" },
        ]},
        { id: "gp-se-developing", name: "Developing", description: "Implementing safety measures", order: 2, indicators: [
          { id: "gp-se-d1", description: "Age-appropriate deployment guidelines in place" },
          { id: "gp-se-d2", description: "Access audits conducted periodically" },
        ]},
        { id: "gp-se-established", name: "Established", description: "Comprehensive safety and equity", order: 3, indicators: [
          { id: "gp-se-s1", description: "Robust content filtering and safety protocols for all AI tools", assessmentCriteria: "Documented safety protocols reviewed annually" },
          { id: "gp-se-s2", description: "Proactive programmes addressing digital divides" },
        ]},
      ],
    },
    {
      id: "gp-ethics", name: "Ethics & Accountability", description: "Transparent governance with clear accountability frameworks",
      order: 3, icon: "Scale", color: "text-purple-600",
      levels: [
        { id: "gp-et-emerging", name: "Emerging", description: "Ethics awareness", order: 1, indicators: [
          { id: "gp-et-e1", description: "Awareness of ethical implications of AI in education" },
        ]},
        { id: "gp-et-developing", name: "Developing", description: "Ethics integration", order: 2, indicators: [
          { id: "gp-et-d1", description: "Ethics review processes for new AI tools" },
          { id: "gp-et-d2", description: "Clear accountability chains documented" },
        ]},
        { id: "gp-et-established", name: "Established", description: "Embedded ethics governance", order: 3, indicators: [
          { id: "gp-et-s1", description: "Independent ethics committee reviewing AI deployment", assessmentCriteria: "Committee meets quarterly with published minutes" },
        ]},
      ],
    },
    {
      id: "gp-evidence", name: "Evidence-Based", description: "Ground decisions in research and continuously evaluate impact",
      order: 4, icon: "CheckCircle", color: "text-amber-600",
      levels: [
        { id: "gp-ev-emerging", name: "Emerging", description: "Beginning evidence gathering", order: 1, indicators: [
          { id: "gp-ev-e1", description: "Awareness of need for evidence-based AI decisions" },
        ]},
        { id: "gp-ev-developing", name: "Developing", description: "Systematic evaluation", order: 2, indicators: [
          { id: "gp-ev-d1", description: "Pilot programmes with defined success metrics" },
        ]},
        { id: "gp-ev-established", name: "Established", description: "Continuous evidence loop", order: 3, indicators: [
          { id: "gp-ev-s1", description: "Longitudinal impact studies informing AI policy revisions", assessmentCriteria: "Annual impact reports published" },
        ]},
      ],
    },
  ],
  metadata: { publicationYear: 2023, documentType: "Policy guidance" },
  useCases: ["Drafting institutional AI policies", "Conducting DPIAs for AI tools", "Building risk registers", "Creating communication packs for stakeholders"],
  crossReferences: ["teacher-competency", "ai-capability"],
  assessmentQuestions: [
    { id: "gp-q1", dimension: "Policy Readiness", question: "How well-defined is your institution's AI usage policy?", options: [
      { value: "gp-a1", label: "We have no formal AI policy yet", level: "acquire" },
      { value: "gp-a2", label: "We have draft guidelines but they're not widely adopted", level: "deepen" },
      { value: "gp-a3", label: "We have a comprehensive, adopted policy with regular reviews", level: "create" },
    ]},
    { id: "gp-q2", dimension: "Data Governance", question: "How does your institution handle data protection for AI tools?", options: [
      { value: "gp-b1", label: "We rely on vendor defaults with no internal review", level: "acquire" },
      { value: "gp-b2", label: "We conduct basic reviews but lack formal DPIAs", level: "deepen" },
      { value: "gp-b3", label: "We have DPIAs for all AI tools with clear data flows", level: "create" },
    ]},
    { id: "gp-q3", dimension: "Stakeholder Communication", question: "How transparent is your institution about AI use with students and staff?", options: [
      { value: "gp-c1", label: "Little to no communication about AI policies", level: "acquire" },
      { value: "gp-c2", label: "Some guidance exists but isn't consistently shared", level: "deepen" },
      { value: "gp-c3", label: "Clear, proactive communication with all stakeholders", level: "create" },
    ]},
    { id: "gp-q4", dimension: "Risk Management", question: "How does your institution identify and mitigate AI-related risks?", options: [
      { value: "gp-d1", label: "Risks are addressed reactively as issues arise", level: "acquire" },
      { value: "gp-d2", label: "We have a basic risk register but it's not comprehensive", level: "deepen" },
      { value: "gp-d3", label: "We maintain a living risk register with mitigation plans", level: "create" },
    ]},
    { id: "gp-q5", dimension: "Equity & Inclusion", question: "How does your institution ensure equitable access to AI tools?", options: [
      { value: "gp-e1", label: "We haven't formally considered equity in AI access", level: "acquire" },
      { value: "gp-e2", label: "We're aware of gaps but haven't addressed them systematically", level: "deepen" },
      { value: "gp-e3", label: "We actively audit and address equity gaps in AI provision", level: "create" },
    ]},
  ],
  assessmentTitle: "Guidance & Policy Readiness Assessment",
  assessmentDescription: "Evaluate your institution's readiness across 5 policy dimensions",
  showInQuiz: true,
  showInDashboard: true,
  showInLanding: true,
  isBackgroundFramework: false,
  compatibility: [],
  sourceFidelity: "official",
  estimatedAssessmentMinutes: 10,
};

// ────────────────────────────────────────────────
// 2. UNESCO Teacher AI Competency
// ────────────────────────────────────────────────
const teacherCompetency: Framework = {
  id: "teacher-competency",
  name: "UNESCO AI Competency Framework for Teachers",
  shortName: "AI CFT",
  description: "The first global framework defining 15 competencies across 5 aspects and 3 progression levels for educators in the age of AI",
  type: "competency",
  scope: "individual",
  source: "UNESCO",
  path: "/frameworks/teacher-competency",
  icon: "Users",
  color: "text-purple-600",
  badgeLabel: "UNESCO Framework",
  targetAudience: ["educator"],
  region: "international",
  overview: `The UNESCO AI Competency Framework for Teachers (2024) is the first ever global framework defining the knowledge, skills, and values teachers must master in the age of AI. It outlines 15 competencies across five aspects and three progression levels, developed with principles of protecting teachers' rights, enhancing human agency, and promoting sustainability. The framework recognises that AI has transformed education into a teacher–AI–student dynamic where the teacher's role shifts from sole knowledge provider to orchestrator. It spans 5 aspects—Human-centred mindset, Ethics of AI, AI Foundations & Applications, AI Pedagogy, and AI for Professional Development—each with 3 progression levels: Acquire (literacy), Deepen (proficiency), and Create (innovation). By 2022 only 7 countries had AI teacher frameworks; this UNESCO model fills the gap with 15 actionable competencies.`,
  keyPrinciples: [
    { id: "principle-1", name: "Ensuring inclusive digital futures", tenets: ["Debunking AI hype", "Understanding threats inherent to the design of AI", "Ensuring human and social values prevail", "Steering AI for human capacity development"] },
    { id: "principle-2", name: "A human-centred approach to AI", tenets: ["Empowering teachers' human-accountable use of AI", "Promoting inclusivity", "Recognizing users' right to question the explainability of AI tools", "Understanding and monitoring the human-controlled impact of AI"] },
    { id: "principle-3", name: "Protecting teachers' rights and iteratively (re)defining teachers' roles" },
    { id: "principle-4", name: "Promoting trustworthy and environmentally sustainable AI for education", tenets: ["Mandating the 'do no harm' principle", "Prioritizing environmentally-friendly AI tools", "Validating trustworthy AI for educational purposes", "Human accountable design and development"] },
    { id: "principle-5", name: "Ensuring applicability for all teachers and reflecting digital evolution" },
    { id: "principle-6", name: "Lifelong professional learning for teachers", tenets: ["Navigate personal progression through transferable competencies", "Guide continuous reflection and improvement of practical performance", "Streamline training and support programmes", "Adapt policies to support lifelong professional learning"] },
  ],
  keyDimensions: [
    // ── Aspect 1: Human-centred mindset ──
    {
      id: "tc-human-centred", name: "Human-centred mindset",
      description: "Defines the values and critical attitudes teachers need to develop towards human-AI interactions. Encourages teachers to always put human rights and needs for human flourishing as the focus of AI in education.",
      order: 1, icon: "Heart", color: "text-rose-600",
      levels: [
        {
          ...ACQUIRE_DEEPEN_CREATE[0], id: "tc-hc-acquire",
          name: "Human agency",
          description: "Teachers have a critical understanding that AI is human-led, and that corporate and individual decisions of AI creators have a profound impact on human autonomy and rights.",
          target: "Teachers with limited or no prior AI knowledge or skills.",
          curricularGoals: [
            { id: "CG1.1.1", description: "Foster critical thinking on AI by organizing teachers to discuss and take perspectives on the dilemma of benefits offered by AI versus the risks of diminishing human autonomy and human agency." },
            { id: "CG1.1.2", description: "Illustrate key steps in the life cycle of AI systems and guide teachers to understand how corporate and individual decisions of creators may affect the impact of AI." },
            { id: "CG1.1.3", description: "Highlight how overreliance on AI can undermine thinking skills and human agency." },
            { id: "CG1.1.4", description: "Offer practices of writing basic tips to help protect human agency when using AI in education, with a specific focus on students with special needs." },
          ],
          indicators: [
            { id: "LO1.1.1", description: "Critically reflect on the benefits, limitations and risks of specific AI tools in their local educational settings and the subject areas and grade levels they teach." },
            { id: "LO1.1.2", description: "Demonstrate an awareness that AI is human-led and the corporate and individual decisions of AI creators affect the impacts on human rights, human agency, individual lives, and societies." },
            { id: "LO1.1.3", description: "Outline the role of humans in the basic steps involved in AI development, from the collection and processing of data to the design of algorithms and functionalities of an AI system." },
            { id: "LO1.1.4", description: "Understand the need to use basic measures to protect human agency in key steps regarding the design and use of AI systems." },
          ],
          contextualActivities: [
            { id: "CA1.1.1", name: "Unpack hype around AI", description: "Critically examine hype around concrete AI tools through basic risk-benefit analysis and by highlighting the central role of humans." },
            { id: "CA1.1.2", name: "Understand why some AI tools should be banned", description: "Demonstrate a basic understanding of why some AI tools should be banned given their potential to diminish human agency and threaten human rights." },
            { id: "CA1.1.3", name: "Spotlight risks", description: "List the potential ways in which teachers' and students' agency may be undermined by certain AI tools." },
            { id: "CA1.1.4", name: "Know basic dos and don'ts", description: "Write daily tips to promote human agency when using AI in teaching and to encourage student agency." },
          ],
        },
        {
          ...ACQUIRE_DEEPEN_CREATE[1], id: "tc-hc-deepen",
          name: "Human accountability",
          description: "Teachers can demonstrate a deepened understanding of human accountability and human determination in the proper deployment and use of AI, as well as a critical capacity to assess AI's capabilities in human-AI decision loops.",
          target: "Teachers who already have some knowledge of AI and some experience of using it in education.",
          curricularGoals: [
            { id: "CG1.2.1", description: "Deepen teachers' understanding of the risks related to the absence of human accountability through examination of use cases of AI for decision loops in educational management, assessment, teaching strategies and student interactions." },
            { id: "CG1.2.2", description: "Develop the understanding that human accountability is a legal obligation by encouraging teachers to debate whether humans or AI should take accountability in AI-assisted decision loops." },
            { id: "CG1.2.3", description: "Build associations between human accountability and teachers' rights by highlighting the changing roles and responsibilities of teachers." },
            { id: "CG1.2.4", description: "Uncover risks related to the absence of users' accountability by encouraging teachers to examine explainable limitations of specific AI tools." },
          ],
          indicators: [
            { id: "LO1.2.1", description: "Understand that human accountability in human-AI decision loops is a legal obligation." },
            { id: "LO1.2.2", description: "Apply local and/or international regulatory frameworks to examine whether the design or use of a specific AI tool diminishes human accountability." },
            { id: "LO1.2.3", description: "Make reference to international or local policies to defend teachers' accountability in using AI in education." },
            { id: "LO1.2.4", description: "Demonstrate teachers' accountability in the decision loops including when determining the appropriateness of AI tools in teaching." },
          ],
          contextualActivities: [
            { id: "CA1.2.1", name: "Human accountability in AI-assisted decision loops is a legal obligation", description: "Draw a concept map of key duty-bearers and their roles in the design, deployment and use of AI in education." },
            { id: "CA1.2.2", name: "Teachers' accountability and rights cannot be usurped by AI", description: "Draft a report on the most relevant regulation(s) that can protect teachers' rights and accountability when adopting AI." },
            { id: "CA1.2.3", name: "Teachers' accountability is a human assurance for ethical and effective uses of AI", description: "Draw a concept map on the feasible roles teachers can play in validating and selecting appropriate AI tools." },
          ],
        },
        {
          ...ACQUIRE_DEEPEN_CREATE[2], id: "tc-hc-create",
          name: "Social responsibility",
          description: "Teachers are able to actively participate in, and contribute to, the building of inclusive AI societies guided by a critical understanding of the implications of AI for societal norms.",
          target: "Teachers who have strong AI knowledge and skills as well as rich experience in using AI in education.",
          curricularGoals: [
            { id: "CG1.3.1", description: "Foster a critical understanding of the importance of protecting social and emotional well-being from commercially-driven AI manipulation." },
            { id: "CG1.3.2", description: "Offer opportunities to reimagine safe, inclusive and just AI societies; organize workshops and collaborative activities." },
            { id: "CG1.3.3", description: "Encourage the internalization of social responsibilities as citizens in an AI society by organizing hands-on workshops." },
          ],
          indicators: [
            { id: "LO1.3.1", description: "Critically evaluate and reflect on the implications of AI for society at large, particularly how it might affect education, work, interpersonal interaction and human connections." },
            { id: "LO1.3.2", description: "Actively contribute to the formation of policies related to AI in education at the institutional, local and/or national level." },
            { id: "LO1.3.3", description: "Personalize and actualize social and civic responsibilities in the era of AI and promote the development of such citizenship qualities through education." },
          ],
          contextualActivities: [
            { id: "CA1.3.1", name: "Teachers' voices on human and planetary well-being", description: "Write thought pieces or essays about how profit-driven AI providers threaten humans' social and emotional well-being and planetary well-being." },
            { id: "CA1.3.2", name: "Reflection on human-centric social relations", description: "Write blogs or champion dialogues on what desirable social relations and social cohesion can look like in the AI era." },
            { id: "CA1.3.3", name: "Rights, obligations, and responsibilities of citizenship in the era of AI", description: "Engage in discussing, consulting on, or contributing to the drafting of policies that define the rights and responsibilities of citizens in the AI era." },
          ],
        },
      ],
    },
    // ── Aspect 2: Ethics of AI ──
    {
      id: "tc-ethics", name: "Ethics of AI",
      description: "Delineates the essential ethical values, principles, regulations and practical ethical rules that teachers need to understand and apply.",
      order: 2, icon: "Shield", color: "text-blue-600",
      levels: [
        {
          ...ACQUIRE_DEEPEN_CREATE[0], id: "tc-et-acquire",
          name: "Ethical principles",
          description: "Teachers have a basic understanding of ethical issues surrounding AI and of the principles required for ethical human-AI interactions.",
          target: "Teachers with limited or no prior AI knowledge or skills.",
          curricularGoals: [
            { id: "CG2.1.1", description: "Surface ethical controversies through a critical examination of use cases of AI tools in education." },
            { id: "CG2.1.2", description: "Facilitate an understanding of essential ethical principles through an examination of use cases related to each of the core ethical principles: do no harm; proportionality; non-discrimination; sustainability; human determination; and transparency and explainability." },
            { id: "CG2.1.3", description: "Build an association between ethical principles and standards through examples of local, national or international regulations regarding the ethics of AI." },
            { id: "CG2.1.4", description: "Advocate for inclusivity in the use of AI and guide teachers to discuss the risks that specific AI tools can pose to inclusion and equity." },
          ],
          indicators: [
            { id: "LO2.1.1", description: "Exemplify fundamental ethical controversies in the use of concrete AI tools from the perspectives of human agency, security, privacy, and linguistic and cultural relevance." },
            { id: "LO2.1.2", description: "Explain the core ethical principles and internalize them through their personal selection and use of AI." },
            { id: "LO2.1.3", description: "Match key articles of regulations with ethical principles and understand their implications for education." },
            { id: "LO2.1.4", description: "Prioritize actions to minimize the negative impact of AI on equity and inclusion when using AI tools in education." },
          ],
          contextualActivities: [
            { id: "CA2.1.1", name: "Perspective taking in ethical dilemmas", description: "Adopt an ethical perspective on the use of AI in schools based on an understanding of multiple dilemmas around privacy, human agency, equity, inclusion, local cultures and languages, and climate change." },
            { id: "CA2.1.2", name: "Knowledge-mapping of ethical principles", description: "Apply basic knowledge-mapping tools to visualize the connections among the different core principles, responses to associated controversies, and their correspondence with regulations." },
            { id: "CA2.1.3", name: "Personal observation of local regulations", description: "Observe whether local AI regulations keep pace with iterations of AI technologies and evaluate applicable regulations." },
            { id: "CA2.1.4", name: "Biases of AI tools", description: "Be mindful of biases of AI tools used in schools and their potential to exclude or marginalize persons with disabilities and students from vulnerable groups." },
          ],
        },
        {
          ...ACQUIRE_DEEPEN_CREATE[1], id: "tc-et-deepen",
          name: "Safe and responsible use",
          description: "Teachers are expected to internalize essential ethical rules for the safe and responsible use of AI, including respecting data privacy, intellectual property rights and other legal frameworks.",
          target: "Teachers who already have some knowledge of AI and some experience of using it in education.",
          curricularGoals: [
            { id: "CG2.2.1", description: "Deepen teachers' understanding of main threats to AI safety at the stages of design and use through analysing case scenarios on typical AI safety risks." },
            { id: "CG2.2.2", description: "Facilitate analyses of typical legal duties when using AI and of the consequences of breaching them." },
            { id: "CG2.2.3", description: "Support teachers to build the association between compliance with regulations on the safe and responsible use of AI and their local contexts." },
          ],
          indicators: [
            { id: "LO2.2.1", description: "Explain typical issues related to AI safety both at institutional and personal levels including: safety by design, safety by use, data ownership, data sovereignty, data privacy." },
            { id: "LO2.2.2", description: "Demonstrate familiarity with locally applicable regulations to protect data privacy and ensure AI safety." },
            { id: "LO2.2.3", description: "Implement measures to safeguard their own and their students' data privacy, ensuring data is collected, used, shared, archived and deleted with consent." },
            { id: "LO2.2.4", description: "Apply guidelines to ensure responsible use of AI by teachers and students in compliance with ethical principles." },
          ],
          contextualActivities: [
            { id: "CA2.2.1", name: "Personal AI safety tracker", description: "Draw and update a conceptual map of typical AI safety issues and frequent incidents, possible threats to institutions and individuals, and mitigation measures." },
            { id: "CA2.2.2", name: "Whitelist personal collections of AI tools", description: "Review the safety of their personal collections of AI tools looking at owners, design ethics, data sources, algorithms, inclusive accessibility and functionality choices." },
            { id: "CA2.2.3", name: "Iteratively update list of dos and don'ts", description: "Observe and evaluate cases of high-risk and irresponsible AI use in schools, and iteratively update the list of dos and don'ts for teachers and students." },
          ],
        },
        {
          ...ACQUIRE_DEEPEN_CREATE[2], id: "tc-et-create",
          name: "Co-creating ethical rules",
          description: "Teachers are able to champion the ethics of AI through critical advocacy, leading discussions that address ethical, sociocultural and environmental concerns, and contributing to the co-creation of ethical rules.",
          target: "Teachers who have strong AI knowledge and skills as well as rich experience in using AI in education.",
          curricularGoals: [
            { id: "CG2.3.1", description: "Foster inquiry into the social impact of AI by organizing teachers' research-based reviews of the social impact of selected AI tools." },
            { id: "CG2.3.2", description: "Enhance critical examination of existing users' guidance published by AI providers." },
            { id: "CG2.3.3", description: "Upgrade knowledge on AI ethics and skills to guide further iterations of ethical rules and standards." },
          ],
          indicators: [
            { id: "LO2.3.1", description: "Critically analyse the social impact of AI from both the global and local perspectives including potential impact on social equity, inclusion, linguistic and cultural diversity, and planetary well-being." },
            { id: "LO2.3.2", description: "Assess the appropriateness and sufficiency of guidance for users of a specific AI tool against the ethical risks rooted in its design." },
            { id: "LO2.3.3", description: "Solidify the view that regulations on AI ethics must be designed by and for human stakeholders; advocate for and participate in the development of regulatory frameworks." },
          ],
          contextualActivities: [
            { id: "CA2.3.1", name: "Localized global view on the social impact of AI", description: "Holistically review the social impact of AI on individual human rights, economic activity, social justice and planetary well-being." },
            { id: "CA2.3.2", name: "Spotlighting ethical gaps in users' guidance", description: "Audit the claims made by AI tool providers against a full list of risks and social impacts." },
            { id: "CA2.3.3", name: "Master teachers as advocates of AI ethics", description: "Play active roles in launching awareness campaigns on the ethics of AI, interpreting ethical principles, and sharing knowledge on relevant regulations." },
            { id: "CA2.3.4", name: "Co-designing ethical prototypes of AI tools", description: "Launch a hypothetical AI development project bringing together teachers, students and technologists to co-design an ethical AI tool." },
          ],
        },
      ],
    },
    // ── Aspect 3: AI foundations and applications ──
    {
      id: "tc-foundations", name: "AI foundations and applications",
      description: "Specifies the conceptual knowledge and transferable operational skills teachers need to understand and apply to support their selection, application and creative customization of AI tools.",
      order: 3, icon: "Lightbulb", color: "text-amber-600",
      levels: [
        {
          ...ACQUIRE_DEEPEN_CREATE[0], id: "tc-fo-acquire",
          name: "Basic AI techniques and applications",
          description: "Teachers are expected to acquire basic conceptual knowledge on AI, including the definition of AI, how AI models are trained, main categories of AI technologies, and the capacity to examine appropriateness of AI tools for education.",
          target: "Teachers with limited or no prior AI knowledge or skills.",
          curricularGoals: [
            { id: "CG3.1.1", description: "Adapt the level of difficulty of basic conceptual knowledge on AI according to teachers' responsibilities and prior experience; illustrate how a specific AI tool is developed based on data and algorithms." },
            { id: "CG3.1.2", description: "Support the hands-on operation of AI tools relevant to teachers' responsibilities to give a basic understanding of how these tools work." },
            { id: "CG3.1.3", description: "Support users' testing of AI tools by introducing a rudimentary method for analysing the reliability and appropriateness of specific AI tools." },
            { id: "CG3.1.4", description: "Support teachers to establish their own collection of AI tools, starting from recommending basic exemplar tools and guiding them to curate trustable AI." },
          ],
          indicators: [
            { id: "LO3.1.1", description: "Demonstrate conceptual knowledge on how AI systems are developed using data, algorithms and computing architecture; exemplify key steps including problem-scoping, design, training, testing, deployment, feedback and iteration." },
            { id: "LO3.1.2", description: "Exemplify what AI is and is not, the main categories of AI techniques, the novel capabilities compared to previous ICT tools, and the core functions of various categories of AI tools." },
            { id: "LO3.1.3", description: "Locate and operate AI tools that are necessary for their daily work in local contexts." },
            { id: "LO3.1.4", description: "Explain the importance of evaluating AI tools to ensure their accessibility, inclusivity, and reliability; undertake basic analyses of appropriateness for education." },
            { id: "LO3.1.5", description: "Start consolidating a personal collection of trustable AI tools relevant to the local language and culture." },
          ],
          contextualActivities: [
            { id: "CA3.1.1", name: "Conceptual mapping of how AI works", description: "Start to draw and iteratively update concept maps showing how AI systems are developed." },
            { id: "CA3.1.2", name: "Extension and enhancement of skills", description: "Extend knowledge on AI tools relevant to the teachers' responsibilities and enhance operational skills." },
            { id: "CA3.1.3", name: "Navigation compass for selection of AI tools", description: "Discern which tools are using AI and which ones are not, and the basic comparative advantages and limitations." },
            { id: "CA3.1.4", name: "Collection of appropriate AI tools", description: "Cooperate with other teachers and school managers to assess and collect validated AI tools, share open-source tools." },
          ],
        },
        {
          ...ACQUIRE_DEEPEN_CREATE[1], id: "tc-fo-deepen",
          name: "Application skills",
          description: "Teachers are expected to proficiently operate AI tools in educational settings; deepen their knowledge of various categories of AI technologies and practical skills concerning data and algorithms.",
          target: "Teachers who already have some knowledge of AI and some experience of using it in education.",
          curricularGoals: [
            { id: "CG3.2.1", description: "Enrich operation and comparison experiences of typical AI tools; guide teachers to analyse similarities and differences of common AI techniques." },
            { id: "CG3.2.2", description: "Scaffold deepened construction of conceptual knowledge by facilitating teachers' research-based learning on how AI systems are trained and tested." },
            { id: "CG3.2.3", description: "Support problem-based learning of operational skills in data, algorithms and coding." },
            { id: "CG3.2.4", description: "Offer hands-on practice to assess the 'ethics by design' of AI tools." },
          ],
          indicators: [
            { id: "LO3.2.1", description: "Proficiently operate commonly used AI tools and exemplify the typical techniques used by these tools." },
            { id: "LO3.2.2", description: "Visually represent how selected AI systems work, including how they are trained and tested." },
            { id: "LO3.2.3", description: "Demonstrate transferable knowledge on data, algorithms and coding and apply it to solve problems." },
            { id: "LO3.2.4", description: "Critically apply knowledge related to data, training, algorithms and models to assess the ethics rooted in the design of AI tools." },
          ],
          contextualActivities: [
            { id: "CA3.2.1", name: "Skillful uses of AI tools in schools", description: "Based on a deepened understanding, skillfully operate widely used AI tools." },
            { id: "CA3.2.2", name: "Visualized know-how on typical categories of AI tools", description: "Draw a concept map or visualized workflow to explain how selected AI systems work." },
            { id: "CA3.2.3", name: "Facilitating students to learn about data, algorithms and coding", description: "Facilitate students or peer teachers to acquire knowledge of data, algorithms and coding." },
            { id: "CA3.2.4", name: "Informed whistleblowing in ethics by design", description: "Apply understanding of how AI is trained to investigate biases and discrimination that may be rooted in datasets and algorithms." },
          ],
        },
        {
          ...ACQUIRE_DEEPEN_CREATE[2], id: "tc-fo-create",
          name: "Creating with AI",
          description: "Teachers are able to customize or modify AI tools in a proficient manner, applying enhanced conceptual knowledge and operational skills to create AI-assisted inclusive learning environments.",
          target: "Teachers who have strong AI knowledge and skills as well as rich experience in using AI in education.",
          curricularGoals: [
            { id: "CG3.3.1", description: "Nurture adaptability and creativity in customizing AI tools; support teachers to integrate skills on data, algorithms, programming and AI models." },
            { id: "CG3.3.2", description: "Foster critical views on open-source AI by supporting teachers to deepen views on advantages, limitations and risks of open-source vs commercial AI tools." },
            { id: "CG3.3.3", description: "Simulate and practice adaptability and creativity in co-creating AI tools through project-based learning." },
            { id: "CG3.3.4", description: "Support teachers to embed values, knowledge and skills into existing repositories of educational AI tools." },
          ],
          indicators: [
            { id: "LO3.3.1", description: "Demonstrate knowledge and skills on AI system design at the level of expert teachers as well as comprehensive competencies to analyse limitations of AI systems." },
            { id: "LO3.3.2", description: "Apply knowledge and skills on data, algorithms, programming and AI models to customize or assemble existing tools or fine-tune open-source AI systems." },
            { id: "LO3.3.3", description: "Revise or define criteria for the comprehensive testing of a self-created AI tool." },
            { id: "LO3.3.4", description: "Contribute to a repository of user-created or tailored AI tools based on personal and institutional needs." },
          ],
          contextualActivities: [
            { id: "CA3.3.1", name: "Driving the design of AI tools for inclusion", description: "Collaborate with a community of co-creators to add functions to existing AI tools or design new ones to facilitate accessibility." },
            { id: "CA3.3.2", name: "Promoting the co-creation of AI tools to support climate-friendly actions", description: "Co-create AI tools or organize hackathons to design tools that promote climate education or climate-friendly actions." },
            { id: "CA3.3.3", name: "Coordinating repositories of educational AI tools", description: "Support the creation of a repository of selected trustable and self-created AI tools that can be shared publicly." },
          ],
        },
      ],
    },
    // ── Aspect 4: AI pedagogy ──
    {
      id: "tc-pedagogy", name: "AI pedagogy",
      description: "Proposes a set of competencies required for purposeful and effective AI-pedagogy integration, covering comprehensive competencies to validate and select appropriate AI tools and integrate them with pedagogical methods.",
      order: 4, icon: "GraduationCap", color: "text-purple-600",
      levels: [
        {
          ...ACQUIRE_DEEPEN_CREATE[0], id: "tc-pd-acquire",
          name: "AI-assisted teaching",
          description: "Teachers are expected to be able to identify and leverage the pedagogical benefits of AI tools to facilitate subject-specific lesson planning, teaching and assessment while mitigating the risks.",
          target: "Teachers with limited or no prior AI knowledge or skills.",
          curricularGoals: [
            { id: "CG4.1.1", description: "Organize lesson analyses based on exemplar videos of teachers using AI tools in the classroom; facilitate understanding of appropriateness of these tools." },
            { id: "CG4.1.2", description: "Encourage teachers to be mindful of scholarly research on the use of AI to support pedagogical activities." },
            { id: "CG4.1.3", description: "Facilitate the transferability of foundational knowledge and skills on AI to teaching by presenting locally accessible and validated AI tools." },
            { id: "CG4.1.4", description: "Facilitate the pedagogical validation of AI and instructional design on AI-assisted teaching including the design-implementation-reflection cycle." },
          ],
          indicators: [
            { id: "LO4.1.1", description: "Demonstrate familiarity with a human-centred mindset, ethical principles and domain-appropriate pedagogical methodologies to analyse sample lessons." },
            { id: "LO4.1.2", description: "Exemplify the main categories of AI systems and applications designed to assist teaching, learning and assessment." },
            { id: "LO4.1.3", description: "Demonstrate familiarity with the use of basic instructional design methods to guide decisions on whether and when to use AI." },
            { id: "LO4.1.4", description: "Find and use basic educational AI tools and/or operate institutionally deployed AI systems." },
          ],
          contextualActivities: [
            { id: "CA4.1.1", name: "Starting from basic teaching needs", description: "Delineate basic needs in teaching and learning assessment. Start from basic needs as the first principle to understand whether a specific AI tool is appropriate." },
            { id: "CA4.1.2", name: "Iterative design-implementation-reflection cycle", description: "Learn and gradually improve ability to design and deliver appropriate AI-assisted teaching through an iterative loop." },
            { id: "CA4.1.3", name: "Evaluating effectiveness against needs", description: "Gain first-hand experience of the limitations, risks and benefits of AI for teaching and learning." },
          ],
        },
        {
          ...ACQUIRE_DEEPEN_CREATE[1], id: "tc-pd-deepen",
          name: "AI-pedagogy integration",
          description: "Teachers are able to adeptly integrate AI into the design and facilitation of student-centred learning practices to foster engagement, support differentiated learning and enhance teacher-student interactions.",
          target: "Teachers who already have some knowledge of AI and some experience of using it in education.",
          curricularGoals: [
            { id: "CG4.2.1", description: "Design and organize learning strategies based on videos of exemplar AI-enhanced learning practice; support teachers to analyse the impact of AI on learning processes." },
            { id: "CG4.2.2", description: "Deepen understanding of the impact of AI by encouraging teachers to discuss selected research reports on impacts of AI on students' agency, thinking and learning." },
            { id: "CG4.2.3", description: "Support the integrated deployment of foundational knowledge and skills on AI to meet the needs of teaching, learning and assessment." },
            { id: "CG4.2.4", description: "Support the transfer from instructional design to learning design in the context of the validation and pedagogical use of AI." },
          ],
          indicators: [
            { id: "LO4.2.1", description: "Adeptly integrate ethical principles, student-centred pedagogical methodologies and interdisciplinary perspectives into learning design practices." },
            { id: "LO4.2.2", description: "Critically evaluate whether various categories of AI present advantages in assisting formative assessment, monitoring learning processes, and enhancing student-centric teaching." },
            { id: "LO4.2.3", description: "Critically examine the appropriateness of AI in formative learning assessment and high-stake examinations; adeptly blend appropriate tools in facilitating AI-assisted formative assessments." },
          ],
          contextualActivities: [
            { id: "CA4.2.1", name: "Mapping of AI tools and application skills", description: "Update the concept map of AI tools to reflect key features and evaluate their pedagogical affordance." },
            { id: "CA4.2.2", name: "Insights into pedagogical assumptions behind AI tools", description: "Cooperate with peers or experts to examine whether the design of general AI systems considers pedagogical implications." },
            { id: "CA4.2.3", name: "Designing students' use of AI for higher-order thinking", description: "Design student-centric teaching and learning activities based on validated educational AI tools." },
            { id: "CA4.2.4", name: "Human-accountable AI-assisted assessments", description: "Debunk myths around the use of AI to automate assessments by examining the risks of AI in usurping human accountability." },
          ],
        },
        {
          ...ACQUIRE_DEEPEN_CREATE[2], id: "tc-pd-create",
          name: "AI-enhanced pedagogical innovation",
          description: "Teachers are able to critically assess AI's impact on teaching, learning and assessment; plan and facilitate AI-immersed learning scenarios; and leverage data to explore student-centred pedagogical innovation.",
          target: "Teachers who have strong AI knowledge and skills as well as rich experience in using AI in education.",
          curricularGoals: [
            { id: "CG4.3.1", description: "Inspire ideas on possible scenarios where AI is used for students' development; facilitate teachers' review of their readiness and empower them to frame feasible innovative learning practices." },
            { id: "CG4.3.2", description: "Scaffold teachers' insights into the interplay between pedagogical principles and pedagogical transformations that AI could trigger." },
            { id: "CG4.3.3", description: "Support the improvisation of skills to create new AI tools or expand existing ones for inquiry- and project-based learning." },
            { id: "CG4.3.4", description: "Incubate the transfer from learning design to scenario design. Organize hands-on practice where teachers co-design curricular practices or human-AI interactive scenarios." },
          ],
          indicators: [
            { id: "LO4.3.1", description: "Critically examine the dynamic interaction between AI advancement and the evolution of pedagogical methodologies; design evidence-based tests of open learning options to harness AI for inquiry-based learning." },
            { id: "LO4.3.2", description: "Assemble AI tools or co-create new AI applications to address needs for inclusive accessibility, personalized learning, and project-based learning." },
            { id: "LO4.3.3", description: "Adeptly design AI-augmented learning scenarios that promote students' higher-order inquiry, project-based learning, critical thinking and co-creations." },
            { id: "LO4.3.4", description: "Design and integrate the use of AI to support learning analytics and adjustment of teaching strategies." },
            { id: "LO4.3.5", description: "Adeptly use AI to generate content across text, audio and video to support co-creation of curricular resources." },
            { id: "LO4.3.6", description: "Streamline the use of AI for teachers' administrative tasks, teaching and learning tasks, engagement with parents and local communities." },
          ],
          contextualActivities: [
            { id: "CA4.3.1", name: "Guiding pedagogical uses of AI while opening new horizons", description: "Uphold human-centred pedagogical principles to guide the design and uses of AI and explore whether existing methodologies are sufficient." },
            { id: "CA4.3.2", name: "Engineering triangular interactions between teachers, students and AI", description: "Navigate the teacher-AI-student triangular relations; design and engineer desirable scenarios of teacher-student, teacher-AI, student-AI and teacher-AI-student interactions." },
            { id: "CA4.3.3", name: "AI empowering students with special needs", description: "Promote assistive AI or co-create assistive AI tools to provide students with disabilities opportunities for empowerment." },
            { id: "CA4.3.4", name: "Human-AI hybrid approach to curricular resources", description: "Continuously engage in the use of AI to facilitate review and production of inclusive curricular resources." },
          ],
        },
      ],
    },
    // ── Aspect 5: AI for professional development ──
    {
      id: "tc-professional", name: "AI for professional development",
      description: "Outlines the emerging competencies teachers need to use AI to drive their own lifelong professional learning and collaborative professional development in view of transforming teaching practice.",
      order: 5, icon: "Sparkles", color: "text-emerald-600",
      levels: [
        {
          ...ACQUIRE_DEEPEN_CREATE[0], id: "tc-pl-acquire",
          name: "Enabling lifelong professional learning",
          description: "Teachers are expected to explore the use of AI tools to enhance their professional development and reflective practices, assess learning needs, and personalize learning pathways.",
          target: "Teachers with limited or no prior AI knowledge or skills.",
          curricularGoals: [
            { id: "CG5.1.1", description: "Nurture teachers' motivation for lifelong professional learning in the AI era by engaging teachers in discussion on the educational implications of rapid AI development." },
            { id: "CG5.1.2", description: "Guide self-assessment on teachers' AI readiness and identify competency gaps." },
            { id: "CG5.1.3", description: "Build awareness of teacher-facing AI by introducing tools that can support professional development." },
            { id: "CG5.1.4", description: "Facilitate the leveraging of AI for professional learning, including content-recommendation platforms." },
          ],
          indicators: [
            { id: "LO5.1.1", description: "Describe the evolution of teachers' rights, working conditions, qualifications and required competencies in the AI era." },
            { id: "LO5.1.2", description: "Exemplify the new knowledge, skills and values required by the teaching profession in local contexts in the AI era." },
            { id: "LO5.1.3", description: "List various AI tools that can be used to support self-assessment, reflective practices and professional learning." },
            { id: "LO5.1.4", description: "Locate and apply teacher-facing AI tools that are affordable and relevant for self-assessment and professional learning." },
          ],
          contextualActivities: [
            { id: "CA5.1.1", name: "Awareness of teachers' basic rights and obligations in the AI era", description: "Delineate the rights that should be protected and the basic working conditions and guidance that should be provided for teachers." },
            { id: "CA5.1.2", name: "Self-assessment of readiness for teaching in the AI era", description: "Conduct assessments of their own readiness and competency gaps and devise possible roadmaps for professional development." },
            { id: "CA5.1.3", name: "Human-directed use of AI to open professional learning horizons", description: "Gain experience using AI-assisted social media to prompt new ideas and recommend peers for coaching or mentoring." },
          ],
        },
        {
          ...ACQUIRE_DEEPEN_CREATE[1], id: "tc-pl-deepen",
          name: "AI to enhance organizational learning",
          description: "Teachers are able to confidently utilize AI tools for tailored participation in collaborative professional learning communities.",
          target: "Teachers who already have some knowledge of AI and some experience of using it in education.",
          curricularGoals: [
            { id: "CG5.2.1", description: "Incite continuous motivation for professional learning and collaboration, supporting teachers to conduct research on how master teachers adapt in AI-rich settings." },
            { id: "CG5.2.2", description: "Facilitate knowledge expansion on AI tools for professional development, introducing locally accessible emerging tools." },
            { id: "CG5.2.3", description: "Deepen teachers' operational skills in the use of data analytics to support professional learning." },
            { id: "CG5.2.4", description: "Offer hands-on practice on assessing deeper ethical issues associated with using AI systems for professional learning." },
          ],
          indicators: [
            { id: "LO5.2.1", description: "Critically analyse their roles in designing and facilitating students' use of AI in their own pedagogical practices." },
            { id: "LO5.2.2", description: "Apply foundational knowledge on data using AI tools to track and analyse their own professional development." },
            { id: "LO5.2.3", description: "Expand knowledge and skills on the use of AI, especially emerging tools, for their own professional development." },
            { id: "LO5.2.4", description: "Evaluate the ethical risks of AI algorithms behind social media platforms and specialized tools; develop guidelines for effective use." },
          ],
          contextualActivities: [
            { id: "CA5.2.1", name: "Autonomous upskilling and peer coaching", description: "Keep pace with emerging AI technologies and their implications, autonomously upskilling and coaching peers." },
            { id: "CA5.2.2", name: "Using data analytics for self-regulated professional development", description: "Apply knowledge on data, algorithms and AI models to draw up analytics of teachers' own professional knowledge and skills." },
            { id: "CA5.2.3", name: "Generative AI simulations for professional development", description: "Utilize existing generative AI tools or customize new ones to create an AI coach that simulates specific professional development scenarios." },
            { id: "CA5.2.4", name: "Human-controlled uses of AI for collaborative professional development", description: "Uncover ethical risks of AI-manipulated platforms and design human-controlled activities to leverage AI for collaborative professional development." },
          ],
        },
        {
          ...ACQUIRE_DEEPEN_CREATE[2], id: "tc-pl-create",
          name: "AI to support professional transformation",
          description: "Teachers are able to customize and modify AI tools to enhance their professional development and continuously test strategies on the effective use of AI.",
          target: "Teachers who have strong AI knowledge and skills as well as rich experience in using AI in education.",
          curricularGoals: [
            { id: "CG5.3.1", description: "Motivate teachers to be agents of change by organizing case studies and discussions on how expert teachers could champion the transformation of education." },
            { id: "CG5.3.2", description: "Enhance skills to use AI to support institutional professional learning through hands-on workshops." },
            { id: "CG5.3.3", description: "Support teachers to customize or assemble AI tools to enable access to professional development for peers with disabilities." },
            { id: "CG5.3.4", description: "Nurture the traits of being creative users of AI to foster self-actualization and transformation." },
          ],
          indicators: [
            { id: "LO5.3.1", description: "Show commitment and persistence in the co-creation and usage of AI tools and methods to fulfil professional and social responsibilities." },
            { id: "LO5.3.2", description: "Blend AI tools and human coaching to facilitate well-informed self-reflection, goal setting and mobilization of knowledge." },
            { id: "LO5.3.3", description: "Where possible, configure or create AI solutions to monitor and critically assess organization-wide professional learning trajectories." },
            { id: "LO5.3.4", description: "Understand the roles of AI to support self-actualization and personalize citizenship in the AI era; contribute to communities' co-creation of AI tools." },
          ],
          contextualActivities: [
            { id: "CA5.3.1", name: "Human-AI hybrid coach for teachers", description: "Build or utilize generative AI toolkits to customize an AI-assisted agent or coach for teachers' professional development." },
            { id: "CA5.3.2", name: "AI-enhanced design of training programmes", description: "Leverage AI tools to expand reviews of existing programmes, extend ideas on training content, and assist production of training courses." },
            { id: "CA5.3.3", name: "Communities for co-creation of AI tools and pedagogical innovations", description: "Lead or engage in collaborative research teams working on innovative pedagogical methodologies and co-creation of trustable AI tools." },
          ],
        },
      ],
    },
  ],
  metadata: {
    publicationYear: 2024,
    totalCompetencies: 15,
    isbn: "978-92-3-100707-1",
    doi: "https://doi.org/10.54675/ZJTE2084",
    license: "CC-BY-SA 3.0 IGO",
    ethicalPrinciples: ["Do no harm", "Proportionality", "Non-discrimination", "Sustainability", "Human determination in human-AI interaction", "Transparency and explainability"],
    crossCuttingThemes: ["Inclusion and accessibility for students with disabilities", "Linguistic and cultural diversity", "Environmental sustainability", "Open-source AI tools", "Human agency and accountability", "Data privacy and protection"],
    implementationStrategies: ["Regulate AI and ensure trustworthy AI tools for education", "Build enabling policies and conditions for the use of AI in education", "Formulate and adopt local AI competency frameworks for teachers", "Design and streamline training and support programmes on AI competencies", "Develop contextual performance-based assessment tools"],
  },
  useCases: ["Designing AI-enhanced lesson plans", "Integrating AI into assessment and feedback", "Leading ethical AI discussions in the classroom", "Building professional development programmes", "Conducting self-assessment of AI readiness", "Co-creating AI tools for education"],
  crossReferences: ["student-competency", "guidance-policy"],
  assessmentQuestions: [
    { id: "q1", dimension: "Human-centred mindset", question: "How comfortable are you considering ethical implications when using AI tools in teaching?", options: [
      { value: "a1", label: "I'm just beginning to learn about AI ethics", level: "acquire" },
      { value: "a2", label: "I regularly consider ethics but need guidance", level: "deepen" },
      { value: "a3", label: "I lead ethical AI discussions and mentor others", level: "create" },
    ]},
    { id: "q2", dimension: "AI foundations and applications", question: "How well do you understand how AI systems work and their limitations?", options: [
      { value: "b1", label: "I have basic awareness of AI concepts", level: "acquire" },
      { value: "b2", label: "I can explain AI capabilities to students", level: "deepen" },
      { value: "b3", label: "I design learning activities around AI principles", level: "create" },
    ]},
    { id: "q3", dimension: "AI pedagogy", question: "To what extent do you integrate AI tools into lesson planning and assessment?", options: [
      { value: "c1", label: "I'm exploring AI tools with guidance", level: "acquire" },
      { value: "c2", label: "I regularly use AI in lesson design", level: "deepen" },
      { value: "c3", label: "I create innovative AI-enhanced curricula", level: "create" },
    ]},
    { id: "q4", dimension: "Ethics of AI", question: "How do you address bias and fairness when using AI in your teaching?", options: [
      { value: "d1", label: "I'm learning to identify bias in AI outputs", level: "acquire" },
      { value: "d2", label: "I implement bias checks in my AI workflows", level: "deepen" },
      { value: "d3", label: "I develop bias detection frameworks for colleagues", level: "create" },
    ]},
    { id: "q5", dimension: "AI for professional development", question: "How actively do you engage in professional development around AI in education?", options: [
      { value: "e1", label: "I occasionally attend AI training sessions", level: "acquire" },
      { value: "e2", label: "I actively seek AI learning opportunities", level: "deepen" },
      { value: "e3", label: "I facilitate AI professional development", level: "create" },
    ]},
  ],
  assessmentTitle: "Teacher AI Competency Self-Assessment",
  assessmentDescription: "Discover your current competency level across 5 key dimensions of the UNESCO AI CFT",
  showInQuiz: true,
  showInDashboard: true,
  showInLanding: true,
  isBackgroundFramework: false,
  compatibility: [],
  sourceFidelity: "official",
  estimatedAssessmentMinutes: 30,
};

// ────────────────────────────────────────────────
// 3. UNESCO Student AI Competency
// ────────────────────────────────────────────────
const studentCompetency: Framework = {
  id: "student-competency",
  name: "UNESCO AI Competency Framework for Students",
  shortName: "Student Competency",
  description: "4 aspects across 3 progression levels building AI literacy and system design skills",
  type: "competency",
  scope: "individual",
  source: "UNESCO",
  path: "/frameworks/student-competency",
  icon: "Brain",
  color: "text-indigo-600",
  badgeLabel: "UNESCO Framework",
  targetAudience: ["student"],
  overview: `The UNESCO AI Competency Framework for Students (2024) is the first global framework defining the values, foundational knowledge and transferable skills students need to critically understand and use AI systems in a safe, effective and meaningful manner. It outlines 12 competencies across four aspects and three progression levels (Understand, Apply, Create), supporting students to grow towards being not only effective and ethical users of AI tools, but also co-creators in the design of more inclusive and environmentally sustainable AI.`,
  keyPrinciples: [
    { id: "principle-1", name: "Fostering a critical approach to AI", description: "Students should be supported to become active co-creators of AI, as well as potential leaders who will define further iterations of AI and its interactions with human society." },
    { id: "principle-2", name: "Prioritizing human-centred interaction with AI", description: "The design and use of AI should serve the development of human capabilities, protect human dignity and agency, and promote justice and sustainability throughout the entire AI life cycle." },
    { id: "principle-3", name: "Encouraging environmentally sustainable AI", description: "Students need a critical understanding of the adverse environmental impact of profit-driven approaches to the design, training and deployment of AI models." },
    { id: "principle-4", name: "Promoting inclusivity in AI competency development", description: "All students should have inclusive access to the environments required for learning about AI at the basic level, and they should be supported to learn how to embed the principle of inclusivity into the design of AI." },
    { id: "principle-5", name: "Building core AI competencies for lifelong learning", description: "Core competencies must include values associated with an ethical and human-centred mindset. The core competencies are brand-agnostic and product-agnostic." },
  ],
  keyDimensions: [
    // ── Aspect 1: Human-centred mindset ──
    {
      id: "sc-human-centred", name: "Human-centred mindset",
      description: "Focuses on students' values, beliefs and critical thinking skills, applied to the examination of whether AI is fit for purpose, whether its use is justified, how humans should interact with it, and what responsibilities individuals and institutions should take on to contribute to the building of safe, inclusive and just AI societies.",
      order: 1, icon: "Heart", color: "text-rose-600",
      levels: [
        {
          id: "sc-hc-understand", name: "Human agency", description: "Students are expected to recognize that AI is human-led and that the decisions of AI creators influence how AI systems impact human rights, human-AI interaction, and their own lives and societies.",
          order: 1, target: "All students",
          curricularGoals: [
            { id: "CG1.1.1", description: "Foster an understanding that AI is human-led: Based on selected AI tools, explain that AI is human-led; facilitate students to develop a stepwise and integral comprehension of human agency covering principles on data ownership, data privacy, protection of human rights, explainability, human control in deployment, and human determination in decision-making." },
            { id: "CG1.1.2", description: "Facilitate an understanding on the necessity of exercising sufficient human control over AI: Expose students to real-world scenarios and guide them to experience the consequences of human oversight in controlling AI." },
            { id: "CG1.1.3", description: "Nurture critical thinking on the dynamic relationship between human agency and machine agency: Expose students to real-world cases in which AI can support human agency; guide students in holding conflict-based debates on dynamic boundaries between human and AI agency." },
          ],
          contextualActivities: [
            { id: "CA1.1.1", name: "Visualizing the abstract concept of human agency throughout the AI life cycle", description: "Ask students to draw concept maps of human agency in key steps of the life cycle of selected AI tools, including data ownership, data privacy, explainability, human-controlled evaluation, and human determination in decision-making." },
            { id: "CA1.1.2", name: "Simulating an AI Act courtroom debate to evaluate creators' intents underlying prohibited AI systems", description: "Based on an age-appropriate interpretation of AI systems prohibited under the EU AI Act, organize students to act as jury members to evaluate selected examples and deliberate on creators' intents and motivations." },
            { id: "CA1.1.3", name: "Scenario-based understanding of human-controlled interaction with AI", description: "Select examples or scenarios in which AI tools are used in workplaces or daily life. Encourage students to recognize AI's contribution in scenarios where human capabilities have limitations, underlining using AI to enhance human capacities while ensuring human control." },
            { id: "CA1.1.4", name: "Debating the dynamic boundary between human agency and machine agency", description: "Based on real-world cases of dilemmas surrounding humans' reliance on machine agency, encourage students to conduct a debate on the changing roles humans and AI may play in AI-supported problem-solving." },
          ],
          learningEnvironments: [
            "Unplugged learning settings like paper-based articles, printed reading materials and worksheets.",
            "Locally available AI tools including mobile phones with AI applications.",
            "Predownloaded or recorded videos and other resources related to specific case studies or scenarios.",
            "Search engines, online videos and supplemental online learning courses.",
          ],
          indicators: [
            { id: "sc-hc-u1", description: "Recognise that AI is human-led and that the decisions of AI creators influence how AI systems impact human rights" },
            { id: "sc-hc-u2", description: "Understand the implications of protecting human agency throughout the design, provision and use of AI" },
            { id: "sc-hc-u3", description: "Understand what it means for AI to be human-controlled and consequences when that is not the case" },
          ],
        },
        {
          id: "sc-hc-apply", name: "Human accountability", description: "Students are expected to recognize that human accountabilities are the legal obligations of AI creators and AI service providers, and understand what human accountabilities they should assume during the design and use of AI.",
          order: 2, target: "All school students",
          curricularGoals: [
            { id: "CG1.2.1", description: "Develop a view that human accountability is a legal obligation of AI creators and AI service providers: Guide students to understand that human AI creators and service providers are accountable for legal issues, violations and infringements that the AI system may cause." },
            { id: "CG1.2.2", description: "Generate the understanding that human accountability is a legal and social responsibility when using AI in making decisions about humanity: Guide students to critically interrogate the genuine capabilities of certain AI tools and debunk the hype." },
            { id: "CG1.2.3", description: "Nurture the personal attitude that human accountability requires personal competencies to steer the purposeful use of AI: Guide students to interrogate how the automation of literature reviews, writing and artistic creation may undermine human thinking processes." },
          ],
          contextualActivities: [
            { id: "CA1.2.1", name: "Writing guidelines on human accountability for AI creators and service providers", description: "Facilitate students to play the roles of AI creators and data owners and discuss their key legal and ethical accountabilities in terms of maintaining human control." },
            { id: "CA1.2.2", name: "Investigating the impact of AI-assisted decisions on humans and avenues of redress within AI regulations", description: "Ask students to find examples in which decisions about humans are determined or greatly influenced by AI, and check whether human accountability is in compliance with applicable regulations." },
            { id: "CA1.2.3", name: "Scenario-based practices of using AI with purpose", description: "Engage students in activities where they use AI tools to purposefully practise their writing skills and foster inquiry-based learning, higher-order thinking and creativity." },
          ],
          learningEnvironments: [
            "Unplugged and/or offline learning settings and resources, including print-based case studies, role-play scripts, videos, worksheets and flipcharts.",
            "Online AI tools, for example learning management systems, social media platforms and generative AI platforms.",
          ],
          indicators: [
            { id: "sc-hc-a1", description: "Recognise that human accountabilities are the legal obligations of AI creators and AI service providers" },
            { id: "sc-hc-a2", description: "Understand what human accountabilities they should assume during the design and use of AI" },
            { id: "sc-hc-a3", description: "Foster awareness that humans should not cede determination to AI when making high-stakes decisions" },
          ],
        },
        {
          id: "sc-hc-create", name: "Citizenship in the era of AI", description: "Students are expected to build critical views on the impact of AI on human societies and expand their human-centred values to promoting the design and use of AI for inclusive and sustainable development.",
          order: 3, target: "Students with strong interest in AI innovation",
          curricularGoals: [
            { id: "CG1.3.1", description: "Foster awareness of being a critical AI citizen: Enable students to gain evidence-based insights into the pervasive adoption of AI. Develop skills in critiquing AI-amplified biases and the effects of AI on social relationships, norms and structures." },
            { id: "CG1.3.2", description: "Nurture personal and social responsibilities in AI societies: Encourage students to share their views on what desirable AI societies would look like and delineate responsibilities for building inclusive, sustainable and just AI societies." },
            { id: "CG1.3.3", description: "Nurture the sense of self-actualization as an AI citizen and the lifelong learning attitude to AI: Guide students to dynamically review the impact of AI adoption across sectors and the competency sets that living in an AI society requires." },
          ],
          contextualActivities: [
            { id: "CA1.3.1", name: "Case studies on conflicts between an inclusive AI society and AI threats", description: "Organize case studies on typical conflicts between an inclusive and just AI society and the risks AI poses to human-centred values. Challenge students to defend positions on how AI can be regulated." },
            { id: "CA1.3.2", name: "Inquiry on the personal social responsibilities of being an AI society citizen", description: "Arrange for students to conduct group discussions on the rights of citizens in an AI society, and jointly outline obligations and responsibilities citizens should assume." },
            { id: "CA1.3.3", name: "Case studies on self-actualization in AI societies", description: "Organize case studies on the adoption of AI in work, life and social practices, and challenge students to review implications for their personal goals and career development." },
          ],
          learningEnvironments: [
            "Unplugged learning settings including worksheets, flipcharts, reports on jobs and career development in AI societies.",
            "Online AI systems or locally available AI tools for experiential and analytical tests.",
          ],
          indicators: [
            { id: "sc-hc-c1", description: "Build critical views on the impact of AI on human societies" },
            { id: "sc-hc-c2", description: "Solidify civic values and sense of social responsibility as a citizen in an AI society" },
            { id: "sc-hc-c3", description: "Reinforce open-minded attitude and lifelong curiosity about learning and using AI" },
          ],
        },
      ],
    },
    // ── Aspect 2: Ethics of AI ──
    {
      id: "sc-ethics", name: "Ethics of AI",
      description: "Represents the ethical value judgements, embodied reflections, and social and emotional skills students require to navigate, understand, practise and contribute to the adaptation of a growing set of principles and regulatory rules relative to the entire life cycle of AI systems.",
      order: 2, icon: "Shield", color: "text-blue-600",
      levels: [
        {
          id: "sc-eth-understand", name: "Embodied ethics", description: "Students are expected to develop a basic understanding of the ethical issues around AI, and the potential impact of AI on human rights, social justice, inclusion, equity and climate change.",
          order: 1, target: "All students",
          curricularGoals: [
            { id: "CG2.1.1", description: "Illustrate dilemmas around AI and identify the main reasons behind ethical conflicts: Guide students to surface dilemma decisions that creators need to make in the design and development of AI." },
            { id: "CG2.1.2", description: "Facilitate scenario-based understandings of ethical principles on AI and their personal implications: Offer students opportunities to discuss age-appropriate real-world cases around the six core AI ethical principles." },
            { id: "CG2.1.3", description: "Guide the embodied reflection and internalization of ethical principles on AI: Guide students to understand the implications of ethical principles for their human rights, data privacy, safety, and human agency." },
          ],
          contextualActivities: [
            { id: "CA2.1.1", name: "Case studies on scenarios containing controversies around AI", description: "Present age-appropriate real-world or simulated scenarios, guide students to surface controversies and draw infographics illustrating the core AI ethical principles." },
            { id: "CA2.1.2", name: "Individual or group reflection on the personal implications of ethical dilemmas", description: "Engage students in group discussion and opinion taking on ethical dilemmas that may arise from uses of AI in daily life and learning in local contexts." },
            { id: "CA2.1.3", name: "Searching for and validating examples of 'AI for the public good'", description: "Organize scoping of examples of AI tools that support the public good, including promoting equity and inclusion, preserving diversity, and increasing sustainability." },
          ],
          learningEnvironments: [
            "Unplugged learning settings including print stories, case studies, worksheets and posters.",
            "Locally available AI tools including mobile phone apps.",
            "Predownloaded or recorded videos related to specific dilemma scenarios.",
            "Search engines, online videos or resources related to case studies.",
          ],
          indicators: [
            { id: "sc-eth-u1", description: "Develop a basic understanding of the ethical issues around AI and their impact on human rights, social justice and inclusion" },
            { id: "sc-eth-u2", description: "Understand and internalize key ethical principles: Do no harm, Proportionality, Non-discrimination, Sustainability, Human determination, Transparency" },
            { id: "sc-eth-u3", description: "Translate ethical principles into reflective practices and uses of AI tools" },
          ],
        },
        {
          id: "sc-eth-apply", name: "Safe and responsible use", description: "Students are expected to carry out responsible AI practices in compliance with ethical principles and locally applicable regulations, be conscious of data privacy risks, and protect their own safety and that of their peers.",
          order: 2, target: "All school students",
          curricularGoals: [
            { id: "CG2.2.1", description: "Foster self-awareness and habitual compliance with ethical principles for the responsible use of AI: Support students to build and update a checkbox of ethical principles for responsible AI practices." },
            { id: "CG2.2.2", description: "Offer opportunities to reinforce self-discipline in the responsible use of AI: Provide students with age-appropriate understanding of their personal, legal and ethical responsibilities." },
            { id: "CG2.2.3", description: "Deepen practical knowledge on the safe use of AI and awareness of locally applicable regulations: Facilitate students to categorize safety risks and practise strategies for safe AI use." },
          ],
          contextualActivities: [
            { id: "CA2.2.1", name: "Designing an 'ethics kit' for the self-disciplined, responsible use of AI", description: "Design simulated scenarios containing potential ethical conflicts. Organize the drafting of an 'ethics kit' that users habitually check when using AI." },
            { id: "CA2.2.2", name: "Simulation of typical AI incidents and risk management", description: "Expose students to simulated AI incidents. Familiarize them with precautionary strategies for ensuring data is collected, used, shared, archived and deleted only with informed consent." },
            { id: "CA2.2.3", name: "Users' reviews of AI creators' policies on data privacy", description: "Encourage students to search for AI creators' policies on data privacy and check whether they comply with relevant regulations." },
            { id: "CA2.2.4", name: "Debate the ownership of AI-generated content", description: "Organize a debate to trigger reflections around the ownership of content created using AI and examine applicable regulations on copyright." },
          ],
          learningEnvironments: [
            "Unplugged learning settings including paper-based worksheets, posters and checklists.",
            "Predownloaded privacy policies and AI regulations.",
            "Locally available AI tools including smartphone apps.",
            "Online AI tools especially platforms containing recommender algorithms and content generators.",
          ],
          indicators: [
            { id: "sc-eth-a1", description: "Carry out responsible AI practices in compliance with ethical principles and locally applicable regulations" },
            { id: "sc-eth-a2", description: "Be conscious of data privacy risks and take measures to ensure informed consent for data handling" },
            { id: "sc-eth-a3", description: "Protect own safety and that of peers when using AI" },
          ],
        },
        {
          id: "sc-eth-create", name: "Ethics by design", description: "Students are expected to adopt an ethics-by-design approach to the design, assessment and use of AI tools as well as the review and adaptation of AI regulations.",
          order: 3, target: "Students with strong interest in AI innovation",
          curricularGoals: [
            { id: "CG2.3.1", description: "Build awareness and understanding on 'ethics by design': Provide conflict-based learning opportunities so students can apply an integral set of ethical principles throughout the AI life cycle." },
            { id: "CG2.3.2", description: "Develop a critical attitude to the ethics-by-design principles behind existing AI systems: Provide students with opportunities to take a holistic approach to evaluating 'ethics by design' of specific AI systems." },
            { id: "CG2.3.3", description: "Cultivating social responsibilities to uphold 'ethics by design' in regulations on AI: Guide students to evaluate how regulations align with the ethics-by-design approach." },
          ],
          contextualActivities: [
            { id: "CA2.3.1", name: "Simulating the due diligence of a 'chief ethics officer'", description: "Design project-based learning where students simulate the role of a chief ethics officer, including drafting a checklist of ethical criteria for auditing key steps of AI system design." },
            { id: "CA2.3.2", name: "Simulating the use of 'ethics label' to audit selected AI tools", description: "Organize students to undertake a mock audit of 'ethics by design' in selected AI tools using an ethics label analogous to a nutrition label for food items." },
            { id: "CA2.3.3", name: "Simulating the use of an ethics matrix to review regulations", description: "Invite students to research an ethics matrix with ethical principles as columns and relevant stakeholders as rows, and apply it to analyse selected regulations." },
          ],
          learningEnvironments: [
            "Unplugged learning settings including worksheets, flipcharts, examples of due diligence checks, ethics labels, privacy policies and regulations on AI.",
            "Locally available AI tools including smartphone apps.",
            "Online AI systems for ethical analysis.",
            "Websites sharing regulations on AI and lawsuits or court cases.",
          ],
          indicators: [
            { id: "sc-eth-c1", description: "Adopt an ethics-by-design approach throughout the AI life cycle" },
            { id: "sc-eth-c2", description: "Apply parameters to assess compliance of AI tools with ethical regulations" },
            { id: "sc-eth-c3", description: "Use an ethical matrix of multi-stakeholders to review AI regulations and inform adaptation" },
          ],
        },
      ],
    },
    // ── Aspect 3: AI techniques and applications ──
    {
      id: "sc-techniques", name: "AI techniques and applications",
      description: "Represents the intrinsically linked conceptual knowledge on AI and associated operational skills, in connection with concrete AI tools or authentic tasks. This aspect serves as the most important and transferable technical foundation.",
      order: 3, icon: "Lightbulb", color: "text-amber-600",
      levels: [
        {
          id: "sc-tech-understand", name: "AI foundations", description: "Students are expected to develop basic knowledge, understanding and skills on AI, particularly with respect to data and algorithms, and understand the importance of interdisciplinary foundational knowledge.",
          order: 1, target: "All students",
          curricularGoals: [
            { id: "CG3.1.1", description: "Exemplify the definition and scope of AI: Facilitate students to understand what AI is and is not; guide students to find and share exemplar tools under the main categories of AI technologies." },
            { id: "CG3.1.2", description: "Develop conceptual knowledge on how AI is trained based on data and algorithms: Foster example-based abstraction of how machine-learning models are trained using supervised, unsupervised and reinforcement learning." },
            { id: "CG3.1.3", description: "Foster open-minded thinking on AI and an interdisciplinary foundation: Enable students to gain knowledge on AI methods such as artificial neural networks, and the difference between strong and weak AI." },
            { id: "CG3.1.4", description: "Concretize human-centred considerations in the design and use of AI: Highlight humans' roles in key steps of the AI life cycle including researchers, data engineers, testers, regulators, and auditors." },
          ],
          contextualActivities: [
            { id: "CA3.1.1", name: "Example-based definition and scope of AI", description: "Investigate and experiment with examples of AI tools. Help students understand what AI is and is not, and the main categories of AI technologies adopted in daily life." },
            { id: "CA3.1.2", name: "Spiral learning from examples to abstract concepts", description: "Use selected examples to guide students to abstract how a machine learning model is trained, including problem definition, data collection, processing, training, evaluation, deployment and iteration." },
            { id: "CA3.1.3", name: "Case analysis of innovative AI tools and innovative uses of AI", description: "Organize students to search for innovative AI tools; guide them to identify the key techniques and main categories of AI used in these applications." },
            { id: "CA3.1.4", name: "Solidifying multidisciplinary foundation for AI with focus on mathematics", description: "Help students grasp that modern AI systems are rooted in mathematics. Nurture essential mathematical skills including algebra, probability, statistics, linear algebra and calculus for understanding ML and neural networks." },
          ],
          learningEnvironments: [
            "Unplugged learning settings including textbooks, essays, worksheets.",
            "Online or downloaded videos introducing AI innovations.",
            "Locally available AI tools including basic AI-assisted smartphone apps.",
            "Online AI tools, for example image/video creators, generative AI models.",
          ],
          indicators: [
            { id: "sc-tech-u1", description: "Develop basic knowledge on AI including data and algorithms" },
            { id: "sc-tech-u2", description: "Understand the three types of AI algorithms: supervised, unsupervised and reinforcement learning" },
            { id: "sc-tech-u3", description: "Connect conceptual knowledge on AI with activities in society and daily life" },
            { id: "sc-tech-u4", description: "Understand the interplay between AI knowledge and STEM, languages and social studies" },
          ],
        },
        {
          id: "sc-tech-apply", name: "Application skills", description: "Students are expected to construct an age-appropriate knowledge structure on data, AI algorithms and programming, and acquire transferable application skills.",
          order: 2, target: "All school students",
          curricularGoals: [
            { id: "CG3.2.1", description: "Strengthen knowledge and skills on data modelling, engineering and analysis: Provide task-based learning on datasets, applying tools or programming languages to acquire, clean and transform data." },
            { id: "CG3.2.2", description: "Acquire age-appropriate technical skills in AI programming: Scaffold understanding of AI algorithms including supervised, unsupervised and reinforcement learning, and their concrete algorithm types." },
            { id: "CG3.2.3", description: "Develop skills to leverage open-source datasets and AI tools: Facilitate acquisition of skills to critically evaluate and leverage open-source AI datasets and algorithm libraries." },
          ],
          contextualActivities: [
            { id: "CA3.2.1", name: "Data biases lab", description: "Provide sample datasets with and without outliers; guide students to conduct hands-on experimentation on how outliers and class imbalance affect model performance." },
            { id: "CA3.2.2", name: "Tailored optional modular courses on various AI algorithms", description: "Tailor open-source datasets and algorithm libraries according to student age and prior knowledge. Develop optional modular courses supporting cohort-based learning." },
            { id: "CA3.2.3", name: "AI hackathons based on variations of authentic tasks", description: "Schedule continuous learning hours for students to conduct task-based hackathons practising transferable AI programming skills." },
            { id: "CA3.2.4", name: "Debunking claims that AI will automate coding", description: "Facilitate research into the professional knowledge demanded by AI system creation, and contemplate how using AI to replace programming skills exacerbates inequality." },
          ],
          learningEnvironments: [
            "Computers with internet connection.",
            "Computer-based samples of datasets or locally accessible public datasets.",
            "Computer-based applications for AI programming or open-source AI programming libraries.",
            "Computer-based or locally accessible online AI tools.",
          ],
          indicators: [
            { id: "sc-tech-a1", description: "Construct age-appropriate knowledge structure on data, AI algorithms and programming" },
            { id: "sc-tech-a2", description: "Critically evaluate and leverage free and/or open-source AI tools, libraries and datasets" },
            { id: "sc-tech-a3", description: "Transfer AI programming skills across complex contexts" },
          ],
        },
        {
          id: "sc-tech-create", name: "Creating AI tools", description: "Students are expected to deepen and apply knowledge on data and algorithms to customize existing AI toolkits to create task-based AI tools.",
          order: 3, target: "Students with strong interest in AI innovation",
          curricularGoals: [
            { id: "CG3.3.1", description: "Challenge and enable advanced skills to develop task-based AI tools: Support mastery of analysing existing AI tools, assessing data needs, deciding on low-code vs programming approaches, and operational customization." },
            { id: "CG3.3.2", description: "Enhance creativity in applying AI knowledge to customize AI toolkits and coding: Design tasks around customizing AI tools to solve authentic tasks; support exploring creative ideas." },
            { id: "CG3.3.3", description: "Equip students with skills to test and optimize self-crafted AI tools: Support students to customize assessment methods, organize peer assessments, and build collaborative skills." },
          ],
          contextualActivities: [
            { id: "CA3.3.1", name: "Task-based enhancement of datasets and programming codes for crafting an AI tool", description: "Organize students to modify or create datasets for real-world contexts, apply AI programming skills to clean and preprocess data, and use it to customize AI models." },
            { id: "CA3.3.2", name: "AI application performance test lab", description: "Guide students to adapt open-source performance metrics (F1 score, confusion matrices, ROC curves) to test performance and technological robustness of crafted AI applications." },
            { id: "CA3.3.3", name: "Comparing customizing datasets/programming codes with low-code development platforms", description: "Organize students to study the differences between the two approaches in terms of human agency, inclusion of local data, cultural diversity, scalability and reusability." },
          ],
          learningEnvironments: [
            "Locally accessible open-source online datasets, AI tools and programming libraries.",
            "Locally accessible open-source data analytics tools.",
            "Locally accessible cloud-based computing resources or locally hosted computing resources.",
          ],
          indicators: [
            { id: "sc-tech-c1", description: "Transfer values, knowledge and skills to crafting AI tools based on existing AI models or toolkits" },
            { id: "sc-tech-c2", description: "Leverage AI-development platforms, enhance datasets, and modify programming codes including open-source options" },
            { id: "sc-tech-c3", description: "Test and optimize self-crafted AI tools using adapted assessment methods" },
          ],
        },
      ],
    },
    // ── Aspect 4: AI system design ──
    {
      id: "sc-design", name: "AI system design",
      description: "Focuses on the systemic design thinking and comprehensive engineering skills required for problem scoping, design, architecture building, training, testing and optimization of AI systems. This aspect mainly targets students who have a particular interest in deepening their knowledge and skills in this field.",
      order: 4, icon: "Target", color: "text-emerald-600",
      levels: [
        {
          id: "sc-des-understand", name: "Problem scoping", description: "Students are expected to understand the importance of 'AI problem scoping' as the starting point for AI innovation, examine whether AI should be used from legal, ethical and logical perspectives, and define problem boundaries, goals and constraints.",
          order: 1, target: "All students",
          curricularGoals: [
            { id: "CG4.1.1", description: "Scaffold critical thinking skills on when AI should not be used: Guide students to develop critical analysis skills to examine reasons why AI should or should not be used to address certain real-world challenges." },
            { id: "CG4.1.2", description: "Support the acquisition of skills in scoping a problem to be solved by an AI system: Based on simulation projects, support learning and practice of skills to identify and scope problems." },
            { id: "CG4.1.3", description: "Develop skills on assessing AI systems' need for data, algorithms and computing resources: Offer opportunities for planning skills by assessing needs and feasibility." },
          ],
          contextualActivities: [
            { id: "CA4.1.1", name: "Simulating the review of project proposals", description: "Organize students to simulate the review of a project proposal. Conduct a debate on whether AI should or should not be used, considering data availability, ethical implications and environmental impact." },
            { id: "CA4.1.2", name: "Simulating problem-scoping and justification for AI system design", description: "Facilitate students to research problems in their daily lives or communities and identify problems that could be addressed by AI; scope and define the problem with a corresponding problem statement." },
            { id: "CA4.1.3", name: "Data preprocessing lab", description: "Using a basic dataset and an existing AI model, organize experiments on training the model based on variations of the dataset, supporting students to apply various data preprocessing techniques." },
          ],
          learningEnvironments: [
            "Unplugged learning settings including worksheets, paper-based case studies, printouts of prototypes or plans.",
            "Digital devices with an internet connection.",
            "Selected online AI systems.",
          ],
          indicators: [
            { id: "sc-des-u1", description: "Examine whether AI should be used in certain situations from legal, ethical and logical perspectives" },
            { id: "sc-des-u2", description: "Define the boundaries, goals and constraints of a problem before attempting to train an AI model" },
            { id: "sc-des-u3", description: "Acquire knowledge and project-planning skills to conceptualize and construct an AI system" },
          ],
        },
        {
          id: "sc-des-apply", name: "Architecture design", description: "Students are expected to cultivate basic methodological knowledge and technical skills to configure a scalable, maintainable and reusable architecture for an AI system covering layers of data, algorithms, models and application interfaces.",
          order: 2, target: "Students with interest in AI",
          curricularGoals: [
            { id: "CG4.2.1", description: "Scaffold the acquisition of methodological knowledge and technical skills on AI architecture: Facilitate students to evaluate AI architectures and configure a prototype encompassing anti-bias data, energy-efficient models, and human-centred design." },
            { id: "CG4.2.2", description: "Support advanced technical skills and project management competencies for AI system building: Offer project-based learning to acquire interdisciplinary technical skills for building prototype AI systems." },
          ],
          contextualActivities: [
            { id: "CA4.2.1", name: "Simulating the evaluation of frameworks and components for AI architectural configuration", description: "Facilitate students to evaluate AI frameworks (TensorFlow, PyTorch, Scikit-learn), simulate evaluation and selection of architecture components, and communicate configuration through diagrams or pseudocode." },
            { id: "CA4.2.2", name: "Simulating the leveraging of resources to build an AI system", description: "Facilitate building simulated AI systems using local or cloud computing platforms, conducting trade-offs between cost, computing capability, robustness and environmental impact." },
          ],
          learningEnvironments: [
            "Videos and metrics for ethical and technical evaluations of AI models.",
            "Computer-based or locally accessible online examples of AI systems.",
            "Computer-based datasets or locally accessible public datasets.",
            "Computer-based AI programming applications or open-source libraries.",
            "Locally hosted or open-source cloud computing resources.",
          ],
          indicators: [
            { id: "sc-des-a1", description: "Configure a scalable, maintainable and reusable architecture for an AI system" },
            { id: "sc-des-a2", description: "Apply deepened human-centred values and ethical principles in configuration, construction and optimization" },
            { id: "sc-des-a3", description: "Leverage datasets, programming tools and computational resources to construct a prototype AI system" },
          ],
        },
        {
          id: "sc-des-create", name: "Iteration and feedback", description: "Students are expected to enhance and apply their interdisciplinary knowledge to evaluate AI models' humanistic appropriateness and methodological robustness, and acquire technical skills to improve datasets, reconfigure algorithms and enhance architectures based on tests and feedback.",
          order: 3, target: "Students with strong interest in AI innovation",
          curricularGoals: [
            { id: "CG4.3.1", description: "Develop the skills to critique AI systems: Provide project-based learning for testing technological robustness and critiquing ethical appropriateness of AI systems through auditing, measuring performance, and studying user feedback." },
            { id: "CG4.3.2", description: "Support building technical skills and social responsibilities in optimizing, reconfiguring or shutting down an AI system: Offer simulations for decisions on iteration based on testing and feedback—optimization, reconfiguration, or shutdown." },
            { id: "CG4.3.3", description: "Foster students' self-identities as co-creators in the AI era: Guide students to nurture responsibilities of being a co-creator and develop their sense of belonging to the larger AI community." },
          ],
          contextualActivities: [
            { id: "CA4.3.1", name: "Simulating the performance-test of an AI system", description: "Organize students to use adapted metrics to evaluate whether an AI model enhances or weakens human capacities, measure performance using metrics (F1, confusion matrices, ROC curves), and synthesize results visually." },
            { id: "CA4.3.2", name: "Simulating AI engineers' corporate decision-making on AI model iteration", description: "Organize students to play AI engineers' roles, integrating feedback to make decisions on optimization, reconfiguration, or shutdown of AI models." },
            { id: "CA4.3.3", name: "Engagement with communities of AI creators", description: "Facilitate interested students to join local or online communities of AI co-creators. Encourage participation in discussions and collaborative development of AI tools." },
          ],
          learningEnvironments: [
            "Locally accessible free and/or open-source AI tools including data analytics tools and programming libraries.",
            "Locally hosted or locally accessible cloud computing resources.",
            "Downloaded and adapted instruments for ethical auditing and performance testing.",
            "Access to applicable regulations on AI or governance frameworks.",
            "Locally accessible online collaborative platforms (e.g. GitHub, arXiv or forum groups).",
          ],
          indicators: [
            { id: "sc-des-c1", description: "Evaluate AI models for humanistic appropriateness and methodological robustness" },
            { id: "sc-des-c2", description: "Improve quality of datasets, reconfigure algorithms and enhance architectures based on tests and feedback" },
            { id: "sc-des-c3", description: "Apply human-centred mindset and ethical principles in simulating decision-making on AI system iteration or shutdown" },
          ],
        },
      ],
    },
  ],
  metadata: {
    publicationYear: 2024,
    isbn: "978-92-3-100709-5",
    doi: "https://doi.org/10.54675/JKJB9835",
    license: "CC-BY-SA 3.0 IGO",
    totalAspects: 4,
    totalLevels: 3,
    totalCompetencyBlocks: 12,
    totalCurricularGoals: 36,
    totalContextualActivities: 39,
    totalLearningEnvironments: 44,
    ethicalPrinciplesReferenced: [
      "Do no harm", "Proportionality", "Non-discrimination", "Sustainability",
      "Human determination in human-AI collaboration", "Transparency and explainability",
      "Safe and responsible use", "Ethics by design",
    ],
    crossCuttingThemes: [
      "Inclusion and accessibility for students with disabilities",
      "Linguistic and cultural diversity", "Environmental sustainability",
      "Open-source AI tools and datasets", "Human agency and accountability",
      "Data privacy and protection", "Brand-agnostic and product-agnostic competencies",
      "Lifelong learning foundation", "Interdisciplinary knowledge integration",
    ],
    implementationStrategies: [
      "Aligning AI competencies as the foundation for national AI strategies",
      "Building interdisciplinary core and cluster AI curricula",
      "Framing future-proofing and locally feasible AI domains as curriculum carriers",
      "Tailoring age-appropriate spiral curricular sequences",
      "Building enabling learning environments for AI curricula",
      "Promoting the professionalization of AI teachers",
      "Guiding cohort-based design and organization of pedagogical activities",
      "Constructing competency-based assessments on progression of key AI aspects",
    ],
    assessmentDomains: [
      { aspect: "Human-centred mindset", types: ["Conflict-based opinion taking", "Conflict-based critical evaluation", "Conflict-based social interactions"] },
      { aspect: "Ethics of AI", types: ["Scenario-based ethical value orientation", "Scenario-based ethical behaviours", "Scenario-based rule-making"] },
      { aspect: "AI techniques and applications", types: ["Problem-based AI knowledge", "Tool-based conceptual insights", "Task-based tool crafting"] },
      { aspect: "AI system design", types: ["Project-based design thinking", "Project-based system configuration", "Project-based iteration"] },
    ],
  },
  useCases: ["Developing student AI literacy programmes", "Embedding AI ethics into curricula", "Creating project-based AI learning", "Assessing student AI competencies", "Building interdisciplinary AI curricula", "Training AI teachers"],
  crossReferences: ["teacher-competency", "guidance-policy"],
  assessmentQuestions: [
    { id: "sc-q1", dimension: "Human-centred mindset", question: "How well do you understand human agency in the context of AI?", options: [
      { value: "sc-a1", label: "I'm starting to learn that AI is human-led", level: "acquire" },
      { value: "sc-a2", label: "I understand human accountability when using AI", level: "deepen" },
      { value: "sc-a3", label: "I advocate for responsible AI citizenship", level: "create" },
    ]},
    { id: "sc-q2", dimension: "Ethics of AI", question: "How do you approach ethical questions when using AI?", options: [
      { value: "sc-b1", label: "I'm aware AI can raise ethical issues", level: "acquire" },
      { value: "sc-b2", label: "I practise safe and responsible AI use", level: "deepen" },
      { value: "sc-b3", label: "I apply ethics-by-design principles", level: "create" },
    ]},
    { id: "sc-q3", dimension: "AI techniques and applications", question: "How well do you understand how AI systems work technically?", options: [
      { value: "sc-c1", label: "I have basic awareness of AI and algorithms", level: "acquire" },
      { value: "sc-c2", label: "I can apply AI programming skills and use open-source tools", level: "deepen" },
      { value: "sc-c3", label: "I can create task-based AI tools from existing toolkits", level: "create" },
    ]},
    { id: "sc-q4", dimension: "AI system design", question: "Can you scope, design and iterate on an AI system?", options: [
      { value: "sc-d1", label: "I understand what problem scoping means for AI", level: "acquire" },
      { value: "sc-d2", label: "I can configure a prototype AI architecture", level: "deepen" },
      { value: "sc-d3", label: "I can iterate on AI systems based on testing and feedback", level: "create" },
    ]},
  ],
  assessmentTitle: "Student AI Competency Self-Assessment",
  assessmentDescription: "Discover your AI competency level across 4 core aspects",
  showInQuiz: true,
  showInDashboard: true,
  showInLanding: true,
  isBackgroundFramework: false,
  compatibility: [],
  sourceFidelity: "official",
  estimatedAssessmentMinutes: 24,
  region: "international",
};

// ────────────────────────────────────────────────
// 4. QS AI Capability Framework
// ────────────────────────────────────────────────
const QS_LEVELS: Omit<import("./framework-types").Level, "indicators">[] = [
  { id: "qs-basic", name: "Basic", description: "Minimal or no systematic AI capability. Activities are ad hoc, reactive, and driven by individuals rather than institutional strategy.", order: 1, target: "Institutions beginning their AI journey" },
  { id: "qs-developing", name: "Developing", description: "Initial structures, policies, and dedicated resources established. Pilot projects and coordinated initiatives underway with cross-functional collaboration emerging.", order: 2, target: "Institutions with active AI strategies building capability through structured programmes" },
  { id: "qs-advanced", name: "Advanced", description: "AI capability fully embedded, continuously optimized, and strategically aligned across the institution. Practices are proactive, data-driven, and regularly evaluated for impact.", order: 3, target: "Institutions aiming for sector leadership with mature, integrated AI capabilities" },
];

const aiCapability: Framework = {
  id: "ai-capability",
  name: "QS AI Capability Framework",
  shortName: "QS AI Capability",
  description: "An open source framework helping universities evaluate and enhance their AI capabilities across 4 pillars, 14 indicators, and 33 sub-indicators.",
  type: "capability",
  scope: "institutional",
  source: "QS",
  region: "international",
  path: "/frameworks/ai-capability",
  icon: "Building2",
  color: "text-cyan-600",
  badgeLabel: "QS Framework",
  targetAudience: ["leader", "admin"],
  overview: `The QS AI Capability Framework provides a structured approach for universities to assess their current AI integration, identify areas for improvement, and make informed strategic decisions. Built upon four key pillars — Governance & Human Commitment, Outreach & Operational Efficiency, Teaching Learning & Assessment, and Research & Scholarship — it segments into 14 indicators and encompasses 33 sub-indicators. The framework was developed in collaboration with AI experts from institutions including Arizona State University, Imperial College London, University of Cambridge, Monash University, and Wharton School, alongside industry partners AWS and Microsoft.`,
  keyPrinciples: [
    { id: "qs-p1", name: "Relevance and rigour", description: "Developed in collaboration with AI experts, industry and academic stakeholders to ensure relevance, rigor, and adaptability." },
    { id: "qs-p2", name: "Multi-stakeholder assessment", description: "Complemented by faculty and student surveys to capture diverse stakeholder insights." },
    { id: "qs-p3", name: "Evidence-based evaluation", description: "Data collection completed by the institution and supported by evidence, with points allocated at each sub-indicator level." },
    { id: "qs-p4", name: "Formative and developmental", description: "Designed to help institutions understand where they are on their AI journey, benchmark against peers, and articulate progress." },
    { id: "qs-p5", name: "Open source taxonomy", description: "Published under Creative Commons CC BY-SA 4.0 license, enabling institutions to use it freely for self-assessment." },
  ],
  keyDimensions: [
    // ── Pillar 1: Governance & Human Commitment ──
    {
      id: "qs-regulatory-ethics", name: "Regulatory & Ethical Standards",
      description: "Pillar 1: Governance — Monitoring adherence to AI regulations, establishing ethical guidelines, and assessing sustainability impact.",
      order: 1, icon: "Scale", color: "text-blue-600",
      levels: QS_LEVELS.map((l, i) => ({
        ...l,
        id: `qs-re-${l.id}`,
        indicators: i === 0 ? [
          { id: "sub-1-1-1-b", description: "No systematic monitoring of AI regulations. Awareness of frameworks like the EU AI Act or national AI strategies is limited to a few individuals. The institution is reactive, with policy conversations occurring only after incidents." },
          { id: "sub-1-1-2-b", description: "Ethical guidelines, if any, are informal statements or borrowed from peer institutions without adaptation. No structured approach to ethical review of AI deployments." },
          { id: "sub-1-1-3-b", description: "No sustainability assessment of AI deployments. Energy consumption of AI tools goes untracked." },
        ] : i === 1 ? [
          { id: "sub-1-1-1-d", description: "The institution actively monitors relevant AI regulations with designated responsibility for regulatory tracking. A schedule for policy review is in place, and the institution participates in sector bodies or consortia for benchmarking." },
          { id: "sub-1-1-2-d", description: "Published ethical guidelines address transparency, fairness, and accountability, though application may be inconsistent across departments. Guidelines are reviewed and updated periodically." },
          { id: "sub-1-1-3-d", description: "Initial sustainability impact assessments have been conducted for major AI procurements. Energy and carbon considerations are part of vendor evaluation criteria." },
        ] : [
          { id: "sub-1-1-1-a", description: "Regulatory monitoring is continuous, with a compliance dashboard tracking adherence to multiple jurisdictional requirements. Policy updates are triggered proactively by emerging regulation. The institution contributes to external regulatory and standards development." },
          { id: "sub-1-1-2-a", description: "Ethical guidelines are operationalized through mandatory review processes for all new AI deployments, with embedded checkpoints rather than just stated principles. Multi-stakeholder consultation informs regular updates." },
          { id: "sub-1-1-3-a", description: "Sustainability impact is quantified: the institution tracks compute costs, energy usage, and carbon footprint of AI systems and publishes this data. Sustainability criteria are embedded in all AI procurement and deployment decisions." },
        ],
      })),
    },
    {
      id: "qs-governance-risk", name: "Governance & Risk Management",
      description: "Pillar 1: Governance — Formal governance structures for AI decision-making, structured risk assessment processes, and ethical procurement practices.",
      order: 2, icon: "ShieldCheck", color: "text-blue-600",
      levels: QS_LEVELS.map((l, i) => ({
        ...l,
        id: `qs-gr-${l.id}`,
        indicators: i === 0 ? [
          { id: "sub-1-2-1-b", description: "No formal AI governance body exists. Decisions about AI adoption are made ad hoc by individual departments or IT. No centralized oversight of institutional AI use." },
          { id: "sub-1-2-2-b", description: "No structured risk assessment process for AI systems. AI tools are adopted without centralized review. Faculty and staff individually adopt tools without institutional vetting." },
          { id: "sub-1-2-3-b", description: "Procurement of AI tools happens without centralized review or ethical criteria. No AI-specific requirements in vendor contracts." },
        ] : i === 1 ? [
          { id: "sub-1-2-1-d", description: "A cross-functional AI steering committee or governance board is established with representation from IT, academic affairs, legal, data privacy, and student affairs." },
          { id: "sub-1-2-2-d", description: "Formal risk assessment is conducted for AI pilots before scaling. An AI tool registry catalogues which tools are in use across campus. Initial risk-management processes and risk profiles are evolving." },
          { id: "sub-1-2-3-d", description: "Procurement guidelines require vendor contracts to include data privacy, security, and ethical AI clauses. Evaluation criteria include responsible AI practices." },
        ] : [
          { id: "sub-1-2-1-a", description: "AI governance is fully integrated into existing institutional decision-making bodies rather than operating as a parallel structure. Every new digital project considers AI implications as standard procedure." },
          { id: "sub-1-2-2-a", description: "Automated risk-management tools monitor AI systems continuously for performance, bias drift, and compliance. Risk assessment is proactive and scenario-based, modeling potential impacts before deployment." },
          { id: "sub-1-2-3-a", description: "Ethical procurement includes third-party auditing of vendor AI practices. Comprehensive due diligence covers model training data provenance, bias testing, and ongoing monitoring commitments." },
        ],
      })),
    },
    {
      id: "qs-conduct-privacy", name: "Code of Conduct & Privacy",
      description: "Pillar 1: Governance — AI-specific codes of conduct governing permissible use, and data protection practices addressing AI-specific privacy risks.",
      order: 3, icon: "Lock", color: "text-blue-600",
      levels: QS_LEVELS.map((l, i) => ({
        ...l,
        id: `qs-cp-${l.id}`,
        indicators: i === 0 ? [
          { id: "sub-1-3-1-b", description: "No AI-specific code of conduct. Reliance on general academic integrity and IT acceptable-use policies that do not address AI-specific scenarios." },
          { id: "sub-1-3-2-b", description: "Data protection meets baseline statutory requirements (e.g. FERPA, GDPR) but does not address AI-specific risks such as model training on institutional data, student profiling, or LLM data leakage." },
        ] : i === 1 ? [
          { id: "sub-1-3-1-d", description: "An AI-specific code of conduct has been published covering permissible uses for staff, faculty, and students. Faculty and students receive training on its requirements." },
          { id: "sub-1-3-2-d", description: "Data protection impact assessments are conducted for AI deployments handling personal data. Clear guidelines govern when AI can and cannot be used for decisions affecting students." },
        ] : [
          { id: "sub-1-3-1-a", description: "The code of conduct is a living document, regularly updated through multi-stakeholder consultation including student input. Violation reporting mechanisms are well-established and accessible. Regular compliance audits verify adherence." },
          { id: "sub-1-3-2-a", description: "Privacy-by-design principles govern all AI system architecture. AI data flows are mapped end-to-end with automated compliance monitoring. The institution operates an AI transparency register disclosing where AI is used in student-facing processes." },
        ],
      })),
    },
    {
      id: "qs-leadership-capability", name: "Leadership & Capability",
      description: "Pillar 1: Governance — Senior leadership responsibility for AI strategy, and structured professional development to build institutional AI capability.",
      order: 4, icon: "Crown", color: "text-blue-600",
      levels: QS_LEVELS.map((l, i) => ({
        ...l,
        id: `qs-lc-${l.id}`,
        indicators: i === 0 ? [
          { id: "sub-1-4-1-b", description: "No senior leader has explicit responsibility for AI strategy. AI is not mentioned in the institutional strategic plan. Individual faculty champions drive experimentation without institutional support." },
          { id: "sub-1-4-2-b", description: "Staff AI training is absent or limited to optional webinars with low attendance. No structured programme for building AI capability." },
        ] : i === 1 ? [
          { id: "sub-1-4-1-d", description: "AI appears in the institutional strategic plan with allocated budget. A senior leader has explicit AI portfolio responsibility. An AI community of practice connects practitioners across departments." },
          { id: "sub-1-4-2-d", description: "Structured professional development programmes cover both tool proficiency and ethical use. Training is available to all staff, though participation and scope may be limited." },
        ] : [
          { id: "sub-1-4-1-a", description: "AI capability is a core institutional priority with board-level visibility. Leadership succession planning includes AI competency requirements. The institution is recognized externally for AI leadership and contributes to sector capability-building." },
          { id: "sub-1-4-2-a", description: "AI literacy is expected of all staff and embedded in promotion and performance criteria. Professional development is continuous, personalized, and role-specific. The institution invests in advanced capability-building beyond basic literacy." },
        ],
      })),
    },
    // ── Pillar 2: Outreach & Operational Efficiency ──
    {
      id: "qs-ai-recruitment", name: "AI Enhanced Recruitment",
      description: "Pillar 2: Outreach — AI integration into student recruitment, admissions processing, and enrollment management.",
      order: 5, icon: "UserPlus", color: "text-purple-600",
      levels: QS_LEVELS.map((l, i) => ({
        ...l,
        id: `qs-ar-${l.id}`,
        indicators: i === 0 ? [
          { id: "sub-2-1-1-b", description: "Application processing is largely manual with minimal automation. Admissions decisions rely on manual review with historical benchmarks. No predictive analytics for enrollment yield." },
          { id: "sub-2-1-2-b", description: "Traditional CRM manages prospect communications through manual segmentation and batch email campaigns. Communications to prospective students are generic and undifferentiated." },
        ] : i === 1 ? [
          { id: "sub-2-1-1-d", description: "AI-powered CRM capabilities score applicant engagement and predict enrollment probability. Application triage uses AI to flag incomplete or high-priority applications. Prescriptive analytics recommend targeted actions." },
          { id: "sub-2-1-2-d", description: "Personalized communication workflows are triggered by prospect behavior such as web visits, event attendance, and inquiry patterns. AI-enhanced lead scoring prioritizes outreach efforts." },
        ] : [
          { id: "sub-2-1-1-a", description: "A fully integrated AI-driven enrollment ecosystem operates with real-time scenario modeling. AI continuously refines predictions from each admissions cycle. Bias-mitigation techniques are built into predictive models to ensure equity." },
          { id: "sub-2-1-2-a", description: "Multi-channel AI orchestration across email, text, chatbot, and social media delivers personalized engagement at scale. Conversion optimization uses predictive modeling for financial aid packaging and outreach timing." },
        ],
      })),
    },
    {
      id: "qs-student-support", name: "Personalised Student Support",
      description: "Pillar 2: Outreach — AI-powered career guidance, real-time student support, and proactive student success interventions.",
      order: 6, icon: "HeartHandshake", color: "text-purple-600",
      levels: QS_LEVELS.map((l, i) => ({
        ...l,
        id: `qs-ss-${l.id}`,
        indicators: i === 0 ? [
          { id: "sub-2-2-1-b", description: "Career services operate through traditional in-person counseling and manual job boards. No AI-powered tools for skills-to-career matching or labor market analytics." },
          { id: "sub-2-2-2-b", description: "All student inquiries handled by staff during business hours. Student support is reactive, with intervention only after failure or withdrawal. No AI chatbots or early-warning systems." },
        ] : i === 1 ? [
          { id: "sub-2-2-1-d", description: "Career services offer AI-assisted resume review and basic skills-to-career matching. Initial integration of labor market data informs career pathway recommendations." },
          { id: "sub-2-2-2-d", description: "AI chatbots handle routine inquiries across admissions, financial aid, and registration. Early-warning systems process engagement data to generate risk scores. AI-powered nudge campaigns deliver personalized reminders." },
        ] : [
          { id: "sub-2-2-1-a", description: "Career coaching uses labor-market analytics and salary data to provide personalized pathway recommendations. AI matches students with alumni mentors, internships, and jobs based on comprehensive profile analysis." },
          { id: "sub-2-2-2-a", description: "AI functions as a comprehensive student success ecosystem integrating academic, financial, career, and wellbeing support. Hybrid models combine AI-generated risk signals with human advisor judgment to orchestrate personalized support pathways." },
        ],
      })),
    },
    {
      id: "qs-faculty-efficiency", name: "Faculty & Administrative Efficiency",
      description: "Pillar 2: Outreach — AI deployment for administrative process automation, operational optimization, and institutional efficiency.",
      order: 7, icon: "Settings", color: "text-purple-600",
      levels: QS_LEVELS.map((l, i) => ({
        ...l,
        id: `qs-fe-${l.id}`,
        indicators: i === 0 ? [
          { id: "sub-2-3-1-b", description: "Manual, spreadsheet-driven administrative processes dominate. Individual staff may use AI for personal productivity without institutional direction. No coordinated AI automation strategy." },
        ] : i === 1 ? [
          { id: "sub-2-3-1-d", description: "AI is deployed in specific high-impact operational areas: automated document processing, AI-assisted scheduling, chatbot-based help desks. Clear ROI measurement for AI pilots is established." },
        ] : [
          { id: "sub-2-3-1-a", description: "AI is embedded across core institutional operations: enrollment processing, financial management, HR, facilities, and compliance. Real-time operational dashboards provide AI-generated insights. Intelligent process automation handles multi-step workflows." },
        ],
      })),
    },
    {
      id: "qs-external-engagement", name: "External Engagement & Partnership",
      description: "Pillar 2: Outreach — Community-facing AI initiatives, external collaboration on AI projects, and strategic industry partnerships.",
      order: 8, icon: "Handshake", color: "text-purple-600",
      levels: QS_LEVELS.map((l, i) => ({
        ...l,
        id: `qs-ee-${l.id}`,
        indicators: i === 0 ? [
          { id: "sub-2-4-1-b", description: "Community learning initiatives do not incorporate AI. The institution neither offers AI-related continuing education nor participates in AI-focused consortia." },
          { id: "sub-2-4-2-b", description: "External collaboration on AI is absent or incidental. No formal partnerships with other institutions or organizations focused on AI." },
          { id: "sub-2-4-3-b", description: "No industry partnerships focused on AI projects. Industry engagement does not include AI-related collaboration." },
        ] : i === 1 ? [
          { id: "sub-2-4-1-d", description: "Community-facing AI literacy programmes are piloted. The institution participates in AI-focused consortia or networks. Some sharing of the institution's AI journey at conferences or through publications." },
          { id: "sub-2-4-2-d", description: "Initial industry partnerships involve AI-related student projects or internship placements. The institution contributes to sector knowledge through case studies or practice reports." },
          { id: "sub-2-4-3-d", description: "Active industry partnerships support applied AI research or student project placements. Initial co-development of AI solutions with industry partners." },
        ] : [
          { id: "sub-2-4-1-a", description: "The institution operates as a regional AI capability hub, offering AI training to community partners, SMEs, and public-sector organizations. Alumni and employer networks are leveraged for AI workforce alignment." },
          { id: "sub-2-4-2-a", description: "Cross-institutional AI collaborations produce shared infrastructure, datasets, or tools. The institution shapes external AI standards and policy through active contributions to national and international bodies." },
          { id: "sub-2-4-3-a", description: "Deep industry partnerships co-develop AI solutions through applied research projects, shared labs, or joint IP arrangements. The institution drives ethical standards and innovation through strategic alliances." },
        ],
      })),
    },
    // ── Pillar 3: Teaching, Learning & Assessment ──
    {
      id: "qs-course-curriculum", name: "Course Design & Curriculum",
      description: "Pillar 3: Teaching — Institutional approach to AI in teaching, curriculum integration of AI literacy, and development of AI-specific courses.",
      order: 9, icon: "BookOpen", color: "text-emerald-600",
      levels: QS_LEVELS.map((l, i) => ({
        ...l,
        id: `qs-cc-${l.id}`,
        indicators: i === 0 ? [
          { id: "sub-3-1-1-b", description: "No agreed institutional approach to AI in teaching. Individual faculty decide independently whether and how to permit or use AI." },
          { id: "sub-3-1-2-b", description: "Curriculum does not address AI literacy. AI topics exist only within computer science or a single specialized department." },
          { id: "sub-3-1-3-b", description: "No formal AI literacy provision. Students encounter AI only through individual course choices or self-directed exploration." },
          { id: "sub-3-1-4-b", description: "No AI-specific courses exist outside computer science, if at all. No micro-credentials or short courses on AI topics." },
        ] : i === 1 ? [
          { id: "sub-3-1-1-d", description: "An institutional framework guides AI use in teaching, with course-level flexibility within agreed principles. Faculty development programmes support curriculum redesign for AI integration." },
          { id: "sub-3-1-2-d", description: "AI literacy is being integrated into general education requirements or specific programmes. Dedicated AI courses have expanded beyond computer science into business, health sciences, and other disciplines." },
          { id: "sub-3-1-3-d", description: "AI literacy modules or workshops are available, covering responsible use, critical evaluation, and practical application. Some programmes include AI literacy as a learning outcome." },
          { id: "sub-3-1-4-d", description: "AI-specific courses are offered in multiple faculties. Micro-credentials or short courses address AI skills for professionals. Course offerings respond to student and employer demand." },
        ] : [
          { id: "sub-3-1-1-a", description: "A comprehensive, regularly updated institutional strategy governs AI in teaching, informed by evidence of impact. Curriculum development is supported by labor-market analytics and employer advisory boards." },
          { id: "sub-3-1-2-a", description: "AI literacy is treated as core academic citizenship, embedded across all disciplines. The institution offers a coherent AI curriculum spanning foundation through advanced levels, including interdisciplinary applications." },
          { id: "sub-3-1-3-a", description: "AI literacy is a graduation requirement or core competency across all programmes. Students develop skills in critical AI evaluation, responsible use, and AI-augmented problem-solving." },
          { id: "sub-3-1-4-a", description: "A comprehensive portfolio of AI courses spans undergraduate, postgraduate, and professional education. Interdisciplinary AI programmes combine technical and domain expertise. The institution maps all AI offerings to workforce competency frameworks." },
        ],
      })),
    },
    {
      id: "qs-learning-support", name: "Personalised Learning & Support",
      description: "Pillar 3: Teaching — AI-powered adaptive learning, predictive student retention, and personalized academic support.",
      order: 10, icon: "Sparkles", color: "text-emerald-600",
      levels: QS_LEVELS.map((l, i) => ({
        ...l,
        id: `qs-ls-${l.id}`,
        indicators: i === 0 ? [
          { id: "sub-3-2-1-b", description: "Standard LMS delivers uniform content to all students. No AI-powered adaptive learning platforms. Learning analytics are limited to basic completion and grade tracking." },
          { id: "sub-3-2-2-b", description: "Student retention efforts rely on aggregate statistics rather than individual prediction. Academic support is generic and available only through scheduled office hours or tutoring centres." },
          { id: "sub-3-2-3-b", description: "Academic support is delivered through traditional in-person tutoring and generic study resources. No AI-powered support tools available to students." },
        ] : i === 1 ? [
          { id: "sub-3-2-1-d", description: "Adaptive learning platforms are piloted in high-enrollment courses, adjusting content difficulty and pacing based on student performance. Learning analytics dashboards track engagement and flag at-risk students." },
          { id: "sub-3-2-2-d", description: "Retention models integrate multiple data streams (grades, attendance, financial status, LMS engagement) to generate individual risk scores. AI-powered tutoring systems provide supplementary support in specific subjects." },
          { id: "sub-3-2-3-d", description: "AI tutoring systems provide supplementary support in specific subjects with basic pedagogical scaffolding. Students have access to AI-powered study tools and writing support." },
        ] : [
          { id: "sub-3-2-1-a", description: "AI-driven personalized learning pathways operate across programmes, with real-time adaptive content adjusting difficulty, pacing, sequencing, and modality. The institution measures and continuously improves personalization efficacy." },
          { id: "sub-3-2-2-a", description: "Integrated analytics across the entire student journey enable predictive pathway optimization. Academic support is proactive and coordinated: AI triggers interventions before students recognize they need help." },
          { id: "sub-3-2-3-a", description: "AI tutoring provides 24/7 personalized support with sophisticated pedagogical scaffolding. Support is integrated across the student journey, combining AI-generated recommendations with human expertise." },
        ],
      })),
    },
    {
      id: "qs-assessment-feedback", name: "Assessment, Grading & Feedback",
      description: "Pillar 3: Teaching — AI tools for real-time feedback, adaptive assessment, and validation of AI-driven assessment for bias.",
      order: 11, icon: "ClipboardCheck", color: "text-emerald-600",
      levels: QS_LEVELS.map((l, i) => ({
        ...l,
        id: `qs-af-${l.id}`,
        indicators: i === 0 ? [
          { id: "sub-3-3-1-b", description: "Feedback is manual and often delayed. No AI tools support the feedback process. Assessment policies either ban AI use entirely or ignore it." },
          { id: "sub-3-3-2-b", description: "Traditional fixed-format assessments are used universally. No adaptive assessment tools. Assessment methods have not changed in response to AI availability." },
          { id: "sub-3-3-3-b", description: "No consideration of algorithmic bias in any grading support tools. AI detection tools (e.g. Turnitin) may be adopted reactively without bias analysis." },
        ] : i === 1 ? [
          { id: "sub-3-3-1-d", description: "Automated formative feedback is deployed in specific courses, providing students with rapid, actionable guidance. Faculty are trained to redesign assessments for the AI era with process documentation and reflection components." },
          { id: "sub-3-3-2-d", description: "A structured framework governs AI in assessment, with levels ranging from 'no AI permitted' through 'AI as study tool' to 'AI output evaluated.' Faculty redesign assessments incorporating process documentation and oral defence elements." },
          { id: "sub-3-3-3-d", description: "Initial bias audits have been conducted on AI-assisted grading or assessment tools. Clear policies distinguish between AI for grading (requiring human review) and AI for feedback generation (permitted with oversight)." },
        ] : [
          { id: "sub-3-3-1-a", description: "AI provides personalized, immediate feedback with diagnostic specificity across all major courses. Multi-modal assessment combines AI-generated evaluation with oral examination, portfolio assessment, and peer review." },
          { id: "sub-3-3-2-a", description: "Adaptive assessment engines adjust question difficulty in real time based on student responses. Assessment practices are fundamentally redesigned around human-AI interaction, critical thinking, and authentic demonstration of learning." },
          { id: "sub-3-3-3-a", description: "Systematic bias validation processes audit all AI assessment tools for fairness across demographic groups. Continuous assessment of AI impact on learning outcomes informs iterative design. The institution contributes to sector-wide assessment innovation." },
        ],
      })),
    },
    // ── Pillar 4: Research & Scholarship ──
    {
      id: "qs-ai-research-practice", name: "AI in Research Practice",
      description: "Pillar 4: Research — AI tools integrated into research workflows, and institutional support for AI-enhanced research methodology.",
      order: 12, icon: "Microscope", color: "text-amber-600",
      levels: QS_LEVELS.map((l, i) => ({
        ...l,
        id: `qs-rp-${l.id}`,
        indicators: i === 0 ? [
          { id: "sub-4-1-1-b", description: "Individual researchers use freely available AI tools ad hoc. No institutional licenses, guidelines, or disclosure requirements for AI in research." },
          { id: "sub-4-1-2-b", description: "AI use in research is untracked and ungoverned. No institutional guidance on AI attribution in publications or grant applications. No IRB updates for AI-specific considerations." },
        ] : i === 1 ? [
          { id: "sub-4-1-1-d", description: "Institutional licenses for AI research tools are available. Shared AI computing resources (GPU clusters, cloud credits) are accessible. AI tools are integrated into specific research workflows." },
          { id: "sub-4-1-2-d", description: "Guidelines establish responsible AI use in research, including disclosure requirements. IRB and ethics review processes address AI-specific considerations. AI-enhanced methodology is taught in some graduate programmes." },
        ] : [
          { id: "sub-4-1-1-a", description: "A comprehensive AI research ecosystem provides institution-controlled tools across all research phases. Custom model fine-tuning on domain-specific datasets is supported by institutional infrastructure. Strategic partnerships provide cutting-edge AI infrastructure." },
          { id: "sub-4-1-2-a", description: "Transparent documentation of AI use throughout the research lifecycle is standard practice. AI-enhanced methodology is taught in all graduate programmes. Internal AI research labs or centres of excellence are established." },
        ],
      })),
    },
    {
      id: "qs-ai-scholarship", name: "Scholarship of AI in Practice",
      description: "Pillar 4: Research — Scholarly inquiry into the institution's own AI practices, generating evidence to inform strategy and contribute to sector knowledge.",
      order: 13, icon: "BookMarked", color: "text-amber-600",
      levels: QS_LEVELS.map((l, i) => ({
        ...l,
        id: `qs-sp-${l.id}`,
        indicators: i === 0 ? [
          { id: "sub-4-2-1-b", description: "No formal scholarly inquiry into the institution's own AI practices. AI adoption generates no systematic evidence base." },
        ] : i === 1 ? [
          { id: "sub-4-2-1-d", description: "Faculty and staff conduct and publish research on the institution's AI implementations. Internal evaluation reports inform decision-making. The institution contributes case studies to sector knowledge." },
        ] : [
          { id: "sub-4-2-1-a", description: "Systematic scholarship of practice is embedded in AI strategy: every major AI deployment includes a built-in evaluation research component. Longitudinal studies track impact. Findings directly feed back into institutional strategy." },
        ],
      })),
    },
    {
      id: "qs-ai-research", name: "AI Research",
      description: "Pillar 4: Research — Academic research in the field of AI, and research examining the impact of AI within specific disciplinary fields.",
      order: 14, icon: "FlaskConical", color: "text-amber-600",
      levels: QS_LEVELS.map((l, i) => ({
        ...l,
        id: `qs-air-${l.id}`,
        indicators: i === 0 ? [
          { id: "sub-4-3-1-b", description: "AI research, if any, is confined to computer science or a single specialized unit. No AI research strategy exists." },
          { id: "sub-4-3-2-b", description: "The institution does not examine AI's disciplinary impact. No research investigates how AI is changing specific fields of study or professional practice." },
        ] : i === 1 ? [
          { id: "sub-4-3-1-d", description: "Active AI research programmes exist within multiple departments with growing interdisciplinary collaboration. The institution has attracted AI research funding and publishes in AI-related venues." },
          { id: "sub-4-3-2-d", description: "Research examines AI's impact within specific disciplines such as healthcare AI, legal AI, and educational AI. Emerging interdisciplinary partnerships support this work. A strategic research plan includes AI as a priority area." },
        ] : [
          { id: "sub-4-3-1-a", description: "The institution is a recognized leader in AI research, publishing influential work, attracting significant funding, and hosting major AI research infrastructure. Research spans fundamental AI science and applied domain-specific AI." },
          { id: "sub-4-3-2-a", description: "Interdisciplinary AI research spans all relevant faculties. The institution contributes to global AI research standards, datasets, and benchmarks. AI impact studies across disciplines inform curriculum and workforce strategy." },
        ],
      })),
    },
  ],
  metadata: {
    totalPillars: 4,
    totalIndicators: 14,
    totalSubIndicators: 33,
    totalCapabilityLevels: 3,
    sourceUrl: "https://www.aicapability.org",
    license: "CC BY-SA 4.0",
    contributors: ["Arizona State University", "AWS", "US Department of Defense", "Galileo Global Education", "GOV.UK", "Imperial College London", "IU International University of Applied Sciences", "Microsoft", "University of Cambridge", "Monash University", "University of Exeter", "University of Sussex Business School", "Wharton School (University of Pennsylvania)"],
    levelSynthesisNote: "The QS AI Capability Framework does not define discrete capability levels in its published open-source materials. The three levels (Basic, Developing, Advanced) and their descriptors were synthesized from sector maturity models for app features requiring a progression model.",
  },
  useCases: ["Institutional AI readiness assessment", "Strategic AI investment planning", "Benchmarking against peer institutions", "90-day AI roadmap generation"],
  crossReferences: ["maturity-the", "maturity-jisc"],
  assessmentQuestions: [
    { id: "ac-q1", dimension: "Regulatory & Ethical Standards", question: "How mature is your institution's AI governance structure?", options: [
      { value: "ac-a1", label: "No formal AI governance exists", level: "acquire" },
      { value: "ac-a2", label: "We have basic governance but it's not institution-wide", level: "deepen" },
      { value: "ac-a3", label: "Comprehensive AI governance with regular reviews", level: "create" },
    ]},
    { id: "ac-q2", dimension: "Course Design & Curriculum", question: "How extensively is AI integrated into teaching and curriculum design?", options: [
      { value: "ac-b1", label: "Individual experiments with no institutional support", level: "acquire" },
      { value: "ac-b2", label: "Supported pilots in several departments", level: "deepen" },
      { value: "ac-b3", label: "Institution-wide AI-enhanced curriculum strategy", level: "create" },
    ]},
    { id: "ac-q3", dimension: "AI in Research Practice", question: "How is AI being used to support research at your institution?", options: [
      { value: "ac-c1", label: "Researchers use AI tools individually, ad hoc", level: "acquire" },
      { value: "ac-c2", label: "Institutional licences and training for AI research tools", level: "deepen" },
      { value: "ac-c3", label: "AI is embedded in research strategy with dedicated support", level: "create" },
    ]},
    { id: "ac-q4", dimension: "Personalised Student Support", question: "How does your institution use AI for student recruitment and support?", options: [
      { value: "ac-d1", label: "No AI use in student-facing services", level: "acquire" },
      { value: "ac-d2", label: "Some AI chatbots or personalisation in use", level: "deepen" },
      { value: "ac-d3", label: "Integrated AI across recruitment, support, and retention", level: "create" },
    ]},
    { id: "ac-q5", dimension: "Leadership & Capability", question: "How prepared is your leadership team to drive AI transformation?", options: [
      { value: "ac-e1", label: "Leadership has limited AI awareness", level: "acquire" },
      { value: "ac-e2", label: "Some leaders champion AI but it's not strategic", level: "deepen" },
      { value: "ac-e3", label: "AI literacy is a leadership priority with clear strategy", level: "create" },
    ]},
  ],
  assessmentTitle: "AI Capability Readiness Assessment",
  assessmentDescription: "Evaluate your institution's AI capability across 4 pillars and 14 indicators",
  showInQuiz: true,
  showInDashboard: true,
  showInLanding: true,
  isBackgroundFramework: false,
  compatibility: [],
  sourceFidelity: "synthesized",
  estimatedAssessmentMinutes: 84,
};

// ────────────────────────────────────────────────
// 5. THE Digital Maturity Index
// ────────────────────────────────────────────────
const THE_MATURITY_LEVELS: Omit<import("./framework-types").Level, "indicators">[] = [
  { id: "incidental", name: "Incidental", description: "Characterized by sporadic, independent digital activities. Limited digital understanding and adaptability, with staff lacking advanced technology skills and training.", order: 1, target: "Institutions at the earliest stage of digital transformation with no coordinated digital strategy" },
  { id: "intentional", name: "Intentional", description: "Digital activities are more purposeful, but not fully streamlined. Efforts in training and integration are ongoing, and technology is increasingly used to improve processes.", order: 2, target: "Institutions beginning to coordinate digital efforts with emerging strategy" },
  { id: "integrated", name: "Integrated", description: "Digital activities are strategic, coordinated, and streamlined. There is a transformation of strategic processes, with enhanced change responsiveness, mature data storage, and analytics-driven decision-making.", order: 3, target: "Institutions with coordinated digital strategy, mature data practices, and digitally skilled workforce" },
  { id: "optimised", name: "Optimised", description: "Institutions are standard-setters with fully automated processes. They lead in digital transformation, integrating technology seamlessly, fostering collaboration, and using data comprehensively to drive decisions.", order: 4, target: "Institutions leading in digital transformation that serve as benchmarks for the sector" },
];

// Helper: base indicators per cross-cutting dimension at each maturity level
// Each dimension has 4-7 base indicators that repeat per pillar, plus a pillar-specific indicator appended
const THE_STRATEGY_INDICATORS: string[][] = [
  ["No formal digital transformation strategy exists for this functional area", "Technology decisions are made reactively without alignment to institutional goals", "IT budget allocation for this area is ad hoc and unplanned", "Change management processes for digital initiatives are absent"],
  ["An emerging digital strategy is being developed for this functional area", "Some technology investments are beginning to align with institutional goals", "IT budget planning is becoming more structured and purposeful", "Initial change management processes are being developed for digital initiatives"],
  ["Digital strategy is fully integrated into planning for this functional area", "Technology investments are aligned with institutional KPIs and strategic goals", "IT budget is coordinated across departments with clear funding strategy", "Change management for digital initiatives is embedded in institutional processes"],
  ["Digital transformation is a core priority with continuous improvement cycles in this area", "Proactive technology forecasting informs strategic planning", "Digital strategy is regularly benchmarked against global standards and peers", "The institution is recognized externally as a leader in digital strategy for this area"],
];

const THE_PEOPLE_INDICATORS: string[][] = [
  ["There is no established culture of exploring or adopting new technologies", "Staff do not engage in ongoing professional development for digital skills", "Dedicated technology support teams are absent or under-resourced", "Digital competence is not recognized in recruitment, evaluations, or promotions", "Digital leadership is not cultivated or modeled by decision-makers"],
  ["An innovative environment for exploring new technologies is beginning to emerge", "Staff are beginning to engage in professional development for digital skills", "A dedicated technology team is being established or expanded", "Digital competence is beginning to be considered in HR processes", "Some leaders are beginning to model and promote digital behaviors"],
  ["A culture of innovation that explores and adopts new technologies is established", "Staff regularly engage in professional development to stay current with technology", "A dedicated, mature technology team supports staff across this functional area", "Digital competence is systematically recognized in recruitment, evaluations, and promotions", "Digital leadership is cultivated through self-assessment and modeling by decision-makers"],
  ["Innovation culture is a defining institutional characteristic with continuous exploration of emerging technologies", "Professional development for digital skills is embedded, continuous, and sector-leading", "Technology support is seamlessly integrated into all functions in this area", "Digital competence is a core requirement across all roles and career pathways", "Digital leadership is a defining feature of governance with leaders serving as sector exemplars"],
];

const THE_TECHNOLOGY_INDICATORS: string[][] = [
  ["Internet access on campus is unreliable or inconsistent for this functional area", "Core digital platforms and tools are not widely available or adopted", "Video conferencing and online communication tools are limited or unavailable", "Emerging technologies such as AI, 5G, blockchain, and immersive tech are not explored", "Network infrastructure lacks flexibility and secure design"],
  ["Internet access is increasingly reliable across most areas", "Core digital platforms are adopted and in use across departments", "Video conferencing and communication tools are available for students and staff", "Some emerging technologies are being explored or piloted", "Network infrastructure improvements are underway with growing attention to security"],
  ["Reliable internet access is provided consistently across the entire campus", "Core digital platforms are fully integrated into workflows across this functional area", "Video conferencing and communication tools are standard across all functions", "Emerging technologies including AI, 5G, and immersive tech are actively adopted", "A flexible, secure network infrastructure is established and maintained"],
  ["Campus-wide internet infrastructure is best-in-class and continuously optimized", "Digital platforms are at the leading edge with continuous innovation", "Communication and collaboration tools are seamlessly integrated across all functions", "The institution leads in the adoption and development of emerging technologies", "Technology infrastructure is fully automated, secure, and serves as a sector benchmark"],
];

const THE_DATA_INDICATORS: string[][] = [
  ["Data collection and storage are inconsistent or not digitized", "Data cannot be easily accessed remotely or across devices", "Departments do not conduct regular analysis of available data", "Data systems are siloed with no integration across platforms", "Predictive analytics are not used for this functional area", "Data security practices are weak or absent"],
  ["Data is increasingly collected and stored in digital formats", "Remote data access on various devices is expanding", "Some departments are conducting analysis of their data", "Initial efforts to integrate data across systems are underway", "Basic predictive analytics are being explored", "Data security awareness is growing with initial measures in place"],
  ["Data is comprehensively collected and stored in well-managed digital systems", "Staff can access data remotely across devices and locations", "Departments routinely conduct data analysis to inform their work", "Data is effectively integrated across systems to maximize usage", "Predictive analytics are used for decision-making in this area", "Data security is managed through comprehensive policies and technology"],
  ["Data storage and management are fully automated and optimized", "Data access is seamless, real-time, and available across all functions", "Advanced data analytics are embedded in processes across this area", "Data ecosystems are fully integrated with no silos between systems", "Predictive modeling comprehensively drives strategy and operations", "Data governance and cybersecurity practices are exemplary and continuously improved"],
];

const THE_UTILIZATION_INDICATORS: string[][] = [
  ["Available technologies are underutilized and not leveraged to improve processes", "Digital tools for collaboration and communication are not effectively used", "Technology is not employed to support evidence-based decision-making", "There is a gap between available technology and its effective application"],
  ["Technology utilization is growing more purposeful with increasing adoption", "Digital tools are beginning to improve collaboration and communication", "Technology is increasingly used to support some decision-making processes", "Efforts are underway to close the gap between technology availability and usage"],
  ["Technology is strategically utilized across this functional area", "Digital tools effectively enhance collaboration and break down silos", "Technology systematically supports evidence-based decision-making", "Available technology is well-matched to institutional needs and fully leveraged"],
  ["Technology utilization is maximized and continuously optimized in this area", "Collaboration is seamless with no communication barriers", "Data and technology comprehensively drive all decisions in this area", "The institution sets sector standards for effective technology utilization"],
];

// Pillar-specific extra indicators per dimension
const THE_PILLAR_SPECIFIC: Record<string, Record<string, string[]>> = {
  tl: {
    technology: [
      "Learning management systems are not widely available or adopted for teaching",
      "LMS is increasingly adopted for course delivery and assessment",
      "LMS and learning technologies are fully integrated into teaching, learning, and assessment",
      "LMS and learning technologies are at the leading edge with sector-leading innovation",
    ],
    data: [
      "Learning analytics and student performance data are not collected or used",
      "Basic learning analytics are being explored for student support",
      "Learning analytics inform personalized student support and course optimization",
      "Advanced learning analytics and predictive models drive personalized learning at scale",
    ],
    utilization: [
      "LMS is not utilized for blended, hybrid, or online learning options",
      "LMS is increasingly utilized for blended and hybrid learning",
      "LMS is fully utilized for diverse learning modalities, providing flexibility and accommodating various learning styles",
      "Learning technologies are used innovatively to create sector-leading flexible learning experiences",
    ],
  },
  re: {
    technology: [
      "Research collaboration platforms and data tools are limited or unavailable",
      "Research collaboration platforms are being adopted and expanded",
      "Research tools and collaboration platforms are fully integrated into the research lifecycle",
      "Research technology infrastructure enables cutting-edge, globally connected research",
    ],
    data: [
      "Research data management practices are absent or inconsistent",
      "Research data management practices are being developed",
      "Research data is well-managed with open data practices and cross-collaboration sharing",
      "Research data management is automated with comprehensive open data practices and global sharing",
    ],
    utilization: [
      "Digital tools are not used for research collaboration or dissemination",
      "Researchers are beginning to use digital networks for collaboration",
      "Digital networks are used effectively for research collaboration and dissemination",
      "Digital scholarship and open research practices are sector-leading",
    ],
  },
  ps: {
    technology: [
      "Administrative and IT service platforms are basic and fragmented",
      "Administrative platforms are becoming more integrated and user-friendly",
      "Administrative and IT platforms effectively support all operational services",
      "Operational technology is fully automated and optimized for efficiency",
    ],
    data: [
      "Operational and administrative data are fragmented across systems",
      "Operational data systems are beginning to be consolidated",
      "Operational data is integrated across HR, finance, and student administration",
      "Operational data drives fully automated, predictive institutional management",
    ],
    utilization: [
      "Administrative platforms are underutilized across operations",
      "Administrative platforms are increasingly adopted with growing IT support",
      "Administrative platforms are effectively used with robust IT support ensuring smooth operation",
      "Operational technology utilization is fully optimized and serves as a sector benchmark",
    ],
  },
  pg: {
    technology: [
      "Governance and decision-support systems are absent or rudimentary",
      "Governance dashboards and planning tools are being introduced",
      "Enterprise-level governance tools and dashboards support strategic decision-making",
      "Governance technology provides real-time, comprehensive institutional intelligence",
    ],
    data: [
      "Institutional performance data is not used for strategic planning",
      "Institutional data dashboards are being developed for strategic use",
      "Institutional data comprehensively supports evidence-based governance and strategic planning",
      "Institutional intelligence is fully automated with real-time, comprehensive governance dashboards",
    ],
    utilization: [
      "Communication tools do not break down organizational silos",
      "Communication tools are beginning to improve cross-department coordination",
      "Communication tools effectively break down silos and support cross-functional governance",
      "Governance processes are fully data-driven with sector-leading strategic technology utilization",
    ],
  },
};

interface THEDimConfig {
  dimKey: string;
  dimName: string;
  dimDesc: string;
  icon: string;
  color: string;
  baseIndicators: string[][];
  pillarSpecificKey?: string;
}

const THE_DIM_CONFIGS: THEDimConfig[] = [
  { dimKey: "strategy", dimName: "Strategy", dimDesc: "Strategic digital transformation planning", icon: "Target", color: "text-blue-600", baseIndicators: THE_STRATEGY_INDICATORS },
  { dimKey: "people", dimName: "People & Culture", dimDesc: "Digital skills, confidence, and organizational culture", icon: "Users", color: "text-purple-600", baseIndicators: THE_PEOPLE_INDICATORS },
  { dimKey: "technology", dimName: "Technology", dimDesc: "Infrastructure integration and emerging tech adoption", icon: "Zap", color: "text-cyan-600", baseIndicators: THE_TECHNOLOGY_INDICATORS, pillarSpecificKey: "technology" },
  { dimKey: "data", dimName: "Data", dimDesc: "Data analytics, integration, and evidence-based decision-making", icon: "BarChart", color: "text-amber-600", baseIndicators: THE_DATA_INDICATORS, pillarSpecificKey: "data" },
  { dimKey: "utilization", dimName: "Utilisation", dimDesc: "Adoption and effective use of digital tools", icon: "TrendingUp", color: "text-emerald-600", baseIndicators: THE_UTILIZATION_INDICATORS, pillarSpecificKey: "utilization" },
];

interface THEPillarConfig {
  pillarKey: string;
  pillarName: string;
  pillarShort: string;
  pillarDesc: string;
}

const THE_PILLARS: THEPillarConfig[] = [
  { pillarKey: "tl", pillarName: "Teaching and Learning", pillarShort: "T&L", pillarDesc: "Accreditation of study programs, teaching and assessment methods, and student/teacher mobility" },
  { pillarKey: "re", pillarName: "Research", pillarShort: "Research", pillarDesc: "Full lifecycle of research activities, from planning and preparation to conducting research, monitoring outcomes, and evaluating results" },
  { pillarKey: "ps", pillarName: "Professional Services", pillarShort: "Prof Services", pillarDesc: "IT support, library services, student administration, staff recruitment, financial management, marketing, procurement, and estate management" },
  { pillarKey: "pg", pillarName: "Planning and Governance", pillarShort: "Planning & Gov", pillarDesc: "Change management, business process development, central IT strategy, and IT budget planning" },
];

// Generate all 20 child dimensions programmatically
let theDimOrder = 0;
const theDimensions: import("./framework-types").FrameworkDimension[] = THE_PILLARS.flatMap((pillar) =>
  THE_DIM_CONFIGS.map((dim) => {
    theDimOrder++;
    const pillarSpecific = dim.pillarSpecificKey ? THE_PILLAR_SPECIFIC[pillar.pillarKey]?.[dim.pillarSpecificKey] : undefined;
    return {
      id: `the-${pillar.pillarKey}-${dim.dimKey}`,
      name: `${dim.dimName} (${pillar.pillarShort})`,
      description: `${pillar.pillarName} — ${dim.dimDesc}`,
      order: theDimOrder,
      icon: dim.icon,
      color: dim.color,
      levels: THE_MATURITY_LEVELS.map((l, li) => {
        const baseInds = dim.baseIndicators[li].map((desc, idx) => ({
          id: `ind-${pillar.pillarKey}-${dim.dimKey}-${li + 1}-${String(idx + 1).padStart(2, "0")}`,
          description: desc,
        }));
        if (pillarSpecific) {
          baseInds.push({
            id: `ind-${pillar.pillarKey}-${dim.dimKey}-${li + 1}-ps`,
            description: pillarSpecific[li],
          });
        }
        return { ...l, id: `the-${pillar.pillarKey}-${dim.dimKey}-${l.id}`, indicators: baseInds };
      }),
    };
  })
);

const maturityTHE: Framework = {
  id: "maturity-the",
  name: "THE Digital Maturity Index",
  shortName: "THE DMI",
  description: "The Digital Maturity Index measures how advanced universities are in their digital transformation journey across 4 pillars, 5 cross-cutting dimensions, and 4 maturity stages.",
  type: "maturity",
  scope: "institutional",
  source: "THE",
  region: "international",
  path: "/frameworks/maturity-the",
  icon: "TrendingUp",
  color: "text-emerald-600",
  badgeLabel: "Times Higher Education",
  targetAudience: ["leader", "admin"],
  overview: `The Times Higher Education Digital Maturity Index (DMI) measures how advanced universities are in their digital transformation journey. Developed through a quantitative survey of 3,863 respondents across 1,949 institutions in 100 countries, the DMI assesses four core pillars (Teaching & Learning, Research, Professional Services, Planning & Governance) across five cross-cutting dimensions (Strategy, People & Culture, Technology, Data, Utilisation) at four maturity stages (Incidental, Intentional, Integrated, Optimised). The framework reveals that technology acquisition without effective utilisation is insufficient, and that people issues — not technology — are the primary barrier to digital transformation.`,
  keyPrinciples: [
    { id: "the-p1", name: "Multidimensional maturity", description: "Digital maturity encompasses strategy, people, culture, technology, data, and utilization, not just technology acquisition." },
    { id: "the-p2", name: "Context-sensitive transformation", description: "Universities must consider their student population, staff needs, funding, and broader institutional and societal goals." },
    { id: "the-p3", name: "Utilization over acquisition", description: "Technology acquisition without effective utilization is insufficient. Effective digital transformation requires both investment in and strategic use of technology." },
    { id: "the-p4", name: "People as the critical factor", description: "Technology initiatives most often fail due to people issues, not technology issues. Digital leadership and workforce development are essential." },
    { id: "the-p5", name: "Data maturity as foundation", description: "The ability to collect, integrate, analyze, and predict from data is foundational to leveraging AI, machine learning, and evidence-based decision-making." },
    { id: "the-p6", name: "Global digital equity", description: "Significant disparities exist between high-resource and low-resource institutions, and these must be addressed for equitable access to education and research." },
    { id: "the-p7", name: "Scalable and adaptable design", description: "The framework is designed to cater to institutions of all sizes globally and can be customized to align with each university's specific KPIs." },
  ],
  keyDimensions: theDimensions,
  metadata: {
    respondents: 3863,
    institutions: 1949,
    countries: 100,
    surveyPeriod: "February 2024 to July 2024",
    maturityLevelNames: ["Incidental", "Intentional", "Integrated", "Optimised"],
    pillars: 4,
    crossCuttingDimensions: 5,
    childDimensions: 20,
    competencyBlocks: 80,
    totalIndicators: 432,
    sourceUrl: "https://www.timeshighereducation.com/digital-maturity-index",
    baseFrameworkReference: "Marks, A., & AL-Ali, M. (2020). Digital Transformation in Higher Education: A Framework for Maturity Assessment.",
    keyFindings: [
      "Technology vs Utilisation gap: Higher-income institutions tend to acquire technology without fully exploiting it",
      "People & Culture is the weakest dimension globally, with digital leadership and competence recognition scoring lowest",
      "Cybersecurity: The human factor remains the weakest link across all regions (67% global vs 77% technology, 74% policy)",
      "Data maturity: Predictive analytics (67%) and data integration (68%) lag behind storage (78%) and access (76%)",
      "Sub-Saharan Africa and Latin America score lowest on technology availability but show strong utilisation relative to availability",
    ],
  },
  useCases: ["Benchmarking digital maturity globally", "Identifying gaps in digital strategy", "Planning technology investments", "Reporting to leadership on transformation progress"],
  crossReferences: ["maturity-jisc", "ai-capability"],
  assessmentQuestions: [
    // ── Teaching & Learning ──
    { id: "the-q1", dimension: "the-tl-strategy", question: "How clearly defined is your institution's digital strategy for teaching and learning?", options: [
      { value: "the-q1-a", label: "No formal digital strategy exists for teaching and learning", level: "incidental" },
      { value: "the-q1-b", label: "An emerging T&L digital strategy is being developed but not yet funded or widely communicated", level: "intentional" },
      { value: "the-q1-c", label: "Digital strategy is fully integrated into T&L planning with aligned KPIs and coordinated funding", level: "integrated" },
      { value: "the-q1-d", label: "T&L digital strategy is continuously improved, benchmarked against global peers, and drives sector innovation", level: "optimised" },
    ]},
    { id: "the-q2", dimension: "the-tl-people", question: "How would you describe the digital culture and skills development among teaching staff?", options: [
      { value: "the-q2-a", label: "Staff lack digital skills and there is no culture of exploring new teaching technologies", level: "incidental" },
      { value: "the-q2-b", label: "Some staff engage in digital skills training but uptake is inconsistent and not incentivised", level: "intentional" },
      { value: "the-q2-c", label: "Regular professional development is established and digital competence is recognised in evaluations", level: "integrated" },
      { value: "the-q2-d", label: "Digital skills development is embedded and continuous; digital leadership is modelled at all levels", level: "optimised" },
    ]},
    { id: "the-q3", dimension: "the-tl-data", question: "How effectively does your institution use learning analytics and student data?", options: [
      { value: "the-q3-a", label: "Learning analytics and student performance data are not collected or used", level: "incidental" },
      { value: "the-q3-b", label: "Basic learning analytics are being explored for student support in some departments", level: "intentional" },
      { value: "the-q3-c", label: "Learning analytics inform personalised student support and course optimisation across programmes", level: "integrated" },
      { value: "the-q3-d", label: "Advanced predictive models drive personalised learning at scale with real-time dashboards", level: "optimised" },
    ]},
    // ── Research ──
    { id: "the-q4", dimension: "the-re-technology", question: "How well are digital research tools and collaboration platforms integrated into the research lifecycle?", options: [
      { value: "the-q4-a", label: "Research collaboration platforms and data tools are limited or unavailable", level: "incidental" },
      { value: "the-q4-b", label: "Research collaboration platforms are being adopted and expanded across some groups", level: "intentional" },
      { value: "the-q4-c", label: "Research tools and collaboration platforms are fully integrated into the research lifecycle", level: "integrated" },
      { value: "the-q4-d", label: "Research technology infrastructure enables cutting-edge, globally connected research", level: "optimised" },
    ]},
    { id: "the-q5", dimension: "the-re-utilization", question: "How effectively do researchers use digital tools for collaboration and dissemination?", options: [
      { value: "the-q5-a", label: "Digital tools are not used for research collaboration or dissemination", level: "incidental" },
      { value: "the-q5-b", label: "Researchers are beginning to use digital networks for collaboration", level: "intentional" },
      { value: "the-q5-c", label: "Digital networks are used effectively for research collaboration and dissemination", level: "integrated" },
      { value: "the-q5-d", label: "Digital scholarship and open research practices are sector-leading", level: "optimised" },
    ]},
    // ── Professional Services ──
    { id: "the-q6", dimension: "the-ps-strategy", question: "How strategically is digital transformation planned for professional services (IT, HR, finance, administration)?", options: [
      { value: "the-q6-a", label: "No coordinated digital strategy exists for professional services; decisions are reactive", level: "incidental" },
      { value: "the-q6-b", label: "An emerging strategy for digitising services is being developed with some alignment to goals", level: "intentional" },
      { value: "the-q6-c", label: "Digital strategy for professional services is fully integrated with institutional KPIs and coordinated funding", level: "integrated" },
      { value: "the-q6-d", label: "Professional services digital strategy is a core institutional priority with continuous improvement cycles", level: "optimised" },
    ]},
    { id: "the-q7", dimension: "the-ps-data", question: "How well is operational and administrative data integrated across professional services?", options: [
      { value: "the-q7-a", label: "Operational data is fragmented across siloed systems with no integration", level: "incidental" },
      { value: "the-q7-b", label: "Data systems are beginning to be consolidated across HR, finance, and student administration", level: "intentional" },
      { value: "the-q7-c", label: "Operational data is effectively integrated across systems to support decision-making", level: "integrated" },
      { value: "the-q7-d", label: "Operational data drives fully automated, predictive institutional management", level: "optimised" },
    ]},
    // ── Planning & Governance ──
    { id: "the-q8", dimension: "the-pg-people", question: "How well does institutional leadership model and promote digital transformation?", options: [
      { value: "the-q8-a", label: "Digital leadership is not cultivated or modelled by decision-makers", level: "incidental" },
      { value: "the-q8-b", label: "Some leaders are beginning to model and promote digital behaviours", level: "intentional" },
      { value: "the-q8-c", label: "Digital leadership is cultivated through self-assessment and actively modelled by governance leaders", level: "integrated" },
      { value: "the-q8-d", label: "Digital leadership is a defining feature of governance with leaders serving as sector exemplars", level: "optimised" },
    ]},
    { id: "the-q9", dimension: "the-pg-technology", question: "How mature are your governance and decision-support technology systems?", options: [
      { value: "the-q9-a", label: "Governance and decision-support systems are absent or rudimentary", level: "incidental" },
      { value: "the-q9-b", label: "Governance dashboards and planning tools are being introduced", level: "intentional" },
      { value: "the-q9-c", label: "Enterprise-level governance tools and dashboards support strategic decision-making", level: "integrated" },
      { value: "the-q9-d", label: "Governance technology provides real-time, comprehensive institutional intelligence", level: "optimised" },
    ]},
    { id: "the-q10", dimension: "the-pg-utilization", question: "How effectively do communication and decision tools break down organisational silos in governance?", options: [
      { value: "the-q10-a", label: "Communication tools do not break down organisational silos", level: "incidental" },
      { value: "the-q10-b", label: "Communication tools are beginning to improve cross-department coordination", level: "intentional" },
      { value: "the-q10-c", label: "Communication tools effectively break down silos and support cross-functional governance", level: "integrated" },
      { value: "the-q10-d", label: "Governance processes are fully data-driven with sector-leading strategic technology utilisation", level: "optimised" },
    ]},
  ],
  assessmentTitle: "Digital Maturity Self-Assessment (THE DMI)",
  assessmentDescription: "Assess your institution's digital maturity across 4 pillars and 5 cross-cutting dimensions (10 questions, ~12 min)",
  showInQuiz: true,
  showInDashboard: true,
  showInLanding: true,
  isBackgroundFramework: false,
  compatibility: [],
  sourceFidelity: "synthesized",
  estimatedAssessmentMinutes: 12,
};

// ────────────────────────────────────────────────
// 6. JISC Digital Maturity Model (DT)
// ────────────────────────────────────────────────
const maturityJISC: Framework = {
  id: "maturity-jisc",
  name: "Digital Transformation Maturity Model (JISC)",
  shortName: "JISC DT Maturity",
  description: "5-area maturity model for institutional digital transformation",
  type: "maturity",
  scope: "institutional",
  source: "JISC",
  path: "/frameworks/maturity-jisc",
  icon: "Target",
  color: "text-amber-600",
  badgeLabel: "JISC Framework",
  targetAudience: ["leader", "admin"],
  overview: `The JISC Digital Transformation Maturity Model helps UK HE providers assess digital maturity across 5 key areas with 3 maturity levels.`,
  keyDimensions: [
    {
      id: "jisc-culture", name: "Organisational Digital Culture", description: "How digital thinking is embedded in institutional identity",
      order: 1, icon: "Users", color: "text-blue-600",
      levels: [
        { id: "jisc-cult-emerging", name: "Emerging to Established", description: "Ad-hoc digital adoption", order: 1, indicators: [
          { id: "jisc-cult-e1", description: "Lack of strategic digital leadership" },
          { id: "jisc-cult-e2", description: "Short-term, project-based digital investment" },
        ]},
        { id: "jisc-cult-enhanced", name: "Established to Enhanced", description: "Proactive digital strategy developing", order: 2, indicators: [
          { id: "jisc-cult-d1", description: "Developing proactive strategic approach to digital", assessmentCriteria: "Strategic plan references digital transformation" },
          { id: "jisc-cult-d2", description: "Enabling effective digital leadership" },
        ]},
        { id: "jisc-cult-mature", name: "Enhanced to Mature", description: "Digital integral to institutional identity", order: 3, indicators: [
          { id: "jisc-cult-m1", description: "Comprehensive integrated strategies across all areas", assessmentCriteria: "Digital embedded in institutional strategic plan" },
          { id: "jisc-cult-m2", description: "Effective digital leadership at all levels" },
        ]},
      ],
    },
    {
      id: "jisc-innovation", name: "Knowledge Creation & Innovation", description: "Support for digital innovation",
      order: 2, icon: "Lightbulb", color: "text-amber-600",
      levels: [
        { id: "jisc-inno-emerging", name: "Emerging to Established", description: "Innovation despite limited support", order: 1, indicators: [
          { id: "jisc-inno-e1", description: "Innovation happens despite, not because of, institutional support" },
        ]},
        { id: "jisc-inno-enhanced", name: "Established to Enhanced", description: "Innovation funding emerging", order: 2, indicators: [
          { id: "jisc-inno-d1", description: "Some innovation funding and support structures exist" },
        ]},
        { id: "jisc-inno-mature", name: "Enhanced to Mature", description: "Systematic innovation support", order: 3, indicators: [
          { id: "jisc-inno-m1", description: "Systematic innovation support with clear pathways", assessmentCriteria: "Documented innovation pipeline with funding" },
        ]},
      ],
    },
    {
      id: "jisc-development", name: "Knowledge Development", description: "Digital skills development programmes",
      order: 3, icon: "GraduationCap", color: "text-purple-600",
      levels: [
        { id: "jisc-dev-emerging", name: "Emerging to Established", description: "Ad-hoc training", order: 1, indicators: [
          { id: "jisc-dev-e1", description: "Dispersed, uncoordinated digital skills training" },
        ]},
        { id: "jisc-dev-enhanced", name: "Established to Enhanced", description: "Structured training developing", order: 2, indicators: [
          { id: "jisc-dev-d1", description: "Structured training programmes but not linked to specific roles" },
        ]},
        { id: "jisc-dev-mature", name: "Enhanced to Mature", description: "Role-specific development", order: 3, indicators: [
          { id: "jisc-dev-m1", description: "Role-specific, progressive digital development programme", assessmentCriteria: "Competency frameworks mapped to job families" },
        ]},
      ],
    },
    {
      id: "jisc-management", name: "Knowledge Management & Use", description: "Systems for managing and sharing digital knowledge",
      order: 4, icon: "Database", color: "text-emerald-600",
      levels: [
        { id: "jisc-mgmt-emerging", name: "Emerging to Established", description: "Knowledge siloed", order: 1, indicators: [
          { id: "jisc-mgmt-e1", description: "Knowledge siloed in departments with limited sharing" },
        ]},
        { id: "jisc-mgmt-enhanced", name: "Established to Enhanced", description: "Cross-team sharing developing", order: 2, indicators: [
          { id: "jisc-mgmt-d1", description: "Some cross-team knowledge sharing but not systematic" },
        ]},
        { id: "jisc-mgmt-mature", name: "Enhanced to Mature", description: "Comprehensive knowledge systems", order: 3, indicators: [
          { id: "jisc-mgmt-m1", description: "Comprehensive knowledge sharing with clear systems", assessmentCriteria: "Knowledge management platform in active use" },
        ]},
      ],
    },
    {
      id: "jisc-exchange", name: "Knowledge Exchange & Partnerships", description: "External collaboration and sector-wide partnerships",
      order: 5, icon: "Share2", color: "text-rose-600",
      levels: [
        { id: "jisc-exch-emerging", name: "Emerging to Established", description: "Limited external collaboration", order: 1, indicators: [
          { id: "jisc-exch-e1", description: "Limited external collaboration on digital matters" },
        ]},
        { id: "jisc-exch-enhanced", name: "Established to Enhanced", description: "Some partnerships", order: 2, indicators: [
          { id: "jisc-exch-d1", description: "Some sector partnerships but not strategic" },
        ]},
        { id: "jisc-exch-mature", name: "Enhanced to Mature", description: "Strategic partnerships", order: 3, indicators: [
          { id: "jisc-exch-m1", description: "Active strategic partnerships driving transformation", assessmentCriteria: "Partnership outcomes documented and reviewed" },
        ]},
      ],
    },
  ],
  metadata: { region: "UK", sector: "Higher Education" },
  useCases: ["Benchmarking against UK HE sector", "Identifying transformation priorities", "Creating actionable roadmaps", "Reporting progress to governing bodies"],
  crossReferences: ["maturity-the", "ai-capability"],
  assessmentQuestions: [
    { id: "jisc-q1", dimension: "Digital Culture", question: "How embedded is digital thinking in your institution's culture?", options: [
      { value: "jisc-a1", label: "Digital is seen as an IT issue, not a cultural one", level: "acquire" },
      { value: "jisc-a2", label: "Growing awareness but not embedded in values", level: "deepen" },
      { value: "jisc-a3", label: "Digital is integral to our institutional identity", level: "create" },
    ]},
    { id: "jisc-q2", dimension: "Knowledge Creation & Innovation", question: "How well does your institution support digital innovation?", options: [
      { value: "jisc-b1", label: "Innovation happens despite, not because of, support", level: "acquire" },
      { value: "jisc-b2", label: "Some innovation funding and support structures exist", level: "deepen" },
      { value: "jisc-b3", label: "Systematic innovation support with clear pathways", level: "create" },
    ]},
    { id: "jisc-q3", dimension: "Knowledge Development", question: "How effective is your approach to digital skills development?", options: [
      { value: "jisc-c1", label: "Ad-hoc training with no strategic plan", level: "acquire" },
      { value: "jisc-c2", label: "Structured training but not linked to roles", level: "deepen" },
      { value: "jisc-c3", label: "Role-specific, progressive digital development", level: "create" },
    ]},
    { id: "jisc-q4", dimension: "Knowledge Management", question: "How well does your institution manage and share digital knowledge?", options: [
      { value: "jisc-d1", label: "Knowledge is siloed in departments", level: "acquire" },
      { value: "jisc-d2", label: "Some cross-team sharing but not systematic", level: "deepen" },
      { value: "jisc-d3", label: "Comprehensive knowledge sharing with clear systems", level: "create" },
    ]},
    { id: "jisc-q5", dimension: "Partnerships & Exchange", question: "How effectively does your institution collaborate externally?", options: [
      { value: "jisc-e1", label: "Limited external collaboration on digital matters", level: "acquire" },
      { value: "jisc-e2", label: "Some sector partnerships but not strategic", level: "deepen" },
      { value: "jisc-e3", label: "Active strategic partnerships driving transformation", level: "create" },
    ]},
  ],
  assessmentTitle: "JISC Digital Transformation Maturity Assessment",
  assessmentDescription: "Assess your organisation's digital maturity across 5 key areas",
  showInQuiz: true,
  showInDashboard: true,
  showInLanding: true,
  isBackgroundFramework: false,
  compatibility: [],
  sourceFidelity: "official",
  estimatedAssessmentMinutes: 60,
  region: "uk",
};

// ────────────────────────────────────────────────
// 7. OECD AI Capability Indicators (2025)
// ────────────────────────────────────────────────

const OECD_LEVELS: Omit<import("./framework-types").Level, "indicators">[] = [
  { id: "level-1", name: "Level 1", description: "Solved AI challenges in highly controlled, structured environments. Often quantitatively superhuman but only within tightly constrained situations.", order: 1, target: "AI systems performing basic, well-defined tasks in controlled settings" },
  { id: "level-2", name: "Level 2", description: "Some variability handled; multiple reasoning types integrated within defined domains. Semi-structured environments with moderate variation.", order: 2, target: "AI systems operating in semi-structured environments with some variability" },
  { id: "level-3", name: "Level 3", description: "Generalised capabilities across multiple data types, domains, or contexts. Standard benchmarks available but lacks real-time learning and full adaptability.", order: 3, target: "AI systems with generalised but still bounded capabilities" },
  { id: "level-4", name: "Level 4", description: "Advanced capabilities including learning through feedback, adapting to dynamic/unfamiliar conditions. Approaches but does not match full human equivalence.", order: 4, target: "AI systems with advanced, near-human capabilities in specific domains" },
  { id: "level-5", name: "Level 5", description: "Full human equivalence across all aspects of the capability. The ceiling against which AI progress is measured.", order: 5, target: "AI systems achieving full human equivalence across all aspects of the capability" },
];

const OECD_DIMS: { id: string; name: string; description: string; icon: string; color: string; currentLevel: number; blocks: { levelIdx: number; desc: string; tasks: string[] }[] }[] = [
  {
    id: "language", name: "Language", icon: "MessageSquare", color: "text-blue-600", currentLevel: 3,
    description: "Language is an essential human ability that provides the foundation for many cognitive tasks. The Language scale covers AI's ability in meaning, modality, multilingualism, knowledge access, reasoning, and learning.",
    blocks: [
      { levelIdx: 0, desc: "Relies on keyword matching or highlighting for language interpretation and generation, with no world knowledge or reasoning capabilities. Monolingual, text-only.", tasks: ["Keyword-based web search"] },
      { levelIdx: 1, desc: "Produces grammatically correct language with single-corpus knowledge and basic analytics. Two modalities in well-resourced languages.", tasks: ["Syntactic parsing"] },
      { levelIdx: 2, desc: "Reliably interprets and generates correct meanings with multi-corpus knowledge, some problem solving, logic and social reasoning. Most modalities and multiple languages.", tasks: ["Essay scoring", "Text classification reflecting multi-corpus knowledge and advanced semantic and syntactic capabilities"] },
      { levelIdx: 3, desc: "Appropriately interprets context, leveraging web-scale world knowledge for complex analysis. All modalities and highly diverse languages including low-resource.", tasks: ["Dialogue depending on contextual understanding, web-scale knowledge and processing diverse language inputs"] },
      { levelIdx: 4, desc: "Nuanced language abilities capturing style, tone and humour with real-time world knowledge and critical thinking. Can learn any language on the fly from small datasets.", tasks: ["Automatic video description generation (video captioning)", "Structured reasoning tasks relying on critical thinking, real-time knowledge and real-world multimodal inputs"] },
    ],
  },
  {
    id: "social-interaction", name: "Social interaction", icon: "Users", color: "text-emerald-600", currentLevel: 2,
    description: "Social intelligence in perceiving, interpreting and responding to social cues in dynamic interpersonal contexts. Integrates embodiment, social memory, identity, communication, affect, perception, and problem solving.",
    blocks: [
      { levelIdx: 0, desc: "Simple, rigid social behaviours with basic movements and emotional cues. Fixed memory, static identity, pre-set scripted responses. Minimal social perception.", tasks: ["Detecting presence of others and solving simple static tasks"] },
      { levelIdx: 1, desc: "Begins to adapt socially, combining movements to express emotions. Limited social memory, recalls events. Basic signal recognition and emotion detection.", tasks: ["Recognising individuals and applying past experiences to recurring problems"] },
      { levelIdx: 2, desc: "Interprets body language, mimics group interactions, maintains evolving personality. Infers social intent and interprets behavioural cues.", tasks: ["Co-ordinating turn-taking at intersections", "Managing simple group dynamics"] },
      { levelIdx: 3, desc: "Highly natural social behaviour adapting gestures to scenarios. Structured social memory, clear group role, handles ambiguity and emotional intensity.", tasks: ["Attracting a waiter's attention", "Determining student disengagement", "Deciding when to interrupt a group"] },
      { levelIdx: 4, desc: "Seamlessly integrates into any social environment. Unlimited adaptive social memory, fully aligned context-aware identity, profound communication.", tasks: ["Describing scenes from another's perspective", "Learning new social norms", "Gauging distant social openness"] },
    ],
  },
  {
    id: "problem-solving", name: "Problem solving", icon: "Puzzle", color: "text-purple-600", currentLevel: 2,
    description: "Integrating qualitative, quantitative and logical information through multi-step reasoning including analysis, prediction, explanation and counterfactual thinking.",
    blocks: [
      { levelIdx: 0, desc: "Operates in structured domains using precise, domain-specific terms. Analyses data for discrepancies and performs planning/scheduling within predefined parameters.", tasks: ["Solving structured problems in mathematics, sciences, medicine or engineering where the problem is specified"] },
      { levelIdx: 1, desc: "Integrates qualitative reasoning (spatial/temporal) with quantitative analysis. Envisions multiple states and transitions, predicting system evolution.", tasks: ["Solving problems where the problem is described using conventional domain abstractions"] },
      { levelIdx: 2, desc: "Handles problems in everyday language, translating to structured models. Incorporates social cognition and theory of mind reasoning.", tasks: ["Solving word problems on standardised tests", "Solving social and ethical reasoning problems directly described"] },
      { levelIdx: 3, desc: "Solves everyday commonsense and professional problems. Builds rapport, leverages social/psychological/physical knowledge, learns from experience.", tasks: ["Interpreting interactions in complex social environments, identifying and developing approaches for problems"] },
      { levelIdx: 4, desc: "Solves complex multidisciplinary problems integrating tacit, social and technical knowledge. Forms long-term relationships, navigates ethical challenges.", tasks: ["Identifying and solving unstructured real-world problems involving social complexity and multiple domains"] },
    ],
  },
  {
    id: "creativity", name: "Creativity", icon: "Sparkles", color: "text-amber-600", currentLevel: 3,
    description: "Creative capabilities from mimicry through novelty and surprise to full intentionality and autonomy. Measured across value, novelty, transformativity, surprise, intentionality, self-assessment, and adaptability.",
    blocks: [
      { levelIdx: 0, desc: "Replicates human outputs to solve non-trivial tasks effectively. Results are valuable but without true creative properties — mimicry as a steppingstone.", tasks: ["Generating recipe variations by substituting ingredients", "Drawing objects with modifications", "Creating simple music following a specific meter and style"] },
      { levelIdx: 1, desc: "Moves beyond imitation to create valuable, novel solutions that differ from training data. Explores possibilities within task constraints.", tasks: ["Painting portraits in historical master styles", "Writing genre-blending short stories", "Developing videogames with automatically generated novel city levels"] },
      { levelIdx: 2, desc: "Generates valuable, novel and surprising outputs, deviating significantly from training data. Generalises skills to new tasks, integrates ideas across domains.", tasks: ["Winning videogames with unexpected strategies", "Participating successfully in political debates", "Composing multimedia installations conveying complex narratives"] },
      { levelIdx: 3, desc: "Incorporates process-oriented creativity, adapting to evolving domains through iterative blind exploratory search. Mirrors general population creativity.", tasks: ["Writing occasion-specific speeches with personal humour", "Composing letters reflecting national mood", "Writing thoughtful journal entries"] },
      { levelIdx: 4, desc: "Achieves intentionality, authenticity and full agency. Creates transformative outputs on par with world-class human creators with autonomous goal-setting.", tasks: ["Designing dominant fashion styles", "Writing acclaimed international bestseller autobiographies", "Designing disruptive technologies"] },
    ],
  },
  {
    id: "metacognition", name: "Metacognition & critical thinking", icon: "Brain", color: "text-rose-600", currentLevel: 2,
    description: "Capability to evaluate own reasoning, calibrate confidence and identify relevant information. Covers critical thinking processes, calibrated self-assessment, and information essentiality.",
    blocks: [
      { levelIdx: 0, desc: "Minimal metacognition — basic interpretation or recognition. Familiar/straightforward subject matter, simple information filtering.", tasks: ["Cooking for guests with dietary requirements — determining recipe adaptability and time estimation"] },
      { levelIdx: 1, desc: "Moderate metacognition — monitoring understanding and adjusting approaches. Partially familiar subjects with ambiguities requiring measured confidence.", tasks: ["Weekly shopping with budget constraints — resolving trade-offs and substitutions based on customer preferences"] },
      { levelIdx: 2, desc: "Significant metacognition — analysis and synthesis of familiar and unfamiliar concepts. Critical evaluation of knowledge, strategic problem solving.", tasks: ["Encountering an unfamiliar door handle — seeking information or trying different approaches"] },
      { levelIdx: 3, desc: "High-level metacognition — active regulation of thought processes. Complex, ambiguous problems in unfamiliar domains with incomplete information.", tasks: ["Performing paperwork — determining if all required attachments are present or need to be requested"] },
      { levelIdx: 4, desc: "Sophisticated metacognition — managing complex trade-offs between goals, resources and skills. Long-term intersecting tasks requiring delegation decisions.", tasks: ["Finding and sending a file with approximate name matching, assessing system access capabilities"] },
    ],
  },
  {
    id: "knowledge-learning-memory", name: "Knowledge, learning & memory", icon: "Database", color: "text-cyan-600", currentLevel: 3,
    description: "Storage, retrieval and acquisition of information. Covers explicit vs implicit knowledge, learning sources, passive vs active learning, generalisation processes, and memory systems.",
    blocks: [
      { levelIdx: 0, desc: "Storing and retrieving structured information through precise computational methods. Formal formats like tables and rules with logical queries.", tasks: ["Precise record keeping: financial accounting, statistics, schedule management"] },
      { levelIdx: 1, desc: "Searching loosely organised information without rigid structuring. Statistical inference connecting search terms with relevant results.", tasks: ["Information search: online shopping, news gathering, travel planning, product reviews"] },
      { levelIdx: 2, desc: "Learning semantics using distributed representations for meaning extraction and generalisation. Advanced algorithms processing massive datasets.", tasks: ["Content generation: writing stories, creating illustrations, summarising information, programming"] },
      { levelIdx: 3, desc: "Learning incrementally through world interaction. Metacognitive awareness to focus on knowledge gaps, balancing exploration and exploitation.", tasks: ["Operating in unknown/uncertain environments: household tasks, elderly support, open-floor industrial settings"] },
      { levelIdx: 4, desc: "Integrating diverse knowledge types, learning methods and memory systems for robust real-time adaptation. Human-like cognitive flexibility.", tasks: ["Open-ended cognitive flexibility: scientific research, public policy decisions, legal argumentation"] },
    ],
  },
  {
    id: "vision", name: "Vision", icon: "Eye", color: "text-indigo-600", currentLevel: 3,
    description: "Visual perception from controlled single-task recognition to full human equivalence. Measured across breadth of objects/scenes, environmental robustness, task diversity, and learning capability.",
    blocks: [
      { levelIdx: 0, desc: "Highly controlled environments with minimal variation. Single task, near-perfect but tightly constrained. Manufacturing inspection, postcode recognition.", tasks: ["Basic object recognition in fixed settings, barcode scanning, quality control with well-organised materials"] },
      { levelIdx: 1, desc: "Handles variations in lighting and sensor position. More flexible with speed/timing and object changes. Specialised but limited.", tasks: ["Face detection, obstacle avoidance in controlled driving, specialised manufacturing inspections"] },
      { levelIdx: 2, desc: "Multiple data types (microscopy, RGB, natural scenes). Some variation handling, multiple subtasks. Human-like in some domains.", tasks: ["Autonomous vehicle navigation, facial recognition, environment mapping for robotic systems"] },
      { levelIdx: 3, desc: "Wide range of data types with significant variation handling. Improves through feedback, performs many tasks, near-human level.", tasks: ["Complex manipulation in dynamic environments: diverse kitchen tasks, assembly line monitoring, intricate quality control"] },
      { levelIdx: 4, desc: "Full human visual capability. Handles all variations including unexpected. Self-feedback learning, full spectrum of visual capabilities.", tasks: ["Complex recognition, dynamic tracking and real-time scene understanding across varied environments"] },
    ],
  },
  {
    id: "manipulation", name: "Manipulation", icon: "Hand", color: "text-orange-600", currentLevel: 2,
    description: "Physical interaction with objects integrating movement, perception (tactile/visual), and cognition for planning and adjustment. From pick-and-place to full human dexterity.",
    blocks: [
      { levelIdx: 0, desc: "Simple pick-and-place in well-organised environments. Rigid objects with basic shapes, predefined paths, wide margins of error.", tasks: ["Moving cereal boxes in a warehouse from pre-taught locations into cases"] },
      { levelIdx: 1, desc: "Low to moderate clutter. Random object placement, variety of shapes, some pliable materials. Controlled conditions.", tasks: ["Picking up toy blocks and placing in storage", "Material handling in controlled factory environments"] },
      { levelIdx: 2, desc: "Moderately cluttered environments. Broader object geometries and challenging materials. Can reorient objects and perform force-based operations.", tasks: ["Reorienting irregular objects", "Setting a table for a meal", "Handling delicate materials requiring force-based manipulation"] },
      { levelIdx: 3, desc: "Significant clutter and occlusions. Rigid and non-rigid objects including moving parts. Moderate force adaptation, stringent time constraints.", tasks: ["Unloading a dishwasher", "Force-based surface manipulation", "Object assembly in cluttered/dynamic environments"] },
      { levelIdx: 4, desc: "Full human manipulation ability. Any environment, diverse objects, exceptional adaptability. Precision, efficiency and robustness equivalent to skilled human.", tasks: ["Helping dress a person", "Search and rescue operations"] },
    ],
  },
  {
    id: "robotic-intelligence", name: "Robotic intelligence", icon: "Bot", color: "text-teal-600", currentLevel: 2,
    description: "Autonomous agent capability coordinating perception, movement, language, social interaction and problem solving. From simple repetitive tasks to full autonomous operation.",
    blocks: [
      { levelIdx: 0, desc: "Simple, repetitive tasks in highly structured, deterministic settings. Pre-specified instructions, no adaptive decisions, no human interaction.", tasks: ["Basic automated assembly in factories", "Robotic vacuum cleaners", "Object sorting in logistics"] },
      { levelIdx: 1, desc: "Predefined tasks in semi-structured environments with some variability. Low to moderate uncertainty, minimal human interaction.", tasks: ["Medical transport robots", "Material-handling robots in factories", "Agricultural robots for fruit picking"] },
      { levelIdx: 2, desc: "Medium-horizon multi-step tasks with some flexibility. Moderate variability, human collaboration, adaptation to dynamic changes.", tasks: ["Hospital robots handling transport and cleaning", "Robots assisting with furniture assembly", "Robot cinematographers filming based on learnt preferences"] },
      { levelIdx: 3, desc: "Multiple tasks with varying complexity. Adapts to dynamic conditions, understands limitations, uses feedback. Long-horizon complex objectives.", tasks: ["Cooking robots selecting ingredients by availability", "Autonomous wheelchairs navigating obstacles", "Autonomous aerial navigation near airports"] },
      { levelIdx: 4, desc: "Multiple complex tasks in unstructured settings with creative goal-setting. Advanced reasoning, common-sense, social intelligence, ethical decision making.", tasks: ["Home-assistance robots for people with disabilities", "Robots performing ethical decision making", "High-performance autonomous driving in diverse environments"] },
    ],
  },
];

const oecdDimensions: import("./framework-types").FrameworkDimension[] = OECD_DIMS.map((dim, i) => ({
  id: `oecd-${dim.id}`,
  name: dim.name,
  description: dim.description,
  order: i + 1,
  icon: dim.icon,
  color: dim.color,
  levels: dim.blocks.map((block) => {
    const lvl = OECD_LEVELS[block.levelIdx];
    return {
      id: `oecd-${dim.id}-${lvl.id}`,
      name: lvl.name,
      description: block.desc,
      order: lvl.order,
      target: lvl.target,
      indicators: block.tasks.map((task, ti) => ({
        id: `oecd-${dim.id}-${lvl.id}-t${ti + 1}`,
        description: task,
      })),
    };
  }),
}));

const oecdIndicators: Framework = {
  id: "oecd-indicators",
  name: "OECD AI Capability Indicators",
  shortName: "OECD AI Indicators",
  description: "Evidence-based framework mapping AI capabilities against nine human abilities across five progressive levels",
  type: "indicators",
  scope: "cross-cutting",
  source: "OECD",
  path: "/frameworks/oecd-indicators",
  icon: "TrendingUp",
  color: "text-rose-600",
  badgeLabel: "OECD Framework",
  targetAudience: ["leader", "educator", "admin", "policymaker", "researcher"],
  overview: `The OECD AI Capability Indicators provide policy makers with an evidence-based framework to understand AI capabilities and compare them to human abilities. Developed over five years by a collaboration of over 50 experts, the indicators cover nine human abilities from Language to Robotic Intelligence. Each indicator is presented as a five-level scale where the most challenging capabilities for AI systems are found towards the top. The indicators describe the progression of AI capabilities up to full human equivalence. Current state-of-the-art AI systems (as of November 2024) are rated at levels 2-3, with Language, Creativity, Knowledge/Learning/Memory and Vision at level 3, and all others at level 2.`,
  keyDimensions: oecdDimensions,
  keyPrinciples: [
    { id: "oecd-p1", name: "Understandable", description: "Communicating AI strengths and limitations in a straightforward manner" },
    { id: "oecd-p2", name: "Policy Relevant", description: "Offering insights into AI's impact on education, employment and the economy" },
    { id: "oecd-p3", name: "Comprehensive", description: "Covering all critical aspects of AI capabilities across nine human abilities" },
    { id: "oecd-p4", name: "Responsive", description: "Tracking AI progress over time through systematic updates" },
    { id: "oecd-p5", name: "Human-Grounded", description: "Grounded in human psychology, linking AI capabilities to human abilities for policy relevance" },
    { id: "oecd-p6", name: "Reliability-Based Rating", description: "To be ranked at a given level, an AI system must consistently and reliably possess most aspects of the capability described" },
    { id: "oecd-p7", name: "Stable & Informative", description: "Framework will remain stable and informative amid rapid AI progress until AI truly surpasses full human performance" },
  ],
  metadata: {
    isbn_print: "978-92-64-53190-1",
    isbn_pdf: "978-92-64-89309-2",
    isbn_html: "978-92-64-83602-0",
    license: "CC BY 4.0",
    source_url: "https://doi.org/10.1787/be745f04-en",
    version: "beta",
    year: 2025,
    publishingBody: "Centre for Educational Research and Innovation (CERI), AI and the Future of Skills (AIFS) project",
    ratings_as_of: "November 2024",
    totalDimensions: 9,
    totalLevels: 5,
    totalCompetencyBlocks: 45,
    totalTypicalTasks: 79,
    currentAiRatings: {
      language: 3, "social-interaction": 2, "problem-solving": 2, creativity: 3,
      metacognition: 2, "knowledge-learning-memory": 3, vision: 3, manipulation: 2, "robotic-intelligence": 2,
    },
  },
  useCases: [
    "Mapping AI progress against human occupational requirements via O*NET",
    "Gap analysis identifying occupations exposed to automation",
    "Identifying transformational AI opportunities in education delivery",
    "Informing curriculum design, teacher roles and student competencies",
    "Providing a framework for defining and measuring AGI",
    "Enabling values-based discussions about AI deployment across the economy",
  ],
  crossReferences: ["teacher-competency", "student-competency", "guidance-policy"],
  assessmentQuestions: [
    { id: "oecd-q1", dimension: "AI Capability Awareness", question: "How well do you understand current AI capability levels across different domains?", options: [
      { value: "oecd-a1", label: "I have limited understanding of what AI can and cannot do", level: "acquire" },
      { value: "oecd-a2", label: "I understand AI strengths in some domains but not others", level: "deepen" },
      { value: "oecd-a3", label: "I can accurately assess AI capabilities across multiple domains", level: "create" },
    ]},
    { id: "oecd-q2", dimension: "Human-AI Comparison", question: "How effectively can you compare AI capabilities to human abilities for specific tasks?", options: [
      { value: "oecd-b1", label: "I tend to over- or under-estimate what AI can do", level: "acquire" },
      { value: "oecd-b2", label: "I can make reasonable comparisons in familiar areas", level: "deepen" },
      { value: "oecd-b3", label: "I systematically evaluate AI vs human capabilities using evidence", level: "create" },
    ]},
    { id: "oecd-q3", dimension: "Policy Application", question: "How do you use knowledge of AI capabilities in decision-making?", options: [
      { value: "oecd-c1", label: "AI capability knowledge doesn't inform my decisions", level: "acquire" },
      { value: "oecd-c2", label: "I consider AI capabilities informally when planning", level: "deepen" },
      { value: "oecd-c3", label: "I use structured AI capability assessments to inform strategy", level: "create" },
    ]},
    { id: "oecd-q4", dimension: "Tracking AI Progress", question: "How do you stay informed about advances in AI capabilities?", options: [
      { value: "oecd-d1", label: "I rely on general news coverage", level: "acquire" },
      { value: "oecd-d2", label: "I follow specific AI research areas periodically", level: "deepen" },
      { value: "oecd-d3", label: "I systematically monitor AI progress using structured frameworks", level: "create" },
    ]},
    { id: "oecd-q5", dimension: "Societal Impact", question: "How well can you assess AI's potential impact on education and employment?", options: [
      { value: "oecd-e1", label: "I'm aware AI will have impacts but unsure of specifics", level: "acquire" },
      { value: "oecd-e2", label: "I can identify likely impacts in my own domain", level: "deepen" },
      { value: "oecd-e3", label: "I can map AI capabilities to occupational tasks and predict implications", level: "create" },
    ]},
  ],
  assessmentTitle: "AI Capability Awareness Assessment (OECD)",
  assessmentDescription: "Evaluate your understanding of AI capabilities across 9 human ability domains",
  showInQuiz: true,
  showInDashboard: true,
  showInLanding: false,
  isBackgroundFramework: true,
  compatibility: [],
  sourceFidelity: "official",
  estimatedAssessmentMinutes: 15,
  region: "international",
};

// ────────────────────────────────────────────────
// Master list & helpers
// ────────────────────────────────────────────────

const ORIGINAL_FRAMEWORKS: Framework[] = [
  guidancePolicy,
  teacherCompetency,
  studentCompetency,
  aiCapability,
  maturityTHE,
  maturityJISC,
  oecdIndicators,
];

export const FRAMEWORKS: Framework[] = [
  ...ORIGINAL_FRAMEWORKS,
  ...ADDITIONAL_FRAMEWORKS,
];

export function getFrameworkById(id: string): Framework | undefined {
  return FRAMEWORKS.find((f) => f.id === id);
}

export function getFrameworkByPath(path: string): Framework | undefined {
  return FRAMEWORKS.find((f) => f.path === path);
}

export function getDashboardFrameworks(): Framework[] {
  return FRAMEWORKS.filter((f) => f.showInDashboard);
}

export function getLandingFrameworks(): Framework[] {
  return FRAMEWORKS.filter((f) => f.showInLanding);
}

export function getQuizFrameworks(): Framework[] {
  return FRAMEWORKS.filter((f) => f.showInQuiz);
}

/** Get only top-level (non-child) dimensions for a framework */
export function getTopLevelDimensions(frameworkId: string): import("./framework-types").FrameworkDimension[] {
  const fw = getFrameworkById(frameworkId);
  if (!fw) return [];
  return fw.keyDimensions.filter((d) => !d.parentDimensionId);
}

/** Get child dimensions for a given parent */
export function getChildDimensions(frameworkId: string, parentId: string): import("./framework-types").FrameworkDimension[] {
  const fw = getFrameworkById(frameworkId);
  if (!fw) return [];
  return fw.keyDimensions.filter((d) => d.parentDimensionId === parentId);
}

/** Flatten all dimensions (including nested) into a flat list */
export function flattenDimensions(frameworkId: string): import("./framework-types").FrameworkDimension[] {
  const fw = getFrameworkById(frameworkId);
  if (!fw) return [];
  return fw.keyDimensions;
}

/** Get frameworks by scope */
export function getFrameworksByScope(scope: import("./framework-types").FrameworkScope): Framework[] {
  return FRAMEWORKS.filter((f) => f.scope === scope);
}

/** Get frameworks by audience */
export function getFrameworksByAudience(audience: string): Framework[] {
  return FRAMEWORKS.filter((f) => f.targetAudience.includes(audience));
}

/** Build a rich-text context block for LLM prompts. */
export function buildLLMContext(frameworkIds?: string[]): string {
  const list = frameworkIds
    ? FRAMEWORKS.filter((f) => frameworkIds.includes(f.id))
    : FRAMEWORKS.filter((f) => !f.isBackgroundFramework);

  return list
    .map(
      (f, i) => {
        const dimensionText = f.keyDimensions
          .filter((d) => !d.parentDimensionId)
          .map((d) => {
            const levelsText = d.levels
              .map((l) => {
                const indicatorText = l.indicators
                  .map((ind) => `      - ${ind.description}${ind.assessmentCriteria ? ` [Criteria: ${ind.assessmentCriteria}]` : ""}`)
                  .join("\n");
                return `    ${l.name} (${l.description}):\n${indicatorText}`;
              })
              .join("\n");
            return `  ${d.name}: ${d.description}\n${levelsText}`;
          })
          .join("\n");

        return `${i + 1}. ${f.name} (${f.source}, ${f.type}, scope: ${f.scope})
   Target: ${f.targetAudience.join(", ")}
   Overview: ${f.overview.substring(0, 200)}...
   Dimensions:
${dimensionText}
   Cross-references: ${f.crossReferences.join(", ")}`;
      }
    )
    .join("\n\n");
}
