// Shared framework context for edge functions (LLM prompts)
// Mirrors the hierarchical schema from src/data/frameworks.ts + frameworks-additional.ts
// Edge functions can't import from src/, so this is a standalone copy.
// Total: 22 frameworks (7 original + 15 additional)
// Last synced: 2026-03-07

interface Indicator {
  id: string;
  description: string;
  assessmentCriteria?: string;
}

interface Level {
  id: string;
  name: string;
  description: string;
  order: number;
  indicators: Indicator[];
}

interface Dimension {
  id: string;
  name: string;
  description: string;
  order: number;
  parentDimensionId?: string;
  levels: Level[];
}

interface FrameworkContext {
  id: string;
  name: string;
  source: string;
  type: string;
  scope: "individual" | "institutional" | "cross-cutting";
  targetAudience: string[];
  dimensions: Dimension[];
  overview: string;
  crossReferences: string[];
}

const frameworks: FrameworkContext[] = [
  // ═══════════════════════════════════════════════════
  // 1. UNESCO Guidance for AI in Education & Research
  // ═══════════════════════════════════════════════════
  {
    id: "guidance-policy",
    name: "UNESCO Guidance for AI in Education & Research",
    source: "UNESCO", type: "policy", scope: "institutional",
    targetAudience: ["leader", "admin", "educator"],
    overview: "UNESCO's first global policy framework addressing GenAI in education (2023). Provides actionable steps for governments and institutions to regulate AI ethically. Covers human-centered design, data governance, content validation, inclusive access, age-appropriate deployment, privacy safeguards, and transparent accountability.",
    crossReferences: ["teacher-competency", "ai-capability"],
    dimensions: [
      { id: "gp-human-centered", name: "Human-Centered AI", description: "Prioritize human agency, dignity, and wellbeing", order: 1, levels: [
        { id: "gp-hc-emerging", name: "Emerging", description: "Initial awareness", order: 1, indicators: [
          { id: "gp-hc-e1", description: "Recognise AI should serve human needs" },
          { id: "gp-hc-e2", description: "Identify stakeholders affected by AI decisions" },
        ]},
        { id: "gp-hc-developing", name: "Developing", description: "Actively applying principles", order: 2, indicators: [
          { id: "gp-hc-d1", description: "Conduct stakeholder impact assessments before AI deployment" },
        ]},
        { id: "gp-hc-established", name: "Established", description: "Embedded governance", order: 3, indicators: [
          { id: "gp-hc-s1", description: "Maintain ongoing human oversight of all AI systems" },
        ]},
      ]},
      { id: "gp-safe-equitable", name: "Safe & Equitable", description: "Safe, age-appropriate use with equitable access", order: 2, levels: [
        { id: "gp-se-emerging", name: "Emerging", description: "Basic safety awareness", order: 1, indicators: [{ id: "gp-se-e1", description: "Awareness of age-appropriateness concerns" }] },
        { id: "gp-se-developing", name: "Developing", description: "Implementing safety measures", order: 2, indicators: [{ id: "gp-se-d1", description: "Age-appropriate deployment guidelines in place" }] },
        { id: "gp-se-established", name: "Established", description: "Comprehensive safety", order: 3, indicators: [{ id: "gp-se-s1", description: "Robust content filtering and safety protocols" }] },
      ]},
      { id: "gp-ethics", name: "Ethics & Accountability", description: "Transparent governance with accountability", order: 3, levels: [
        { id: "gp-et-emerging", name: "Emerging", description: "Ethics awareness", order: 1, indicators: [{ id: "gp-et-e1", description: "Awareness of ethical implications" }] },
        { id: "gp-et-developing", name: "Developing", description: "Ethics integration", order: 2, indicators: [{ id: "gp-et-d1", description: "Ethics review for new AI tools" }] },
        { id: "gp-et-established", name: "Established", description: "Embedded ethics", order: 3, indicators: [{ id: "gp-et-s1", description: "Independent ethics committee reviewing AI" }] },
      ]},
      { id: "gp-evidence", name: "Evidence-Based", description: "Decisions grounded in research", order: 4, levels: [
        { id: "gp-ev-emerging", name: "Emerging", description: "Beginning evidence gathering", order: 1, indicators: [{ id: "gp-ev-e1", description: "Awareness of need for evidence" }] },
        { id: "gp-ev-developing", name: "Developing", description: "Systematic evaluation", order: 2, indicators: [{ id: "gp-ev-d1", description: "Pilot programmes with success metrics" }] },
        { id: "gp-ev-established", name: "Established", description: "Continuous evidence loop", order: 3, indicators: [{ id: "gp-ev-s1", description: "Longitudinal impact studies" }] },
      ]},
    ],
  },

  // ═══════════════════════════════════════════════════
  // 2. UNESCO Teacher AI Competency Framework
  // ═══════════════════════════════════════════════════
  {
    id: "teacher-competency",
    name: "UNESCO Teacher AI Competency Framework",
    source: "UNESCO", type: "competency", scope: "individual",
    targetAudience: ["educator"],
    overview: "Defines knowledge, skills, and values educators need in an AI-augmented teaching environment. 5 dimensions × 3 levels (Acquire, Deepen, Create) = 15 competencies. Covers human-centered mindset, ethics, AI foundations, pedagogy, and professional learning.",
    crossReferences: ["student-competency", "guidance-policy"],
    dimensions: [
      { id: "tc-human-centered", name: "Human-Centered Mindset", description: "Protect teacher rights, enhance agency", order: 1, levels: [
        { id: "tc-hc-acquire", name: "Acquire", description: "Foundational knowledge", order: 1, indicators: [{ id: "tc-hc-a1", description: "Recognise AI's role in supporting, not replacing, teachers" }] },
        { id: "tc-hc-deepen", name: "Deepen", description: "Applied skills", order: 2, indicators: [{ id: "tc-hc-d1", description: "Apply inclusive design principles when selecting AI tools" }] },
        { id: "tc-hc-create", name: "Create", description: "Innovation & leadership", order: 3, indicators: [{ id: "tc-hc-c1", description: "Lead institutional conversations on human-centered AI" }] },
      ]},
      { id: "tc-ethics", name: "Ethics of AI", description: "Navigate bias, privacy, accountability", order: 2, levels: [
        { id: "tc-et-acquire", name: "Acquire", description: "Awareness", order: 1, indicators: [{ id: "tc-et-a1", description: "Recognise common ethical issues" }] },
        { id: "tc-et-deepen", name: "Deepen", description: "Application", order: 2, indicators: [{ id: "tc-et-d1", description: "Implement bias detection checks" }] },
        { id: "tc-et-create", name: "Create", description: "Leadership", order: 3, indicators: [{ id: "tc-et-c1", description: "Develop ethical guidelines for AI use" }] },
      ]},
      { id: "tc-foundations", name: "AI Foundations", description: "Understand AI systems and limitations", order: 3, levels: [
        { id: "tc-fo-acquire", name: "Acquire", description: "Awareness", order: 1, indicators: [{ id: "tc-fo-a1", description: "Explain what AI is vs traditional software" }] },
        { id: "tc-fo-deepen", name: "Deepen", description: "Application", order: 2, indicators: [{ id: "tc-fo-d1", description: "Explain how LLMs work and their limitations" }] },
        { id: "tc-fo-create", name: "Create", description: "Leadership", order: 3, indicators: [{ id: "tc-fo-c1", description: "Design learning activities about AI principles" }] },
      ]},
      { id: "tc-pedagogy", name: "AI Pedagogy", description: "Design AI-enhanced lessons", order: 4, levels: [
        { id: "tc-pd-acquire", name: "Acquire", description: "Awareness", order: 1, indicators: [{ id: "tc-pd-a1", description: "Use AI for basic lesson preparation" }] },
        { id: "tc-pd-deepen", name: "Deepen", description: "Application", order: 2, indicators: [{ id: "tc-pd-d1", description: "Integrate AI into formative assessment" }] },
        { id: "tc-pd-create", name: "Create", description: "Leadership", order: 3, indicators: [{ id: "tc-pd-c1", description: "Create innovative AI-enhanced curricula" }] },
      ]},
      { id: "tc-professional", name: "Professional Learning", description: "Continuous development with AI", order: 5, levels: [
        { id: "tc-pl-acquire", name: "Acquire", description: "Awareness", order: 1, indicators: [{ id: "tc-pl-a1", description: "Attend AI training sessions" }] },
        { id: "tc-pl-deepen", name: "Deepen", description: "Application", order: 2, indicators: [{ id: "tc-pl-d1", description: "Actively share AI resources with peers" }] },
        { id: "tc-pl-create", name: "Create", description: "Leadership", order: 3, indicators: [{ id: "tc-pl-c1", description: "Facilitate AI professional development programmes" }] },
      ]},
    ],
  },

  // ═══════════════════════════════════════════════════
  // 3. UNESCO Student AI Competency Framework
  // ═══════════════════════════════════════════════════
  {
    id: "student-competency",
    name: "UNESCO Student AI Competency Framework",
    source: "UNESCO", type: "competency", scope: "individual",
    targetAudience: ["student"],
    overview: "Prepares learners across 4 dimensions with 3 levels (Foundational, Intermediate, Advanced). Covers AI use, understanding, ethics, and system design.",
    crossReferences: ["teacher-competency", "guidance-policy"],
    dimensions: [
      { id: "sc-use", name: "AI Use & Application", description: "Apply AI tools to real-world problems", order: 1, levels: [
        { id: "sc-use-found", name: "Foundational", description: "Guided exploration", order: 1, indicators: [{ id: "sc-use-f1", description: "Use AI tools with guidance" }] },
        { id: "sc-use-inter", name: "Intermediate", description: "Independent application", order: 2, indicators: [{ id: "sc-use-i1", description: "Independently select appropriate AI tools" }] },
        { id: "sc-use-adv", name: "Advanced", description: "Creation & leadership", order: 3, indicators: [{ id: "sc-use-a1", description: "Creatively combine multiple AI tools" }] },
      ]},
      { id: "sc-understanding", name: "AI Understanding", description: "How AI works and societal impact", order: 2, levels: [
        { id: "sc-und-found", name: "Foundational", description: "Basic awareness", order: 1, indicators: [{ id: "sc-und-f1", description: "Know AI exists in everyday products" }] },
        { id: "sc-und-inter", name: "Intermediate", description: "Critical analysis", order: 2, indicators: [{ id: "sc-und-i1", description: "Explain training data, bias, limitations" }] },
        { id: "sc-und-adv", name: "Advanced", description: "Advanced analysis", order: 3, indicators: [{ id: "sc-und-a1", description: "Critically analyse AI architectures" }] },
      ]},
      { id: "sc-ethics", name: "AI Ethics & Values", description: "Bias, privacy, responsible use", order: 3, levels: [
        { id: "sc-eth-found", name: "Foundational", description: "Awareness", order: 1, indicators: [{ id: "sc-eth-f1", description: "Aware AI can be biased" }] },
        { id: "sc-eth-inter", name: "Intermediate", description: "Application", order: 2, indicators: [{ id: "sc-eth-i1", description: "Apply ethical principles to AI choices" }] },
        { id: "sc-eth-adv", name: "Advanced", description: "Advocacy", order: 3, indicators: [{ id: "sc-eth-a1", description: "Advocate for responsible AI use" }] },
      ]},
      { id: "sc-design", name: "AI System Design", description: "Design and evaluate AI solutions", order: 4, levels: [
        { id: "sc-des-found", name: "Foundational", description: "Exploration", order: 1, indicators: [{ id: "sc-des-f1", description: "Understand AI systems are human-built" }] },
        { id: "sc-des-inter", name: "Intermediate", description: "Prototyping", order: 2, indicators: [{ id: "sc-des-i1", description: "Plan and prototype simple AI apps" }] },
        { id: "sc-des-adv", name: "Advanced", description: "Independent design", order: 3, indicators: [{ id: "sc-des-a1", description: "Design, test, and iterate AI solutions" }] },
      ]},
    ],
  },

  // ═══════════════════════════════════════════════════
  // 4. QS AI Capability Framework
  // ═══════════════════════════════════════════════════
  {
    id: "ai-capability",
    name: "QS AI Capability Framework",
    source: "QS", type: "capability", scope: "institutional",
    targetAudience: ["leader", "admin"],
    overview: "4 pillars, 14 categories, 30+ sub-indicators for institutional AI readiness. Maps governance, outreach, teaching, and research capabilities.",
    crossReferences: ["maturity-the", "maturity-jisc"],
    dimensions: [
      { id: "qs-governance", name: "Governance & Regulatory", description: "Standards, risk, privacy, leadership", order: 1, levels: [
        { id: "qs-gov-basic", name: "Basic", description: "Ad-hoc governance", order: 1, indicators: [{ id: "qs-gov-b1", description: "Awareness of regulatory requirements" }] },
        { id: "qs-gov-developing", name: "Developing", description: "Formal structures", order: 2, indicators: [{ id: "qs-gov-d1", description: "AI governance committee established" }] },
        { id: "qs-gov-advanced", name: "Advanced", description: "Comprehensive governance", order: 3, indicators: [{ id: "qs-gov-a1", description: "Integrated governance with regular reviews" }] },
      ]},
      { id: "qs-outreach", name: "Outreach & Commitment", description: "Recruitment, support, partnerships", order: 2, levels: [
        { id: "qs-out-basic", name: "Basic", description: "Limited AI engagement", order: 1, indicators: [{ id: "qs-out-b1", description: "Exploring AI for recruitment" }] },
        { id: "qs-out-developing", name: "Developing", description: "Active pilots", order: 2, indicators: [{ id: "qs-out-d1", description: "AI chatbots in student services" }] },
        { id: "qs-out-advanced", name: "Advanced", description: "Integrated AI engagement", order: 3, indicators: [{ id: "qs-out-a1", description: "Personalised AI across student lifecycle" }] },
      ]},
      { id: "qs-teaching", name: "Teaching & Learning", description: "Curriculum, personalisation, assessment", order: 3, levels: [
        { id: "qs-tl-basic", name: "Basic", description: "Individual experiments", order: 1, indicators: [{ id: "qs-tl-b1", description: "Some faculty using AI tools" }] },
        { id: "qs-tl-developing", name: "Developing", description: "Departmental integration", order: 2, indicators: [{ id: "qs-tl-d1", description: "AI-assisted learning in pilots" }] },
        { id: "qs-tl-advanced", name: "Advanced", description: "Institution-wide pedagogy", order: 3, indicators: [{ id: "qs-tl-a1", description: "AI curriculum standards adopted" }] },
      ]},
      { id: "qs-research", name: "Research & Innovation", description: "AI tools, scholarship, field research", order: 4, levels: [
        { id: "qs-res-basic", name: "Basic", description: "Individual adoption", order: 1, indicators: [{ id: "qs-res-b1", description: "Researchers use AI ad hoc" }] },
        { id: "qs-res-developing", name: "Developing", description: "Institutional support", order: 2, indicators: [{ id: "qs-res-d1", description: "Institutional AI research licences" }] },
        { id: "qs-res-advanced", name: "Advanced", description: "Strategic integration", order: 3, indicators: [{ id: "qs-res-a1", description: "AI integral to research strategy" }] },
      ]},
    ],
  },

  // ═══════════════════════════════════════════════════
  // 5. THE Digital Maturity Index
  // ═══════════════════════════════════════════════════
  // 4 pillars × 5 cross-cutting dimensions = 20 child dimensions, each with 4 maturity levels
  {
    id: "maturity-the",
    name: "THE Digital Maturity Index",
    source: "THE", type: "maturity", scope: "institutional",
    targetAudience: ["leader", "admin"],
    overview: "Based on 3,863 respondents from 1,949 institutions in 100 countries. 4 pillars (Teaching & Learning, Research, Professional Services, Planning & Governance) × 5 cross-cutting dimensions (Strategy, People & Culture, Technology, Data, Utilisation) × 4 maturity levels (Incidental→Intentional→Integrated→Optimised).",
    crossReferences: ["maturity-jisc", "ai-capability"],
    dimensions: [
      // ─── Pillar 1: Teaching & Learning ───
      { id: "the-tl-strategy", name: "Strategy (T&L)", description: "Teaching & Learning — Strategic digital transformation planning", order: 1, levels: [
        { id: "the-tl-strategy-incidental", name: "Incidental", description: "Sporadic, independent digital activities with no coordinated strategy", order: 1, indicators: [
          { id: "the-tl-str-1-01", description: "No formal digital transformation strategy exists for teaching and learning" },
          { id: "the-tl-str-1-02", description: "Technology decisions for T&L are made reactively without alignment to institutional goals" },
          { id: "the-tl-str-1-03", description: "IT budget allocation for teaching technology is ad hoc and unplanned" },
        ]},
        { id: "the-tl-strategy-intentional", name: "Intentional", description: "Purposeful digital activities with emerging strategy", order: 2, indicators: [
          { id: "the-tl-str-2-01", description: "An emerging digital strategy is being developed for teaching and learning" },
          { id: "the-tl-str-2-02", description: "Some technology investments are beginning to align with institutional teaching goals" },
          { id: "the-tl-str-2-03", description: "IT budget planning for T&L is becoming more structured and purposeful" },
        ]},
        { id: "the-tl-strategy-integrated", name: "Integrated", description: "Digital strategy fully integrated into T&L planning", order: 3, indicators: [
          { id: "the-tl-str-3-01", description: "Digital strategy is fully integrated into teaching and learning planning" },
          { id: "the-tl-str-3-02", description: "Technology investments for T&L are aligned with institutional KPIs and strategic goals" },
          { id: "the-tl-str-3-03", description: "Change management for digital teaching initiatives is embedded in institutional processes" },
        ]},
        { id: "the-tl-strategy-optimised", name: "Optimised", description: "Continuous improvement and sector leadership", order: 4, indicators: [
          { id: "the-tl-str-4-01", description: "Digital transformation is a core priority with continuous improvement cycles in teaching and learning" },
          { id: "the-tl-str-4-02", description: "Proactive technology forecasting informs T&L strategic planning" },
          { id: "the-tl-str-4-03", description: "The institution is recognized externally as a leader in digital teaching strategy" },
        ]},
      ]},
      { id: "the-tl-people", name: "People & Culture (T&L)", description: "Teaching & Learning — Digital skills, confidence, and organizational culture", order: 2, levels: [
        { id: "the-tl-people-incidental", name: "Incidental", description: "No established culture of exploring teaching technologies", order: 1, indicators: [
          { id: "the-tl-ppl-1-01", description: "There is no established culture of exploring or adopting new teaching technologies" },
          { id: "the-tl-ppl-1-02", description: "Teaching staff do not engage in ongoing professional development for digital skills" },
          { id: "the-tl-ppl-1-03", description: "Digital competence is not recognized in recruitment, evaluations, or promotions for educators" },
        ]},
        { id: "the-tl-people-intentional", name: "Intentional", description: "Emerging innovation culture for teaching", order: 2, indicators: [
          { id: "the-tl-ppl-2-01", description: "An innovative environment for exploring new teaching technologies is beginning to emerge" },
          { id: "the-tl-ppl-2-02", description: "Teaching staff are beginning to engage in professional development for digital skills" },
          { id: "the-tl-ppl-2-03", description: "Digital competence is beginning to be considered in HR processes for educators" },
        ]},
        { id: "the-tl-people-integrated", name: "Integrated", description: "Culture of innovation with regular digital CPD", order: 3, indicators: [
          { id: "the-tl-ppl-3-01", description: "A culture of innovation that explores and adopts new teaching technologies is established" },
          { id: "the-tl-ppl-3-02", description: "Teaching staff regularly engage in professional development to stay current with technology" },
          { id: "the-tl-ppl-3-03", description: "Digital competence is systematically recognized in recruitment, evaluations, and promotions" },
        ]},
        { id: "the-tl-people-optimised", name: "Optimised", description: "Innovation is a defining institutional characteristic", order: 4, indicators: [
          { id: "the-tl-ppl-4-01", description: "Innovation culture is a defining institutional characteristic with continuous exploration of emerging teaching technologies" },
          { id: "the-tl-ppl-4-02", description: "Professional development for digital teaching skills is embedded, continuous, and sector-leading" },
          { id: "the-tl-ppl-4-03", description: "Digital competence is a core requirement across all teaching roles and career pathways" },
        ]},
      ]},
      { id: "the-tl-technology", name: "Technology (T&L)", description: "Teaching & Learning — Infrastructure integration and emerging tech adoption", order: 3, levels: [
        { id: "the-tl-technology-incidental", name: "Incidental", description: "Unreliable infrastructure and limited platform availability", order: 1, indicators: [
          { id: "the-tl-tch-1-01", description: "Internet access on campus is unreliable or inconsistent for teaching activities" },
          { id: "the-tl-tch-1-02", description: "Emerging technologies such as AI and immersive tech are not explored for teaching" },
          { id: "the-tl-tch-1-03", description: "Learning management systems are not widely available or adopted for teaching" },
        ]},
        { id: "the-tl-technology-intentional", name: "Intentional", description: "Improving reliability with growing LMS adoption", order: 2, indicators: [
          { id: "the-tl-tch-2-01", description: "Internet access is increasingly reliable across most teaching areas" },
          { id: "the-tl-tch-2-02", description: "Some emerging technologies are being explored or piloted for teaching" },
          { id: "the-tl-tch-2-03", description: "LMS is increasingly adopted for course delivery and assessment" },
        ]},
        { id: "the-tl-technology-integrated", name: "Integrated", description: "Reliable infrastructure with integrated learning technologies", order: 3, indicators: [
          { id: "the-tl-tch-3-01", description: "Reliable internet access is provided consistently across the entire campus" },
          { id: "the-tl-tch-3-02", description: "Emerging technologies including AI and immersive tech are actively adopted for teaching" },
          { id: "the-tl-tch-3-03", description: "LMS and learning technologies are fully integrated into teaching, learning, and assessment" },
        ]},
        { id: "the-tl-technology-optimised", name: "Optimised", description: "Best-in-class infrastructure with sector-leading learning tech", order: 4, indicators: [
          { id: "the-tl-tch-4-01", description: "Campus-wide internet infrastructure is best-in-class and continuously optimized" },
          { id: "the-tl-tch-4-02", description: "The institution leads in the adoption and development of emerging teaching technologies" },
          { id: "the-tl-tch-4-03", description: "LMS and learning technologies are at the leading edge with sector-leading innovation" },
        ]},
      ]},
      { id: "the-tl-data", name: "Data (T&L)", description: "Teaching & Learning — Data analytics, integration, and evidence-based decision-making", order: 4, levels: [
        { id: "the-tl-data-incidental", name: "Incidental", description: "Inconsistent data collection with no learning analytics", order: 1, indicators: [
          { id: "the-tl-dat-1-01", description: "Data collection and storage for teaching are inconsistent or not digitized" },
          { id: "the-tl-dat-1-02", description: "Data systems for T&L are siloed with no integration across platforms" },
          { id: "the-tl-dat-1-03", description: "Learning analytics and student performance data are not collected or used" },
        ]},
        { id: "the-tl-data-intentional", name: "Intentional", description: "Emerging digital data collection with basic learning analytics", order: 2, indicators: [
          { id: "the-tl-dat-2-01", description: "Teaching data is increasingly collected and stored in digital formats" },
          { id: "the-tl-dat-2-02", description: "Initial efforts to integrate T&L data across systems are underway" },
          { id: "the-tl-dat-2-03", description: "Basic learning analytics are being explored for student support" },
        ]},
        { id: "the-tl-data-integrated", name: "Integrated", description: "Comprehensive data with analytics-informed teaching", order: 3, indicators: [
          { id: "the-tl-dat-3-01", description: "Teaching data is comprehensively collected and stored in well-managed digital systems" },
          { id: "the-tl-dat-3-02", description: "Data for T&L is effectively integrated across systems to maximize usage" },
          { id: "the-tl-dat-3-03", description: "Learning analytics inform personalized student support and course optimization" },
        ]},
        { id: "the-tl-data-optimised", name: "Optimised", description: "Fully automated data with predictive learning models", order: 4, indicators: [
          { id: "the-tl-dat-4-01", description: "Teaching data storage and management are fully automated and optimized" },
          { id: "the-tl-dat-4-02", description: "Data ecosystems for T&L are fully integrated with no silos between systems" },
          { id: "the-tl-dat-4-03", description: "Advanced learning analytics and predictive models drive personalized learning at scale" },
        ]},
      ]},
      { id: "the-tl-utilization", name: "Utilisation (T&L)", description: "Teaching & Learning — Adoption and effective use of digital tools", order: 5, levels: [
        { id: "the-tl-utilization-incidental", name: "Incidental", description: "Available teaching technologies underutilized", order: 1, indicators: [
          { id: "the-tl-utl-1-01", description: "Available teaching technologies are underutilized and not leveraged to improve pedagogy" },
          { id: "the-tl-utl-1-02", description: "There is a gap between available technology and its effective application in teaching" },
          { id: "the-tl-utl-1-03", description: "LMS is not utilized for blended, hybrid, or online learning options" },
        ]},
        { id: "the-tl-utilization-intentional", name: "Intentional", description: "Growing adoption with emerging blended learning", order: 2, indicators: [
          { id: "the-tl-utl-2-01", description: "Technology utilization in teaching is growing more purposeful with increasing adoption" },
          { id: "the-tl-utl-2-02", description: "Efforts are underway to close the gap between teaching technology availability and usage" },
          { id: "the-tl-utl-2-03", description: "LMS is increasingly utilized for blended and hybrid learning" },
        ]},
        { id: "the-tl-utilization-integrated", name: "Integrated", description: "Strategic utilization with diverse learning modalities", order: 3, indicators: [
          { id: "the-tl-utl-3-01", description: "Technology is strategically utilized across all teaching and learning activities" },
          { id: "the-tl-utl-3-02", description: "Technology systematically supports evidence-based teaching decisions" },
          { id: "the-tl-utl-3-03", description: "LMS is fully utilized for diverse learning modalities, providing flexibility and accommodating various learning styles" },
        ]},
        { id: "the-tl-utilization-optimised", name: "Optimised", description: "Maximized utilization with sector-leading flexible learning", order: 4, indicators: [
          { id: "the-tl-utl-4-01", description: "Technology utilization in teaching is maximized and continuously optimized" },
          { id: "the-tl-utl-4-02", description: "Data and technology comprehensively drive all teaching and learning decisions" },
          { id: "the-tl-utl-4-03", description: "Learning technologies are used innovatively to create sector-leading flexible learning experiences" },
        ]},
      ]},
      // ─── Pillar 2: Research ───
      { id: "the-re-strategy", name: "Strategy (Research)", description: "Research — Strategic digital transformation planning", order: 6, levels: [
        { id: "the-re-strategy-incidental", name: "Incidental", description: "No coordinated digital research strategy", order: 1, indicators: [
          { id: "the-re-str-1-01", description: "No formal digital transformation strategy exists for research" },
          { id: "the-re-str-1-02", description: "Technology decisions for research are made reactively without alignment to institutional goals" },
          { id: "the-re-str-1-03", description: "Change management processes for digital research initiatives are absent" },
        ]},
        { id: "the-re-strategy-intentional", name: "Intentional", description: "Emerging digital research strategy", order: 2, indicators: [
          { id: "the-re-str-2-01", description: "An emerging digital strategy is being developed for research" },
          { id: "the-re-str-2-02", description: "Some research technology investments are beginning to align with institutional goals" },
          { id: "the-re-str-2-03", description: "Initial change management processes are being developed for digital research initiatives" },
        ]},
        { id: "the-re-strategy-integrated", name: "Integrated", description: "Digital strategy fully integrated into research planning", order: 3, indicators: [
          { id: "the-re-str-3-01", description: "Digital strategy is fully integrated into research planning" },
          { id: "the-re-str-3-02", description: "Research technology investments are aligned with institutional KPIs and strategic goals" },
          { id: "the-re-str-3-03", description: "IT budget for research is coordinated across departments with clear funding strategy" },
        ]},
        { id: "the-re-strategy-optimised", name: "Optimised", description: "Continuous improvement and sector leadership in research strategy", order: 4, indicators: [
          { id: "the-re-str-4-01", description: "Digital transformation is a core priority with continuous improvement cycles in research" },
          { id: "the-re-str-4-02", description: "Digital research strategy is regularly benchmarked against global standards and peers" },
          { id: "the-re-str-4-03", description: "The institution is recognized externally as a leader in digital research strategy" },
        ]},
      ]},
      { id: "the-re-people", name: "People & Culture (Research)", description: "Research — Digital skills, confidence, and organizational culture", order: 7, levels: [
        { id: "the-re-people-incidental", name: "Incidental", description: "No culture of exploring research technologies", order: 1, indicators: [
          { id: "the-re-ppl-1-01", description: "There is no established culture of exploring or adopting new research technologies" },
          { id: "the-re-ppl-1-02", description: "Research staff do not engage in ongoing professional development for digital skills" },
          { id: "the-re-ppl-1-03", description: "Dedicated research technology support teams are absent or under-resourced" },
        ]},
        { id: "the-re-people-intentional", name: "Intentional", description: "Emerging innovation culture for research", order: 2, indicators: [
          { id: "the-re-ppl-2-01", description: "An innovative environment for exploring new research technologies is beginning to emerge" },
          { id: "the-re-ppl-2-02", description: "Research staff are beginning to engage in professional development for digital skills" },
          { id: "the-re-ppl-2-03", description: "A dedicated research technology team is being established or expanded" },
        ]},
        { id: "the-re-people-integrated", name: "Integrated", description: "Established innovation culture with regular CPD", order: 3, indicators: [
          { id: "the-re-ppl-3-01", description: "A culture of innovation that explores and adopts new research technologies is established" },
          { id: "the-re-ppl-3-02", description: "Research staff regularly engage in professional development to stay current with technology" },
          { id: "the-re-ppl-3-03", description: "A dedicated, mature technology team supports research staff across the institution" },
        ]},
        { id: "the-re-people-optimised", name: "Optimised", description: "Sector-leading digital research culture", order: 4, indicators: [
          { id: "the-re-ppl-4-01", description: "Innovation culture is a defining institutional characteristic with continuous exploration of emerging research technologies" },
          { id: "the-re-ppl-4-02", description: "Professional development for digital research skills is embedded, continuous, and sector-leading" },
          { id: "the-re-ppl-4-03", description: "Digital leadership in research is a defining feature of governance with leaders serving as sector exemplars" },
        ]},
      ]},
      { id: "the-re-technology", name: "Technology (Research)", description: "Research — Infrastructure integration and emerging tech adoption", order: 8, levels: [
        { id: "the-re-technology-incidental", name: "Incidental", description: "Limited research platforms and collaboration tools", order: 1, indicators: [
          { id: "the-re-tch-1-01", description: "Core digital platforms and tools for research are not widely available or adopted" },
          { id: "the-re-tch-1-02", description: "Emerging technologies such as AI are not explored for research" },
          { id: "the-re-tch-1-03", description: "Research collaboration platforms and data tools are limited or unavailable" },
        ]},
        { id: "the-re-technology-intentional", name: "Intentional", description: "Research platforms being adopted and expanded", order: 2, indicators: [
          { id: "the-re-tch-2-01", description: "Core digital platforms for research are adopted and in use across departments" },
          { id: "the-re-tch-2-02", description: "Some emerging technologies are being explored or piloted for research" },
          { id: "the-re-tch-2-03", description: "Research collaboration platforms are being adopted and expanded" },
        ]},
        { id: "the-re-technology-integrated", name: "Integrated", description: "Fully integrated research tools and collaboration platforms", order: 3, indicators: [
          { id: "the-re-tch-3-01", description: "Core digital platforms are fully integrated into research workflows" },
          { id: "the-re-tch-3-02", description: "Emerging technologies including AI are actively adopted for research" },
          { id: "the-re-tch-3-03", description: "Research tools and collaboration platforms are fully integrated into the research lifecycle" },
        ]},
        { id: "the-re-technology-optimised", name: "Optimised", description: "Cutting-edge, globally connected research infrastructure", order: 4, indicators: [
          { id: "the-re-tch-4-01", description: "Digital platforms for research are at the leading edge with continuous innovation" },
          { id: "the-re-tch-4-02", description: "The institution leads in the adoption and development of emerging research technologies" },
          { id: "the-re-tch-4-03", description: "Research technology infrastructure enables cutting-edge, globally connected research" },
        ]},
      ]},
      { id: "the-re-data", name: "Data (Research)", description: "Research — Data analytics, integration, and evidence-based decision-making", order: 9, levels: [
        { id: "the-re-data-incidental", name: "Incidental", description: "Inconsistent research data management", order: 1, indicators: [
          { id: "the-re-dat-1-01", description: "Research data collection and storage are inconsistent or not digitized" },
          { id: "the-re-dat-1-02", description: "Research data systems are siloed with no integration across platforms" },
          { id: "the-re-dat-1-03", description: "Research data management practices are absent or inconsistent" },
        ]},
        { id: "the-re-data-intentional", name: "Intentional", description: "Developing research data management practices", order: 2, indicators: [
          { id: "the-re-dat-2-01", description: "Research data is increasingly collected and stored in digital formats" },
          { id: "the-re-dat-2-02", description: "Initial efforts to integrate research data across systems are underway" },
          { id: "the-re-dat-2-03", description: "Research data management practices are being developed" },
        ]},
        { id: "the-re-data-integrated", name: "Integrated", description: "Well-managed research data with open data practices", order: 3, indicators: [
          { id: "the-re-dat-3-01", description: "Research data is comprehensively collected and stored in well-managed digital systems" },
          { id: "the-re-dat-3-02", description: "Predictive analytics are used for research decision-making" },
          { id: "the-re-dat-3-03", description: "Research data is well-managed with open data practices and cross-collaboration sharing" },
        ]},
        { id: "the-re-data-optimised", name: "Optimised", description: "Automated research data with comprehensive open data", order: 4, indicators: [
          { id: "the-re-dat-4-01", description: "Research data storage and management are fully automated and optimized" },
          { id: "the-re-dat-4-02", description: "Data ecosystems for research are fully integrated with no silos between systems" },
          { id: "the-re-dat-4-03", description: "Research data management is automated with comprehensive open data practices and global sharing" },
        ]},
      ]},
      { id: "the-re-utilization", name: "Utilisation (Research)", description: "Research — Adoption and effective use of digital tools", order: 10, levels: [
        { id: "the-re-utilization-incidental", name: "Incidental", description: "Digital tools not used for research collaboration", order: 1, indicators: [
          { id: "the-re-utl-1-01", description: "Available research technologies are underutilized and not leveraged to improve processes" },
          { id: "the-re-utl-1-02", description: "Technology is not employed to support evidence-based research decision-making" },
          { id: "the-re-utl-1-03", description: "Digital tools are not used for research collaboration or dissemination" },
        ]},
        { id: "the-re-utilization-intentional", name: "Intentional", description: "Researchers beginning to use digital networks", order: 2, indicators: [
          { id: "the-re-utl-2-01", description: "Research technology utilization is growing more purposeful with increasing adoption" },
          { id: "the-re-utl-2-02", description: "Technology is increasingly used to support some research decision-making processes" },
          { id: "the-re-utl-2-03", description: "Researchers are beginning to use digital networks for collaboration" },
        ]},
        { id: "the-re-utilization-integrated", name: "Integrated", description: "Effective digital collaboration and dissemination", order: 3, indicators: [
          { id: "the-re-utl-3-01", description: "Technology is strategically utilized across all research activities" },
          { id: "the-re-utl-3-02", description: "Technology systematically supports evidence-based research decision-making" },
          { id: "the-re-utl-3-03", description: "Digital networks are used effectively for research collaboration and dissemination" },
        ]},
        { id: "the-re-utilization-optimised", name: "Optimised", description: "Sector-leading digital scholarship and open research", order: 4, indicators: [
          { id: "the-re-utl-4-01", description: "Technology utilization in research is maximized and continuously optimized" },
          { id: "the-re-utl-4-02", description: "Data and technology comprehensively drive all research decisions" },
          { id: "the-re-utl-4-03", description: "Digital scholarship and open research practices are sector-leading" },
        ]},
      ]},
      // ─── Pillar 3: Professional Services ───
      { id: "the-ps-strategy", name: "Strategy (Prof Services)", description: "Professional Services — Strategic digital transformation planning", order: 11, levels: [
        { id: "the-ps-strategy-incidental", name: "Incidental", description: "No coordinated digital strategy for operations", order: 1, indicators: [
          { id: "the-ps-str-1-01", description: "No formal digital transformation strategy exists for professional services" },
          { id: "the-ps-str-1-02", description: "Technology decisions for operations are made reactively without alignment to institutional goals" },
          { id: "the-ps-str-1-03", description: "IT budget allocation for professional services is ad hoc and unplanned" },
        ]},
        { id: "the-ps-strategy-intentional", name: "Intentional", description: "Emerging digital strategy for operations", order: 2, indicators: [
          { id: "the-ps-str-2-01", description: "An emerging digital strategy is being developed for professional services" },
          { id: "the-ps-str-2-02", description: "Some operational technology investments are beginning to align with institutional goals" },
          { id: "the-ps-str-2-03", description: "Initial change management processes are being developed for digital service initiatives" },
        ]},
        { id: "the-ps-strategy-integrated", name: "Integrated", description: "Digital strategy fully integrated into operations planning", order: 3, indicators: [
          { id: "the-ps-str-3-01", description: "Digital strategy is fully integrated into professional services planning" },
          { id: "the-ps-str-3-02", description: "Technology investments for operations are aligned with institutional KPIs and strategic goals" },
          { id: "the-ps-str-3-03", description: "IT budget for professional services is coordinated across departments with clear funding strategy" },
        ]},
        { id: "the-ps-strategy-optimised", name: "Optimised", description: "Continuous improvement and sector leadership in operational strategy", order: 4, indicators: [
          { id: "the-ps-str-4-01", description: "Digital transformation is a core priority with continuous improvement cycles in professional services" },
          { id: "the-ps-str-4-02", description: "Proactive technology forecasting informs operational strategic planning" },
          { id: "the-ps-str-4-03", description: "The institution is recognized externally as a leader in digital operational strategy" },
        ]},
      ]},
      { id: "the-ps-people", name: "People & Culture (Prof Services)", description: "Professional Services — Digital skills, confidence, and organizational culture", order: 12, levels: [
        { id: "the-ps-people-incidental", name: "Incidental", description: "No culture of exploring operational technologies", order: 1, indicators: [
          { id: "the-ps-ppl-1-01", description: "There is no established culture of exploring or adopting new operational technologies" },
          { id: "the-ps-ppl-1-02", description: "Professional services staff do not engage in ongoing professional development for digital skills" },
          { id: "the-ps-ppl-1-03", description: "Digital leadership is not cultivated or modeled by operational decision-makers" },
        ]},
        { id: "the-ps-people-intentional", name: "Intentional", description: "Emerging innovation culture for operations", order: 2, indicators: [
          { id: "the-ps-ppl-2-01", description: "An innovative environment for exploring new operational technologies is beginning to emerge" },
          { id: "the-ps-ppl-2-02", description: "Professional services staff are beginning to engage in professional development for digital skills" },
          { id: "the-ps-ppl-2-03", description: "Some operational leaders are beginning to model and promote digital behaviors" },
        ]},
        { id: "the-ps-people-integrated", name: "Integrated", description: "Established innovation culture with mature support teams", order: 3, indicators: [
          { id: "the-ps-ppl-3-01", description: "A culture of innovation that explores and adopts new operational technologies is established" },
          { id: "the-ps-ppl-3-02", description: "Professional services staff regularly engage in professional development to stay current with technology" },
          { id: "the-ps-ppl-3-03", description: "A dedicated, mature technology team supports staff across professional services" },
        ]},
        { id: "the-ps-people-optimised", name: "Optimised", description: "Sector-leading digital operational culture", order: 4, indicators: [
          { id: "the-ps-ppl-4-01", description: "Innovation culture is a defining institutional characteristic with continuous exploration of emerging operational technologies" },
          { id: "the-ps-ppl-4-02", description: "Professional development for digital operational skills is embedded, continuous, and sector-leading" },
          { id: "the-ps-ppl-4-03", description: "Digital competence is a core requirement across all professional services roles and career pathways" },
        ]},
      ]},
      { id: "the-ps-technology", name: "Technology (Prof Services)", description: "Professional Services — Infrastructure integration and emerging tech adoption", order: 13, levels: [
        { id: "the-ps-technology-incidental", name: "Incidental", description: "Basic and fragmented administrative platforms", order: 1, indicators: [
          { id: "the-ps-tch-1-01", description: "Core digital platforms and tools for operations are not widely available or adopted" },
          { id: "the-ps-tch-1-02", description: "Network infrastructure lacks flexibility and secure design for services" },
          { id: "the-ps-tch-1-03", description: "Administrative and IT service platforms are basic and fragmented" },
        ]},
        { id: "the-ps-technology-intentional", name: "Intentional", description: "Administrative platforms becoming more integrated", order: 2, indicators: [
          { id: "the-ps-tch-2-01", description: "Core digital platforms for operations are adopted and in use across departments" },
          { id: "the-ps-tch-2-02", description: "Network infrastructure improvements are underway with growing attention to security" },
          { id: "the-ps-tch-2-03", description: "Administrative platforms are becoming more integrated and user-friendly" },
        ]},
        { id: "the-ps-technology-integrated", name: "Integrated", description: "Effective operational platforms with secure infrastructure", order: 3, indicators: [
          { id: "the-ps-tch-3-01", description: "Core digital platforms are fully integrated into operational workflows" },
          { id: "the-ps-tch-3-02", description: "A flexible, secure network infrastructure is established and maintained" },
          { id: "the-ps-tch-3-03", description: "Administrative and IT platforms effectively support all operational services" },
        ]},
        { id: "the-ps-technology-optimised", name: "Optimised", description: "Fully automated and optimized operational technology", order: 4, indicators: [
          { id: "the-ps-tch-4-01", description: "Digital platforms for operations are at the leading edge with continuous innovation" },
          { id: "the-ps-tch-4-02", description: "Technology infrastructure is fully automated, secure, and serves as a sector benchmark" },
          { id: "the-ps-tch-4-03", description: "Operational technology is fully automated and optimized for efficiency" },
        ]},
      ]},
      { id: "the-ps-data", name: "Data (Prof Services)", description: "Professional Services — Data analytics, integration, and evidence-based decision-making", order: 14, levels: [
        { id: "the-ps-data-incidental", name: "Incidental", description: "Fragmented operational data across systems", order: 1, indicators: [
          { id: "the-ps-dat-1-01", description: "Operational data collection and storage are inconsistent or not digitized" },
          { id: "the-ps-dat-1-02", description: "Data security practices for professional services are weak or absent" },
          { id: "the-ps-dat-1-03", description: "Operational and administrative data are fragmented across systems" },
        ]},
        { id: "the-ps-data-intentional", name: "Intentional", description: "Operational data systems being consolidated", order: 2, indicators: [
          { id: "the-ps-dat-2-01", description: "Operational data is increasingly collected and stored in digital formats" },
          { id: "the-ps-dat-2-02", description: "Data security awareness is growing with initial measures in place for services" },
          { id: "the-ps-dat-2-03", description: "Operational data systems are beginning to be consolidated" },
        ]},
        { id: "the-ps-data-integrated", name: "Integrated", description: "Integrated data across HR, finance, and administration", order: 3, indicators: [
          { id: "the-ps-dat-3-01", description: "Operational data is comprehensively collected and stored in well-managed digital systems" },
          { id: "the-ps-dat-3-02", description: "Data security is managed through comprehensive policies and technology for services" },
          { id: "the-ps-dat-3-03", description: "Operational data is integrated across HR, finance, and student administration" },
        ]},
        { id: "the-ps-data-optimised", name: "Optimised", description: "Fully automated, predictive institutional management", order: 4, indicators: [
          { id: "the-ps-dat-4-01", description: "Operational data storage and management are fully automated and optimized" },
          { id: "the-ps-dat-4-02", description: "Data governance and cybersecurity practices for services are exemplary and continuously improved" },
          { id: "the-ps-dat-4-03", description: "Operational data drives fully automated, predictive institutional management" },
        ]},
      ]},
      { id: "the-ps-utilization", name: "Utilisation (Prof Services)", description: "Professional Services — Adoption and effective use of digital tools", order: 15, levels: [
        { id: "the-ps-utilization-incidental", name: "Incidental", description: "Administrative platforms underutilized", order: 1, indicators: [
          { id: "the-ps-utl-1-01", description: "Available operational technologies are underutilized and not leveraged to improve processes" },
          { id: "the-ps-utl-1-02", description: "Digital tools for operational collaboration and communication are not effectively used" },
          { id: "the-ps-utl-1-03", description: "Administrative platforms are underutilized across operations" },
        ]},
        { id: "the-ps-utilization-intentional", name: "Intentional", description: "Growing adoption with increasing IT support", order: 2, indicators: [
          { id: "the-ps-utl-2-01", description: "Operational technology utilization is growing more purposeful with increasing adoption" },
          { id: "the-ps-utl-2-02", description: "Digital tools are beginning to improve operational collaboration and communication" },
          { id: "the-ps-utl-2-03", description: "Administrative platforms are increasingly adopted with growing IT support" },
        ]},
        { id: "the-ps-utilization-integrated", name: "Integrated", description: "Effective platforms with robust IT support", order: 3, indicators: [
          { id: "the-ps-utl-3-01", description: "Technology is strategically utilized across all professional services" },
          { id: "the-ps-utl-3-02", description: "Digital tools effectively enhance operational collaboration and break down silos" },
          { id: "the-ps-utl-3-03", description: "Administrative platforms are effectively used with robust IT support ensuring smooth operation" },
        ]},
        { id: "the-ps-utilization-optimised", name: "Optimised", description: "Fully optimized operational technology as sector benchmark", order: 4, indicators: [
          { id: "the-ps-utl-4-01", description: "Technology utilization in professional services is maximized and continuously optimized" },
          { id: "the-ps-utl-4-02", description: "Data and technology comprehensively drive all operational decisions" },
          { id: "the-ps-utl-4-03", description: "Operational technology utilization is fully optimized and serves as a sector benchmark" },
        ]},
      ]},
      // ─── Pillar 4: Planning & Governance ───
      { id: "the-pg-strategy", name: "Strategy (Planning & Gov)", description: "Planning & Governance — Strategic digital transformation planning", order: 16, levels: [
        { id: "the-pg-strategy-incidental", name: "Incidental", description: "No coordinated digital governance strategy", order: 1, indicators: [
          { id: "the-pg-str-1-01", description: "No formal digital transformation strategy exists for institutional governance" },
          { id: "the-pg-str-1-02", description: "Technology decisions for governance are made reactively without alignment to institutional goals" },
          { id: "the-pg-str-1-03", description: "Change management processes for digital governance initiatives are absent" },
        ]},
        { id: "the-pg-strategy-intentional", name: "Intentional", description: "Emerging digital governance strategy", order: 2, indicators: [
          { id: "the-pg-str-2-01", description: "An emerging digital strategy is being developed for institutional governance" },
          { id: "the-pg-str-2-02", description: "Some governance technology investments are beginning to align with institutional goals" },
          { id: "the-pg-str-2-03", description: "IT budget planning for governance is becoming more structured and purposeful" },
        ]},
        { id: "the-pg-strategy-integrated", name: "Integrated", description: "Digital strategy fully integrated into governance planning", order: 3, indicators: [
          { id: "the-pg-str-3-01", description: "Digital strategy is fully integrated into institutional governance and planning" },
          { id: "the-pg-str-3-02", description: "Technology investments for governance are aligned with institutional KPIs and strategic goals" },
          { id: "the-pg-str-3-03", description: "Change management for digital governance initiatives is embedded in institutional processes" },
        ]},
        { id: "the-pg-strategy-optimised", name: "Optimised", description: "Continuous improvement and sector leadership in governance strategy", order: 4, indicators: [
          { id: "the-pg-str-4-01", description: "Digital transformation is a core priority with continuous improvement cycles in governance" },
          { id: "the-pg-str-4-02", description: "Digital governance strategy is regularly benchmarked against global standards and peers" },
          { id: "the-pg-str-4-03", description: "The institution is recognized externally as a leader in digital governance strategy" },
        ]},
      ]},
      { id: "the-pg-people", name: "People & Culture (Planning & Gov)", description: "Planning & Governance — Digital skills, confidence, and organizational culture", order: 17, levels: [
        { id: "the-pg-people-incidental", name: "Incidental", description: "No culture of digital governance leadership", order: 1, indicators: [
          { id: "the-pg-ppl-1-01", description: "There is no established culture of exploring or adopting new governance technologies" },
          { id: "the-pg-ppl-1-02", description: "Governance staff do not engage in ongoing professional development for digital skills" },
          { id: "the-pg-ppl-1-03", description: "Digital leadership is not cultivated or modeled by governance decision-makers" },
        ]},
        { id: "the-pg-people-intentional", name: "Intentional", description: "Emerging digital governance leadership", order: 2, indicators: [
          { id: "the-pg-ppl-2-01", description: "An innovative environment for exploring new governance technologies is beginning to emerge" },
          { id: "the-pg-ppl-2-02", description: "Governance staff are beginning to engage in professional development for digital skills" },
          { id: "the-pg-ppl-2-03", description: "Some governance leaders are beginning to model and promote digital behaviors" },
        ]},
        { id: "the-pg-people-integrated", name: "Integrated", description: "Governance leaders cultivate digital leadership through self-assessment", order: 3, indicators: [
          { id: "the-pg-ppl-3-01", description: "A culture of innovation that explores and adopts new governance technologies is established" },
          { id: "the-pg-ppl-3-02", description: "Governance staff regularly engage in professional development to stay current with technology" },
          { id: "the-pg-ppl-3-03", description: "Digital leadership is cultivated through self-assessment and modeling by governance decision-makers" },
        ]},
        { id: "the-pg-people-optimised", name: "Optimised", description: "Sector-leading digital governance leadership", order: 4, indicators: [
          { id: "the-pg-ppl-4-01", description: "Innovation culture is a defining institutional characteristic with continuous exploration of emerging governance technologies" },
          { id: "the-pg-ppl-4-02", description: "Professional development for digital governance skills is embedded, continuous, and sector-leading" },
          { id: "the-pg-ppl-4-03", description: "Digital leadership in governance is a defining feature with leaders serving as sector exemplars" },
        ]},
      ]},
      { id: "the-pg-technology", name: "Technology (Planning & Gov)", description: "Planning & Governance — Infrastructure integration and emerging tech adoption", order: 18, levels: [
        { id: "the-pg-technology-incidental", name: "Incidental", description: "Absent or rudimentary governance systems", order: 1, indicators: [
          { id: "the-pg-tch-1-01", description: "Core digital platforms and tools for governance are not widely available or adopted" },
          { id: "the-pg-tch-1-02", description: "Emerging technologies such as AI are not explored for governance" },
          { id: "the-pg-tch-1-03", description: "Governance and decision-support systems are absent or rudimentary" },
        ]},
        { id: "the-pg-technology-intentional", name: "Intentional", description: "Governance dashboards and planning tools being introduced", order: 2, indicators: [
          { id: "the-pg-tch-2-01", description: "Core digital platforms for governance are adopted and in use across departments" },
          { id: "the-pg-tch-2-02", description: "Some emerging technologies are being explored or piloted for governance" },
          { id: "the-pg-tch-2-03", description: "Governance dashboards and planning tools are being introduced" },
        ]},
        { id: "the-pg-technology-integrated", name: "Integrated", description: "Enterprise-level governance tools supporting strategic decisions", order: 3, indicators: [
          { id: "the-pg-tch-3-01", description: "Core digital platforms are fully integrated into governance workflows" },
          { id: "the-pg-tch-3-02", description: "Emerging technologies including AI are actively adopted for governance and planning" },
          { id: "the-pg-tch-3-03", description: "Enterprise-level governance tools and dashboards support strategic decision-making" },
        ]},
        { id: "the-pg-technology-optimised", name: "Optimised", description: "Real-time, comprehensive institutional intelligence", order: 4, indicators: [
          { id: "the-pg-tch-4-01", description: "Digital platforms for governance are at the leading edge with continuous innovation" },
          { id: "the-pg-tch-4-02", description: "The institution leads in the adoption and development of emerging governance technologies" },
          { id: "the-pg-tch-4-03", description: "Governance technology provides real-time, comprehensive institutional intelligence" },
        ]},
      ]},
      { id: "the-pg-data", name: "Data (Planning & Gov)", description: "Planning & Governance — Data analytics, integration, and evidence-based decision-making", order: 19, levels: [
        { id: "the-pg-data-incidental", name: "Incidental", description: "Institutional performance data not used for planning", order: 1, indicators: [
          { id: "the-pg-dat-1-01", description: "Data collection and storage for governance are inconsistent or not digitized" },
          { id: "the-pg-dat-1-02", description: "Predictive analytics are not used for institutional planning" },
          { id: "the-pg-dat-1-03", description: "Institutional performance data is not used for strategic planning" },
        ]},
        { id: "the-pg-data-intentional", name: "Intentional", description: "Institutional data dashboards being developed", order: 2, indicators: [
          { id: "the-pg-dat-2-01", description: "Governance data is increasingly collected and stored in digital formats" },
          { id: "the-pg-dat-2-02", description: "Basic predictive analytics are being explored for institutional planning" },
          { id: "the-pg-dat-2-03", description: "Institutional data dashboards are being developed for strategic use" },
        ]},
        { id: "the-pg-data-integrated", name: "Integrated", description: "Comprehensive evidence-based governance and planning", order: 3, indicators: [
          { id: "the-pg-dat-3-01", description: "Governance data is comprehensively collected and stored in well-managed digital systems" },
          { id: "the-pg-dat-3-02", description: "Predictive analytics are used for governance decision-making" },
          { id: "the-pg-dat-3-03", description: "Institutional data comprehensively supports evidence-based governance and strategic planning" },
        ]},
        { id: "the-pg-data-optimised", name: "Optimised", description: "Fully automated institutional intelligence with real-time dashboards", order: 4, indicators: [
          { id: "the-pg-dat-4-01", description: "Governance data storage and management are fully automated and optimized" },
          { id: "the-pg-dat-4-02", description: "Predictive modeling comprehensively drives governance strategy and operations" },
          { id: "the-pg-dat-4-03", description: "Institutional intelligence is fully automated with real-time, comprehensive governance dashboards" },
        ]},
      ]},
      { id: "the-pg-utilization", name: "Utilisation (Planning & Gov)", description: "Planning & Governance — Adoption and effective use of digital tools", order: 20, levels: [
        { id: "the-pg-utilization-incidental", name: "Incidental", description: "Communication tools do not break down silos", order: 1, indicators: [
          { id: "the-pg-utl-1-01", description: "Available governance technologies are underutilized and not leveraged to improve processes" },
          { id: "the-pg-utl-1-02", description: "Technology is not employed to support evidence-based governance decision-making" },
          { id: "the-pg-utl-1-03", description: "Communication tools do not break down organizational silos" },
        ]},
        { id: "the-pg-utilization-intentional", name: "Intentional", description: "Communication tools beginning to improve coordination", order: 2, indicators: [
          { id: "the-pg-utl-2-01", description: "Governance technology utilization is growing more purposeful with increasing adoption" },
          { id: "the-pg-utl-2-02", description: "Technology is increasingly used to support some governance decision-making processes" },
          { id: "the-pg-utl-2-03", description: "Communication tools are beginning to improve cross-department coordination" },
        ]},
        { id: "the-pg-utilization-integrated", name: "Integrated", description: "Effective cross-functional governance with digital tools", order: 3, indicators: [
          { id: "the-pg-utl-3-01", description: "Technology is strategically utilized across all governance activities" },
          { id: "the-pg-utl-3-02", description: "Available technology is well-matched to governance needs and fully leveraged" },
          { id: "the-pg-utl-3-03", description: "Communication tools effectively break down silos and support cross-functional governance" },
        ]},
        { id: "the-pg-utilization-optimised", name: "Optimised", description: "Fully data-driven governance with sector-leading utilization", order: 4, indicators: [
          { id: "the-pg-utl-4-01", description: "Technology utilization in governance is maximized and continuously optimized" },
          { id: "the-pg-utl-4-02", description: "Collaboration is seamless with no communication barriers across governance" },
          { id: "the-pg-utl-4-03", description: "Governance processes are fully data-driven with sector-leading strategic technology utilization" },
        ]},
      ]},
    ],
  },

  // ═══════════════════════════════════════════════════
  // 6. JISC Digital Maturity Model
  // ═══════════════════════════════════════════════════
  {
    id: "maturity-jisc",
    name: "JISC Digital Maturity Model",
    source: "JISC", type: "maturity", scope: "institutional",
    targetAudience: ["leader", "admin"],
    overview: "UK HE digital maturity assessment. 3 levels × 5 components. Common language for sector-wide transformation.",
    crossReferences: ["maturity-the", "ai-capability"],
    dimensions: [
      { id: "jisc-culture", name: "Organisational Digital Culture", description: "Digital thinking in institutional identity", order: 1, levels: [
        { id: "jisc-cult-emerging", name: "Emerging to Established", description: "Ad-hoc adoption", order: 1, indicators: [{ id: "jisc-cult-e1", description: "Lack of strategic digital leadership" }] },
        { id: "jisc-cult-enhanced", name: "Established to Enhanced", description: "Proactive strategy", order: 2, indicators: [{ id: "jisc-cult-d1", description: "Developing proactive digital approach" }] },
        { id: "jisc-cult-mature", name: "Enhanced to Mature", description: "Integral to identity", order: 3, indicators: [{ id: "jisc-cult-m1", description: "Comprehensive integrated strategies" }] },
      ]},
      { id: "jisc-innovation", name: "Knowledge Creation & Innovation", description: "Support for innovation", order: 2, levels: [
        { id: "jisc-inno-emerging", name: "Emerging to Established", description: "Despite limited support", order: 1, indicators: [{ id: "jisc-inno-e1", description: "Innovation despite limited support" }] },
        { id: "jisc-inno-enhanced", name: "Established to Enhanced", description: "Funding emerging", order: 2, indicators: [{ id: "jisc-inno-d1", description: "Innovation funding structures" }] },
        { id: "jisc-inno-mature", name: "Enhanced to Mature", description: "Systematic support", order: 3, indicators: [{ id: "jisc-inno-m1", description: "Clear innovation pathways" }] },
      ]},
      { id: "jisc-development", name: "Knowledge Development", description: "Digital skills programmes", order: 3, levels: [
        { id: "jisc-dev-emerging", name: "Emerging to Established", description: "Ad-hoc training", order: 1, indicators: [{ id: "jisc-dev-e1", description: "Dispersed skills training" }] },
        { id: "jisc-dev-enhanced", name: "Established to Enhanced", description: "Structured training", order: 2, indicators: [{ id: "jisc-dev-d1", description: "Structured but not role-linked" }] },
        { id: "jisc-dev-mature", name: "Enhanced to Mature", description: "Role-specific", order: 3, indicators: [{ id: "jisc-dev-m1", description: "Progressive role-specific development" }] },
      ]},
      { id: "jisc-management", name: "Knowledge Management & Use", description: "Managing digital knowledge", order: 4, levels: [
        { id: "jisc-mgmt-emerging", name: "Emerging to Established", description: "Siloed", order: 1, indicators: [{ id: "jisc-mgmt-e1", description: "Knowledge siloed in departments" }] },
        { id: "jisc-mgmt-enhanced", name: "Established to Enhanced", description: "Cross-team sharing", order: 2, indicators: [{ id: "jisc-mgmt-d1", description: "Some cross-team sharing" }] },
        { id: "jisc-mgmt-mature", name: "Enhanced to Mature", description: "Comprehensive systems", order: 3, indicators: [{ id: "jisc-mgmt-m1", description: "Clear knowledge management systems" }] },
      ]},
      { id: "jisc-exchange", name: "Knowledge Exchange & Partnerships", description: "External collaboration", order: 5, levels: [
        { id: "jisc-exch-emerging", name: "Emerging to Established", description: "Limited", order: 1, indicators: [{ id: "jisc-exch-e1", description: "Limited external collaboration" }] },
        { id: "jisc-exch-enhanced", name: "Established to Enhanced", description: "Some partnerships", order: 2, indicators: [{ id: "jisc-exch-d1", description: "Non-strategic partnerships" }] },
        { id: "jisc-exch-mature", name: "Enhanced to Mature", description: "Strategic", order: 3, indicators: [{ id: "jisc-exch-m1", description: "Strategic partnerships driving transformation" }] },
      ]},
    ],
  },

  // ═══════════════════════════════════════════════════
  // 7. OECD AI Capability Indicators
  // ═══════════════════════════════════════════════════
  {
    id: "oecd-indicators",
    name: "OECD AI Capability Indicators",
    source: "OECD", type: "indicators", scope: "cross-cutting",
    targetAudience: ["leader", "educator", "admin"],
    overview: "Assess task characteristics and AI readiness. 4 task types with augment-vs-automate decisions and safeguard requirements.",
    crossReferences: ["ai-capability", "guidance-policy"],
    dimensions: [
      { id: "oecd-routine", name: "Routine Cognitive", description: "Data entry, reports, scheduling — high readiness", order: 1, levels: [
        { id: "oecd-rc-high", name: "High AI Readiness", description: "Largely automatable", order: 1, indicators: [
          { id: "oecd-rc-h1", description: "Data entry and formatting" },
          { id: "oecd-rc-h2", description: "Report generation from structured data" },
          { id: "oecd-rc-h3", description: "Schedule management" },
        ]},
      ]},
      { id: "oecd-nonroutine", name: "Non-Routine Cognitive", description: "Research, policy, strategy — medium readiness", order: 2, levels: [
        { id: "oecd-nrc-medium", name: "Medium AI Readiness", description: "AI augmentation with oversight", order: 1, indicators: [
          { id: "oecd-nrc-m1", description: "Research design and literature review" },
          { id: "oecd-nrc-m2", description: "Policy development and analysis" },
        ]},
      ]},
      { id: "oecd-interpersonal", name: "Interpersonal", description: "Counselling, collaboration, mentorship — low readiness", order: 3, levels: [
        { id: "oecd-ip-low", name: "Low AI Readiness", description: "Requires human connection", order: 1, indicators: [
          { id: "oecd-ip-l1", description: "Student counselling and pastoral care" },
          { id: "oecd-ip-l2", description: "Team collaboration and conflict resolution" },
        ]},
      ]},
      { id: "oecd-manual", name: "Manual/Physical", description: "Lab work, equipment, fieldwork — variable readiness", order: 4, levels: [
        { id: "oecd-mp-variable", name: "Variable AI Readiness", description: "Context-dependent", order: 1, indicators: [
          { id: "oecd-mp-v1", description: "Laboratory experiments" },
          { id: "oecd-mp-v2", description: "Field studies and site visits" },
        ]},
      ]},
    ],
  },

  // ═══════════════════════════════════════════════════
  // 8. AILit Framework
  // ═══════════════════════════════════════════════════
  {
    id: "ailit",
    name: "AI Literacy Framework (AILit)",
    source: "AILit Consortium", type: "competency", scope: "individual",
    targetAudience: ["educator", "student"],
    overview: "4 domains with 22 competences across 4 proficiency levels (Novice→Expert). Covers AI Fundamentals, Application, Ethics, and AI Futures. Designed for cross-disciplinary AI literacy education.",
    crossReferences: ["teacher-competency", "student-competency", "digcomp"],
    dimensions: [
      { id: "ailit-fundamentals", name: "AI Fundamentals", description: "Core knowledge of AI concepts, data, and algorithms", order: 1, levels: [
        { id: "ailit-fund-novice", name: "Novice", description: "Basic awareness", order: 1, indicators: [{ id: "ailit-fund-n1", description: "Define AI, ML, and deep learning concepts" }] },
        { id: "ailit-fund-intermediate", name: "Intermediate", description: "Applied understanding", order: 2, indicators: [{ id: "ailit-fund-i1", description: "Explain how training data shapes model outputs" }] },
        { id: "ailit-fund-advanced", name: "Advanced", description: "Critical analysis", order: 3, indicators: [{ id: "ailit-fund-a1", description: "Evaluate model architectures and their trade-offs" }] },
        { id: "ailit-fund-expert", name: "Expert", description: "Innovation & leadership", order: 4, indicators: [{ id: "ailit-fund-e1", description: "Design novel AI solutions for domain-specific problems" }] },
      ]},
      { id: "ailit-application", name: "AI Application", description: "Using AI tools effectively in professional contexts", order: 2, levels: [
        { id: "ailit-app-novice", name: "Novice", description: "Guided use", order: 1, indicators: [{ id: "ailit-app-n1", description: "Use common AI tools with guidance" }] },
        { id: "ailit-app-intermediate", name: "Intermediate", description: "Independent use", order: 2, indicators: [{ id: "ailit-app-i1", description: "Select appropriate AI tools for specific tasks" }] },
        { id: "ailit-app-advanced", name: "Advanced", description: "Strategic integration", order: 3, indicators: [{ id: "ailit-app-a1", description: "Integrate AI tools into complex workflows" }] },
        { id: "ailit-app-expert", name: "Expert", description: "Transformation", order: 4, indicators: [{ id: "ailit-app-e1", description: "Transform practices through innovative AI application" }] },
      ]},
      { id: "ailit-ethics", name: "AI Ethics & Society", description: "Navigating ethical implications of AI", order: 3, levels: [
        { id: "ailit-eth-novice", name: "Novice", description: "Awareness", order: 1, indicators: [{ id: "ailit-eth-n1", description: "Recognise AI raises ethical questions" }] },
        { id: "ailit-eth-intermediate", name: "Intermediate", description: "Application", order: 2, indicators: [{ id: "ailit-eth-i1", description: "Apply ethical frameworks to AI decisions" }] },
        { id: "ailit-eth-advanced", name: "Advanced", description: "Advocacy", order: 3, indicators: [{ id: "ailit-eth-a1", description: "Evaluate systemic impacts of AI on equity and justice" }] },
        { id: "ailit-eth-expert", name: "Expert", description: "Leadership", order: 4, indicators: [{ id: "ailit-eth-e1", description: "Lead ethical AI governance initiatives" }] },
      ]},
      { id: "ailit-futures", name: "AI Futures", description: "Anticipating and shaping AI's trajectory", order: 4, levels: [
        { id: "ailit-fut-novice", name: "Novice", description: "Awareness", order: 1, indicators: [{ id: "ailit-fut-n1", description: "Aware AI is rapidly evolving" }] },
        { id: "ailit-fut-intermediate", name: "Intermediate", description: "Engagement", order: 2, indicators: [{ id: "ailit-fut-i1", description: "Track emerging AI trends and their implications" }] },
        { id: "ailit-fut-advanced", name: "Advanced", description: "Strategic thinking", order: 3, indicators: [{ id: "ailit-fut-a1", description: "Anticipate AI impacts on professional domains" }] },
        { id: "ailit-fut-expert", name: "Expert", description: "Visioning", order: 4, indicators: [{ id: "ailit-fut-e1", description: "Shape institutional AI strategy and roadmaps" }] },
      ]},
    ],
  },

  // ═══════════════════════════════════════════════════
  // 9–15. BDC Role Profiles (from source JSON data)
  // Uses correct JISC Discovery Tool 3-level individual model:
  // Developing / Capable / Proficient
  // ═══════════════════════════════════════════════════
  ...[
    {
      id: "bdc-individual", name: "JISC BDC Individual Framework",
      audience: ["all_staff", "all_students"],
      overview: "Generic digital capability framework for all roles. 6 capability areas × 3 capability levels (Developing → Capable → Proficient).",
    },
    {
      id: "bdc-teacher-he", name: "JISC BDC Teacher HE Profile",
      audience: ["educator", "teaching_staff_he"],
      overview: "Teaching-focused digital capabilities for HE lecturers. 6 capability areas × 3 levels (Developing → Capable → Proficient). Maps to PSF 2023 dimensions.",
    },
    {
      id: "bdc-researcher", name: "JISC BDC Researcher Profile",
      audience: ["researchers", "research_students"],
      overview: "Research-focused digital capabilities. 6 capability areas × 3 levels (Developing → Capable → Proficient). Emphasises open research and data management. Maps to Vitae RDF.",
    },
    {
      id: "bdc-professional-services", name: "JISC BDC Professional Services Profile",
      audience: ["professional_services_staff"],
      overview: "Digital capabilities for professional services staff. 6 capability areas × 3 levels (Developing → Capable → Proficient).",
    },
    {
      id: "bdc-learning-technology", name: "JISC BDC Learning Technology Profile",
      audience: ["learning_technologists"],
      overview: "Digital capabilities for learning technology specialists. 6 capability areas × 3 levels (Developing → Capable → Proficient). Bridges technical and pedagogical perspectives.",
    },
    {
      id: "bdc-digital-leader", name: "JISC BDC Digital Leader Profile",
      audience: ["senior_leaders", "strategic_leaders", "governors"],
      overview: "Strategic digital leadership capabilities. 6 capability areas × 3 levels (Developing → Capable → Proficient). AI strategy and responsible AI governance prominent.",
    },
    {
      id: "bdc-educational-developer", name: "JISC BDC Educational Developer Profile",
      audience: ["educational_developers"],
      overview: "Digital capabilities for educational development professionals. 6 capability areas × 3 levels (Developing → Capable → Proficient). Bridges individual teacher capability and institutional strategy.",
    },
  ].map(profile => ({
    id: profile.id, name: profile.name, source: "JISC", type: "capability" as const,
    scope: "individual" as const,
    targetAudience: profile.audience,
    overview: profile.overview,
    crossReferences: ["maturity-jisc-ai"],
    dimensions: [
      { id: `${profile.id}-prof`, name: "Digital Proficiency and Productivity", description: "Using digital devices, networks, applications, AI, software and services effectively and productively", order: 1, levels: [
        { id: `${profile.id}-prof-developing`, name: "Developing", description: "Awareness, exploration and guided experimentation with AI-enhanced productivity tools", order: 1, indicators: [
          { id: `${profile.id}-prof-dev-1`, description: "Understand what AI is and how it relates to existing digital tools" },
          { id: `${profile.id}-prof-dev-2`, description: "Identify AI-powered features in tools they already use" },
          { id: `${profile.id}-prof-dev-3`, description: "Experiment with AI tools to enhance personal productivity" },
          { id: `${profile.id}-prof-dev-4`, description: "Evaluate AI-generated outputs for accuracy and fitness for purpose" },
        ]},
        { id: `${profile.id}-prof-capable`, name: "Capable", description: "Confident, systematic AI-augmented professional practice", order: 2, indicators: [
          { id: `${profile.id}-prof-cap-1`, description: "Integrate AI tools systematically into professional workflows" },
          { id: `${profile.id}-prof-cap-2`, description: "Select appropriate AI and non-AI tools for different professional tasks" },
          { id: `${profile.id}-prof-cap-3`, description: "Support colleagues in basic AI tool adoption" },
        ]},
        { id: `${profile.id}-prof-proficient`, name: "Proficient", description: "Leading AI integration and shaping institutional strategy", order: 3, indicators: [
          { id: `${profile.id}-prof-pro-1`, description: "Lead AI-enhanced productivity practices across a team or department" },
          { id: `${profile.id}-prof-pro-2`, description: "Mentor colleagues in effective and responsible AI tool use" },
          { id: `${profile.id}-prof-pro-3`, description: "Shape institutional and sector approaches to AI-enhanced productivity" },
        ]},
      ]},
      { id: `${profile.id}-creation`, name: "Digital Creation, Problem-Solving and Innovation", description: "Digital production of content including AI-generated content; using digital evidence to solve problems", order: 2, levels: [
        { id: `${profile.id}-creat-developing`, name: "Developing", description: "Understanding and experimenting with AI content generation", order: 1, indicators: [
          { id: `${profile.id}-creat-dev-1`, description: "Understand how AI can generate and assist with content creation" },
          { id: `${profile.id}-creat-dev-2`, description: "Experiment with AI tools for content creation including text, image and code generation" },
          { id: `${profile.id}-creat-dev-3`, description: "Use AI to generate initial drafts or prototypes for professional tasks" },
        ]},
        { id: `${profile.id}-creat-capable`, name: "Capable", description: "Routine AI-enhanced creation and problem-solving", order: 2, indicators: [
          { id: `${profile.id}-creat-cap-1`, description: "Integrate AI into creative and problem-solving workflows" },
          { id: `${profile.id}-creat-cap-2`, description: "Evaluate AI-generated solutions critically before implementation" },
        ]},
        { id: `${profile.id}-creat-proficient`, name: "Proficient", description: "Leading AI innovation and shaping sector practices", order: 3, indicators: [
          { id: `${profile.id}-creat-pro-1`, description: "Lead AI-enhanced innovation within team or department" },
          { id: `${profile.id}-creat-pro-2`, description: "Develop frameworks for evaluating AI-generated content quality" },
          { id: `${profile.id}-creat-pro-3`, description: "Pioneer new AI-enabled approaches to digital creation and innovation" },
        ]},
      ]},
      { id: `${profile.id}-learning`, name: "Digital Learning and Development", description: "Learning in digital settings including AI-enhanced learning and digital teaching practices", order: 3, levels: [
        { id: `${profile.id}-learn-developing`, name: "Developing", description: "Awareness and exploration of AI in learning", order: 1, indicators: [
          { id: `${profile.id}-learn-dev-1`, description: "Understand how AI can support learning and professional development" },
          { id: `${profile.id}-learn-dev-2`, description: "Experiment with AI-powered learning tools for personal development" },
        ]},
        { id: `${profile.id}-learn-capable`, name: "Capable", description: "Systematic AI-enhanced learning practice", order: 2, indicators: [
          { id: `${profile.id}-learn-cap-1`, description: "Integrate AI into CPD and learning activities systematically" },
        ]},
        { id: `${profile.id}-learn-proficient`, name: "Proficient", description: "Leading AI-enhanced learning approaches", order: 3, indicators: [
          { id: `${profile.id}-learn-pro-1`, description: "Lead institutional approaches to AI-enhanced learning and development" },
          { id: `${profile.id}-learn-pro-2`, description: "Shape sector approaches to AI in learning and professional development" },
        ]},
      ]},
      { id: `${profile.id}-literacies`, name: "Information, Data and Media Literacies", description: "Finding, evaluating, managing and sharing digital information including AI-generated content", order: 4, levels: [
        { id: `${profile.id}-lit-developing`, name: "Developing", order: 1, description: "Awareness and exploration of AI's impact on information", indicators: [
          { id: `${profile.id}-lit-dev-1`, description: "Understand how AI affects information reliability and media authenticity" },
          { id: `${profile.id}-lit-dev-2`, description: "Experiment with AI tools for information gathering and data analysis" },
        ]},
        { id: `${profile.id}-lit-capable`, name: "Capable", order: 2, description: "Systematic evaluation of AI-generated information", indicators: [
          { id: `${profile.id}-lit-cap-1`, description: "Systematically evaluate AI-generated information for accuracy and bias" },
        ]},
        { id: `${profile.id}-lit-proficient`, name: "Proficient", order: 3, description: "Leading AI-era information literacy practices", indicators: [
          { id: `${profile.id}-lit-pro-1`, description: "Lead institutional data literacy practices including AI-generated content evaluation" },
          { id: `${profile.id}-lit-pro-2`, description: "Shape sector approaches to AI-era information and media literacy" },
        ]},
      ]},
      { id: `${profile.id}-comms`, name: "Digital Communication, Collaboration and Participation", description: "Communicating effectively in digital media including AI-powered collaboration tools", order: 5, levels: [
        { id: `${profile.id}-com-developing`, name: "Developing", order: 1, description: "Awareness and exploration of AI collaboration tools", indicators: [
          { id: `${profile.id}-com-dev-1`, description: "Understand how AI can enhance digital communication and collaboration" },
          { id: `${profile.id}-com-dev-2`, description: "Experiment with AI-powered communication and collaboration tools" },
        ]},
        { id: `${profile.id}-com-capable`, name: "Capable", order: 2, description: "Routine AI-enhanced collaboration", indicators: [
          { id: `${profile.id}-com-cap-1`, description: "Integrate AI tools into team collaboration workflows" },
        ]},
        { id: `${profile.id}-com-proficient`, name: "Proficient", order: 3, description: "Leading AI-powered collaboration practices", indicators: [
          { id: `${profile.id}-com-pro-1`, description: "Lead AI-enhanced collaboration practices across teams" },
          { id: `${profile.id}-com-pro-2`, description: "Shape sector approaches to AI-powered collaboration and participation" },
        ]},
      ]},
      { id: `${profile.id}-identity`, name: "Digital Identity and Wellbeing", description: "Managing professional digital identity and wellbeing in AI-mediated environments", order: 6, levels: [
        { id: `${profile.id}-id-developing`, name: "Developing", order: 1, description: "Awareness and exploration of AI's impact on identity and wellbeing", indicators: [
          { id: `${profile.id}-id-dev-1`, description: "Understand AI's impact on digital identity, privacy and wellbeing" },
          { id: `${profile.id}-id-dev-2`, description: "Explore how AI affects personal data and digital footprint" },
        ]},
        { id: `${profile.id}-id-capable`, name: "Capable", order: 2, description: "Proactive management of digital identity in AI contexts", indicators: [
          { id: `${profile.id}-id-cap-1`, description: "Manage digital identity and wellbeing proactively in AI-mediated environments" },
        ]},
        { id: `${profile.id}-id-proficient`, name: "Proficient", order: 3, description: "Leading digital wellbeing practices for AI era", indicators: [
          { id: `${profile.id}-id-pro-1`, description: "Lead institutional digital wellbeing practices for AI-mediated work" },
          { id: `${profile.id}-id-pro-2`, description: "Shape sector approaches to digital identity and wellbeing in AI era" },
        ]},
      ]},
    ],
  } as FrameworkContext)),


  // ═══════════════════════════════════════════════════
  // 16. JISC AI Maturity Model
  // ═══════════════════════════════════════════════════
  {
    id: "jisc-ai-maturity",
    name: "JISC AI Maturity Model",
    source: "JISC", type: "maturity", scope: "institutional",
    targetAudience: ["leader", "admin"],
    overview: "UK-focused institutional AI maturity assessment. 5 components (Strategy, People, Technology, Data, Ethics) × 3 maturity stages. Helps HE/FE institutions benchmark their AI adoption journey.",
    crossReferences: ["maturity-jisc", "ai-capability", "guidance-policy"],
    dimensions: [
      { id: "jisc-ai-strategy", name: "AI Strategy & Leadership", description: "Strategic vision and governance for AI", order: 1, levels: [
        { id: "jisc-ai-str-emerging", name: "Emerging", description: "No formal AI strategy", order: 1, indicators: [{ id: "jisc-ai-str-e1", description: "Ad-hoc AI experimentation without strategic direction" }] },
        { id: "jisc-ai-str-developing", name: "Developing", description: "Strategy in formation", order: 2, indicators: [{ id: "jisc-ai-str-d1", description: "AI strategy being drafted with stakeholder input" }] },
        { id: "jisc-ai-str-mature", name: "Mature", description: "Embedded AI strategy", order: 3, indicators: [{ id: "jisc-ai-str-m1", description: "AI strategy integrated with institutional objectives" }] },
      ]},
      { id: "jisc-ai-people", name: "People & Skills", description: "Workforce AI capability development", order: 2, levels: [
        { id: "jisc-ai-ppl-emerging", name: "Emerging", description: "Limited awareness", order: 1, indicators: [{ id: "jisc-ai-ppl-e1", description: "Few staff have AI skills" }] },
        { id: "jisc-ai-ppl-developing", name: "Developing", description: "Training programmes", order: 2, indicators: [{ id: "jisc-ai-ppl-d1", description: "Structured AI training being rolled out" }] },
        { id: "jisc-ai-ppl-mature", name: "Mature", description: "Organisation-wide capability", order: 3, indicators: [{ id: "jisc-ai-ppl-m1", description: "AI skills embedded in all role profiles" }] },
      ]},
      { id: "jisc-ai-tech", name: "Technology & Infrastructure", description: "AI-ready technical foundations", order: 3, levels: [
        { id: "jisc-ai-tech-emerging", name: "Emerging", description: "Basic infrastructure", order: 1, indicators: [{ id: "jisc-ai-tech-e1", description: "No dedicated AI infrastructure" }] },
        { id: "jisc-ai-tech-developing", name: "Developing", description: "Building capacity", order: 2, indicators: [{ id: "jisc-ai-tech-d1", description: "AI platforms being piloted" }] },
        { id: "jisc-ai-tech-mature", name: "Mature", description: "Integrated AI stack", order: 3, indicators: [{ id: "jisc-ai-tech-m1", description: "Enterprise AI platforms supporting multiple use cases" }] },
      ]},
      { id: "jisc-ai-data", name: "Data Readiness", description: "Data quality and governance for AI", order: 4, levels: [
        { id: "jisc-ai-data-emerging", name: "Emerging", description: "Siloed data", order: 1, indicators: [{ id: "jisc-ai-data-e1", description: "Data quality inconsistent and ungoverned" }] },
        { id: "jisc-ai-data-developing", name: "Developing", description: "Governance forming", order: 2, indicators: [{ id: "jisc-ai-data-d1", description: "Data governance framework being implemented" }] },
        { id: "jisc-ai-data-mature", name: "Mature", description: "AI-ready data", order: 3, indicators: [{ id: "jisc-ai-data-m1", description: "High-quality, governed data pipelines for AI" }] },
      ]},
      { id: "jisc-ai-ethics", name: "Ethics & Responsible AI", description: "Ethical AI governance and transparency", order: 5, levels: [
        { id: "jisc-ai-eth-emerging", name: "Emerging", description: "Awareness only", order: 1, indicators: [{ id: "jisc-ai-eth-e1", description: "Ethical concerns recognised but not addressed" }] },
        { id: "jisc-ai-eth-developing", name: "Developing", description: "Policies forming", order: 2, indicators: [{ id: "jisc-ai-eth-d1", description: "AI ethics policy being developed" }] },
        { id: "jisc-ai-eth-mature", name: "Mature", description: "Embedded ethics", order: 3, indicators: [{ id: "jisc-ai-eth-m1", description: "Ethics review integrated into AI procurement and deployment" }] },
      ]},
    ],
  },

  // ═══════════════════════════════════════════════════
  // 17. DEC AI Literacy Framework
  // ═══════════════════════════════════════════════════
  {
    id: "dec-ai-literacy",
    name: "DEC AI Literacy Framework",
    source: "DEC", type: "competency", scope: "individual",
    targetAudience: ["educator", "student"],
    overview: "5 dimensions × 4 proficiency levels (Awareness→Mastery). Covers understanding AI & data, using AI tools, creating with AI, AI ethics & society, and AI in discipline contexts. Includes learning outcomes and contextual activities per block.",
    crossReferences: ["ailit", "digcomp", "teacher-competency"],
    dimensions: [
      { id: "dec-understanding", name: "Understanding AI and Data", description: "How AI systems work, data processing, and AI-generated output", order: 1, levels: [
        { id: "dec-und-aware", name: "Awareness", description: "Basic AI concepts", order: 1, indicators: [{ id: "dec-und-a1", description: "Define AI and identify common AI applications in daily life" }] },
        { id: "dec-und-explore", name: "Exploration", description: "Deeper understanding", order: 2, indicators: [{ id: "dec-und-e1", description: "Analyse how AI processes data and generates outputs" }] },
        { id: "dec-und-practice", name: "Practice", description: "Applied knowledge", order: 3, indicators: [{ id: "dec-und-p1", description: "Evaluate AI system outputs for accuracy and bias" }] },
        { id: "dec-und-mastery", name: "Mastery", description: "Expert evaluation", order: 4, indicators: [{ id: "dec-und-m1", description: "Critically assess AI architectures and their societal implications" }] },
      ]},
      { id: "dec-using", name: "Using AI Tools", description: "Effective and critical use of AI applications", order: 2, levels: [
        { id: "dec-use-aware", name: "Awareness", description: "Basic use", order: 1, indicators: [{ id: "dec-use-a1", description: "Use AI tools for simple, guided tasks" }] },
        { id: "dec-use-explore", name: "Exploration", description: "Independent use", order: 2, indicators: [{ id: "dec-use-e1", description: "Select and apply AI tools independently for specific tasks" }] },
        { id: "dec-use-practice", name: "Practice", description: "Strategic use", order: 3, indicators: [{ id: "dec-use-p1", description: "Integrate AI tools into complex professional workflows" }] },
        { id: "dec-use-mastery", name: "Mastery", description: "Transformative use", order: 4, indicators: [{ id: "dec-use-m1", description: "Innovate new approaches through advanced AI tool combinations" }] },
      ]},
      { id: "dec-creating", name: "Creating with AI", description: "Co-creating content and solutions using AI", order: 3, levels: [
        { id: "dec-cre-aware", name: "Awareness", description: "Basic creation", order: 1, indicators: [{ id: "dec-cre-a1", description: "Use AI to assist with simple content creation" }] },
        { id: "dec-cre-explore", name: "Exploration", description: "Creative application", order: 2, indicators: [{ id: "dec-cre-e1", description: "Co-create content using AI with iterative refinement" }] },
        { id: "dec-cre-practice", name: "Practice", description: "Advanced creation", order: 3, indicators: [{ id: "dec-cre-p1", description: "Design AI-enhanced solutions for complex problems" }] },
        { id: "dec-cre-mastery", name: "Mastery", description: "Innovation", order: 4, indicators: [{ id: "dec-cre-m1", description: "Lead innovative AI-powered creation projects" }] },
      ]},
      { id: "dec-ethics", name: "AI Ethics and Society", description: "Social, ethical, and regulatory implications", order: 4, levels: [
        { id: "dec-eth-aware", name: "Awareness", description: "Basic awareness", order: 1, indicators: [{ id: "dec-eth-a1", description: "Recognise ethical issues in AI use" }] },
        { id: "dec-eth-explore", name: "Exploration", description: "Critical thinking", order: 2, indicators: [{ id: "dec-eth-e1", description: "Analyse ethical implications of AI decisions" }] },
        { id: "dec-eth-practice", name: "Practice", description: "Applied ethics", order: 3, indicators: [{ id: "dec-eth-p1", description: "Apply ethical frameworks to real AI scenarios" }] },
        { id: "dec-eth-mastery", name: "Mastery", description: "Governance", order: 4, indicators: [{ id: "dec-eth-m1", description: "Develop and advocate for responsible AI policies" }] },
      ]},
      { id: "dec-discipline", name: "AI in Discipline Contexts", description: "Domain-specific AI integration", order: 5, levels: [
        { id: "dec-dis-aware", name: "Awareness", description: "Field awareness", order: 1, indicators: [{ id: "dec-dis-a1", description: "Identify how AI is used in one's own discipline" }] },
        { id: "dec-dis-explore", name: "Exploration", description: "Field exploration", order: 2, indicators: [{ id: "dec-dis-e1", description: "Explore discipline-specific AI tools and applications" }] },
        { id: "dec-dis-practice", name: "Practice", description: "Field integration", order: 3, indicators: [{ id: "dec-dis-p1", description: "Apply AI meaningfully within professional practice" }] },
        { id: "dec-dis-mastery", name: "Mastery", description: "Field leadership", order: 4, indicators: [{ id: "dec-dis-m1", description: "Pioneer AI-driven innovation in professional domain" }] },
      ]},
    ],
  },

  // ═══════════════════════════════════════════════════
  // 18. DigComp 3.0
  // ═══════════════════════════════════════════════════
  {
    id: "digcomp",
    name: "DigComp 3.0: European Digital Competence Framework",
    source: "EU JRC", type: "competency", scope: "individual",
    targetAudience: ["educator", "student", "admin"],
    overview: "5 competence areas, 21 competences, 4 proficiency bands (Basic→Highly Advanced). 362 competence statements and 523 learning outcomes. Includes AI relevance labels (14% AI-Explicit, 68% AI-Implicit). The EU reference framework for digital skills.",
    crossReferences: ["teacher-competency", "ailit", "dec-ai-literacy"],
    dimensions: [
      { id: "dc-info", name: "Information & Data Literacy", description: "Browsing, searching, evaluating, managing data", order: 1, levels: [
        { id: "dc-info-basic", name: "Basic", description: "Foundation (1–2)", order: 1, indicators: [{ id: "dc-info-b1", description: "Identify information needs and search using digital tools" }] },
        { id: "dc-info-inter", name: "Intermediate", description: "Intermediate (3–4)", order: 2, indicators: [{ id: "dc-info-i1", description: "Evaluate reliability of information sources" }] },
        { id: "dc-info-adv", name: "Advanced", description: "Advanced (5–6)", order: 3, indicators: [{ id: "dc-info-a1", description: "Develop strategies for complex information retrieval including AI-powered search" }] },
        { id: "dc-info-hadv", name: "Highly Advanced", description: "Highly Specialised (7–8)", order: 4, indicators: [{ id: "dc-info-h1", description: "Create innovative solutions for knowledge management" }] },
      ]},
      { id: "dc-comm", name: "Communication & Collaboration", description: "Interacting, sharing, engaging through digital tech", order: 2, levels: [
        { id: "dc-comm-basic", name: "Basic", description: "Foundation", order: 1, indicators: [{ id: "dc-comm-b1", description: "Interact using basic digital communication tools" }] },
        { id: "dc-comm-inter", name: "Intermediate", description: "Intermediate", order: 2, indicators: [{ id: "dc-comm-i1", description: "Share and collaborate using appropriate platforms" }] },
        { id: "dc-comm-adv", name: "Advanced", description: "Advanced", order: 3, indicators: [{ id: "dc-comm-a1", description: "Manage digital identities and engage with AI-mediated communication" }] },
        { id: "dc-comm-hadv", name: "Highly Advanced", description: "Highly Specialised", order: 4, indicators: [{ id: "dc-comm-h1", description: "Innovate communication strategies using emerging technologies" }] },
      ]},
      { id: "dc-content", name: "Digital Content Creation", description: "Creating, editing, integrating digital content", order: 3, levels: [
        { id: "dc-cont-basic", name: "Basic", description: "Foundation", order: 1, indicators: [{ id: "dc-cont-b1", description: "Create simple digital content" }] },
        { id: "dc-cont-inter", name: "Intermediate", description: "Intermediate", order: 2, indicators: [{ id: "dc-cont-i1", description: "Modify and integrate content with copyright awareness" }] },
        { id: "dc-cont-adv", name: "Advanced", description: "Advanced", order: 3, indicators: [{ id: "dc-cont-a1", description: "Create complex content using AI co-creation tools" }] },
        { id: "dc-cont-hadv", name: "Highly Advanced", description: "Highly Specialised", order: 4, indicators: [{ id: "dc-cont-h1", description: "Develop innovative content solutions and programming approaches" }] },
      ]},
      { id: "dc-safety", name: "Safety", description: "Protecting devices, data, health, environment", order: 4, levels: [
        { id: "dc-safe-basic", name: "Basic", description: "Foundation", order: 1, indicators: [{ id: "dc-safe-b1", description: "Protect devices and personal data at basic level" }] },
        { id: "dc-safe-inter", name: "Intermediate", description: "Intermediate", order: 2, indicators: [{ id: "dc-safe-i1", description: "Apply security measures and manage digital wellbeing" }] },
        { id: "dc-safe-adv", name: "Advanced", description: "Advanced", order: 3, indicators: [{ id: "dc-safe-a1", description: "Evaluate AI safety risks and implement mitigation strategies" }] },
        { id: "dc-safe-hadv", name: "Highly Advanced", description: "Highly Specialised", order: 4, indicators: [{ id: "dc-safe-h1", description: "Develop organisational safety policies for AI systems" }] },
      ]},
      { id: "dc-problem", name: "Problem Solving", description: "Solving technical problems, identifying needs, creativity", order: 5, levels: [
        { id: "dc-prob-basic", name: "Basic", description: "Foundation", order: 1, indicators: [{ id: "dc-prob-b1", description: "Identify basic technical problems and digital needs" }] },
        { id: "dc-prob-inter", name: "Intermediate", description: "Intermediate", order: 2, indicators: [{ id: "dc-prob-i1", description: "Select digital tools creatively to solve problems" }] },
        { id: "dc-prob-adv", name: "Advanced", description: "Advanced", order: 3, indicators: [{ id: "dc-prob-a1", description: "Apply computational thinking and AI approaches to complex problems" }] },
        { id: "dc-prob-hadv", name: "Highly Advanced", description: "Highly Specialised", order: 4, indicators: [{ id: "dc-prob-h1", description: "Create innovative solutions addressing digital competence gaps" }] },
      ]},
    ],
  },

  // ═══════════════════════════════════════════════════
  // 19. ISTE Standards for Students
  // ═══════════════════════════════════════════════════
  {
    id: "iste-students",
    name: "ISTE Standards for Students v4.02",
    source: "ISTE", type: "competency", scope: "individual",
    targetAudience: ["student"],
    overview: "7 standards with 28 indicators for student digital learning. Covers Empowered Learner, Digital Citizen, Knowledge Constructor, Innovative Designer, Computational Thinker, Creative Communicator, and Global Collaborator.",
    crossReferences: ["student-competency", "digcomp"],
    dimensions: [
      { id: "iste-s-empowered", name: "Empowered Learner", description: "Leverage technology to take an active role in learning", order: 1, levels: [
        { id: "iste-s-emp-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-s-emp-1a", description: "Articulate and set personal learning goals, develop strategies leveraging technology" },
          { id: "iste-s-emp-1b", description: "Build networks and customize learning environments" },
          { id: "iste-s-emp-1c", description: "Use technology to seek feedback that informs and improves practice" },
          { id: "iste-s-emp-1d", description: "Understand the fundamental concepts of technology operations and demonstrate the ability to transfer knowledge to new technologies and situations" },
        ]},
      ]},
      { id: "iste-s-citizen", name: "Digital Citizen", description: "Recognise rights and responsibilities in a digital world", order: 2, levels: [
        { id: "iste-s-cit-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-s-cit-2a", description: "Cultivate and manage digital identity and reputation" },
          { id: "iste-s-cit-2b", description: "Engage in positive, safe, legal and ethical behaviour online" },
          { id: "iste-s-cit-2c", description: "Demonstrate an understanding of and respect for rights and obligations of using intellectual property" },
          { id: "iste-s-cit-2d", description: "Manage personal data to maintain digital privacy and security" },
        ]},
      ]},
      { id: "iste-s-knowledge", name: "Knowledge Constructor", description: "Curate and construct knowledge using digital resources", order: 3, levels: [
        { id: "iste-s-know-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-s-know-3a", description: "Plan and employ effective research strategies" },
          { id: "iste-s-know-3b", description: "Evaluate the accuracy, perspective, credibility and relevance of information" },
          { id: "iste-s-know-3c", description: "Curate information from digital resources using a variety of tools and methods" },
          { id: "iste-s-know-3d", description: "Build knowledge by actively exploring real-world issues and problems" },
        ]},
      ]},
      { id: "iste-s-designer", name: "Innovative Designer", description: "Use design process to solve problems", order: 4, levels: [
        { id: "iste-s-des-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-s-des-4a", description: "Know and use a deliberate design process for generating ideas and solving problems" },
          { id: "iste-s-des-4b", description: "Select and use digital tools to plan and manage a design process" },
          { id: "iste-s-des-4c", description: "Develop, test and refine prototypes" },
          { id: "iste-s-des-4d", description: "Exhibit a tolerance for ambiguity, perseverance and capacity to work with open-ended problems" },
        ]},
      ]},
      { id: "iste-s-computational", name: "Computational Thinker", description: "Develop and employ strategies for understanding and solving problems with technology", order: 5, levels: [
        { id: "iste-s-comp-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-s-comp-5a", description: "Formulate problem definitions suited for technology-assisted methods" },
          { id: "iste-s-comp-5b", description: "Collect data or identify relevant data sets" },
          { id: "iste-s-comp-5c", description: "Break problems into component parts and use algorithmic thinking" },
          { id: "iste-s-comp-5d", description: "Understand how automation works and use algorithmic thinking to develop solutions" },
        ]},
      ]},
      { id: "iste-s-communicator", name: "Creative Communicator", description: "Communicate clearly using variety of platforms and tools", order: 6, levels: [
        { id: "iste-s-comm-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-s-comm-6a", description: "Choose appropriate platforms and tools for meeting desired objectives" },
          { id: "iste-s-comm-6b", description: "Create original works or responsibly repurpose digital resources" },
          { id: "iste-s-comm-6c", description: "Communicate complex ideas using digital tools" },
          { id: "iste-s-comm-6d", description: "Publish or present content that customises the message for intended audiences" },
        ]},
      ]},
      { id: "iste-s-collaborator", name: "Global Collaborator", description: "Use digital tools to broaden perspectives and enrich learning with others", order: 7, levels: [
        { id: "iste-s-collab-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-s-collab-7a", description: "Use digital tools to connect with learners from diverse backgrounds" },
          { id: "iste-s-collab-7b", description: "Use collaborative technologies to work with others" },
          { id: "iste-s-collab-7c", description: "Contribute constructively to project teams" },
          { id: "iste-s-collab-7d", description: "Explore local and global issues and use collaborative technologies to investigate solutions" },
        ]},
      ]},
    ],
  },

  // ═══════════════════════════════════════════════════
  // 20. ISTE Standards for Educators
  // ═══════════════════════════════════════════════════
  {
    id: "iste-educators",
    name: "ISTE Standards for Educators v4.02",
    source: "ISTE", type: "competency", scope: "individual",
    targetAudience: ["educator"],
    overview: "7 standards across Empowered Professional and Learning Catalyst groups, with 24 indicators. Covers Learner, Leader, Citizen, Collaborator, Designer, Facilitator, and Analyst roles.",
    crossReferences: ["teacher-competency", "iste-students"],
    dimensions: [
      { id: "iste-e-learner", name: "Learner", description: "Continuously improve practice through learning", order: 1, levels: [
        { id: "iste-e-learn-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-e-learn-1a", description: "Set professional learning goals to explore and apply pedagogical approaches enabled by technology" },
          { id: "iste-e-learn-1b", description: "Pursue professional interests by creating and actively participating in learning networks" },
          { id: "iste-e-learn-1c", description: "Stay current with research that supports improved student learning outcomes with technology" },
        ]},
      ]},
      { id: "iste-e-leader", name: "Leader", description: "Seek opportunities for leadership in technology", order: 2, levels: [
        { id: "iste-e-lead-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-e-lead-2a", description: "Shape, advance and accelerate a shared vision for technology-empowered learning" },
          { id: "iste-e-lead-2b", description: "Advocate for equitable access to educational technology" },
          { id: "iste-e-lead-2c", description: "Model for colleagues the identification, exploration, evaluation and adoption of new digital resources" },
        ]},
      ]},
      { id: "iste-e-citizen", name: "Citizen", description: "Inspire students to contribute positively to the digital world", order: 3, levels: [
        { id: "iste-e-cit-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-e-cit-3a", description: "Create experiences for learners to make positive, socially responsible contributions" },
          { id: "iste-e-cit-3b", description: "Establish a learning culture that promotes curiosity and critical examination of online resources" },
          { id: "iste-e-cit-3c", description: "Mentor students in safe, legal and ethical practices with digital tools" },
          { id: "iste-e-cit-3d", description: "Model and promote management of personal data and digital identity" },
        ]},
      ]},
      { id: "iste-e-collaborator", name: "Collaborator", description: "Collaborate with colleagues and students to improve practice", order: 4, levels: [
        { id: "iste-e-collab-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-e-collab-4a", description: "Dedicate planning time to collaborate with colleagues to create authentic learning experiences" },
          { id: "iste-e-collab-4b", description: "Collaborate and co-learn with students to discover and use new digital resources" },
          { id: "iste-e-collab-4c", description: "Use collaborative tools to expand communication with parents, colleagues, and community" },
          { id: "iste-e-collab-4d", description: "Demonstrate cultural competency interacting with students, parents and colleagues" },
        ]},
      ]},
      { id: "iste-e-designer", name: "Designer", description: "Design authentic, learner-driven activities", order: 5, levels: [
        { id: "iste-e-des-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-e-des-5a", description: "Use technology to create, adapt and personalise learning experiences" },
          { id: "iste-e-des-5b", description: "Design authentic learning activities that align with content area standards and use digital tools" },
          { id: "iste-e-des-5c", description: "Explore and apply instructional design principles to create innovative digital learning environments" },
        ]},
      ]},
      { id: "iste-e-facilitator", name: "Facilitator", description: "Facilitate learning with technology", order: 6, levels: [
        { id: "iste-e-fac-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-e-fac-6a", description: "Foster a culture where students take ownership of their learning goals in both independent and group settings" },
          { id: "iste-e-fac-6b", description: "Manage the use of technology and student learning strategies in digital platforms" },
          { id: "iste-e-fac-6c", description: "Create learning opportunities that challenge students to use a design process and computational thinking" },
          { id: "iste-e-fac-6d", description: "Model and nurture creativity and creative expression to communicate ideas" },
        ]},
      ]},
      { id: "iste-e-analyst", name: "Analyst", description: "Understand and use data to drive instruction", order: 7, levels: [
        { id: "iste-e-ana-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-e-ana-7a", description: "Provide alternative ways for students to demonstrate competency using technology" },
          { id: "iste-e-ana-7b", description: "Use technology to design and implement formative and summative assessments" },
          { id: "iste-e-ana-7c", description: "Use assessment data to guide progress and communicate with students, parents and colleagues" },
        ]},
      ]},
    ],
  },

  // ═══════════════════════════════════════════════════
  // 21. ISTE Standards for Coaches
  // ═══════════════════════════════════════════════════
  {
    id: "iste-coaches",
    name: "ISTE Standards for Coaches v4.02",
    source: "ISTE", type: "competency", scope: "individual",
    targetAudience: ["educator", "leader"],
    overview: "7 standards with 26 indicators for technology coaches. Covers Change Agent, Connected Learner, Collaborator, Learning Designer, Professional Learning Facilitator, Data-Driven Decision Maker, and Digital Citizen Advocate.",
    crossReferences: ["iste-educators", "teacher-competency"],
    dimensions: [
      { id: "iste-c-change", name: "Change Agent", description: "Create a shared vision for technology integration", order: 1, levels: [
        { id: "iste-c-cha-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-c-cha-1a", description: "Create a shared vision and culture for using technology to learn and accelerate transformation" },
          { id: "iste-c-cha-1b", description: "Facilitate equitable use of digital learning tools and content" },
          { id: "iste-c-cha-1c", description: "Cultivate a supportive coaching culture for technology use" },
          { id: "iste-c-cha-1d", description: "Recognize and remove barriers to equitable technology access" },
        ]},
      ]},
      { id: "iste-c-connected", name: "Connected Learner", description: "Model continuous learning with technology", order: 2, levels: [
        { id: "iste-c-con-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-c-con-2a", description: "Pursue professional learning to deepen expertise and leadership capacity" },
          { id: "iste-c-con-2b", description: "Actively participate in professional learning networks" },
          { id: "iste-c-con-2c", description: "Explore and apply emerging technologies and pedagogical approaches" },
        ]},
      ]},
      { id: "iste-c-collaborator", name: "Collaborator", description: "Partner with educators to enhance practice", order: 3, levels: [
        { id: "iste-c-col-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-c-col-3a", description: "Establish trusting and respectful coaching relationships" },
          { id: "iste-c-col-3b", description: "Partner with educators to identify digital learning needs" },
          { id: "iste-c-col-3c", description: "Partner with educators to evaluate digital tools and content" },
          { id: "iste-c-col-3d", description: "Personalize support based on educator needs" },
        ]},
      ]},
      { id: "iste-c-designer", name: "Learning Designer", description: "Design innovative learning with technology", order: 4, levels: [
        { id: "iste-c-des-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-c-des-4a", description: "Collaborate with educators to design learning experiences using technology" },
          { id: "iste-c-des-4b", description: "Collaborate with educators to design and develop digitally-rich assessments" },
          { id: "iste-c-des-4c", description: "Model the use of instructional design principles with educators" },
          { id: "iste-c-des-4d", description: "Explore AI-powered tools to enhance teaching and learning" },
        ]},
      ]},
      { id: "iste-c-facilitator", name: "Professional Learning Facilitator", description: "Facilitate professional development", order: 5, levels: [
        { id: "iste-c-fac-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-c-fac-5a", description: "Design professional learning based on needs assessments" },
          { id: "iste-c-fac-5b", description: "Model effective use of technology for learning" },
          { id: "iste-c-fac-5c", description: "Provide follow-up support to ensure implementation" },
          { id: "iste-c-fac-5d", description: "Evaluate professional learning impact on educator practice" },
        ]},
      ]},
      { id: "iste-c-data", name: "Data-Driven Decision Maker", description: "Use data to inform coaching", order: 6, levels: [
        { id: "iste-c-dat-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-c-dat-6a", description: "Assist educators in using data to inform instruction" },
          { id: "iste-c-dat-6b", description: "Support educators in examining data for equity gaps" },
          { id: "iste-c-dat-6c", description: "Support educators in using qualitative and quantitative data" },
          { id: "iste-c-dat-6d", description: "Model ethical use of student data and privacy practices" },
        ]},
      ]},
      { id: "iste-c-citizen", name: "Digital Citizen Advocate", description: "Model and promote digital citizenship", order: 7, levels: [
        { id: "iste-c-cit-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-c-cit-7a", description: "Inspire and encourage educators to promote digital citizenship" },
          { id: "iste-c-cit-7b", description: "Partner with educators to promote safe online behaviour" },
          { id: "iste-c-cit-7c", description: "Promote accessibility practices and Universal Design for Learning" },
        ]},
      ]},
    ],
  },

  // ═══════════════════════════════════════════════════
  // 22. ISTE Standards for Leaders
  // ═══════════════════════════════════════════════════
  {
    id: "iste-leaders",
    name: "ISTE Standards for Education Leaders v4.02",
    source: "ISTE", type: "competency", scope: "institutional",
    targetAudience: ["leader", "admin"],
    overview: "5 standards with 23 indicators for education leaders. Covers Visionary Planner, Systems Designer, Empowering Leader, Connected Learner, and Equity and Citizenship Advocate.",
    crossReferences: ["iste-educators", "guidance-policy", "ai-capability"],
    dimensions: [
      { id: "iste-l-visionary", name: "Visionary Planner", description: "Create a shared vision for educational technology", order: 1, levels: [
        { id: "iste-l-vis-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-l-vis-1a", description: "Engage stakeholders in developing and adopting a shared vision" },
          { id: "iste-l-vis-1b", description: "Build on technology-savvy leaders and invest in capacity of others" },
          { id: "iste-l-vis-1c", description: "Evaluate progress on the technology development plan" },
          { id: "iste-l-vis-1d", description: "Share lessons learned with other education leaders" },
        ]},
      ]},
      { id: "iste-l-systems", name: "Systems Designer", description: "Build teams and systems to implement technology vision", order: 2, levels: [
        { id: "iste-l-sys-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-l-sys-2a", description: "Build creative teams and systems to achieve technology vision" },
          { id: "iste-l-sys-2b", description: "Ensure robust infrastructure for security, privacy, and safety" },
          { id: "iste-l-sys-2c", description: "Protect privacy and security of data" },
          { id: "iste-l-sys-2d", description: "Model and promote interoperability and open data standards" },
          { id: "iste-l-sys-2e", description: "Make informed decisions about technology procurement" },
        ]},
      ]},
      { id: "iste-l-empowering", name: "Empowering Leader", description: "Create a culture of innovation", order: 3, levels: [
        { id: "iste-l-emp-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-l-emp-3a", description: "Empower educators to exercise professional agency in technology use" },
          { id: "iste-l-emp-3b", description: "Champion policies and provide resources for technology integration" },
          { id: "iste-l-emp-3c", description: "Inspire innovation, experimentation and a willingness to learn from and adapt to failure" },
          { id: "iste-l-emp-3d", description: "Support educators in using technology to advance learning" },
          { id: "iste-l-emp-3e", description: "Develop shared understanding of how technology can enhance learning experiences" },
        ]},
      ]},
      { id: "iste-l-connected", name: "Connected Learner", description: "Model and promote continuous learning", order: 4, levels: [
        { id: "iste-l-con-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-l-con-4a", description: "Set goals to remain current on emerging technologies for learning" },
          { id: "iste-l-con-4b", description: "Participate regularly in online professional learning networks" },
          { id: "iste-l-con-4c", description: "Use technology to engage in mentoring and networking" },
          { id: "iste-l-con-4d", description: "Develop the skills needed to lead and navigate change and promote digital age learning" },
        ]},
      ]},
      { id: "iste-l-equity", name: "Equity & Citizenship Advocate", description: "Model and promote equitable and inclusive digital citizenship", order: 5, levels: [
        { id: "iste-l-eq-prof", name: "Proficient", description: "Demonstrates standard", order: 1, indicators: [
          { id: "iste-l-eq-5a", description: "Ensure equitable access to technology and digital resources" },
          { id: "iste-l-eq-5b", description: "Model digital citizenship by critically evaluating online resources" },
          { id: "iste-l-eq-5c", description: "Cultivate responsible online behaviour including ethical use of AI" },
          { id: "iste-l-eq-5d", description: "Protect student data and privacy" },
          { id: "iste-l-eq-5e", description: "Proactively address accessibility needs through UDL principles" },
        ]},
      ]},
    ],
  },
];

/** Returns the raw framework data array for schema validation and direct access. */
export function getFrameworkData(): FrameworkContext[] {
  return frameworks;
}

/** Lightweight one-line-per-framework index (~500 tokens instead of ~23K). */
export function getFrameworkIndex(): string {
  return frameworks
    .map(
      (f, i) =>
        `${i + 1}. ${f.name} (${f.scope}, ${f.type}, audience: ${f.targetAudience.join("/")})`
    )
    .join("\n");
}

/** Returns the human-readable name for a framework ID, or the ID itself if not found. */
export function getFrameworkNameById(id: string): string {
  const f = frameworks.find((fw) => fw.id === id);
  return f ? f.name : id;
}

/** Returns framework IDs relevant to a user's role and assessed frameworks. */
export function getRelevantFrameworkIds(
  assessedFrameworkIds: string[],
  currentFrameworkId?: string | null,
  maxTotal: number = 3
): string[] {
  const ids = new Set<string>();
  if (currentFrameworkId) ids.add(currentFrameworkId);
  for (const id of assessedFrameworkIds) {
    if (ids.size >= maxTotal) break;
    ids.add(id);
  }
  return [...ids];
}

/** Returns full context for multiple frameworks by ID. */
export function getFrameworkContextByIds(ids: string[]): string {
  return ids
    .map((id) => getFrameworkContextById(id))
    .filter(Boolean)
    .join("\n\n---\n\n");
}

export function getFrameworkContext(): string {
  return frameworks
    .map((f, i) => {
      const dimText = f.dimensions
        .filter((d) => !d.parentDimensionId)
        .map((d) => {
          const levelText = d.levels
            .map((l) => {
              const indText = l.indicators.map((ind) => `        - ${ind.description}`).join("\n");
              return `      ${l.name}: ${l.description}\n${indText}`;
            })
            .join("\n");
          return `    ${d.name}: ${d.description}\n${levelText}`;
        })
        .join("\n");

      return `${i + 1}. ${f.name} (${f.source}, ${f.type}, scope: ${f.scope})
   Target audience: ${f.targetAudience.join(", ")}
   Overview: ${f.overview}
   Dimensions:
${dimText}
   Related frameworks: ${f.crossReferences.join(", ")}`;
    })
    .join("\n\n");
}

export function getFrameworkContextById(id: string): string | null {
  const f = frameworks.find((fw) => fw.id === id);
  if (!f) return null;

  const dimText = f.dimensions
    .map((d) => {
      const levelText = d.levels
        .map((l) => {
          const indText = l.indicators.map((ind) => `      - ${ind.description}`).join("\n");
          return `    ${l.name}: ${l.description}\n${indText}`;
        })
        .join("\n");
      return `  ${d.name}: ${d.description}\n${levelText}`;
    })
    .join("\n");

  return `${f.name} (${f.source}, ${f.type}, scope: ${f.scope})
Target audience: ${f.targetAudience.join(", ")}
Overview: ${f.overview}
Dimensions:
${dimText}
Related frameworks: ${f.crossReferences.join(", ")}`;
}

export function getFrameworkPaths(): Record<string, string> {
  const paths: Record<string, string> = {};
  for (const f of frameworks) {
    paths[f.name] = `/frameworks/${f.id}`;
  }
  return paths;
}
