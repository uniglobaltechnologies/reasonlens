/**
 * THE Digital Maturity Index — Word Report Generator
 *
 * Generates a professional institutional assessment report as a .docx file.
 * Uses the `docx` library (already installed for Policy page).
 */
import {
  Document,
  Packer,
  Paragraph,
  TextRun,
  HeadingLevel,
  Table,
  TableRow,
  TableCell,
  WidthType,
  AlignmentType,
  BorderStyle,
  PageBreak,
  ShadingType,
  TableLayoutType,
  Footer,
  PageNumber,
  Header,
} from "docx";
import { getFrameworkById } from "@/data/frameworks";

// ── Types ──────────────────────────────────────

interface DimensionResult {
  dimension_id: string;
  dimension_name: string;
  assigned_level: string;
  assigned_level_order: number;
  confidence: "high" | "medium" | "low";
  answer_count: number;
  answer_distribution: Record<string, number>;
}

export interface TheReportData {
  results: DimensionResult[];
  frameworkId: string;
  sessionId: string;
  completedAt: Date;
  context: {
    institution_size?: string | null;
    institution_type?: string | null;
    region?: string | null;
    funding_model?: string | null;
    respondent_role?: string | null;
    respondent_institutional_visibility?: string | null;
    digital_infrastructure_baseline?: string | null;
  };
  scenarioCount: number;
  interpretiveReport?: {
    executive_summary: string;
    pillar_teaching_learning: string;
    pillar_research: string;
    pillar_professional_services: string;
    pillar_planning_governance: string;
    recommendations: string;
    methodology_version?: string;
    generated_at?: string;
  };
}

// ── Constants ──────────────────────────────────

const PILLARS = [
  { key: "tl", name: "Teaching & Learning", prefix: "the-tl-" },
  { key: "re", name: "Research", prefix: "the-re-" },
  { key: "ps", name: "Professional Services", prefix: "the-ps-" },
  { key: "pg", name: "Planning & Governance", prefix: "the-pg-" },
];

const DIMENSIONS = ["strategy", "people", "technology", "data", "utilization"];
const DIMENSION_LABELS: Record<string, string> = {
  strategy: "Strategy",
  people: "People & Culture",
  technology: "Technology",
  data: "Data",
  utilization: "Utilisation",
};

const LEVEL_NAMES: Record<number, string> = {
  1: "Incidental",
  2: "Intentional",
  3: "Integrated",
  4: "Optimised",
};

const THIN_BORDER = {
  top: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
  bottom: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
  left: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
  right: { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" },
} as const;

const NO_BORDER = {
  top: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
} as const;

// ── Helpers ────────────────────────────────────

function resultsByPillar(results: DimensionResult[]) {
  return PILLARS.map((pillar) => ({
    ...pillar,
    dimensions: results.filter((r) => r.dimension_id.startsWith(pillar.prefix)),
  }));
}

function averageOrder(dims: DimensionResult[]): number {
  if (dims.length === 0) return 0;
  return dims.reduce((sum, d) => sum + d.assigned_level_order, 0) / dims.length;
}

function levelDistribution(results: DimensionResult[]) {
  const dist: Record<string, number> = {};
  for (const r of results) {
    dist[r.assigned_level] = (dist[r.assigned_level] || 0) + 1;
  }
  return dist;
}

function confidenceCounts(results: DimensionResult[]) {
  let high = 0, medium = 0, low = 0;
  for (const r of results) {
    if (r.confidence === "high") high++;
    else if (r.confidence === "medium") medium++;
    else low++;
  }
  return { high, medium, low };
}

function formatDate(d: Date): string {
  return d.toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
}

function cleanLabel(s?: string | null): string {
  if (!s) return "—";
  return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function shadingForLevel(order: number) {
  const colors: Record<number, string> = {
    1: "FDE8E8", // soft red
    2: "FEF3CD", // soft amber
    3: "D1FAE5", // soft green
    4: "DBEAFE", // soft blue
  };
  return { type: ShadingType.SOLID, color: colors[order] || "FFFFFF" };
}

// ── Cell builders ──────────────────────────────

function headerCell(text: string, width?: number): TableCell {
  return new TableCell({
    children: [new Paragraph({
      children: [new TextRun({ text, bold: true, size: 18, font: "Calibri", color: "FFFFFF" })],
      spacing: { before: 40, after: 40 },
    })],
    shading: { type: ShadingType.SOLID, color: "2D3748" },
    borders: THIN_BORDER,
    ...(width ? { width: { size: width, type: WidthType.PERCENTAGE } } : {}),
  });
}

function dataCell(text: string, options?: { shading?: ReturnType<typeof shadingForLevel>; bold?: boolean; width?: number }): TableCell {
  return new TableCell({
    children: [new Paragraph({
      children: [new TextRun({ text, size: 18, font: "Calibri", bold: options?.bold, color: options?.shading ? "1A202C" : undefined })],
      spacing: { before: 40, after: 40 },
    })],
    shading: options?.shading,
    borders: THIN_BORDER,
    ...(options?.width ? { width: { size: options.width, type: WidthType.PERCENTAGE } } : {}),
  });
}

// ── Section builders ───────────────────────────

function buildCoverPage(data: TheReportData): Paragraph[] {
  const ctx = data.context;
  return [
    new Paragraph({ spacing: { before: 2400 } }),
    new Paragraph({
      children: [new TextRun({ text: "THE Digital Maturity Index", size: 56, bold: true, font: "Calibri", color: "1A365D" })],
      alignment: AlignmentType.CENTER,
    }),
    new Paragraph({
      children: [new TextRun({ text: "Institutional Assessment Report", size: 32, font: "Calibri", color: "4A5568" })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 400 },
    }),
    new Paragraph({
      children: [new TextRun({ text: "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", size: 20, color: "CBD5E0" })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 400 },
    }),
    new Paragraph({
      children: [new TextRun({ text: `Institution type: ${cleanLabel(ctx.institution_type)}`, size: 22, font: "Calibri", color: "4A5568" })],
      alignment: AlignmentType.CENTER,
    }),
    new Paragraph({
      children: [new TextRun({ text: `Size: ${cleanLabel(ctx.institution_size)}  |  Region: ${cleanLabel(ctx.region)}  |  Funding: ${cleanLabel(ctx.funding_model)}`, size: 22, font: "Calibri", color: "4A5568" })],
      alignment: AlignmentType.CENTER,
    }),
    new Paragraph({
      children: [new TextRun({ text: `Respondent: ${cleanLabel(ctx.respondent_role)} (${cleanLabel(ctx.respondent_institutional_visibility)} visibility)`, size: 22, font: "Calibri", color: "4A5568" })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
    }),
    new Paragraph({
      children: [new TextRun({ text: `Assessment completed: ${formatDate(data.completedAt)}`, size: 20, font: "Calibri", color: "718096" })],
      alignment: AlignmentType.CENTER,
    }),
    new Paragraph({
      children: [new TextRun({ text: `${data.scenarioCount} scenarios  |  Session ${data.sessionId.slice(0, 8)}`, size: 18, font: "Calibri", color: "A0AEC0" })],
      alignment: AlignmentType.CENTER,
      spacing: { after: 600 },
    }),
    new Paragraph({
      children: [new TextRun({ text: "Generated by ReasonLens — reasonlens.com", size: 18, font: "Calibri", italics: true, color: "A0AEC0" })],
      alignment: AlignmentType.CENTER,
    }),
    new Paragraph({ children: [new PageBreak()] }),
  ];
}

function buildExecutiveSummary(data: TheReportData): Paragraph[] {
  const dist = levelDistribution(data.results);
  const conf = confidenceCounts(data.results);
  const pillared = resultsByPillar(data.results);

  const pillarAverages = pillared.map((p) => ({ name: p.name, avg: averageOrder(p.dimensions) }));
  const strongest = pillarAverages.reduce((a, b) => (a.avg >= b.avg ? a : b));
  const weakest = pillarAverages.reduce((a, b) => (a.avg <= b.avg ? a : b));

  const overallAvg = averageOrder(data.results);
  const overallLevel = LEVEL_NAMES[Math.round(overallAvg)] || "Intentional";

  const children: Paragraph[] = [
    new Paragraph({ text: "Executive Summary", heading: HeadingLevel.HEADING_1, spacing: { after: 200 } }),
    new Paragraph({
      children: [
        new TextRun({ text: `This assessment evaluated your institution across ${data.results.length} dimensions using ${data.scenarioCount} realistic institutional scenarios. `, size: 22, font: "Calibri" }),
        new TextRun({ text: `Your overall maturity profile centres around the ${overallLevel} level.`, size: 22, font: "Calibri", bold: true }),
      ],
      spacing: { after: 200 },
    }),
  ];

  // Level distribution
  children.push(new Paragraph({ text: "Maturity Profile", heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 } }));

  const distRows = Object.entries(LEVEL_NAMES).map(([order, name]) => {
    const count = dist[name] || 0;
    return new TableRow({
      children: [
        dataCell(name, { bold: true, width: 30 }),
        dataCell(`${count} dimension${count !== 1 ? "s" : ""}`, { shading: shadingForLevel(Number(order)), width: 70 }),
      ],
    });
  });

  children.push(new Table({
    rows: distRows,
    width: { size: 100, type: WidthType.PERCENTAGE },
    layout: TableLayoutType.FIXED,
  }));

  // Pillar summary
  children.push(new Paragraph({ spacing: { before: 200 } }));
  children.push(new Paragraph({
    children: [
      new TextRun({ text: "Strongest pillar: ", size: 22, font: "Calibri" }),
      new TextRun({ text: strongest.name, size: 22, font: "Calibri", bold: true }),
      new TextRun({ text: ` (avg ${strongest.avg.toFixed(1)}/4)`, size: 22, font: "Calibri", color: "718096" }),
    ],
  }));
  children.push(new Paragraph({
    children: [
      new TextRun({ text: "Weakest pillar: ", size: 22, font: "Calibri" }),
      new TextRun({ text: weakest.name, size: 22, font: "Calibri", bold: true }),
      new TextRun({ text: ` (avg ${weakest.avg.toFixed(1)}/4)`, size: 22, font: "Calibri", color: "718096" }),
    ],
    spacing: { after: 200 },
  }));

  // Confidence summary
  children.push(new Paragraph({
    children: [
      new TextRun({ text: `Confidence: `, size: 22, font: "Calibri" }),
      new TextRun({ text: `${conf.high} high`, size: 22, font: "Calibri", bold: true, color: "276749" }),
      new TextRun({ text: `, ${conf.medium} medium`, size: 22, font: "Calibri", color: "975A16" }),
      new TextRun({ text: `, ${conf.low} low`, size: 22, font: "Calibri", color: "9B2C2C" }),
    ],
    spacing: { after: 100 },
  }));

  if (conf.low > 0) {
    children.push(new Paragraph({
      children: [new TextRun({ text: `${conf.low} dimension${conf.low > 1 ? "s" : ""} showed low confidence — responses were inconsistent, suggesting deeper investigation may be needed.`, size: 20, font: "Calibri", italics: true, color: "718096" })],
      spacing: { after: 200 },
    }));
  }

  children.push(new Paragraph({ children: [new PageBreak()] }));
  return children;
}

function buildPillarSection(pillarName: string, dims: DimensionResult[], framework: ReturnType<typeof getFrameworkById>): Paragraph[] {
  const children: Paragraph[] = [
    new Paragraph({ text: pillarName, heading: HeadingLevel.HEADING_1, spacing: { after: 200 } }),
  ];

  // Summary table
  const headerRow = new TableRow({
    children: [
      headerCell("Dimension", 35),
      headerCell("Level", 25),
      headerCell("Confidence", 20),
      headerCell("Distribution", 20),
    ],
  });

  const dataRows = dims.map((d) => {
    const dimLabel = d.dimension_name.replace(/ \(.*\)$/, ""); // strip pillar suffix
    const distStr = Object.entries(d.answer_distribution)
      .map(([level, count]) => `${level}: ${count}`)
      .join(", ");

    return new TableRow({
      children: [
        dataCell(dimLabel, { bold: true, width: 35 }),
        dataCell(d.assigned_level, { shading: shadingForLevel(d.assigned_level_order), width: 25 }),
        dataCell(d.confidence.charAt(0).toUpperCase() + d.confidence.slice(1), { width: 20 }),
        dataCell(distStr, { width: 20 }),
      ],
    });
  });

  children.push(new Table({
    rows: [headerRow, ...dataRows],
    width: { size: 100, type: WidthType.PERCENTAGE },
    layout: TableLayoutType.FIXED,
  }));

  // Dimension detail
  for (const d of dims) {
    const dimLabel = d.dimension_name.replace(/ \(.*\)$/, "");
    children.push(new Paragraph({ text: dimLabel, heading: HeadingLevel.HEADING_2, spacing: { before: 300, after: 100 } }));

    children.push(new Paragraph({
      children: [
        new TextRun({ text: "Assigned level: ", size: 22, font: "Calibri" }),
        new TextRun({ text: d.assigned_level, size: 22, font: "Calibri", bold: true }),
        new TextRun({ text: `  (${d.confidence} confidence)`, size: 20, font: "Calibri", color: "718096" }),
      ],
    }));

    // Get level description from framework
    if (framework) {
      const fwDim = framework.keyDimensions.find((fd) => fd.id === d.dimension_id);
      if (fwDim) {
        const level = fwDim.levels.find((l) => l.name.toLowerCase() === d.assigned_level.toLowerCase());
        if (level) {
          children.push(new Paragraph({
            children: [new TextRun({ text: level.description, size: 20, font: "Calibri", italics: true, color: "4A5568" })],
            spacing: { before: 60, after: 100 },
          }));

          // Key indicators for this level
          if (level.indicators.length > 0) {
            children.push(new Paragraph({
              children: [new TextRun({ text: "Key indicators at this level:", size: 20, font: "Calibri", bold: true })],
              spacing: { before: 60 },
            }));
            for (const ind of level.indicators.slice(0, 4)) {
              children.push(new Paragraph({
                children: [new TextRun({ text: ind.description, size: 20, font: "Calibri" })],
                bullet: { level: 0 },
              }));
            }
          }
        }
      }
    }
  }

  children.push(new Paragraph({ children: [new PageBreak()] }));
  return children;
}

function heatmapCell(result: DimensionResult | undefined): TableCell {
  if (!result) return dataCell("—");
  const confDots = result.confidence === "high" ? "●●●" : result.confidence === "medium" ? "●●○" : "●○○";
  return new TableCell({
    children: [
      new Paragraph({
        children: [new TextRun({ text: result.assigned_level, size: 17, font: "Calibri", bold: true, color: "1A202C" })],
        alignment: AlignmentType.CENTER,
        spacing: { before: 30, after: 0 },
      }),
      new Paragraph({
        children: [new TextRun({ text: confDots, size: 14, font: "Calibri", color: "718096" })],
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 30 },
      }),
    ],
    shading: shadingForLevel(result.assigned_level_order),
    borders: THIN_BORDER,
  });
}

function buildHeatMap(data: TheReportData): Paragraph[] {
  const children: Paragraph[] = [
    new Paragraph({ text: "Maturity Heat Map", heading: HeadingLevel.HEADING_1, spacing: { after: 200 } }),
    new Paragraph({
      children: [new TextRun({ text: "Rows = pillars, columns = cross-cutting dimensions. Cell colour encodes maturity level; dots indicate confidence (●●● high, ●●○ medium, ●○○ low).", size: 20, font: "Calibri", color: "718096" })],
      spacing: { after: 200 },
    }),
  ];

  const dimHeaders = DIMENSIONS.map((d) => headerCell(DIMENSION_LABELS[d] || d));
  const headerRow = new TableRow({
    children: [headerCell("Pillar", 25), ...dimHeaders],
  });

  const dataRows = PILLARS.map((pillar) => {
    const cells = DIMENSIONS.map((dim) => {
      const result = data.results.find((r) => r.dimension_id === `${pillar.prefix}${dim}`);
      return heatmapCell(result);
    });
    return new TableRow({
      children: [dataCell(pillar.name, { bold: true, width: 25 }), ...cells],
    });
  });

  children.push(new Table({
    rows: [headerRow, ...dataRows],
    width: { size: 100, type: WidthType.PERCENTAGE },
    layout: TableLayoutType.FIXED,
  }));

  // Legend
  children.push(new Paragraph({ spacing: { before: 200 } }));
  children.push(new Paragraph({
    children: [
      new TextRun({ text: "Legend: ", size: 18, font: "Calibri", bold: true }),
      new TextRun({ text: "Red = Incidental  |  Amber = Intentional  |  Green = Integrated  |  Blue = Optimised", size: 18, font: "Calibri", color: "718096" }),
    ],
  }));

  children.push(new Paragraph({ children: [new PageBreak()] }));
  return children;
}

function buildRecommendations(data: TheReportData): Paragraph[] {
  const children: Paragraph[] = [
    new Paragraph({ text: "Recommendations", heading: HeadingLevel.HEADING_1, spacing: { after: 200 } }),
  ];

  // Priority areas (Incidental)
  const incidental = data.results.filter((r) => r.assigned_level_order === 1);
  if (incidental.length > 0) {
    children.push(new Paragraph({ text: "Priority Areas for Strategic Planning", heading: HeadingLevel.HEADING_2, spacing: { before: 100, after: 100 } }));
    children.push(new Paragraph({
      children: [new TextRun({ text: `${incidental.length} dimension${incidental.length > 1 ? "s" : ""} scored at the Incidental level, indicating sporadic digital activity without coordinated strategy. These represent the highest-impact opportunities for improvement.`, size: 22, font: "Calibri" })],
      spacing: { after: 100 },
    }));
    for (const r of incidental) {
      children.push(new Paragraph({
        children: [new TextRun({ text: `${r.dimension_name}`, size: 22, font: "Calibri", bold: true })],
        bullet: { level: 0 },
      }));
    }
  }

  // Low confidence
  const lowConf = data.results.filter((r) => r.confidence === "low");
  if (lowConf.length > 0) {
    children.push(new Paragraph({ text: "Dimensions Requiring Deeper Investigation", heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 } }));
    children.push(new Paragraph({
      children: [new TextRun({ text: `${lowConf.length} dimension${lowConf.length > 1 ? "s" : ""} showed low confidence — scenario responses were inconsistent, suggesting the assigned level may not fully reflect institutional reality. Consider targeted follow-up with relevant stakeholders.`, size: 22, font: "Calibri" })],
      spacing: { after: 100 },
    }));
    for (const r of lowConf) {
      children.push(new Paragraph({
        children: [
          new TextRun({ text: `${r.dimension_name}`, size: 22, font: "Calibri", bold: true }),
          new TextRun({ text: ` — assigned ${r.assigned_level} but with divergent responses`, size: 22, font: "Calibri", color: "718096" }),
        ],
        bullet: { level: 0 },
      }));
    }
  }

  // Strengths (Optimised)
  const optimised = data.results.filter((r) => r.assigned_level_order === 4);
  if (optimised.length > 0) {
    children.push(new Paragraph({ text: "Strengths — Maintain and Share Best Practice", heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 } }));
    children.push(new Paragraph({
      children: [new TextRun({ text: `${optimised.length} dimension${optimised.length > 1 ? "s" : ""} scored at the Optimised level, indicating sector-leading practice. Consider documenting and sharing these approaches through case studies and peer networks.`, size: 22, font: "Calibri" })],
      spacing: { after: 100 },
    }));
    for (const r of optimised) {
      children.push(new Paragraph({
        children: [new TextRun({ text: `${r.dimension_name}`, size: 22, font: "Calibri", bold: true })],
        bullet: { level: 0 },
      }));
    }
  }

  // Intentional/Integrated growth areas
  const middle = data.results.filter((r) => r.assigned_level_order === 2 || r.assigned_level_order === 3);
  if (middle.length > 0) {
    children.push(new Paragraph({ text: "Growth Trajectory", heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 } }));
    children.push(new Paragraph({
      children: [new TextRun({ text: `${middle.length} dimensions are at Intentional or Integrated level — purposeful activity is underway. Focus on closing specific gaps to advance to the next maturity stage.`, size: 22, font: "Calibri" })],
      spacing: { after: 100 },
    }));
  }

  if (incidental.length === 0 && lowConf.length === 0 && optimised.length === 0 && middle.length === 0) {
    children.push(new Paragraph({
      children: [new TextRun({ text: "No specific recommendations generated — results are balanced across dimensions.", size: 22, font: "Calibri" })],
    }));
  }

  children.push(new Paragraph({ children: [new PageBreak()] }));
  return children;
}

function buildMethodology(data: TheReportData): Paragraph[] {
  return [
    new Paragraph({ text: "Methodology", heading: HeadingLevel.HEADING_1, spacing: { after: 200 } }),

    new Paragraph({ text: "Assessment Approach", heading: HeadingLevel.HEADING_2, spacing: { after: 100 } }),
    new Paragraph({
      children: [new TextRun({ text: "This assessment used a Situational Judgement Test (SJT) methodology. The respondent was presented with realistic institutional scenarios and asked to select the response that best reflects what they would most likely do. Response options were not labelled with maturity levels — the mapping was determined after selection based on validated level descriptors.", size: 20, font: "Calibri" })],
      spacing: { after: 200 },
    }),

    new Paragraph({ text: "Scoring Algorithm", heading: HeadingLevel.HEADING_2, spacing: { after: 100 } }),
    new Paragraph({
      children: [new TextRun({ text: "Each dimension was assessed through scenarios targeting specific maturity boundaries (e.g., Incidental → Intentional, Integrated → Optimised). The scoring algorithm evaluates each boundary independently:", size: 20, font: "Calibri" })],
      spacing: { after: 100 },
    }),
    new Paragraph({ children: [new TextRun({ text: "Pass: All responses at or above the boundary threshold — advance to next level", size: 20, font: "Calibri" })], bullet: { level: 0 } }),
    new Paragraph({ children: [new TextRun({ text: "Partial: Mixed responses at the boundary — remain at current level, confidence reduced", size: 20, font: "Calibri" })], bullet: { level: 0 } }),
    new Paragraph({ children: [new TextRun({ text: "Fail: Responses below the boundary — remain at current level", size: 20, font: "Calibri" })], bullet: { level: 0 } }),
    new Paragraph({ spacing: { after: 100 } }),
    new Paragraph({
      children: [new TextRun({ text: "The algorithm is conservative: it assigns the highest level for which all boundaries below are clearly passed, and stops at the first partial or failed boundary.", size: 20, font: "Calibri" })],
      spacing: { after: 200 },
    }),

    new Paragraph({ text: "Confidence Levels", heading: HeadingLevel.HEADING_2, spacing: { after: 100 } }),
    new Paragraph({ children: [new TextRun({ text: "High: All scenario responses for this dimension agree, pointing to a consistent maturity level", size: 20, font: "Calibri" })], bullet: { level: 0 } }),
    new Paragraph({ children: [new TextRun({ text: "Medium: Minor disagreement between responses (e.g., partial pass at one boundary)", size: 20, font: "Calibri" })], bullet: { level: 0 } }),
    new Paragraph({ children: [new TextRun({ text: "Low: Significant disagreement — responses are contradictory or skip boundaries. The assigned level may not fully reflect institutional reality.", size: 20, font: "Calibri" })], bullet: { level: 0 } }),
    new Paragraph({ spacing: { after: 200 } }),

    new Paragraph({ text: "Session Details", heading: HeadingLevel.HEADING_2, spacing: { after: 100 } }),
    new Paragraph({
      children: [new TextRun({ text: `Scenarios completed: ${data.scenarioCount}`, size: 20, font: "Calibri" })],
    }),
    new Paragraph({
      children: [new TextRun({ text: `Dimensions scored: ${data.results.length}`, size: 20, font: "Calibri" })],
    }),
    new Paragraph({
      children: [new TextRun({ text: `Assessment date: ${formatDate(data.completedAt)}`, size: 20, font: "Calibri" })],
    }),
    new Paragraph({
      children: [new TextRun({ text: `Session ID: ${data.sessionId}`, size: 20, font: "Calibri" })],
      spacing: { after: 200 },
    }),

    new Paragraph({ text: "About the THE Digital Maturity Index", heading: HeadingLevel.HEADING_2, spacing: { after: 100 } }),
    new Paragraph({
      children: [new TextRun({ text: "The THE Digital Maturity Index was developed by Times Higher Education through a quantitative survey of 3,863 respondents across 1,949 institutions in 100 countries (February–July 2024). It assesses digital maturity across 4 pillars (Teaching & Learning, Research, Professional Services, Planning & Governance) and 5 cross-cutting dimensions (Strategy, People & Culture, Technology, Data, Utilisation) at 4 maturity stages (Incidental, Intentional, Integrated, Optimised).", size: 20, font: "Calibri" })],
      spacing: { after: 100 },
    }),
    new Paragraph({
      children: [new TextRun({ text: "Scenarios in this assessment were created by ReasonLens based on the THE Digital Maturity Index framework.", size: 20, font: "Calibri", italics: true, color: "718096" })],
    }),
  ];
}

// ── Markdown-to-docx converter (for interpretive report) ──────────

function markdownToDocxParagraphs(md: string): Paragraph[] {
  const paragraphs: Paragraph[] = [];
  const lines = md.split("\n");
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Skip empty lines
    if (line.trim() === "") { i++; continue; }

    // Headings
    if (line.startsWith("### ")) {
      paragraphs.push(new Paragraph({
        children: [new TextRun({ text: line.slice(4).trim(), size: 24, bold: true, font: "Calibri", color: "2D3748" })],
        spacing: { before: 200, after: 100 },
      }));
      i++; continue;
    }
    if (line.startsWith("## ")) {
      paragraphs.push(new Paragraph({
        text: line.slice(3).trim(),
        heading: HeadingLevel.HEADING_2,
        spacing: { before: 300, after: 100 },
      }));
      i++; continue;
    }

    // Bullet points
    if (line.match(/^[-*]\s/)) {
      paragraphs.push(new Paragraph({
        children: parseInlineFormatting(line.replace(/^[-*]\s/, "").trim()),
        bullet: { level: 0 },
        spacing: { after: 60 },
      }));
      i++; continue;
    }

    // Numbered lists — render as indented paragraphs with number prefix
    const numMatch = line.match(/^(\d+)\.\s/);
    if (numMatch) {
      const numText = numMatch[1] + ". ";
      paragraphs.push(new Paragraph({
        children: [
          new TextRun({ text: numText, bold: true, size: 22, font: "Calibri" }),
          ...parseInlineFormatting(line.replace(/^\d+\.\s/, "").trim()),
        ],
        indent: { left: 360 },
        spacing: { after: 60 },
      }));
      i++; continue;
    }

    // Regular paragraph
    paragraphs.push(new Paragraph({
      children: parseInlineFormatting(line.trim()),
      spacing: { after: 100 },
    }));
    i++;
  }

  return paragraphs;
}

function parseInlineFormatting(text: string): TextRun[] {
  const runs: TextRun[] = [];
  // Match **bold**, *italic*, and plain text
  const regex = /(\*\*(.+?)\*\*|\*(.+?)\*|([^*]+))/g;
  let match;
  while ((match = regex.exec(text)) !== null) {
    if (match[2]) {
      runs.push(new TextRun({ text: match[2], bold: true, size: 22, font: "Calibri" }));
    } else if (match[3]) {
      runs.push(new TextRun({ text: match[3], italics: true, size: 22, font: "Calibri" }));
    } else if (match[4]) {
      runs.push(new TextRun({ text: match[4], size: 22, font: "Calibri" }));
    }
  }
  if (runs.length === 0) {
    runs.push(new TextRun({ text, size: 22, font: "Calibri" }));
  }
  return runs;
}

function buildInterpretiveReport(data: TheReportData): Paragraph[] {
  const ir = data.interpretiveReport;
  if (!ir) return [];

  const paragraphs: Paragraph[] = [];

  // Page break + Part 2 header
  paragraphs.push(new Paragraph({ children: [new PageBreak()] }));
  paragraphs.push(new Paragraph({
    text: "Part 2: AI-Powered Interpretive Analysis",
    heading: HeadingLevel.HEADING_1,
    spacing: { after: 200 },
  }));

  // AI disclaimer
  paragraphs.push(new Paragraph({
    children: [new TextRun({
      text: `The following interpretation was generated by ReasonLens AI based on your assessment results, institutional context, and the information you provided. It uses the ReasonLens Interpretive Methodology v${ir.methodology_version || "1.0"}.`,
      size: 20, font: "Calibri", italics: true, color: "718096",
    })],
    spacing: { after: 300 },
  }));

  // Executive Summary
  paragraphs.push(...markdownToDocxParagraphs(ir.executive_summary));
  paragraphs.push(new Paragraph({ children: [new PageBreak()] }));

  // Pillar analyses
  const pillarSections = [
    { content: ir.pillar_teaching_learning, label: "Teaching & Learning" },
    { content: ir.pillar_research, label: "Research" },
    { content: ir.pillar_professional_services, label: "Professional Services" },
    { content: ir.pillar_planning_governance, label: "Planning & Governance" },
  ];
  for (const ps of pillarSections) {
    paragraphs.push(...markdownToDocxParagraphs(ps.content));
    paragraphs.push(new Paragraph({ spacing: { after: 200 } }));
  }

  paragraphs.push(new Paragraph({ children: [new PageBreak()] }));

  // Recommendations
  paragraphs.push(...markdownToDocxParagraphs(ir.recommendations));

  return paragraphs;
}

// ── Main export ────────────────────────────────

export async function generateTheReport(data: TheReportData): Promise<void> {
  const framework = getFrameworkById("maturity-the");
  const pillared = resultsByPillar(data.results);

  const sections: Paragraph[] = [];

  // Cover page
  sections.push(...buildCoverPage(data));

  // Executive summary
  sections.push(...buildExecutiveSummary(data));

  // Results by pillar
  for (const pillar of pillared) {
    if (pillar.dimensions.length > 0) {
      sections.push(...buildPillarSection(pillar.name, pillar.dimensions, framework));
    }
  }

  // Heat map
  sections.push(...buildHeatMap(data));

  // AI interpretive report (Part 2) — inserted between data and template recommendations
  if (data.interpretiveReport) {
    sections.push(...buildInterpretiveReport(data));
  }

  // Recommendations
  sections.push(...buildRecommendations(data));

  // Methodology
  sections.push(...buildMethodology(data));

  const doc = new Document({
    styles: {
      default: {
        heading1: { run: { size: 32, bold: true, font: "Calibri", color: "1A365D" } },
        heading2: { run: { size: 26, bold: true, font: "Calibri", color: "2D3748" } },
        document: { run: { size: 22, font: "Calibri" } },
      },
    },
    sections: [{
      properties: {
        page: {
          margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 }, // ~2cm
        },
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            children: [new TextRun({ text: "THE Digital Maturity Index — Assessment Report", size: 16, font: "Calibri", color: "A0AEC0", italics: true })],
            alignment: AlignmentType.RIGHT,
          })],
        }),
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            children: [
              new TextRun({ text: "Page ", size: 16, font: "Calibri", color: "A0AEC0" }),
              new TextRun({ children: [PageNumber.CURRENT], size: 16, font: "Calibri", color: "A0AEC0" }),
              new TextRun({ text: "  |  Generated by ReasonLens", size: 16, font: "Calibri", color: "A0AEC0" }),
            ],
            alignment: AlignmentType.CENTER,
          })],
        }),
      },
      children: sections,
    }],
  });

  const blob = await Packer.toBlob(doc);
  const a = document.createElement("a");
  const url = URL.createObjectURL(blob);
  a.href = url;
  a.download = `THE-DMI-Report-${data.completedAt.toISOString().slice(0, 10)}.docx`;
  a.click();
  URL.revokeObjectURL(url);
}
