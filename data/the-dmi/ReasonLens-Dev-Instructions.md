# ReasonLens: Scenario-Based Assessment -- Developer Instructions

## Overview

This document covers everything needed to implement contextualised, scenario-based framework assessment in ReasonLens. It references two research data bundles (CFT pilot completed, THE DMI in progress) and documents all normalisation, recalibration, and codebase corrections required.

---

## 1. What Exists Today and What's Wrong With It

### Current Assessment Flow

Users pick a framework, see raw dimension/level/indicator text, select a level per dimension from a dropdown, and results are saved to `assessment_results`. Every downstream feature (learning paths, policy generator, copilot, progress dashboard) reads from this table.

### Problems This Work Fixes

**Assessment validity:** Self-selection against abstract descriptors captures self-perception, not competence. Scenario-based assessment measures behavioural alignment instead. The research data provides 20 validated scenarios for the CFT pilot, with THE DMI scenarios being produced separately.

**Context blindness:** The same indicator means different things for different practitioners. The research data includes a persona-context matrix defining which contextual factors matter and how they affect interpretation, plus context-tagged scenarios that can be filtered by user profile.

**Downstream contamination:** Invalid assessments produce invalid LLM outputs. Scenario-derived levels flowing into the same `assessment_results` table improve every downstream consumer without those consumers needing modification.

**Data model errors:** The BDC profile level structure in the codebase is wrong. The `levelToScore()` function fails for multiple framework vocabularies. The framework recommender enum is incomplete. These must be fixed regardless of whether scenario assessment ships.

---

## 2. Research Data Bundle (CFT Pilot)

Located in the zip file `reasonlens-research-bundle.zip`:

```
reasonlens-research/
  methodological-foundation.md              # Theory, DB schema, scoring, licensing, validation
  cft-pilot/
    README.md                               # Quality audit results + integration guide
    frameworks/teacher-competency/
      indicator-operationalisation.json      88KB   15 objects (5 dims x 3 levels)
      level-boundaries.json                  28KB   10 objects (5 dims x 2 boundaries)
      practitioner-language.json             38KB   15 objects (1 per indicator)
      scenarios.json                         55KB   20 objects (2 per boundary)
    cross-framework/
      mappings.json                          35KB   27 objects (14 DigComp + 12 BDC + 1 non-mapping)
    context/
      persona-context-matrix.json            20KB   8 objects (1 per contextual factor)
```

All files are valid JSON. All IDs match the codebase (`teacher-competency`, `tc-ethics`, `tc-et-d1`, etc.). Full quality audit results are in `cft-pilot/README.md`.

### THE DMI Bundle (Forthcoming)

Same structure, adapted for institutional maturity assessment. Key differences from CFT:

- 80 indicator operationalisation objects (20 child dimensions x 4 levels) instead of 15
- 60 level boundary objects (20 dims x 3 boundaries) instead of 10
- 120 scenarios (20 dims x 3 boundaries x 2 each) instead of 20
- Institutional framing throughout (scenarios describe what institutions do, not individuals)
- Additional fields: `cross_dimension_dependencies`, `respondent_visibility_requirements`
- Different context factors: `institution_size`, `funding_model`, `digital_infrastructure_baseline`, `respondent_role`, `respondent_institutional_visibility`

---

## 3. Codebase Fixes Required (Do These First)

These are independent of scenario assessment and should be done regardless.

### 3.1 BDC Profile Level Model (CRITICAL)

**Problem:** `api/src/shared/framework-context.ts` and `app/src/data/frameworks.ts` define all 7 BDC profiles with a 5-level organisational maturity model:

```
Approaching and Understanding -> Experimenting and Exploring -> Operational -> Embedded -> Optimised/Transformed
```

This is wrong. These levels come from JISC's institutional maturity models (Digital Assessment Maturity Model, Digital Transformation Maturity Model). The individual BDC Discovery Tool uses 3 levels:

```
Developing -> Capable -> Proficient
```

**Impact:**
- All 7 BDC profile definitions have wrong level structures (420 indicators assigned to wrong levels)
- `levelToScore()` in `policy-recommender.ts` cannot match any BDC level name (all default to score 2)
- The cross-framework mappings in the research bundle use the correct 3-level model, so they are inconsistent with the current codebase
- Any user who assesses against a BDC profile gets unreliable downstream results

**Files to change:**
- `api/src/shared/framework-context.ts` -- all 7 BDC profile dimension/level definitions
- `app/src/data/frameworks.ts` -- mirror the same corrections
- `api/src/functions/policy-recommender.ts` -- `levelToScore()` needs BDC entries

**What the correct structure looks like:**

Each BDC profile has 6 elements (dimensions), each with 3 levels:

| Level | Name | Description | Normalised Score |
|---|---|---|---|
| 1 | Developing | Most room for improvement; building awareness and early skills | 1.7 |
| 2 | Capable | Competent and confident; systematic practice | 3.3 |
| 3 | Proficient | Strong, advanced practice; leadership and innovation | 5.0 |

The indicators per level need to be redistributed. The current codebase has ~10 indicators per BDC dimension spread across 5 levels. These need to be consolidated into 3 levels. The research team's practitioner language mapping (forthcoming for BDC) will provide the correct indicator-to-level assignments, but in the interim a reasonable approach is:

- Current levels 1-2 (Approaching, Experimenting) collapse into Developing
- Current level 3 (Operational) maps to Capable
- Current levels 4-5 (Embedded, Optimised) collapse into Proficient

### 3.2 levelToScore() Comprehensive Mapping

**Problem:** The function in `api/src/functions/policy-recommender.ts` uses string matching that fails for multiple framework vocabularies. Anything it cannot match defaults to score 2.

**Current matching (incomplete):**

```typescript
// Only handles: emerging, developing, established, defined, advanced, embedded, leading, optimising
// Fails for: BDC levels, AILit, DEC, DigComp Highly Advanced, ISTE, OECD
```

**Required comprehensive mapping:**

```typescript
const LEVEL_SCORE_MAP: Record<string, number> = {
  // UNESCO CFT (3 levels)
  "acquire": 1.7,
  "deepen": 3.3,
  "create": 5.0,

  // UNESCO Student (3 levels)
  "foundational": 1.7,
  "intermediate": 3.3,
  "advanced": 5.0,

  // UNESCO Guidance / JISC maturity models (3 levels)
  "emerging": 1.7,
  "developing": 3.3,
  "established": 5.0,
  "mature": 5.0,

  // THE Digital Maturity (4 levels)
  "incidental": 1.25,
  "intentional": 2.5,
  "integrated": 3.75,
  "optimised": 5.0,

  // QS AI Capability (3 levels)
  "basic": 1.7,
  // "developing" already mapped above
  // "advanced" already mapped above

  // JISC Digital Maturity (3 levels, compound names)
  "emerging to established": 1.7,
  "established to enhanced": 3.3,
  "enhanced to mature": 5.0,

  // BDC Discovery Tool CORRECT 3-level model
  // "developing" already mapped above
  "capable": 3.3,
  "proficient": 5.0,

  // AILit (4 levels)
  "novice": 1.25,
  // "intermediate" already mapped above
  // "advanced" already mapped above
  "expert": 5.0,

  // DEC AI Literacy (4 levels)
  "awareness": 1.25,
  "exploration": 2.5,
  "practice": 3.75,
  "mastery": 5.0,

  // DigComp 3.0 (4 levels)
  // "basic" already mapped above
  // "intermediate" already mapped above
  // "advanced" already mapped above
  "highly advanced": 5.0,

  // BDC WRONG 5-level model (keep for backward compat until migration)
  "approaching and understanding": 1.0,
  "experimenting and exploring": 2.0,
  "operational": 3.0,
  "embedded": 4.0,
  "optimised/transformed": 5.0,
};

function levelToScore(level: string): number {
  const normalised = level.toLowerCase().trim();
  if (LEVEL_SCORE_MAP[normalised] !== undefined) {
    return LEVEL_SCORE_MAP[normalised];
  }
  // Partial matching as fallback
  for (const [key, score] of Object.entries(LEVEL_SCORE_MAP)) {
    if (normalised.includes(key) || key.includes(normalised)) {
      return score;
    }
  }
  // Unknown level -- log a warning, return middle score
  console.warn(`levelToScore: unrecognised level "${level}", defaulting to 2.5`);
  return 2.5;
}
```

Note: ISTE frameworks have a single "Proficient" level (met/not-met) and OECD has task readiness levels (High/Medium/Low/Variable), neither of which should feed into `levelToScore()` for policy recommendation. Add a guard:

```typescript
const NON_SCORING_FRAMEWORKS = new Set([
  'iste-students', 'iste-educators', 'iste-coaches', 'iste-leaders',
  'oecd-indicators'
]);
```

### 3.3 FRAMEWORK_NAMES_ENUM Completion

**File:** `api/src/shared/prompt-preamble.ts`

**Problem:** 18 entries, 22 frameworks exist. Missing 4:

```typescript
// Add these to FRAMEWORK_NAMES_ENUM:
"AI Literacy Framework (AILit)",
"DEC AI Literacy Framework",
"JISC AI Maturity Model",
"ISTE Standards for Coaches v4.02",
```

Also verify that every enum entry exactly matches the `name` field in `framework-context.ts`. Known mismatches:

| Enum value | framework-context.ts name | Match? |
|---|---|---|
| "BDC Individual Contributor Profile" | "JISC BDC Individual Framework" | NO |
| "BDC Digital Leader Profile" | "JISC BDC Digital Leader Profile" | Partial |

Align all enum values to exact `name` field values from `framework-context.ts`.

### 3.4 Product Name in Prompts

**File:** `api/src/shared/prompt-preamble.ts`

Replace "LearnAI Scope" with "ReasonLens" throughout `PLATFORM_PREAMBLE`.

### 3.5 learning-path-ai.ts framework_name Bug

**File:** `api/src/functions/learning-path-ai.ts`

**Problem:** Line uses `$2` for both `framework_id` and `framework_name`:

```typescript
// Current (wrong):
VALUES ($1, $2, $2, $3, $4, now())

// Fixed:
VALUES ($1, $2, $3, $4, $5, now())
```

Either pass the actual framework name or look it up from framework data.

---

## 4. New Database Tables

Full SQL is in `methodological-foundation.md` Section 2.1. Summary:

| Table | Purpose | Rows (CFT pilot) |
|---|---|---|
| `scenario_bank` | Scenario stems with context tags and boundary targeting | 20 |
| `scenario_responses` | Response options per scenario with level mappings | ~70 (3-4 per scenario) |
| `scenario_sessions` | Assessment session lifecycle per user | 1 per user per assessment |
| `scenario_answers` | Individual responses within a session | 20 per CFT session |
| `item_calibration` | Psychometric data per scenario (Phase 2, empty initially) | 0 initially |
| `user_assessment_context` | Onboarding context (subject, tools, etc.) | 1 per user |

Create as migration `db/006_scenario_assessment.sql`.

### Data Loading

The JSON research files load directly into these tables. The mapping is:

| JSON file | Target table | Loading logic |
|---|---|---|
| `scenarios.json` | `scenario_bank` + `scenario_responses` | Each scenario object becomes 1 `scenario_bank` row. Each `responses[]` item becomes 1 `scenario_responses` row. |
| `indicator-operationalisation.json` | Not stored in DB directly | Used by frontend for personalised view and by API for prompt enrichment. Store as static data in `api/src/shared/` and `app/src/data/`. |
| `level-boundaries.json` | Not stored in DB directly | Used by scoring logic and personalised "next level" descriptions. Same treatment. |
| `practitioner-language.json` | Not stored in DB directly | Used by frontend UI to replace raw indicator text. |
| `persona-context-matrix.json` | Informs `user_assessment_context` schema | Determines which fields to collect in onboarding. |
| `mappings.json` | Could be stored in a `framework_mappings` table or kept as static data | Used by copilot and progress dashboard for cross-referencing. |

### Write a seed script

```bash
# Example: load scenarios into DB from JSON
node db/seed-scenarios.js cft-pilot/frameworks/teacher-competency/scenarios.json
```

The script should:
1. Read the JSON array
2. For each scenario, INSERT into `scenario_bank` (stem, framework_id, dimension_id, boundary levels, context_tags)
3. For each response within the scenario, INSERT into `scenario_responses` (response text, mapped level, nuisance flag)
4. Be idempotent (ON CONFLICT DO UPDATE)

---

## 5. New API Endpoints

### 5.1 Onboarding Context

```
POST /user-assessment-context
Auth: JWT required
Body: {
  "subject_area": "english",
  "institution_level": "secondary",
  "current_ai_tools": ["ChatGPT", "Grammarly"],
  "primary_frustration": "Marking takes too long",
  "years_of_experience": "6-10",
  "management_responsibility": "none"
}
Returns: { "success": true }
```

Upserts into `user_assessment_context`. Fields come from the persona-context matrix (8 factors for individual frameworks, 7 different factors for institutional frameworks).

### 5.2 Scenario Session

```
POST /scenario-sessions
Auth: JWT required
Body: { "framework_id": "teacher-competency" }
Returns: {
  "session_id": "uuid",
  "scenarios": [ ...filtered scenario objects... ],
  "estimated_time_minutes": 15
}
```

Creates a `scenario_sessions` row. Fetches scenarios from `scenario_bank` filtered by the user's context tags from `user_assessment_context`. Returns the full scenario objects (stem + responses, but NOT the level mappings -- those stay server-side to prevent gaming).

**Important:** Do not send `maps_to_level`, `is_attractive_nuisance`, `nuisance_explanation`, or `discrimination_notes` to the frontend. The client should only see `scenario_id`, `scenario_stem`, and `responses[].{id, text}`.

### 5.3 Scenario Answer

```
POST /scenario-answers
Auth: JWT required
Body: {
  "session_id": "uuid",
  "scenario_id": "CFT-ETH-AD-01",
  "response_id": "C",
  "time_to_respond_seconds": 34
}
Returns: {
  "recorded": true,
  "remaining": 17
}
```

Looks up the response's `maps_to_level` server-side. Stores in `scenario_answers`. Does NOT return the mapped level to the user mid-assessment (to prevent them adjusting subsequent answers).

### 5.4 Session Completion

```
POST /scenario-sessions/:id/complete
Auth: JWT required
Returns: {
  "results": [
    {
      "dimension": "Ethics of AI",
      "dimension_id": "tc-ethics",
      "level": "Deepen",
      "confidence": "high",
      "scenario_agreement": "2/2 scenarios agreed"
    },
    ...
  ]
}
```

Aggregates all answers for the session. Scoring algorithm (Phase 1, deterministic):

```
For each dimension:
  1. Collect all scenario answers for this dimension
  2. Map each to its level
  3. If all agree: assign that level, confidence = "high"
  4. If disagree by 1 level: assign the LOWER level, confidence = "medium"
  5. If disagree by 2+ levels: assign the LOWER level, confidence = "low"
```

Then write to existing tables for backward compatibility:

```sql
-- Write to assessment_results (existing table, existing consumers)
INSERT INTO assessment_results (user_id, framework_id, framework_name, question_id, dimension, selected_level)
VALUES ($user_id, $framework_id, $framework_name, $session_id, $dimension, $derived_level);

-- Update framework_progress (existing table)
INSERT INTO framework_progress (user_id, framework_id, framework_name, progress, completed_items, total_items)
VALUES ($user_id, $framework_id, $name, 100, $num_dimensions, $num_dimensions)
ON CONFLICT (user_id, framework_id) DO UPDATE SET ...;
```

This means all existing downstream consumers (copilot, learning path, policy recommender, progress dashboard, badge system) work without modification. They read `assessment_results` and get scenario-derived levels instead of self-selected levels.

### 5.5 Backward Compatibility

The old self-selection flow should continue working in parallel. Some frameworks won't have scenario data for a while (or ever, for licensing reasons). The frontend should:

1. Check if scenarios exist for the framework: `GET /scenarios?framework_id=X`
2. If yes, offer scenario-based assessment
3. If no, fall back to self-selection with personalised descriptions from `practitioner-language.json`

The `assessment_results` table should gain a column to distinguish methods:

```sql
ALTER TABLE assessment_results ADD COLUMN assessment_method TEXT DEFAULT 'self_report'
  CHECK (assessment_method IN ('self_report', 'scenario'));
```

Downstream consumers can optionally use this to weight or flag results differently (e.g. copilot could say "Ethics: Deepen (scenario-assessed)" vs "Ethics: Deepen (self-reported)").

---

## 6. Frontend Changes

### 6.1 Onboarding Flow

Before first assessment, collect context via `POST /user-assessment-context`. Present as a brief "Help us personalise your experience" step. Fields depend on framework type:

**Individual frameworks (CFT, DigComp, BDC, etc.):**
- Subject area (dropdown)
- Institution level (dropdown)
- Institution type (dropdown)
- Region (dropdown)
- Current AI tools (free text, comma-separated)
- Primary frustration (free text)
- Years of experience (dropdown)
- Management responsibility (dropdown)

**Institutional frameworks (THE, JISC maturity, QS):**
- Institution size (dropdown)
- Institution type (dropdown)
- Region (dropdown)
- Funding model (dropdown)
- Respondent role (dropdown)
- Respondent institutional visibility (dropdown)
- Digital infrastructure baseline (dropdown)

### 6.2 Scenario Assessment UI

Replace the level-picker with a scenario presentation flow:

1. Show scenario stem (2-4 sentences describing a situation)
2. Show 3-4 response options as radio buttons or cards
3. User selects one, clicks Next
4. Repeat for all scenarios (20 for CFT, 40 for THE)
5. On completion, show results per dimension with confidence indicators

**Do not show:**
- Level labels on response options
- Which option is "best"
- Running scores during assessment
- Attractive nuisance flags

**Do show:**
- Progress indicator (scenario 5 of 20)
- Estimated time remaining
- The scenario stem prominently, responses below
- Shuffle response order per scenario (prevent position bias)

### 6.3 Personalised Framework View

After assessment, replace the raw framework display with personalised content from the research data:

For each assessed dimension, show:
- `practitioner-language.json` -> `plain_language_rewrite` instead of raw indicator text
- `indicator-operationalisation.json` -> `good_enough_guidance` for "what matters for your role"
- `level-boundaries.json` -> `threshold_evidence.minimum` for "to reach the next level, you need..."
- `indicator-operationalisation.json` -> `evidence_types` for "evidence you could collect"

### 6.4 Cross-Framework View

Use `mappings.json` to show connections on the progress dashboard:

- "Your CFT Ethics at Deepen maps to DigComp Safety at Intermediate (moderate confidence)"
- Only show `strong` and `moderate` confidence mappings
- Respect `direction`: if `unidirectional_source_implies_target`, only show the mapping when viewing the source framework
- Show the documented non-mapping for AI Pedagogy -> DigComp as a gap: "AI Pedagogy has no equivalent in DigComp 3.0 because DigComp is a citizen framework, not a teaching framework"

---

## 7. Prompt Updates

### 7.1 Copilot System Prompt

The research data enables three prompt improvements:

**Replace the all-frameworks dump.** Currently `getFrameworkContext()` injects all 22 frameworks (~15,000+ tokens) into every copilot message. Replace with a lightweight index (~500 tokens) plus full detail only for the active framework. New helper needed:

```typescript
export function getFrameworkIndex(): string {
  return frameworks
    .map((f, i) => `${i+1}. ${f.name} (${f.scope}, ${f.type}, audience: ${f.targetAudience.join("/")})`)
    .join("\n");
}
```

**Inject assessment confidence.** When building the copilot's personalisation context, include `assessment_method` from the new column:

```
ASSESSMENT PROFILE:
  Ethics of AI: Deepen (scenario-assessed, high confidence)
  AI Pedagogy: Acquire (self-reported)
```

This lets the copilot weight its recommendations appropriately.

**Inject observable behaviours.** For the active framework, load indicator operationalisation data and include `observable_behaviours` for the user's current and next levels. This gives the copilot concrete actions to recommend instead of generic advice.

### 7.2 Learning Path Prompt

Inject from indicator-operationalisation:
- `observable_behaviours` for the next level (what the user needs to demonstrate)
- `evidence_types` (what artifacts to produce)
- `prerequisites` (what to do first)
- `good_enough_guidance` (when to stop)

This transforms the learning path from "improve on Ethics" to "conduct a structured bias evaluation of one AI tool you currently use, documenting: what data it collects, whether outputs differ by student group, and whether it complies with your school's data policy. This produces a Tool Evaluation Report suitable for your portfolio."

### 7.3 Policy Generator

The research data doesn't directly change the policy generator, but the improved assessment accuracy from scenario-based assessment means the policy generator receives better grounding data. The `assessment_summary` should be enriched with confidence levels once scenario assessment is live.

---

## 8. Licensing Constraints (Read Before Building)

The methodological foundation document has the full analysis. Critical constraints:

| Framework | Can we build scenarios? | Can we use commercially? | Action needed |
|---|---|---|---|
| UNESCO (CFT, Student, Guidance) | Yes (CC BY-SA 3.0 IGO) | Yes | Attribution + ShareAlike |
| DigComp 3.0 | Yes (EU reuse policy) | Yes | Attribution + "modified from source" statement |
| JISC BDC | Permission-dependent (CC BY-NC-ND or BY-NC-SA) | No (NC clause) | Contact JISC for commercial reuse permission |
| ISTE | Permitted educational use | Requires licensing for commercial | Classify deployments; gate ISTE if commercial |
| OECD | Yes (CC BY 4.0) | Yes | Attribution |
| QS | Yes (CC BY-SA 4.0) | Yes | ShareAlike |
| AILit | Yes (CC BY-SA 4.0) | Yes | ShareAlike |
| DEC | No derivatives allowed | Likely not without permission | Do not modify DEC text; create scenarios from original research |
| THE | No open licence found | Uncertain | Contact THE before creating derivatives |

**For the CFT pilot:** No licensing issues. UNESCO CC BY-SA 3.0 IGO allows adaptation with attribution and share-alike.

**For the THE DMI:** Contact THE for explicit reuse terms before publishing scenario derivatives. The scenarios themselves are original ReasonLens content (not copied from THE), but they are derived from THE's framework structure, which may require permission depending on THE's interpretation.

**For BDC profiles:** If ReasonLens has any commercial pathway (paid tiers, institutional licensing, consultancy), JISC BDC content cannot be included without explicit permission due to the NC clause.

---

## 9. Implementation Order

1. **Codebase fixes** (Section 3): BDC levels, levelToScore(), enum, product name, learning path bug. These are independent and should be done first.

2. **Database migration** (Section 4): Create `db/006_scenario_assessment.sql` with the 6 new tables.

3. **Data loading**: Seed script to load CFT scenarios from the research bundle into `scenario_bank` and `scenario_responses`.

4. **API endpoints** (Section 5): Onboarding context, scenario sessions, answers, completion. The completion endpoint is the most complex (scoring algorithm + backward-compatible writes to `assessment_results`).

5. **Frontend**: Onboarding flow, scenario assessment UI, results display.

6. **Personalised framework view** (Section 6.3): Use research data to replace raw indicator text.

7. **Prompt updates** (Section 7): Copilot framework index, confidence injection, observable behaviour injection.

8. **Cross-framework view** (Section 6.4): Use mappings data on progress dashboard.

Items 1-4 can be built and tested without frontend changes. The old self-selection flow continues working throughout.
