import { describe, expect, it } from "vitest";
import { scoreSession, type ScenarioAnswer } from "./scenario-scoring";

describe("scoreSession", () => {
  it("keeps the conservative minimum scorer for non-THE frameworks", () => {
    const answers: ScenarioAnswer[] = [
      {
        scenario_id: "CFT-ETH-01",
        dimension_id: "tc-ethics",
        dimension_name: "Ethics of AI",
        mapped_level: "Acquire",
        level_order: 1,
      },
      {
        scenario_id: "CFT-ETH-02",
        dimension_id: "tc-ethics",
        dimension_name: "Ethics of AI",
        mapped_level: "Deepen",
        level_order: 2,
      },
    ];

    const [result] = scoreSession(answers, { frameworkId: "teacher-competency" });

    expect(result.assigned_level).toBe("Acquire");
    expect(result.assigned_level_order).toBe(1);
    expect(result.confidence).toBe("medium");
  });

  it("scores THE dimensions by contiguous boundary passes", () => {
    const answers: ScenarioAnswer[] = [
      {
        scenario_id: "THE-TLS-IN-01",
        dimension_id: "the-tl-strategy",
        dimension_name: "Teaching & Learning: Strategy",
        mapped_level: "Intentional",
        level_order: 2,
        target_boundary: "Incidental-Intentional",
      },
      {
        scenario_id: "THE-TLS-IN-02",
        dimension_id: "the-tl-strategy",
        dimension_name: "Teaching & Learning: Strategy",
        mapped_level: "Intentional",
        level_order: 2,
        target_boundary: "Incidental-Intentional",
      },
      {
        scenario_id: "THE-TLS-NI-01",
        dimension_id: "the-tl-strategy",
        dimension_name: "Teaching & Learning: Strategy",
        mapped_level: "Integrated",
        level_order: 3,
        target_boundary: "Intentional-Integrated",
      },
      {
        scenario_id: "THE-TLS-NI-02",
        dimension_id: "the-tl-strategy",
        dimension_name: "Teaching & Learning: Strategy",
        mapped_level: "Integrated",
        level_order: 3,
        target_boundary: "Intentional-Integrated",
      },
      {
        scenario_id: "THE-TLS-IO-01",
        dimension_id: "the-tl-strategy",
        dimension_name: "Teaching & Learning: Strategy",
        mapped_level: "Optimised",
        level_order: 4,
        target_boundary: "Integrated-Optimised",
      },
      {
        scenario_id: "THE-TLS-IO-02",
        dimension_id: "the-tl-strategy",
        dimension_name: "Teaching & Learning: Strategy",
        mapped_level: "Optimised",
        level_order: 4,
        target_boundary: "Integrated-Optimised",
      },
    ];

    const [result] = scoreSession(answers, { frameworkId: "maturity-the" });

    expect(result.assigned_level).toBe("Optimised");
    expect(result.assigned_level_order).toBe(4);
    expect(result.confidence).toBe("high");
  });

  it("downgrades THE confidence for below-boundary contradictions and keeps the lower level", () => {
    const answers: ScenarioAnswer[] = [
      {
        scenario_id: "THE-TLS-IN-01",
        dimension_id: "the-tl-strategy",
        dimension_name: "Teaching & Learning: Strategy",
        mapped_level: "Intentional",
        level_order: 2,
        target_boundary: "Incidental-Intentional",
      },
      {
        scenario_id: "THE-TLS-IN-02",
        dimension_id: "the-tl-strategy",
        dimension_name: "Teaching & Learning: Strategy",
        mapped_level: "Intentional",
        level_order: 2,
        target_boundary: "Incidental-Intentional",
      },
      {
        scenario_id: "THE-TLS-NI-01",
        dimension_id: "the-tl-strategy",
        dimension_name: "Teaching & Learning: Strategy",
        mapped_level: "Integrated",
        level_order: 3,
        target_boundary: "Intentional-Integrated",
      },
      {
        scenario_id: "THE-TLS-NI-02",
        dimension_id: "the-tl-strategy",
        dimension_name: "Teaching & Learning: Strategy",
        mapped_level: "Incidental",
        level_order: 1,
        target_boundary: "Intentional-Integrated",
      },
      {
        scenario_id: "THE-TLS-IO-01",
        dimension_id: "the-tl-strategy",
        dimension_name: "Teaching & Learning: Strategy",
        mapped_level: "Optimised",
        level_order: 4,
        target_boundary: "Integrated-Optimised",
      },
      {
        scenario_id: "THE-TLS-IO-02",
        dimension_id: "the-tl-strategy",
        dimension_name: "Teaching & Learning: Strategy",
        mapped_level: "Optimised",
        level_order: 4,
        target_boundary: "Integrated-Optimised",
      },
    ];

    const [result] = scoreSession(answers, { frameworkId: "maturity-the" });

    expect(result.assigned_level).toBe("Intentional");
    expect(result.assigned_level_order).toBe(2);
    expect(result.confidence).toBe("low");
  });
});
