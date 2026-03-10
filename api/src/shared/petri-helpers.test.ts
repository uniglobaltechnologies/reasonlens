import { describe, it, expect } from "vitest";
import {
  parseScoresFromXml,
  parseScoresFromJson,
  parseScores,
  extractTextContent,
  getMessagesFromParsed,
  extractAssistantResponses,
  percentile,
  summarizeScores,
  isErrorTranscript,
  analyzeTranscriptsForErrors,
  TOXICITY_FLAG_THRESHOLD,
  TOXICITY_CRITICAL_THRESHOLD,
} from "./petri-helpers";

// --- parseScoresFromXml ---
describe("parseScoresFromXml", () => {
  it("parses valid XML scores", () => {
    const content = `Some text <scores>1. needs_attention: 3/10\n2. toxicity: 7/10</scores> more text`;
    const scores = parseScoresFromXml(content);
    expect(scores).toEqual({ needs_attention: 3, toxicity: 7 });
  });

  it("returns {} when no <scores> tag", () => {
    expect(parseScoresFromXml("no scores here")).toEqual({});
  });

  it("handles decimal scores", () => {
    const content = `<scores>1. bias: 4.5/10</scores>`;
    expect(parseScoresFromXml(content)).toEqual({ bias: 4.5 });
  });

  it("handles case-insensitive labels", () => {
    const content = `<scores>1. Toxicity: 5/10\n2. BIAS: 3/10</scores>`;
    const scores = parseScoresFromXml(content);
    expect(scores).toEqual({ toxicity: 5, bias: 3 });
  });

  it("handles scores without leading numbers", () => {
    const content = `<scores>toxicity: 2/10</scores>`;
    expect(parseScoresFromXml(content)).toEqual({ toxicity: 2 });
  });
});

// --- parseScoresFromJson ---
describe("parseScoresFromJson", () => {
  it("parses valid v2 JSON scores", () => {
    const content = JSON.stringify({
      metadata: { judge_output: { scores: { toxicity: 0.3, bias: 0.7, needs_attention: 2 } } },
    });
    const scores = parseScoresFromJson(content);
    expect(scores).toEqual({ toxicity: 0.3, bias: 0.7, needs_attention: 2 });
  });

  it("returns {} for invalid JSON", () => {
    expect(parseScoresFromJson("not json")).toEqual({});
  });

  it("returns {} when metadata.judge_output.scores is missing", () => {
    expect(parseScoresFromJson(JSON.stringify({ metadata: {} }))).toEqual({});
    expect(parseScoresFromJson(JSON.stringify({ other: "data" }))).toEqual({});
  });

  it("skips non-numeric string values", () => {
    const content = JSON.stringify({
      metadata: { judge_output: { scores: { toxicity: 0.5, label: "high", note: "text" } } },
    });
    // "high" and "text" → Number("high") = NaN, Number.isFinite(NaN) = false → skipped
    expect(parseScoresFromJson(content)).toEqual({ toxicity: 0.5 });
  });

  it("lowercases keys", () => {
    const content = JSON.stringify({
      metadata: { judge_output: { scores: { Toxicity: 1, BIAS: 2 } } },
    });
    expect(parseScoresFromJson(content)).toEqual({ toxicity: 1, bias: 2 });
  });
});

// --- parseScores ---
describe("parseScores", () => {
  it("tries JSON first", () => {
    const content = JSON.stringify({
      metadata: { judge_output: { scores: { toxicity: 0.5 } } },
    });
    expect(parseScores(content)).toEqual({ toxicity: 0.5 });
  });

  it("falls back to XML when JSON has no scores", () => {
    const content = `<scores>1. toxicity: 8/10</scores>`;
    expect(parseScores(content)).toEqual({ toxicity: 8 });
  });

  it("returns {} when neither format matches", () => {
    expect(parseScores("plain text with no scores")).toEqual({});
  });
});

// --- extractTextContent ---
describe("extractTextContent", () => {
  it("returns string as-is", () => {
    expect(extractTextContent("hello")).toBe("hello");
  });

  it("extracts .text property", () => {
    expect(extractTextContent({ text: "world" })).toBe("world");
  });

  it("recurses into .content", () => {
    expect(extractTextContent({ content: "nested" })).toBe("nested");
    expect(extractTextContent({ content: { text: "deep" } })).toBe("deep");
  });

  it("extracts .message property", () => {
    expect(extractTextContent({ message: "msg" })).toBe("msg");
  });

  it("joins arrays", () => {
    expect(extractTextContent(["a", "b", "c"])).toBe("a b c");
  });

  it("returns empty string for null/undefined", () => {
    expect(extractTextContent(null)).toBe("");
    expect(extractTextContent(undefined)).toBe("");
  });

  it("stringifies unknown objects", () => {
    expect(extractTextContent({ foo: "bar" })).toBe('{"foo":"bar"}');
  });
});

// --- getMessagesFromParsed ---
describe("getMessagesFromParsed", () => {
  it("prefers target_messages with assistant msgs", () => {
    const parsed = {
      target_messages: [{ role: "assistant", content: "hi" }],
      messages: [{ role: "user", content: "hello" }],
    };
    expect(getMessagesFromParsed(parsed)).toBe(parsed.target_messages);
  });

  it("falls back to messages when target_messages has no assistant", () => {
    const parsed = {
      target_messages: [{ role: "user", content: "hi" }],
      messages: [{ role: "assistant", content: "hello" }],
    };
    expect(getMessagesFromParsed(parsed)).toBe(parsed.messages);
  });

  it("checks metadata.target_messages", () => {
    const parsed = {
      metadata: { target_messages: [{ role: "assistant", content: "hi" }] },
    };
    expect(getMessagesFromParsed(parsed)).toBe(parsed.metadata.target_messages);
  });

  it("handles conversation.turns", () => {
    const parsed = { conversation: { turns: [{ role: "assistant" }] } };
    expect(getMessagesFromParsed(parsed)).toBe(parsed.conversation.turns);
  });

  it("handles top-level turns", () => {
    const parsed = { turns: [{ role: "user" }] };
    expect(getMessagesFromParsed(parsed)).toBe(parsed.turns);
  });

  it("returns null for null/non-object", () => {
    expect(getMessagesFromParsed(null)).toBeNull();
    expect(getMessagesFromParsed("string")).toBeNull();
    expect(getMessagesFromParsed(42)).toBeNull();
  });

  it("handles array input directly", () => {
    const arr = [{ role: "assistant", content: "hi" }];
    expect(getMessagesFromParsed(arr)).toBe(arr);
  });
});

// --- extractAssistantResponses ---
describe("extractAssistantResponses", () => {
  it("extracts v1 assistant role messages", () => {
    const transcripts = [{
      content: JSON.stringify({
        messages: [
          { role: "user", content: "hi" },
          { role: "assistant", content: "hello there" },
          { role: "user", content: "thanks" },
          { role: "assistant", content: "you're welcome" },
        ],
      }),
    }];
    const responses = extractAssistantResponses(transcripts);
    expect(responses).toEqual(["hello there", "you're welcome"]);
  });

  it("extracts v2 tool messages with <target_response> XML", () => {
    const transcripts = [{
      content: JSON.stringify({
        messages: [
          { role: "user", content: "hi" },
          { role: "tool", content: "Observation: <target_response>I can help with that</target_response>" },
          { role: "tool", content: "No target response here" },
          { role: "tool", content: "<target_response attr='x'>Second response</target_response>" },
        ],
      }),
    }];
    const responses = extractAssistantResponses(transcripts);
    expect(responses).toEqual(["I can help with that", "Second response"]);
  });

  it("extracts from target_messages fallback", () => {
    const transcripts = [{
      content: JSON.stringify({
        messages: [{ role: "user", content: "hi" }],
        target_messages: [
          { role: "user", content: "prompt" },
          { role: "assistant", content: "target reply" },
        ],
      }),
    }];
    const responses = extractAssistantResponses(transcripts);
    expect(responses).toContain("target reply");
  });

  it("skips empty/null content transcripts", () => {
    const transcripts = [
      { content: null },
      { content: "" },
      { content: undefined },
    ];
    expect(extractAssistantResponses(transcripts as any)).toEqual([]);
  });

  it("skips unparseable JSON content", () => {
    const transcripts = [{ content: "not json {{{" }];
    expect(extractAssistantResponses(transcripts)).toEqual([]);
  });

  it("skips empty assistant content", () => {
    const transcripts = [{
      content: JSON.stringify({
        messages: [
          { role: "assistant", content: "" },
          { role: "assistant", content: "   " },
        ],
      }),
    }];
    expect(extractAssistantResponses(transcripts)).toEqual([]);
  });
});

// --- percentile ---
describe("percentile", () => {
  it("returns 0 for empty array", () => {
    expect(percentile([], 95)).toBe(0);
  });

  it("computes p95 correctly", () => {
    const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    expect(percentile(values, 95)).toBe(10);
  });

  it("computes p50 (median) correctly", () => {
    expect(percentile([1, 2, 3, 4, 5], 50)).toBe(3);
  });

  it("handles single element", () => {
    expect(percentile([42], 95)).toBe(42);
  });
});

// --- summarizeScores ---
describe("summarizeScores", () => {
  it("computes avg, max, p95, flagged, critical", () => {
    const scores = [
      { toxicity: 0.3 },
      { toxicity: 0.6 },
      { toxicity: 0.9 },
    ];
    const { labelStats, summary } = summarizeScores(scores);
    expect(labelStats.toxicity.avg).toBeCloseTo(0.6);
    expect(labelStats.toxicity.max).toBe(0.9);
    expect(labelStats.toxicity.flagged).toBe(2); // 0.6 and 0.9 >= 0.5
    expect(labelStats.toxicity.critical).toBe(1); // 0.9 >= 0.8
    expect(summary?.label).toBe("toxicity");
  });

  it("uses toxicity as summary label when present", () => {
    const scores = [{ bias: 0.1, toxicity: 0.2 }];
    const { summary } = summarizeScores(scores);
    expect(summary?.label).toBe("toxicity");
  });

  it("uses first label when toxicity not present", () => {
    const scores = [{ bias: 0.3 }];
    const { summary } = summarizeScores(scores);
    expect(summary?.label).toBe("bias");
  });

  it("returns null summary for empty input", () => {
    const { summary } = summarizeScores([]);
    expect(summary).toBeNull();
  });
});

// --- isErrorTranscript ---
describe("isErrorTranscript", () => {
  it("detects ERROR saving transcript pattern", () => {
    const t = { content: "blah ERROR  Error  saving  transcript blah" };
    expect(isErrorTranscript(t).isError).toBe(true);
  });

  it("detects NoneType judge_output pattern", () => {
    const t = { content: "NoneType object has no attribute judge_output" };
    expect(isErrorTranscript(t).isError).toBe(true);
  });

  it("detects NoneType generic pattern", () => {
    const t = { content: "'NoneType' object has no attribute 'foo'" };
    expect(isErrorTranscript(t).isError).toBe(true);
  });

  it("detects transcript_hook.py pattern", () => {
    const t = { content: "File transcript_hook.py:42 error occurred" };
    expect(isErrorTranscript(t).isError).toBe(true);
  });

  it("detects Python traceback pattern", () => {
    const t = { content: "Traceback (most recent call last):\n  File..." };
    expect(isErrorTranscript(t).isError).toBe(true);
  });

  it("detects none/none with no judge scores", () => {
    const t = { content: "model: none/none output", judge_scores: {} };
    expect(isErrorTranscript(t).isError).toBe(true);
  });

  it("does not flag none/none when judge_scores present", () => {
    const t = { content: "model: none/none output", judge_scores: { toxicity: 1 } };
    expect(isErrorTranscript(t).isError).toBe(false);
  });

  it("returns isError: false for valid transcript", () => {
    const t = { content: "Normal conversation transcript", judge_scores: { toxicity: 3 } };
    expect(isErrorTranscript(t).isError).toBe(false);
  });
});

// --- analyzeTranscriptsForErrors ---
describe("analyzeTranscriptsForErrors", () => {
  it("returns allErrors for empty input", () => {
    const result = analyzeTranscriptsForErrors([]);
    expect(result.hasValidResults).toBe(false);
    expect(result.allErrors).toBe(true);
  });

  it("detects valid transcripts with judge_scores", () => {
    const transcripts = [
      { content: "valid", judge_scores: { toxicity: 1 } },
    ];
    const result = analyzeTranscriptsForErrors(transcripts);
    expect(result.hasValidResults).toBe(true);
    expect(result.allErrors).toBe(false);
  });

  it("detects valid transcripts with parseable scores in content", () => {
    const transcripts = [{
      content: JSON.stringify({ metadata: { judge_output: { scores: { toxicity: 2 } } } }),
      judge_scores: {},
    }];
    const result = analyzeTranscriptsForErrors(transcripts);
    expect(result.hasValidResults).toBe(true);
  });

  it("detects all-error transcripts", () => {
    const transcripts = [
      { content: "Traceback (most recent call last):\n  File..." },
    ];
    const result = analyzeTranscriptsForErrors(transcripts);
    expect(result.hasValidResults).toBe(false);
    expect(result.allErrors).toBe(true);
    expect(result.errorMessages.length).toBeGreaterThan(0);
  });
});
