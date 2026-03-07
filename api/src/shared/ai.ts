import {
  GoogleGenerativeAI,
  Content,
  FunctionDeclaration,
  GenerateContentResult,
  Tool,
} from "@google/generative-ai";

let genAI: GoogleGenerativeAI | null = null;

function getClient(): GoogleGenerativeAI {
  if (!genAI) {
    const key = process.env.GOOGLE_AI_API_KEY;
    if (!key) throw new Error("GOOGLE_AI_API_KEY not configured");
    genAI = new GoogleGenerativeAI(key);
  }
  return genAI;
}

// Convert OpenAI-style messages to Google AI format
function toGoogleContents(
  messages: Array<{ role: string; content: string }>
): Content[] {
  return messages
    .filter((m) => m.role !== "system")
    .map((m) => ({
      role: m.role === "assistant" ? "model" : "user",
      parts: [{ text: m.content }],
    }));
}

// Non-streaming generation
export async function generateContent(
  modelName: string,
  systemPrompt: string,
  messages: Array<{ role: string; content: string }>
): Promise<string> {
  const model = getClient().getGenerativeModel({
    model: modelName,
    systemInstruction: systemPrompt,
  });
  const result = await model.generateContent({
    contents: toGoogleContents(messages),
  });
  return result.response.text();
}

// Streaming generation — returns an async iterable of text chunks
export async function* generateContentStream(
  modelName: string,
  systemPrompt: string,
  messages: Array<{ role: string; content: string }>
): AsyncGenerator<string> {
  const model = getClient().getGenerativeModel({
    model: modelName,
    systemInstruction: systemPrompt,
  });
  const result = await model.generateContentStream({
    contents: toGoogleContents(messages),
  });
  for await (const chunk of result.stream) {
    const text = chunk.text();
    if (text) yield text;
  }
}

// Tool calling (structured output)
export async function generateWithTools(
  modelName: string,
  systemPrompt: string,
  messages: Array<{ role: string; content: string }>,
  functionDeclarations: FunctionDeclaration[]
): Promise<Record<string, any> | null> {
  const model = getClient().getGenerativeModel({
    model: modelName,
    systemInstruction: systemPrompt,
    tools: [{ functionDeclarations }] as Tool[],
  });

  const result: GenerateContentResult = await model.generateContent({
    contents: toGoogleContents(messages),
  });

  const candidate = result.response.candidates?.[0];
  const functionCall = candidate?.content?.parts?.find(
    (p) => "functionCall" in p
  );

  if (functionCall && "functionCall" in functionCall) {
    return (functionCall as any).functionCall.args as Record<string, any>;
  }
  return null;
}

// Create SSE response from streaming generator
export function createSSEResponse(
  stream: AsyncGenerator<string>
): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    async start(controller) {
      try {
        for await (const chunk of stream) {
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ content: chunk })}\n\n`)
          );
        }
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      } catch (err) {
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({ error: String(err) })}\n\n`
          )
        );
      } finally {
        controller.close();
      }
    },
  });
}
