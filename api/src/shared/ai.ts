import { AzureOpenAI } from "openai";
import type {
  ChatCompletionMessageParam,
  ChatCompletionTool,
} from "openai/resources/chat/completions";

let client: AzureOpenAI | null = null;

function getClient(): AzureOpenAI {
  if (!client) {
    const endpoint = process.env.AZURE_OPENAI_ENDPOINT;
    const apiKey = process.env.AZURE_OPENAI_API_KEY;
    if (!endpoint || !apiKey) {
      throw new Error("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be configured");
    }
    client = new AzureOpenAI({
      endpoint,
      apiKey,
      apiVersion: "2024-12-01-preview",
    });
  }
  return client;
}

function getDeployment(): string {
  return process.env.AZURE_OPENAI_DEPLOYMENT || "gpt-5.2";
}

function buildMessages(
  systemPrompt: string,
  messages: Array<{ role: string; content: string }>
): ChatCompletionMessageParam[] {
  return [
    { role: "system", content: systemPrompt },
    ...messages.map((m) => ({
      role: m.role as "user" | "assistant",
      content: m.content,
    })),
  ];
}

// Non-streaming generation
export async function generateContent(
  systemPrompt: string,
  messages: Array<{ role: string; content: string }>
): Promise<string> {
  const result = await getClient().chat.completions.create({
    model: getDeployment(),
    messages: buildMessages(systemPrompt, messages),
  });
  return result.choices[0]?.message?.content || "";
}

// Streaming generation — returns an async iterable of text chunks
export async function* generateContentStream(
  systemPrompt: string,
  messages: Array<{ role: string; content: string }>
): AsyncGenerator<string> {
  const stream = await getClient().chat.completions.create({
    model: getDeployment(),
    messages: buildMessages(systemPrompt, messages),
    stream: true,
  });
  for await (const chunk of stream) {
    const text = chunk.choices[0]?.delta?.content;
    if (text) yield text;
  }
}

// Tool calling (structured output)
export async function generateWithTools(
  systemPrompt: string,
  messages: Array<{ role: string; content: string }>,
  tools: ChatCompletionTool[]
): Promise<Record<string, any> | null> {
  const result = await getClient().chat.completions.create({
    model: getDeployment(),
    messages: buildMessages(systemPrompt, messages),
    tools,
    tool_choice: "auto",
  });

  const toolCall = result.choices[0]?.message?.tool_calls?.[0] as any;
  if (toolCall?.function?.arguments) {
    return JSON.parse(toolCall.function.arguments);
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
