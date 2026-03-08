import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Send,
  Loader2,
  Bot,
  User,
  Play,
  CheckCircle2,
  Sparkles,
  GraduationCap,
  MessageSquare,
} from "lucide-react";
import { apiPost, isAuthenticated } from "@/lib/api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  extractedConfig?: ExtractedConfig;
}

interface ExtractedConfig {
  use_case: string;
  subject?: string;
  level?: string;
  target_model_hint?: string;
  suggested_config?: {
    scenario_pack: string;
    target_model: string;
    auditor_model: string;
    judge_model: string;
    max_turns: number;
    posthoc_packs: string[];
  };
  confirmation_message?: string;
  ready_to_run?: boolean;
}

const suggestedPrompts = [
  {
    icon: GraduationCap,
    text: "I'm a Year 10 maths teacher and want to use ChatGPT as a homework helper",
  },
  {
    icon: MessageSquare,
    text: "Test if Gemini is safe for university students writing essays",
  },
  {
    icon: Sparkles,
    text: "Evaluate Claude for primary school science lessons",
  },
];

export default function SimpleAuditChat() {
  const navigate = useNavigate();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Hi! I'm here to help you test your AI tool for education. Tell me what AI you want to test and how you plan to use it with your students. For example:\n\n\"I'm a Year 10 maths teacher and want to use ChatGPT as a homework helper\"",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [extractedConfig, setExtractedConfig] = useState<ExtractedConfig | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    if (!isAuthenticated()) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: "Please sign in first to run audits.",
          timestamp: new Date(),
        },
      ]);
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const result = await apiPost<{
        response: string;
        confirmation_message?: string;
        ready_to_run?: boolean;
        extracted: ExtractedConfig;
        suggested_config?: any;
      }>("/parse-audit-intent", { message: userMessage.content });

      const config: ExtractedConfig = {
        ...result.extracted,
        suggested_config: result.suggested_config,
        confirmation_message: result.confirmation_message,
        ready_to_run: result.ready_to_run,
      };

      setExtractedConfig(config);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: result.response,
        timestamp: new Date(),
        extractedConfig: config,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: `Sorry, something went wrong: ${err.message}`,
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunAudit = async () => {
    if (!extractedConfig?.suggested_config || isRunning) return;
    if (!isAuthenticated()) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: "Please sign in first to run audits.",
          timestamp: new Date(),
        },
      ]);
      return;
    }
    setIsRunning(true);

    try {
      const config = extractedConfig.suggested_config;
      const result = await apiPost<{ run_id: string }>("/run-petri-audit", {
        scenario_pack: config.scenario_pack,
        auditor_model: config.auditor_model,
        target_model: config.target_model,
        judge_model: config.judge_model,
        max_turns: config.max_turns,
        posthoc_packs: config.posthoc_packs,
      });

      navigate(`/audit/runs/${result.run_id}`);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: `Failed to start audit: ${err.message}`,
          timestamp: new Date(),
        },
      ]);
      setIsRunning(false);
    }
  };

  const sendDirectMessage = async (text: string) => {
    if (!text.trim() || isLoading) return;
    if (!isAuthenticated()) {
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "assistant", content: "Please sign in first to run audits.", timestamp: new Date() },
      ]);
      return;
    }

    const userMessage: Message = { id: crypto.randomUUID(), role: "user", content: text.trim(), timestamp: new Date() };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const result = await apiPost<{
        response: string; confirmation_message?: string; ready_to_run?: boolean; extracted: ExtractedConfig; suggested_config?: any;
      }>("/parse-audit-intent", { message: userMessage.content });

      const config: ExtractedConfig = { ...result.extracted, suggested_config: result.suggested_config, confirmation_message: result.confirmation_message, ready_to_run: result.ready_to_run };
      setExtractedConfig(config);

      setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "assistant", content: result.response, timestamp: new Date(), extractedConfig: config }]);
    } catch (err: any) {
      setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "assistant", content: `Sorry, something went wrong: ${err.message}`, timestamp: new Date() }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-2xl">
      {/* Chat area */}
      <div className="rounded-xl border border-border bg-card overflow-hidden">
        {/* Header */}
        <div className="px-4 py-3 border-b border-border flex items-center gap-2">
          <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
            <Bot className="h-4 w-4 text-primary-foreground" />
          </div>
          <div>
            <p className="text-sm font-medium text-foreground">ReasonLens Audit Assistant</p>
            <p className="text-xs text-muted-foreground">Describe your AI use case</p>
          </div>
        </div>

        {/* Messages */}
        <div className="h-[400px] overflow-y-auto p-4 space-y-4">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3 ${msg.role === "user" ? "justify-end" : ""}`}
            >
              {msg.role === "assistant" && (
                <div className="w-7 h-7 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <Bot className="h-3.5 w-3.5 text-primary" />
                </div>
              )}
              <div
                className={`max-w-[80%] rounded-lg px-4 py-2.5 text-sm ${
                  msg.role === "user"
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted text-foreground"
                }`}
              >
                <p className="whitespace-pre-wrap">{msg.content}</p>
              </div>
              {msg.role === "user" && (
                <div className="w-7 h-7 rounded-full bg-muted flex items-center justify-center flex-shrink-0 mt-0.5">
                  <User className="h-3.5 w-3.5 text-muted-foreground" />
                </div>
              )}
            </div>
          ))}

          {isLoading && (
            <div className="flex gap-3">
              <div className="w-7 h-7 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                <Bot className="h-3.5 w-3.5 text-primary" />
              </div>
              <div className="bg-muted rounded-lg px-4 py-2.5">
                <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Ready to run banner */}
        {extractedConfig?.ready_to_run && (
          <div className="px-4 py-3 bg-green-500/10 border-t border-green-500/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-green-500" />
                <span className="text-sm font-medium text-foreground">
                  Ready to run audit
                </span>
              </div>
              <button
                onClick={handleRunAudit}
                disabled={isRunning}
                className="inline-flex items-center gap-2 px-4 py-2 bg-green-600 text-white text-sm font-medium rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50"
              >
                {isRunning ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
                {isRunning ? "Starting..." : "Run Audit"}
              </button>
            </div>
          </div>
        )}

        {/* Suggested prompts (show only if no user messages yet) */}
        {messages.length === 1 && (
          <div className="px-4 py-3 border-t border-border space-y-2">
            <p className="text-xs text-muted-foreground">Try one of these:</p>
            {suggestedPrompts.map((prompt, i) => {
              const Icon = prompt.icon;
              return (
                <button
                  key={i}
                  onClick={() => sendDirectMessage(prompt.text)}
                  className="w-full text-left px-3 py-2 text-sm rounded-lg border border-border hover:bg-muted/50 transition-colors flex items-center gap-2"
                >
                  <Icon className="h-4 w-4 text-primary flex-shrink-0" />
                  {prompt.text}
                </button>
              );
            })}
          </div>
        )}

        {/* Input */}
        <form
          onSubmit={handleSubmit}
          className="px-4 py-3 border-t border-border flex gap-2"
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Describe the AI tool and how you want to use it..."
            className="flex-1 px-3 py-2 text-sm bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="px-3 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50"
          >
            <Send className="h-4 w-4" />
          </button>
        </form>
      </div>
    </div>
  );
}
