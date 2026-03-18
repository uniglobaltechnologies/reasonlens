import { useState, useRef, useEffect } from "react";
import { useLocation } from "react-router-dom";
import { MessageCircle, X, Send, Loader2, Plus, Bot, User } from "lucide-react";
import { apiStream } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
}

const PAGE_LABELS: Record<string, string> = {
  "/": "Hub",
  "/audit": "Test an AI Tool",
  "/audit/runs": "My Audit Runs",
  "/assess": "Assess Your AI Readiness",
  "/learning-path": "Learning Path",
  "/frameworks": "Framework Explorer",
  "/policy": "Policy Generator",
  "/evaluate": "Can AI Do This?",
  "/my-progress": "My Progress",
  "/portfolio": "Portfolio",
  "/badges": "Badges",
};

const SUGGESTED_PROMPTS: Record<string, string[]> = {
  "/": ["Where should I start?", "What's the difference between an audit and an assessment?"],
  "/assess": ["Which framework should I start with?", "What do the levels mean?"],
  "/learning-path": ["How do I action these recommendations this month?", "Which recommendation should I prioritise first?"],
  "/frameworks": ["How do the 22 frameworks relate to each other?", "Which framework suits an educator?"],
  "/policy": ["What policy type do I need?", "How is the policy grounded in evidence?"],
  "/evaluate": ["How do I interpret the feasibility score?", "What does 'augment' vs 'automate' mean?"],
  "/audit": ["What scenarios should I choose?", "What does PETRI test for?"],
};

// Pages where copilot doesn't add value — hide it
const HIDE_ON: string[] = ["/auth"];

function getPageLabel(pathname: string): string {
  if (PAGE_LABELS[pathname]) return PAGE_LABELS[pathname];
  if (pathname.startsWith("/audit/runs/")) return "Audit Run Detail";
  if (pathname.startsWith("/assess/")) return "Framework Assessment";
  if (pathname.startsWith("/learning-path/")) return "Learning Path";
  if (pathname.startsWith("/frameworks/")) return "Framework Detail";
  return "ReasonLens";
}

function getFrameworkId(pathname: string): string | null {
  const m = pathname.match(/^\/frameworks\/(.+)$/);
  return m ? m[1] : null;
}

function getSuggestions(pathname: string): string[] {
  if (SUGGESTED_PROMPTS[pathname]) return SUGGESTED_PROMPTS[pathname];
  if (pathname.startsWith("/frameworks/")) return ["Explain this framework's levels", "How does this compare to DigComp?"];
  if (pathname.startsWith("/assess/")) return ["What does each level mean?", "Which dimension should I focus on first?"];
  if (pathname.startsWith("/learning-path/")) return ["How should I tackle these actions in 30 days?", "Which actions need evidence in my portfolio?"];
  return [];
}

export default function Copilot() {
  const location = useLocation();
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const pathname = location.pathname;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (isOpen) inputRef.current?.focus();
  }, [isOpen]);

  // Reset conversation only when navigating to a different top-level section
  const section = pathname.split("/")[1] || "";
  useEffect(() => {
    setMessages([]);
  }, [section]);

  if (HIDE_ON.includes(pathname)) return null;

  const sendMessage = async (text: string) => {
    if (!text.trim() || isLoading) return;

    const userMsg: Message = { role: "user", content: text };
    const next = [...messages, userMsg];
    setMessages(next);
    setInput("");
    setIsLoading(true);

    // Placeholder for streaming response
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    await apiStream(
      "/copilot-chat",
      {
        messages: next,
        context: {
          page: getPageLabel(pathname),
          frameworkId: getFrameworkId(pathname),
        },
      },
      (chunk) => {
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: "assistant",
            content: updated[updated.length - 1].content + chunk,
          };
          return updated;
        });
      },
      () => setIsLoading(false),
      (_err: string) => {
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: "assistant",
            content: "Sorry, something went wrong. Please try again.",
          };
          return updated;
        });
        setIsLoading(false);
      }
    );
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(input);
  };

  const suggestions = getSuggestions(pathname);

  return (
    <>
      {/* Floating button */}
      <button
        onClick={() => setIsOpen((o) => !o)}
        className="fixed bottom-6 right-6 z-50 h-14 w-14 rounded-full bg-primary text-primary-foreground shadow-lg hover:bg-primary/90 transition-colors flex items-center justify-center"
        aria-label={isOpen ? "Close AI Copilot" : "Open AI Copilot"}
      >
        {isOpen ? <X className="h-6 w-6" /> : <MessageCircle className="h-6 w-6" />}
      </button>

      {/* Panel */}
      {isOpen && (
        <div className="fixed bottom-24 right-6 z-50 w-[360px] max-w-[calc(100vw-2rem)] rounded-xl border border-border bg-card shadow-2xl flex flex-col overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 bg-primary text-primary-foreground">
            <div className="flex items-center gap-2">
              <MessageCircle className="h-4 w-4" />
              <span className="text-sm font-semibold">AI Copilot</span>
              <span className="text-xs opacity-70">— {getPageLabel(pathname)}</span>
            </div>
            <button
              onClick={() => setMessages([])}
              className="opacity-70 hover:opacity-100 transition-opacity"
              title="New conversation"
            >
              <Plus className="h-4 w-4" />
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 h-[380px] overflow-y-auto p-4 space-y-4">
            {messages.length === 0 && (
              <div className="text-center py-4">
                <p className="text-sm text-muted-foreground mb-4">
                  Hi! I'm your AI literacy assistant. Ask me anything about frameworks, assessments, audits, or policies.
                </p>
                {suggestions.length > 0 && (
                  <div className="space-y-2">
                    {suggestions.map((s, i) => (
                      <button
                        key={i}
                        onClick={() => sendMessage(s)}
                        className="w-full text-left px-3 py-2 text-sm rounded-lg border border-border hover:bg-muted/50 transition-colors"
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            {messages.map((msg, i) => (
              <div key={i} className={`flex gap-2.5 ${msg.role === "user" ? "justify-end" : ""}`}>
                {msg.role === "assistant" && (
                  <div className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <Bot className="h-3.5 w-3.5 text-primary" />
                  </div>
                )}
                <div
                  className={`max-w-[80%] rounded-lg px-3 py-2 text-sm ${
                    msg.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted text-foreground"
                  }`}
                >
                  {msg.content ? (
                    <p className="whitespace-pre-wrap">{msg.content}</p>
                  ) : (
                    <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                  )}
                </div>
                {msg.role === "user" && (
                  <div className="w-6 h-6 rounded-full bg-muted flex items-center justify-center flex-shrink-0 mt-0.5">
                    <User className="h-3.5 w-3.5 text-muted-foreground" />
                  </div>
                )}
              </div>
            ))}
            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <form onSubmit={handleSubmit} className="px-4 py-3 border-t border-border flex gap-2">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question..."
              disabled={isLoading}
              className="flex-1 px-3 py-2 text-sm bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50"
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
      )}
    </>
  );
}
