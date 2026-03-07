import { useState } from "react";
import { Link } from "react-router-dom";
import { MessageSquare, Settings2, ArrowLeft, History } from "lucide-react";
import Header from "@/components/Header";
import SimpleAuditChat from "@/components/audit/SimpleAuditChat";
import ProAuditWizard from "@/components/audit/ProAuditWizard";

type AuditMode = "simple" | "pro";

export default function Audit() {
  const [mode, setMode] = useState<AuditMode>(() => {
    const stored = localStorage.getItem("audit-mode-preference");
    return stored === "pro" ? "pro" : "simple";
  });

  const selectMode = (m: AuditMode) => {
    setMode(m);
    localStorage.setItem("audit-mode-preference", m);
  };

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto px-4 sm:px-6 py-8">
        {/* Top nav */}
        <div className="flex items-center justify-between mb-8">
          <Link
            to="/"
            className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Hub
          </Link>
          <Link
            to="/audit/runs"
            className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <History className="h-4 w-4" />
            My Audit Runs
          </Link>
        </div>

        <h2 className="text-2xl sm:text-3xl font-bold text-foreground mb-2">
          Test an AI Tool
        </h2>
        <p className="text-muted-foreground mb-8">
          Run automated safety audits on AI tools. Choose Simple mode for a
          guided experience, or Pro mode for full control.
        </p>

        {/* Mode Toggle */}
        <div className="grid grid-cols-2 gap-4 max-w-md mb-8">
          <button
            onClick={() => selectMode("simple")}
            className={`p-4 rounded-xl border-2 text-left transition-all ${
              mode === "simple"
                ? "border-primary bg-primary/5 shadow-md"
                : "border-border hover:border-primary/50 hover:bg-muted/50"
            }`}
          >
            <div
              className={`w-10 h-10 rounded-full flex items-center justify-center mb-2 ${
                mode === "simple"
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-muted-foreground"
              }`}
            >
              <MessageSquare className="h-5 w-5" />
            </div>
            <h4 className="font-semibold text-foreground">Simple</h4>
            <p className="text-xs text-muted-foreground">
              Describe your use case and we'll set up the audit
            </p>
          </button>
          <button
            onClick={() => selectMode("pro")}
            className={`p-4 rounded-xl border-2 text-left transition-all ${
              mode === "pro"
                ? "border-primary bg-primary/5 shadow-md"
                : "border-border hover:border-primary/50 hover:bg-muted/50"
            }`}
          >
            <div
              className={`w-10 h-10 rounded-full flex items-center justify-center mb-2 ${
                mode === "pro"
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-muted-foreground"
              }`}
            >
              <Settings2 className="h-5 w-5" />
            </div>
            <h4 className="font-semibold text-foreground">Pro</h4>
            <p className="text-xs text-muted-foreground">
              Select scenarios, models, and parameters manually
            </p>
          </button>
        </div>

        {/* Mode Content */}
        {mode === "simple" ? <SimpleAuditChat /> : <ProAuditWizard />}
      </div>
    </div>
  );
}
