import { useState, useEffect } from "react";
import { ScanEye, Lock } from "lucide-react";

const STORAGE_KEY = "rl_access_granted";

export default function PasswordGate({ children }: { children: React.ReactNode }) {
  const [unlocked, setUnlocked] = useState(false);
  const [input, setInput] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    if (sessionStorage.getItem(STORAGE_KEY) === "1") {
      setUnlocked(true);
    }
  }, []);

  if (unlocked) return <>{children}</>;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim().toLowerCase() === "earlyaccess") {
      sessionStorage.setItem(STORAGE_KEY, "1");
      setUnlocked(true);
    } else {
      setError("Incorrect password");
      setInput("");
    }
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center px-4">
      <div className="w-full max-w-sm space-y-6 text-center">
        <div className="flex justify-center">
          <ScanEye className="h-12 w-12 text-primary" />
        </div>
        <h1 className="text-2xl font-bold text-foreground">ReasonLens</h1>
        <p className="text-muted-foreground text-sm">
          Enter the access password to continue.
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          {error && (
            <p className="text-sm text-red-500">{error}</p>
          )}
          <div className="relative">
            <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <input
              type="password"
              value={input}
              onChange={(e) => { setInput(e.target.value); setError(""); }}
              placeholder="Access password"
              autoFocus
              className="w-full rounded-lg border border-border bg-card pl-10 pr-4 py-3 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
          <button
            type="submit"
            className="w-full rounded-lg bg-primary px-4 py-3 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            Enter
          </button>
        </form>
      </div>
    </div>
  );
}
