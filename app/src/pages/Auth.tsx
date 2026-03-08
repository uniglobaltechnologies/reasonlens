import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ScanEye, ArrowLeft, Loader2 } from "lucide-react";
import { setToken, apiPost, ApiError } from "@/lib/api";

type Mode = "login" | "signup" | "set-password";

export default function Auth() {
  const navigate = useNavigate();
  const [mode, setMode] = useState<Mode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      let action: string;
      let body: Record<string, string>;

      if (mode === "signup") {
        action = "signup";
        body = { email, password, full_name: name };
      } else if (mode === "set-password") {
        action = "set-password";
        body = { email, password };
      } else {
        action = "login";
        body = { email, password };
      }

      const res = await apiPost<{ token: string; user: any }>(
        `/auth?action=${action}`,
        body
      );

      setToken(res.token);
      navigate("/");
    } catch (err: any) {
      // Handle pre-bcrypt accounts that need a password set
      if (err instanceof ApiError && err.status === 409) {
        try {
          const parsed = JSON.parse(err.message);
          if (parsed === "password_not_set" || err.message.includes("password_not_set")) {
            setMode("set-password");
            setPassword("");
            setError("This account needs a password. Please set one below.");
            return;
          }
        } catch {
          // Check raw message
          if (err.message.includes("password_not_set") || err.message.includes("needs a password")) {
            setMode("set-password");
            setPassword("");
            setError("This account needs a password. Please set one below.");
            return;
          }
        }
      }
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const title =
    mode === "signup" ? "Create Account" :
    mode === "set-password" ? "Set Password" :
    "Sign In";

  const buttonLabel =
    mode === "signup" ? "Create Account" :
    mode === "set-password" ? "Set Password" :
    "Sign In";

  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="w-full max-w-sm px-4">
        <Link to="/" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-8">
          <ArrowLeft className="h-4 w-4" />Back to Hub
        </Link>
        <div className="flex items-center gap-2 mb-8 justify-center">
          <ScanEye className="h-8 w-8 text-primary" />
          <h1 className="text-2xl font-bold text-foreground">ReasonLens</h1>
        </div>

        <div className="p-6 rounded-xl border border-border bg-card">
          <h2 className="text-lg font-semibold text-foreground mb-4 text-center">
            {title}
          </h2>

          {error && (
            <p className={`text-sm mb-4 text-center ${mode === "set-password" ? "text-amber-500" : "text-red-500"}`}>
              {error}
            </p>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {mode === "signup" && (
              <div>
                <label className="text-sm font-medium text-foreground block mb-1">Full Name</label>
                <input type="text" value={name} onChange={(e) => setName(e.target.value)} className="w-full px-3 py-2 text-sm bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50" />
              </div>
            )}
            <div>
              <label className="text-sm font-medium text-foreground block mb-1">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                disabled={mode === "set-password"}
                className="w-full px-3 py-2 text-sm bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-60"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-foreground block mb-1">
                {mode === "set-password" ? "New Password" : "Password"}
              </label>
              <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={8} className="w-full px-3 py-2 text-sm bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50" />
              {(mode === "signup" || mode === "set-password") && (
                <p className="text-xs text-muted-foreground mt-1">Minimum 8 characters</p>
              )}
            </div>
            <button type="submit" disabled={loading} className="w-full py-2.5 bg-primary text-primary-foreground text-sm font-medium rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50 inline-flex items-center justify-center gap-2">
              {loading && <Loader2 className="h-4 w-4 animate-spin" />}
              {buttonLabel}
            </button>
          </form>

          {mode !== "set-password" && (
            <p className="text-sm text-center text-muted-foreground mt-4">
              {mode === "signup" ? "Already have an account?" : "Don't have an account?"}{" "}
              <button onClick={() => { setMode(mode === "signup" ? "login" : "signup"); setError(""); }} className="text-primary hover:underline">
                {mode === "signup" ? "Sign in" : "Sign up"}
              </button>
            </p>
          )}

          {mode === "set-password" && (
            <p className="text-sm text-center text-muted-foreground mt-4">
              <button onClick={() => { setMode("login"); setError(""); setPassword(""); }} className="text-primary hover:underline">
                Back to sign in
              </button>
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
