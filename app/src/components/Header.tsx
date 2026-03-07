import { Link } from "react-router-dom";
import { ScanEye, LogIn, LogOut, BarChart3, FolderOpen, Award } from "lucide-react";
import { isAuthenticated, clearToken } from "@/lib/api";
import ThemeToggle from "./ThemeToggle";

export default function Header() {
  const authed = isAuthenticated();

  const handleLogout = () => {
    clearToken();
    window.location.href = "/";
  };

  return (
    <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4 sm:px-6 py-3 sm:py-4 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2">
          <ScanEye className="h-6 w-6 text-primary" />
          <div>
            <h1 className="text-xl font-bold text-foreground">ReasonLens</h1>
            <p className="text-xs text-muted-foreground hidden sm:block">
              Clarity Through Ethical AI Evaluation
            </p>
          </div>
        </Link>
        <div className="flex items-center gap-2 sm:gap-3">
          {authed && (
            <>
              <Link
                to="/my-progress"
                className="hidden sm:inline-flex items-center gap-1.5 px-3 py-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                <BarChart3 className="h-4 w-4" />
                Progress
              </Link>
              <Link
                to="/portfolio"
                className="hidden sm:inline-flex items-center gap-1.5 px-3 py-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                <FolderOpen className="h-4 w-4" />
                Portfolio
              </Link>
              <Link
                to="/badges"
                className="hidden sm:inline-flex items-center gap-1.5 px-3 py-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                <Award className="h-4 w-4" />
                Badges
              </Link>
              <button
                onClick={handleLogout}
                className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg border border-border hover:bg-muted transition-colors"
              >
                <LogOut className="h-4 w-4" />
                <span className="hidden sm:inline">Sign Out</span>
              </button>
            </>
          )}
          {!authed && (
            <Link
              to="/auth"
              className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg border border-border hover:bg-muted transition-colors"
            >
              <LogIn className="h-4 w-4" />
              <span className="hidden sm:inline">Sign In</span>
            </Link>
          )}
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
