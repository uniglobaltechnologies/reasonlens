import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, Lock, Loader2 } from "lucide-react";
import Header from "@/components/Header";
import { apiGet, isAuthenticated } from "@/lib/api";

interface Badge {
  id: string;
  name: string;
  description: string;
  icon: string;
  category: string;
  points: number;
  earned: boolean;
  earned_at: string | null;
}

const ICON_MAP: Record<string, string> = {
  Compass: "🧭",
  Sparkles: "✨",
  FlaskConical: "🧪",
  FileText: "📄",
  Flame: "🔥",
  Users: "👥",
  BookMarked: "📚",
  Library: "🏛️",
  Zap: "⚡",
  Crown: "👑",
  Heart: "❤️",
  UserCheck: "🤝",
};

export default function Badges() {
  const [badges, setBadges] = useState<Badge[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isAuthenticated()) {
      setLoading(false);
      setError("Please sign in to view your badges.");
      return;
    }

    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const res = await apiGet<{ badges: Badge[] }>("/check-badge-criteria");
        if (!cancelled) setBadges(res.badges);
      } catch (err: any) {
        if (!cancelled) setError(err?.message || "Failed to load badges.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void load();
    return () => { cancelled = true; };
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto px-4 sm:px-6 py-8 max-w-3xl">
        <Link to="/" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-8">
          <ArrowLeft className="h-4 w-4" />Back to Hub
        </Link>
        <h2 className="text-2xl font-bold text-foreground mb-6">Badges</h2>

        {loading && (
          <div className="p-6 rounded-xl border border-border bg-card flex items-center gap-3 mb-6">
            <Loader2 className="h-5 w-5 animate-spin text-primary" />
            <p className="text-sm text-muted-foreground">Loading badges...</p>
          </div>
        )}

        {error && (
          <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 mb-6">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
          {badges.map((b) => (
            <div key={b.id} className={`p-5 rounded-xl border border-border bg-card text-center ${b.earned ? "" : "opacity-50"}`}>
              <div className="text-3xl mb-2">{ICON_MAP[b.icon] || "🏅"}</div>
              <h4 className="font-semibold text-foreground text-sm">{b.name}</h4>
              <p className="text-xs text-muted-foreground mt-1">{b.description}</p>
              {b.earned ? (
                <p className="text-xs text-green-600 mt-2">Earned</p>
              ) : (
                <Lock className="h-3.5 w-3.5 text-muted-foreground mx-auto mt-2" />
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
