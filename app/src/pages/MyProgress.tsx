import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, BarChart3, Award, FolderOpen, Loader2 } from "lucide-react";
import Header from "@/components/Header";
import { apiGet, isAuthenticated } from "@/lib/api";

interface ProgressData {
  assessmentCount: number;
  badgeCount: number;
  portfolioCount: number;
}

export default function MyProgress() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ProgressData>({
    assessmentCount: 0,
    badgeCount: 0,
    portfolioCount: 0,
  });

  useEffect(() => {
    if (!isAuthenticated()) {
      setLoading(false);
      setError("Please sign in to view your progress.");
      return;
    }

    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const res = await apiGet<ProgressData>("/user-progress");
        if (!cancelled) setData(res);
      } catch (err: any) {
        if (!cancelled) setError(err?.message || "Failed to load progress.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto px-4 sm:px-6 py-8 max-w-3xl">
        <Link to="/" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-8">
          <ArrowLeft className="h-4 w-4" />Back to Hub
        </Link>
        <h2 className="text-2xl font-bold text-foreground mb-6">My Progress</h2>

        {loading && (
          <div className="p-6 rounded-xl border border-border bg-card flex items-center gap-3 mb-6">
            <Loader2 className="h-5 w-5 animate-spin text-primary" />
            <p className="text-sm text-muted-foreground">Loading progress...</p>
          </div>
        )}

        {error && (
          <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 mb-6">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
          <div className="p-5 rounded-xl border border-border bg-card text-center">
            <BarChart3 className="h-8 w-8 text-primary mx-auto mb-2" />
            <p className="text-2xl font-bold text-foreground">{data.assessmentCount}</p>
            <p className="text-sm text-muted-foreground">Assessments</p>
          </div>
          <div className="p-5 rounded-xl border border-border bg-card text-center">
            <Award className="h-8 w-8 text-amber-500 mx-auto mb-2" />
            <p className="text-2xl font-bold text-foreground">{data.badgeCount}</p>
            <p className="text-sm text-muted-foreground">Badges</p>
          </div>
          <div className="p-5 rounded-xl border border-border bg-card text-center">
            <FolderOpen className="h-8 w-8 text-green-500 mx-auto mb-2" />
            <p className="text-2xl font-bold text-foreground">{data.portfolioCount}</p>
            <p className="text-sm text-muted-foreground">Portfolio Items</p>
          </div>
        </div>

        {!loading && !error && data.assessmentCount === 0 && data.badgeCount === 0 && data.portfolioCount === 0 && (
          <div className="p-8 rounded-xl border-2 border-dashed border-border text-center">
            <p className="text-muted-foreground mb-4">Complete an assessment to see your progress here.</p>
            <Link to="/assess" className="px-4 py-2 bg-primary text-primary-foreground text-sm font-medium rounded-lg hover:bg-primary/90 transition-colors">Start Assessment</Link>
          </div>
        )}
      </div>
    </div>
  );
}
