import { Link } from "react-router-dom";
import { ArrowLeft, Award, Lock } from "lucide-react";
import Header from "@/components/Header";

const badges = [
  { id: "framework_explorer", name: "Framework Explorer", desc: "Complete your first assessment", icon: "🧭", earned: false },
  { id: "creator", name: "Creator", desc: "Achieve Create level in any dimension", icon: "✨", earned: false },
  { id: "lab_rat", name: "Lab Rat", desc: "Complete your first practice lab", icon: "🧪", earned: false },
  { id: "documenter", name: "Documenter", desc: "Add 5 portfolio items", icon: "📄", earned: false },
  { id: "committed", name: "Committed", desc: "Log in for 7 consecutive days", icon: "🔥", earned: false },
  { id: "collaborator", name: "Collaborator", desc: "Share 3 portfolio items", icon: "👥", earned: false },
];

export default function Badges() {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto px-4 sm:px-6 py-8 max-w-3xl">
        <Link to="/" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-8">
          <ArrowLeft className="h-4 w-4" />Back to Hub
        </Link>
        <h2 className="text-2xl font-bold text-foreground mb-6">Badges</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
          {badges.map((b) => (
            <div key={b.id} className={`p-5 rounded-xl border border-border bg-card text-center ${b.earned ? "" : "opacity-50"}`}>
              <div className="text-3xl mb-2">{b.icon}</div>
              <h4 className="font-semibold text-foreground text-sm">{b.name}</h4>
              <p className="text-xs text-muted-foreground mt-1">{b.desc}</p>
              {!b.earned && <Lock className="h-3.5 w-3.5 text-muted-foreground mx-auto mt-2" />}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
