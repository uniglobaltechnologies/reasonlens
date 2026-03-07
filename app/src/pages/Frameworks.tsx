import { Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import Header from "@/components/Header";
import { FRAMEWORKS } from "@/data/frameworks";

export default function Frameworks() {
  const individual = FRAMEWORKS.filter((f) => f.scope === "individual" && f.showInDashboard);
  const institutional = FRAMEWORKS.filter((f) => f.scope === "institutional" && f.showInDashboard);
  const crossCutting = FRAMEWORKS.filter((f) => f.scope === "cross-cutting" && f.showInDashboard);

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto px-4 sm:px-6 py-8">
        <Link to="/" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-8">
          <ArrowLeft className="h-4 w-4" />Back to Hub
        </Link>

        <h2 className="text-2xl sm:text-3xl font-bold text-foreground mb-2">Explore Frameworks</h2>
        <p className="text-muted-foreground mb-8">Browse 22 international AI literacy and digital competence frameworks.</p>

        {[
          { title: "Individual Competency", frameworks: individual, color: "primary" },
          { title: "Institutional Maturity", frameworks: institutional, color: "accent" },
          { title: "Cross-Cutting", frameworks: crossCutting, color: "warning" },
        ].map((section) => (
          section.frameworks.length > 0 && (
            <div key={section.title} className="mb-10">
              <h3 className="text-lg font-semibold text-foreground mb-4">{section.title}</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {section.frameworks.map((fw) => (
                  <Link
                    key={fw.id}
                    to={`/frameworks/${fw.id}`}
                    className="p-5 rounded-xl border border-border bg-card hover:border-primary/50 hover:shadow-md hover:-translate-y-0.5 transition-all"
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-primary/10 text-primary">
                        {fw.source}
                      </span>
                      {fw.sourceFidelity === "official" && (
                        <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-green-500/10 text-green-600">Official</span>
                      )}
                    </div>
                    <h4 className="font-semibold text-foreground mb-1">{fw.shortName || fw.name}</h4>
                    <p className="text-sm text-muted-foreground line-clamp-2">{fw.description}</p>
                    <div className="mt-3 flex items-center gap-3 text-xs text-muted-foreground">
                      <span>{fw.keyDimensions.length} dimensions</span>
                      <span>·</span>
                      <span>{fw.targetAudience.slice(0, 2).join(", ")}</span>
                    </div>
                  </Link>
                ))}
              </div>
            </div>
          )
        ))}
      </div>
    </div>
  );
}
