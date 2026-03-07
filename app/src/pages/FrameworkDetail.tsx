import { useParams, Link } from "react-router-dom";
import { ArrowLeft, Play } from "lucide-react";
import Header from "@/components/Header";
import { getFrameworkById } from "@/data/frameworks";

export default function FrameworkDetail() {
  const { id } = useParams<{ id: string }>();
  const fw = id ? getFrameworkById(id) : undefined;

  if (!fw) {
    return (
      <div className="min-h-screen bg-background">
        <Header />
        <div className="container mx-auto px-4 py-12 text-center">
          <p className="text-muted-foreground">Framework not found.</p>
          <Link to="/frameworks" className="text-primary hover:underline mt-4 inline-block">Browse frameworks</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto px-4 sm:px-6 py-8 max-w-4xl">
        <Link to="/frameworks" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-6">
          <ArrowLeft className="h-4 w-4" />All Frameworks
        </Link>

        <div className="flex items-center gap-2 mb-2">
          <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-primary/10 text-primary">{fw.source}</span>
          <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-muted text-muted-foreground">{fw.type}</span>
          <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-muted text-muted-foreground">{fw.scope}</span>
        </div>

        <h2 className="text-2xl sm:text-3xl font-bold text-foreground mb-3">{fw.name}</h2>
        <p className="text-muted-foreground mb-6">{fw.overview || fw.description}</p>

        <Link
          to={`/assess/${fw.id}`}
          className="inline-flex items-center gap-2 px-5 py-2.5 bg-primary text-primary-foreground text-sm font-medium rounded-lg hover:bg-primary/90 transition-colors mb-8"
        >
          <Play className="h-4 w-4" />
          Start Assessment
        </Link>

        <h3 className="text-lg font-semibold text-foreground mb-4">Dimensions</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8">
          {fw.keyDimensions.map((dim) => (
            <div key={dim.id} className="p-4 rounded-xl border border-border bg-card">
              <h4 className="font-semibold text-foreground mb-1">{dim.name}</h4>
              <p className="text-sm text-muted-foreground mb-3">{dim.description}</p>
              {dim.levels.length > 0 && (
                <div className="space-y-1">
                  {dim.levels.map((level) => (
                    <div key={level.id} className="flex items-start gap-2">
                      <span className="text-xs font-medium px-1.5 py-0.5 rounded bg-primary/10 text-primary whitespace-nowrap mt-0.5">
                        {level.name}
                      </span>
                      <p className="text-xs text-muted-foreground line-clamp-2">{level.description}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>

        {fw.targetAudience.length > 0 && (
          <>
            <h3 className="text-lg font-semibold text-foreground mb-3">Target Audience</h3>
            <div className="flex flex-wrap gap-2 mb-8">
              {fw.targetAudience.map((a) => (
                <span key={a} className="text-xs px-3 py-1 rounded-full bg-muted text-muted-foreground">{a}</span>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
