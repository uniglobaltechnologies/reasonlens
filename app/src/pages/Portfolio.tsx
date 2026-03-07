import { Link } from "react-router-dom";
import { ArrowLeft, Plus, FolderOpen } from "lucide-react";
import Header from "@/components/Header";

export default function Portfolio() {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto px-4 sm:px-6 py-8 max-w-3xl">
        <Link to="/" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-8">
          <ArrowLeft className="h-4 w-4" />Back to Hub
        </Link>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-foreground">Evidence Portfolio</h2>
          <button className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground text-sm font-medium rounded-lg hover:bg-primary/90 transition-colors">
            <Plus className="h-4 w-4" />Add Evidence
          </button>
        </div>
        <div className="p-12 rounded-xl border-2 border-dashed border-border text-center">
          <FolderOpen className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <p className="text-muted-foreground mb-2">No portfolio items yet.</p>
          <p className="text-sm text-muted-foreground">Add documents, links, reflections, or videos as evidence of your AI competencies.</p>
        </div>
      </div>
    </div>
  );
}
