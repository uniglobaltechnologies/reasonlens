import { Link } from "react-router-dom";
import { ArrowLeft, BarChart3, Award, FolderOpen } from "lucide-react";
import Header from "@/components/Header";

export default function MyProgress() {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto px-4 sm:px-6 py-8 max-w-3xl">
        <Link to="/" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-8">
          <ArrowLeft className="h-4 w-4" />Back to Hub
        </Link>
        <h2 className="text-2xl font-bold text-foreground mb-6">My Progress</h2>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
          <div className="p-5 rounded-xl border border-border bg-card text-center">
            <BarChart3 className="h-8 w-8 text-primary mx-auto mb-2" />
            <p className="text-2xl font-bold text-foreground">0</p>
            <p className="text-sm text-muted-foreground">Assessments</p>
          </div>
          <div className="p-5 rounded-xl border border-border bg-card text-center">
            <Award className="h-8 w-8 text-amber-500 mx-auto mb-2" />
            <p className="text-2xl font-bold text-foreground">0</p>
            <p className="text-sm text-muted-foreground">Badges</p>
          </div>
          <div className="p-5 rounded-xl border border-border bg-card text-center">
            <FolderOpen className="h-8 w-8 text-green-500 mx-auto mb-2" />
            <p className="text-2xl font-bold text-foreground">0</p>
            <p className="text-sm text-muted-foreground">Portfolio Items</p>
          </div>
        </div>

        <div className="p-8 rounded-xl border-2 border-dashed border-border text-center">
          <p className="text-muted-foreground mb-4">Complete an assessment to see your progress here.</p>
          <Link to="/assess" className="px-4 py-2 bg-primary text-primary-foreground text-sm font-medium rounded-lg hover:bg-primary/90 transition-colors">Start Assessment</Link>
        </div>
      </div>
    </div>
  );
}
