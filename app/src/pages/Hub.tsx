import { Link } from "react-router-dom";
import {
  ScanEye,
  Shield,
  BarChart3,
  FileText,
  Compass,
  HelpCircle,
  LogIn,
} from "lucide-react";

const actions = [
  {
    title: "Test an AI Tool",
    description:
      "Run safety audits on any AI tool. Automated red-teaming with PETRI evaluates safety, bias, and compliance.",
    icon: Shield,
    href: "/audit",
    color: "primary",
  },
  {
    title: "Assess Your AI Readiness",
    description:
      "Self-assess against 22 international frameworks. Get personalised learning paths based on your gaps.",
    icon: BarChart3,
    href: "/assess",
    color: "accent",
  },
  {
    title: "Generate a Policy",
    description:
      "AI-grounded policy drafts for your institution. Regulatory context from EU AI Act, UK DfE, and FERPA.",
    icon: FileText,
    href: "/policy",
    color: "success",
  },
  {
    title: "Explore Frameworks",
    description:
      "Browse all 22 AI literacy and digital competence frameworks. UNESCO, DigComp, JISC, ISTE, and more.",
    icon: Compass,
    href: "/frameworks",
    color: "warning",
  },
  {
    title: "Can AI Do This?",
    description:
      "Evaluate any educational task for AI feasibility. Get safeguards, risks, and implementation guidance.",
    icon: HelpCircle,
    href: "/evaluate",
    color: "destructive",
  },
];

const colorMap: Record<string, string> = {
  primary: "border-primary/30 hover:border-primary hover:shadow-primary/10",
  accent: "border-accent/30 hover:border-accent hover:shadow-accent/10",
  success: "border-green-500/30 hover:border-green-500 hover:shadow-green-500/10",
  warning: "border-amber-500/30 hover:border-amber-500 hover:shadow-amber-500/10",
  destructive: "border-red-500/30 hover:border-red-500 hover:shadow-red-500/10",
};

const iconColorMap: Record<string, string> = {
  primary: "bg-primary/10 text-primary",
  accent: "bg-cyan-500/10 text-cyan-500",
  success: "bg-green-500/10 text-green-500",
  warning: "bg-amber-500/10 text-amber-500",
  destructive: "bg-red-500/10 text-red-500",
};

export default function Hub() {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 sm:px-6 py-3 sm:py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ScanEye className="h-6 w-6 text-primary" />
            <div>
              <h1 className="text-xl font-bold text-foreground">ReasonLens</h1>
              <p className="text-xs text-muted-foreground hidden sm:block">
                Clarity Through Ethical AI Evaluation
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Link
              to="/auth"
              className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg border border-border hover:bg-muted transition-colors"
            >
              <LogIn className="h-4 w-4" />
              <span className="hidden sm:inline">Sign In</span>
            </Link>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="container mx-auto px-4 sm:px-6 py-12 sm:py-20 text-center">
        <h2 className="text-3xl sm:text-5xl font-bold text-foreground mb-4 leading-tight">
          Navigate AI in Education
          <br />
          <span className="text-primary">Safely, Ethically, With Evidence.</span>
        </h2>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto mb-12">
          Test AI tools for safety. Assess institutional readiness against 22
          international frameworks. Generate evidence-based policies. All in one
          platform.
        </p>

        {/* Action Cards */}
        <div className="max-w-5xl mx-auto">
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-6">
            What would you like to do?
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
            {actions.map((action) => {
              const Icon = action.icon;
              return (
                <Link
                  key={action.href}
                  to={action.href}
                  className={`group relative p-6 rounded-xl border-2 bg-card text-left transition-all duration-300 hover:shadow-lg hover:-translate-y-1 ${colorMap[action.color]}`}
                >
                  <div
                    className={`w-12 h-12 rounded-lg flex items-center justify-center mb-4 ${iconColorMap[action.color]}`}
                  >
                    <Icon className="h-6 w-6" />
                  </div>
                  <h4 className="text-lg font-semibold text-foreground mb-2">
                    {action.title}
                  </h4>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {action.description}
                  </p>
                </Link>
              );
            })}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="border-t border-border bg-muted/30 py-16">
        <div className="container mx-auto px-4 sm:px-6">
          <h3 className="text-2xl font-bold text-foreground text-center mb-12">
            How It Works
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto">
            {[
              {
                step: "1",
                title: "Choose Your Path",
                desc: "Test an AI tool, assess your readiness, generate a policy, or explore frameworks.",
              },
              {
                step: "2",
                title: "Get Evidence",
                desc: "Automated audits, framework-grounded assessments, and regulatory-aware analysis.",
              },
              {
                step: "3",
                title: "Take Action",
                desc: "Personalised learning paths, policy drafts, and evidence portfolios for your institution.",
              },
            ].map((item) => (
              <div key={item.step} className="text-center">
                <div className="w-12 h-12 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-lg font-bold mx-auto mb-4">
                  {item.step}
                </div>
                <h4 className="text-lg font-semibold text-foreground mb-2">
                  {item.title}
                </h4>
                <p className="text-sm text-muted-foreground">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-8">
        <div className="container mx-auto px-4 sm:px-6 text-center text-sm text-muted-foreground">
          <p>
            ReasonLens by{" "}
            <a
              href="https://aiforglobaleducation.org"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              AI For Global Education
            </a>
          </p>
        </div>
      </footer>
    </div>
  );
}
