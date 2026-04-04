import { useState, useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import {
  Shield,
  BarChart3,
  FileText,
  Compass,
  HelpCircle,
  ArrowRight,
  Sparkles,
  Target,
  Rocket,
  ScanEye,
  ExternalLink,
} from "lucide-react";
import Header from "@/components/Header";
import GlobeCanvas from "@/components/GlobeCanvas";
import { aifgeLinks } from "@/lib/constants";

/* ------------------------------------------------------------------ */
/*  Custom hooks                                                       */
/* ------------------------------------------------------------------ */

function useScrollFadeIn() {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          el.classList.add("is-visible");
          observer.unobserve(el);
        }
      },
      { threshold: 0.15 }
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return ref;
}

function useCountUp(target: number, duration = 2000) {
  const [value, setValue] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  const started = useRef(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !started.current) {
          started.current = true;
          const start = performance.now();
          const step = (now: number) => {
            const progress = Math.min((now - start) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            setValue(Math.round(eased * target));
            if (progress < 1) requestAnimationFrame(step);
          };
          requestAnimationFrame(step);
          observer.unobserve(el);
        }
      },
      { threshold: 0.5 }
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, [target, duration]);

  return { ref, value };
}

/* ------------------------------------------------------------------ */
/*  Data                                                               */
/* ------------------------------------------------------------------ */

const actions = [
  {
    title: "Test an AI Tool",
    description:
      "Run safety audits on any AI tool. Automated red-teaming with PETRI evaluates safety, bias, and compliance.",
    icon: Shield,
    href: "/audit",
    accent: "border-t-aifge-teal",
    iconBg: "bg-aifge-teal/10 text-aifge-teal",
  },
  {
    title: "Assess Your AI Readiness",
    description:
      "Self-assess against 22 international frameworks. Get personalised learning paths based on your gaps.",
    icon: BarChart3,
    href: "/assess",
    accent: "border-t-aifge-orange",
    iconBg: "bg-aifge-orange/10 text-aifge-orange",
  },
  {
    title: "Generate a Policy",
    description:
      "AI-grounded policy drafts for your institution. Regulatory context from EU AI Act, UK DfE, and FERPA.",
    icon: FileText,
    href: "/policy",
    accent: "border-t-green-500",
    iconBg: "bg-green-500/10 text-green-500",
  },
  {
    title: "Explore Frameworks",
    description:
      "Browse all 22 AI literacy and digital competence frameworks. UNESCO, DigComp, JISC, ISTE, and more.",
    icon: Compass,
    href: "/frameworks",
    accent: "border-t-aifge-teal",
    iconBg: "bg-aifge-teal/10 text-aifge-teal",
  },
  {
    title: "Can AI Do This?",
    description:
      "Evaluate any educational task for AI feasibility. Get safeguards, risks, and implementation guidance.",
    icon: HelpCircle,
    href: "/evaluate",
    accent: "border-t-aifge-orange",
    iconBg: "bg-aifge-orange/10 text-aifge-orange",
  },
];

const heroStats = [
  { value: "22+", label: "Frameworks" },
  { value: "200+", label: "Indicators" },
  { value: "4+", label: "Regions" },
];

const impactStats = [
  { value: 22, label: "Frameworks" },
  { value: 6, label: "Policy Types" },
  { value: 4, label: "Regions" },
  { value: 5, label: "Pathways" },
];

const steps = [
  {
    num: "1",
    icon: Sparkles,
    title: "Choose Your Path",
    desc: "Test an AI tool, assess your readiness, generate a policy, or explore frameworks.",
  },
  {
    num: "2",
    icon: Target,
    title: "Get Evidence",
    desc: "Automated audits, framework-grounded assessments, and regulatory-aware analysis.",
  },
  {
    num: "3",
    icon: Rocket,
    title: "Take Action",
    desc: "Personalised learning paths, policy drafts, and evidence portfolios for your institution.",
  },
];

const trustBadges = [
  "UNESCO",
  "OECD",
  "European Commission",
  "JISC",
  "ISTE",
  "DfE",
  "NIST",
];

/* ------------------------------------------------------------------ */
/*  Stat Counter component                                             */
/* ------------------------------------------------------------------ */

function StatCounter({ target, label }: { target: number; label: string }) {
  const { ref, value } = useCountUp(target);
  return (
    <div className="text-center px-6 py-4">
      <span
        ref={ref}
        className="block text-4xl sm:text-5xl font-bold text-aifge-teal"
      >
        {value}
      </span>
      <span className="text-sm text-muted-foreground mt-1 block">
        {label}
      </span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Hub Page                                                           */
/* ------------------------------------------------------------------ */

export default function Hub() {
  const statsRef = useScrollFadeIn();
  const cardsRef = useScrollFadeIn();
  const timelineRef = useScrollFadeIn();
  const trustRef = useScrollFadeIn();
  const ctaRef = useScrollFadeIn();

  return (
    <div className="min-h-screen bg-background">
      <Header />

      {/* ====== HERO (AIFGE-style) ====== */}
      <section
        className="relative overflow-hidden"
        style={{
          background: `
            radial-gradient(1100px 520px at 78% 88%, rgba(255,145,77,0.95) 0%, rgba(255,145,77,0.70) 28%, rgba(255,145,77,0.38) 52%, rgba(255,145,77,0.00) 75%),
            radial-gradient(360px 220px at 78% 96%, rgba(255,145,77,0.90) 0%, rgba(255,145,77,0.00) 70%),
            radial-gradient(900px 650px at 18% 14%, rgba(15,164,198,0.46) 0%, rgba(15,164,198,0.18) 45%, rgba(15,164,198,0) 70%),
            linear-gradient(118deg, #0fa4c6 0%, #061233 28%, #201347 70%, #201347 100%)
          `,
        }}
      >
        <div className="container mx-auto px-4 sm:px-6">
          <div
            className="grid items-center gap-4 py-12 sm:py-16 lg:py-20"
            style={{
              gridTemplateColumns: "minmax(280px, 680px) minmax(320px, 1fr)",
            }}
          >
            {/* Left: Text */}
            <div className="max-w-[680px]">
              <h1
                className="text-4xl sm:text-5xl lg:text-6xl font-bold leading-[0.98] mb-4 text-white tracking-tight"
                style={{ fontFamily: "'Roboto', 'Inter', system-ui, sans-serif", textWrap: "balance" } as React.CSSProperties}
              >
                Navigate AI
                <br />
                in Global
                <br />
                Education
              </h1>

              {/* Orange accent bar */}
              <div
                className="h-2 rounded-full my-5"
                style={{
                  width: "clamp(160px, 20vw, 260px)",
                  background: "linear-gradient(90deg, #ffb678, #ff8a3d)",
                  boxShadow: "0 0 26px rgba(255,138,61,0.55)",
                }}
              />

              <p className="text-white/90 text-base sm:text-lg leading-relaxed max-w-lg mb-6">
                Test AI tools for safety. Assess institutional readiness against
                22 international frameworks. Generate evidence-based policies.
                All in one platform by{" "}
                <a
                  href="https://aiforglobaleducation.org"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-aifge-teal hover:underline"
                >
                  AI For Global Education
                </a>
                .
              </p>

              {/* Hero stats */}
              <div className="flex gap-8 sm:gap-12 border-t border-white/10 pt-4 flex-wrap mb-6">
                {heroStats.map((s) => (
                  <div key={s.label}>
                    <div
                      className="text-2xl sm:text-3xl font-bold text-white"
                      style={{ fontVariantNumeric: "tabular-nums", fontFeatureSettings: "'tnum' 1" }}
                    >
                      {s.value}
                    </div>
                    <div className="text-white/60 text-sm">{s.label}</div>
                  </div>
                ))}
              </div>

              {/* CTAs */}
              <div className="flex flex-wrap gap-4">
                <Link
                  to="/assess"
                  className="inline-flex items-center gap-2 px-6 py-3 text-white font-semibold rounded-full bg-gradient-cta transition-all hover:shadow-lg hover:shadow-orange-500/30 hover:-translate-y-0.5"
                  style={{ boxShadow: "0 0 20px rgba(255,138,61,0.5)" }}
                >
                  Get Started
                  <ArrowRight className="h-4 w-4" />
                </Link>
                <Link
                  to="/audit"
                  className="inline-flex items-center gap-2 px-6 py-3 border-2 border-white/30 text-white font-semibold rounded-full hover:bg-white/10 transition-colors"
                >
                  Test an AI Tool
                </Link>
              </div>
            </div>

            {/* Right: Globe */}
            <div className="hidden lg:flex flex-col items-center justify-center relative z-10 aspect-square" style={{ height: "clamp(420px, 56vw, 700px)" }}>
              <GlobeCanvas />
            </div>
          </div>
        </div>
      </section>

      {/* ====== IMPACT STATS ====== */}
      <section className="py-8 sm:py-12">
        <div
          ref={statsRef}
          className="fade-in-on-scroll container mx-auto px-4 sm:px-6"
        >
          <div className="max-w-3xl mx-auto grid grid-cols-2 sm:grid-cols-4 divide-y sm:divide-y-0 sm:divide-x divide-border">
            {impactStats.map((s) => (
              <StatCounter key={s.label} target={s.value} label={s.label} />
            ))}
          </div>
        </div>
      </section>

      {/* ====== ACTION PATHWAYS ====== */}
      <section className="py-12 sm:py-20">
        <div
          ref={cardsRef}
          className="fade-in-on-scroll container mx-auto px-4 sm:px-6"
        >
          <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider text-center mb-2">
            Your Pathways
          </h2>
          <h3 className="text-2xl sm:text-3xl font-bold text-foreground text-center mb-10">
            What Would You Like To Do?
          </h3>

          <div className="max-w-5xl mx-auto grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5 sm:gap-6">
            {actions.map((action, i) => {
              const Icon = action.icon;
              return (
                <Link
                  key={action.href}
                  to={action.href}
                  className={`group relative p-7 rounded-xl border border-border border-t-4 ${action.accent} bg-card text-left transition-all duration-300 hover:shadow-lg hover:-translate-y-1 fade-in-delay-${Math.min(i + 1, 3)}`}
                >
                  <div
                    className={`w-12 h-12 rounded-lg flex items-center justify-center mb-4 ${action.iconBg}`}
                  >
                    <Icon className="h-6 w-6" />
                  </div>
                  <h4 className="text-lg font-semibold text-foreground mb-2">
                    {action.title}
                  </h4>
                  <p className="text-sm text-muted-foreground leading-relaxed mb-4">
                    {action.description}
                  </p>
                  <span className="inline-flex items-center gap-1 text-sm font-medium text-aifge-teal opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                    Explore
                    <ArrowRight className="h-3.5 w-3.5 translate-x-0 group-hover:translate-x-1 transition-transform duration-300" />
                  </span>
                </Link>
              );
            })}
          </div>
        </div>
      </section>

      {/* ====== HOW IT WORKS (TIMELINE) ====== */}
      <section className="border-t border-border bg-muted/30 py-16 sm:py-20">
        <div
          ref={timelineRef}
          className="fade-in-on-scroll container mx-auto px-4 sm:px-6"
        >
          <h3 className="text-2xl sm:text-3xl font-bold text-foreground text-center mb-14">
            How It Works
          </h3>

          <div className="relative max-w-4xl mx-auto">
            {/* Connecting line — desktop */}
            <div className="hidden md:block absolute top-10 left-[16.67%] right-[16.67%] h-0.5 bg-border" />

            <div className="grid grid-cols-1 md:grid-cols-3 gap-10 md:gap-8">
              {steps.map((item) => {
                const StepIcon = item.icon;
                return (
                  <div key={item.num} className="text-center relative">
                    <div className="flex flex-col items-center">
                      <div className="w-6 h-6 rounded-full flex items-center justify-center mb-3">
                        <StepIcon className="h-5 w-5 text-aifge-teal" />
                      </div>
                      <div className="relative z-10 w-14 h-14 rounded-full bg-aifge-navy text-white flex items-center justify-center text-xl font-bold ring-4 ring-aifge-teal/20 mb-5">
                        {item.num}
                      </div>
                    </div>
                    <h4 className="text-lg font-semibold text-foreground mb-2">
                      {item.title}
                    </h4>
                    <p className="text-sm text-muted-foreground max-w-xs mx-auto">
                      {item.desc}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </section>

      {/* ====== TRUST BAR ====== */}
      <section className="py-12 sm:py-16 border-t border-border">
        <div
          ref={trustRef}
          className="fade-in-on-scroll container mx-auto px-4 sm:px-6 text-center"
        >
          <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground mb-6">
            Built on Global Standards
          </p>
          <div className="flex flex-wrap justify-center gap-3 sm:gap-4 max-w-3xl mx-auto">
            {trustBadges.map((badge) => (
              <span
                key={badge}
                className="px-4 py-2 text-sm font-medium text-muted-foreground bg-muted/50 border border-border rounded-full"
              >
                {badge}
              </span>
            ))}
          </div>
          <p className="text-xs text-muted-foreground mt-5">
            Regulatory coverage: EU AI Act &middot; UK DfE Guidance &middot; US
            FERPA
          </p>
        </div>
      </section>

      {/* ====== CTA SECTION ====== */}
      <section
        className="relative py-16 sm:py-20"
        style={{
          background: "linear-gradient(118deg, #0fa4c6 0%, #061233 40%, #201347 100%)",
        }}
      >
        <div
          ref={ctaRef}
          className="fade-in-on-scroll container mx-auto px-4 sm:px-6 text-center"
        >
          <h3 className="text-2xl sm:text-3xl font-bold text-white mb-4">
            Ready to get started?
          </h3>
          <p className="text-white/70 max-w-lg mx-auto mb-8">
            Join educators and institutions using ReasonLens to navigate AI
            safely and responsibly.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <Link
              to="/assess"
              className="inline-flex items-center gap-2 px-6 py-3 text-white font-semibold rounded-full bg-gradient-cta transition-all hover:shadow-lg hover:shadow-orange-500/30"
            >
              Assess Your Readiness
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              to="/audit"
              className="inline-flex items-center gap-2 px-6 py-3 border-2 border-white/40 text-white font-semibold rounded-full hover:bg-white/10 transition-colors"
            >
              Test an AI Tool
            </Link>
          </div>
        </div>
      </section>

      {/* ====== FOOTER ====== */}
      <footer className="bg-aifge-navy text-white border-t border-white/10 py-12">
        <div className="container mx-auto px-4 sm:px-6">
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-8 mb-8">
            {/* Brand */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <ScanEye className="h-5 w-5 text-aifge-teal" />
                <span className="font-bold text-white">ReasonLens</span>
              </div>
              <p className="text-sm text-white/60 leading-relaxed">
                Evidence-based AI literacy for global education. A project by{" "}
                <a
                  href="https://aiforglobaleducation.org"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-aifge-teal hover:underline"
                >
                  AI For Global Education
                </a>
                .
              </p>
            </div>

            {/* Pathways */}
            <div>
              <h4 className="text-sm font-semibold text-white mb-3">
                Pathways
              </h4>
              <ul className="space-y-2 text-sm text-white/60">
                <li>
                  <Link to="/audit" className="hover:text-white transition-colors">
                    AI Tool Audits
                  </Link>
                </li>
                <li>
                  <Link to="/assess" className="hover:text-white transition-colors">
                    Readiness Assessment
                  </Link>
                </li>
                <li>
                  <Link to="/policy" className="hover:text-white transition-colors">
                    Policy Generator
                  </Link>
                </li>
                <li>
                  <Link to="/evaluate" className="hover:text-white transition-colors">
                    Task Evaluator
                  </Link>
                </li>
              </ul>
            </div>

            {/* Resources */}
            <div>
              <h4 className="text-sm font-semibold text-white mb-3">
                Resources
              </h4>
              <ul className="space-y-2 text-sm text-white/60">
                <li>
                  <Link to="/frameworks" className="hover:text-white transition-colors">
                    Framework Explorer
                  </Link>
                </li>
                <li>
                  <Link to="/portfolio" className="hover:text-white transition-colors">
                    Evidence Portfolio
                  </Link>
                </li>
                <li>
                  <Link to="/badges" className="hover:text-white transition-colors">
                    Badges
                  </Link>
                </li>
                <li>
                  <Link to="/my-progress" className="hover:text-white transition-colors">
                    My Progress
                  </Link>
                </li>
              </ul>
            </div>

            {/* AIFGE */}
            <div>
              <h4 className="text-sm font-semibold text-white mb-3">
                AI For Global Education
              </h4>
              <ul className="space-y-2 text-sm text-white/60">
                {aifgeLinks.map((link) => (
                  <li key={link.href}>
                    <a href={link.href} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 hover:text-white transition-colors">
                      {link.label} <ExternalLink className="h-3 w-3 opacity-40" />
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="border-t border-white/10 pt-6 text-center text-xs text-white/40">
            &copy; {new Date().getFullYear()} AI For Global Education. All
            rights reserved.
          </div>
        </div>
      </footer>
    </div>
  );
}
