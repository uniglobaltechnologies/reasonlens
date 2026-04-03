import { useState, useRef, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import {
  ScanEye,
  LogIn,
  LogOut,
  BarChart3,
  FolderOpen,
  Award,
  Menu,
  X,
  ChevronDown,
  ExternalLink,
} from "lucide-react";
import { isAuthenticated, clearToken } from "@/lib/api";
import ThemeToggle from "./ThemeToggle";
import AIGELogo from "@/assets/aige-logo.png";

const navLinks = [
  { label: "Home", to: "/" },
  { label: "Assess", to: "/assess" },
  { label: "Frameworks", to: "/frameworks" },
  { label: "Audit", to: "/audit" },
  { label: "Policy", to: "/policy" },
  { label: "Evaluate", to: "/evaluate" },
];

const aifgeLinks = [
  { label: "Courses", href: "https://aiforglobaleducation.org/courses/" },
  { label: "Resources", href: "https://aiforglobaleducation.org/resources/" },
  { label: "Volunteering", href: "https://aiforglobaleducation.org/volunteering/" },
  { label: "About Us", href: "https://aiforglobaleducation.org/about-us/" },
];

export default function Header() {
  const authed = isAuthenticated();
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [aifgeOpen, setAifgeOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const handleLogout = () => {
    clearToken();
    window.location.href = "/";
  };

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setAifgeOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  // Close mobile menu on route change
  useEffect(() => {
    setMobileOpen(false);
    setAifgeOpen(false);
  }, [location.pathname]);

  const isActive = (path: string) =>
    path === "/" ? location.pathname === "/" : location.pathname.startsWith(path);

  return (
    <header className="bg-[#061233] border-b border-white/10 sticky top-0 z-50">
      <div className="container mx-auto px-4 sm:px-6 py-3 flex items-center justify-between">
        {/* Left: Logo + brand */}
        <div className="flex items-center gap-3 sm:gap-4">
          <a
            href="https://aiforglobaleducation.org"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:opacity-80 transition-opacity"
          >
            <img src={AIGELogo} alt="AI For Global Education" className="h-9 w-9 rounded" />
          </a>
          <div className="h-8 w-px bg-white/20 hidden sm:block" />
          <Link to="/" className="flex items-center gap-2">
            <ScanEye className="h-5 w-5 text-[#0fa4c6]" />
            <div>
              <h1 className="text-lg font-bold text-white leading-tight">ReasonLens</h1>
              <p className="text-[10px] text-white/50 hidden sm:block leading-tight">
                by AI For Global Education
              </p>
            </div>
          </Link>
        </div>

        {/* Center: Desktop nav links */}
        <nav className="hidden lg:flex items-center gap-1">
          {navLinks.map((link) => (
            <Link
              key={link.to}
              to={link.to}
              className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
                isActive(link.to)
                  ? "text-white bg-white/10"
                  : "text-white/70 hover:text-white hover:bg-white/5"
              }`}
            >
              {link.label}
            </Link>
          ))}

          {/* AIFGE dropdown */}
          <div ref={dropdownRef} className="relative">
            <button
              onClick={() => setAifgeOpen(!aifgeOpen)}
              className="flex items-center gap-1 px-3 py-1.5 text-sm text-white/70 hover:text-white hover:bg-white/5 rounded-md transition-colors"
            >
              AIFGE
              <ChevronDown className={`h-3.5 w-3.5 transition-transform ${aifgeOpen ? "rotate-180" : ""}`} />
            </button>
            {aifgeOpen && (
              <div className="absolute right-0 top-full mt-1 w-48 bg-[#0a1a3a] border border-white/10 rounded-lg shadow-xl py-1 z-50">
                {aifgeLinks.map((link) => (
                  <a
                    key={link.href}
                    href={link.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-between px-4 py-2 text-sm text-white/80 hover:text-white hover:bg-white/5 transition-colors"
                  >
                    {link.label}
                    <ExternalLink className="h-3 w-3 opacity-40" />
                  </a>
                ))}
              </div>
            )}
          </div>
        </nav>

        {/* Right: Auth + CTA + mobile toggle */}
        <div className="flex items-center gap-2 sm:gap-3">
          {authed && (
            <>
              <Link
                to="/my-progress"
                className="hidden xl:inline-flex items-center gap-1.5 px-3 py-1.5 text-sm text-white/60 hover:text-white transition-colors"
              >
                <BarChart3 className="h-4 w-4" />
                Progress
              </Link>
              <Link
                to="/portfolio"
                className="hidden xl:inline-flex items-center gap-1.5 px-3 py-1.5 text-sm text-white/60 hover:text-white transition-colors"
              >
                <FolderOpen className="h-4 w-4" />
                Portfolio
              </Link>
              <Link
                to="/badges"
                className="hidden xl:inline-flex items-center gap-1.5 px-3 py-1.5 text-sm text-white/60 hover:text-white transition-colors"
              >
                <Award className="h-4 w-4" />
                Badges
              </Link>
              <button
                onClick={handleLogout}
                className="hidden sm:inline-flex items-center gap-2 px-3 py-1.5 text-sm text-white/70 hover:text-white border border-white/20 rounded-lg transition-colors"
              >
                <LogOut className="h-4 w-4" />
                <span className="hidden sm:inline">Sign Out</span>
              </button>
            </>
          )}
          {!authed && (
            <Link
              to="/auth"
              className="hidden sm:inline-flex items-center gap-2 px-3 py-1.5 text-sm text-white/70 hover:text-white border border-white/20 rounded-lg transition-colors"
            >
              <LogIn className="h-4 w-4" />
              <span>Sign In</span>
            </Link>
          )}

          <Link
            to="/assess"
            className="hidden sm:inline-flex items-center px-4 py-2 text-sm font-semibold text-white rounded-full transition-all hover:shadow-lg hover:shadow-orange-500/30"
            style={{ background: "linear-gradient(90deg, #ffb678, #ff8a3d)" }}
          >
            Get Started
          </Link>

          <ThemeToggle />

          {/* Mobile hamburger */}
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="lg:hidden p-2 text-white/70 hover:text-white transition-colors"
            aria-label="Toggle menu"
          >
            {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {mobileOpen && (
        <div className="lg:hidden border-t border-white/10 bg-[#061233]">
          <div className="container mx-auto px-4 py-4 space-y-1">
            {navLinks.map((link) => (
              <Link
                key={link.to}
                to={link.to}
                className={`block px-4 py-2.5 text-sm rounded-lg transition-colors ${
                  isActive(link.to)
                    ? "text-white bg-white/10"
                    : "text-white/70 hover:text-white hover:bg-white/5"
                }`}
              >
                {link.label}
              </Link>
            ))}

            <div className="border-t border-white/10 my-2 pt-2">
              <p className="px-4 py-1 text-xs font-semibold text-white/40 uppercase tracking-wider">
                AI For Global Education
              </p>
              {aifgeLinks.map((link) => (
                <a
                  key={link.href}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center justify-between px-4 py-2.5 text-sm text-white/70 hover:text-white hover:bg-white/5 rounded-lg transition-colors"
                >
                  {link.label}
                  <ExternalLink className="h-3 w-3 opacity-40" />
                </a>
              ))}
            </div>

            <div className="border-t border-white/10 my-2 pt-3 space-y-2">
              {!authed && (
                <Link
                  to="/auth"
                  className="flex items-center gap-2 px-4 py-2.5 text-sm text-white/70 hover:text-white rounded-lg transition-colors"
                >
                  <LogIn className="h-4 w-4" />
                  Sign In
                </Link>
              )}
              {authed && (
                <>
                  <Link to="/my-progress" className="flex items-center gap-2 px-4 py-2.5 text-sm text-white/70 hover:text-white rounded-lg transition-colors">
                    <BarChart3 className="h-4 w-4" /> Progress
                  </Link>
                  <Link to="/portfolio" className="flex items-center gap-2 px-4 py-2.5 text-sm text-white/70 hover:text-white rounded-lg transition-colors">
                    <FolderOpen className="h-4 w-4" /> Portfolio
                  </Link>
                  <Link to="/badges" className="flex items-center gap-2 px-4 py-2.5 text-sm text-white/70 hover:text-white rounded-lg transition-colors">
                    <Award className="h-4 w-4" /> Badges
                  </Link>
                  <button onClick={handleLogout} className="flex items-center gap-2 px-4 py-2.5 text-sm text-white/70 hover:text-white rounded-lg transition-colors w-full text-left">
                    <LogOut className="h-4 w-4" /> Sign Out
                  </button>
                </>
              )}
              <Link
                to="/assess"
                className="block text-center px-4 py-2.5 text-sm font-semibold text-white rounded-full"
                style={{ background: "linear-gradient(90deg, #ffb678, #ff8a3d)" }}
              >
                Get Started
              </Link>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
