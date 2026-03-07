import { useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, FileText, Loader2, Copy, Download, Check } from "lucide-react";
import Header from "@/components/Header";
import { apiStream } from "@/lib/api";

const policyTypes = [
  { id: "ai-acceptable-use", name: "AI Acceptable Use Policy", desc: "Defines permitted and prohibited uses of AI tools" },
  { id: "ai-governance", name: "AI Governance Policy", desc: "Establishes strategic AI governance structures" },
  { id: "ai-assessment-integrity", name: "AI Assessment Integrity Policy", desc: "Governs AI use in educational assessment" },
  { id: "staff-ai-development", name: "Staff AI Development Policy", desc: "Defines AI literacy requirements and training" },
  { id: "ai-data-governance", name: "AI Data Governance Policy", desc: "Governs data processing and privacy in AI systems" },
  { id: "student-ai-guidance", name: "Student AI Guidance", desc: "Student-facing guidance on responsible AI use" },
];

const regions = [
  { id: "uk", name: "UK (DfE Guidance)" },
  { id: "eu", name: "EU (AI Act)" },
  { id: "us", name: "US (FERPA/NIST)" },
  { id: "international", name: "International" },
];

export default function Policy() {
  const [step, setStep] = useState(1);
  const [selectedType, setSelectedType] = useState("");
  const [institution, setInstitution] = useState("");
  const [region, setRegion] = useState("uk");
  const [content, setContent] = useState("");
  const [generating, setGenerating] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleGenerate = async () => {
    setGenerating(true);
    setContent("");
    setStep(3);

    await apiStream(
      "/policy-generator",
      { policy_type: selectedType, institution_name: institution || "Our Institution", region },
      (chunk) => setContent((prev) => prev + chunk),
      () => setGenerating(false),
      (err) => { setContent(`Error: ${err}`); setGenerating(false); }
    );
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto px-4 sm:px-6 py-8 max-w-3xl">
        <Link to="/" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-8">
          <ArrowLeft className="h-4 w-4" />Back to Hub
        </Link>
        <h2 className="text-2xl sm:text-3xl font-bold text-foreground mb-2">Generate a Policy</h2>
        <p className="text-muted-foreground mb-8">AI-grounded policy drafts for your institution, citing framework indicators and regulatory provisions.</p>

        {/* Step indicators */}
        <div className="flex items-center gap-2 mb-8">
          {[1, 2, 3].map((s) => (
            <div key={s} className={`h-1.5 flex-1 rounded-full ${step >= s ? "bg-primary" : "bg-muted"}`} />
          ))}
        </div>

        {step === 1 && (
          <>
            <h3 className="text-lg font-semibold text-foreground mb-4">Select Policy Type</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
              {policyTypes.map((pt) => (
                <button
                  key={pt.id}
                  onClick={() => { setSelectedType(pt.id); setStep(2); }}
                  className="p-4 rounded-xl border-2 border-border hover:border-primary/50 text-left transition-all hover:shadow-sm"
                >
                  <FileText className="h-5 w-5 text-primary mb-2" />
                  <h4 className="font-semibold text-foreground text-sm">{pt.name}</h4>
                  <p className="text-xs text-muted-foreground mt-1">{pt.desc}</p>
                </button>
              ))}
            </div>
          </>
        )}

        {step === 2 && (
          <>
            <h3 className="text-lg font-semibold text-foreground mb-4">Configure</h3>
            <div className="space-y-4 mb-6">
              <div>
                <label className="text-sm font-medium text-foreground block mb-1">Institution Name</label>
                <input value={institution} onChange={(e) => setInstitution(e.target.value)} placeholder="e.g. UniGlobal University" className="w-full px-3 py-2 text-sm bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50" />
              </div>
              <div>
                <label className="text-sm font-medium text-foreground block mb-1">Region</label>
                <div className="grid grid-cols-2 gap-2">
                  {regions.map((r) => (
                    <button key={r.id} onClick={() => setRegion(r.id)} className={`px-3 py-2 text-sm rounded-lg border-2 transition-all ${region === r.id ? "border-primary bg-primary/5" : "border-border hover:border-primary/30"}`}>
                      {r.name}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <div className="flex gap-3">
              <button onClick={() => setStep(1)} className="px-4 py-2 text-sm border border-border rounded-lg hover:bg-muted">Back</button>
              <button onClick={handleGenerate} className="px-4 py-2 bg-primary text-primary-foreground text-sm font-medium rounded-lg hover:bg-primary/90">Generate Draft</button>
            </div>
          </>
        )}

        {step === 3 && (
          <>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-foreground">
                {generating && <Loader2 className="h-4 w-4 animate-spin inline mr-2" />}
                {generating ? "Generating..." : "Policy Draft"}
              </h3>
              {content && !generating && (
                <div className="flex gap-2">
                  <button onClick={handleCopy} className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm border border-border rounded-lg hover:bg-muted">
                    {copied ? <Check className="h-3.5 w-3.5 text-green-500" /> : <Copy className="h-3.5 w-3.5" />}
                    {copied ? "Copied" : "Copy"}
                  </button>
                  <button onClick={() => { const blob = new Blob([content], { type: "text/plain" }); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `${selectedType}-draft.txt`; a.click(); }} className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm border border-border rounded-lg hover:bg-muted">
                    <Download className="h-3.5 w-3.5" />Download
                  </button>
                </div>
              )}
            </div>
            <div className="p-6 rounded-xl border border-border bg-card min-h-[300px]">
              <div className="prose prose-sm max-w-none dark:prose-invert whitespace-pre-wrap text-sm text-foreground">
                {content || "Generating policy draft..."}
              </div>
            </div>
            {!generating && (
              <button onClick={() => { setStep(1); setContent(""); }} className="mt-4 px-4 py-2 text-sm border border-border rounded-lg hover:bg-muted">
                Generate Another
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
}
