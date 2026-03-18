import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, Plus, FolderOpen, Loader2, X, Trash2, FileText, LinkIcon, PenLine, Video } from "lucide-react";
import Header from "@/components/Header";
import { apiGet, apiPost, apiDelete, isAuthenticated } from "@/lib/api";

interface PortfolioItem {
  id: string;
  title: string;
  description: string | null;
  artifact_type: string;
  file_url: string | null;
  visibility: string;
  created_at: string;
}

const ARTIFACT_TYPES = [
  { value: "document", label: "Document", icon: FileText },
  { value: "link", label: "Link", icon: LinkIcon },
  { value: "reflection", label: "Reflection", icon: PenLine },
  { value: "video", label: "Video", icon: Video },
];

export default function Portfolio() {
  const [items, setItems] = useState<PortfolioItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [artifactType, setArtifactType] = useState("document");
  const [fileUrl, setFileUrl] = useState("");

  useEffect(() => {
    if (!isAuthenticated()) {
      setLoading(false);
      setError("Please sign in to view your portfolio.");
      return;
    }

    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const res = await apiGet<{ items: PortfolioItem[] }>("/portfolio");
        if (!cancelled) setItems(res.items);
      } catch (err: any) {
        if (!cancelled) setError(err?.message || "Failed to load portfolio.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void load();
    return () => { cancelled = true; };
  }, []);

  const resetForm = () => {
    setTitle("");
    setDescription("");
    setArtifactType("document");
    setFileUrl("");
    setFormError(null);
  };

  const handleAdd = async () => {
    if (!title.trim()) {
      setFormError("Title is required.");
      return;
    }
    setSaving(true);
    setFormError(null);
    try {
      const res = await apiPost<{ id: string }>("/portfolio", {
        title: title.trim(),
        description: description.trim() || undefined,
        artifact_type: artifactType,
        file_url: fileUrl.trim() || undefined,
      });
      setItems((prev) => [
        {
          id: res.id,
          title: title.trim(),
          description: description.trim() || null,
          artifact_type: artifactType,
          file_url: fileUrl.trim() || null,
          visibility: "private",
          created_at: new Date().toISOString(),
        },
        ...prev,
      ]);
      resetForm();
      setShowForm(false);
    } catch (err: any) {
      setFormError(err?.message || "Failed to add item.");
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await apiDelete("/portfolio", { id });
      setItems((prev) => prev.filter((i) => i.id !== id));
    } catch (err: any) {
      setError(err?.message || "Failed to delete item.");
    }
  };

  const getTypeIcon = (type: string) => {
    const t = ARTIFACT_TYPES.find((a) => a.value === type);
    return t ? t.icon : FileText;
  };

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto px-4 sm:px-6 py-8 max-w-3xl">
        <Link to="/" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-8">
          <ArrowLeft className="h-4 w-4" />Back to Hub
        </Link>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-foreground">Evidence Portfolio</h2>
          <button
            onClick={() => { setShowForm(true); resetForm(); }}
            className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground text-sm font-medium rounded-lg hover:bg-primary/90 transition-colors"
          >
            <Plus className="h-4 w-4" />Add Evidence
          </button>
        </div>

        {error && (
          <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 mb-6">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        {showForm && (
          <div className="p-5 rounded-xl border border-border bg-card mb-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-foreground">Add Evidence</h3>
              <button onClick={() => setShowForm(false)} className="text-muted-foreground hover:text-foreground">
                <X className="h-5 w-5" />
              </button>
            </div>

            {formError && (
              <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 mb-4">
                <p className="text-sm text-red-600">{formError}</p>
              </div>
            )}

            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium text-foreground block mb-1">Title *</label>
                <input
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="e.g. Lesson plan using AI for Year 8 Science"
                  className="w-full px-3 py-2 text-sm bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50"
                />
              </div>
              <div>
                <label className="text-sm font-medium text-foreground block mb-1">Type</label>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  {ARTIFACT_TYPES.map((t) => {
                    const Icon = t.icon;
                    return (
                      <button
                        key={t.value}
                        onClick={() => setArtifactType(t.value)}
                        className={`flex items-center gap-2 px-3 py-2 text-sm rounded-lg border-2 transition-all ${
                          artifactType === t.value ? "border-primary bg-primary/5" : "border-border hover:border-primary/30"
                        }`}
                      >
                        <Icon className="h-4 w-4" />
                        {t.label}
                      </button>
                    );
                  })}
                </div>
              </div>
              <div>
                <label className="text-sm font-medium text-foreground block mb-1">URL / Link</label>
                <input
                  value={fileUrl}
                  onChange={(e) => setFileUrl(e.target.value)}
                  placeholder="https://..."
                  className="w-full px-3 py-2 text-sm bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50"
                />
              </div>
              <div>
                <label className="text-sm font-medium text-foreground block mb-1">Description</label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Describe what this evidence demonstrates..."
                  rows={3}
                  className="w-full px-3 py-2 text-sm bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none"
                />
              </div>
              <button
                onClick={handleAdd}
                disabled={saving}
                className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground text-sm font-medium rounded-lg hover:bg-primary/90 transition-colors disabled:opacity-50"
              >
                {saving && <Loader2 className="h-4 w-4 animate-spin" />}
                {saving ? "Saving..." : "Add to Portfolio"}
              </button>
            </div>
          </div>
        )}

        {loading && (
          <div className="p-6 rounded-xl border border-border bg-card flex items-center gap-3">
            <Loader2 className="h-5 w-5 animate-spin text-primary" />
            <p className="text-sm text-muted-foreground">Loading portfolio...</p>
          </div>
        )}

        {!loading && items.length === 0 && !error && (
          <div className="p-12 rounded-xl border-2 border-dashed border-border text-center">
            <FolderOpen className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
            <p className="text-muted-foreground mb-2">No portfolio items yet.</p>
            <p className="text-sm text-muted-foreground">Add documents, links, reflections, or videos as evidence of your AI competencies.</p>
          </div>
        )}

        {!loading && items.length > 0 && (
          <div className="space-y-3">
            {items.map((item) => {
              const Icon = getTypeIcon(item.artifact_type);
              return (
                <div key={item.id} className="p-4 rounded-xl border border-border bg-card flex items-start justify-between gap-3">
                  <div className="flex items-start gap-3">
                    <div className="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                      <Icon className="h-4 w-4 text-primary" />
                    </div>
                    <div>
                      <p className="font-medium text-foreground text-sm">{item.title}</p>
                      {item.description && <p className="text-xs text-muted-foreground mt-1 line-clamp-2">{item.description}</p>}
                      <div className="flex items-center gap-2 mt-1.5">
                        <span className="text-xs text-muted-foreground capitalize">{item.artifact_type}</span>
                        <span className="text-xs text-muted-foreground">·</span>
                        <span className="text-xs text-muted-foreground">{new Date(item.created_at).toLocaleDateString()}</span>
                        {item.file_url && /^https?:\/\//i.test(item.file_url) && (
                          <>
                            <span className="text-xs text-muted-foreground">·</span>
                            <a href={item.file_url} target="_blank" rel="noopener noreferrer" className="text-xs text-primary hover:underline">View</a>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={() => handleDelete(item.id)}
                    className="text-muted-foreground hover:text-red-500 transition-colors flex-shrink-0"
                    title="Delete"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
