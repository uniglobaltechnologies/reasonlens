import { Shield } from "lucide-react";

export default function ProAuditWizard() {
  return (
    <div className="max-w-2xl">
      <div className="rounded-xl border-2 border-dashed border-border p-12 text-center">
        <Shield className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-foreground mb-2">
          Pro Audit Wizard
        </h3>
        <p className="text-sm text-muted-foreground">
          Select scenario packs, choose models, and configure audit parameters
          manually. Coming soon — use Simple mode for now.
        </p>
      </div>
    </div>
  );
}
