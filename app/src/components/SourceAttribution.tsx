interface SourceAttributionProps {
  attribution: {
    source_framework?: string;
    source_licence?: string;
    content_type?: string;
    attribution_text?: string;
  } | null | undefined;
}

export default function SourceAttribution({ attribution }: SourceAttributionProps) {
  if (!attribution?.source_framework) return null;

  return (
    <div className="text-xs text-muted-foreground border-t border-border pt-3 mt-4">
      <p>{attribution.attribution_text || `Source: ${attribution.source_framework}`}</p>
      {attribution.source_licence && (
        <p className="mt-0.5">Licence: {attribution.source_licence}</p>
      )}
    </div>
  );
}
