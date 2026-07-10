/* ResultViewer.jsx -- the real UMR / UIR JSON pane for a finished job.
 *
 * In the design kit this component existed but was never mounted, and its
 * contents were two hardcoded string constants (`sampleUMR`, `sampleJSON`)
 * showing a fake "Attention Is All You Need" result. It now fetches the
 * actual artefacts:
 *
 *   UMR      -> GET /api/umr/<job_id>      (markdown, the agent-facing view)
 *   UIR JSON -> GET /api/result/<job_id>   (intent-filtered when an intent was set)
 *   Download -> GET /api/download/<job_id> (always the full document)
 *
 * The JSON tab fetches lazily: on a large document /api/result is the
 * expensive call and most users never open it.
 *
 * IIFE-wrapped: see app.jsx.
 */

(function () {

const { Tabs, Button, Badge } = window.ApertureDesignSystem_0a9afd;

const PRE_STYLE = {
  margin: 0,
  background: "var(--surface-tile-1)",
  color: "var(--text-on-dark)",
  borderRadius: "var(--radius-sm)",
  padding: "16px 18px",
  fontFamily: "var(--font-mono)",
  fontSize: "var(--text-mono-size)",
  lineHeight: "var(--text-mono-leading)",
  maxHeight: "48vh",
  overflow: "auto",
  whiteSpace: "pre-wrap",
};

/** Friendly names for the pipeline's internal extraction routes. */
const ROUTE_LABELS = {
  pdf: "PDFplumber",
  docling: "Docling",
  pptx: "Native PPTX walker",
  text: "Text walker",
  image: "Fireworks AI vision",
  skip: "skipped",
};

function ResultViewer({ job }) {
  const [view, setView] = React.useState("umr");
  const [umr, setUmr] = React.useState(null);
  const [uir, setUir] = React.useState(null);
  const [error, setError] = React.useState("");
  const [copied, setCopied] = React.useState(false);

  const jobId = job.job_id;

  // UMR is the default view, so fetch it as soon as the job lands.
  React.useEffect(() => {
    let cancelled = false;
    setUmr(null); setUir(null); setError(""); setView("umr");
    window.MonadLabsAPI.umr(jobId)
      .then((text) => { if (!cancelled) setUmr(text); })
      .catch((e) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, [jobId]);

  // JSON only when asked for.
  React.useEffect(() => {
    if (view !== "json" || uir !== null) return;
    let cancelled = false;
    window.MonadLabsAPI.result(jobId)
      .then((doc) => { if (!cancelled) setUir(doc); })
      .catch((e) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, [view, jobId, uir]);

  const meta = job.result || {};
  const intent = job.intent;
  const shownText = view === "umr" ? umr : uir ? JSON.stringify(uir, null, 2) : null;

  async function copy() {
    if (!shownText) return;
    try {
      await navigator.clipboard.writeText(shownText);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      setError("Clipboard write was blocked by the browser.");
    }
  }

  const routeLabel = meta.source_route ? (ROUTE_LABELS[meta.source_route] || meta.source_route) : null;

  return (
    <div style={{ fontFamily: "var(--font-text)", width: "100%", maxWidth: 900 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12, gap: 16, flexWrap: "wrap" }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontFamily: "var(--font-display)", fontSize: 18, fontWeight: 600, color: "var(--text-ink)", display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{job.filename || "UIR v1.0"}</span>
            {meta.source_format && (
              <Badge kind="neutral">
                {meta.source_format}{routeLabel ? ` · ${routeLabel}` : ""}
              </Badge>
            )}
          </div>
          <div style={{ color: "var(--text-muted-48)", fontSize: "var(--text-caption-size)", marginTop: 2 }}>
            {meta.chunk_count != null ? `${meta.chunk_count} chunks` : "—"}
            {meta.entity_count != null ? ` · ${meta.entity_count} entities` : ""}
            {meta.elapsed_seconds != null ? ` · ${Number(meta.elapsed_seconds).toFixed(2)}s` : ""}
            {intent ? ` · ${intent.matched_chunks}/${intent.total_chunks} chunks match “${intent.query}”` : ""}
            {intent && intent.no_match_fallback ? " · expanded to full document (no keyword hit)" : ""}
          </div>
        </div>

        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <Tabs
            tabs={[{ value: "umr", label: "UMR" }, { value: "json", label: "UIR JSON" }]}
            active={view}
            onChange={setView}
          />
          <Button variant="dark-utility" onClick={copy} disabled={!shownText}>
            {copied ? "Copied" : "Copy"}
          </Button>
          <a
            href={window.MonadLabsAPI.downloadUrl(jobId)}
            download
            style={{
              background: "var(--text-ink)", color: "var(--text-on-dark)",
              borderRadius: "var(--radius-sm)", padding: "8px 15px",
              fontSize: "var(--text-button-utility-size)", textDecoration: "none",
            }}
          >
            Download
          </a>
        </div>
      </div>

      {error && (
        <div role="alert" style={{ background: "var(--status-error-bg)", color: "var(--status-error)", borderRadius: "var(--radius-sm)", padding: "10px 14px", fontSize: "var(--text-caption-size)", marginBottom: 10 }}>
          {error}
        </div>
      )}

      <pre style={{ ...PRE_STYLE, borderLeft: view === "umr" ? "3px solid var(--accent-primary-on-dark)" : "3px solid transparent" }}>
        {shownText === null ? "Loading…" : shownText}
      </pre>
    </div>
  );
}

window.ConsoleResultViewer = ResultViewer;

})();
