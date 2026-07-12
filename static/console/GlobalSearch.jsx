/* GlobalSearch.jsx -- a command-palette-style document search overlay.
 *
 * Opened from the IconRail's search button (works from any tab). Debounced
 * POST /api/search ranks passages across every converted document by content
 * semantics + title (title matches first, badged). Clicking a result opens
 * the source document in the file browser (the caller's onOpenDocument swaps
 * to the Upload tab and selects the file). Esc closes.
 *
 * Apple design: parchment surface, rounded.lg, the single Action Blue accent,
 * no decorative shadows (a soft ring separates the panel from the scrim).
 *
 * IIFE-wrapped: see app.jsx.
 */

(function () {

const API = window.MonadLabsAPI;
const { Badge } = window.ApertureDesignSystem_0a9afd;

function snippet(text, max = 140) {
  const s = (text || "").replace(/\s+/g, " ").trim();
  return s.length > max ? s.slice(0, max) + "…" : s;
}

function GlobalSearch({ open, onClose, onOpenDocument }) {
  const [q, setQ] = React.useState("");
  const [results, setResults] = React.useState([]);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState("");
  const inputRef = React.useRef(null);
  const timer = React.useRef(null);

  // Focus + reset on open; ignore Esc/close when not open.
  React.useEffect(() => {
    if (open) {
      setQ(""); setResults([]); setError(""); setLoading(false);
      setTimeout(() => inputRef.current && inputRef.current.focus(), 10);
    }
  }, [open]);

  // Debounced search.
  React.useEffect(() => {
    if (!open) return;
    const query = q.trim();
    if (!query) { setResults([]); setLoading(false); setError(""); return; }
    setLoading(true); setError("");
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      API.search(query)
        .then(({ results: rs }) => { setResults(rs || []); setLoading(false); })
        .catch((e) => {
          if (API.isUnauthorized(e)) { window.location.reload(); return; }
          setError(e.message); setLoading(false);
        });
    }, 250);
    return () => { if (timer.current) clearTimeout(timer.current); };
  }, [q, open]);

  React.useEffect(() => {
    if (!open) return;
    function onKey(e) { if (e.key === "Escape") onClose(); }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  function pick(r) {
    if (onOpenDocument) onOpenDocument(r.job_id);
    onClose();
  }

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed", inset: 0, background: "rgba(0,0,0,0.35)",
        display: "flex", alignItems: "flex-start", justifyContent: "center",
        paddingTop: "12vh", zIndex: 200, fontFamily: "var(--font-text)",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: "100%", maxWidth: 560, background: "var(--surface-canvas)",
          borderRadius: "var(--radius-lg)", border: "1px solid var(--border-hairline)",
          boxShadow: "var(--shadow-ring)", overflow: "hidden", maxHeight: "70vh",
          display: "flex", flexDirection: "column",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "14px 16px", borderBottom: "1px solid var(--border-hairline)" }}>
          <window.LucideIcon name="search" size={18} style={{ color: "var(--text-muted-48)" }} />
          <input
            ref={inputRef}
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search your documents…"
            style={{
              flex: 1, border: "none", outline: "none", background: "transparent",
              fontFamily: "var(--font-text)", fontSize: "var(--text-body-size)",
              color: "var(--text-ink)",
            }}
          />
          <button
            onClick={onClose}
            aria-label="Close"
            style={{ border: "none", background: "transparent", cursor: "pointer", color: "var(--text-muted-48)", display: "flex" }}
          >
            <window.LucideIcon name="x" size={18} />
          </button>
        </div>

        <div style={{ overflow: "auto", flex: 1 }}>
          {!q.trim() && (
            <div style={{ padding: "28px 16px", color: "var(--text-muted-48)", fontSize: "var(--text-caption-size)", textAlign: "center" }}>
              Search by meaning or by document title. Title matches rank first.
            </div>
          )}
          {q.trim() && loading && (
            <div style={{ padding: "20px 16px", color: "var(--text-muted-48)", fontSize: "var(--text-caption-size)" }}>Searching…</div>
          )}
          {q.trim() && !loading && error && (
            <div style={{ margin: "12px 16px", color: "var(--status-error)", fontSize: "var(--text-caption-size)" }}>{error}</div>
          )}
          {q.trim() && !loading && !error && results.length === 0 && (
            <div style={{ padding: "20px 16px", color: "var(--text-muted-48)", fontSize: "var(--text-caption-size)" }}>No passages matched.</div>
          )}
          {results.map((r, i) => (
            <button
              key={`${r.job_id}-${r.chunk_id}-${i}`}
              onClick={() => pick(r)}
              style={{
                display: "flex", flexDirection: "column", gap: 4, width: "100%",
                textAlign: "left", border: "none", background: "transparent",
                cursor: "pointer", padding: "12px 16px",
                borderBottom: "1px solid var(--border-hairline)",
              }}
              onMouseEnter={(e) => { e.currentTarget.style.background = "var(--surface-parchment)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <window.LucideIcon name="file-text" size={14} style={{ color: "var(--text-muted-48)", flexShrink: 0 }} />
                <span style={{ fontSize: "var(--text-caption-size)", fontWeight: 600, color: "var(--text-ink)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>
                  {r.doc_title}
                </span>
                {r.page != null && <span style={{ fontSize: "var(--text-micro-legal-size)", color: "var(--text-muted-48)" }}>p. {r.page}</span>}
                {r.title_match && <Badge kind="neutral" style={{ padding: "2px 8px" }}>title</Badge>}
              </div>
              <div style={{ fontSize: "var(--text-caption-size)", color: "var(--text-muted-80)", lineHeight: 1.4 }}>
                {snippet(r.text)}
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

window.ConsoleGlobalSearch = GlobalSearch;

})();
