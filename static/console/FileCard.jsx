/* FileCard.jsx -- one tile in the Google-Drive-like file grid.
 *
 * Apple design: a `rounded.lg` white card with a hairline border and NO
 * chrome shadow. The only shadow in the system is `--shadow-product`, and it
 * is reserved for "product imagery" -- here, the file's rendered preview. The
 * card itself stays flat; the preview thumbnail resting on it gets the soft
 * drop. Press state is `transform: scale(0.95)`, never a colour change.
 *
 * Preview source: GET /api/thumb/<job_id> renders PDF page 1 to PNG and
 * serves images verbatim; other types 404, which swaps the <img> out for a
 * filetype-matched Lucide icon (see EXT_ICON). Processing/error cards show
 * status chips instead of a preview.
 *
 * IIFE-wrapped: see app.jsx.
 */

(function () {

const API = window.MonadLabsAPI;

/** Extension -> Lucide icon name for the no-preview fallback. */
const EXT_ICON = {
  pdf: "file-text",
  doc: "file-text", docx: "file-text",
  ppt: "file-presentation", pptx: "file-presentation",
  xls: "file-spreadsheet", xlsx: "file-spreadsheet", csv: "file-spreadsheet",
  epub: "book",
  html: "globe", htm: "globe",
  png: "image", jpg: "image", jpeg: "image", gif: "image",
  webp: "image", bmp: "image", tiff: "image", tif: "image",
  mp4: "clapperboard", avi: "clapperboard", mov: "clapperboard",
  webm: "clapperboard", mkv: "clapperboard", flv: "clapperboard",
  wmv: "clapperboard", m4v: "clapperboard",
  mp3: "music", wav: "music", m4a: "music",
  flac: "music", ogg: "music", aac: "music", wma: "music",
  txt: "file-code", md: "file-code",
  py: "file-code", js: "file-code", jsx: "file-code",
  ts: "file-code", tsx: "file-code", json: "file-code",
  xml: "file-code", ipynb: "notebook-text",
};

function extOf(name) {
  const i = String(name || "").lastIndexOf(".");
  return i >= 0 ? name.slice(i + 1).toLowerCase() : "";
}

function iconFor(name) {
  return EXT_ICON[extOf(name)] || "file";
}

/** A square preview tile: rendered thumbnail, or a filetype icon fallback. */
function PreviewTile({ file, size }) {
  const [broken, setBroken] = React.useState(false);
  const jobId = file.job && file.job.job_id;
  const canRender = file.status === "done" && jobId;

  React.useEffect(() => { setBroken(false); }, [jobId, file.status]);

  const frame = (children) => (
    <div style={{
      width: size, height: size, borderRadius: "var(--radius-md)",
      background: "var(--surface-parchment)",
      display: "flex", alignItems: "center", justifyContent: "center",
      overflow: "hidden",
    }}>
      {children}
    </div>
  );

  if (canRender && !broken) {
    return (
      <div style={{ width: size, height: size, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <img
          src={API.thumbUrl(jobId)}
          alt=""
          onError={() => setBroken(true)}
          style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain", borderRadius: "var(--radius-md)", boxShadow: "var(--shadow-product)" }}
        />
      </div>
    );
  }
  // Fallback / non-image types: a centred filetype glyph, no shadow.
  return frame(
    <window.LucideIcon name={file.status === "error" ? "file-warning" : iconFor(file.name)} size={size * 0.4} style={{ color: "var(--text-muted-80)" }} />
  );
}

function StatusChip({ status }) {
  if (status === "processing") {
    return (
      <div className="ap-spin" style={{ position: "absolute", top: 8, right: 8, width: 18, height: 18, borderRadius: "50%", border: "2px solid transparent", borderTopColor: "var(--accent-primary)", borderRightColor: "var(--accent-primary)" }} />
    );
  }
  if (status === "done") {
    return (
      <div className="ap-badge-pop" style={{ position: "absolute", top: 8, right: 8, width: 20, height: 20, borderRadius: "50%", background: "var(--status-success)", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", border: "2px solid var(--surface-canvas)" }}>
        <window.LucideIcon name="check" size={11} />
      </div>
    );
  }
  if (status === "error") {
    return (
      <div className="ap-badge-pop" style={{ position: "absolute", top: 8, right: 8, width: 20, height: 20, borderRadius: "50%", background: "var(--status-error)", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", border: "2px solid var(--surface-canvas)" }}>
        <window.LucideIcon name="x" size={11} />
      </div>
    );
  }
  return null;
}

function FileCard({ file, selected, onSelect, onDelete }) {
  const meta = (file.job && file.job.result) || {};
  const fmt = meta.source_format && meta.source_format !== "UNKNOWN" ? meta.source_format : null;
  const chunks = meta.chunk_count != null ? `${meta.chunk_count} chunks` : null;

  return (
    <button
      onClick={() => onSelect(file.id)}
      className="ap-file-icon"
      style={{
        position: "relative", display: "flex", flexDirection: "column",
        textAlign: "left", cursor: "pointer", padding: 12,
        background: "var(--surface-canvas)",
        border: `1px solid ${selected ? "var(--accent-primary)" : "var(--border-hairline)"}`,
        borderRadius: "var(--radius-lg)",
        transition: "transform var(--duration-press) var(--ease-standard), border-color var(--duration-fast) ease",
      }}
      onMouseDown={(e) => { e.currentTarget.style.transform = "scale(var(--scale-press))"; }}
      onMouseUp={(e) => { e.currentTarget.style.transform = "scale(1)"; }}
      onMouseLeave={(e) => { e.currentTarget.style.transform = "scale(1)"; }}
    >
      <div style={{ position: "relative", width: "100%", aspectRatio: "1 / 1", marginBottom: 10, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <PreviewTile file={file} size={150} />
        <StatusChip status={file.status} />
        {onDelete && (file.status === "done" || file.status === "error") && (
          <div
            className="ap-file-delete" role="button" aria-label={`Remove ${file.name}`}
            onClick={(e) => { e.stopPropagation(); onDelete(file.id); }}
            style={{ position: "absolute", bottom: 6, right: 6, width: 24, height: 24, borderRadius: "50%", background: "var(--status-error)", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", border: "2px solid var(--surface-canvas)", cursor: "pointer" }}
          >
            <window.LucideIcon name="trash-2" size={12} />
          </div>
        )}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 6, minHeight: 20 }}>
        <window.LucideIcon name={iconFor(file.name)} size={14} style={{ color: "var(--text-muted-48)", flexShrink: 0 }} />
        <span style={{ fontSize: "var(--text-caption-size)", fontWeight: 600, color: "var(--text-ink)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {file.name}
        </span>
      </div>
      <div style={{ fontSize: "var(--text-micro-legal-size)", color: "var(--text-muted-48)", marginTop: 4, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {[fmt, chunks].filter(Boolean).join(" · ") || (file.status === "processing" ? "Converting…" : "—")}
      </div>
    </button>
  );
}

window.ConsoleFileCard = FileCard;

})();
