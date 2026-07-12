/* FileDetail.jsx -- the enlarged preview + tabbed inspector for one file.
 *
 * Replaces the old center-pane <ResultViewer> usage. Layout: an enlarged
 * preview at the left (same /api/thumb source + the single --shadow-product,
 * with a filetype-icon fallback), and a tabbed panel at the right with four
 * tabs -- Metadata / UMR / UIR / Chunks -- built from the Aperture
 * design-system <Tabs>. UMR and UIR reuse the existing fetch contract
 * (/api/umr, /api/result); the Chunks tab walks the UIR's structure tree,
 * flattening the ChunkNode leaves into a scrollable list.
 *
 * IIFE-wrapped: see app.jsx.
 */

(function () {

const { Tabs, Button, Badge } = window.ApertureDesignSystem_0a9afd;
const API = window.MonadLabsAPI;

const PRE_STYLE = {
  margin: 0,
  background: "var(--surface-tile-1)",
  color: "var(--text-on-dark)",
  borderRadius: "var(--radius-sm)",
  padding: "16px 18px",
  fontFamily: "var(--font-mono)",
  fontSize: "var(--text-mono-size)",
  lineHeight: "var(--text-mono-leading)",
  maxHeight: "60vh",
  overflow: "auto",
  whiteSpace: "pre-wrap",
};

const ROUTE_LABELS = {
  pdf: "PDFplumber", docling: "Docling", pptx: "Native PPTX walker",
  text: "Text walker", image: "Fireworks AI vision", skip: "skipped",
};

const EXT_ICON = {
  pdf: "file-text", doc: "file-text", docx: "file-text",
  ppt: "file-presentation", pptx: "file-presentation",
  xls: "file-spreadsheet", xlsx: "file-spreadsheet", csv: "file-spreadsheet",
  epub: "book", html: "globe", htm: "globe",
  png: "image", jpg: "image", jpeg: "image", gif: "image",
  webp: "image", bmp: "image", tiff: "image", tif: "image",
  avif: "image", heic: "image", heif: "image",
  mp4: "clapperboard", avi: "clapperboard", mov: "clapperboard",
  webm: "clapperboard", mkv: "clapperboard", flv: "clapperboard",
  wmv: "clapperboard", m4v: "clapperboard",
  mp3: "music", wav: "music", m4a: "music",
  flac: "music", ogg: "music", aac: "music", wma: "music",
  txt: "file-code", md: "file-code", py: "file-code", js: "file-code",
  jsx: "file-code", ts: "file-code", tsx: "file-code", json: "file-code",
  xml: "file-code", ipynb: "notebook-text",
  eml: "mail", msg: "mail",
};

function extOf(name) {
  const i = String(name || "").lastIndexOf(".");
  return i >= 0 ? name.slice(i + 1).toLowerCase() : "";
}
function iconFor(name) { return EXT_ICON[extOf(name)] || "file"; }

/** Recursively collect every ChunkNode under a structure node. */
function flattenChunks(node) {
  if (!node) return [];
  if (node.type === "chunk") return [node];
  const children = node.children || node.items || [];
  return (Array.isArray(children) ? children : []).flatMap(flattenChunks);
}

const _VIDEO_EXT = new Set(["mp4", "avi", "mov", "webm", "mkv", "flv", "wmv", "m4v"]);
const _AUDIO_EXT = new Set(["mp3", "wav", "m4a", "flac", "ogg", "aac", "wma"]);

function isVideo(name) { return _VIDEO_EXT.has(extOf(name)); }
function isAudio(name) { return _AUDIO_EXT.has(extOf(name)); }

/** Big preview: media player for video/audio, thumbnail for images/PDFs, icon on error. */
function BigPreview({ job, name }) {
  const [broken, setBroken] = React.useState(false);
  const jobId = job && job.job_id;
  React.useEffect(() => { setBroken(false); }, [jobId]);

  if (!jobId || broken) {
    return (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
        gap: 10, width: "100%", minHeight: 220, maxHeight: "46vh", padding: 24,
        background: "var(--surface-parchment)", borderRadius: "var(--radius-lg)", color: "var(--text-muted-48)" }}>
        <window.LucideIcon name={iconFor(name)} size={64} />
        <span style={{ fontSize: "var(--text-caption-size)" }}>No graphical preview for this file type</span>
      </div>
    );
  }

  if (isVideo(name)) {
    return (
      <div style={{ width: "100%", padding: 16, background: "var(--surface-parchment)", borderRadius: "var(--radius-lg)" }}>
        <video
          src={API.originalUrl(jobId)}
          controls
          style={{ width: "100%", maxHeight: "46vh", borderRadius: "var(--radius-sm)", boxShadow: "var(--shadow-product)" }}
          poster={API.thumbUrl(jobId)}
        />
      </div>
    );
  }

  if (isAudio(name)) {
    return (
      <div style={{ width: "100%", padding: 24, background: "var(--surface-parchment)", borderRadius: "var(--radius-lg)" }}>
        <audio src={API.originalUrl(jobId)} controls style={{ width: "100%" }} />
      </div>
    );
  }

  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center",
      width: "100%", minHeight: 220, maxHeight: "46vh", padding: 16,
      background: "var(--surface-parchment)", borderRadius: "var(--radius-lg)" }}>
      <img
        src={API.thumbUrl(jobId)} alt=""
        onError={() => setBroken(true)}
        style={{ maxWidth: "100%", maxHeight: "46vh", objectFit: "contain", borderRadius: "var(--radius-sm)", boxShadow: "var(--shadow-product)" }}
      />
    </div>
  );
}

function MetaRow({ label, children }) {
  return (
    <div style={{ display: "flex", gap: 12, padding: "8px 0", borderBottom: "1px solid var(--border-hairline)" }}>
      <div style={{ width: 130, flexShrink: 0, color: "var(--text-muted-48)", fontSize: "var(--text-caption-size)" }}>{label}</div>
      <div style={{ color: "var(--text-ink)", fontSize: "var(--text-caption-size)", minWidth: 0 }}>{children}</div>
    </div>
  );
}

function MetadataTab({ job, folders }) {
  const meta = (job && job.result) || {};
  const intent = job && job.intent;
  const routeLabel = meta.source_route ? (ROUTE_LABELS[meta.source_route] || meta.source_route) : null;
  const folder = folders && folders.find((f) => f.id === (job && job.folder_id));
  const fmt = (t) => (t == null ? "—" : new Date(t * 1000).toLocaleString());

  return (
    <div style={{ fontFamily: "var(--font-text)" }}>
      <MetaRow label="Filename">{job && job.filename ? job.filename : "—"}</MetaRow>
      <MetaRow label="Format">
        {meta.source_format && meta.source_format !== "UNKNOWN" ? (
          <Badge kind="neutral">{meta.source_format}{routeLabel ? ` · ${routeLabel}` : ""}</Badge>
        ) : "—"}
      </MetaRow>
      <MetaRow label="Chunks">{meta.chunk_count != null ? meta.chunk_count : "—"}</MetaRow>
      <MetaRow label="Entities">{meta.entity_count != null ? meta.entity_count : "—"}</MetaRow>
      <MetaRow label="Elapsed">{meta.elapsed_seconds != null ? `${Number(meta.elapsed_seconds).toFixed(2)}s` : "—"}</MetaRow>
      <MetaRow label="Submitted">{fmt(job && job.submitted_at)}</MetaRow>
      <MetaRow label="Finished">{fmt(job && job.finished_at)}</MetaRow>
      <MetaRow label="Folder">{folder ? folder.name : "All files"}</MetaRow>
      {intent && (
        <MetaRow label="Intent">
          {intent.matched_chunks}/{intent.total_chunks} chunks match “{intent.query}”
          {intent.no_match_fallback ? " · expanded to full document" : ""}
        </MetaRow>
      )}
      {job && job.error && <MetaRow label="Error"><span style={{ color: "var(--status-error)" }}>{job.error}</span></MetaRow>}
    </div>
  );
}

function ChunksTab({ jobId }) {
  const [uir, setUir] = React.useState(null);
  const [error, setError] = React.useState("");

  React.useEffect(() => {
    let cancelled = false;
    setUir(null); setError("");
    API.result(jobId)
      .then((doc) => { if (!cancelled) setUir(doc); })
      .catch((e) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, [jobId]);

  if (error) return <div style={{ color: "var(--status-error)", fontSize: "var(--text-caption-size)" }}>{error}</div>;
  if (!uir) return <div style={{ color: "var(--text-muted-48)", fontSize: "var(--text-caption-size)" }}>Loading…</div>;

  const root = (uir.structure && (uir.structure.root || uir.structure)) || uir.structure;
  const chunks = flattenChunks(root);
  if (!chunks.length) return <div style={{ color: "var(--text-muted-48)", fontSize: "var(--text-caption-size)" }}>No chunks.</div>;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10, fontFamily: "var(--font-text)" }}>
      {chunks.map((c, i) => {
        const mf = c.modal_features || {};
        const videoSeg = mf.video_segment || {};
        const visualFrames = videoSeg.visual_frames || [];
        const speaker = videoSeg.speaker;
        const start = videoSeg.start;
        const end = videoSeg.end;
        const hasVideo = visualFrames.length > 0 || (start != null && end != null);
        const hasAudio = c.text && !hasVideo; // crude heuristic: video chunks have visual frames

        return (
          <div key={c.id || i} style={{ background: "var(--surface-parchment)", border: "1px solid var(--border-hairline)", borderRadius: "var(--radius-sm)", padding: "12px 14px" }}>
            <div style={{ display: "flex", gap: 10, alignItems: "center", fontSize: "var(--text-micro-legal-size)", color: "var(--text-muted-48)", marginBottom: 6 }}>
              <Badge kind="neutral">p. {c.page != null ? c.page : "?"}</Badge>
              <span>{c.token_count != null ? `${c.token_count} tokens` : ""}</span>
              {c.confidence != null && <span>· conf {Number(c.confidence).toFixed(2)}</span>}
              {hasVideo && (
                <span style={{ marginLeft: "auto", fontWeight: 600, color: "var(--accent-primary)" }}>
                  {start != null && end != null ? `${Math.floor(start / 60)}:${Math.floor(start % 60).toString().padStart(2, "0")} - ${Math.floor(end / 60)}:${Math.floor(end % 60).toString().padStart(2, "0")}` : ""}
                  {speaker && speaker !== "UNKNOWN" ? ` · ${speaker}` : ""}
                </span>
              )}
            </div>

            {/* Visual frames: rendered as distinct annotation blocks */}
            {visualFrames.length > 0 && (
              <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 8 }}>
                {visualFrames.map((vf, j) => {
                  const ts = vf.timestamp;
                  const mins = Math.floor(ts / 60);
                  const secs = Math.floor(ts % 60).toString().padStart(2, "0");
                  return (
                    <div key={j} style={{
                      display: "flex", alignItems: "center", gap: 8,
                      padding: "6px 10px", background: "var(--blue-50)", borderRadius: "var(--radius-xs)",
                      fontSize: "var(--text-caption-size)", color: "var(--text-ink)",
                    }}>
                      <window.LucideIcon name="clapperboard" size={14} style={{ color: "var(--accent-primary)", flexShrink: 0 }} />
                      <span style={{ fontWeight: 600, color: "var(--accent-primary)", flexShrink: 0 }}>{mins}:{secs}</span>
                      <span>{vf.description}</span>
                    </div>
                  );
                })}
              </div>
            )}

            <pre style={{ ...PRE_STYLE, background: "var(--surface-canvas)", color: "var(--text-ink)", border: "1px solid var(--border-hairline)", maxHeight: 180 }}>{c.text}</pre>
          </div>
        );
      })}
    </div>
  );
}

function CodeTab({ jobId, mode }) {
  // mode === "umr" fetches markdown text; "uir" fetches JSON and pretty-prints.
  const [text, setText] = React.useState(null);
  const [error, setError] = React.useState("");

  React.useEffect(() => {
    let cancelled = false;
    setText(null); setError("");
    const p = mode === "umr" ? API.umr(jobId) : API.result(jobId).then((d) => JSON.stringify(d, null, 2));
    p.then((t) => { if (!cancelled) setText(t); })
     .catch((e) => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; };
  }, [jobId, mode]);

  if (error) return <div style={{ color: "var(--status-error)", fontSize: "var(--text-caption-size)" }}>{error}</div>;
  if (text === null) return <div style={{ color: "var(--text-muted-48)", fontSize: "var(--text-caption-size)" }}>Loading…</div>;
  return <pre style={PRE_STYLE}>{text}</pre>;
}

function FileDetail({ file, folders, onBack, onDelete, onMoveFile }) {
  const [view, setView] = React.useState("metadata");
  const [copied, setCopied] = React.useState(false);
  const job = file.job;
  const jobId = job && job.job_id;
  const meta = (job && job.result) || {};
  const routeLabel = meta.source_route ? (ROUTE_LABELS[meta.source_route] || meta.source_route) : null;

  async function copy() {
    let text;
    if (view === "umr") text = await API.umr(jobId).catch(() => null);
    else if (view === "uir") text = JSON.stringify(await API.result(jobId).catch(() => null), null, 2);
    else if (view === "chunks") {
      const doc = await API.result(jobId).catch(() => null);
      text = doc ? JSON.stringify(flattenChunks(doc.structure && (doc.structure.root || doc.structure) || doc.structure), null, 2) : null;
    } else { text = JSON.stringify(job, null, 2); }
    if (!text) return;
    try { await navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 1200); }
    catch { /* clipboard blocked -- ignore */ }
  }

  const canDownload = file.status === "done" && jobId;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", width: "100%", fontFamily: "var(--font-text)" }}>
      {/* Toolbar */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "12px 20px", borderBottom: "1px solid var(--border-hairline)", flexShrink: 0 }}>
        <button onClick={onBack} title="Back to files"
          style={{ border: "none", background: "transparent", cursor: "pointer", padding: 4, color: "var(--accent-primary)", display: "flex" }}>
          <window.LucideIcon name="arrow-left" size={20} />
        </button>
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, overflow: "hidden" }}>
            <span style={{ fontFamily: "var(--font-display)", fontSize: 18, fontWeight: 600, color: "var(--text-ink)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {file.name}
            </span>
            {meta.source_format && meta.source_format !== "UNKNOWN" && (
              <Badge kind="neutral">{meta.source_format}{routeLabel ? ` · ${routeLabel}` : ""}</Badge>
            )}
          </div>
          <div style={{ fontSize: "var(--text-caption-size)", color: "var(--text-muted-48)", marginTop: 2 }}>
            {meta.chunk_count != null ? `${meta.chunk_count} chunks` : ""}
            {meta.entity_count != null ? ` · ${meta.entity_count} entities` : ""}
            {meta.elapsed_seconds != null ? ` · ${Number(meta.elapsed_seconds).toFixed(2)}s` : ""}
          </div>
        </div>
        <Button variant="dark-utility" onClick={copy} disabled={file.status !== "done"}>
          {copied ? "Copied" : "Copy"}
        </Button>
        {canDownload && (
          <a
            href={API.downloadUrl(jobId)} download
            style={{
              background: "var(--text-ink)", color: "var(--text-on-dark)",
              borderRadius: "var(--radius-sm)", padding: "8px 15px",
              fontSize: "var(--text-button-utility-size)", textDecoration: "none",
            }}
          >Download</a>
        )}
        {onDelete && (
          <button onClick={() => onDelete(file.id)} title="Delete file"
            style={{ border: "none", background: "transparent", cursor: "pointer", padding: 4, color: "var(--status-error)", display: "flex" }}>
            <window.LucideIcon name="trash-2" size={18} />
          </button>
        )}
      </div>

      {/* Body: preview + tabs */}
      <div style={{ flex: 1, display: "flex", gap: 20, padding: 20, overflow: "auto", alignItems: "flex-start" }}>
        <div style={{ flex: "0 1 360px", minWidth: 240, position: "sticky", top: 20 }}>
          {file.status === "processing" && (
            <div style={{ textAlign: "center", padding: 40, color: "var(--text-muted-48)" }}>
              <div className="ap-spin" style={{ width: 36, height: 36, margin: "0 auto 12px", borderRadius: "50%", border: "3px solid transparent", borderTopColor: "var(--accent-primary)", borderRightColor: "var(--accent-primary)" }} />
              <div style={{ fontSize: "var(--text-caption-size)" }}>{file.stage || "Converting"} · {file.percent || 0}%</div>
            </div>
          )}
          {file.status === "error" && (
            <div style={{ padding: 20, background: "var(--status-error-bg)", borderRadius: "var(--radius-lg)", color: "var(--status-error)" }}>
              <div style={{ fontWeight: 600, marginBottom: 6 }}>Failed to convert</div>
              <pre style={{ margin: 0, whiteSpace: "pre-wrap", fontSize: "var(--text-micro-legal-size)" }}>{file.error || "Unknown error"}</pre>
            </div>
          )}
          {file.status === "done" && <BigPreview job={job} name={file.name} />}

          {file.status === "done" && onMoveFile && (
            <div style={{ marginTop: 12 }}>
              <label style={{ fontSize: "var(--text-micro-legal-size)", color: "var(--text-muted-48)", display: "block", marginBottom: 4 }}>Move to folder</label>
              <select
                value={(job && job.folder_id) == null ? "" : String(job.folder_id)}
                onChange={(e) => onMoveFile(file.id, e.target.value === "" ? null : Number(e.target.value))}
                style={{ width: "100%", padding: "8px 10px", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-hairline)", background: "var(--surface-canvas)", color: "var(--text-ink)", fontSize: "var(--text-caption-size)" }}
              >
                <option value="">All files</option>
                {(folders || []).map((f) => <option key={f.id} value={String(f.id)}>{f.name}</option>)}
              </select>
            </div>
          )}
        </div>

        <div style={{ flex: 1, minWidth: 0 }}>
          {file.status === "done" && (
            <>
              <div style={{ marginBottom: 14 }}>
                <Tabs
                  tabs={[
                    { value: "metadata", label: "Metadata" },
                    { value: "umr", label: "UMR" },
                    { value: "uir", label: "UIR" },
                    { value: "chunks", label: "Chunks" },
                  ]}
                  active={view}
                  onChange={setView}
                />
              </div>
              {view === "metadata" && <MetadataTab job={job} folders={folders} />}
              {view === "umr" && <CodeTab jobId={jobId} mode="umr" />}
              {view === "uir" && <CodeTab jobId={jobId} mode="uir" />}
              {view === "chunks" && <ChunksTab jobId={jobId} />}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

window.ConsoleFileDetail = FileDetail;

})();
