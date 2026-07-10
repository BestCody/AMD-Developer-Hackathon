/* UploadStage.jsx -- dropzone, converting animation, document folder.
 *
 * This is the component the design kit faked. Its original `pickFakeFiles()`
 * ignored the change/drop event entirely and appended a name from a
 * hardcoded pool (`invoice-q3.pdf`, `board-deck.pdf`, ...) indexed by
 * `files.length % pool.length`, then a setTimeout flipped it to "done" after
 * 2.2s. Upload contract.pdf, watch invoice-q3.pdf appear.
 *
 * Now: the real File goes to POST /api/run, and the converging-rings
 * animation reports the pipeline's actual stage and percentage from
 * /api/status. The success badge pops when the job really finishes.
 *
 * IIFE-wrapped: see app.jsx.
 */

(function () {

const UPLOAD_STAGE_LABELS = {
  queued: "Queued",
  ingest: "Ingesting",
  route: "Routing format",
  extract: "Extracting layout",
  tables: "Extracting tables",
  chunk: "Chunking",
  enrich: "Enriching",
  embed: "Embedding",
  done: "Finishing up",
};

function stageLabel(stage) {
  if (!stage) return "Working";
  return UPLOAD_STAGE_LABELS[stage] || stage.charAt(0).toUpperCase() + stage.slice(1);
}

function ConvertingAnimation({ file }) {
  const pct = Math.max(0, Math.min(100, file.percent || 0));
  return (
    <div style={{ textAlign: "center" }}>
      <div style={{ position: "relative", width: 96, height: 96, margin: "0 auto 24px" }}>
        <div className="ap-ring ap-ring--1" />
        <div className="ap-ring ap-ring--2" />
        <div className="ap-ring ap-ring--3" />
        <img
          src="/static/ds/assets/logo/aperture-mark.png"
          alt=""
          style={{ width: 34, height: 34, borderRadius: 8, position: "absolute", top: "50%", left: "50%", transform: "translate(-50%,-50%)" }}
        />
      </div>
      <div style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-tagline-size)", fontWeight: 600, color: "var(--text-ink)" }}>
        Converting {file.name}
      </div>
      {/* Real stage + percent from /api/status, not a decorative caption. */}
      <div style={{ fontSize: "var(--text-caption-size)", color: "var(--text-muted-48)", marginTop: 6 }}>
        {stageLabel(file.stage)} · {pct}%
      </div>
      <div style={{ width: 260, height: 3, background: "var(--gray-100)", borderRadius: 2, margin: "14px auto 0", overflow: "hidden" }}>
        <div
          style={{
            width: `${pct}%`, height: "100%", background: "var(--accent-primary)",
            transition: "width var(--duration-standard) var(--ease-standard)",
          }}
        />
      </div>
    </div>
  );
}

function FileIcon({ file, selected, onClick, onDelete }) {
  return (
    <button onClick={onClick} className="ap-file-icon" style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6, border: "none", background: "transparent", cursor: "pointer", padding: 4 }}>
      <div style={{ position: "relative", width: 56, height: 56 }}>
        <div
          style={{
            width: 56, height: 56, borderRadius: "var(--radius-md)",
            background: "var(--surface-canvas)",
            boxShadow: selected ? "0 0 0 2px var(--accent-primary)" : "var(--shadow-ring)",
            display: "flex", alignItems: "center", justifyContent: "center",
            color: "var(--text-muted-80)",
          }}
        >
          <i data-lucide={file.status === "error" ? "file-warning" : "file-text"} style={{ width: 22, height: 22 }}></i>
        </div>

        {file.status === "processing" && (
          <div className="ap-spin" style={{ position: "absolute", inset: -3, borderRadius: "var(--radius-md)", border: "2px solid transparent", borderTopColor: "var(--accent-primary)", borderRightColor: "var(--accent-primary)" }} />
        )}

        {file.status === "done" && (
          <div className="ap-badge-pop" style={{ position: "absolute", bottom: -4, right: -4, width: 20, height: 20, borderRadius: "50%", background: "var(--status-success)", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", border: "2px solid var(--surface-parchment)" }}>
            <i data-lucide="check" style={{ width: 11, height: 11 }}></i>
          </div>
        )}

        {file.status === "error" && (
          <div className="ap-badge-pop" style={{ position: "absolute", bottom: -4, right: -4, width: 20, height: 20, borderRadius: "50%", background: "var(--status-error)", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", border: "2px solid var(--surface-parchment)" }}>
            <i data-lucide="x" style={{ width: 11, height: 11 }}></i>
          </div>
        )}

        <div
          className="ap-file-delete" role="button" aria-label={`Remove ${file.name}`}
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          style={{ position: "absolute", top: -6, right: -6, width: 22, height: 22, borderRadius: "50%", background: "var(--status-error)", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", border: "2px solid var(--surface-parchment)", cursor: "pointer" }}
        >
          <img src="/static/ds/assets/icons/trash-alpha.png" alt="" style={{ width: 12, height: 12, display: "block" }} />
        </div>
      </div>
      <span style={{ fontSize: 11, color: "var(--text-muted-80)", textAlign: "center", maxWidth: 70, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {file.name}
      </span>
    </button>
  );
}

function UploadStage({ files, selectedId, onSelectFile, onAddFiles, onDeleteFile }) {
  const [dragging, setDragging] = React.useState(false);
  const [folderWidth, setFolderWidth] = React.useState(240);
  const resizing = React.useRef(false);
  const inputRef = React.useRef(null);

  const hasFiles = files.length > 0;
  const selected = files.find((f) => f.id === selectedId) || null;
  const maxMb = (window.MONADLABS_CONFIG && window.MONADLABS_CONFIG.maxUploadMb) || 64;

  React.useEffect(() => { if (window.lucide) window.lucide.createIcons(); });

  React.useEffect(() => {
    function onMove(e) {
      if (!resizing.current) return;
      setFolderWidth(Math.min(480, Math.max(180, window.innerWidth - e.clientX)));
    }
    function onUp() {
      resizing.current = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    }
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  function startResize(e) {
    e.preventDefault();
    resizing.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }

  // These two are the whole point: hand the actual FileList upward.
  function handleInputChange(e) {
    const picked = Array.from(e.target.files || []);
    if (picked.length) onAddFiles(picked);
    e.target.value = ""; // let the same file be re-picked
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    const dropped = Array.from((e.dataTransfer && e.dataTransfer.files) || []);
    if (dropped.length) onAddFiles(dropped);
  }

  return (
    <div style={{ display: "flex", height: "100%", width: "100%" }}>
      {/* One input for the whole stage. It must stay mounted regardless of
          which pane is showing, because the folder's "Add file" button
          clicks it too. */}
      <input ref={inputRef} type="file" multiple hidden onChange={handleInputChange} />

      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: 40, overflow: "auto" }}>
        {!selected && (
          <div style={{ display: "flex", flexDirection: "column", gap: 18, width: 460 }}>
            <div
              role="button"
              tabIndex={0}
              onClick={() => inputRef.current && inputRef.current.click()}
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); inputRef.current && inputRef.current.click(); } }}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              style={{
                display: "flex", flexDirection: "column", alignItems: "center", gap: 16,
                padding: "56px 40px",
                border: `1.5px dashed ${dragging ? "var(--accent-primary)" : "var(--border-hairline)"}`,
                borderRadius: "var(--radius-lg)",
                background: dragging ? "var(--blue-50)" : "var(--surface-canvas)",
                cursor: "pointer",
                transition: "border-color var(--duration-fast) ease, background var(--duration-fast) ease",
              }}
            >
              <div style={{ width: 64, height: 64, borderRadius: "var(--radius-full)", background: "var(--blue-50)", color: "var(--accent-primary)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <i data-lucide="upload-cloud" style={{ width: 28, height: 28 }}></i>
              </div>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-tagline-size)", fontWeight: 600, color: "var(--text-ink)" }}>
                  Drop a file to convert it
                </div>
                <div style={{ fontSize: "var(--text-caption-size)", color: "var(--text-muted-48)", marginTop: 4 }}>
                  PDF · DOCX · PPTX · XLSX · EPUB · HTML · images · text — max {maxMb} MB
                </div>
              </div>
            </div>

          </div>
        )}

        {selected && selected.status === "processing" && <ConvertingAnimation file={selected} />}

        {selected && selected.status === "error" && (
          <div style={{ textAlign: "center", maxWidth: 520 }}>
            <div style={{ width: 64, height: 64, margin: "0 auto 16px", borderRadius: "var(--radius-full)", background: "var(--status-error-bg)", color: "var(--status-error)", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <i data-lucide="x" style={{ width: 28, height: 28 }}></i>
            </div>
            <div style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-tagline-size)", fontWeight: 600, color: "var(--text-ink)" }}>
              {selected.name} failed to convert
            </div>
            {/* The real pipeline error, verbatim. Swallowing it here would make
                every failure look identical and undebuggable. */}
            <pre style={{ textAlign: "left", marginTop: 14, background: "var(--surface-parchment)", border: "1px solid var(--border-hairline)", borderRadius: "var(--radius-sm)", padding: "12px 14px", fontFamily: "var(--font-mono)", fontSize: "var(--text-mono-size)", color: "var(--status-error)", whiteSpace: "pre-wrap", overflow: "auto", maxHeight: 200 }}>
              {selected.error || "Unknown error"}
            </pre>
          </div>
        )}

        {selected && selected.status === "done" && <window.ConsoleResultViewer job={selected.job} />}
      </div>

      {hasFiles && (
        <div onMouseDown={startResize} style={{ width: 6, flexShrink: 0, cursor: "col-resize", position: "relative", zIndex: 2 }}>
          <div style={{ position: "absolute", top: 0, bottom: 0, left: "50%", width: 1, background: "var(--border-hairline)" }} />
        </div>
      )}

      {hasFiles && (
        <aside style={{ width: folderWidth, flexShrink: 0, background: "var(--surface-parchment)", padding: "24px 16px", overflow: "auto" }}>
          <div style={{ fontSize: "var(--text-caption-strong-size)", fontWeight: 600, color: "var(--text-muted-48)", marginBottom: 14, padding: "0 4px" }}>
            Documents
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(64px, 1fr))", gap: 14 }}>
            {files.map((f) => (
              <FileIcon
                key={f.id} file={f} selected={f.id === selectedId}
                onClick={() => onSelectFile(f.id)}
                onDelete={() => onDeleteFile(f.id)}
              />
            ))}
            <button
              onClick={() => inputRef.current && inputRef.current.click()}
              aria-label="Add another file"
              style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6, border: "none", background: "transparent", cursor: "pointer", padding: 4 }}
            >
              <div style={{ width: 56, height: 56, borderRadius: "var(--radius-md)", border: "1.5px dashed var(--border-hairline)", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--accent-primary)" }}>
                <i data-lucide="plus" style={{ width: 20, height: 20 }}></i>
              </div>
              <span style={{ fontSize: 11, color: "var(--text-muted-48)" }}>Add file</span>
            </button>
          </div>
        </aside>
      )}
    </div>
  );
}

window.ConsoleUploadStage = UploadStage;

})();
