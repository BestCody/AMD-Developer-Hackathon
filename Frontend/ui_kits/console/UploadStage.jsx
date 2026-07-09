/** Upload stage — empty dropzone, converting animation, and the side "folder" of file icons. */
function UploadStage({ files, selectedId, onSelectFile, onAddFiles, onDeleteFile }) {
  const [dragging, setDragging] = React.useState(false);
  const [folderWidth, setFolderWidth] = React.useState(220);
  const resizing = React.useRef(false);
  const hasFiles = files.length > 0;
  const selected = files.find((f) => f.id === selectedId) || null;
  const anyProcessing = files.some((f) => f.status === "processing");

  function startResize(e) {
    e.preventDefault();
    resizing.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }

  React.useEffect(() => {
    function onMove(e) {
      if (!resizing.current) return;
      const distFromRight = window.innerWidth - e.clientX;
      const next = Math.min(480, Math.max(180, distFromRight));
      setFolderWidth(next);
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

  function pickFakeFiles() {
    const pool = ["invoice-q3.pdf", "board-deck.pdf", "support-call.mp4", "architecture.png", "release-notes.docx"];
    const name = pool[files.length % pool.length];
    onAddFiles([name]);
  }

  return (
    <div style={{ display: "flex", height: "100%", width: "100%" }}>
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: 40 }}>
        {!selected && (
          <label
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); pickFakeFiles(); }}
            style={{
              width: 420,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 16,
              padding: "56px 40px",
              border: `1.5px dashed ${dragging ? "var(--accent-primary)" : "var(--border-hairline)"}`,
              borderRadius: "var(--radius-lg)",
              background: dragging ? "var(--blue-50)" : "var(--surface-canvas)",
              cursor: "pointer",
              transition: "border-color var(--duration-fast) ease, background var(--duration-fast) ease",
            }}
          >
            <input type="file" multiple hidden onChange={pickFakeFiles} />
            <div style={{ width: 64, height: 64, borderRadius: "var(--radius-full)", background: "var(--blue-50)", color: "var(--accent-primary)", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <i data-lucide="upload-cloud" style={{ width: 28, height: 28 }}></i>
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-tagline-size)", fontWeight: 600, color: "var(--text-ink)" }}>
                Drop a file to convert it
              </div>
              <div style={{ fontSize: "var(--text-caption-size)", color: "var(--text-muted-48)", marginTop: 4 }}>
                PDFs, videos, screenshots, etc.
              </div>
            </div>
          </label>
        )}

        {selected && selected.status === "processing" && <ConvertingAnimation name={selected.name} />}

        {selected && selected.status === "done" && (
          <div style={{ textAlign: "center", maxWidth: 360 }}>
            <div style={{ width: 64, height: 64, margin: "0 auto 16px", borderRadius: "var(--radius-full)", background: "var(--status-success-bg)", color: "var(--status-success)", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <i data-lucide="check" style={{ width: 28, height: 28 }}></i>
            </div>
            <div style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-tagline-size)", fontWeight: 600, color: "var(--text-ink)" }}>{selected.name}</div>
            <div style={{ fontSize: "var(--text-caption-size)", color: "var(--text-muted-48)", marginTop: 4 }}>
              Converted to UIR · ready for your agent to query
            </div>
          </div>
        )}

      </div>

      {hasFiles && (
        <div
          onMouseDown={startResize}
          style={{
            width: 6, flexShrink: 0, cursor: "col-resize", background: "transparent",
            position: "relative", zIndex: 2,
          }}
        >
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
              <FileIcon key={f.id} file={f} selected={f.id === selectedId} onClick={() => onSelectFile(f.id)} onDelete={() => onDeleteFile(f.id)} />
            ))}
            <button
              onClick={pickFakeFiles}
              aria-label="Add another file"
              style={{
                display: "flex", flexDirection: "column", alignItems: "center", gap: 6,
                border: "none", background: "transparent", cursor: "pointer", padding: 4,
              }}
            >
              <div
                style={{
                  width: 56, height: 56, borderRadius: "var(--radius-md)",
                  border: "1.5px dashed var(--border-hairline)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  color: "var(--accent-primary)",
                }}
              >
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

function FileIcon({ file, selected, onClick, onDelete }) {
  const badgeShown = file.status === "done";
  return (
    <button
      onClick={onClick}
      className="ap-file-icon"
      style={{
        display: "flex", flexDirection: "column", alignItems: "center", gap: 6,
        border: "none", background: "transparent", cursor: "pointer", padding: 4,
      }}
    >
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
          <i data-lucide="file-text" style={{ width: 22, height: 22 }}></i>
        </div>
        {file.status === "processing" && (
          <div className="ap-spin" style={{
            position: "absolute", inset: -3, borderRadius: "var(--radius-md)",
            border: "2px solid transparent", borderTopColor: "var(--accent-primary)", borderRightColor: "var(--accent-primary)",
          }} />
        )}
        {badgeShown && (
          <div
            className="ap-badge-pop"
            style={{
              position: "absolute", bottom: -4, right: -4, width: 20, height: 20, borderRadius: "50%",
              background: "var(--status-success)", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center",
              border: "2px solid var(--surface-parchment)",
            }}
          >
            <i data-lucide="check" style={{ width: 11, height: 11 }}></i>
          </div>
        )}
        <div
          className="ap-file-delete"
          role="button"
          aria-label={`Delete ${file.name}`}
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          style={{
            position: "absolute", top: -6, right: -6, width: 22, height: 22, borderRadius: "50%",
            background: "var(--status-error)", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center",
            border: "2px solid var(--surface-parchment)", cursor: "pointer",
          }}
        >
          <img
            src="../../assets/icons/trash-alpha.png"
            alt=""
            style={{ width: 12, height: 12, display: "block" }}
          />
        </div>
      </div>
      <span style={{ fontSize: 11, color: "var(--text-muted-80)", textAlign: "center", maxWidth: 70, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {file.name}
      </span>
    </button>
  );
}

function ConvertingAnimation({ name }) {
  return (
    <div style={{ textAlign: "center" }}>
      <div style={{ position: "relative", width: 96, height: 96, margin: "0 auto 24px" }}>
        <div className="ap-ring ap-ring--1" />
        <div className="ap-ring ap-ring--2" />
        <div className="ap-ring ap-ring--3" />
        <img src="../../assets/logo/aperture-mark.png" alt="" style={{ width: 34, height: 34, borderRadius: 8, position: "absolute", top: "50%", left: "50%", transform: "translate(-50%,-50%)" }} />
      </div>
      <div style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-tagline-size)", fontWeight: 600, color: "var(--text-ink)" }}>
        Converting {name}
      </div>
      <div style={{ fontSize: "var(--text-caption-size)", color: "var(--text-muted-48)", marginTop: 4 }}>
        Layout → tables → chunks → embeddings
      </div>
    </div>
  );
}

window.ConsoleUploadStage = UploadStage;
