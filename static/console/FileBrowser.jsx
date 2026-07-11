/* FileBrowser.jsx -- the Upload tab's Google-Drive-like workspace.
 *
 * Replaces UploadStage as the body of the "upload" tab. Three regions:
 *   left   -- <FileTree> folder rail (window.ConsoleFileTree)
 *   main   -- a responsive grid of <FileCard> tiles, with a "New" pill that
 *             expands to "File upload" / "New folder", and drag-and-drop that
 *             uploads into the currently-open folder.
 *   detail -- selecting a card swaps the grid for <FileDetail> (enlarged
 *             preview + Metadata/UMR/UIR/Chunks tabs); the back arrow returns
 *             to the grid.
 *
 * Apple design: single Action Blue accent, pill CTAs, rounded.lg cards with a
 * hairline border and no chrome shadow, the one --shadow-product reserved for
 * the preview imagery inside the card. Press = scale(0.95).
 *
 * IIFE-wrapped: see app.jsx.
 */

(function () {

const FileTree = window.ConsoleFileTree;
const FileCard = window.ConsoleFileCard;
const FileDetail = window.ConsoleFileDetail;
const { Button } = window.ApertureDesignSystem_0a9afd;

/** A file's folder: the server-side folder_id once known, else the optimistic
 *  folderId stamped on the row at upload time. null = root. */
function effectiveFolder(f) {
  if (f.job && f.job.folder_id != null) return f.job.folder_id;
  if (f.folderId != null) return f.folderId;
  return null;
}

function NewMenu({ onUpload, onNewFolder }) {
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef(null);

  React.useEffect(() => {
    if (!open) return;
    function onDoc(e) { if (ref.current && !ref.current.contains(e.target)) setOpen(false); }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const item = (icon, label, fn) => (
    <button
      onClick={() => { setOpen(false); fn(); }}
      style={{
        display: "flex", alignItems: "center", gap: 10, width: "100%", textAlign: "left",
        border: "none", background: "transparent", cursor: "pointer", padding: "8px 12px",
        fontSize: "var(--text-caption-size)", color: "var(--text-ink)", borderRadius: "var(--radius-xs)",
      }}
      onMouseEnter={(e) => { e.currentTarget.style.background = "var(--gray-100)"; }}
      onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
    >
      <i data-lucide={icon} style={{ width: 16, height: 16, color: "var(--text-muted-80)" }} />
      {label}
    </button>
  );

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <Button variant="primary" onClick={() => setOpen((v) => !v)}>
        <i data-lucide="plus" style={{ width: 16, height: 16 }} />
        New
      </Button>
      {open && (
        <div style={{
          position: "absolute", top: "100%", right: 0, marginTop: 6, zIndex: 40,
          background: "var(--surface-canvas)", border: "1px solid var(--border-hairline)",
          borderRadius: "var(--radius-sm)", padding: 4, minWidth: 180, boxShadow: "var(--shadow-ring)",
        }}>
          {item("upload", "File upload", onUpload)}
          {item("folder-plus", "New folder", onNewFolder)}
        </div>
      )}
    </div>
  );
}

function FileBrowser(props) {
  const {
    files, folders, currentFolderId, selectedId,
    onSelectFile, onSelectFolder,
    onAddFiles, onDeleteFile, onMoveFile,
    onNewFolder, onRenameFolder, onDeleteFolder,
  } = props;

  const [dragging, setDragging] = React.useState(false);
  const inputRef = React.useRef(null);

  React.useEffect(() => { if (window.lucide) window.lucide.createIcons(); });

  const selected = selectedId ? files.find((f) => f.id === selectedId) : null;
  const currentFolder = folders && folders.find((f) => f.id === currentFolderId);
  const title = currentFolderId == null ? "All files" : (currentFolder ? currentFolder.name : "All files");

  // Counts per folder + root, for the tree badges.
  const counts = { root: 0 };
  (folders || []).forEach((f) => { counts[f.id] = 0; });
  files.forEach((f) => {
    const fid = effectiveFolder(f);
    if (fid == null) counts.root += 1;
    else if (counts[fid] != null) counts[fid] += 1;
  });

  const visible = files
    .filter((f) => effectiveFolder(f) === (currentFolderId == null ? null : currentFolderId))
    .sort((a, b) => (b.job && b.job.submitted_at || 0) - (a.job && a.job.submitted_at || 0));

  function handleInputChange(e) {
    const picked = Array.from(e.target.files || []);
    if (picked.length) onAddFiles(picked, currentFolderId);
    e.target.value = "";
  }
  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    const dropped = Array.from((e.dataTransfer && e.dataTransfer.files) || []);
    if (dropped.length) onAddFiles(dropped, currentFolderId);
  }
  function pickFiles() { if (inputRef.current) inputRef.current.click(); }
  function newFolder() {
    const name = window.prompt("Folder name");
    if (name && name.trim() && onNewFolder) onNewFolder(name.trim());
  }

  return (
    <div style={{ display: "flex", height: "100%", width: "100%" }}>
      <input ref={inputRef} type="file" multiple hidden onChange={handleInputChange} />

      <FileTree
        folders={folders || []}
        currentFolderId={currentFolderId}
        counts={counts}
        onSelectFolder={onSelectFolder}
        onNewFolder={newFolder}
        onRenameFolder={onRenameFolder}
        onDeleteFolder={onDeleteFolder}
      />

      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, overflow: "hidden" }}>
        {selected ? (
          <FileDetail
            file={selected}
            folders={folders || []}
            onBack={() => onSelectFile(null)}
            onDelete={onDeleteFile}
            onMoveFile={onMoveFile}
          />
        ) : (
          <>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, padding: "16px 24px", borderBottom: "1px solid var(--border-hairline)", flexShrink: 0 }}>
              <div>
                <div style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-tagline-size)", fontWeight: 600, color: "var(--text-ink)" }}>{title}</div>
                <div style={{ fontSize: "var(--text-caption-size)", color: "var(--text-muted-48)", marginTop: 2 }}>
                  {visible.length} file{visible.length === 1 ? "" : "s"}
                </div>
              </div>
              <NewMenu onUpload={pickFiles} onNewFolder={newFolder} />
            </div>

            <div
              style={{ flex: 1, overflow: "auto", padding: 24, position: "relative" }}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={(e) => { if (e.currentTarget === e.target) setDragging(false); }}
              onDrop={handleDrop}
            >
              {visible.length > 0 ? (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 20 }}>
                  {visible.map((f) => (
                    <FileCard
                      key={f.id}
                      file={f}
                      selected={f.id === selectedId}
                      onSelect={onSelectFile}
                      onDelete={onDeleteFile}
                    />
                  ))}
                </div>
              ) : (
                <div
                  style={{
                    display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                    gap: 14, height: "100%", minHeight: 260, padding: 40, textAlign: "center",
                    border: `1.5px dashed ${dragging ? "var(--accent-primary)" : "var(--border-hairline)"}`,
                    borderRadius: "var(--radius-lg)",
                    background: dragging ? "var(--blue-50)" : "transparent",
                    transition: "border-color var(--duration-fast) ease, background var(--duration-fast) ease",
                  }}
                >
                  <div style={{ width: 56, height: 56, borderRadius: "var(--radius-full)", background: "var(--blue-50)", color: "var(--accent-primary)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <i data-lucide="upload-cloud" style={{ width: 26, height: 26 }} />
                  </div>
                  <div>
                    <div style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-body-size)", fontWeight: 600, color: "var(--text-ink)" }}>
                      Drop files here to convert them
                    </div>
                    <div style={{ fontSize: "var(--text-caption-size)", color: "var(--text-muted-48)", marginTop: 4 }}>
                      or use “New” — PDF · DOCX · PPTX · XLSX · EPUB · HTML · images · text
                    </div>
                  </div>
                  <Button variant="secondary-pill" onClick={pickFiles}>Choose a file</Button>
                </div>
              )}

              {dragging && visible.length > 0 && (
                <div style={{ position: "absolute", inset: 0, background: "rgba(238,246,255,0.6)", border: "2px dashed var(--accent-primary)", borderRadius: "var(--radius-lg)", display: "flex", alignItems: "center", justifyContent: "center", pointerEvents: "none", color: "var(--accent-primary)", fontFamily: "var(--font-display)", fontSize: 18, fontWeight: 600 }}>
                  Drop to upload into {title}
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

window.ConsoleFileBrowser = FileBrowser;

})();
