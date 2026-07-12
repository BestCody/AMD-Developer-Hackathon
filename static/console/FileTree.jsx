/* FileTree.jsx -- the file-browser's left folder rail.
 *
 * Replaces the old right-hand "Documents" icon grid with a Google-Drive-like
 * folder tree. Apple design: parchment surface, SF Pro Text, the single
 * Action Blue for the selected row, no chrome shadows. "All files" is a
 * pseudo-folder at the top (folder_id === null); user folders sit below with
 * file-count badges. A hover kebab on each user folder opens a tiny menu with
 * Rename / Delete (rename uses a prompt -- pragmatic for a console tool).
 *
 * IIFE-wrapped: see app.jsx.
 */

(function () {

function CountBadge({ count }) {
  if (!count) return null;
  return (
    <span style={{
      fontSize: "var(--text-micro-legal-size)", fontWeight: 600,
      color: "var(--text-muted-48)", background: "var(--gray-100)",
      borderRadius: "var(--radius-pill)", padding: "1px 7px", minWidth: 16, textAlign: "center",
    }}>{count}</span>
  );
}

function FolderRow({ name, icon, count, selected, onClick, trailing }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 8, width: "100%",
      padding: "8px 10px", borderRadius: "var(--radius-sm)", cursor: "pointer",
      background: selected ? "var(--blue-50)" : "transparent",
      color: selected ? "var(--accent-primary)" : "var(--text-ink)",
      transition: "background var(--duration-fast) ease",
    }}
      onClick={onClick}
      onMouseEnter={(e) => { if (!selected) e.currentTarget.style.background = "var(--gray-100)"; }}
      onMouseLeave={(e) => { if (!selected) e.currentTarget.style.background = "transparent"; }}
    >
      <window.LucideIcon name={icon} size={16} style={{ flexShrink: 0, color: selected ? "var(--accent-primary)" : "var(--text-muted-80)" }} />
      <span style={{ fontSize: "var(--text-caption-size)", fontWeight: selected ? 600 : 400, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>
        {name}
      </span>
      {trailing || (count != null ? <CountBadge count={count} /> : null)}
    </div>
  );
}

function KebabMenu({ onRename, onDelete }) {
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef(null);

  React.useEffect(() => {
    if (!open) return;
    function onDoc(e) { if (ref.current && !ref.current.contains(e.target)) setOpen(false); }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const item = (label, fn, danger) => (
    <button
      onClick={(e) => { e.stopPropagation(); setOpen(false); fn(); }}
      style={{
        display: "block", width: "100%", textAlign: "left", border: "none",
        background: "transparent", cursor: "pointer", padding: "6px 10px",
        fontSize: "var(--text-caption-size)", color: danger ? "var(--status-error)" : "var(--text-ink)",
        borderRadius: "var(--radius-xs)",
      }}
      onMouseEnter={(e) => { e.currentTarget.style.background = "var(--gray-100)"; }}
      onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
    >{label}</button>
  );

  return (
    <div ref={ref} style={{ position: "relative", flexShrink: 0 }} onClick={(e) => e.stopPropagation()}>
      <button
        onClick={(e) => { e.stopPropagation(); setOpen((v) => !v); }}
        aria-label="Folder actions"
        style={{ border: "none", background: "transparent", cursor: "pointer", padding: 2, display: "flex", color: "var(--text-muted-48)" }}
      >
        <window.LucideIcon name="ellipsis" size={16} />
      </button>
      {open && (
        <div style={{
          position: "absolute", right: 0, top: "100%", marginTop: 4, zIndex: 30,
          background: "var(--surface-canvas)", border: "1px solid var(--border-hairline)",
          borderRadius: "var(--radius-sm)", padding: 4, minWidth: 120, boxShadow: "var(--shadow-ring)",
        }}>
          {item("Rename", onRename, false)}
          {item("Delete", onDelete, true)}
        </div>
      )}
    </div>
  );
}

function FileTree({ folders, currentFolderId, counts, onSelectFolder, onNewFolder, onRenameFolder, onDeleteFolder }) {
  // counts: { [folderId]: number, "root": number }
  const rootCount = (counts && counts.root) || 0;

  return (
    <aside style={{
      width: 232, flexShrink: 0, background: "var(--surface-parchment)",
      padding: "20px 12px", overflow: "auto", borderRight: "1px solid var(--border-hairline)",
      fontFamily: "var(--font-text)",
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14, padding: "0 10px" }}>
        <span style={{ fontSize: "var(--text-caption-strong-size)", fontWeight: 600, color: "var(--text-muted-48)" }}>Library</span>
        <button
          onClick={onNewFolder}
          aria-label="New folder"
          title="New folder"
          style={{ border: "none", background: "transparent", cursor: "pointer", padding: 2, color: "var(--accent-primary)", display: "flex" }}
        >
          <window.LucideIcon name="folder-plus" size={16} />
        </button>
      </div>

      <FolderRow
        name="All files" icon="folder" count={rootCount}
        selected={currentFolderId == null}
        onClick={() => onSelectFolder(null)}
      />

      <div style={{ height: 1, background: "var(--border-hairline)", margin: "10px 8px" }} />

      {folders.length === 0 && (
        <div style={{ fontSize: "var(--text-micro-legal-size)", color: "var(--text-muted-48)", padding: "4px 10px" }}>
          No folders yet
        </div>
      )}

      {folders.map((f) => (
        <FolderRow
          key={f.id}
          name={f.name} icon="folder" count={f.file_count != null ? f.file_count : (counts && counts[f.id]) || 0}
          selected={currentFolderId === f.id}
          onClick={() => onSelectFolder(f.id)}
          trailing={
            (onRenameFolder || onDeleteFolder) ? (
              <KebabMenu
                onRename={() => {
                  const name = window.prompt("Rename folder", f.name);
                  if (name && name.trim() && onRenameFolder) onRenameFolder(f.id, name.trim());
                }}
                onDelete={() => {
                  if (window.confirm(`Delete folder “${f.name}”? Its files move to All files.`) && onDeleteFolder) onDeleteFolder(f.id);
                }}
              />
            ) : null
          }
        />
      ))}
    </aside>
  );
}

window.ConsoleFileTree = FileTree;

})();
