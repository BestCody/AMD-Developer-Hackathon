/* app.jsx -- console root: session bootstrap, upload orchestration, routing.
 *
 * State that used to be faked here (the design kit's `nextId++` counter and a
 * `setTimeout(2200 + i*400)` that flipped files to "done") is now derived from
 * the backend: a job id from POST /api/run, and real stage/percent from
 * polling GET /api/status until the job settles.
 *
 * Wrapped in an IIFE. Every `type="text/babel"` script shares one global
 * lexical scope, so a top-level `const IconRail` here collided with
 * IconRail.jsx's `function IconRail` -- a SyntaxError that aborted this
 * whole file, so nothing ever mounted and the page rendered blank. Each
 * console file keeps its locals to itself and communicates only through
 * the `window.Console*` globals it explicitly publishes.
 */

(function () {

// `crypto.randomUUID` only exists in *secure contexts* (HTTPS or localhost).
// The console binds 0.0.0.0, so a page loaded over HTTP from a LAN IP throws
// `crypto.randomUUID is not a function` at the first upload. `getRandomValues`
// is callable in every context (its randomness strength is what's downgraded
// outside secure contexts -- irrelevant for a temp id), so we use it to mint
// a UUIDv4 ourselves.
function randomUUID() {
  if (typeof crypto?.randomUUID === "function") return crypto.randomUUID();
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
  bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant 10xx
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

const IconRail = window.ConsoleIconRail;
const FileBrowser = window.ConsoleFileBrowser;
const FireworksChat = window.ConsoleFireworksChat;
const ChatsPanel = window.ConsoleChatsPanel;
const GlobalSearch = window.ConsoleGlobalSearch;
const LoginScreen = window.ConsoleLoginScreen;
const SignUpScreen = window.ConsoleSignUpScreen;

const API = window.MonadLabsAPI;

function Splash({ children }) {
  return (
    <div style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted-48)", fontFamily: "var(--font-text)" }}>
      {children}
    </div>
  );
}

function App() {
  // `undefined` = still checking the session; null = anonymous.
  const [user, setUser] = React.useState(undefined);
  const [authScreen, setAuthScreen] = React.useState("login");
  const [tab, setTab] = React.useState("upload");
  const [files, setFiles] = React.useState([]);
  const [selectedId, setSelectedId] = React.useState(null);
  // File-browser library state: folders (server-persisted) + which folder is
  // open in the left tree. ``null`` = the "All files" root.
  const [folders, setFolders] = React.useState([]);
  const [currentFolderId, setCurrentFolderId] = React.useState(null);
  // Global document search overlay (opened from the IconRail, any tab).
  const [searchOpen, setSearchOpen] = React.useState(false);

  // Ask the server who we are. The session lives in an HttpOnly cookie, so
  // this is the only way to know -- the page cannot read it.
  React.useEffect(() => {
    API.me()
      .then(({ user }) => setUser(user))
      .catch(() => setUser(null));
  }, []);

  // Rehydrate the library after a reload. Jobs + folders are persisted server
  // side (SQLite), so this returns whatever survived since the server last
  // started. Folders and in-flight jobs are both restored here.
  React.useEffect(() => {
    if (!user) return;
    let cancelled = false;
    API.listJobs()
      .then(({ jobs }) => {
        if (cancelled) return;
        setFiles(jobs.map((j) => ({
          id: j.job_id,
          name: j.filename || j.job_id.slice(0, 8),
          status: j.status === "done" ? "done" : j.status === "error" ? "error" : "processing",
          stage: j.stage,
          percent: j.percent,
          error: j.error,
          job: j,
        })));
        // Resume polling for anything still in flight when we reconnected.
        jobs.filter((j) => j.status === "queued" || j.status === "running")
            .forEach((j) => watchJob(j.job_id));
      })
      .catch(() => { /* folder stays empty; not fatal */ });
    API.listFolders()
      .then(({ folders: fs }) => { if (!cancelled) setFolders(fs || []); })
      .catch(() => { /* folders optional */ });
    return () => { cancelled = true; };
  }, [user]);

  function refreshFolders() {
    API.listFolders()
      .then(({ folders: fs }) => setFolders(fs || []))
      .catch(() => { /* keep current */ });
  }

  function patchFile(id, patch) {
    setFiles((prev) => prev.map((f) => (f.id === id ? { ...f, ...patch } : f)));
  }

  /** Poll a job to completion, mirroring real stage/percent into the folder. */
  async function watchJob(jobId) {
    try {
      const settled = await API.pollUntilSettled(jobId, (tick) => {
        patchFile(jobId, { stage: tick.stage, percent: tick.percent, job: tick });
      });
      if (settled.status === "done") {
        patchFile(jobId, { status: "done", percent: 100, job: settled });
      } else {
        patchFile(jobId, { status: "error", error: settled.error, job: settled });
      }
    } catch (err) {
      if (API.isUnauthorized(err)) { setUser(null); return; }
      patchFile(jobId, { status: "error", error: err.message });
    }
  }

  /** Real files in, real jobs out.
   *
   * `folderId` lands the upload in the currently-open folder (null = root).
   * No intent is sent: the console converts the whole document. `API.run`
   * and POST /api/run still accept the optional `intent` field, and
   * /api/result still filters on it when a job carried one.
   */
  async function addFiles(fileList, folderId) {
    for (const file of fileList) {
      // Optimistic row keyed by a temp id, swapped for the real job id below.
      // Stamp folderId so the pending row shows in the right tree folder.
      // We deliberately do NOT setSelectedId here: a file opens only on an
      // explicit click, not the moment it starts uploading. Auto-opening
      // mid-upload meant N files fought for the detail pane and the user was
      // pulled into a still-converting document instead of seeing the grid.
      const tempId = `pending-${randomUUID()}`;
      setFiles((prev) => [...prev, { id: tempId, name: file.name, status: "processing", stage: "queued", percent: 0, folderId: folderId == null ? null : folderId }]);

      try {
        const { job_id } = await API.run(file, undefined, folderId);
        setFiles((prev) => prev.map((f) => (f.id === tempId ? { ...f, id: job_id } : f)));
        watchJob(job_id);
      } catch (err) {
        if (API.isUnauthorized(err)) { setUser(null); return; }
        // e.g. 400 unsupported file type, 413 too large. Show the server's words.
        patchFile(tempId, { status: "error", error: err.message });
      }
    }
  }

  /** Delete a job for real: remove its upload + outputs on disk, then drop it. */
  async function removeJob(id) {
    const file = files.find((f) => f.id === id);
    const jobId = file && file.job && file.job.job_id;
    setSelectedId((prev) => (prev === id ? null : prev));
    setFiles((prev) => prev.filter((f) => f.id !== id));
    if (jobId) {
      try { await API.deleteJob(jobId); } catch { /* already removed from UI */ }
      refreshFolders();
    }
  }

  /** Move a job into a folder (id) or to the root (null); mirror locally. */
  async function moveJob(id, folderId) {
    const file = files.find((f) => f.id === id);
    const jobId = file && file.job && file.job.job_id;
    patchFile(id, { job: { ...(file && file.job), folder_id: folderId }, folderId });
    if (jobId) {
      try { await API.moveJob(jobId, folderId); } catch { /* optimistic; revert on next list */ }
      refreshFolders();
    }
  }

  async function createFolder(name) {
    try { await API.createFolder(name); refreshFolders(); }
    catch (err) { if (API.isUnauthorized(err)) setUser(null); }
  }
  async function renameFolder(fid, name) {
    try { await API.renameFolder(fid, name); refreshFolders(); }
    catch (err) { if (API.isUnauthorized(err)) setUser(null); }
  }
  async function deleteFolder(fid) {
    try {
      await API.deleteFolder(fid);
      if (currentFolderId === fid) setCurrentFolderId(null);
      // The folder's jobs fall back to the root server-side; reflect that.
      setFiles((prev) => prev.map((f) => {
        const jfid = f.job && f.job.folder_id;
        if (jfid === fid) return { ...f, job: { ...f.job, folder_id: null } };
        return f;
      }));
      refreshFolders();
    } catch (err) { if (API.isUnauthorized(err)) setUser(null); }
  }

  async function logout() {
    try { await API.logout(); } finally {
      setUser(null);
      setFiles([]);
      setSelectedId(null);
      setFolders([]);
      setCurrentFolderId(null);
      setTab("upload");
      setSearchOpen(false);
    }
  }

  /** Open a document from the global search: jump to the Upload tab and
   *  select the file. FileBrowser finds the selected file across all folders,
   *  so FileDetail opens regardless of the currently-open folder. */
  function openDocument(jobId) {
    setTab("upload");
    setSelectedId(jobId);
    setSearchOpen(false);
  }

  if (user === undefined) return <Splash>Loading…</Splash>;

  if (user === null) {
    return authScreen === "signup" ? (
      <SignUpScreen onAuthed={setUser} onBack={() => setAuthScreen("login")} />
    ) : (
      <LoginScreen onAuthed={setUser} onSignUp={() => setAuthScreen("signup")} />
    );
  }

  return (
    <div className="console">
      <IconRail active={tab} onChange={setTab} user={user} onLogout={logout}
        onOpenSearch={() => setSearchOpen(true)} />
      {tab === "upload" && (
        <FileBrowser
          files={files}
          folders={folders}
          currentFolderId={currentFolderId}
          selectedId={selectedId}
          onSelectFile={setSelectedId}
          onSelectFolder={setCurrentFolderId}
          onAddFiles={addFiles}
          onDeleteFile={removeJob}
          onMoveFile={moveJob}
          onNewFolder={createFolder}
          onRenameFolder={renameFolder}
          onDeleteFolder={deleteFolder}
        />
      )}
      {tab === "fireworks" && <FireworksChat files={files} />}
      {tab === "chats" && <ChatsPanel user={user} />}
      <GlobalSearch open={searchOpen} onClose={() => setSearchOpen(false)} onOpenDocument={openDocument} />
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

})();
