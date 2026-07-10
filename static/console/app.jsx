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

const IconRail = window.ConsoleIconRail;
const UploadStage = window.ConsoleUploadStage;
const GeminiChat = window.ConsoleGeminiChat;
const ChatsPanel = window.ConsoleChatsPanel;
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

  // Ask the server who we are. The session lives in an HttpOnly cookie, so
  // this is the only way to know -- the page cannot read it.
  React.useEffect(() => {
    API.me()
      .then(({ user }) => setUser(user))
      .catch(() => setUser(null));
  }, []);

  // Rehydrate the document folder after a reload. Jobs are in-memory server
  // side, so this returns whatever survived since the server last started.
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
    return () => { cancelled = true; };
  }, [user]);

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
   * No intent is sent: the console converts the whole document. `API.run`
   * and POST /api/run still accept the optional `intent` field, and
   * /api/result still filters on it when a job carried one.
   */
  async function addFiles(fileList) {
    for (const file of fileList) {
      // Optimistic row keyed by a temp id, swapped for the real job id below.
      const tempId = `pending-${crypto.randomUUID()}`;
      setFiles((prev) => [...prev, { id: tempId, name: file.name, status: "processing", stage: "queued", percent: 0 }]);
      setSelectedId(tempId);

      try {
        const { job_id } = await API.run(file);
        setFiles((prev) => prev.map((f) => (f.id === tempId ? { ...f, id: job_id } : f)));
        setSelectedId((cur) => (cur === tempId ? job_id : cur));
        watchJob(job_id);
      } catch (err) {
        if (API.isUnauthorized(err)) { setUser(null); return; }
        // e.g. 400 unsupported file type, 413 too large. Show the server's words.
        patchFile(tempId, { status: "error", error: err.message });
      }
    }
  }

  function deleteFile(id) {
    // Client-side only: the server has no delete route, so this hides the row
    // without removing the upload or its UIR output from disk.
    setFiles((prev) => prev.filter((f) => f.id !== id));
    setSelectedId((prev) => (prev === id ? null : prev));
  }

  async function logout() {
    try { await API.logout(); } finally {
      setUser(null);
      setFiles([]);
      setSelectedId(null);
      setTab("upload");
    }
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
      <IconRail active={tab} onChange={setTab} user={user} onLogout={logout} />
      {tab === "upload" && (
        <UploadStage
          files={files}
          selectedId={selectedId}
          onSelectFile={setSelectedId}
          onAddFiles={addFiles}
          onDeleteFile={deleteFile}
        />
      )}
      {tab === "gemini" && <GeminiChat files={files} />}
      {tab === "chats" && <ChatsPanel user={user} />}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

})();
