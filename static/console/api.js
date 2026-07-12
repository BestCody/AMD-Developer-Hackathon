/* api.js -- the console's only channel to the Flask backend.
 *
 * Every function here hits a real endpoint. If you find yourself adding a
 * setTimeout to simulate a result, you are writing a mock, not a feature --
 * put it in a test instead.
 *
 * Session auth is a cookie, so every call sends `credentials: "same-origin"`.
 * A 401 means the session lapsed; callers should bounce the user to login
 * rather than silently swallowing it.
 */
(() => {
  "use strict";

  class ApiError extends Error {
    constructor(message, status) {
      super(message);
      this.name = "ApiError";
      this.status = status;
    }
  }

  /** Unauthenticated session. Callers treat this as "show the login screen". */
  const isUnauthorized = (err) => err instanceof ApiError && err.status === 401;

  async function request(path, { method = "GET", body, headers } = {}) {
    let resp;
    try {
      resp = await fetch(path, {
        method,
        credentials: "same-origin",
        headers: headers || (body instanceof FormData ? undefined : { "Content-Type": "application/json" }),
        body: body instanceof FormData ? body : body ? JSON.stringify(body) : undefined,
      });
    } catch (e) {
      throw new ApiError(`Network error: ${e.message}`, 0);
    }

    if (resp.status === 204) return null;

    const isJson = (resp.headers.get("content-type") || "").includes("application/json");
    const payload = isJson ? await resp.json().catch(() => null) : await resp.text();

    if (!resp.ok) {
      const msg = (payload && payload.error) || (typeof payload === "string" && payload) || `HTTP ${resp.status}`;
      throw new ApiError(msg, resp.status);
    }
    return payload;
  }

  // -- auth -----------------------------------------------------------------

  const me = () => request("/api/auth/me");
  const login = (email, password) => request("/api/auth/login", { method: "POST", body: { email, password } });
  const signup = (email, password, name) => request("/api/auth/signup", { method: "POST", body: { email, password, name } });
  const logout = () => request("/api/auth/logout", { method: "POST" });

  // -- pipeline -------------------------------------------------------------

  /** Upload a real File. Returns { job_id }.
   *
   * `folderId` (optional) lands the document in that folder; omit/null puts
   * it at the "All files" root. `intent` is the optional reader-mode query.
   */
  function run(file, intent, folderId) {
    const fd = new FormData();
    fd.append("file", file);
    if (intent && intent.trim()) fd.append("intent", intent.trim());
    if (folderId != null && folderId !== "") fd.append("folder", String(folderId));
    return request("/api/run", { method: "POST", body: fd });
  }

  const status = (jobId) => request(`/api/status/${encodeURIComponent(jobId)}`);
  const listJobs = (folderId) => {
    const qs = folderId === undefined ? "" : `?folder_id=${folderId == null ? "" : encodeURIComponent(folderId)}`;
    return request(`/api/jobs${qs}`);
  };
  const result = (jobId) => request(`/api/result/${encodeURIComponent(jobId)}`);
  const umr = (jobId) => request(`/api/umr/${encodeURIComponent(jobId)}`);
  const downloadUrl = (jobId) => `/api/download/${encodeURIComponent(jobId)}`;
  const thumbUrl = (jobId) => `/api/thumb/${encodeURIComponent(jobId)}`;
  const originalUrl = (jobId) => `/api/original/${encodeURIComponent(jobId)}`;

  // -- folders / library -------------------------------------------------

  const listFolders = () => request("/api/folders");
  const createFolder = (name) => request("/api/folders", { method: "POST", body: { name } });
  const renameFolder = (id, name) => request(`/api/folders/${encodeURIComponent(id)}`, { method: "PATCH", body: { name } });
  const deleteFolder = (id) => request(`/api/folders/${encodeURIComponent(id)}`, { method: "DELETE" });
  /** Move a job into a folder (folderId) or to the root (null). */
  const moveJob = (jobId, folderId) =>
    request(`/api/jobs/${encodeURIComponent(jobId)}`, { method: "PATCH", body: { folder_id: folderId } });
  /** Permanently delete a job and its on-disk artefacts. */
  const deleteJob = (jobId) => request(`/api/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" });

  /**
   * Poll /api/status until the job leaves the running set.
   * `onTick` receives every intermediate payload so the UI can show the real
   * stage name and percentage instead of a decorative spinner.
   * Returns the terminal payload (status "done" or "error").
   */
  async function pollUntilSettled(jobId, onTick, { intervalMs = 700, signal } = {}) {
    for (;;) {
      if (signal && signal.aborted) throw new ApiError("cancelled", 0);
      let data;
      try {
        data = await status(jobId);
      } catch (e) {
        if (isUnauthorized(e)) throw e;
        // Transient network blip: keep polling rather than failing the job.
        await sleep(intervalMs);
        continue;
      }
      if (onTick) onTick(data);
      if (data.status === "done" || data.status === "error") return data;
      await sleep(intervalMs);
    }
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // -- chat -----------------------------------------------------------------

  /** Ask a grounded question. `history` is [{role, content}, ...].
   *  The agent may call tools (search/get_more_sources); the response includes
   *  `tool_steps` describing each call. */
  const chat = (message, history, jobIds) =>
    request("/api/chat", { method: "POST", body: { message, history: history || [], job_ids: jobIds } });

  /** Semantic + title-priority passage search across the caller's documents.
   *  Returns { results: [{job_id, doc_title, page, text, score, title_match}] }. */
  const search = (query, jobIds) =>
    request("/api/search", { method: "POST", body: { query, job_ids: jobIds } });

  // -- conversations (Chats panel) ------------------------------------------

  const listConversations = () => request("/api/conversations");
  /** Start (or reopen) a 1:1 thread with the person at `peerEmail`. */
  const createConversation = (peerEmail) =>
    request("/api/conversations", { method: "POST", body: { peer_email: peerEmail || "" } });
  const deleteConversation = (cid) =>
    request(`/api/conversations/${encodeURIComponent(cid)}`, { method: "DELETE" });
  const conversationMessages = (cid) =>
    request(`/api/conversations/${encodeURIComponent(cid)}/messages`);
  /**
   * Post a message to a thread. If `text` starts with "@fireworks", the server
   * answers the remainder from the user's documents. Returns
   * { user_message, reply } -- reply is null for a plain note.
   */
  const sendConversationMessage = (cid, text) =>
    request(`/api/conversations/${encodeURIComponent(cid)}/messages`, { method: "POST", body: { text } });

  /** Autocomplete: registered users whose email starts with `prefix`. */
  const searchUsers = (prefix) =>
    request(`/api/users/search?q=${encodeURIComponent(prefix)}`);

  window.MonadLabsAPI = {
    ApiError,
    isUnauthorized,
    me, login, signup, logout,
    run, status, listJobs, result, umr, downloadUrl, thumbUrl, originalUrl, pollUntilSettled,
    listFolders, createFolder, renameFolder, deleteFolder,
    moveJob, deleteJob,
    chat, search,
    listConversations, createConversation, deleteConversation,
    conversationMessages, sendConversationMessage, searchUsers,
  };
})();
