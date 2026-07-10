/* ChatsPanel.jsx -- persistent chat threads, opened from the profile avatar.
 *
 * This replaces the design kit's mock. That version seeded three fictional
 * people (Priya Shah, Marcus Lee, Data Team) into `useState` with a
 * `nextConvoId++` counter and never touched a backend -- opening a thread
 * showed hardcoded lines. Shipping fake conversations in a working product
 * reads as broken, so the mock was dropped in the port.
 *
 * This is the real thing:
 *   - Threads and messages persist per user in SQLite (see
 *     uir_pipeline/conversations.py); they survive reloads and restarts.
 *   - A message that starts with "gemini:" is a command: the text after the
 *     colon is answered from the caller's own converted documents (the same
 *     grounded retrieval /api/chat uses), and the reply is stored with the
 *     citations it was given. Anything else is a plain note in the thread.
 *
 * The "gemini:" keyword is the product's chosen invocation word; the model
 * behind it is Fireworks, the same one the Copilot tab uses.
 *
 * IIFE-wrapped: see app.jsx.
 */

(function () {

const GEMINI_HINT = 'Message, or type "gemini: <question>" to ask your documents';

/** Strip a leading "gemini:" (any case) -> the question, or null for a note. */
function geminiQuestion(text) {
  const m = /^\s*gemini:\s*/i.exec(text || "");
  return m ? (text.slice(m[0].length)).trim() : null;
}

function Citations({ items }) {
  const [open, setOpen] = React.useState(false);
  if (!items || !items.length) return null;
  return (
    <div style={{ marginTop: 8 }}>
      <button
        onClick={() => setOpen((v) => !v)}
        style={{ border: "none", background: "transparent", cursor: "pointer", padding: 0, color: "var(--accent-primary)", fontSize: "var(--text-caption-size)", fontFamily: "var(--font-text)" }}
      >
        {open ? "Hide" : "Show"} {items.length} source{items.length > 1 ? "s" : ""}
      </button>
      {open && (
        <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 8 }}>
          {items.map((c, i) => (
            <div key={`${c.chunk_id || i}-${i}`} style={{ background: "var(--surface-parchment)", border: "1px solid var(--border-hairline)", borderRadius: "var(--radius-sm)", padding: "10px 12px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                <span className="ap-cite" style={{ fontWeight: 600, fontSize: "var(--text-caption-strong-size)", color: "var(--text-ink)" }}>[{i + 1}]</span>
                <span style={{ fontSize: "var(--text-caption-size)", color: "var(--text-muted-80)" }}>
                  {c.doc_title}{c.page != null ? `, p. ${c.page}` : ""}
                </span>
                {c.score != null && <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--text-muted-48)" }}>score {c.score}</span>}
              </div>
              <div style={{ fontSize: "var(--text-caption-size)", color: "var(--text-muted-80)", lineHeight: 1.5, maxHeight: 88, overflow: "auto" }}>
                {c.text}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function GeminiAvatar() {
  return (
    <div style={{
      width: 22, height: 22, borderRadius: "50%", background: "var(--accent-primary)", color: "#fff",
      display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginBottom: 2,
    }}>
      <i data-lucide="sparkles" style={{ width: 12, height: 12 }}></i>
    </div>
  );
}

function Message({ m }) {
  const mine = m.role === "user";
  const isCommand = mine && geminiQuestion(m.content) !== null;
  return (
    <div style={{ display: "flex", justifyContent: mine ? "flex-end" : "flex-start", alignItems: "flex-end", gap: 8 }}>
      {!mine && <GeminiAvatar />}
      <div style={{ maxWidth: "74%" }}>
        <div
          style={{
            padding: "12px 16px",
            borderRadius: "var(--radius-lg)",
            fontSize: "var(--text-body-size)",
            lineHeight: "var(--text-body-leading)",
            whiteSpace: "pre-wrap",
            background: mine ? "var(--accent-primary)" : "var(--blue-50)",
            color: mine ? "var(--on-accent)" : "var(--text-ink)",
          }}
        >
          {m.content}
        </div>
        {isCommand && (
          <div style={{ marginTop: 4, fontSize: 11, color: "var(--text-muted-48)", textAlign: "right" }}>
            asked Gemini
          </div>
        )}
        {!mine && <Citations items={m.citations} />}
        {!mine && m.grounded === false && (
          <div style={{ marginTop: 6, fontSize: 11, color: "var(--text-muted-48)" }}>
            No passage scored high enough to answer from.
          </div>
        )}
      </div>
    </div>
  );
}

function ChatsPanel() {
  const [conversations, setConversations] = React.useState(null); // null = loading
  const [listError, setListError] = React.useState("");
  const [search, setSearch] = React.useState("");

  const [openId, setOpenId] = React.useState(null);
  const [messages, setMessages] = React.useState([]);
  const [threadLoading, setThreadLoading] = React.useState(false);
  const [draft, setDraft] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState("");
  const scrollRef = React.useRef(null);

  const API = window.MonadLabsAPI;

  React.useEffect(() => { if (window.lucide) window.lucide.createIcons(); });
  React.useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, busy]);

  function bounceIfUnauth(err) {
    if (API.isUnauthorized(err)) { window.location.reload(); return true; }
    return false;
  }

  async function refreshList() {
    try {
      const { conversations: list } = await API.listConversations();
      setConversations(list);
    } catch (err) {
      if (bounceIfUnauth(err)) return;
      setConversations([]);
      setListError(err.message);
    }
  }

  React.useEffect(() => { refreshList(); }, []);

  const open = conversations && conversations.find((c) => c.id === openId);

  async function openThread(cid) {
    setOpenId(cid);
    setMessages([]);
    setError("");
    setThreadLoading(true);
    try {
      const { messages: msgs } = await API.conversationMessages(cid);
      setMessages(msgs);
    } catch (err) {
      if (bounceIfUnauth(err)) return;
      setError(err.message);
    } finally {
      setThreadLoading(false);
    }
  }

  async function newChat() {
    try {
      const { conversation } = await API.createConversation();
      setConversations((prev) => [conversation, ...(prev || [])]);
      openThread(conversation.id);
    } catch (err) {
      if (bounceIfUnauth(err)) return;
      setListError(err.message);
    }
  }

  async function removeChat(cid) {
    try {
      await API.deleteConversation(cid);
      setConversations((prev) => (prev || []).filter((c) => c.id !== cid));
      if (openId === cid) { setOpenId(null); setMessages([]); }
    } catch (err) {
      if (bounceIfUnauth(err)) return;
      setListError(err.message);
    }
  }

  async function send() {
    const text = draft.trim();
    if (!text || busy || openId == null) return;
    setError("");
    setDraft("");
    // Optimistic user bubble; the server persists it regardless of what the
    // model does, so this never diverges from what a reload would show.
    setMessages((m) => [...m, { id: `pending-${Date.now()}`, role: "user", content: text }]);
    setBusy(true);
    const isCommand = geminiQuestion(text) !== null;
    try {
      const res = await API.sendConversationMessage(openId, text);
      // Swap the optimistic bubble for the persisted one, then append a reply.
      setMessages((m) => {
        const kept = m.filter((x) => typeof x.id !== "string" || !x.id.startsWith("pending-"));
        const next = [...kept, res.user_message];
        if (res.reply) next.push(res.reply);
        return next;
      });
      refreshList(); // preview + auto-title may have changed
    } catch (err) {
      if (bounceIfUnauth(err)) return;
      // The user message is saved server-side even on model failure; keep the
      // optimistic bubble and show the error rather than faking an answer.
      setError(isCommand ? `Gemini couldn't answer: ${err.message}` : err.message);
    } finally {
      setBusy(false);
    }
  }

  // ---- thread view --------------------------------------------------------
  if (openId != null && open) {
    return (
      <div style={{ display: "flex", flexDirection: "column", height: "100%", maxWidth: 720, margin: "0 auto", width: "100%" }}>
        <div style={{ padding: "28px 8px 16px", display: "flex", alignItems: "center", gap: 12 }}>
          <button
            onClick={() => { setOpenId(null); setMessages([]); }}
            aria-label="Back"
            style={{ border: "1px solid var(--border-hairline)", background: "var(--surface-canvas)", cursor: "pointer", color: "var(--text-ink)", display: "flex", alignItems: "center", justifyContent: "center", width: 36, height: 36, borderRadius: "var(--radius-full)", flexShrink: 0 }}
          >
            <i data-lucide="chevron-left" style={{ width: 20, height: 20 }}></i>
          </button>
          <div style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-tagline-size)", fontWeight: 600, color: "var(--text-ink)", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {open.title}
          </div>
          <button
            onClick={() => removeChat(open.id)}
            aria-label="Delete conversation"
            style={{ border: "none", background: "transparent", cursor: "pointer", color: "var(--text-muted-48)", display: "flex", alignItems: "center", justifyContent: "center", width: 36, height: 36, flexShrink: 0 }}
          >
            <i data-lucide="trash-2" style={{ width: 18, height: 18 }}></i>
          </button>
        </div>

        <div ref={scrollRef} style={{ flex: 1, overflow: "auto", display: "flex", flexDirection: "column", gap: 14, padding: "0 8px 24px" }}>
          {threadLoading && (
            <div style={{ color: "var(--text-muted-48)", fontSize: "var(--text-caption-size)" }}>Loading…</div>
          )}
          {!threadLoading && messages.length === 0 && (
            <div style={{ color: "var(--text-muted-48)", fontSize: "var(--text-body-size)", marginTop: 8 }}>
              Nothing here yet. Type a note, or start with <b>gemini:</b> to ask your documents.
            </div>
          )}
          {messages.map((m) => <Message key={m.id} m={m} />)}

          {busy && (
            <div style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--text-muted-48)", fontSize: "var(--text-caption-size)" }}>
              <div className="ap-spin" style={{ width: 14, height: 14, borderRadius: "50%", border: "2px solid var(--gray-200)", borderTopColor: "var(--accent-primary)" }} />
              {geminiQuestion(draft) !== null ? "Searching your documents…" : "Saving…"}
            </div>
          )}
          {error && (
            <div role="alert" style={{ background: "var(--status-error-bg)", color: "var(--status-error)", borderRadius: "var(--radius-sm)", padding: "10px 14px", fontSize: "var(--text-caption-size)" }}>
              {error}
            </div>
          )}
        </div>

        <div style={{ padding: "0 8px 28px", display: "flex", gap: 10, alignItems: "center" }}>
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
            placeholder={GEMINI_HINT}
            disabled={busy}
            style={{ flex: 1, fontFamily: "var(--font-text)", fontSize: "var(--text-body-size)", color: "var(--text-ink)", background: "var(--surface-canvas)", border: "1px solid var(--border-hairline)", borderRadius: "var(--radius-pill)", height: 48, padding: "0 20px", outline: "none" }}
          />
          <button
            onClick={send} aria-label="Send" disabled={busy || !draft.trim()}
            style={{ width: 48, height: 48, borderRadius: "var(--radius-full)", border: "none", background: busy || !draft.trim() ? "var(--gray-200)" : "var(--accent-primary)", color: "var(--on-accent)", display: "flex", alignItems: "center", justifyContent: "center", cursor: busy || !draft.trim() ? "not-allowed" : "pointer", flexShrink: 0 }}
          >
            <i data-lucide="arrow-up" style={{ width: 18, height: 18 }}></i>
          </button>
        </div>
      </div>
    );
  }

  // ---- list view ----------------------------------------------------------
  const filtered = (conversations || []).filter((c) =>
    (c.title || "").toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div style={{ maxWidth: 720, margin: "0 auto", width: "100%", height: "100%", overflow: "auto" }}>
      <div style={{ padding: "32px 8px 6px", fontFamily: "var(--font-display)", fontSize: "var(--text-display-md-size)", fontWeight: 600, letterSpacing: "var(--text-display-md-tracking)", color: "var(--text-ink)" }}>
        Chats
      </div>
      <div style={{ padding: "0 8px 18px", fontSize: "var(--text-caption-size)", color: "var(--text-muted-48)" }}>
        Notes to yourself. Start a line with <b>gemini:</b> to ask your converted documents.
      </div>

      <div style={{ padding: "0 8px 16px", display: "flex", gap: 10, alignItems: "center" }}>
        <button
          onClick={newChat}
          aria-label="New chat"
          style={{ width: 44, height: 44, borderRadius: "var(--radius-full)", flexShrink: 0, border: "1px solid var(--border-hairline)", background: "var(--surface-canvas)", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", color: "var(--text-ink)" }}
        >
          <i data-lucide="plus" style={{ width: 20, height: 20 }}></i>
        </button>
        <div style={{ flex: 1, position: "relative", display: "flex", alignItems: "center" }}>
          <i data-lucide="search" style={{ width: 16, height: 16, position: "absolute", left: 14, color: "var(--text-muted-48)" }}></i>
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search conversations…"
            style={{ width: "100%", height: 44, fontFamily: "var(--font-text)", fontSize: "var(--text-body-size)", color: "var(--text-ink)", background: "var(--surface-canvas)", border: "1px solid var(--border-hairline)", borderRadius: "var(--radius-pill)", padding: "0 16px 0 40px", outline: "none", boxSizing: "border-box" }}
          />
        </div>
      </div>

      {listError && (
        <div role="alert" style={{ margin: "0 8px 12px", background: "var(--status-error-bg)", color: "var(--status-error)", borderRadius: "var(--radius-sm)", padding: "10px 14px", fontSize: "var(--text-caption-size)" }}>
          {listError}
        </div>
      )}

      {conversations === null ? (
        <div style={{ padding: "8px", color: "var(--text-muted-48)", fontSize: "var(--text-caption-size)" }}>Loading…</div>
      ) : filtered.length === 0 ? (
        <div style={{ padding: "8px", color: "var(--text-muted-48)", fontSize: "var(--text-body-size)" }}>
          {conversations.length === 0 ? "No conversations yet. Hit + to start one." : "No matches."}
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column" }}>
          {filtered.map((c) => (
            <div
              key={c.id}
              className="ap-file-icon"
              style={{ display: "flex", alignItems: "center", gap: 14, padding: "14px 8px", borderBottom: "1px solid var(--border-hairline)" }}
            >
              <button
                onClick={() => openThread(c.id)}
                style={{ display: "flex", alignItems: "center", gap: 14, flex: 1, minWidth: 0, border: "none", background: "transparent", cursor: "pointer", textAlign: "left", padding: 0 }}
              >
                <div style={{ width: 40, height: 40, borderRadius: "50%", background: "var(--gray-100)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, color: "var(--text-muted-80)" }}>
                  <i data-lucide="message-square" style={{ width: 18, height: 18 }}></i>
                </div>
                <div style={{ overflow: "hidden" }}>
                  <div style={{ fontSize: "var(--text-body-strong-size)", fontWeight: "var(--text-body-strong-weight)", color: "var(--text-ink)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c.title}</div>
                  <div style={{ fontSize: "var(--text-caption-size)", color: c.last_role === "assistant" ? "var(--accent-primary)" : "var(--text-muted-48)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {c.preview || "Empty conversation"}
                  </div>
                </div>
              </button>
              <button
                onClick={() => removeChat(c.id)}
                aria-label="Delete conversation"
                className="ap-file-delete"
                style={{ border: "none", background: "transparent", cursor: "pointer", color: "var(--text-muted-48)", display: "flex", alignItems: "center", justifyContent: "center", width: 32, height: 32, flexShrink: 0 }}
              >
                <i data-lucide="trash-2" style={{ width: 16, height: 16 }}></i>
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

window.ConsoleChatsPanel = ChatsPanel;

})();
