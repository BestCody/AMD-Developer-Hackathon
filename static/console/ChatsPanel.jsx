/* ChatsPanel.jsx -- conversations with other people, opened from the profile avatar.
 *
 * This replaces the design kit's mock, which seeded three fictional people
 * (Priya Shah, Marcus Lee, Data Team) into `useState` with a `nextConvoId++`
 * counter and never touched a backend.
 *
 * Real behaviour:
 *   - A conversation is a 1:1 thread between two people, each identified by
 *     email. You start one from the "+" screen by entering an address (they
 *     see the thread when they sign up with it). Both members share the
 *     messages; the thread polls so the other person's replies appear.
 *   - Within a thread, a message that starts with "@fireworks" is a command:
 *     the remainder is answered from *your* converted documents (the same
 *     grounded retrieval the Copilot tab uses) and the reply is posted into
 *     the shared thread, so both people see the question and the answer.
 *
 * "@fireworks" is the product's chosen invocation word; the model behind it is
 * Fireworks, the same one Copilot uses.
 *
 * IIFE-wrapped: see app.jsx.
 */

(function () {

const FIREWORKS_HINT = 'Message, or type "@fireworks <question>" to ask your documents';
const POLL_MS = 4000;

const Markdown = window.ConsoleMarkdown;
const { Badge } = window.ApertureDesignSystem_0a9afd;

/** Compact chips showing the agent's tool calls before a fireworks answer. */
function ToolSteps({ steps }) {
  if (!steps || !steps.length) return null;
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 6, marginLeft: 30 }}>
      {steps.map((s, i) => {
        const isMore = s.tool === "get_more_sources";
        const label = isMore
          ? `Fetched ${s.n_results} more source${s.n_results === 1 ? "" : "s"}`
          : `Searched “${s.query}” — ${s.n_results} source${s.n_results === 1 ? "" : "s"}`;
        return (
          <span key={i} style={{
            display: "inline-flex", alignItems: "center", gap: 5,
            background: "var(--gray-100)", color: "var(--text-muted-80)",
            borderRadius: "var(--radius-pill)", padding: "3px 10px",
            fontSize: "var(--text-micro-legal-size)", fontWeight: 600,
          }}>
            <window.LucideIcon name={isMore ? "plus-circle" : "search"} size={12} />
            {label}
          </span>
        );
      })}
    </div>
  );
}

/** Strip a leading "@fireworks" (any case) -> the question, or null for a message. */
function fireworksQuestion(text) {
  const m = /^\s*@fireworks\b\s*/i.exec(text || "");
  return m ? text.slice(m[0].length).trim() : null;
}

/** A display name from an email: the local part, else the raw string. */
function nameOf(email) {
  if (!email) return "Unknown";
  const local = email.split("@")[0];
  return local.charAt(0).toUpperCase() + local.slice(1);
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

function Message({ m, myEmail }) {
  const isAssistant = m.role === "assistant";
  const mine = !isAssistant && (m.sender_email || "").toLowerCase() === myEmail;
  const isCommand = mine && fireworksQuestion(m.content) !== null;
  return (
    <div style={{ display: "flex", justifyContent: mine ? "flex-end" : "flex-start", alignItems: "flex-end", gap: 8 }}>
      {isAssistant && (
        <div style={{ width: 22, height: 22, borderRadius: "50%", background: "var(--accent-primary)", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginBottom: 2 }}>
          <window.LucideIcon name="sparkles" size={12} />
        </div>
      )}
      <div style={{ maxWidth: "74%" }}>
        {isAssistant && <ToolSteps steps={m.tool_steps} />}
        <div
          style={{
            padding: "12px 16px",
            borderRadius: "var(--radius-lg)",
            fontSize: "var(--text-body-size)",
            lineHeight: "var(--text-body-leading)",
            whiteSpace: isAssistant ? "normal" : "pre-wrap",
            background: mine ? "var(--accent-primary)" : isAssistant ? "var(--blue-50)" : "var(--gray-100)",
            color: mine ? "var(--on-accent)" : "var(--text-ink)",
          }}
        >
          {isAssistant ? <Markdown text={m.content} /> : m.content}
        </div>
        {isCommand && (
          <div style={{ marginTop: 4, fontSize: 11, color: "var(--text-muted-48)", textAlign: "right" }}>asked Fireworks</div>
        )}
        {isAssistant && <Citations items={m.citations} />}
        {isAssistant && m.grounded === false && (
          <div style={{ marginTop: 6, fontSize: 11, color: "var(--text-muted-48)" }}>
            No passage scored high enough to answer from.
          </div>
        )}
      </div>
    </div>
  );
}

function ChatsPanel({ user, files }) {
  const myEmail = ((user && user.email) || "").toLowerCase();
  const API = window.MonadLabsAPI;

  const [conversations, setConversations] = React.useState(null); // null = loading
  const [listError, setListError] = React.useState("");
  const [search, setSearch] = React.useState("");

  // @mention file autocomplete state
  const [mentionDrop, setMentionDrop] = React.useState(false);
  const [mentionQuery, setMentionQuery] = React.useState("");
  const mentionRef = React.useRef(null);
  const convertedFiles = (files || []).filter((f) => f.status === "done" && f.job);

  const [showNew, setShowNew] = React.useState(false);
  const [newEmail, setNewEmail] = React.useState("");
  const [newError, setNewError] = React.useState("");
  const [newSuggestions, setNewSuggestions] = React.useState([]);
  const newSuggestTimer = React.useRef(null);

  const [openId, setOpenId] = React.useState(null);
  const [peer, setPeer] = React.useState("");
  const [peerRegistered, setPeerRegistered] = React.useState(false);
  const [messages, setMessages] = React.useState([]);
  const [threadLoading, setThreadLoading] = React.useState(false);
  const [draft, setDraft] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState("");
  const scrollRef = React.useRef(null);
  const busyRef = React.useRef(false);
  const msgInputRef = React.useRef(null);
  const [cmdDrop, setCmdDrop] = React.useState(false);
  React.useEffect(() => { busyRef.current = busy; }, [busy]);

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

  async function loadMessages(cid, { quiet } = {}) {
    if (!quiet) setThreadLoading(true);
    try {
      const { conversation, messages: msgs } = await API.conversationMessages(cid);
      if (conversation) {
        setPeer(conversation.peer_email || "");
        setPeerRegistered(!!conversation.peer_registered);
      }
      setMessages(msgs);
    } catch (err) {
      if (bounceIfUnauth(err)) return;
      if (!quiet) setError(err.message);
    } finally {
      if (!quiet) setThreadLoading(false);
    }
  }

  function openThread(cid, peerEmail, isReg) {
    setOpenId(cid);
    setPeer(peerEmail || "");
    setPeerRegistered(!!isReg);
    setMessages([]);
    setError("");
    setShowNew(false);
    loadMessages(cid);
  }

  // Poll the open thread so the other person's messages arrive. Skip while a
  // send is in flight so an optimistic bubble is never clobbered mid-request.
  React.useEffect(() => {
    if (openId == null) return;
    const t = setInterval(() => { if (!busyRef.current) loadMessages(openId, { quiet: true }); }, POLL_MS);
    return () => clearInterval(t);
  }, [openId]);

  async function startChat() {
    const email = newEmail.trim();
    if (!email) return;
    setNewError("");
    try {
      const { conversation } = await API.createConversation(email);
      setNewEmail("");
      await refreshList();
      openThread(conversation.id, conversation.peer_email, conversation.peer_registered);
    } catch (err) {
      if (bounceIfUnauth(err)) return;
      setNewError(err.message);
    }
  }

  async function leaveChat(cid) {
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
    const isCommand = fireworksQuestion(text) !== null;
    setMessages((m) => [...m, { id: `pending-${Date.now()}`, role: "user", sender_email: myEmail, content: text }]);
    setBusy(true);
    try {
      const res = await API.sendConversationMessage(openId, text);
      setMessages((m) => {
        const kept = m.filter((x) => typeof x.id !== "string" || !x.id.startsWith("pending-"));
        const next = [...kept, res.user_message];
        if (res.reply) next.push(res.reply);
        return next;
      });
      refreshList();
    } catch (err) {
      if (bounceIfUnauth(err)) return;
      setError(isCommand ? `Fireworks couldn't answer: ${err.message}` : err.message);
    } finally {
      setBusy(false);
    }
  }

  // ---- new-chat screen ----------------------------------------------------
  if (showNew) {
    return (
      <div style={{ maxWidth: 480, margin: "60px auto", width: "100%", display: "flex", flexDirection: "column", gap: 20, padding: "0 8px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", position: "relative", gap: 12 }}>
          <button
            onClick={() => { setShowNew(false); setNewError(""); }}
            aria-label="Back"
            style={{ border: "1px solid var(--border-hairline)", background: "var(--surface-canvas)", cursor: "pointer", color: "var(--text-ink)", display: "flex", alignItems: "center", justifyContent: "center", width: 36, height: 36, borderRadius: "var(--radius-full)", flexShrink: 0, position: "absolute", left: 0 }}
          >
            <window.LucideIcon name="chevron-left" size={20} />
          </button>
          <div style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-tagline-size)", fontWeight: 600, color: "var(--text-ink)" }}>New chat</div>
        </div>
        <div style={{ fontSize: "var(--text-caption-size)", color: "var(--text-muted-48)" }}>
          Enter the email of the person you want to chat with.
        </div>
        <div style={{ position: "relative" }}>
          <input
            type="email"
            value={newEmail}
            onChange={(e) => {
              const v = e.target.value;
              setNewEmail(v);
              setNewSuggestions([]);
              if (newSuggestTimer.current) clearTimeout(newSuggestTimer.current);
              if (v.trim().length < 2) return;
              newSuggestTimer.current = setTimeout(() => {
                API.searchUsers(v.trim())
                  .then(({ users }) => setNewSuggestions(users || []))
                  .catch(() => {});
              }, 200);
            }}
            onBlur={() => setTimeout(() => setNewSuggestions([]), 150)}
            onKeyDown={(e) => e.key === "Enter" && startChat()}
            placeholder="name@company.com"
            autoFocus
            style={{ width: "100%", fontFamily: "var(--font-text)", fontSize: "var(--text-body-size)", color: "var(--text-ink)", background: "var(--surface-canvas)", border: "1px solid var(--border-hairline)", borderRadius: "var(--radius-sm)", height: 48, padding: "0 16px", outline: "none", boxSizing: "border-box" }}
          />
          {newSuggestions.length > 0 && (
            <div style={{
              position: "absolute", top: "calc(100% + 4px)", left: 0, right: 0,
              background: "var(--surface-canvas)", border: "1px solid var(--border-hairline)",
              borderRadius: "var(--radius-sm)", maxHeight: 200, overflow: "auto",
              zIndex: 10, boxShadow: "var(--shadow-ring)",
            }}>
              {newSuggestions.map((u) => (
                <button
                  key={u.id}
                  onClick={() => { setNewEmail(u.email); setNewSuggestions([]); }}
                  style={{
                    display: "flex", alignItems: "center", gap: 10,
                    width: "100%", padding: "10px 14px", border: "none",
                    background: "transparent", cursor: "pointer", textAlign: "left",
                    fontFamily: "var(--font-text)", fontSize: "var(--text-body-size)",
                    color: "var(--text-ink)", borderBottom: "1px solid var(--border-hairline)",
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.background = "var(--surface-parchment)"}
                  onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}
                >
                  <window.LucideIcon name="user" size={14} style={{ color: "var(--text-muted-48)", flexShrink: 0 }} />
                  <div style={{ overflow: "hidden" }}>
                    <div style={{ fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{u.email}</div>
                    <div style={{ fontSize: "var(--text-caption-size)", color: "var(--text-muted-48)" }}>{u.name || "No name"}</div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
        {newError && (
          <div role="alert" style={{ background: "var(--status-error-bg)", color: "var(--status-error)", borderRadius: "var(--radius-sm)", padding: "10px 14px", fontSize: "var(--text-caption-size)" }}>
            {newError}
          </div>
        )}
        <button
          onClick={startChat}
          style={{ width: "100%", background: "var(--surface-black)", color: "#fff", border: "none", borderRadius: "var(--radius-pill)", height: 48, fontFamily: "var(--font-text)", fontSize: "var(--text-body-size)", fontWeight: 500, cursor: "pointer" }}
        >
          Start chat
        </button>
      </div>
    );
  }

  // ---- thread view --------------------------------------------------------
  if (openId != null) {
    return (
      <div style={{ display: "flex", flexDirection: "column", height: "100%", maxWidth: 720, margin: "0 auto", width: "100%" }}>
        <div style={{ padding: "28px 8px 16px", display: "flex", alignItems: "center", gap: 12 }}>
          <button
            onClick={() => { setOpenId(null); setMessages([]); }}
            aria-label="Back"
            style={{ border: "1px solid var(--border-hairline)", background: "var(--surface-canvas)", cursor: "pointer", color: "var(--text-ink)", display: "flex", alignItems: "center", justifyContent: "center", width: 36, height: 36, borderRadius: "var(--radius-full)", flexShrink: 0 }}
          >
            <window.LucideIcon name="chevron-left" size={20} />
          </button>
          <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column", gap: 2 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, overflow: "hidden" }}>
              <span style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-tagline-size)", fontWeight: 600, color: "var(--text-ink)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {nameOf(peer)}
              </span>
              <Badge kind={peerRegistered ? "success" : "neutral"} style={{ padding: "2px 8px", fontSize: "var(--text-micro-legal-size)", flexShrink: 0 }}>
                {peerRegistered ? "Member" : "Pending"}
              </Badge>
            </div>
            <div style={{ fontSize: "var(--text-caption-size)", color: "var(--text-muted-48)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {peer}
            </div>
          </div>
          <button
            onClick={() => leaveChat(openId)}
            aria-label="Leave conversation"
            style={{ border: "none", background: "transparent", cursor: "pointer", color: "var(--text-muted-48)", display: "flex", alignItems: "center", justifyContent: "center", width: 36, height: 36, flexShrink: 0 }}
          >
            <window.LucideIcon name="trash-2" size={18} />
          </button>
        </div>

        <div ref={scrollRef} style={{ flex: 1, overflow: "auto", display: "flex", flexDirection: "column", gap: 14, padding: "0 8px 24px" }}>
          {threadLoading && <div style={{ color: "var(--text-muted-48)", fontSize: "var(--text-caption-size)" }}>Loading…</div>}
          {!threadLoading && messages.length === 0 && (
            <div style={{ color: "var(--text-muted-48)", fontSize: "var(--text-body-size)", marginTop: 8 }}>
              Say hi to {nameOf(peer)}, or start with <b>@fireworks</b> to ask your documents.
            </div>
          )}
          {messages.map((m) => <Message key={m.id} m={m} myEmail={myEmail} />)}

          {busy && (
            <div style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--text-muted-48)", fontSize: "var(--text-caption-size)" }}>
              <div className="ap-spin" style={{ width: 14, height: 14, borderRadius: "50%", border: "2px solid var(--gray-200)", borderTopColor: "var(--accent-primary)" }} />
              {fireworksQuestion(draft) !== null ? "Searching your documents…" : "Sending…"}
            </div>
          )}
          {error && (
            <div role="alert" style={{ background: "var(--status-error-bg)", color: "var(--status-error)", borderRadius: "var(--radius-sm)", padding: "10px 14px", fontSize: "var(--text-caption-size)" }}>
              {error}
            </div>
          )}
        </div>

        <div style={{ padding: "0 8px 28px", display: "flex", gap: 10, alignItems: "center" }}>
          <div style={{ position: "relative", flex: 1 }} ref={mentionRef}>
            <input
              ref={msgInputRef}
              value={draft}
              onChange={(e) => {
                const v = e.target.value;
                setDraft(v);
                const t = v.trimStart();
                const isFireworks = /^\s*@fireworks\b/.test(v);
                setCmdDrop(t.startsWith("@") && (t === "@" || t[1] !== " ") && !isFireworks);
                // File mention autocomplete: @filename anywhere in the text
                const lastAt = v.lastIndexOf("@");
                if (lastAt >= 0 && !/\s/.test(v.slice(lastAt + 1))) {
                  const q = v.slice(lastAt + 1).toLowerCase();
                  setMentionQuery(q);
                  setMentionDrop(q.length > 0);
                } else {
                  setMentionDrop(false);
                  setMentionQuery("");
                }
              }}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
              placeholder={FIREWORKS_HINT}
              disabled={busy}
              style={{ width: "100%", fontFamily: "var(--font-text)", fontSize: "var(--text-body-size)", color: "var(--text-ink)", background: "var(--surface-canvas)", border: "1px solid var(--border-hairline)", borderRadius: "var(--radius-pill)", height: 48, padding: "0 20px", outline: "none", boxSizing: "border-box" }}
            />
            {cmdDrop && (
              <div style={{
                position: "absolute", bottom: "calc(100% + 4px)", left: 0, right: 0,
                background: "var(--surface-canvas)", border: "1px solid var(--border-hairline)",
                borderRadius: "var(--radius-sm)", boxShadow: "var(--shadow-ring)", zIndex: 10,
              }}>
                <button
                  onClick={() => { setDraft("@fireworks "); setCmdDrop(false); setTimeout(() => msgInputRef.current && msgInputRef.current.focus(), 0); }}
                  style={{
                    display: "flex", alignItems: "center", gap: 8, width: "100%", padding: "10px 14px",
                    border: "none", background: "transparent", cursor: "pointer", textAlign: "left",
                    fontFamily: "var(--font-text)", fontSize: "var(--text-body-size)", color: "var(--text-ink)",
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.background = "var(--surface-parchment)"}
                  onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}
                >
                  <window.LucideIcon name="at-sign" size={14} style={{ color: "var(--text-muted-48)" }} />
                  <span style={{ fontWeight: 600 }}>@fireworks</span>
                  <span style={{ color: "var(--text-muted-48)", fontSize: "var(--text-caption-size)" }}>Ask your documents</span>
                </button>
              </div>
            )}
            {mentionDrop && (
              <div style={{
                position: "absolute", bottom: "calc(100% + 4px)", left: 0, right: 0, maxHeight: 200, overflow: "auto",
                background: "var(--surface-canvas)", border: "1px solid var(--border-hairline)",
                borderRadius: "var(--radius-sm)", boxShadow: "var(--shadow-ring)", zIndex: 10,
              }}>
                {convertedFiles.filter((f) => f.name.toLowerCase().includes(mentionQuery.toLowerCase())).slice(0, 8).map((f) => (
                  <button
                    key={f.id}
                    onClick={() => {
                      const lastAt = draft.lastIndexOf("@");
                      const nextDraft = draft.slice(0, lastAt) + "@" + f.name + " " + draft.slice(lastAt + 1 + mentionQuery.length);
                      setDraft(nextDraft);
                      setMentionDrop(false);
                      setMentionQuery("");
                      setTimeout(() => msgInputRef.current && msgInputRef.current.focus(), 0);
                    }}
                    style={{
                      display: "flex", alignItems: "center", gap: 8, width: "100%", padding: "10px 14px",
                      border: "none", background: "transparent", cursor: "pointer", textAlign: "left",
                      fontFamily: "var(--font-text)", fontSize: "var(--text-body-size)", color: "var(--text-ink)",
                      borderBottom: "1px solid var(--border-hairline)",
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.background = "var(--surface-parchment)"}
                    onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}
                  >
                    <window.LucideIcon name="file-text" size={14} style={{ color: "var(--text-muted-48)" }} />
                    <span style={{ fontWeight: 600 }}>@{f.name}</span>
                  </button>
                ))}
                {convertedFiles.filter((f) => f.name.toLowerCase().includes(mentionQuery.toLowerCase())).length === 0 && (
                  <div style={{ padding: "10px 14px", color: "var(--text-muted-48)", fontSize: "var(--text-caption-size)" }}>
                    No matching files.
                  </div>
                )}
              </div>
            )}
          </div>
          <button
            onClick={send} aria-label="Send" disabled={busy || !draft.trim()}
            style={{ width: 48, height: 48, borderRadius: "var(--radius-full)", border: "none", background: busy || !draft.trim() ? "var(--gray-200)" : "var(--accent-primary)", color: "var(--on-accent)", display: "flex", alignItems: "center", justifyContent: "center", cursor: busy || !draft.trim() ? "not-allowed" : "pointer", flexShrink: 0 }}
          >
            <window.LucideIcon name="arrow-up" size={18} />
          </button>
        </div>
      </div>
    );
  }

  // ---- list view ----------------------------------------------------------
  const filtered = (conversations || []).filter((c) =>
    (c.peer_email || "").toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div style={{ maxWidth: 720, margin: "0 auto", width: "100%", height: "100%", overflow: "auto" }}>
      <div style={{ padding: "32px 8px 6px", fontFamily: "var(--font-display)", fontSize: "var(--text-display-md-size)", fontWeight: 600, letterSpacing: "var(--text-display-md-tracking)", color: "var(--text-ink)" }}>
        Chats
      </div>
      <div style={{ padding: "0 8px 18px", fontSize: "var(--text-caption-size)", color: "var(--text-muted-48)" }}>
        Message a teammate by email. Inside a chat, start a line with <b>@fireworks</b> to ask your converted documents.
      </div>

      <div style={{ padding: "0 8px 16px", display: "flex", gap: 10, alignItems: "center" }}>
        <button
          onClick={() => { setShowNew(true); setNewError(""); }}
          aria-label="New chat"
          style={{ width: 44, height: 44, borderRadius: "var(--radius-full)", flexShrink: 0, border: "1px solid var(--border-hairline)", background: "var(--surface-canvas)", display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", color: "var(--text-ink)" }}
        >
          <window.LucideIcon name="plus" size={20} />
        </button>
        <div style={{ flex: 1, position: "relative", display: "flex", alignItems: "center" }}>
          <window.LucideIcon name="search" size={16} style={{ position: "absolute", left: 14, top: "50%", transform: "translateY(-50%)", color: "var(--text-muted-48)", pointerEvents: "none" }} />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search people…"
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
          {conversations.length === 0 ? "No conversations yet. Hit + to message someone." : "No matches."}
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
                onClick={() => openThread(c.id, c.peer_email, c.peer_registered)}
                style={{ display: "flex", alignItems: "center", gap: 14, flex: 1, minWidth: 0, border: "none", background: "transparent", cursor: "pointer", textAlign: "left", padding: 0 }}
              >
                <div style={{ width: 40, height: 40, borderRadius: "50%", background: "var(--gray-100)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, color: "var(--text-muted-80)" }}>
                  <window.LucideIcon name="user" size={18} />
                </div>
                <div style={{ overflow: "hidden" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: "var(--text-body-strong-size)", fontWeight: "var(--text-body-strong-weight)", color: "var(--text-ink)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {nameOf(c.peer_email)}
                    <Badge kind={c.peer_registered ? "success" : "neutral"} style={{ padding: "2px 8px", fontSize: "var(--text-micro-legal-size)" }}>
                      {c.peer_registered ? "Member" : "Pending"}
                    </Badge>
                  </div>
                  <div style={{ fontSize: "var(--text-caption-size)", color: c.last_role === "assistant" ? "var(--accent-primary)" : "var(--text-muted-48)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {c.peer_email} — {c.preview || "No messages yet"}
                  </div>
                </div>
              </button>
              <button
                onClick={() => leaveChat(c.id)}
                aria-label="Leave conversation"
                className="ap-file-delete"
                style={{ border: "none", background: "transparent", cursor: "pointer", color: "var(--text-muted-48)", display: "flex", alignItems: "center", justifyContent: "center", width: 32, height: 32, flexShrink: 0 }}
              >
                <window.LucideIcon name="trash-2" size={16} />
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
