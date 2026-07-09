/** Chats — tapped from the profile avatar. Lists past conversations; a thread highlights
    Gemini's turns with a small talking-sparkle badge whenever a line starts with "Gemini:". */
function ChatsPanel() {
  const [conversations, setConversations] = React.useState([
    {
      id: 1,
      name: "Priya Shah",
      preview: "Gemini: I've summarized the board deck — 3 action items.",
      messages: [
        { from: "them", text: "Can you pull the Q3 numbers from the deck?" },
        { from: "me", text: "Sure, uploading it now." },
        { from: "them", text: "Gemini: I've summarized the board deck — revenue up 12%, 3 open action items, and a flagged low-confidence table on page 9." },
      ],
    },
    {
      id: 2,
      name: "Marcus Lee",
      preview: "Sounds good, thanks!",
      messages: [
        { from: "me", text: "Sent you the converted invoice batch." },
        { from: "them", text: "Sounds good, thanks!" },
      ],
    },
    {
      id: 3,
      name: "Data Team",
      preview: "Gemini: 42 chunks embedded, ready to query.",
      messages: [
        { from: "them", text: "Did the attention paper finish processing?" },
        { from: "them", text: "Gemini: 42 chunks embedded, ready to query. Confidence averaged 0.97 across the document." },
      ],
    },
  ]);
  let nextConvoId = 4;

  const [openId, setOpenId] = React.useState(null);
  const [showNewChat, setShowNewChat] = React.useState(false);
  const [newChatEmail, setNewChatEmail] = React.useState("");
  const [search, setSearch] = React.useState("");
  const [draft, setDraft] = React.useState("");
  const open = conversations.find((c) => c.id === openId) || null;

  const filtered = conversations.filter((c) =>
    c.name.toLowerCase().includes(search.toLowerCase())
  );

  React.useEffect(() => {
    if (window.lucide) window.lucide.createIcons();
  }, [openId, showNewChat, conversations]);

  function createChat() {
    const email = newChatEmail.trim();
    if (!email) return;
    const name = email.split("@")[0];
    const id = nextConvoId++;
    const convo = { id, name, preview: "New conversation", messages: [] };
    setConversations((prev) => [convo, ...prev]);
    setNewChatEmail("");
    setShowNewChat(false);
    setOpenId(id);
  }

  function renderMessage(m, i) {
    const isGemini = m.text.startsWith("Gemini:");
    const text = isGemini ? m.text.replace(/^Gemini:\s*/, "") : m.text;
    const mine = m.from === "me";
    return (
      <div key={i} style={{ display: "flex", justifyContent: mine ? "flex-end" : "flex-start", alignItems: "flex-end", gap: 8 }}>
        {isGemini && (
          <div className="ap-gemini-talk" style={{
            width: 22, height: 22, borderRadius: "50%", background: "var(--accent-primary)", color: "#fff",
            display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
          }}>
            <i data-lucide="sparkles" style={{ width: 12, height: 12 }}></i>
          </div>
        )}
        <div
          style={{
            maxWidth: "70%",
            padding: "12px 16px",
            borderRadius: "var(--radius-lg)",
            fontSize: "var(--text-body-size)",
            lineHeight: "var(--text-body-leading)",
            background: mine ? "var(--accent-primary)" : isGemini ? "var(--blue-50)" : "var(--gray-100)",
            color: mine ? "var(--on-accent)" : "var(--text-ink)",
          }}
        >
          {text}
        </div>
      </div>
    );
  }

  if (showNewChat) {
    return (
      <div style={{ maxWidth: 480, margin: "60px auto", width: "100%", display: "flex", flexDirection: "column", gap: 20, padding: "0 8px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", position: "relative", gap: 12 }}>
          <button
            onClick={() => setShowNewChat(false)}
            aria-label="Back"
            style={{
              border: "1px solid var(--border-hairline)", background: "var(--surface-canvas)", cursor: "pointer",
              color: "#000", display: "flex", alignItems: "center", justifyContent: "center",
              width: 36, height: 36, borderRadius: "var(--radius-full)", flexShrink: 0,
              position: "absolute", left: 0,
            }}
          >
            <i data-lucide="chevron-left" style={{ width: 20, height: 20 }}></i>
          </button>
          <div style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-tagline-size)", fontWeight: 600, color: "var(--text-ink)" }}>
            New chat
          </div>
        </div>
        <div style={{ fontSize: "var(--text-caption-size)", color: "var(--text-muted-48)" }}>
          Enter the email of the person you want to chat with.
        </div>
        <input
          type="email"
          value={newChatEmail}
          onChange={(e) => setNewChatEmail(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && createChat()}
          placeholder="name@company.com"
          autoFocus
          style={{
            width: "100%", fontFamily: "var(--font-text)", fontSize: "var(--text-body-size)", color: "#000",
            background: "var(--surface-canvas)", border: "1px solid var(--border-hairline)", borderRadius: "var(--radius-sm)",
            height: 48, padding: "0 16px", outline: "none", boxSizing: "border-box",
          }}
        />
        <button
          onClick={createChat}
          style={{
            width: "100%", background: "#000", color: "#fff", border: "none", borderRadius: "var(--radius-pill)",
            height: 48, fontFamily: "var(--font-text)", fontSize: "var(--text-body-size)", fontWeight: 500, cursor: "pointer",
          }}
        >
          Start chat
        </button>
      </div>
    );
  }

  function sendMsg() {
    if (!draft.trim() || !open) return;
    const text = draft.trim();
    setConversations((prev) =>
      prev.map((c) => (c.id === open.id ? { ...c, messages: [...c.messages, { from: "me", text }], preview: text } : c))
    );
    setDraft("");
  }

  if (open) {
    return (
      <div style={{ display: "flex", flexDirection: "column", height: "100%", maxWidth: 720, margin: "0 auto", width: "100%" }}>
        <div style={{ padding: "28px 8px 16px", display: "flex", alignItems: "center", gap: 12 }}>
          <button
            onClick={() => setOpenId(null)}
            aria-label="Back"
            style={{
              border: "1px solid var(--border-hairline)", background: "var(--surface-canvas)", cursor: "pointer",
              color: "#000", display: "flex", alignItems: "center", justifyContent: "center",
              width: 36, height: 36, borderRadius: "var(--radius-full)", flexShrink: 0,
            }}
          >
            <i data-lucide="chevron-left" style={{ width: 20, height: 20 }}></i>
          </button>
          <div style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-tagline-size)", fontWeight: 600, color: "var(--text-ink)" }}>{open.name}</div>
        </div>
        <div style={{ flex: 1, overflow: "auto", display: "flex", flexDirection: "column", gap: 14, padding: "0 8px 24px" }}>
          {open.messages.map(renderMessage)}
        </div>
        <div style={{ padding: "0 8px 28px", display: "flex", gap: 10, alignItems: "center" }}>
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && sendMsg()}
            placeholder="Message…"
            style={{
              flex: 1, fontFamily: "var(--font-text)", fontSize: "var(--text-body-size)", color: "#000",
              background: "var(--surface-canvas)", border: "1px solid var(--border-hairline)", borderRadius: "var(--radius-pill)",
              height: 48, padding: "0 20px", outline: "none",
            }}
          />
          <button
            onClick={sendMsg}
            aria-label="Send"
            style={{
              width: 48, height: 48, borderRadius: "var(--radius-full)", border: "none",
              background: "#000", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center",
              cursor: "pointer", flexShrink: 0,
            }}
          >
            <i data-lucide="arrow-up" style={{ width: 18, height: 18 }}></i>
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: 720, margin: "0 auto", width: "100%", height: "100%", overflow: "auto" }}>
      <div style={{ padding: "32px 8px 20px", fontFamily: "var(--font-display)", fontSize: "var(--text-display-md-size)", fontWeight: 600, letterSpacing: "var(--text-display-md-tracking)", color: "var(--text-ink)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        Chats
        <button
          onClick={() => { localStorage.removeItem("aperture_logged_in"); window.location.reload(); }}
          style={{
            fontFamily: "var(--font-text)", fontSize: "var(--text-caption-size)", fontWeight: 500,
            color: "var(--text-muted-48)", background: "transparent", border: "1px solid var(--border-hairline)",
            borderRadius: "var(--radius-pill)", padding: "8px 16px", cursor: "pointer",
          }}
        >
          Log out
        </button>
      </div>
      <div style={{ padding: "0 8px 16px", display: "flex", gap: 10, alignItems: "center" }}>
        <button
          onClick={() => setShowNewChat(true)}
          aria-label="New chat"
          style={{
            width: 44, height: 44, borderRadius: "var(--radius-full)", flexShrink: 0,
            border: "1px solid var(--border-hairline)", background: "var(--surface-canvas)",
            display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer", color: "#000",
          }}
        >
          <i data-lucide="plus" style={{ width: 20, height: 20 }}></i>
        </button>
        <div style={{ flex: 1, position: "relative", display: "flex", alignItems: "center" }}>
          <i data-lucide="search" style={{ width: 16, height: 16, position: "absolute", left: 14, color: "var(--text-muted-48)" }}></i>
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search usernames…"
            style={{
              width: "100%", height: 44, fontFamily: "var(--font-text)", fontSize: "var(--text-body-size)", color: "#000",
              background: "var(--surface-canvas)", border: "1px solid var(--border-hairline)", borderRadius: "var(--radius-pill)",
              padding: "0 16px 0 40px", outline: "none", boxSizing: "border-box",
            }}
          />
        </div>
      </div>
      <div style={{ display: "flex", flexDirection: "column" }}>
        {filtered.map((c) => (
          <button
            key={c.id}
            onClick={() => setOpenId(c.id)}
            style={{
              display: "flex", alignItems: "center", gap: 14, padding: "14px 8px", border: "none",
              borderBottom: "1px solid var(--border-hairline)", background: "transparent", cursor: "pointer", textAlign: "left",
            }}
          >
            <div style={{ width: 40, height: 40, borderRadius: "50%", background: "var(--gray-100)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, color: "var(--text-muted-80)" }}>
              <i data-lucide="user" style={{ width: 18, height: 18 }}></i>
            </div>
            <div style={{ overflow: "hidden" }}>
              <div style={{ fontSize: "var(--text-body-strong-size)", fontWeight: "var(--text-body-strong-weight)", color: "var(--text-ink)" }}>{c.name}</div>
              <div style={{ fontSize: "var(--text-caption-size)", color: c.preview.startsWith("Gemini:") ? "var(--accent-primary)" : "var(--text-muted-48)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {c.preview}
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

window.ConsoleChatsPanel = ChatsPanel;
