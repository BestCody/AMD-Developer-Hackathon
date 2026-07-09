/** Chat with Gemini about your converted documents — Aperture addition. */
function CopilotChat({ files }) {
  const converted = files.filter((f) => f.status === "done");
  const [messages, setMessages] = React.useState([
    { from: "assistant", text: "I can see everything you've converted so far. Let me know if you want me to do something with them." },
  ]);
  const [draft, setDraft] = React.useState("");

  function send() {
    if (!draft.trim()) return;
    const q = draft.trim();
    setMessages((m) => [...m, { from: "user", text: q }]);
    setDraft("");
    setTimeout(() => {
      const reply = converted.length
        ? `Based on ${converted[converted.length - 1].name}, here's what I found — ${converted.length} document${converted.length > 1 ? "s" : ""} are structured and ready to query with full provenance.`
        : "Upload a document first and I'll be able to answer questions about it with citations back to the source chunks.";
      setMessages((m) => [...m, { from: "assistant", text: reply }]);
    }, 700);
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", maxWidth: 720, margin: "0 auto", width: "100%" }}>
      <div style={{ padding: "32px 8px 20px" }}>
        <div style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-display-md-size)", fontWeight: 600, letterSpacing: "var(--text-display-md-tracking)", color: "var(--text-ink)" }}>
          Gemini
        </div>
      </div>

      <div style={{ flex: 1, overflow: "auto", display: "flex", flexDirection: "column", gap: 14, padding: "0 8px" }}>
        {messages.map((m, i) => (
          <div key={i} style={{ display: "flex", justifyContent: m.from === "user" ? "flex-end" : "flex-start" }}>
            <div
              style={{
                maxWidth: "72%",
                padding: "12px 16px",
                borderRadius: "var(--radius-lg)",
                fontSize: "var(--text-body-size)",
                lineHeight: "var(--text-body-leading)",
                background: m.from === "user" ? "var(--accent-primary)" : "var(--gray-100)",
                color: m.from === "user" ? "var(--on-accent)" : "var(--text-ink)",
              }}
            >
              {m.text}
            </div>
          </div>
        ))}
      </div>

      <div style={{ padding: "18px 8px 28px", display: "flex", gap: 10, alignItems: "center" }}>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Ask about your documents…"
          style={{
            flex: 1,
            fontFamily: "var(--font-text)",
            fontSize: "var(--text-body-size)",
            color: "var(--text-ink)",
            background: "var(--surface-canvas)",
            border: "1px solid var(--border-hairline)",
            borderRadius: "var(--radius-pill)",
            height: 48,
            padding: "0 20px",
            outline: "none",
          }}
        />
        <button
          onClick={send}
          aria-label="Send"
          style={{
            width: 48,
            height: 48,
            borderRadius: "var(--radius-full)",
            border: "none",
            background: "var(--accent-primary)",
            color: "var(--on-accent)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
            flexShrink: 0,
          }}
        >
          <i data-lucide="arrow-up" style={{ width: 18, height: 18 }}></i>
        </button>
      </div>
    </div>
  );
}

window.ConsoleCopilotChat = CopilotChat;
