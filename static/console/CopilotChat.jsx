/* CopilotChat.jsx -- grounded Q&A over the user's converted documents.
 *
 * The design kit's version was a setTimeout(700) that produced a template
 * string ("Based on <filename>, here's what I found — N documents are
 * structured and ready to query with full provenance."). It never called a
 * model and never read a document.
 *
 * This one POSTs to /api/chat, which retrieves the top-scoring chunks from
 * the caller's own UIR documents and sends them to a Fireworks chat model
 * under a grounding instruction. The passages that were actually put in
 * front of the model come back as `citations` and are rendered under the
 * answer, so a claim can be checked against its source.
 *
 * Honesty note: grounding is enforced by prompt. The citations show what the
 * model was *given*, not proof of what it used.
 */

const { Badge } = window.ApertureDesignSystem_0a9afd;

function Citations({ items }) {
  const [open, setOpen] = React.useState(false);
  if (!items || !items.length) return null;
  return (
    <div style={{ marginTop: 10 }}>
      <button
        onClick={() => setOpen((v) => !v)}
        style={{ border: "none", background: "transparent", cursor: "pointer", padding: 0, color: "var(--accent-primary)", fontSize: "var(--text-caption-size)", fontFamily: "var(--font-text)" }}
      >
        {open ? "Hide" : "Show"} {items.length} source{items.length > 1 ? "s" : ""}
      </button>
      {open && (
        <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 8 }}>
          {items.map((c, i) => (
            <div key={`${c.chunk_id}-${i}`} style={{ background: "var(--surface-parchment)", border: "1px solid var(--border-hairline)", borderRadius: "var(--radius-sm)", padding: "10px 12px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                <span className="ap-cite" style={{ fontWeight: 600, fontSize: "var(--text-caption-strong-size)", color: "var(--text-ink)" }}>[{i + 1}]</span>
                <span style={{ fontSize: "var(--text-caption-size)", color: "var(--text-muted-80)" }}>
                  {c.doc_title}{c.page != null ? `, p. ${c.page}` : ""}
                </span>
                <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--text-muted-48)" }}>score {c.score}</span>
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

function CopilotChat({ files }) {
  const converted = files.filter((f) => f.status === "done");
  const [messages, setMessages] = React.useState([]);
  const [draft, setDraft] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState("");
  const scrollRef = React.useRef(null);

  React.useEffect(() => { if (window.lucide) window.lucide.createIcons(); });
  React.useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, busy]);

  async function send() {
    const q = draft.trim();
    if (!q || busy) return;
    setError("");
    setDraft("");
    const next = [...messages, { role: "user", content: q }];
    setMessages(next);
    setBusy(true);

    try {
      // Send prior turns so follow-ups ("what about page 3?") resolve.
      const history = messages.map((m) => ({ role: m.role, content: m.content }));
      const res = await window.MonadLabsAPI.chat(q, history);
      setMessages((m) => [...m, {
        role: "assistant",
        content: res.answer,
        citations: res.citations || [],
        grounded: res.grounded,
      }]);
    } catch (err) {
      if (window.MonadLabsAPI.isUnauthorized(err)) { window.location.reload(); return; }
      // Surface the failure as a banner, not as an assistant message -- a
      // model that failed to answer must not look like it answered.
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const empty = messages.length === 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", maxWidth: 760, margin: "0 auto", width: "100%" }}>
      <div style={{ padding: "32px 8px 12px", display: "flex", alignItems: "center", gap: 12 }}>
        <div style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-display-md-size)", fontWeight: 600, letterSpacing: "var(--text-display-md-tracking)", color: "var(--text-ink)" }}>
          Copilot
        </div>
        <Badge kind={converted.length ? "success" : "neutral"}>
          {converted.length} document{converted.length === 1 ? "" : "s"} indexed
        </Badge>
      </div>

      <div ref={scrollRef} style={{ flex: 1, overflow: "auto", display: "flex", flexDirection: "column", gap: 14, padding: "0 8px" }}>
        {empty && (
          <div style={{ color: "var(--text-muted-48)", fontSize: "var(--text-body-size)", marginTop: 8 }}>
            {converted.length
              ? "Ask a question about your converted documents. Answers cite the chunks they came from."
              : "Convert a document first — I answer only from documents you've uploaded."}
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} style={{ display: "flex", justifyContent: m.role === "user" ? "flex-end" : "flex-start" }}>
            <div style={{ maxWidth: "78%" }}>
              <div
                style={{
                  padding: "12px 16px",
                  borderRadius: "var(--radius-lg)",
                  fontSize: "var(--text-body-size)",
                  lineHeight: "var(--text-body-leading)",
                  whiteSpace: "pre-wrap",
                  background: m.role === "user" ? "var(--accent-primary)" : "var(--gray-100)",
                  color: m.role === "user" ? "var(--on-accent)" : "var(--text-ink)",
                }}
              >
                {m.content}
              </div>
              {m.role === "assistant" && <Citations items={m.citations} />}
              {m.role === "assistant" && m.grounded === false && (
                <div style={{ marginTop: 6, fontSize: 11, color: "var(--text-muted-48)" }}>
                  No passage scored high enough to answer from.
                </div>
              )}
            </div>
          </div>
        ))}

        {busy && (
          <div style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--text-muted-48)", fontSize: "var(--text-caption-size)" }}>
            <div className="ap-spin" style={{ width: 14, height: 14, borderRadius: "50%", border: "2px solid var(--gray-200)", borderTopColor: "var(--accent-primary)" }} />
            Searching your documents…
          </div>
        )}

        {error && (
          <div role="alert" style={{ background: "var(--status-error-bg)", color: "var(--status-error)", borderRadius: "var(--radius-sm)", padding: "10px 14px", fontSize: "var(--text-caption-size)" }}>
            {error}
          </div>
        )}
      </div>

      <div style={{ padding: "18px 8px 28px", display: "flex", gap: 10, alignItems: "center" }}>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
          placeholder="Ask about your documents…"
          disabled={busy}
          style={{
            flex: 1, fontFamily: "var(--font-text)", fontSize: "var(--text-body-size)",
            color: "var(--text-ink)", background: "var(--surface-canvas)",
            border: "1px solid var(--border-hairline)", borderRadius: "var(--radius-pill)",
            height: 48, padding: "0 20px", outline: "none",
          }}
        />
        <button
          onClick={send} aria-label="Send" disabled={busy || !draft.trim()}
          style={{
            width: 48, height: 48, borderRadius: "var(--radius-full)", border: "none",
            background: busy || !draft.trim() ? "var(--gray-200)" : "var(--accent-primary)",
            color: "var(--on-accent)", display: "flex", alignItems: "center",
            justifyContent: "center", cursor: busy || !draft.trim() ? "not-allowed" : "pointer", flexShrink: 0,
          }}
        >
          <i data-lucide="arrow-up" style={{ width: 18, height: 18 }}></i>
        </button>
      </div>
    </div>
  );
}

window.ConsoleCopilotChat = CopilotChat;
