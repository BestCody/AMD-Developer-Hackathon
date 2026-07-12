/* IconRail.jsx -- left rail: Upload / Fireworks, plus the signed-in user.
 *
 * The design kit's rail had a third "Profile → Chats" tab backed by a
 * hardcoded list of fictional conversations ("Priya Shah", "Marcus Lee").
 * There is no messaging backend, so that panel is not shipped. The rail's
 * bottom slot shows who you actually are and lets you sign out.
 *
 * IIFE-wrapped: see app.jsx.
 */

(function () {

function IconRail({ active, onChange, user, onLogout, onOpenSearch }) {
  const items = [
    { id: "upload", label: "Upload", icon: "upload-cloud" },
    { id: "fireworks", label: "Fireworks", icon: "sparkles" },
    { id: "chats", label: "Chats", icon: "message-circle" },
  ];

  const [accountOpen, setAccountOpen] = React.useState(false);
  const accountRef = React.useRef(null);

  const initial = ((user && (user.name || user.email)) || "?").trim().charAt(0).toUpperCase();

  return (
    <div
      style={{
        width: 92,
        flexShrink: 0,
        background: "var(--surface-black)",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 10,
        padding: "20px 0",
      }}
    >
      <img
        src="/static/ds/assets/logo/aperture-mark.png"
        alt="MonadLabs"
        style={{ width: 39, height: 39, borderRadius: 10, marginBottom: 16 }}
      />

      {items.map((it) => {
        const isActive = it.id === active;
        return (
          <button
            key={it.id}
            onClick={() => onChange(it.id)}
            aria-label={it.label}
            aria-current={isActive}
            style={{
              width: 64,
              height: 64,
              borderRadius: "var(--radius-md)",
              border: "none",
              background: isActive ? "var(--accent-primary)" : "transparent",
              color: isActive ? "var(--on-accent)" : "var(--text-muted-on-dark)",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              gap: 4,
              cursor: "pointer",
              transition:
                "background var(--duration-fast) ease, color var(--duration-fast) ease, transform var(--duration-press) var(--ease-standard)",
            }}
            onMouseDown={(e) => (e.currentTarget.style.transform = "scale(var(--scale-press))")}
            onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
            onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
          >
            <window.LucideIcon name={it.icon} size={26} />
            <span style={{ fontSize: 11, fontWeight: 600 }}>{it.label}</span>
          </button>
        );
      })}

      {/* Global document search -- opens a command-palette-style overlay that
          searches every converted document (content semantics + title, title
          first) and jumps to the result's file. Available from any tab. */}
      <button
        onClick={onOpenSearch}
        aria-label="Search documents"
        title="Search documents"
        style={{
          width: 64, height: 64, borderRadius: "var(--radius-md)",
          border: "none", background: "transparent",
          color: "var(--text-muted-on-dark)",
          display: "flex", flexDirection: "column", alignItems: "center",
          justifyContent: "center", gap: 4, cursor: "pointer",
          transition: "background var(--duration-fast) ease, color var(--duration-fast) ease, transform var(--duration-press) var(--ease-standard)",
        }}
        onMouseDown={(e) => (e.currentTarget.style.transform = "scale(var(--scale-press))")}
        onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
        onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
      >
        <window.LucideIcon name="search" size={26} />
        <span style={{ fontSize: 11, fontWeight: 600 }}>Search</span>
      </button>

      <div style={{ flex: 1 }} />

      {/* Account dropdown: avatar button opens a small menu with user info + sign out. */}
      <div style={{ position: "relative" }} ref={accountRef}>
        <button
          onClick={() => setAccountOpen((v) => !v)}
          aria-label="Account"
          aria-haspopup="menu"
          aria-expanded={accountOpen}
          title={user ? user.email : "Account"}
          style={{
            width: 40, height: 40, borderRadius: "50%",
            background: accountOpen ? "var(--accent-primary)" : "var(--gray-700)",
            color: "var(--white)", border: "2px solid transparent",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 14, fontWeight: 600, marginBottom: 4, cursor: "pointer",
            transition: "background var(--duration-fast) ease, transform var(--duration-press) var(--ease-standard)",
          }}
          onMouseDown={(e) => (e.currentTarget.style.transform = "scale(var(--scale-press))")}
          onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
          onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
        >
          {initial}
        </button>
        {accountOpen && (
          <div style={{
            position: "absolute", bottom: "calc(100% + 6px)", left: 0, width: 180,
            background: "var(--surface-canvas)", border: "1px solid var(--border-hairline)",
            borderRadius: "var(--radius-sm)", boxShadow: "var(--shadow-ring)", zIndex: 20,
            padding: "10px 0",
            fontFamily: "var(--font-text)",
          }}>
            <div style={{ padding: "0 14px 8px", fontSize: 12, color: "var(--text-muted-48)", borderBottom: "1px solid var(--border-hairline)" }}>
              <div style={{ fontWeight: 600, color: "var(--text-ink)", fontSize: 13, marginBottom: 2 }}>
                {user && (user.name || user.email) ? (user.name || user.email) : "Account"}
              </div>
              {user && user.email && <div>{user.email}</div>}
            </div>
            <button
              onClick={() => { setAccountOpen(false); onLogout(); }}
              style={{
                width: "100%", padding: "8px 14px", border: "none", background: "transparent",
                cursor: "pointer", color: "var(--status-error)", fontSize: 13,
                fontFamily: "var(--font-text)", textAlign: "left",
                display: "flex", alignItems: "center", gap: 8,
              }}
            >
              <window.LucideIcon name="log-out" size={14} />
              Sign out
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

window.ConsoleIconRail = IconRail;

})();
