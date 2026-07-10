/* IconRail.jsx -- left rail: Upload / Copilot, plus the signed-in user.
 *
 * The design kit's rail had a third "Profile → Chats" tab backed by a
 * hardcoded list of fictional conversations ("Priya Shah", "Marcus Lee").
 * There is no messaging backend, so that panel is not shipped. The rail's
 * bottom slot shows who you actually are and lets you sign out.
 */

function IconRail({ active, onChange, user, onLogout }) {
  const items = [
    { id: "upload", label: "Upload", icon: "upload-cloud" },
    { id: "copilot", label: "Copilot", icon: "sparkles" },
  ];

  React.useEffect(() => {
    if (window.lucide) window.lucide.createIcons();
  });

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
            <i data-lucide={it.icon} style={{ width: 26, height: 26 }}></i>
            <span style={{ fontSize: 11, fontWeight: 600 }}>{it.label}</span>
          </button>
        );
      })}

      <div style={{ flex: 1 }} />

      <div
        title={user ? user.email : ""}
        style={{
          width: 36, height: 36, borderRadius: "50%",
          background: "var(--gray-700)", color: "var(--white)",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 14, fontWeight: 600, marginBottom: 4,
        }}
      >
        {initial}
      </div>
      <button
        onClick={onLogout}
        style={{
          border: "none", background: "transparent", cursor: "pointer",
          color: "var(--text-muted-on-dark)", fontSize: 11,
          fontFamily: "var(--font-text)",
        }}
      >
        Sign out
      </button>
    </div>
  );
}

window.ConsoleIconRail = IconRail;
