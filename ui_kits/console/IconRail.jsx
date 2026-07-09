/** Left icon rail — profile / Upload / Gemini, unified list for consistent spacing. Aperture addition. */
function IconRail({ active, onChange }) {
  const items = [
    { id: "chats", label: "Profile", img: "../../assets/icons/profile-alpha.png" },
    { id: "upload", label: "Upload", icon: "upload-cloud" },
    { id: "gemini", label: "Gemini", icon: "sparkles" },
  ];
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
      <img src="../../assets/logo/aperture-mark.png" alt="Aperture" style={{ width: 39, height: 39, borderRadius: 10, marginBottom: 16 }} />

      {items.map((it) => {
        const isActive = it.id === active;
        return (
          <button
            key={it.id}
            onClick={() => onChange(it.id)}
            aria-label={it.label}
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
              transition: "background var(--duration-fast) ease, color var(--duration-fast) ease, transform var(--duration-press) var(--ease-standard)",
            }}
            onMouseDown={(e) => (e.currentTarget.style.transform = "scale(var(--scale-press))")}
            onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
            onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
          >
            {it.img ? (
              <img src={it.img} alt="" style={{ width: 26, height: 26 }} />
            ) : (
              <i data-lucide={it.icon} style={{ width: 26, height: 26 }}></i>
            )}
            <span style={{ fontSize: 11, fontWeight: 500 }}>{it.label}</span>
          </button>
        );
      })}
    </div>
  );
}

window.ConsoleIconRail = IconRail;
