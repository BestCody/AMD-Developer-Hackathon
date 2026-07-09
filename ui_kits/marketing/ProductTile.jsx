const bg = {
  light: "var(--surface-canvas)",
  parchment: "var(--surface-parchment)",
  dark: "var(--surface-tile-1)",
  dark2: "var(--surface-tile-2)",
};
const fg = { light: "var(--text-ink)", parchment: "var(--text-ink)", dark: "var(--text-on-dark)", dark2: "var(--text-on-dark)" };
const muted = { light: "var(--text-muted-48)", parchment: "var(--text-muted-48)", dark: "var(--text-muted-on-dark)", dark2: "var(--text-muted-on-dark)" };

/** Full-bleed product tile — the section-rhythm building block of the whole site. */
function ProductTile({ tone = "light", eyebrow, title, tagline, ctas, art, align = "center" }) {
  return (
    <section
      style={{
        background: bg[tone],
        color: fg[tone],
        padding: "var(--space-section) var(--space-xl)",
        textAlign: align,
        display: "flex",
        flexDirection: "column",
        alignItems: align === "center" ? "center" : "flex-start",
        gap: 16,
      }}
    >
      {eyebrow && (
        <div style={{ fontFamily: "var(--font-display)", fontSize: "var(--text-tagline-size)", fontWeight: "var(--text-tagline-weight)", letterSpacing: "var(--text-tagline-tracking)", color: tone.startsWith("dark") ? "var(--accent-primary-on-dark)" : "var(--accent-primary)" }}>
          {eyebrow}
        </div>
      )}
      <h2
        style={{
          margin: 0,
          fontFamily: "var(--font-display)",
          fontSize: "var(--text-display-lg-size)",
          fontWeight: "var(--text-display-lg-weight)",
          lineHeight: "var(--text-display-lg-leading)",
          letterSpacing: "var(--text-display-lg-tracking)",
          maxWidth: 640,
        }}
      >
        {title}
      </h2>
      {tagline && (
        <p style={{ margin: 0, fontFamily: "var(--font-display)", fontSize: "var(--text-lead-size)", fontWeight: "var(--text-lead-weight)", lineHeight: "var(--text-lead-leading)", color: muted[tone], maxWidth: 560 }}>
          {tagline}
        </p>
      )}
      {ctas && <div style={{ display: "flex", gap: 12, marginTop: 8 }}>{ctas}</div>}
      {art && <div style={{ marginTop: 32 }}>{art}</div>}
    </section>
  );
}

window.MarketingProductTile = ProductTile;
