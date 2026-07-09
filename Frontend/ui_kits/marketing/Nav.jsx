/** Ultra-thin black global nav — 44px, matches DESIGN-apple.md's `global-nav`. */
function Nav() {

  return (
    <nav
      style={{
        position: "sticky",
        top: 0,
        zIndex: 50,
        height: 44,
        background: "var(--surface-black)",
        color: "var(--text-on-dark)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 28,
        fontFamily: "var(--font-text)",
        fontSize: "var(--text-nav-link-size)",
        letterSpacing: "var(--text-nav-link-tracking)",
      }}
    >
      <img src="../../assets/logo/aperture-mark.png" alt="Aperture" style={{ width: 18, height: 18, borderRadius: 5, position: "absolute", left: 24 }} />
      <a href="#product" style={{ color: "inherit" }}>Product</a>
      <a href="#pipeline" style={{ color: "inherit" }}>Pipeline</a>
      <a href="#console" style={{ color: "inherit" }}>Console</a>
      <a href="#pricing" style={{ color: "inherit" }}>Pricing</a>
      <span style={{ position: "absolute", right: 24, display: "flex", gap: 16 }}>
        <a href="#" style={{ color: "inherit" }}>Search</a>
        <a href="#" style={{ color: "inherit" }}>Sign in</a>
      </span>
    </nav>
  );
}

window.MarketingNav = Nav;
