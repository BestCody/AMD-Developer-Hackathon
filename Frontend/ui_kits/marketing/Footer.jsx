/** Parchment footer — dense link columns at the source's `dense-link` (2.41 leading) grammar. */
function Footer() {
  const cols = [
    { h: "Product", links: ["Pipeline", "Console", "UIR schema", "Integrations"] },
    { h: "Developers", links: ["Docs", "API reference", "GitHub", "Changelog"] },
    { h: "Company", links: ["About", "Careers", "Blog", "Contact"] },
  ];
  return (
    <footer style={{ background: "var(--surface-parchment)", color: "var(--text-muted-80)", padding: "var(--space-xxl) var(--space-xl)", fontFamily: "var(--font-text)" }}>
      <div style={{ display: "flex", gap: 64, maxWidth: "var(--container-max)", margin: "0 auto", flexWrap: "wrap" }}>
        {cols.map((c) => (
          <div key={c.h}>
            <div style={{ fontSize: "var(--text-caption-strong-size)", fontWeight: "var(--text-caption-strong-weight)", color: "var(--text-ink)", marginBottom: 4 }}>{c.h}</div>
            {c.links.map((l) => (
              <div key={l} style={{ fontSize: "var(--text-dense-link-size)", lineHeight: "var(--text-dense-link-leading)" }}>
                <a href="#" style={{ color: "var(--text-muted-80)" }}>{l}</a>
              </div>
            ))}
          </div>
        ))}
      </div>
      <div style={{ maxWidth: "var(--container-max)", margin: "40px auto 0", borderTop: "1px solid var(--border-hairline)", paddingTop: 20, fontSize: "var(--text-fine-print-size)", color: "var(--text-muted-48)" }}>
        Copyright © 2026 Aperture. All rights reserved.
      </div>
    </footer>
  );
}

window.MarketingFooter = Footer;
