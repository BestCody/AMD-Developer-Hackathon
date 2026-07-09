import React from "react";

/** Store-utility-card grammar: white, hairline border, radius-lg, lg padding. The all-purpose content container. */
export function Card({ image, title, subtitle, action, children, style }) {
  return (
    <div
      style={{
        background: "var(--surface-canvas)",
        border: "1px solid var(--border-hairline)",
        borderRadius: "var(--radius-lg)",
        padding: "var(--space-lg)",
        fontFamily: "var(--font-text)",
        ...style,
      }}
    >
      {image && (
        <div style={{ borderRadius: "var(--radius-sm)", overflow: "hidden", marginBottom: 16, background: "var(--surface-parchment)" }}>
          {image}
        </div>
      )}
      {title && <div style={{ fontSize: "var(--text-body-strong-size)", fontWeight: "var(--text-body-strong-weight)", color: "var(--text-ink)" }}>{title}</div>}
      {subtitle && <div style={{ fontSize: "var(--text-body-size)", color: "var(--text-muted-48)", marginTop: 2 }}>{subtitle}</div>}
      {children && <div style={{ marginTop: 12 }}>{children}</div>}
      {action && <div style={{ marginTop: 12 }}>{action}</div>}
    </div>
  );
}
