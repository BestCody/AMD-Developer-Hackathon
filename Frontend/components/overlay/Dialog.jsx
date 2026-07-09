import React from "react";

/** Centered modal dialog — canvas surface, radius-lg, no shadow-on-chrome per the source's rules; separation comes from a dim scrim only. */
export function Dialog({ open, title, children, actions, onClose }) {
  if (!open) return null;
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.4)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 100,
      }}
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "var(--surface-canvas)",
          borderRadius: "var(--radius-lg)",
          padding: "var(--space-xl)",
          minWidth: 360,
          maxWidth: 480,
          fontFamily: "var(--font-text)",
        }}
      >
        {title && (
          <h3 style={{ margin: "0 0 12px", fontFamily: "var(--font-display)", fontSize: "var(--text-display-md-size)", fontWeight: 600, letterSpacing: "-0.01em", color: "var(--text-ink)" }}>
            {title}
          </h3>
        )}
        <div style={{ fontSize: "var(--text-body-size)", lineHeight: "var(--text-body-leading)", color: "var(--text-ink)" }}>{children}</div>
        {actions && <div style={{ display: "flex", gap: 12, justifyContent: "flex-end", marginTop: 24 }}>{actions}</div>}
      </div>
    </div>
  );
}
