import React from "react";

/**
 * Bottom-floating notification bar. Uses the same frosted-parchment
 * treatment as the source's floating-sticky-bar / sub-nav-frosted.
 */
export function Toast({ children, action, onDismiss, style }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 16,
        background: "rgba(245,245,247,0.8)",
        backdropFilter: "saturate(180%) blur(20px)",
        WebkitBackdropFilter: "saturate(180%) blur(20px)",
        borderRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-ring)",
        padding: "14px 20px",
        fontFamily: "var(--font-text)",
        fontSize: "var(--text-body-size)",
        color: "var(--text-ink)",
        ...style,
      }}
    >
      <span>{children}</span>
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        {action}
        {onDismiss && (
          <button onClick={onDismiss} style={{ border: "none", background: "transparent", color: "var(--text-muted-48)", cursor: "pointer", fontSize: 18, lineHeight: 1 }}>
            ×
          </button>
        )}
      </div>
    </div>
  );
}
