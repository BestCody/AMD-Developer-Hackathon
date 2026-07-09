import React from "react";

/**
 * Configurator-style option chip — tappable pill with a selected state
 * (2px accent-focus border), matching the source's iPhone buy-page grid.
 */
export function Tag({ selected = false, children, onClick, style }) {
  return (
    <button
      onClick={onClick}
      style={{
        fontFamily: "var(--font-text)",
        fontSize: "var(--text-caption-size)",
        color: "var(--text-ink)",
        background: "var(--surface-canvas)",
        border: selected ? "2px solid var(--accent-primary-focus)" : "1px solid var(--border-hairline)",
        borderRadius: "var(--radius-pill)",
        padding: selected ? "11px 15px" : "12px 16px",
        cursor: "pointer",
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
        ...style,
      }}
    >
      {children}
    </button>
  );
}
