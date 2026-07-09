import React from "react";

const kindStyle = {
  processing: { bg: "var(--status-processing-bg)", fg: "var(--status-processing)" },
  success: { bg: "var(--status-success-bg)", fg: "var(--status-success)" },
  warning: { bg: "var(--status-warning-bg)", fg: "var(--status-warning)" },
  error: { bg: "var(--status-error-bg)", fg: "var(--status-error)" },
  neutral: { bg: "var(--gray-100)", fg: "var(--text-muted-80)" },
};

/** Status pill — Aperture addition for pipeline job states. */
export function Badge({ kind = "neutral", children, style }) {
  const k = kindStyle[kind] || kindStyle.neutral;
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        padding: "4px 12px",
        borderRadius: "var(--radius-pill)",
        fontFamily: "var(--font-text)",
        fontSize: "var(--text-caption-strong-size)",
        fontWeight: "var(--text-caption-strong-weight)",
        background: k.bg,
        color: k.fg,
        ...style,
      }}
    >
      {children}
    </span>
  );
}
