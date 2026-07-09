import React from "react";

export function Checkbox({ checked, onChange, label, style, ...rest }) {
  return (
    <label style={{ display: "inline-flex", alignItems: "center", gap: 10, cursor: "pointer", fontFamily: "var(--font-text)", fontSize: "var(--text-body-size)", color: "var(--text-ink)", ...style }}>
      <span
        onClick={() => onChange && onChange(!checked)}
        style={{
          width: 20,
          height: 20,
          borderRadius: 6,
          border: checked ? "none" : "1.5px solid var(--border-hairline)",
          background: checked ? "var(--accent-primary)" : "var(--surface-canvas)",
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          transition: "background var(--duration-fast) ease",
          flexShrink: 0,
        }}
      >
        {checked && (
          <svg width="12" height="10" viewBox="0 0 12 10" fill="none">
            <path d="M1 5L4.3 8.3L11 1.5" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        )}
      </span>
      {label}
    </label>
  );
}
