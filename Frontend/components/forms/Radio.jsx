import React from "react";

export function Radio({ checked, onChange, label, style, ...rest }) {
  return (
    <label style={{ display: "inline-flex", alignItems: "center", gap: 10, cursor: "pointer", fontFamily: "var(--font-text)", fontSize: "var(--text-body-size)", color: "var(--text-ink)", ...style }}>
      <span
        onClick={() => onChange && onChange(true)}
        style={{
          width: 20,
          height: 20,
          borderRadius: "50%",
          border: checked ? "6px solid var(--accent-primary)" : "1.5px solid var(--border-hairline)",
          background: "var(--surface-canvas)",
          display: "inline-flex",
          flexShrink: 0,
          transition: "border var(--duration-fast) ease",
        }}
      />
      {label}
    </label>
  );
}
