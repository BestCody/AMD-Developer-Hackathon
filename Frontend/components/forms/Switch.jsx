import React from "react";

export function Switch({ checked, onChange, label, style }) {
  return (
    <label style={{ display: "inline-flex", alignItems: "center", gap: 10, cursor: "pointer", fontFamily: "var(--font-text)", fontSize: "var(--text-body-size)", color: "var(--text-ink)", ...style }}>
      <span
        onClick={() => onChange && onChange(!checked)}
        style={{
          width: 40,
          height: 24,
          borderRadius: "var(--radius-pill)",
          background: checked ? "var(--accent-primary)" : "var(--gray-200)",
          position: "relative",
          transition: "background var(--duration-fast) ease",
          flexShrink: 0,
        }}
      >
        <span
          style={{
            position: "absolute",
            top: 2,
            left: checked ? 18 : 2,
            width: 20,
            height: 20,
            borderRadius: "50%",
            background: "#fff",
            boxShadow: "0 1px 3px rgba(0,0,0,0.25)",
            transition: "left var(--duration-fast) var(--ease-standard)",
          }}
        />
      </span>
      {label}
    </label>
  );
}
