import React from "react";

/**
 * Text input. `pill` mode matches the source's search-input grammar
 * (full pill, 44px height); `default` is a standard rectangular field
 * for forms the source doesn't define (Aperture addition).
 */
export function Input({ variant = "default", icon, placeholder, value, onChange, style, ...rest }) {
  const isPill = variant === "pill";
  return (
    <div style={{ position: "relative", display: "inline-flex", alignItems: "center", width: "100%" }}>
      {icon && (
        <span style={{ position: "absolute", left: 16, color: "var(--text-muted-48)", display: "flex" }}>{icon}</span>
      )}
      <input
        placeholder={placeholder}
        value={value}
        onChange={onChange}
        style={{
          width: "100%",
          fontFamily: "var(--font-text)",
          fontSize: "var(--text-body-size)",
          color: "var(--text-ink)",
          background: "var(--surface-canvas)",
          border: isPill ? "1px solid rgba(0,0,0,0.08)" : "1px solid var(--border-hairline)",
          borderRadius: isPill ? "var(--radius-pill)" : "var(--radius-sm)",
          height: isPill ? 44 : 40,
          padding: icon ? "12px 20px 12px 42px" : "12px 20px",
          outline: "none",
          transition: "border-color var(--duration-fast) ease, box-shadow var(--duration-fast) ease",
          ...style,
        }}
        onFocus={(e) => {
          e.target.style.borderColor = "var(--accent-primary-focus)";
          e.target.style.boxShadow = "0 0 0 3px rgba(0,113,227,0.15)";
        }}
        onBlur={(e) => {
          e.target.style.borderColor = isPill ? "rgba(0,0,0,0.08)" : "var(--border-hairline)";
          e.target.style.boxShadow = "none";
        }}
        {...rest}
      />
    </div>
  );
}
