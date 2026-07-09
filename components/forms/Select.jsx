import React from "react";

/** Minimal native-backed select, styled to match Input's rectangular grammar. */
export function Select({ options = [], value, onChange, style, ...rest }) {
  return (
    <select
      value={value}
      onChange={onChange}
      style={{
        fontFamily: "var(--font-text)",
        fontSize: "var(--text-body-size)",
        color: "var(--text-ink)",
        background: "var(--surface-canvas)",
        border: "1px solid var(--border-hairline)",
        borderRadius: "var(--radius-sm)",
        height: 40,
        padding: "0 36px 0 14px",
        appearance: "none",
        backgroundImage:
          "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%237a7a7a' stroke-width='1.5' fill='none'/%3E%3C/svg%3E\")",
        backgroundRepeat: "no-repeat",
        backgroundPosition: "right 14px center",
        cursor: "pointer",
        ...style,
      }}
      {...rest}
    >
      {options.map((opt) => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  );
}
