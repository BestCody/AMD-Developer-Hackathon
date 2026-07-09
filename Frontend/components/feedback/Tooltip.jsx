import React from "react";

/** Small dark tooltip — Aperture addition, minimal and quiet like the rest of the chrome. */
export function Tooltip({ label, children }) {
  const [show, setShow] = React.useState(false);
  return (
    <span
      style={{ position: "relative", display: "inline-flex" }}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      {show && (
        <span
          style={{
            position: "absolute",
            bottom: "calc(100% + 8px)",
            left: "50%",
            transform: "translateX(-50%)",
            background: "var(--text-ink)",
            color: "var(--text-on-dark)",
            fontFamily: "var(--font-text)",
            fontSize: "var(--text-caption-size)",
            padding: "6px 10px",
            borderRadius: "var(--radius-xs)",
            whiteSpace: "nowrap",
            zIndex: 10,
          }}
        >
          {label}
        </span>
      )}
    </span>
  );
}
