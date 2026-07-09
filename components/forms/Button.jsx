import React from "react";

const sizePad = {
  default: "11px 22px",
  large: "14px 28px",
  utility: "8px 15px",
  capsule: "8px 14px",
};

/** @type {React.CSSProperties} */
const base = {
  fontFamily: "var(--font-text)",
  border: "none",
  cursor: "pointer",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  gap: "8px",
  transition: "transform var(--duration-press) var(--ease-standard), opacity var(--duration-fast) ease",
  whiteSpace: "nowrap",
};

function variantStyle(variant) {
  switch (variant) {
    case "secondary-pill":
      return {
        background: "transparent",
        color: "var(--accent-primary)",
        border: "1px solid var(--accent-primary)",
        borderRadius: "var(--radius-pill)",
        fontSize: "17px",
        fontWeight: 400,
        padding: sizePad.default,
      };
    case "dark-utility":
      return {
        background: "var(--text-ink)",
        color: "var(--text-on-dark)",
        borderRadius: "var(--radius-sm)",
        fontSize: "var(--text-button-utility-size)",
        letterSpacing: "var(--text-button-utility-tracking)",
        fontWeight: 400,
        padding: sizePad.utility,
      };
    case "pearl-capsule":
      return {
        background: "var(--surface-pearl)",
        color: "var(--text-muted-80)",
        border: "3px solid var(--border-divider-soft)",
        borderRadius: "var(--radius-md)",
        fontSize: "var(--text-caption-size)",
        fontWeight: 400,
        padding: sizePad.capsule,
      };
    case "store-hero":
      return {
        background: "var(--accent-primary)",
        color: "var(--on-accent)",
        borderRadius: "var(--radius-pill)",
        fontSize: "var(--text-button-large-size)",
        fontWeight: 300,
        padding: sizePad.large,
      };
    case "primary":
    default:
      return {
        background: "var(--accent-primary)",
        color: "var(--on-accent)",
        borderRadius: "var(--radius-pill)",
        fontSize: "17px",
        fontWeight: 400,
        padding: sizePad.default,
      };
  }
}

/**
 * Aperture's action button. The full-pill radius on `primary` IS the brand's
 * action signal — reserve it for the one thing you want clicked.
 * Press state is always `scale(0.95)`, never a color or shadow change.
 */
export function Button({ variant = "primary", disabled = false, children, onClick, style, ...rest }) {
  const [pressed, setPressed] = React.useState(false);
  const vs = variantStyle(variant);
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      onMouseDown={() => setPressed(true)}
      onMouseUp={() => setPressed(false)}
      onMouseLeave={() => setPressed(false)}
      style={{
        ...base,
        ...vs,
        opacity: disabled ? 0.45 : 1,
        cursor: disabled ? "not-allowed" : "pointer",
        transform: pressed && !disabled ? "scale(var(--scale-press))" : "scale(1)",
        ...style,
      }}
      {...rest}
    >
      {children}
    </button>
  );
}
