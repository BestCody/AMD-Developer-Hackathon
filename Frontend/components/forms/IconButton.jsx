import React from "react";

/**
 * Circular chip button for controls floating over photography or dense
 * toolbars (carousel arrows, close buttons, in-image controls).
 */
export function IconButton({ icon, translucent = true, size = 44, onClick, label, style, ...rest }) {
  const [pressed, setPressed] = React.useState(false);
  return (
    <button
      onClick={onClick}
      aria-label={label}
      onMouseDown={() => setPressed(true)}
      onMouseUp={() => setPressed(false)}
      onMouseLeave={() => setPressed(false)}
      style={{
        width: size,
        height: size,
        borderRadius: "var(--radius-full)",
        border: "none",
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        cursor: "pointer",
        background: translucent ? "var(--surface-chip-translucent)" : "var(--surface-canvas)",
        color: "var(--text-ink)",
        boxShadow: translucent ? "none" : "var(--shadow-ring)",
        transform: pressed ? "scale(var(--scale-press))" : "scale(1)",
        transition: "transform var(--duration-press) var(--ease-standard)",
        ...style,
      }}
      {...rest}
    >
      {icon}
    </button>
  );
}
