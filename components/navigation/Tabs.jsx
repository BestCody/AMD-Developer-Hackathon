import React from "react";

/** Segmented toggle group — matches the source repo's own UMR/JSON view switch. */
export function Tabs({ tabs, active, onChange, style }) {
  return (
    <div
      role="tablist"
      style={{
        display: "inline-flex",
        background: "var(--gray-100)",
        borderRadius: "var(--radius-md)",
        padding: 2,
        gap: 2,
        ...style,
      }}
    >
      {tabs.map((t) => {
        const isActive = t.value === active;
        return (
          <button
            key={t.value}
            role="tab"
            aria-selected={isActive}
            onClick={() => onChange && onChange(t.value)}
            style={{
              appearance: "none",
              border: "none",
              background: isActive ? "var(--surface-canvas)" : "transparent",
              color: isActive ? "var(--text-ink)" : "var(--text-muted-48)",
              fontFamily: "var(--font-text)",
              fontWeight: 600,
              fontSize: "var(--text-caption-strong-size)",
              padding: "7px 14px",
              borderRadius: "var(--radius-sm)",
              cursor: "pointer",
              boxShadow: isActive ? "var(--shadow-ring)" : "none",
              transition: "background var(--duration-fast) ease, color var(--duration-fast) ease",
            }}
          >
            {t.label}
          </button>
        );
      })}
    </div>
  );
}
