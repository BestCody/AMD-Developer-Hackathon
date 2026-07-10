/* @ds-bundle: {"format":4,"namespace":"ApertureDesignSystem_0a9afd","components":[{"name":"Card","sourcePath":"components/data/Card.jsx"},{"name":"Badge","sourcePath":"components/feedback/Badge.jsx"},{"name":"Tag","sourcePath":"components/feedback/Tag.jsx"},{"name":"Toast","sourcePath":"components/feedback/Toast.jsx"},{"name":"Tooltip","sourcePath":"components/feedback/Tooltip.jsx"},{"name":"Button","sourcePath":"components/forms/Button.jsx"},{"name":"Checkbox","sourcePath":"components/forms/Checkbox.jsx"},{"name":"IconButton","sourcePath":"components/forms/IconButton.jsx"},{"name":"Input","sourcePath":"components/forms/Input.jsx"},{"name":"Radio","sourcePath":"components/forms/Radio.jsx"},{"name":"Select","sourcePath":"components/forms/Select.jsx"},{"name":"Switch","sourcePath":"components/forms/Switch.jsx"},{"name":"Tabs","sourcePath":"components/navigation/Tabs.jsx"},{"name":"Dialog","sourcePath":"components/overlay/Dialog.jsx"}],"inlinedExternals":[],"unexposedExports":[]} */

(() => {

const __ds_ns = (window.ApertureDesignSystem_0a9afd = window.ApertureDesignSystem_0a9afd || {});

const __ds_scope = {};

(__ds_ns.__errors = __ds_ns.__errors || []);

// components/data/Card.jsx
try { (() => {
/** Store-utility-card grammar: white, hairline border, radius-lg, lg padding. The all-purpose content container. */
function Card({
  image,
  title,
  subtitle,
  action,
  children,
  style
}) {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      background: "var(--surface-canvas)",
      border: "1px solid var(--border-hairline)",
      borderRadius: "var(--radius-lg)",
      padding: "var(--space-lg)",
      fontFamily: "var(--font-text)",
      ...style
    }
  }, image && /*#__PURE__*/React.createElement("div", {
    style: {
      borderRadius: "var(--radius-sm)",
      overflow: "hidden",
      marginBottom: 16,
      background: "var(--surface-parchment)"
    }
  }, image), title && /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: "var(--text-body-strong-size)",
      fontWeight: "var(--text-body-strong-weight)",
      color: "var(--text-ink)"
    }
  }, title), subtitle && /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: "var(--text-body-size)",
      color: "var(--text-muted-48)",
      marginTop: 2
    }
  }, subtitle), children && /*#__PURE__*/React.createElement("div", {
    style: {
      marginTop: 12
    }
  }, children), action && /*#__PURE__*/React.createElement("div", {
    style: {
      marginTop: 12
    }
  }, action));
}
Object.assign(__ds_scope, { Card });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/data/Card.jsx", error: String((e && e.message) || e) }); }

// components/feedback/Badge.jsx
try { (() => {
const kindStyle = {
  processing: {
    bg: "var(--status-processing-bg)",
    fg: "var(--status-processing)"
  },
  success: {
    bg: "var(--status-success-bg)",
    fg: "var(--status-success)"
  },
  warning: {
    bg: "var(--status-warning-bg)",
    fg: "var(--status-warning)"
  },
  error: {
    bg: "var(--status-error-bg)",
    fg: "var(--status-error)"
  },
  neutral: {
    bg: "var(--gray-100)",
    fg: "var(--text-muted-80)"
  }
};

/** Status pill — Aperture addition for pipeline job states. */
function Badge({
  kind = "neutral",
  children,
  style
}) {
  const k = kindStyle[kind] || kindStyle.neutral;
  return /*#__PURE__*/React.createElement("span", {
    style: {
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
      ...style
    }
  }, children);
}
Object.assign(__ds_scope, { Badge });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/feedback/Badge.jsx", error: String((e && e.message) || e) }); }

// components/feedback/Tag.jsx
try { (() => {
/**
 * Configurator-style option chip — tappable pill with a selected state
 * (2px accent-focus border), matching the source's iPhone buy-page grid.
 */
function Tag({
  selected = false,
  children,
  onClick,
  style
}) {
  return /*#__PURE__*/React.createElement("button", {
    onClick: onClick,
    style: {
      fontFamily: "var(--font-text)",
      fontSize: "var(--text-caption-size)",
      color: "var(--text-ink)",
      background: "var(--surface-canvas)",
      border: selected ? "2px solid var(--accent-primary-focus)" : "1px solid var(--border-hairline)",
      borderRadius: "var(--radius-pill)",
      padding: selected ? "11px 15px" : "12px 16px",
      cursor: "pointer",
      display: "inline-flex",
      alignItems: "center",
      gap: 8,
      ...style
    }
  }, children);
}
Object.assign(__ds_scope, { Tag });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/feedback/Tag.jsx", error: String((e && e.message) || e) }); }

// components/feedback/Toast.jsx
try { (() => {
/**
 * Bottom-floating notification bar. Uses the same frosted-parchment
 * treatment as the source's floating-sticky-bar / sub-nav-frosted.
 */
function Toast({
  children,
  action,
  onDismiss,
  style
}) {
  return /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      gap: 16,
      background: "rgba(245,245,247,0.8)",
      backdropFilter: "saturate(180%) blur(20px)",
      WebkitBackdropFilter: "saturate(180%) blur(20px)",
      borderRadius: "var(--radius-lg)",
      boxShadow: "var(--shadow-ring)",
      padding: "14px 20px",
      fontFamily: "var(--font-text)",
      fontSize: "var(--text-body-size)",
      color: "var(--text-ink)",
      ...style
    }
  }, /*#__PURE__*/React.createElement("span", null, children), /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      gap: 8,
      alignItems: "center"
    }
  }, action, onDismiss && /*#__PURE__*/React.createElement("button", {
    onClick: onDismiss,
    style: {
      border: "none",
      background: "transparent",
      color: "var(--text-muted-48)",
      cursor: "pointer",
      fontSize: 18,
      lineHeight: 1
    }
  }, "\xD7")));
}
Object.assign(__ds_scope, { Toast });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/feedback/Toast.jsx", error: String((e && e.message) || e) }); }

// components/feedback/Tooltip.jsx
try { (() => {
/** Small dark tooltip — Aperture addition, minimal and quiet like the rest of the chrome. */
function Tooltip({
  label,
  children
}) {
  const [show, setShow] = React.useState(false);
  return /*#__PURE__*/React.createElement("span", {
    style: {
      position: "relative",
      display: "inline-flex"
    },
    onMouseEnter: () => setShow(true),
    onMouseLeave: () => setShow(false)
  }, children, show && /*#__PURE__*/React.createElement("span", {
    style: {
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
      zIndex: 10
    }
  }, label));
}
Object.assign(__ds_scope, { Tooltip });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/feedback/Tooltip.jsx", error: String((e && e.message) || e) }); }

// components/forms/Button.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
const sizePad = {
  default: "11px 22px",
  large: "14px 28px",
  utility: "8px 15px",
  capsule: "8px 14px"
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
  whiteSpace: "nowrap"
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
        padding: sizePad.default
      };
    case "dark-utility":
      return {
        background: "var(--text-ink)",
        color: "var(--text-on-dark)",
        borderRadius: "var(--radius-sm)",
        fontSize: "var(--text-button-utility-size)",
        letterSpacing: "var(--text-button-utility-tracking)",
        fontWeight: 400,
        padding: sizePad.utility
      };
    case "pearl-capsule":
      return {
        background: "var(--surface-pearl)",
        color: "var(--text-muted-80)",
        border: "3px solid var(--border-divider-soft)",
        borderRadius: "var(--radius-md)",
        fontSize: "var(--text-caption-size)",
        fontWeight: 400,
        padding: sizePad.capsule
      };
    case "store-hero":
      return {
        background: "var(--accent-primary)",
        color: "var(--on-accent)",
        borderRadius: "var(--radius-pill)",
        fontSize: "var(--text-button-large-size)",
        fontWeight: 300,
        padding: sizePad.large
      };
    case "primary":
    default:
      return {
        background: "var(--accent-primary)",
        color: "var(--on-accent)",
        borderRadius: "var(--radius-pill)",
        fontSize: "17px",
        fontWeight: 400,
        padding: sizePad.default
      };
  }
}

/**
 * Aperture's action button. The full-pill radius on `primary` IS the brand's
 * action signal — reserve it for the one thing you want clicked.
 * Press state is always `scale(0.95)`, never a color or shadow change.
 */
function Button({
  variant = "primary",
  disabled = false,
  children,
  onClick,
  style,
  ...rest
}) {
  const [pressed, setPressed] = React.useState(false);
  const vs = variantStyle(variant);
  return /*#__PURE__*/React.createElement("button", _extends({
    onClick: onClick,
    disabled: disabled,
    onMouseDown: () => setPressed(true),
    onMouseUp: () => setPressed(false),
    onMouseLeave: () => setPressed(false),
    style: {
      ...base,
      ...vs,
      opacity: disabled ? 0.45 : 1,
      cursor: disabled ? "not-allowed" : "pointer",
      transform: pressed && !disabled ? "scale(var(--scale-press))" : "scale(1)",
      ...style
    }
  }, rest), children);
}
Object.assign(__ds_scope, { Button });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/Button.jsx", error: String((e && e.message) || e) }); }

// components/forms/Checkbox.jsx
try { (() => {
function Checkbox({
  checked,
  onChange,
  label,
  style,
  ...rest
}) {
  return /*#__PURE__*/React.createElement("label", {
    style: {
      display: "inline-flex",
      alignItems: "center",
      gap: 10,
      cursor: "pointer",
      fontFamily: "var(--font-text)",
      fontSize: "var(--text-body-size)",
      color: "var(--text-ink)",
      ...style
    }
  }, /*#__PURE__*/React.createElement("span", {
    onClick: () => onChange && onChange(!checked),
    style: {
      width: 20,
      height: 20,
      borderRadius: 6,
      border: checked ? "none" : "1.5px solid var(--border-hairline)",
      background: checked ? "var(--accent-primary)" : "var(--surface-canvas)",
      display: "inline-flex",
      alignItems: "center",
      justifyContent: "center",
      transition: "background var(--duration-fast) ease",
      flexShrink: 0
    }
  }, checked && /*#__PURE__*/React.createElement("svg", {
    width: "12",
    height: "10",
    viewBox: "0 0 12 10",
    fill: "none"
  }, /*#__PURE__*/React.createElement("path", {
    d: "M1 5L4.3 8.3L11 1.5",
    stroke: "white",
    strokeWidth: "1.8",
    strokeLinecap: "round",
    strokeLinejoin: "round"
  }))), label);
}
Object.assign(__ds_scope, { Checkbox });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/Checkbox.jsx", error: String((e && e.message) || e) }); }

// components/forms/IconButton.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * Circular chip button for controls floating over photography or dense
 * toolbars (carousel arrows, close buttons, in-image controls).
 */
function IconButton({
  icon,
  translucent = true,
  size = 44,
  onClick,
  label,
  style,
  ...rest
}) {
  const [pressed, setPressed] = React.useState(false);
  return /*#__PURE__*/React.createElement("button", _extends({
    onClick: onClick,
    "aria-label": label,
    onMouseDown: () => setPressed(true),
    onMouseUp: () => setPressed(false),
    onMouseLeave: () => setPressed(false),
    style: {
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
      ...style
    }
  }, rest), icon);
}
Object.assign(__ds_scope, { IconButton });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/IconButton.jsx", error: String((e && e.message) || e) }); }

// components/forms/Input.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/**
 * Text input. `pill` mode matches the source's search-input grammar
 * (full pill, 44px height); `default` is a standard rectangular field
 * for forms the source doesn't define (Aperture addition).
 */
function Input({
  variant = "default",
  icon,
  placeholder,
  value,
  onChange,
  style,
  ...rest
}) {
  const isPill = variant === "pill";
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: "relative",
      display: "inline-flex",
      alignItems: "center",
      width: "100%"
    }
  }, icon && /*#__PURE__*/React.createElement("span", {
    style: {
      position: "absolute",
      left: 16,
      color: "var(--text-muted-48)",
      display: "flex"
    }
  }, icon), /*#__PURE__*/React.createElement("input", _extends({
    placeholder: placeholder,
    value: value,
    onChange: onChange,
    style: {
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
      ...style
    },
    onFocus: e => {
      e.target.style.borderColor = "var(--accent-primary-focus)";
      e.target.style.boxShadow = "0 0 0 3px rgba(0,113,227,0.15)";
    },
    onBlur: e => {
      e.target.style.borderColor = isPill ? "rgba(0,0,0,0.08)" : "var(--border-hairline)";
      e.target.style.boxShadow = "none";
    }
  }, rest)));
}
Object.assign(__ds_scope, { Input });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/Input.jsx", error: String((e && e.message) || e) }); }

// components/forms/Radio.jsx
try { (() => {
function Radio({
  checked,
  onChange,
  label,
  style,
  ...rest
}) {
  return /*#__PURE__*/React.createElement("label", {
    style: {
      display: "inline-flex",
      alignItems: "center",
      gap: 10,
      cursor: "pointer",
      fontFamily: "var(--font-text)",
      fontSize: "var(--text-body-size)",
      color: "var(--text-ink)",
      ...style
    }
  }, /*#__PURE__*/React.createElement("span", {
    onClick: () => onChange && onChange(true),
    style: {
      width: 20,
      height: 20,
      borderRadius: "50%",
      border: checked ? "6px solid var(--accent-primary)" : "1.5px solid var(--border-hairline)",
      background: "var(--surface-canvas)",
      display: "inline-flex",
      flexShrink: 0,
      transition: "border var(--duration-fast) ease"
    }
  }), label);
}
Object.assign(__ds_scope, { Radio });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/Radio.jsx", error: String((e && e.message) || e) }); }

// components/forms/Select.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/** Minimal native-backed select, styled to match Input's rectangular grammar. */
function Select({
  options = [],
  value,
  onChange,
  style,
  ...rest
}) {
  return /*#__PURE__*/React.createElement("select", _extends({
    value: value,
    onChange: onChange,
    style: {
      fontFamily: "var(--font-text)",
      fontSize: "var(--text-body-size)",
      color: "var(--text-ink)",
      background: "var(--surface-canvas)",
      border: "1px solid var(--border-hairline)",
      borderRadius: "var(--radius-sm)",
      height: 40,
      padding: "0 36px 0 14px",
      appearance: "none",
      backgroundImage: "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%237a7a7a' stroke-width='1.5' fill='none'/%3E%3C/svg%3E\")",
      backgroundRepeat: "no-repeat",
      backgroundPosition: "right 14px center",
      cursor: "pointer",
      ...style
    }
  }, rest), options.map(opt => /*#__PURE__*/React.createElement("option", {
    key: opt.value,
    value: opt.value
  }, opt.label)));
}
Object.assign(__ds_scope, { Select });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/Select.jsx", error: String((e && e.message) || e) }); }

// components/forms/Switch.jsx
try { (() => {
function Switch({
  checked,
  onChange,
  label,
  style
}) {
  return /*#__PURE__*/React.createElement("label", {
    style: {
      display: "inline-flex",
      alignItems: "center",
      gap: 10,
      cursor: "pointer",
      fontFamily: "var(--font-text)",
      fontSize: "var(--text-body-size)",
      color: "var(--text-ink)",
      ...style
    }
  }, /*#__PURE__*/React.createElement("span", {
    onClick: () => onChange && onChange(!checked),
    style: {
      width: 40,
      height: 24,
      borderRadius: "var(--radius-pill)",
      background: checked ? "var(--accent-primary)" : "var(--gray-200)",
      position: "relative",
      transition: "background var(--duration-fast) ease",
      flexShrink: 0
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      position: "absolute",
      top: 2,
      left: checked ? 18 : 2,
      width: 20,
      height: 20,
      borderRadius: "50%",
      background: "#fff",
      boxShadow: "0 1px 3px rgba(0,0,0,0.25)",
      transition: "left var(--duration-fast) var(--ease-standard)"
    }
  })), label);
}
Object.assign(__ds_scope, { Switch });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/Switch.jsx", error: String((e && e.message) || e) }); }

// components/navigation/Tabs.jsx
try { (() => {
/** Segmented toggle group — matches the source repo's own UMR/JSON view switch. */
function Tabs({
  tabs,
  active,
  onChange,
  style
}) {
  return /*#__PURE__*/React.createElement("div", {
    role: "tablist",
    style: {
      display: "inline-flex",
      background: "var(--gray-100)",
      borderRadius: "var(--radius-md)",
      padding: 2,
      gap: 2,
      ...style
    }
  }, tabs.map(t => {
    const isActive = t.value === active;
    return /*#__PURE__*/React.createElement("button", {
      key: t.value,
      role: "tab",
      "aria-selected": isActive,
      onClick: () => onChange && onChange(t.value),
      style: {
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
        transition: "background var(--duration-fast) ease, color var(--duration-fast) ease"
      }
    }, t.label);
  }));
}
Object.assign(__ds_scope, { Tabs });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/navigation/Tabs.jsx", error: String((e && e.message) || e) }); }

// components/overlay/Dialog.jsx
try { (() => {
/** Centered modal dialog — canvas surface, radius-lg, no shadow-on-chrome per the source's rules; separation comes from a dim scrim only. */
function Dialog({
  open,
  title,
  children,
  actions,
  onClose
}) {
  if (!open) return null;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      position: "fixed",
      inset: 0,
      background: "rgba(0,0,0,0.4)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      zIndex: 100
    },
    onClick: onClose
  }, /*#__PURE__*/React.createElement("div", {
    onClick: e => e.stopPropagation(),
    style: {
      background: "var(--surface-canvas)",
      borderRadius: "var(--radius-lg)",
      padding: "var(--space-xl)",
      minWidth: 360,
      maxWidth: 480,
      fontFamily: "var(--font-text)"
    }
  }, title && /*#__PURE__*/React.createElement("h3", {
    style: {
      margin: "0 0 12px",
      fontFamily: "var(--font-display)",
      fontSize: "var(--text-display-md-size)",
      fontWeight: 600,
      letterSpacing: "-0.01em",
      color: "var(--text-ink)"
    }
  }, title), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: "var(--text-body-size)",
      lineHeight: "var(--text-body-leading)",
      color: "var(--text-ink)"
    }
  }, children), actions && /*#__PURE__*/React.createElement("div", {
    style: {
      display: "flex",
      gap: 12,
      justifyContent: "flex-end",
      marginTop: 24
    }
  }, actions)));
}
Object.assign(__ds_scope, { Dialog });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/overlay/Dialog.jsx", error: String((e && e.message) || e) }); }

__ds_ns.Card = __ds_scope.Card;

__ds_ns.Badge = __ds_scope.Badge;

__ds_ns.Tag = __ds_scope.Tag;

__ds_ns.Toast = __ds_scope.Toast;

__ds_ns.Tooltip = __ds_scope.Tooltip;

__ds_ns.Button = __ds_scope.Button;

__ds_ns.Checkbox = __ds_scope.Checkbox;

__ds_ns.IconButton = __ds_scope.IconButton;

__ds_ns.Input = __ds_scope.Input;

__ds_ns.Radio = __ds_scope.Radio;

__ds_ns.Select = __ds_scope.Select;

__ds_ns.Switch = __ds_scope.Switch;

__ds_ns.Tabs = __ds_scope.Tabs;

__ds_ns.Dialog = __ds_scope.Dialog;

})();
