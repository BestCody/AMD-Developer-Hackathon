/* LucideIcon.jsx -- React-safe wrapper for Lucide icons.
 *
 * The CDN lucide library replaces <i data-lucide="..."> elements with SVGs,
 * which breaks React's virtual DOM reconciliation (NotFoundError on removeChild).
 * This component renders each icon inside a container span that React manages;
 * the inner <i> → SVG replacement happens inside the container via a layout
 * effect, so React only sees the stable span boundary.
 *
 * IIFE-wrapped: see app.jsx.
 */

(function () {

function LucideIcon(props) {
  const { name, size, style, className } = props;
  const ref = React.useRef(null);

  React.useLayoutEffect(function () {
    const container = ref.current;
    if (!container || !window.lucide) return;

    // Clear any previous icon (SVG or <i>).
    container.innerHTML = "";

    const i = document.createElement("i");
    i.setAttribute("data-lucide", name);
    if (size != null) {
      const sizeStr = typeof size === "number" ? size + "px" : String(size);
      i.style.width = sizeStr;
      i.style.height = sizeStr;
    }
    if (style) {
      Object.assign(i.style, style);
    }

    container.appendChild(i);

    if (window.lucide.createIcons) {
      try {
        window.lucide.createIcons({ nodes: [i] });
      } catch (err) {
        // If lucide fails, the <i> stays as a fallback.
      }
    }
  }, [name, size, JSON.stringify(style)]);

  return React.createElement("span", {
    ref: ref,
    className: className,
    style: { display: "inline-flex", alignItems: "center", justifyContent: "center" },
  });
}

window.LucideIcon = LucideIcon;

})();
