/* Markdown.jsx -- sanitized markdown renderer for chat messages.
 *
 * The chat model emits markdown (bold, lists, code, links, tables). This
 * component parses it with `marked` and sanitizes with `DOMPurify` before
 * injecting via dangerouslySetInnerHTML. Both libraries are loaded from CDN
 * in templates/console.html. The `.ap-md` class (also defined in console.html)
 * styles the rendered elements with the Apple design tokens -- no second
 * accent colour, SF Pro/Inter type, mono code on the dark tile surface.
 *
 * IIFE-wrapped: see app.jsx.
 */

(function () {

function Markdown({ text }) {
  const html = React.useMemo(() => {
    if (!text) return "";
    try {
      const raw = window.marked.parse(text, { breaks: true, gfm: true });
      return window.DOMPurify ? window.DOMPurify.sanitize(raw) : raw;
    } catch (e) {
      return String(text);  // fall back to plain text if the parser chokes
    }
  }, [text]);
  return (
    <div className="ap-md" dangerouslySetInnerHTML={{ __html: html }} />
  );
}

window.ConsoleMarkdown = Markdown;

})();
