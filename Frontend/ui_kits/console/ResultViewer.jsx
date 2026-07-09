const sampleUMR = `# Attention Is All You Need

## Abstract
The dominant sequence transduction models are based on complex
recurrent or convolutional neural networks...

## 1. Introduction
Recurrent neural networks, long short-term memory [13] and gated
recurrent [7] neural networks in particular, have been firmly
established as state of the art approaches...`;

const sampleJSON = `{
  "uiR_version": "1.0",
  "id": "doc_8f2a1e3b",
  "modal_type": "document",
  "metadata": { "title": "Attention Is All You Need", "page_count": 15 },
  "structure": {
    "root": {
      "children": [
        { "id": "section_01", "type": "section", "title": "Abstract",
          "children": [ { "id": "chunk_001", "type": "chunk", "token_count": 241, "confidence": 0.97 } ] }
      ]
    }
  }
}`;

/** UMR / UIR JSON result viewer — recreates the source repo's toggle-group result pane. */
function ResultViewer({ Tabs, Button }) {
  const [view, setView] = React.useState("umr");
  return (
    <div style={{ fontFamily: "var(--font-text)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <div style={{ fontFamily: "var(--font-display)", fontSize: 18, fontWeight: 600, color: "var(--text-ink)" }}>
          UIR v1.0 <span style={{ color: "var(--text-muted-48)", fontWeight: 400, fontSize: 14 }}>· 15 pages · 42 chunks</span>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <Tabs tabs={[{ value: "umr", label: "UMR" }, { value: "json", label: "UIR JSON" }]} active={view} onChange={setView} />
          <Button variant="dark-utility">Copy</Button>
          <Button variant="dark-utility">Download</Button>
        </div>
      </div>
      <pre
        style={{
          margin: 0,
          background: "var(--surface-tile-1)",
          color: "var(--text-on-dark)",
          borderRadius: "var(--radius-sm)",
          padding: "16px 18px",
          fontFamily: "var(--font-mono)",
          fontSize: "var(--text-mono-size)",
          lineHeight: "var(--text-mono-leading)",
          maxHeight: 280,
          overflow: "auto",
          whiteSpace: "pre-wrap",
          borderLeft: view === "umr" ? "3px solid var(--accent-primary-on-dark)" : "3px solid transparent",
        }}
      >
        {view === "umr" ? sampleUMR : sampleJSON}
      </pre>
    </div>
  );
}

window.ConsoleResultViewer = ResultViewer;
