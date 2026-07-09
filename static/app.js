/* app.js -- front-end logic for the Phase N UIR web UI.
   Now also handles the optional ``intent`` reader-query input. */

(() => {
  const $ = (sel) => document.querySelector(sel);

  const fileInput = $("#file-input");
  const fileLabel = $("#file-label");
  const runBtn = $("#run-btn");
  const intentInput = $("#intent-input");
  const drop = document.querySelector(".file-drop");
  const statusSection = $("#status-section");
  const stageText = $("#stage-text");
  const pctText = $("#pct-text");
  const progressFill = $("#progress-fill");
  const jobIdEl = $("#job-id");
  const elapsedEl = $("#elapsed");
  const resultSection = $("#result-section");
  const resultTitle = $("#result-title");
  const resultSummary = $("#result-summary");
  const umrOutput = $("#umr-output");
  const jsonOutput = $("#json-output");
  const copyBtn = $("#copy-btn");
  const downloadLink = $("#download-link");
  const viewUmrBtn = $("#view-umr-btn");
  const viewJsonBtn = $("#view-json-btn");
  const errorSection = $("#error-section");
  const errorOutput = $("#error-output");

  let currentJobId = null;
  let pollHandle = null;
  // Most-recent /api/status payload -- used by renderResult to surface
  // the intent-filter summary alongside the UIR preview.
  let lastStatus = null;
  // Most-recent UIR-doc fetch (cached for the JSON-tab swap). UMR is
  // already cached as ``umrOutput.textContent`` so a tab switch doesn't
  // re-fetch. Both cached locally because flipping tabs on a slow
  // network would otherwise feel sluggish.
  let lastUirDoc = null;

  function setStatus({ stage, percent }) {
    stageText.textContent = stage || "—";
    pctText.textContent = `${percent || 0}%`;
    progressFill.style.width = `${percent || 0}%`;
  }

  function show(id)  { document.getElementById(id)?.classList.remove("hidden"); }
  function hide(id)  { document.getElementById(id)?.classList.add("hidden"); }

  function reset() {
    hide("status-section");
    hide("result-section");
    hide("error-section");
    umrOutput.textContent = "";
    jsonOutput.textContent = "";
    errorOutput.textContent = "";
    progressFill.style.width = "0%";
    stageText.textContent = "—";
    pctText.textContent = "0%";
    jobIdEl.textContent = "—";
    elapsedEl.textContent = "0.0s";
    lastUirDoc = null;
    setActiveView("umr");
    // Keep the intent text across ``Run again`` clicks so the user can
    // hit replay on a slightly different PDF without re-typing.
    if (pollHandle) { clearInterval(pollHandle); pollHandle = null; }
    currentJobId = null;
    runBtn.disabled = false;
    runBtn.textContent = "Run pipeline";
  }

  function setActiveView(which) {
    // which === "umr" | "json"
    const showUmr = which !== "json";
    umrOutput.classList.toggle("hidden", !showUmr);
    jsonOutput.classList.toggle("hidden", showUmr);
    viewUmrBtn.classList.toggle("active", showUmr);
    viewJsonBtn.classList.toggle("active", !showUmr);
    viewUmrBtn.setAttribute("aria-selected", String(showUmr));
    viewJsonBtn.setAttribute("aria-selected", String(!showUmr));
    copyBtn.textContent = showUmr ? "Copy UMR" : "Copy JSON";
  }

  viewUmrBtn?.addEventListener("click", () => setActiveView("umr"));
  viewJsonBtn?.addEventListener("click", () => {
    // On-demand fetch so a slow /api/result only pays when the user
    // explicitly requests the verbose JSON view.
    if (!currentJobId) return;
    if (!lastUirDoc) fetchAndFillJson();
    setActiveView("json");
  });

  function setFile(file) {
    if (!file) { runBtn.disabled = true; fileLabel.textContent = "Click to choose a PDF"; return; }
    fileLabel.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    runBtn.disabled = false;
  }

  fileInput.addEventListener("change", (e) => setFile(e.target.files[0]));
  // Drag-and-drop support.
  ["dragenter", "dragover"].forEach((evt) =>
    drop.addEventListener(evt, (e) => { e.preventDefault(); drop.classList.add("is-dragging"); }));
  ["dragleave", "drop"].forEach((evt) =>
    drop.addEventListener(evt, (e) => { e.preventDefault(); drop.classList.remove("is-dragging"); }));
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files?.[0];
    if (f) { fileInput.files = e.dataTransfer.files; setFile(f); }
  });

  // Pressing Enter on the intent input jumps straight to submission if a
  // file is already picked. Saves a round-trip when the user has filled
  // the query but not clicked the button.
  intentInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && fileInput.files?.[0] && !runBtn.disabled) {
      e.preventDefault();
      submitJob(fileInput.files[0]);
    }
  });

  runBtn.addEventListener("click", () => fileInput.files?.[0] && submitJob(fileInput.files[0]));

  async function submitJob(file) {
    reset();
    runBtn.disabled = true;
    runBtn.textContent = "Uploading…";
    show("status-section");
    const fd = new FormData();
    fd.append("file", file);
    const userIntent = (intentInput?.value || "").trim();
    if (userIntent) fd.append("intent", userIntent);
    let resp;
    try {
      resp = await fetch("/api/run", { method: "POST", body: fd });
    } catch (e) {
      return showError(`Network error: ${e.message}`);
    }
    if (!resp.ok) {
      const text = await resp.text();
      return showError(`Upload failed: ${resp.status} ${text}`);
    }
    const { job_id } = await resp.json();
    currentJobId = job_id;
    jobIdEl.textContent = job_id.slice(0, 8) + "…";
    runBtn.textContent = "Running…";
    setStatus({ stage: "ingest", percent: 5 });
    poll();
    pollHandle = setInterval(poll, 700);
  }

  async function poll() {
    if (!currentJobId) return;
    let data;
    try {
      const r = await fetch(`/api/status/${currentJobId}`);
      data = await r.json();
    } catch (e) { return; }   // transient: try again next tick
    setStatus({ stage: data.stage, percent: data.percent });
    lastStatus = data;
    if (data.finished_at && data.submitted_at) {
      const s = Math.max(0, data.finished_at - data.submitted_at);
      elapsedEl.textContent = `${s.toFixed(2)}s`;
    } else if (data.submitted_at) {
      const s = Math.max(0, (Date.now() / 1000) - data.submitted_at);
      elapsedEl.textContent = `${s.toFixed(1)}s`;
    }
    if (data.status === "done") {
      clearInterval(pollHandle); pollHandle = null;
      runBtn.disabled = false; runBtn.textContent = "Run again";
      renderResult(data.result);
    } else if (data.status === "error") {
      clearInterval(pollHandle); pollHandle = null;
      runBtn.disabled = false; runBtn.textContent = "Run again";
      showError(data.error || "unknown error");
    }
  }

  function renderIntentSummary(baseText, intentSummary) {
    if (!intentSummary) return baseText;
    const kw = intentSummary.keywords && intentSummary.keywords.length
      ? `keywords: ${intentSummary.keywords.join(", ")}`
      : "no extractable keywords";
    const fallback = intentSummary.no_match_fallback
      ? " · expanded to full document (no keyword hit)"
      : "";
    return `${baseText} · ${intentSummary.matched_chunks} of ${intentSummary.total_chunks} chunks (intent: "${intentSummary.query}"; ${kw})${fallback}`;
  }

  async function renderResult(meta) {
    if (!currentJobId) return;
    // Fetch the UMR companion file (Phase 17 -- agent-facing view) and
    // also the UIR JSON so the user can flip tabs on demand. UMR is
    // primary; JSON is opt-in for debugging.
    let umrText = "";
    let uirDoc = null;
    try {
      const r = await fetch(`/api/umr/${currentJobId}`);
      if (r.ok) umrText = await r.text();
    } catch (e) { /* swallow; we'll render metadata below */ }
    try {
      const r = await fetch(`/api/result/${currentJobId}`);
      if (r.ok) uirDoc = await r.json();
    } catch (e) { /* swallow; JSON view will be empty until tab click */ }
    lastUirDoc = uirDoc;
    const intentSummary = lastStatus && lastStatus.intent;

    if (umrText) {
      umrOutput.textContent = umrText;
      // Also set the JSON output if we have it, so the JSON tab works without refetch
      if (uirDoc) {
        jsonOutput.textContent = JSON.stringify(uirDoc, null, 2);
      }
      // The summary line still uses UIR metadata for accurate counts
      // (chunk_count is the source of truth, not the UMR content).
      const id = (uirDoc && uirDoc.id) || (meta && meta.uir_id) || "unknown";
      const title = (uirDoc && uirDoc.metadata && uirDoc.metadata.title) || "";
      const nChunks = (meta && meta.chunk_count) ??
        ((uirDoc && uirDoc.structure && uirDoc.structure.root &&
          uirDoc.structure.root.children)
            ? uirDoc.structure.root.children.length : "?");
      const titleBit = title ? ` · ${title}` : "";
      const baseSummary = `${id}${titleBit} · ${nChunks} chunks`;
      resultSummary.textContent = renderIntentSummary(baseSummary, intentSummary);
      resultTitle.firstChild.nodeValue = "UMR ";
      // Set download link to the UIR file (consistent with the /api/download endpoint)
      downloadLink.download = `${id}.uir.json`;
    } else {
      // Back-compat: UMR endpoint missing \u2014 fall through to JSON-only
      // rendering so the UI never breaks when the WSGI server is older.
      if (uirDoc) {
        const id = uirDoc.id || (meta && meta.uir_id) || "unknown";
        const title = (uirDoc.metadata && uirDoc.metadata.title) || "";
        const nChunks = (uirDoc.structure && uirDoc.structure.root &&
                          uirDoc.structure.root.children)
          ? uirDoc.structure.root.children.length
          : (meta && meta.chunk_count) ?? "?";
        const nEntities = (uirDoc.semantics && uirDoc.semantics.entities)
          ? uirDoc.semantics.entities.length
          : (meta && meta.entity_count) ?? "?";
        const titleBit = title ? ` \u00b7 ${title}` : "";
        const baseSummary = `${id}${titleBit} \u00b7 ${nChunks} chunks \u00b7 ${nEntities} entities`;
        resultSummary.textContent = renderIntentSummary(baseSummary, intentSummary);
        jsonOutput.textContent = JSON.stringify(uirDoc, null, 2);
        downloadLink.download = `${id}.uir.json`;
      } else if (meta) {
        const baseSummary = `${meta.chunk_count} chunks \u00b7 ${meta.entity_count} entities \u00b7 ${meta.elapsed_seconds}s`;
        resultSummary.textContent = " " + baseSummary;
        jsonOutput.textContent = JSON.stringify(meta, null, 2);
        downloadLink.download = (meta.uir_id || currentJobId) + ".uir.json";
      } else {
        return showError("Pipeline returned no result.");
      }
    }
    show("result-section");
    setActiveView("umr");  // default: UMR markdown
    downloadLink.href = `/api/download/${currentJobId}`;
  }

  async function fetchAndFillJson() {
    if (!currentJobId) return;
    try {
      const r = await fetch(`/api/result/${currentJobId}`);
      if (r.ok) {
        lastUirDoc = await r.json();
        jsonOutput.textContent = JSON.stringify(lastUirDoc, null, 2);
      } else {
        jsonOutput.textContent = `HTTP ${r.status}: ${await r.text()}`;
      }
    } catch (e) {
      jsonOutput.textContent = `Network error: ${e.message}`;
    }
  }

  function showError(msg) {
    errorOutput.textContent = msg || "Unknown error";
    show("error-section");
  }

  copyBtn.addEventListener("click", async () => {
    let target;
    if (viewUmrBtn.classList.contains("active")) {
      target = umrOutput;
    } else {
      target = jsonOutput;
    }
    const label = target === umrOutput ? "Copy UMR" : "Copy JSON";
    try {
      await navigator.clipboard.writeText(target.textContent);
      copyBtn.textContent = "Copied!";
      setTimeout(() => (copyBtn.textContent = label), 1200);
    } catch {
      copyBtn.textContent = "Copy failed";
      setTimeout(() => (copyBtn.textContent = label), 1200);
    }
  });
})();
