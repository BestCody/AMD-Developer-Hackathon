/* app.js -- front-end logic for the Phase N UIR web UI.
   No build pipeline: a single vanilla-JS module is plenty. */

(() => {
  const $ = (sel) => document.querySelector(sel);

  const fileInput = $("#file-input");
  const fileLabel = $("#file-label");
  const runBtn = $("#run-btn");
  const drop = document.querySelector(".file-drop");
  const statusSection = $("#status-section");
  const stageText = $("#stage-text");
  const pctText = $("#pct-text");
  const progressFill = $("#progress-fill");
  const jobIdEl = $("#job-id");
  const elapsedEl = $("#elapsed");
  const resultSection = $("#result-section");
  const resultSummary = $("#result-summary");
  const jsonOutput = $("#json-output");
  const copyBtn = $("#copy-btn");
  const downloadLink = $("#download-link");
  const errorSection = $("#error-section");
  const errorOutput = $("#error-output");

  let currentJobId = null;
  let pollHandle = null;

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
    jsonOutput.textContent = "";
    errorOutput.textContent = "";
    progressFill.style.width = "0%";
    stageText.textContent = "—";
    pctText.textContent = "0%";
    jobIdEl.textContent = "—";
    elapsedEl.textContent = "0.0s";
    if (pollHandle) { clearInterval(pollHandle); pollHandle = null; }
    currentJobId = null;
    runBtn.disabled = false;
    runBtn.textContent = "Run pipeline";
  }

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

  runBtn.addEventListener("click", () => fileInput.files[0] && submitJob(fileInput.files[0]));

  async function submitJob(file) {
    reset();
    runBtn.disabled = true;
    runBtn.textContent = "Uploading…";
    show("status-section");
    const fd = new FormData();
    fd.append("file", file);
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
    elapsedEl.textContent = `${(data.percent / 100 * 0).toFixed(1)}s`; // placeholder
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

  async function renderResult(meta) {
    if (!currentJobId) return;
    // Fetch the full UIR document for in-browser rendering.  Fall back to
    // the metadata-only view if it fails (e.g. transient network glitch).
    let uirDoc = null;
    try {
      const r = await fetch(`/api/result/${currentJobId}`);
      if (r.ok) uirDoc = await r.json();
    } catch (e) { /* swallow; we'll render metadata below */ }

    if (uirDoc) {
      const id = uirDoc.id || (meta && meta.uir_id) || "unknown";
      const title = (uirDoc.metadata && uirDoc.metadata.title) || "";
      const nChunks = (uirDoc.structure && uirDoc.structure.root && uirDoc.structure.root.children)
        ? uirDoc.structure.root.children.length
        : (meta && meta.chunk_count) ?? "?";
      const nEntities = (uirDoc.semantics && uirDoc.semantics.entities)
        ? uirDoc.semantics.entities.length
        : (meta && meta.entity_count) ?? "?";
      const titleBit = title ? ` · ${title}` : "";
      resultSummary.textContent = ` ${id}${titleBit} · ${nChunks} chunks · ${nEntities} entities`;
      jsonOutput.textContent = JSON.stringify(uirDoc, null, 2);
      downloadLink.download = `${id}.uir.json`;
    } else {
      if (!meta) return showError("Pipeline returned no result.");
      const summary = `${meta.chunk_count} chunks \u00b7 ${meta.entity_count} entities \u00b7 ${meta.elapsed_seconds}s`;
      resultSummary.textContent = " " + summary;
      jsonOutput.textContent = JSON.stringify(meta, null, 2);
      downloadLink.download = (meta.uir_id || currentJobId) + ".uir.json";
    }
    show("result-section");
    downloadLink.href = `/api/download/${currentJobId}`;
  }

  function showError(msg) {
    errorOutput.textContent = msg || "Unknown error";
    show("error-section");
  }

  copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(jsonOutput.textContent);
      copyBtn.textContent = "Copied!";
      setTimeout(() => (copyBtn.textContent = "Copy JSON"), 1200);
    } catch {
      copyBtn.textContent = "Copy failed";
      setTimeout(() => (copyBtn.textContent = "Copy JSON"), 1200);
    }
  });
})();
