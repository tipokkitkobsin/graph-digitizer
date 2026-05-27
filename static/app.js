/**
 * Graph Digitizer — HF Spaces / local Docker SPA.
 * Multi-file batch predict: choose any number of images, click Predict once,
 * see one result card per image. Combined CSV (with source_file column) at end.
 */

const $ = (sel) => {
  const el = document.querySelector(sel);
  if (!el) throw new Error(`missing element ${sel}`);
  return el;
};

const fmtBytes = (n) => {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 ** 2).toFixed(1)} MB`;
};

const fmtCalib = (c) =>
  c ? `${c.slope.toFixed(4)}·px + ${c.intercept.toFixed(2)} (n=${c.n_ticks})` : "—";

const escapeHtml = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// --------- selection chips ----------

let selectedChips = []; // [{file: File, objectURL: string}]
let batchResults = []; // [{filename, ok, resp?, error?}]

function renderChips() {
  const container = $("#file-chips");
  const label = $("#image-input-label");
  container.innerHTML = "";
  if (!selectedChips.length) {
    label.textContent = "Choose image(s)";
    container.classList.add("empty");
    return;
  }
  container.classList.remove("empty");
  label.textContent = `${selectedChips.length} file${selectedChips.length === 1 ? "" : "s"} selected`;
  selectedChips.forEach((c, idx) => {
    const div = document.createElement("div");
    div.className = "chip";
    div.innerHTML = `
      <img class="chip-thumb" src="${c.objectURL}" alt="">
      <div class="chip-text">
        <span class="chip-name" title="${escapeHtml(c.file.name)}">${escapeHtml(c.file.name)}</span>
        <span class="chip-size">${fmtBytes(c.file.size)}</span>
      </div>
      <button type="button" class="chip-remove" data-idx="${idx}" title="Remove">×</button>
    `;
    container.appendChild(div);
  });
  container.querySelectorAll(".chip-remove").forEach((btn) => {
    btn.addEventListener("click", () => removeChip(Number(btn.dataset.idx)));
  });
}

function setSelection(files) {
  // Snapshot File refs first (input.files is a live FileList — clearing the input
  // afterward would invalidate it).
  const fileArr = files ? Array.from(files) : [];
  for (const c of selectedChips) URL.revokeObjectURL(c.objectURL);
  selectedChips = [];
  for (const f of fileArr) {
    selectedChips.push({ file: f, objectURL: URL.createObjectURL(f) });
  }
  renderChips();
}

function removeChip(idx) {
  if (idx < 0 || idx >= selectedChips.length) return;
  URL.revokeObjectURL(selectedChips[idx].objectURL);
  selectedChips.splice(idx, 1);
  renderChips();
  // Note: NOT clearing input.value here — that would invalidate remaining chips' File blobs.
}

function clearAll() {
  for (const c of selectedChips) URL.revokeObjectURL(c.objectURL);
  selectedChips = [];
  $("#image-file").value = "";
  renderChips();
  $("#results-list").innerHTML = "";
  $("#batch-actions").classList.add("hidden");
  const msg = $("#predict-msg");
  msg.textContent = "";
  msg.className = "msg";
  batchResults = [];
}

// --------- health + info ----------

async function refreshHealth() {
  const pill = $("#status-pill");
  try {
    const r = await fetch("/api/healthz");
    const j = await r.json();
    if (j.ok && j.model_present) {
      pill.className = "pill pill-ok";
      pill.textContent = j.model_loaded ? "ready" : "ready · model lazy-loads on first predict";
    } else if (j.ok && !j.model_present) {
      pill.className = "pill pill-err";
      pill.textContent = `missing weight: ${j.model_path}`;
    } else {
      pill.className = "pill pill-warn";
      pill.textContent = "server up, model not present";
    }
  } catch {
    pill.className = "pill pill-err";
    pill.textContent = "server unreachable";
  }
}

async function loadInfo() {
  try {
    const r = await fetch("/api/info");
    const j = await r.json();
    $("#model-info").textContent = `model: ${j.model_path}  ·  ${j.classes.length} classes`;
    $("#model-name").textContent = j.model_path;
  } catch {}
}

// --------- batch predict ----------

async function runBatchPredict() {
  const msg = $("#predict-msg");
  const list = $("#results-list");
  const actions = $("#batch-actions");
  const summary = $("#batch-summary");
  const submitBtn = $("#predict-btn");
  const clearBtn = $("#clear-selection");

  if (!selectedChips.length) {
    msg.className = "msg error";
    msg.textContent = "no files chosen";
    return;
  }

  list.innerHTML = "";
  batchResults = [];
  actions.classList.add("hidden");
  msg.className = "msg";
  submitBtn.disabled = true;
  clearBtn.disabled = true;

  const total = selectedChips.length;
  for (let i = 0; i < total; i++) {
    const file = selectedChips[i].file;
    msg.textContent = `predicting ${i + 1} of ${total}: ${file.name}`;
    const card = appendPendingCard(file.name);
    try {
      const form = new FormData();
      form.append("file", file);
      const r = await fetch("/api/predict", { method: "POST", body: form });
      const j = await r.json();
      if (!r.ok || !j.ok) {
        const errText = j.detail ?? j.phase4?.error ?? `HTTP ${r.status}`;
        renderErrorCard(card, file.name, errText);
        batchResults.push({ filename: file.name, ok: false, error: errText });
      } else {
        renderResultCard(card, file.name, j);
        batchResults.push({ filename: file.name, ok: true, resp: j });
      }
    } catch (e) {
      renderErrorCard(card, file.name, e.message);
      batchResults.push({ filename: file.name, ok: false, error: e.message });
    }
  }

  submitBtn.disabled = false;
  clearBtn.disabled = false;

  const okCount = batchResults.filter((r) => r.ok).length;
  const errCount = batchResults.length - okCount;
  const totalPts = batchResults.reduce((s, r) => s + (r.resp?.n_points ?? 0), 0);
  msg.className = errCount === 0 ? "msg success" : "msg error";
  msg.textContent = `done: ${okCount}/${total} succeeded · ${totalPts} points extracted`;
  summary.className = errCount === 0 ? "msg success" : "msg error";
  summary.textContent = `combined CSV will include ${okCount} image(s) and ${totalPts} rows`;
  if (okCount > 0) actions.classList.remove("hidden");
}

function appendPendingCard(filename) {
  const tpl = $("#result-card-template");
  const card = tpl.content.firstElementChild.cloneNode(true);
  card.querySelector(".result-filename").textContent = filename;
  const status = card.querySelector(".result-status");
  status.className = "result-status pill pill-warn";
  status.textContent = "predicting…";
  $("#results-list").appendChild(card);
  return card;
}

function renderErrorCard(card, filename, errText) {
  const status = card.querySelector(".result-status");
  status.className = "result-status pill pill-err";
  status.textContent = "error";
  card.querySelector(".result-body").innerHTML =
    `<div class="msg error" style="padding:16px">${escapeHtml(filename)}: ${escapeHtml(errText)}</div>`;
}

function renderResultCard(card, filename, j) {
  const status = card.querySelector(".result-status");
  status.className = "result-status pill pill-ok";
  status.textContent = `${j.n_series} series · ${j.n_points} pts`;

  card.querySelector(".result-image").src = `data:image/png;base64,${j.annotated_png_b64}`;

  const p = j.phase4;
  const dl = card.querySelector(".result-summary");
  dl.innerHTML = `
    <dt>chart</dt><dd>${p.chart_class ?? "?"} (conf ${(p.plot_conf ?? 0).toFixed(3)})</dd>
    <dt>x calib</dt><dd>${fmtCalib(p.x_calib)}</dd>
    <dt>y calib</dt><dd>${fmtCalib(p.y_calib)}</dd>
    <dt>OCR x</dt><dd>${(p.x_numbers_ocr ?? []).join(", ") || "—"}</dd>
    <dt>OCR y</dt><dd>${(p.y_numbers_ocr ?? []).join(", ") || "—"}</dd>
    <dt>fallback</dt><dd>x=${p.used_fallback?.x_axis ? "yes" : "no"}, y=${p.used_fallback?.y_axis ? "yes" : "no"}</dd>
    <dt>legend</dt><dd>${p.legend_xyxy ? "detected (excluded)" : "—"}</dd>
  `;

  const tbody = card.querySelector(".result-series tbody");
  for (const s of j.series) {
    const markers = new Set(s.points.map((pt) => pt.marker).filter(Boolean));
    const markerStr = markers.size ? Array.from(markers).join(", ") : "—";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>#${s.series_idx}</td>
      <td><span class="swatch" style="background:${s.color_hex}"></span><span class="mono">${s.color_hex}</span></td>
      <td>${markerStr}</td>
      <td class="mono">${s.n_points}</td>
    `;
    tbody.appendChild(tr);
  }
}

// --------- combined CSV download ----------

function downloadCombinedCsv() {
  const ok = batchResults.filter((r) => r.ok && r.resp);
  if (!ok.length) return;
  const lines = ["source_file,series,color_hex,marker,x,y,pixel_x,pixel_y"];
  for (const r of ok) {
    const rows = (r.resp.csv ?? "").trim().split("\n");
    for (let i = 1; i < rows.length; i++) {
      lines.push(`${r.filename},${rows[i]}`);
    }
  }
  const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "graph_digitizer_batch.csv";
  a.click();
  URL.revokeObjectURL(url);
}

// --------- wiring ----------

document.addEventListener("DOMContentLoaded", () => {
  refreshHealth();
  loadInfo();

  $("#image-file").addEventListener("change", (e) => {
    setSelection(e.target.files);
  });

  $("#predict-form").addEventListener("submit", (e) => {
    e.preventDefault();
    runBatchPredict();
  });

  $("#clear-selection").addEventListener("click", clearAll);
  $("#download-csv").addEventListener("click", downloadCombinedCsv);
});
