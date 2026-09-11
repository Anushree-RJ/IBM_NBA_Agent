/**
 * AccrediAI – NBA Accreditation Assistant
 * Frontend application script.
 *
 * Sections handled:
 *  - Sidebar navigation & collapse
 *  - AI Chat (general RAG Q&A)
 *  - Document Management (upload with category, table listing, filter, delete)
 *  - SAR Assistance (topic chips, structured response rendering)
 *  - CO-PO Mapping Assistant (CO/PO text areas, NBA standard POs, response rendering)
 *  - Knowledge Base (card list, delete, refresh)
 *  - Health-check polling
 *  - Toast notifications & loading overlay
 */

"use strict";

/* =============================================================
   API endpoints
   ============================================================= */
const API = {
  UPLOAD:     "/api/upload",
  DOCUMENTS:  "/api/documents",
  CATEGORIES: "/api/documents/categories",
  CHAT:       "/api/chat",
  SAR:        "/api/sar-assist",
  COPO:       "/api/copo-map",
  HEALTH:     "/api/health",
  STATS:      "/api/stats",
};

/* =============================================================
   State
   ============================================================= */
let uploadQueue = [];

/* =============================================================
   DOM refs — global chrome
   ============================================================= */
const sidebarToggle  = document.getElementById("sidebarToggle");
const statusDot      = document.getElementById("statusDot");
const statusLabel    = document.getElementById("statusLabel");
const docCountBadge  = document.getElementById("docCountBadge");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingMsg     = document.getElementById("loadingMsg");
const toastContainer = document.getElementById("toastContainer");

/* DOM refs — chat */
const chatMessages  = document.getElementById("chatMessages");
const chatEmpty     = document.getElementById("chatEmpty");
const chatInput     = document.getElementById("chatInput");
const sendBtn       = document.getElementById("sendBtn");
const clearChatBtn  = document.getElementById("clearChatBtn");
const sourcesList   = document.getElementById("sourcesList");

/* DOM refs — document management */
const docCategory     = document.getElementById("docCategory");
const dropzone        = document.getElementById("dropzone");
const fileInput       = document.getElementById("fileInput");
const uploadQueueEl   = document.getElementById("uploadQueue");
const uploadBtn       = document.getElementById("uploadBtn");
const uploadResults   = document.getElementById("uploadResults");
const filterCategory  = document.getElementById("filterCategory");
const refreshDocsBtn  = document.getElementById("refreshDocsBtn");
const docTableWrapper = document.getElementById("docTableWrapper");

/* DOM refs — SAR */
const sarTopic       = document.getElementById("sarTopic");
const sarSubmitBtn   = document.getElementById("sarSubmitBtn");
const sarResponseArea= document.getElementById("sarResponseArea");

/* DOM refs — CO-PO */
const coInput        = document.getElementById("coInput");
const poInput        = document.getElementById("poInput");
const copoSubmitBtn  = document.getElementById("copoSubmitBtn");
const copoResponseArea = document.getElementById("copoResponseArea");
const loadNbaPos     = document.getElementById("loadNbaPos");
const clearCoBtn     = document.getElementById("clearCoBtn");
const clearPoBtn     = document.getElementById("clearPoBtn");
const addCoBtn       = document.getElementById("addCoBtn");

/* DOM refs — knowledge base */
const kbList        = document.getElementById("kbList");
const refreshKbBtn  = document.getElementById("refreshKbBtn");

/* =============================================================
   Utilities
   ============================================================= */
function showLoading(msg = "Processing…") {
  loadingMsg.textContent = msg;
  loadingOverlay.setAttribute("aria-hidden", "false");
  loadingOverlay.classList.add("visible");
}
function hideLoading() {
  loadingOverlay.setAttribute("aria-hidden", "true");
  loadingOverlay.classList.remove("visible");
}

function toast(message, type = "info", ms = 4500) {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = message;
  toastContainer.appendChild(el);
  setTimeout(() => el.remove(), ms);
}

function formatBytes(b) {
  if (b < 1024) return b + " B";
  if (b < 1048576) return (b / 1024).toFixed(1) + " KB";
  return (b / 1048576).toFixed(1) + " MB";
}

function formatTime(iso) {
  return new Date(iso || Date.now()).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatDate(iso) {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString(undefined, { day: "2-digit", month: "short", year: "numeric" });
}

/** Safe HTML escape */
function esc(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

/** Escape + convert newlines to <br> */
function nl2br(s) { return esc(s).replace(/\n/g, "<br>"); }

/* =============================================================
   Sidebar toggle
   ============================================================= */
sidebarToggle.addEventListener("click", () => {
  const collapsed = document.body.classList.toggle("sidebar-collapsed");
  sidebarToggle.setAttribute("aria-expanded", String(!collapsed));
});

/* =============================================================
   Navigation
   ============================================================= */
document.querySelectorAll(".nav-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const sec = btn.dataset.section;
    document.querySelectorAll(".nav-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    const view = document.getElementById(`view-${sec}`);
    if (view) view.classList.add("active");

    // Side-effects on navigation
    if (sec === "dashboard")  loadDashboard();
    if (sec === "knowledge")  loadKnowledgeBase();
    if (sec === "documents")  loadDocumentTable();
    if (sec === "history")    loadHistory();
    if (sec === "settings")   loadSettings();
  });
});

/* =============================================================
   Health check
   ============================================================= */
async function checkHealth() {
  try {
    const data = await fetch(API.HEALTH).then(r => r.json());
    const n = data.document_count ?? 0;
    statusDot.className     = "status-dot " + (data.status === "ok" ? "online" : "offline");
    statusLabel.textContent = `${n} doc${n !== 1 ? "s" : ""} indexed`;
    docCountBadge.textContent = `${n} doc${n !== 1 ? "s" : ""}`;
  } catch {
    statusDot.className     = "status-dot offline";
    statusLabel.textContent = "Offline";
  }
}
checkHealth();
setInterval(checkHealth, 30_000);

/* =============================================================
   CHAT — input handling
   ============================================================= */
chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + "px";
  sendBtn.disabled = !chatInput.value.trim();
});
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!sendBtn.disabled) handleSend(); }
});
sendBtn.addEventListener("click", handleSend);

document.querySelectorAll(".example-q").forEach((btn) => {
  btn.addEventListener("click", () => {
    chatInput.value = btn.dataset.q;
    chatInput.dispatchEvent(new Event("input"));
    chatInput.focus();
    handleSend();
  });
});

clearChatBtn.addEventListener("click", () => {
  Array.from(chatMessages.children).forEach((el) => { if (el !== chatEmpty) el.remove(); });
  chatEmpty.style.display = "";
  renderSources([]);
  chatInput.focus();
});

/* =============================================================
   CHAT — send / receive
   ============================================================= */
async function handleSend() {
  const question = chatInput.value.trim();
  if (!question) return;

  chatEmpty.style.display = "none";
  chatInput.value = "";
  chatInput.style.height = "auto";
  sendBtn.disabled = true;

  appendUserBubble(question);
  const typingEl = appendTypingIndicator();

  try {
    const res  = await fetch(API.CHAT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    typingEl.remove();

    if (!res.ok || data.error) {
      appendAIBubble(`I encountered an error: ${esc(data.error || "Unknown error.")}`, null, [], true);
      toast("Query failed.", "error");
    } else {
      appendAIBubble(data.answer, data.timestamp, data.sources || []);
      renderSources(data.sources || []);
    }
  } catch {
    typingEl.remove();
    appendAIBubble("Network error. Please check your connection and try again.", null, [], true);
    toast("Network error.", "error");
  }
  sendBtn.disabled = false;
  chatInput.focus();
}

/* =============================================================
   CHAT — bubble builders
   ============================================================= */
function appendUserBubble(text) {
  const row = document.createElement("div");
  row.className = "msg-row user";
  row.innerHTML = `
    <div class="msg-avatar" aria-hidden="true">U</div>
    <div class="msg-body">
      <div class="msg-bubble">${nl2br(text)}</div>
      <div class="msg-time">${formatTime()}</div>
    </div>`;
  chatMessages.appendChild(row);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendAIBubble(text, timestamp, sources = [], isError = false) {
  const row = document.createElement("div");
  row.className = `msg-row ai${isError ? " msg-error" : ""}`;
  row.setAttribute("aria-live", "polite");

  let inlineSrc = "";
  if (sources.length) {
    const items = sources.map(s => {
      const pg = s.page ? `, Page ${s.page}` : "";
      return `<strong>${esc(s.document + pg)}</strong>`;
    }).join(" &nbsp;·&nbsp; ");
    inlineSrc = `<div class="msg-inline-sources"><strong>Sources:</strong> ${items}</div>`;
  }

  row.innerHTML = `
    <div class="msg-avatar" aria-hidden="true">AI</div>
    <div class="msg-body">
      <div class="msg-bubble">${nl2br(text)}${inlineSrc}</div>
      <div class="msg-time">${formatTime(timestamp)}</div>
    </div>`;
  chatMessages.appendChild(row);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendTypingIndicator() {
  const row = document.createElement("div");
  row.className = "msg-row ai typing-row";
  row.setAttribute("aria-label", "AI is thinking");
  row.innerHTML = `
    <div class="msg-avatar" aria-hidden="true">AI</div>
    <div class="msg-body">
      <div class="msg-bubble">
        <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
      </div>
    </div>`;
  chatMessages.appendChild(row);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return row;
}

function renderSources(sources) {
  if (!sources || !sources.length) {
    sourcesList.innerHTML = '<p class="sources-placeholder">No sources matched this query.</p>';
    return;
  }
  sourcesList.innerHTML = sources.map(s => `
    <div class="source-card">
      <div class="source-card-doc">${esc(s.document)}</div>
      ${s.page ? `<div class="source-card-page">Page ${s.page}${s.section ? " · " + esc(s.section) : ""}</div>` : ""}
      <div class="source-card-excerpt">${esc(s.excerpt)}</div>
      <div class="source-card-score">Relevance score: ${s.relevance_score}</div>
    </div>`).join("");
}

/* =============================================================
   DOCUMENT MANAGEMENT — upload
   ============================================================= */
dropzone.addEventListener("click", (e) => {
  // Only open picker when clicking the dropzone itself (not the hidden input)
  if (e.target !== fileInput) fileInput.click();
});
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
dropzone.addEventListener("dragover",  (e) => { e.preventDefault(); dropzone.classList.add("drag-over"); });
dropzone.addEventListener("dragleave", ()  => dropzone.classList.remove("drag-over"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault(); dropzone.classList.remove("drag-over");
  addToQueue([...e.dataTransfer.files]);
});
fileInput.addEventListener("change", () => { addToQueue([...fileInput.files]); fileInput.value = ""; });

const ALLOWED = ["pdf", "docx", "txt", "xlsx"];

function addToQueue(files) {
  files.forEach(file => {
    const ext = file.name.split(".").pop().toLowerCase();
    if (!ALLOWED.includes(ext)) { toast(`Unsupported file type: ${file.name}`, "error"); return; }
    if (file.size > 50 * 1024 * 1024) { toast(`File too large (max 50 MB): ${file.name}`, "error"); return; }
    if (!uploadQueue.find(f => f.name === file.name)) uploadQueue.push(file);
  });
  renderQueue();
}

function renderQueue() {
  uploadQueueEl.innerHTML = uploadQueue.map((file, i) => `
    <div class="queue-item" id="qi-${i}">
      <span class="qi-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
        </svg>
      </span>
      <span class="qi-name" title="${esc(file.name)}">${esc(file.name)}</span>
      <span class="qi-size">${formatBytes(file.size)}</span>
      <button class="qi-remove" onclick="removeFromQueue(${i})" aria-label="Remove ${esc(file.name)}">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
    </div>`).join("");
  uploadBtn.disabled = uploadQueue.length === 0;
}

window.removeFromQueue = (idx) => { uploadQueue.splice(idx, 1); renderQueue(); };

uploadBtn.addEventListener("click", uploadAll);

async function uploadAll() {
  if (!uploadQueue.length) return;
  uploadBtn.disabled = true;
  uploadResults.innerHTML = "";
  const category = docCategory.value || "Other";

  for (let i = 0; i < uploadQueue.length; i++) {
    const file  = uploadQueue[i];
    const qiEl  = document.getElementById(`qi-${i}`);

    if (qiEl) {
      const badge = document.createElement("span");
      badge.className = "qi-status uploading"; badge.textContent = "Uploading…";
      qiEl.appendChild(badge);
    }

    const fd = new FormData();
    fd.append("file", file);
    fd.append("category", category);

    try {
      const res  = await fetch(API.UPLOAD, { method: "POST", body: fd });
      const data = await res.json();
      const msg  = document.createElement("div");
      if (res.ok) {
        msg.className   = "upload-msg success";
        msg.textContent = data.message || `'${file.name}' indexed successfully.`;
        const badge = qiEl?.querySelector(".qi-status");
        if (badge) { badge.className = "qi-status success"; badge.textContent = "Indexed"; }
      } else {
        msg.className   = "upload-msg error";
        msg.textContent = `${file.name}: ${data.error}`;
        const badge = qiEl?.querySelector(".qi-status");
        if (badge) { badge.className = "qi-status error"; badge.textContent = "Failed"; }
      }
      uploadResults.appendChild(msg);
    } catch {
      const msg = document.createElement("div");
      msg.className   = "upload-msg error";
      msg.textContent = `${file.name}: Network error.`;
      uploadResults.appendChild(msg);
    }
  }

  uploadQueue = [];
  renderQueue();
  checkHealth();
  loadDocumentTable();   // refresh table after upload
  toast("Upload complete.", "success");
}

/* =============================================================
   DOCUMENT MANAGEMENT — table listing
   ============================================================= */
refreshDocsBtn.addEventListener("click", loadDocumentTable);
filterCategory.addEventListener("change", loadDocumentTable);

async function loadDocumentTable() {
  const cat = filterCategory.value;
  const url = cat ? `${API.DOCUMENTS}?category=${encodeURIComponent(cat)}` : API.DOCUMENTS;
  try {
    const docs = await fetch(url).then(r => r.json());
    renderDocTable(docs);
  } catch {
    toast("Failed to load documents.", "error");
  }
}

function renderDocTable(docs) {
  if (!docs || !docs.length) {
    docTableWrapper.innerHTML = `
      <table class="doc-table doc-table-wrap"><thead>
        <tr>
          <th>File Name</th><th>Type</th><th>Category</th>
          <th>Uploaded</th><th>Status</th><th>Chunks</th><th>Action</th>
        </tr>
      </thead><tbody>
        <tr class="doc-empty-row"><td colspan="7">No documents found. Upload NBA documents to get started.</td></tr>
      </tbody></table>`;
    return;
  }

  const rows = docs.map(doc => {
    const ext  = (doc.file_type || "FILE").toLowerCase();
    const cat  = doc.category || "Other";
    const stat = doc.status || "indexed";
    return `
      <tr>
        <td><span class="doc-fname" title="${esc(doc.filename)}">${esc(doc.filename)}</span></td>
        <td><span class="doc-type-badge ${esc(ext)}">${esc(doc.file_type || "FILE")}</span></td>
        <td><span class="doc-cat-badge">${esc(cat)}</span></td>
        <td>${formatDate(doc.uploaded_at)}</td>
        <td><span class="status-badge ${esc(stat)}">${esc(stat.charAt(0).toUpperCase() + stat.slice(1))}</span></td>
        <td>${doc.chunks ?? "—"}</td>
        <td>
          <button class="btn-del-doc" onclick="deleteDoc('${esc(doc.id)}', '${esc(doc.filename)}')" aria-label="Remove ${esc(doc.filename)}">
            Delete
          </button>
        </td>
      </tr>`;
  }).join("");

  docTableWrapper.innerHTML = `
    <div class="doc-table-wrap">
      <table class="doc-table">
        <thead>
          <tr>
            <th>File Name</th><th>Type</th><th>Category</th>
            <th>Uploaded</th><th>Status</th><th>Chunks</th><th>Action</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

window.deleteDoc = async function (id, name) {
  if (!confirm(`Remove "${name}" from the knowledge base? The index will be rebuilt.`)) return;
  showLoading("Removing document and rebuilding index…");
  try {
    const res  = await fetch(`${API.DOCUMENTS}/${id}`, { method: "DELETE" });
    const data = await res.json();
    if (res.ok) {
      toast(data.message, "success");
      loadDocumentTable();
      loadKnowledgeBase();
      checkHealth();
    } else {
      toast(`Error: ${data.error}`, "error");
    }
  } catch {
    toast("Failed to remove document.", "error");
  } finally {
    hideLoading();
  }
};

/* =============================================================
   SAR ASSISTANCE
   ============================================================= */

/* Enable/disable submit based on textarea content */
sarTopic.addEventListener("input", () => {
  sarSubmitBtn.disabled = !sarTopic.value.trim();
});

/* Quick-topic chips */
document.querySelectorAll(".chip[data-sar]").forEach(chip => {
  chip.addEventListener("click", () => {
    sarTopic.value = chip.dataset.sar;
    sarTopic.dispatchEvent(new Event("input"));
    // Highlight active chip
    document.querySelectorAll(".chip[data-sar]").forEach(c => c.classList.remove("active"));
    chip.classList.add("active");
  });
});

sarSubmitBtn.addEventListener("click", handleSARSubmit);

async function handleSARSubmit() {
  const topic = sarTopic.value.trim();
  if (!topic) return;

  sarSubmitBtn.disabled = true;
  showLoading("Generating SAR guidance…");
  sarResponseArea.innerHTML = "";

  try {
    const res  = await fetch(API.SAR, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic }),
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      sarResponseArea.innerHTML = `
        <div class="upload-msg error">${esc(data.error || "Unknown error.")}</div>`;
      toast("SAR assistance failed.", "error");
    } else {
      renderSARResponse(data);
    }
  } catch {
    sarResponseArea.innerHTML = `<div class="upload-msg error">Network error. Please try again.</div>`;
    toast("Network error.", "error");
  } finally {
    sarSubmitBtn.disabled = !sarTopic.value.trim();
    hideLoading();
  }
}

function renderSARResponse(data) {
  // Extract inline disclaimer if Granite included it, otherwise render separately
  const body = data.response || "";

  // Source mini-cards
  const srcHtml = (data.sources || []).map(s => `
    <div class="source-card">
      <div class="source-card-doc">${esc(s.document)}</div>
      ${s.page ? `<div class="source-card-page">Page ${s.page}</div>` : ""}
      <div class="source-card-excerpt">${esc(s.excerpt)}</div>
    </div>`).join("") || `<p class="sources-placeholder">No specific sources matched this topic.</p>`;

  sarResponseArea.innerHTML = `
    <div class="sar-response-card">
      <div class="sar-response-header">
        <span class="sar-response-title">SAR Guidance: ${esc(data.topic)}</span>
        <span class="sar-response-meta">${formatTime(data.timestamp)}</span>
      </div>
      <div class="sar-response-body">${nl2br(body)}</div>
      <div class="sar-disclaimer">
        ⚠️ <strong>Disclaimer:</strong> This AI-generated content is based solely on the uploaded documents. All information must be reviewed and verified by faculty before inclusion in any official SAR submission.
      </div>
      <div class="sar-sources">
        <div class="sar-sources-title">Source Documents</div>
        ${srcHtml}
      </div>
    </div>`;
}

/* =============================================================
   CO-PO MAPPING ASSISTANT
   ============================================================= */

/* NBA Standard POs (12 POs as defined by NBA for UG Engineering programmes) */
const NBA_STANDARD_POS = [
  "PO1: Engineering Knowledge – Apply knowledge of mathematics, science, engineering fundamentals and an engineering specialisation to the solution of complex engineering problems.",
  "PO2: Problem Analysis – Identify, formulate, review research literature, and analyse complex engineering problems using first principles of mathematics, natural sciences and engineering sciences.",
  "PO3: Design/Development of Solutions – Design solutions for complex engineering problems that meet specified needs with appropriate consideration for public health, safety, cultural, societal, and environmental considerations.",
  "PO4: Conduct Investigations of Complex Problems – Conduct investigations of complex problems using research-based knowledge and research methods to provide valid conclusions.",
  "PO5: Modern Tool Usage – Create, select, and apply appropriate techniques, resources, and modern engineering and IT tools to complex engineering activities.",
  "PO6: The Engineer and Society – Apply reasoning informed by contextual knowledge to assess societal, health, safety, legal, and cultural issues and responsibilities relevant to professional engineering practice.",
  "PO7: Environment and Sustainability – Understand the impact of professional engineering solutions in societal and environmental contexts and demonstrate knowledge for sustainable development.",
  "PO8: Ethics – Apply ethical principles and commit to professional ethics and responsibilities and norms of engineering practice.",
  "PO9: Individual and Team Work – Function effectively as an individual, and as a member or leader in diverse teams and multidisciplinary settings.",
  "PO10: Communication – Communicate effectively on complex engineering activities with the engineering community and society at large.",
  "PO11: Project Management and Finance – Demonstrate knowledge and understanding of engineering and management principles and apply these to manage projects in multidisciplinary environments.",
  "PO12: Life-long Learning – Recognise the need for, and have the preparation and ability to engage in independent and life-long learning in the broadest context of technological change.",
];

/* Enable/disable CO-PO submit */
function updateCoPoBtn() {
  copoSubmitBtn.disabled = !coInput.value.trim() || !poInput.value.trim();
}
coInput.addEventListener("input", updateCoPoBtn);
poInput.addEventListener("input", updateCoPoBtn);

loadNbaPos.addEventListener("click", () => {
  poInput.value = NBA_STANDARD_POS.join("\n");
  updateCoPoBtn();
  toast("NBA Standard POs (12) loaded.", "success");
});

clearCoBtn.addEventListener("click", () => { coInput.value = ""; updateCoPoBtn(); });
clearPoBtn.addEventListener("click", () => { poInput.value = ""; updateCoPoBtn(); });

/* Add a numbered CO placeholder */
let coCount = 0;
addCoBtn.addEventListener("click", () => {
  coCount++;
  const n = coCount;
  const cur = coInput.value.trim();
  coInput.value = cur ? cur + `\nCO${n}: ` : `CO${n}: `;
  coInput.focus();
  coInput.setSelectionRange(coInput.value.length, coInput.value.length);
  updateCoPoBtn();
});

copoSubmitBtn.addEventListener("click", handleCoPoSubmit);

async function handleCoPoSubmit() {
  const cosRaw = coInput.value.trim();
  const posRaw = poInput.value.trim();
  if (!cosRaw || !posRaw) return;

  // Parse non-empty lines
  const cos = cosRaw.split("\n").map(l => l.trim()).filter(Boolean);
  const pos = posRaw.split("\n").map(l => l.trim()).filter(Boolean);

  if (!cos.length || !pos.length) {
    toast("Please enter at least one CO and one PO.", "warning"); return;
  }

  copoSubmitBtn.disabled = true;
  showLoading("Generating CO-PO mapping…");
  copoResponseArea.innerHTML = "";

  try {
    const res  = await fetch(API.COPO, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ course_outcomes: cos, program_outcomes: pos }),
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      copoResponseArea.innerHTML = `<div class="upload-msg error">${esc(data.error || "Unknown error.")}</div>`;
      toast("CO-PO mapping failed.", "error");
    } else {
      renderCoPoResponse(data);
    }
  } catch {
    copoResponseArea.innerHTML = `<div class="upload-msg error">Network error. Please try again.</div>`;
    toast("Network error.", "error");
  } finally {
    copoSubmitBtn.disabled = false;
    hideLoading();
  }
}

function renderCoPoResponse(data) {
  const body = data.response || "";

  const srcHtml = (data.sources || []).map(s => `
    <div class="source-card">
      <div class="source-card-doc">${esc(s.document)}</div>
      ${s.page ? `<div class="source-card-page">Page ${s.page}</div>` : ""}
      <div class="source-card-excerpt">${esc(s.excerpt)}</div>
    </div>`).join("") || `<p class="sources-placeholder">No specific sources matched this request.</p>`;

  copoResponseArea.innerHTML = `
    <div class="copo-response-card">
      <div class="copo-response-header">
        <span class="copo-response-title">CO-PO Mapping Suggestion (${data.course_outcomes?.length || 0} COs × ${data.program_outcomes?.length || 0} POs)</span>
        <span style="font-size:12px;color:var(--muted)">${formatTime(data.timestamp)}</span>
      </div>
      <div class="copo-response-body">${nl2br(body)}</div>
      <div class="copo-disclaimer">
        ⚠️ <strong>Disclaimer:</strong> This mapping is AI-suggested based on uploaded documents only. It does NOT constitute an officially approved NBA mapping. Faculty must review, modify, and approve the final mapping before submission.
      </div>
      <div class="copo-sources">
        <div class="copo-sources-title">Source Documents</div>
        ${srcHtml}
      </div>
    </div>`;
}

/* =============================================================
   KNOWLEDGE BASE — listing & deletion
   ============================================================= */
refreshKbBtn.addEventListener("click", loadKnowledgeBase);

async function loadKnowledgeBase() {
  showLoading("Loading knowledge base…");
  try {
    const docs = await fetch(API.DOCUMENTS).then(r => r.json());
    renderKB(docs);
    const n = docs.length;
    docCountBadge.textContent = `${n} doc${n !== 1 ? "s" : ""}`;
  } catch {
    toast("Failed to load knowledge base.", "error");
  } finally {
    hideLoading();
  }
}

function renderKB(docs) {
  if (!docs || !docs.length) {
    kbList.innerHTML = `
      <div class="kb-empty">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
          <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
        </svg>
        <p>No documents uploaded yet.<br/>Upload NBA documents to start querying.</p>
      </div>`;
    return;
  }

  kbList.innerHTML = docs.map(doc => {
    const date  = formatDate(doc.uploaded_at);
    const pages = doc.pages ? `${doc.pages} pages · ` : "";
    const cat   = doc.category ? ` · ${esc(doc.category)}` : "";
    return `
      <div class="kb-card" id="kbdoc-${doc.id}">
        <div class="kb-card-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
            <line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>
          </svg>
        </div>
        <div class="kb-card-info">
          <div class="kb-card-name" title="${esc(doc.filename)}">${esc(doc.filename)}</div>
          <div class="kb-card-meta">${pages}${doc.chunks} chunks · ${date}${cat}</div>
        </div>
        <button class="kb-card-del" onclick="deleteDoc('${esc(doc.id)}', '${esc(doc.filename)}')" aria-label="Remove ${esc(doc.filename)}">
          Remove
        </button>
      </div>`;
  }).join("");
}

/* =============================================================
   CHAT HISTORY — load from /api/stats recent_queries
   ============================================================= */

const refreshHistoryBtn = document.getElementById("refreshHistoryBtn");
if (refreshHistoryBtn) refreshHistoryBtn.addEventListener("click", loadHistory);

async function loadHistory() {
  const historyList = document.getElementById("historyList");
  if (!historyList) return;
  historyList.innerHTML = `<div class="history-empty">Loading…</div>`;
  try {
    const data = await fetch(API.STATS).then(r => r.json());
    const queries = data.recent_queries || [];
    if (!queries.length) {
      historyList.innerHTML = `<div class="history-empty">No queries yet. Start by asking a question in AI Chat.</div>`;
      return;
    }
    const TYPE_LABELS = { chat: "Chat", sar: "SAR", copo: "CO-PO" };
    historyList.innerHTML = queries.map(q => {
      const typeClass = `history-type-${esc(q.type || "chat")}`;
      const typeLabel = TYPE_LABELS[q.type] || "Chat";
      return `
        <div class="history-item">
          <span class="history-type-badge ${typeClass}">${typeLabel}</span>
          <div>
            <div class="history-question">${esc(q.question || "—")}</div>
            ${q.time ? `<div class="history-time">${formatDate(q.time)} · ${formatTime(q.time)}</div>` : ""}
          </div>
        </div>`;
    }).join("");
  } catch {
    historyList.innerHTML = `<div class="history-empty">Failed to load history.</div>`;
  }
}

/* =============================================================
   SETTINGS — load health + stats for connection status panel
   ============================================================= */

async function loadSettings() {
  const settingsStatus = document.getElementById("settingsStatus");
  if (!settingsStatus) return;
  settingsStatus.innerHTML = `<div style="color:var(--muted);font-size:13px;padding:8px 0">Checking connection…</div>`;
  try {
    const [health, stats] = await Promise.all([
      fetch(API.HEALTH).then(r => r.json()),
      fetch(API.STATS).then(r => r.json()),
    ]);
    const kbReady = health.knowledge_base_ready;
    const isOk    = health.status === "ok";
    settingsStatus.innerHTML = `
      <div class="settings-status-card">
        <div class="ssc-label">Backend Status</div>
        <div class="ssc-value ${isOk ? "ok" : "err"}">${isOk ? "✓ Online" : "✗ Offline"}</div>
      </div>
      <div class="settings-status-card">
        <div class="ssc-label">Knowledge Base</div>
        <div class="ssc-value ${kbReady ? "ok" : ""}">${kbReady ? "✓ Ready" : "Empty"}</div>
      </div>
      <div class="settings-status-card">
        <div class="ssc-label">LLM Model</div>
        <div class="ssc-value">${esc(health.model || stats.model || "ibm/granite-4-h-small")}</div>
      </div>
      <div class="settings-status-card">
        <div class="ssc-label">Documents</div>
        <div class="ssc-value">${esc(String(health.document_count ?? 0))}</div>
      </div>
      <div class="settings-status-card">
        <div class="ssc-label">Vector Store</div>
        <div class="ssc-value">${esc(stats.vector_store || "FAISS")}</div>
      </div>
      <div class="settings-status-card">
        <div class="ssc-label">Embed Model</div>
        <div class="ssc-value" style="font-size:11px">${esc(stats.embed_model || "all-MiniLM-L6-v2")}</div>
      </div>`;
  } catch {
    settingsStatus.innerHTML = `<div class="settings-status-card"><div class="ssc-label">Status</div><div class="ssc-value err">✗ Backend unreachable</div></div>`;
  }
}

/* =============================================================
   DASHBOARD — stats + recent queries
   ============================================================= */

/**
 * Load /api/stats and render the dashboard section:
 *  - 4 stat cards  (total_documents, indexed_documents, total_queries, kb_status)
 *  - Recent queries list
 *  - System information (model, vector store, embed model)
 */
async function loadDashboard() {
  const dashStats  = document.getElementById("dashStats");
  const recentList = document.getElementById("dashRecentQueries");
  const sysInfo    = document.getElementById("dashSysInfo");
  if (!dashStats) return;

  // Show skeleton while loading
  dashStats.innerHTML = `
    <div class="stat-card stat-card-skeleton"></div>
    <div class="stat-card stat-card-skeleton"></div>
    <div class="stat-card stat-card-skeleton"></div>
    <div class="stat-card stat-card-skeleton"></div>`;

  try {
    const data = await fetch(API.STATS).then(r => r.json());

    // ---- Stat cards ----
    const cards = [
      {
        icon: `<svg class="stat-card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>`,
        value: data.total_documents ?? 0,
        label: "Total Documents",
      },
      {
        icon: `<svg class="stat-card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`,
        value: data.indexed_documents ?? 0,
        label: "Indexed Documents",
      },
      {
        icon: `<svg class="stat-card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>`,
        value: data.total_queries ?? 0,
        label: "Total Queries",
      },
      {
        icon: `<svg class="stat-card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>`,
        value: data.knowledge_base_ready ? "Ready" : "Empty",
        label: "Knowledge Base",
      },
    ];

    dashStats.innerHTML = cards.map(c => `
      <div class="stat-card">
        ${c.icon}
        <div class="stat-value">${esc(String(c.value))}</div>
        <div class="stat-label">${esc(c.label)}</div>
      </div>`).join("");

    // ---- Recent queries ----
    const queries = data.recent_queries || [];
    if (queries.length === 0) {
      recentList.innerHTML = `<li class="recent-query-empty">No queries yet. Ask something in the AI Chat section.</li>`;
    } else {
      recentList.innerHTML = queries.map((q, i) => `
        <li class="recent-query-item">
          <span class="recent-query-num">${i + 1}</span>
          <div style="flex:1">
            <div class="recent-query-text">${esc(q.question || q)}</div>
            ${q.time ? `<div class="recent-query-time">${formatDate(q.time)} · ${formatTime(q.time)}</div>` : ""}
          </div>
        </li>`).join("");
    }

    // ---- System info ----
    const model       = data.model        || "ibm/granite-4-h-small";
    const vectorStore = data.vector_store || "FAISS";
    const embedModel  = data.embed_model  || "all-MiniLM-L6-v2";
    sysInfo.innerHTML = `
      <dt>LLM</dt>              <dd>${esc(model)}</dd>
      <dt>Vector Store</dt>     <dd>${esc(vectorStore)}</dd>
      <dt>Embed Model</dt>      <dd>${esc(embedModel)}</dd>
      <dt>SAR Queries</dt>      <dd>${esc(String(data.sar_queries ?? 0))}</dd>
      <dt>CO-PO Queries</dt>    <dd>${esc(String(data.copo_queries ?? 0))}</dd>
      <dt>Platform</dt>         <dd>IBM watsonx.ai</dd>`;

  } catch (err) {
    dashStats.innerHTML = `<div class="upload-msg error" style="grid-column:1/-1">Failed to load statistics. Make sure the backend is running.</div>`;
  }
}

/* =============================================================
   Init
   ============================================================= */
document.addEventListener("DOMContentLoaded", () => {
  chatInput.focus();
  // Load document table on page load so counts are accurate
  loadDocumentTable();
  // Pre-load dashboard stats in background
  loadDashboard();
});
