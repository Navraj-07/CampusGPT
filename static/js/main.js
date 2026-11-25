/* CampusGPT Frontend Controller */

(() => {
  const state = {
    documents: [],
    selectedDocIds: new Set(),
    primaryDocId: null,
    courseFilter: "ALL",
    apiConfigured: true,
    currentSummaryMd: "",
    currentStudyMd: "",
    currentQuizQuestions: [],
    currentQuizArtifactId: null,
  };

  // DOM references
  const courseNameInput = document.getElementById("courseNameInput");
  const courseFilterSelect = document.getElementById("courseFilterSelect");
  const uploadDropzone = document.getElementById("uploadDropzone");
  const pdfFileInput = document.getElementById("pdfFileInput");
  const browsePdfBtn = document.getElementById("browsePdfBtn");
  const uploadSpinner = document.getElementById("uploadSpinner");
  const documentList = document.getElementById("documentList");
  const docCountBadge = document.getElementById("docCountBadge");
  const selectAllDocsBtn = document.getElementById("selectAllDocsBtn");
  const activeContextLabel = document.getElementById("activeContextLabel");
  const engineModeSelect = document.getElementById("engineModeSelect");
  const apiKeyStatusBadge = document.getElementById("apiKeyStatusBadge");
  const apiAlertBanner = document.getElementById("apiAlertBanner");
  const apiAlertText = document.getElementById("apiAlertText");

  const chatMessages = document.getElementById("chatMessages");
  const chatForm = document.getElementById("chatForm");
  const chatInput = document.getElementById("chatInput");
  const sendChatBtn = document.getElementById("sendChatBtn");
  const clearChatBtn = document.getElementById("clearChatBtn");

  const generateSummaryBtn = document.getElementById("generateSummaryBtn");
  const exportSummaryBtn = document.getElementById("exportSummaryBtn");
  const summaryOutput = document.getElementById("summaryOutput");

  const studyCount = document.getElementById("studyCount");
  const quizDifficulty = document.getElementById("quizDifficulty");
  const generateQuizBtn = document.getElementById("generateQuizBtn");
  const generateVivaBtn = document.getElementById("generateVivaBtn");
  const exportStudyMdBtn = document.getElementById("exportStudyMdBtn");
  const exportAnkiBtn = document.getElementById("exportAnkiBtn");
  const quizVivaOutput = document.getElementById("quizVivaOutput");

  const metricsTabBtn = document.getElementById("metrics-tab");
  const refreshMetricsBtn = document.getElementById("refreshMetricsBtn");
  const metricsCards = document.getElementById("metricsCards");
  const savedArtifactsList = document.getElementById("savedArtifactsList");
  const recentMetricsTable = document.getElementById("recentMetricsTable");

  const docPreviewTitle = document.getElementById("docPreviewTitle");
  const docPreviewBody = document.getElementById("docPreviewBody");

  // ---------- Utilities ----------

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text ?? "";
    return div.innerHTML;
  }

  function renderMarkdown(markdownText) {
    const raw = markdownText || "";
    if (window.marked && window.DOMPurify) {
      const parsed = window.marked.parse(raw, { breaks: true, gfm: true });
      return window.DOMPurify.sanitize(parsed);
    }
    let html = escapeHtml(raw);
    html = html.replace(/```([\s\S]*?)```/g, "<pre><code>$1</code></pre>");
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    html = html.replace(/^### (.*)$/gm, "<h3>$1</h3>");
    html = html.replace(/^## (.*)$/gm, "<h2>$1</h2>");
    html = html.replace(/^# (.*)$/gm, "<h1>$1</h1>");
    html = html.replace(/\n/g, "<br>");
    return html;
  }

  function renderMath(container) {
    if (container && window.renderMathInElement) {
      try {
        window.renderMathInElement(container, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false },
            { left: "\\(", right: "\\)", display: false },
            { left: "\\[", right: "\\]", display: true },
          ],
          throwOnError: false,
        });
      } catch {
        // Ignore malformed math expressions gracefully
      }
    }
  }

  function showToast(message, variant = "primary") {
    const container = document.getElementById("toastContainer");
    if (!container) return;
    const toastEl = document.createElement("div");
    toastEl.className = `toast align-items-center text-bg-${variant} border-0`;
    toastEl.setAttribute("role", "alert");
    toastEl.innerHTML = `
      <div class="d-flex">
        <div class="toast-body">${escapeHtml(message)}</div>
        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
      </div>
    `;
    container.appendChild(toastEl);
    const toast = new bootstrap.Toast(toastEl, { delay: 3600 });
    toast.show();
    toastEl.addEventListener("hidden.bs.toast", () => toastEl.remove());
  }

  function setButtonLoading(button, isLoading) {
    if (!button) return;
    const spinner = button.querySelector(".spinner-border");
    button.disabled = isLoading;
    if (spinner) {
      spinner.classList.toggle("d-none", !isLoading);
    }
  }

  function formatBytes(bytes) {
    if (!bytes || bytes < 1024) return `${bytes || 0} B`;
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(1)} KB`;
    return `${(kb / 1024).toFixed(2)} MB`;
  }

  function downloadBlob(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function showApiWarning(message) {
    if (apiAlertBanner && apiAlertText) {
      apiAlertText.textContent = message;
      apiAlertBanner.classList.remove("d-none");
      apiAlertBanner.classList.add("d-flex");
    }
  }

  async function apiFetch(url, options = {}) {
    const response = await fetch(url, options);
    let data = {};
    try {
      data = await response.json();
    } catch {
      data = {};
    }
    if (response.status === 401) {
      window.location.href = "/login";
      throw new Error("Session expired. Redirecting to login.");
    }
    if (!response.ok) {
      const errMsg = data.error || `Request failed (${response.status})`;
      if (data.code === "MISSING_API_KEY" || data.code === "INVALID_API_KEY") {
        showApiWarning(errMsg);
      }
      throw new Error(errMsg);
    }
    return data;
  }

  function getSelectedMode() {
    return engineModeSelect ? engineModeSelect.value : "auto";
  }

  // ---------- Document Preview Modal ----------

  async function openDocumentPreview(docId, page = 1) {
    const doc = state.documents.find((d) => d.id === Number(docId));
    const docName = doc ? doc.original_filename : `Document #${docId}`;
    const isPdf = !doc || doc.file_type === "pdf" || docName.toLowerCase().endsWith(".pdf");

    docPreviewTitle.textContent = `${docName} — Page/Section ${page}`;
    const modal = new bootstrap.Modal(document.getElementById("docPreviewModal"));

    if (isPdf) {
      const url = `/api/documents/${docId}/view#page=${page}`;
      docPreviewBody.innerHTML = `<iframe src="${url}" style="width:100%;height:100%;border:0;" title="PDF Preview"></iframe>`;
      modal.show();
      return;
    }

    docPreviewBody.innerHTML = `<div class="p-4 text-center text-muted"><div class="spinner-border text-primary"></div></div>`;
    modal.show();
    try {
      const data = await apiFetch(`/api/documents/${docId}/view`);
      const chunks = data.chunks || [];
      docPreviewBody.innerHTML = `
        <div class="p-3 overflow-auto h-100">
          ${chunks
            .map(
              (c) => `
              <div class="p-3 mb-2 rounded border ${Number(c.page) === Number(page) ? "border-primary bg-primary-subtle" : "bg-light"}">
                <div class="fw-semibold small text-primary mb-1">Page / Slide ${c.page}</div>
                <div class="small" style="white-space: pre-wrap;">${escapeHtml(c.text)}</div>
              </div>`
            )
            .join("")}
        </div>`;
    } catch (err) {
      docPreviewBody.innerHTML = `<div class="p-3 text-danger">${escapeHtml(err.message)}</div>`;
    }
  }

  // ---------- Health & Context ----------

  async function checkHealth() {
    try {
      const data = await apiFetch("/api/health");
      state.apiConfigured = Boolean(data.gemini_configured);
      if (apiKeyStatusBadge) {
        apiKeyStatusBadge.classList.remove("text-bg-secondary", "text-bg-success", "text-bg-warning");
        if (state.apiConfigured) {
          apiKeyStatusBadge.classList.add("text-bg-success");
          apiKeyStatusBadge.innerHTML = `<i class="bi bi-check-circle-fill"></i> Gemini Ready`;
        } else {
          apiKeyStatusBadge.classList.add("text-bg-warning");
          apiKeyStatusBadge.innerHTML = `<i class="bi bi-exclamation-triangle-fill"></i> Local RAG Mode`;
          showApiWarning(
            data.message ||
              "GEMINI_API_KEY is missing. Local Offline RAG still works; add a Gemini key to CampusGPT/.env for cloud LLM answers."
          );
        }
      }
    } catch (err) {
      console.error("Health check error:", err);
    }
  }

  function updateContextLabel() {
    const count = state.selectedDocIds.size;
    if (count === 0) {
      activeContextLabel.textContent =
        "No document selected. Upload or select a file on the left.";
      return;
    }
    const primaryDoc = state.documents.find((d) => d.id === state.primaryDocId);
    const primaryName = primaryDoc ? primaryDoc.original_filename : "Document";
    if (count === 1) {
      activeContextLabel.innerHTML = `Active Document: <strong>${escapeHtml(primaryName)}</strong>`;
    } else {
      activeContextLabel.innerHTML = `Active Scope: <strong>${count} files selected</strong> (Primary: <strong>${escapeHtml(primaryName)}</strong>)`;
    }
  }

  function getPrimaryDocId() {
    if (state.primaryDocId && state.selectedDocIds.has(state.primaryDocId)) {
      return state.primaryDocId;
    }
    const first = state.selectedDocIds.values().next();
    return first.done ? null : first.value;
  }

  // ---------- Document Manager ----------

  function refreshCourseFilterOptions() {
    if (!courseFilterSelect) return;
    const courses = [...new Set(state.documents.map((d) => d.course_name || "General"))].sort();
    const prev = state.courseFilter;
    courseFilterSelect.innerHTML =
      `<option value="ALL">All Courses (${state.documents.length})</option>` +
      courses.map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("");
    if (prev === "ALL" || courses.includes(prev)) {
      courseFilterSelect.value = prev;
    } else {
      state.courseFilter = "ALL";
    }
  }

  function renderDocuments() {
    refreshCourseFilterOptions();
    const visibleDocs =
      state.courseFilter === "ALL"
        ? state.documents
        : state.documents.filter((d) => (d.course_name || "General") === state.courseFilter);

    docCountBadge.textContent = String(state.documents.length);
    if (!visibleDocs.length) {
      documentList.innerHTML = `
        <div class="text-center text-muted py-4 small">
          <i class="bi bi-inbox fs-3 d-block mb-2"></i>
          No files in this view.
        </div>`;
      updateContextLabel();
      return;
    }

    documentList.innerHTML = visibleDocs
      .map((doc) => {
        const checked = state.selectedDocIds.has(doc.id);
        const isPrimary = state.primaryDocId === doc.id;
        const extLabel = (doc.file_type || "pdf").toUpperCase();
        return `
          <div class="doc-item ${isPrimary ? "active" : ""}" data-doc-id="${doc.id}">
            <div class="d-flex align-items-start gap-2">
              <input
                type="checkbox"
                class="form-check-input mt-1 doc-checkbox"
                data-doc-id="${doc.id}"
                ${checked ? "checked" : ""}
                title="Include in Hybrid RAG search"
              />
              <div class="flex-grow-1 doc-select-area" role="button" data-doc-id="${doc.id}">
                <div class="doc-title text-dark">${escapeHtml(doc.original_filename)}</div>
                <div class="d-flex flex-wrap align-items-center gap-1 mt-1">
                  <span class="badge bg-light text-secondary border x-small">${escapeHtml(extLabel)}</span>
                  <span class="badge bg-primary-subtle text-primary x-small">${escapeHtml(doc.course_name || "General")}</span>
                </div>
                <div class="x-small text-muted mt-1">
                  ${doc.page_count} pages · ${doc.chunk_count} chunks · ${formatBytes(doc.file_size)}
                </div>
              </div>
              <div class="d-flex flex-column gap-1">
                <button type="button" class="btn btn-sm btn-outline-secondary border-0 preview-doc-btn" data-doc-id="${doc.id}" title="Preview Document">
                  <i class="bi bi-eye"></i>
                </button>
                <button type="button" class="btn btn-sm btn-outline-danger border-0 delete-doc-btn" data-doc-id="${doc.id}" title="Delete File">
                  <i class="bi bi-trash"></i>
                </button>
              </div>
            </div>
          </div>
        `;
      })
      .join("");

    documentList.querySelectorAll(".doc-checkbox").forEach((cb) => {
      cb.addEventListener("change", (e) => {
        const id = Number(e.target.dataset.docId);
        if (e.target.checked) {
          state.selectedDocIds.add(id);
          if (!state.primaryDocId) state.primaryDocId = id;
        } else {
          state.selectedDocIds.delete(id);
          if (state.primaryDocId === id) {
            state.primaryDocId = getPrimaryDocId();
          }
        }
        renderDocuments();
      });
    });

    documentList.querySelectorAll(".doc-select-area").forEach((el) => {
      el.addEventListener("click", () => {
        const id = Number(el.dataset.docId);
        state.selectedDocIds.add(id);
        state.primaryDocId = id;
        renderDocuments();
        loadStoredSummary(id);
        loadChatHistory();
      });
    });

    documentList.querySelectorAll(".preview-doc-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        openDocumentPreview(Number(btn.dataset.docId), 1);
      });
    });

    documentList.querySelectorAll(".delete-doc-btn").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const id = Number(btn.dataset.docId);
        const doc = state.documents.find((d) => d.id === id);
        if (!confirm(`Delete "${doc?.original_filename || "this file"}" and its vector index?`)) {
          return;
        }
        try {
          await apiFetch(`/api/documents/${id}`, { method: "DELETE" });
          state.selectedDocIds.delete(id);
          if (state.primaryDocId === id) state.primaryDocId = null;
          showToast("Document deleted.", "secondary");
          await loadDocuments();
          await loadChatHistory();
        } catch (err) {
          showToast(err.message, "danger");
        }
      });
    });

    updateContextLabel();
  }

  function loadStoredSummary(docId) {
    const doc = state.documents.find((d) => d.id === docId);
    if (doc && doc.summary) {
      state.currentSummaryMd = doc.summary;
      summaryOutput.innerHTML = `<div class="markdown-body">${renderMarkdown(doc.summary)}</div>`;
      renderMath(summaryOutput);
      exportSummaryBtn.classList.remove("d-none");
    }
  }

  async function loadDocuments() {
    try {
      const data = await apiFetch("/api/documents");
      state.documents = data.documents || [];
      const validIds = new Set(state.documents.map((d) => d.id));
      state.selectedDocIds = new Set(
        [...state.selectedDocIds].filter((id) => validIds.has(id))
      );
      if (!validIds.has(state.primaryDocId)) {
        state.primaryDocId = null;
      }
      if (state.documents.length > 0 && state.selectedDocIds.size === 0) {
        const firstId = state.documents[0].id;
        state.selectedDocIds.add(firstId);
        state.primaryDocId = firstId;
      }
      renderDocuments();
      if (state.primaryDocId) {
        loadStoredSummary(state.primaryDocId);
      }
    } catch (err) {
      showToast(err.message, "danger");
    }
  }

  async function uploadStudyFile(file) {
    if (!file) return;
    const allowed = [".pdf", ".docx", ".pptx", ".txt", ".md"];
    const lower = file.name.toLowerCase();
    if (!allowed.some((ext) => lower.endsWith(ext))) {
      showToast("Supported formats: PDF, DOCX, PPTX, TXT, MD.", "warning");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("course_name", (courseNameInput?.value || "General").trim() || "General");

    uploadSpinner.classList.remove("d-none");
    uploadSpinner.classList.add("d-flex");
    try {
      const data = await apiFetch("/api/documents/upload", {
        method: "POST",
        body: formData,
      });
      const uploadedDoc = data.document;
      showToast(
        `Indexed "${uploadedDoc.original_filename}" (${uploadedDoc.chunk_count} chunks).`,
        "success"
      );
      state.selectedDocIds.add(uploadedDoc.id);
      state.primaryDocId = uploadedDoc.id;
      await loadDocuments();
    } catch (err) {
      showToast(err.message, "danger");
    } finally {
      uploadSpinner.classList.add("d-none");
      uploadSpinner.classList.remove("d-flex");
      pdfFileInput.value = "";
    }
  }

  // ---------- Streaming Academic Chat ----------

  function buildSourcesHtml(sources) {
    if (!sources || !sources.length) return "";
    return `
      <div class="d-flex flex-wrap gap-1 mt-2 pt-2 border-top">
        ${sources
          .map(
            (s) =>
              `<button type="button" class="source-chip border-0" data-doc-id="${s.doc_id}" data-page="${s.page}" title="Click to preview page ${s.page}">
                <i class="bi bi-bookmark"></i>${escapeHtml(s.doc_name || "Doc")} · p.${s.page}
              </button>`
          )
          .join("")}
      </div>`;
  }

  function bindSourceChips(container) {
    container.querySelectorAll(".source-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        openDocumentPreview(Number(chip.dataset.docId), Number(chip.dataset.page || 1));
      });
    });
  }

  function appendChatBubble(role, content, sources = [], meta = null) {
    const emptyEl = chatMessages.querySelector(".chat-empty");
    if (emptyEl) emptyEl.remove();

    const wrapper = document.createElement("div");
    wrapper.className = `chat-bubble ${role}`;

    if (role === "user") {
      wrapper.innerHTML = `<div class="fw-normal">${escapeHtml(content).replace(/\n/g, "<br>")}</div>`;
    } else {
      const metaHtml = meta
        ? `<div class="x-small text-muted mt-1"><i class="bi bi-cpu me-1"></i>${escapeHtml(meta.engine_used)} · ${meta.latency_ms} ms</div>`
        : "";
      wrapper.innerHTML = `<div class="markdown-body">${renderMarkdown(content)}</div>${buildSourcesHtml(sources)}${metaHtml}`;
      renderMath(wrapper);
      bindSourceChips(wrapper);
    }

    chatMessages.appendChild(wrapper);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return wrapper;
  }

  async function loadChatHistory() {
    try {
      const primaryId = getPrimaryDocId();
      const query = primaryId ? `?document_id=${primaryId}` : "";
      const data = await apiFetch(`/api/chat/history${query}`);
      const messages = data.messages || [];

      if (!messages.length) {
        chatMessages.innerHTML = `
          <div class="chat-empty text-center text-muted py-5">
            <i class="bi bi-robot fs-1 text-primary opacity-75 d-block mb-2"></i>
            <h3 class="h6 fw-bold">Ask Anything About Your Course Materials</h3>
            <p class="small mb-0">Upload a PDF, Word, or PowerPoint file, keep it checked on the left, and ask a question below.</p>
          </div>`;
        return;
      }

      chatMessages.innerHTML = "";
      messages.forEach((msg) => {
        let sources = [];
        try {
          sources = JSON.parse(msg.sources_json || "[]");
        } catch {
          sources = [];
        }
        appendChatBubble(msg.role, msg.content, sources);
      });
    } catch (err) {
      console.error("Failed to load chat history:", err);
    }
  }

  async function handleChatSubmit(event) {
    event.preventDefault();
    const question = chatInput.value.trim();
    if (!question) return;

    const docIds = [...state.selectedDocIds];
    if (!docIds.length) {
      showToast("Please upload and select at least one document first.", "warning");
      return;
    }

    appendChatBubble("user", question);
    chatInput.value = "";
    setButtonLoading(sendChatBtn, true);

    const assistantBubble = appendChatBubble("assistant", "_Thinking..._");
    const mdBody = assistantBubble.querySelector(".markdown-body");
    let accumulated = "";
    let latestSources = [];

    try {
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          document_ids: docIds,
          primary_document_id: getPrimaryDocId(),
          mode: getSelectedMode(),
        }),
      });

      if (response.status === 401) {
        window.location.href = "/login";
        return;
      }
      if (!response.ok) {
        const errPayload = await response.json().catch(() => ({}));
        throw new Error(errPayload.error || `Request failed (${response.status})`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const events = buffer.split("\n\n");
        buffer = events.pop() || "";

        for (const rawEvent of events) {
          const line = rawEvent.trim();
          if (!line.startsWith("data:")) continue;
          const jsonStr = line.slice(5).trim();
          if (!jsonStr) continue;
          const evt = JSON.parse(jsonStr);

          if (evt.type === "sources") {
            latestSources = evt.sources || [];
          } else if (evt.type === "chunk") {
            accumulated += evt.text || "";
            mdBody.innerHTML = renderMarkdown(accumulated);
            chatMessages.scrollTop = chatMessages.scrollHeight;
          } else if (evt.type === "error") {
            throw new Error(evt.error || "Streaming error");
          } else if (evt.type === "done") {
            assistantBubble.innerHTML =
              `<div class="markdown-body">${renderMarkdown(accumulated)}</div>` +
              buildSourcesHtml(latestSources) +
              `<div class="x-small text-muted mt-1"><i class="bi bi-cpu me-1"></i>${escapeHtml(evt.engine_used)} · ${evt.latency_ms} ms</div>`;
            renderMath(assistantBubble);
            bindSourceChips(assistantBubble);
          }
        }
      }
    } catch (err) {
      mdBody.innerHTML = renderMarkdown(`**Error:** ${err.message}`);
      showToast(err.message, "danger");
    } finally {
      setButtonLoading(sendChatBtn, false);
    }
  }

  // ---------- Summary / Interactive Quiz / Viva ----------

  async function handleGenerateSummary() {
    const docId = getPrimaryDocId();
    if (!docId) {
      showToast("Select a document first.", "warning");
      return;
    }
    setButtonLoading(generateSummaryBtn, true);
    summaryOutput.innerHTML = `
      <div class="text-center py-5 text-muted">
        <div class="spinner-border text-primary mb-2" role="status"></div>
        <div class="small">Generating structured academic summary...</div>
      </div>`;

    try {
      const data = await apiFetch("/api/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_id: docId, mode: getSelectedMode() }),
      });
      state.currentSummaryMd = data.summary;
      summaryOutput.innerHTML = `
        <div class="x-small text-muted mb-2"><i class="bi bi-cpu me-1"></i>${escapeHtml(data.engine_used)} · ${data.latency_ms} ms</div>
        <div class="markdown-body">${renderMarkdown(data.summary)}</div>`;
      renderMath(summaryOutput);
      exportSummaryBtn.classList.remove("d-none");
      const doc = state.documents.find((d) => d.id === docId);
      if (doc) doc.summary = data.summary;
    } catch (err) {
      summaryOutput.innerHTML = `<div class="alert alert-danger mb-0">${escapeHtml(err.message)}</div>`;
    } finally {
      setButtonLoading(generateSummaryBtn, false);
    }
  }

  function renderInteractiveQuiz(questions, artifactId, docId, metaLabel = "") {
    state.currentQuizQuestions = questions;
    state.currentQuizArtifactId = artifactId;
    const letters = ["A", "B", "C", "D"];

    quizVivaOutput.innerHTML = `
      ${metaLabel ? `<div class="x-small text-muted mb-2"><i class="bi bi-cpu me-1"></i>${escapeHtml(metaLabel)}</div>` : ""}
      <form id="interactiveQuizForm">
        ${questions
          .map(
            (q, qIdx) => `
            <div class="quiz-question-card" data-q-idx="${qIdx}">
              <div class="d-flex justify-content-between align-items-start gap-2 mb-2">
                <div class="fw-semibold small">${qIdx + 1}. ${escapeHtml(q.question)}</div>
                <button type="button" class="source-chip border-0 flex-shrink-0" data-doc-id="${docId}" data-page="${q.page || 1}">
                  p.${q.page || 1}
                </button>
              </div>
              <div class="quiz-options">
                ${q.options
                  .map(
                    (opt, oIdx) => `
                    <label class="quiz-option-label small" data-q-idx="${qIdx}" data-o-idx="${oIdx}">
                      <input type="radio" name="quiz_q_${qIdx}" value="${oIdx}" class="form-check-input mt-0" />
                      <span><strong>${letters[oIdx]})</strong> ${escapeHtml(opt)}</span>
                    </label>`
                  )
                  .join("")}
              </div>
              <div class="quiz-feedback small mt-2 d-none"></div>
            </div>`
          )
          .join("")}
        <div class="d-flex align-items-center justify-content-between gap-2 mt-3">
          <button type="submit" class="btn btn-success btn-sm px-3">
            <i class="bi bi-check2-circle me-1"></i>Submit &amp; Grade Quiz
          </button>
          <div id="quizScoreBanner" class="fw-bold small"></div>
        </div>
      </form>`;

    renderMath(quizVivaOutput);
    bindSourceChips(quizVivaOutput);

    const quizForm = document.getElementById("interactiveQuizForm");
    quizForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      let score = 0;
      questions.forEach((q, qIdx) => {
        const selectedInput = quizForm.querySelector(`input[name="quiz_q_${qIdx}"]:checked`);
        const chosen = selectedInput ? Number(selectedInput.value) : -1;
        const card = quizForm.querySelector(`.quiz-question-card[data-q-idx="${qIdx}"]`);
        const feedbackEl = card.querySelector(".quiz-feedback");
        const labels = card.querySelectorAll(".quiz-option-label");

        labels.forEach((lbl) => {
          lbl.classList.remove("correct", "wrong");
          const oIdx = Number(lbl.dataset.oIdx);
          if (oIdx === q.correct_index) {
            lbl.classList.add("correct");
          } else if (oIdx === chosen && chosen !== q.correct_index) {
            lbl.classList.add("wrong");
          }
        });

        const isCorrect = chosen === q.correct_index;
        if (isCorrect) score += 1;
        feedbackEl.classList.remove("d-none");
        feedbackEl.innerHTML = `<span class="fw-semibold ${isCorrect ? "text-success" : "text-danger"}">${isCorrect ? "Correct!" : `Correct answer: ${letters[q.correct_index]})`}</span> ${escapeHtml(q.explanation || "")}`;
      });

      const banner = document.getElementById("quizScoreBanner");
      const pct = Math.round((score / Math.max(1, questions.length)) * 100);
      banner.innerHTML = `<span class="badge text-bg-${pct >= 70 ? "success" : "warning"} fs-6">Score: ${score}/${questions.length} (${pct}%)</span>`;

      if (artifactId) {
        try {
          await apiFetch(`/api/quiz/${artifactId}/score`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ score, total_questions: questions.length }),
          });
          showToast(`Saved score: ${score}/${questions.length}`, "success");
        } catch {
          // Non-fatal
        }
      }
    });
  }

  async function handleGenerateStudyMaterial(mode) {
    const docId = getPrimaryDocId();
    if (!docId) {
      showToast("Select a document first.", "warning");
      return;
    }

    const isQuiz = mode === "quiz";
    const btn = isQuiz ? generateQuizBtn : generateVivaBtn;
    const endpoint = isQuiz ? "/api/quiz" : "/api/viva";
    const payload = {
      document_id: docId,
      num_questions: Number(studyCount.value || 5),
      difficulty: quizDifficulty.value || "medium",
      mode: getSelectedMode(),
    };

    setButtonLoading(btn, true);
    quizVivaOutput.innerHTML = `
      <div class="text-center py-5 text-muted">
        <div class="spinner-border text-primary mb-2" role="status"></div>
        <div class="small">Generating ${isQuiz ? "interactive MCQ quiz" : "Viva Q&A"}...</div>
      </div>`;

    try {
      const data = await apiFetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const mdContent = isQuiz ? data.quiz : data.viva;
      state.currentStudyMd = mdContent || "";
      exportStudyMdBtn.classList.toggle("d-none", !state.currentStudyMd);

      if (isQuiz && Array.isArray(data.questions) && data.questions.length > 0) {
        exportAnkiBtn.classList.remove("d-none");
        renderInteractiveQuiz(
          data.questions,
          data.artifact_id,
          docId,
          `${data.engine_used} · ${data.latency_ms} ms`
        );
      } else {
        exportAnkiBtn.classList.add("d-none");
        state.currentQuizQuestions = [];
        quizVivaOutput.innerHTML = `
          <div class="x-small text-muted mb-2"><i class="bi bi-cpu me-1"></i>${escapeHtml(data.engine_used)} · ${data.latency_ms} ms</div>
          <div class="markdown-body">${renderMarkdown(mdContent)}</div>`;
        renderMath(quizVivaOutput);
      }
    } catch (err) {
      quizVivaOutput.innerHTML = `<div class="alert alert-danger mb-0">${escapeHtml(err.message)}</div>`;
    } finally {
      setButtonLoading(btn, false);
    }
  }

  // ---------- History & Metrics ----------

  async function loadMetricsAndArtifacts() {
    try {
      const [metrics, artifactsData] = await Promise.all([
        apiFetch("/api/metrics"),
        apiFetch("/api/artifacts"),
      ]);

      metricsCards.innerHTML = `
        <div class="col-6 col-md-3">
          <div class="metric-kpi-card">
            <div class="x-small text-muted">Total RAG Queries</div>
            <div class="h4 fw-bold mb-0">${metrics.total_queries}</div>
            <div class="x-small text-secondary">${metrics.cloud_queries} Cloud · ${metrics.local_queries} Local</div>
          </div>
        </div>
        <div class="col-6 col-md-3">
          <div class="metric-kpi-card">
            <div class="x-small text-muted">Avg Response Latency</div>
            <div class="h4 fw-bold mb-0">${metrics.avg_latency_ms} ms</div>
            <div class="x-small text-secondary">End-to-end RAG + LLM</div>
          </div>
        </div>
        <div class="col-6 col-md-3">
          <div class="metric-kpi-card">
            <div class="x-small text-muted">Avg Hybrid Top Score</div>
            <div class="h4 fw-bold mb-0">${metrics.avg_top_score}</div>
            <div class="x-small text-secondary">FAISS + BM25 RRF</div>
          </div>
        </div>
        <div class="col-6 col-md-3">
          <div class="metric-kpi-card">
            <div class="x-small text-muted">Quiz Performance</div>
            <div class="h4 fw-bold mb-0">${metrics.avg_quiz_pct}%</div>
            <div class="x-small text-secondary">${metrics.quiz_attempts} Graded Attempts</div>
          </div>
        </div>`;

      const artifacts = artifactsData.artifacts || [];
      if (!artifacts.length) {
        savedArtifactsList.innerHTML = `<div class="text-muted small text-center py-4">No saved quizzes, summaries, or vivas yet.</div>`;
      } else {
        savedArtifactsList.innerHTML = artifacts
          .map(
            (a) => `
            <div class="border rounded p-2 mb-2 bg-light d-flex justify-content-between align-items-center gap-2">
              <div class="overflow-hidden">
                <div class="fw-semibold small text-truncate">${escapeHtml(a.title)}</div>
                <div class="x-small text-muted">
                  <span class="badge text-bg-secondary">${escapeHtml(a.artifact_type.toUpperCase())}</span>
                  ${a.score !== null && a.score !== undefined ? `· Score ${a.score}/${a.total_questions}` : ""}
                  · ${escapeHtml(a.engine_used)}
                </div>
              </div>
              <div class="d-flex gap-1 flex-shrink-0">
                <button type="button" class="btn btn-sm btn-outline-primary open-artifact-btn" data-artifact-id="${a.id}">Open</button>
                <button type="button" class="btn btn-sm btn-outline-danger delete-artifact-btn" data-artifact-id="${a.id}"><i class="bi bi-trash"></i></button>
              </div>
            </div>`
          )
          .join("");

        savedArtifactsList.querySelectorAll(".open-artifact-btn").forEach((btn) => {
          btn.addEventListener("click", () => {
            const art = artifacts.find((x) => x.id === Number(btn.dataset.artifactId));
            if (!art) return;
            if (art.artifact_type === "summary") {
              state.currentSummaryMd = art.raw_markdown;
              summaryOutput.innerHTML = `<div class="markdown-body">${renderMarkdown(art.raw_markdown)}</div>`;
              renderMath(summaryOutput);
              exportSummaryBtn.classList.remove("d-none");
              bootstrap.Tab.getOrCreateInstance(document.getElementById("summary-tab")).show();
            } else {
              state.currentStudyMd = art.raw_markdown;
              exportStudyMdBtn.classList.remove("d-none");
              let parsedQuestions = [];
              try {
                parsedQuestions = JSON.parse(art.content_json || "[]");
              } catch {
                parsedQuestions = [];
              }
              if (art.artifact_type === "quiz" && parsedQuestions.length) {
                exportAnkiBtn.classList.remove("d-none");
                renderInteractiveQuiz(parsedQuestions, art.id, art.document_id, art.engine_used);
              } else {
                exportAnkiBtn.classList.add("d-none");
                quizVivaOutput.innerHTML = `<div class="markdown-body">${renderMarkdown(art.raw_markdown)}</div>`;
                renderMath(quizVivaOutput);
              }
              bootstrap.Tab.getOrCreateInstance(document.getElementById("quiz-tab")).show();
            }
          });
        });

        savedArtifactsList.querySelectorAll(".delete-artifact-btn").forEach((btn) => {
          btn.addEventListener("click", async () => {
            await apiFetch(`/api/artifacts/${Number(btn.dataset.artifactId)}`, { method: "DELETE" });
            loadMetricsAndArtifacts();
          });
        });
      }

      const recent = metrics.recent_queries || [];
      if (!recent.length) {
        recentMetricsTable.innerHTML = `<div class="text-muted small text-center py-4">No queries logged yet.</div>`;
      } else {
        recentMetricsTable.innerHTML = `
          <table class="table table-sm table-hover x-small mb-0">
            <thead><tr><th>Endpoint</th><th>Engine</th><th>Latency</th><th>Top Score</th></tr></thead>
            <tbody>
              ${recent
                .map(
                  (r) => `
                  <tr>
                    <td>${escapeHtml(r.endpoint)}</td>
                    <td><code>${escapeHtml(r.engine_used)}</code></td>
                    <td>${r.latency_ms} ms</td>
                    <td>${Number(r.top_score).toFixed(3)}</td>
                  </tr>`
                )
                .join("")}
            </tbody>
          </table>`;
      }
    } catch (err) {
      showToast(err.message, "danger");
    }
  }

  // ---------- Event Wiring ----------

  if (courseFilterSelect) {
    courseFilterSelect.addEventListener("change", (e) => {
      state.courseFilter = e.target.value;
      renderDocuments();
    });
  }

  if (browsePdfBtn && pdfFileInput) {
    browsePdfBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      pdfFileInput.click();
    });
    uploadDropzone.addEventListener("click", () => pdfFileInput.click());
    pdfFileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        uploadStudyFile(e.target.files[0]);
      }
    });
    uploadDropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      uploadDropzone.classList.add("drag-over");
    });
    uploadDropzone.addEventListener("dragleave", () => {
      uploadDropzone.classList.remove("drag-over");
    });
    uploadDropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      uploadDropzone.classList.remove("drag-over");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        uploadStudyFile(e.dataTransfer.files[0]);
      }
    });
  }

  if (selectAllDocsBtn) {
    selectAllDocsBtn.addEventListener("click", () => {
      if (state.selectedDocIds.size === state.documents.length) {
        state.selectedDocIds.clear();
        state.primaryDocId = null;
        selectAllDocsBtn.textContent = "Select All";
      } else {
        state.documents.forEach((d) => state.selectedDocIds.add(d.id));
        if (!state.primaryDocId && state.documents.length) {
          state.primaryDocId = state.documents[0].id;
        }
        selectAllDocsBtn.textContent = "Deselect All";
      }
      renderDocuments();
    });
  }

  if (chatForm) {
    chatForm.addEventListener("submit", handleChatSubmit);
    chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event("submit", { cancelable: true }));
      }
    });
  }

  if (clearChatBtn) {
    clearChatBtn.addEventListener("click", async () => {
      try {
        const primaryId = getPrimaryDocId();
        const query = primaryId ? `?document_id=${primaryId}` : "";
        await apiFetch(`/api/chat/history${query}`, { method: "DELETE" });
        await loadChatHistory();
        showToast("Chat history cleared.", "secondary");
      } catch (err) {
        showToast(err.message, "danger");
      }
    });
  }

  if (generateSummaryBtn) generateSummaryBtn.addEventListener("click", handleGenerateSummary);
  if (generateQuizBtn) generateQuizBtn.addEventListener("click", () => handleGenerateStudyMaterial("quiz"));
  if (generateVivaBtn) generateVivaBtn.addEventListener("click", () => handleGenerateStudyMaterial("viva"));

  if (exportSummaryBtn) {
    exportSummaryBtn.addEventListener("click", () => {
      if (!state.currentSummaryMd) return;
      downloadBlob("campusgpt_summary.md", state.currentSummaryMd, "text/markdown;charset=utf-8");
    });
  }

  if (exportStudyMdBtn) {
    exportStudyMdBtn.addEventListener("click", () => {
      if (!state.currentStudyMd) return;
      downloadBlob("campusgpt_study_prep.md", state.currentStudyMd, "text/markdown;charset=utf-8");
    });
  }

  if (exportAnkiBtn) {
    exportAnkiBtn.addEventListener("click", () => {
      if (!state.currentQuizQuestions.length) return;
      const letters = ["A", "B", "C", "D"];
      const csvRows = state.currentQuizQuestions.map((q) => {
        const front = `"${(q.question + " | " + q.options.map((o, i) => `${letters[i]}) ${o}`).join(" ")).replace(/"/g, '""')}"`;
        const back = `"${(`${letters[q.correct_index]}) ${q.options[q.correct_index]} — ${q.explanation || ""} (Page ${q.page || 1})`).replace(/"/g, '""')}"`;
        return `${front},${back}`;
      });
      downloadBlob("campusgpt_anki_flashcards.csv", csvRows.join("\n"), "text/csv;charset=utf-8");
    });
  }

  if (metricsTabBtn) metricsTabBtn.addEventListener("shown.bs.tab", loadMetricsAndArtifacts);
  if (refreshMetricsBtn) refreshMetricsBtn.addEventListener("click", loadMetricsAndArtifacts);

  checkHealth();
  loadDocuments().then(loadChatHistory);
})();
