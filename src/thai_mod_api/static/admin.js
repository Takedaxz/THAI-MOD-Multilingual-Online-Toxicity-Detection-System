const adminHealthBadge = document.getElementById("adminHealthBadge");
const adminUserBadge = document.getElementById("adminUserBadge");
const logoutButton = document.getElementById("logoutButton");
const adminModelName = document.getElementById("adminModelName");
const adminDeploymentMode = document.getElementById("adminDeploymentMode");
const adminCacheStatus = document.getElementById("adminCacheStatus");
const adminStatus = document.getElementById("adminStatus");
const adminThreshold = document.getElementById("adminThreshold");
const adminLoaded = document.getElementById("adminLoaded");
const adminDatasetRows = document.getElementById("adminDatasetRows");
const adminMetricsList = document.getElementById("adminMetricsList");
const monitoringStatus = document.getElementById("monitoringStatus");
const monitoringPsi = document.getElementById("monitoringPsi");
const monitoringPrimaryFeature = document.getElementById("monitoringPrimaryFeature");
const monitoringRecentCount = document.getElementById("monitoringRecentCount");
const monitoringWindowSize = document.getElementById("monitoringWindowSize");
const monitoringReferenceInfo = document.getElementById("monitoringReferenceInfo");
const monitoringDetailsGrid = document.getElementById("monitoringDetailsGrid");
const monitoringGeneratedAt = document.getElementById("monitoringGeneratedAt");
const monitoringLogPath = document.getElementById("monitoringLogPath");
const monitoringLogCount = document.getElementById("monitoringLogCount");
const monitoringEventsTable = document.getElementById("monitoringEventsTable");
const refreshMonitoringButton = document.getElementById("refreshMonitoringButton");
const clearMonitoringLogButton = document.getElementById("clearMonitoringLogButton");
const trainLrCandidateButton = document.getElementById("trainLrCandidateButton");
const promoteLrCandidateButton = document.getElementById("promoteLrCandidateButton");
const trainCandidateButton = document.getElementById("trainCandidateButton");
const promoteCandidateButton = document.getElementById("promoteCandidateButton");
const refreshModelUpdateButton = document.getElementById("refreshModelUpdateButton");
const modelUpdateStatus = document.getElementById("modelUpdateStatus");
const modelUpdateKind = document.getElementById("modelUpdateKind");
const modelUpdateStarted = document.getElementById("modelUpdateStarted");
const modelUpdateFinished = document.getElementById("modelUpdateFinished");
const modelUpdateReturnCode = document.getElementById("modelUpdateReturnCode");
const modelUpdateLogPath = document.getElementById("modelUpdateLogPath");
const modelUpdateMessage = document.getElementById("modelUpdateMessage");
const reviewedExamplesCount = document.getElementById("reviewedExamplesCount");
const reviewedExamplesPath = document.getElementById("reviewedExamplesPath");
const confusionCells = {
  tn: document.querySelector("#cm-tn .confusion-cell-value"),
  fp: document.querySelector("#cm-fp .confusion-cell-value"),
  fn: document.querySelector("#cm-fn .confusion-cell-value"),
  tp: document.querySelector("#cm-tp .confusion-cell-value"),
};
const confusionCellBoxes = {
  tn: document.getElementById("cm-tn"),
  fp: document.getElementById("cm-fp"),
  fn: document.getElementById("cm-fn"),
  tp: document.getElementById("cm-tp"),
};

function formatMetricValue(key, value) {
  if (typeof value !== "number") {
    return String(value);
  }

  if (key === "test_size") {
    return value.toLocaleString();
  }

  return `${(value * 100).toFixed(2)}%`;
}

function renderMetrics(metrics) {
  adminMetricsList.innerHTML = "";

  Object.entries(metrics).forEach(([key, value]) => {
    if (key === "confusion_matrix") {
      return;
    }

    const item = document.createElement("div");
    item.className = "system-item";
    item.innerHTML = `
      <span>${key.replaceAll("_", " ")}</span>
      <strong>${formatMetricValue(key, value)}</strong>
    `;
    adminMetricsList.appendChild(item);
  });
}

function renderConfusionMatrix(matrix) {
  if (!Array.isArray(matrix) || matrix.length !== 2 || matrix.some((row) => !Array.isArray(row) || row.length !== 2)) {
    Object.values(confusionCells).forEach((node) => {
      node.textContent = "-";
    });
    Object.values(confusionCellBoxes).forEach((node) => {
      node.style.opacity = "0.7";
    });
    return;
  }

  const values = {
    tn: Number(matrix[0][0]) || 0,
    fp: Number(matrix[0][1]) || 0,
    fn: Number(matrix[1][0]) || 0,
    tp: Number(matrix[1][1]) || 0,
  };
  const maxValue = Math.max(...Object.values(values), 1);

  Object.entries(values).forEach(([key, value]) => {
    confusionCells[key].textContent = value.toLocaleString();

    const intensity = value / maxValue;
    const alpha = 0.16 + intensity * 0.56;
    const isCorrect = key === "tn" || key === "tp";
    const color = isCorrect ? `rgba(83, 214, 160, ${alpha.toFixed(3)})` : `rgba(255, 123, 134, ${alpha.toFixed(3)})`;
    confusionCellBoxes[key].style.background = `linear-gradient(180deg, ${color}, rgba(12, 19, 36, 0.92))`;
    confusionCellBoxes[key].style.borderColor = isCorrect ? "rgba(83, 214, 160, 0.35)" : "rgba(255, 123, 134, 0.35)";
    confusionCellBoxes[key].style.opacity = "1";
  });
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function formatNumber(value, decimals = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(decimals);
}

function formatDateTime(value) {
  if (!value) {
    return "-";
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return String(value);
  }

  return parsed.toLocaleString();
}

function formatLanguageMix(languageMix) {
  if (!languageMix || typeof languageMix !== "object") {
    return "Unavailable";
  }

  const order = [
    ["thai_only", "Thai"],
    ["english_only", "English"],
    ["mixed_script", "Mixed"],
    ["other", "Other"],
  ];

  return order
    .filter(([key]) => languageMix[key] !== undefined)
    .map(([key, label]) => `${label} ${formatPercent(languageMix[key])}`)
    .join(" | ");
}

function formatMonitoringState(drift, windowInfo) {
  const requestCount = Number(windowInfo?.prediction_count || 0);

  if (!drift || drift.confidence === "collecting_data" || drift.status === "collecting_data") {
    return requestCount === 0 ? "COLLECTING DATA" : "COLLECTING DATA";
  }

  const baseStatus = String(drift.status || "-").replaceAll("_", " ").toUpperCase();
  if (drift.confidence === "provisional") {
    return `${baseStatus} (PROVISIONAL)`;
  }

  return baseStatus;
}

function buildMonitoringFootnote(report) {
  const drift = report?.drift_analysis;
  const requestCount = Number(report?.monitoring_window?.prediction_count || 0);
  const minRequired = Number(report?.monitoring_window?.min_required || 20);
  const fullConfidence = Number(report?.monitoring_window?.full_confidence_required || 50);

  if (!drift || drift.confidence === "collecting_data" || drift.status === "collecting_data") {
    if (requestCount === 0) {
      return "No recent requests logged yet.";
    }

    return `Collecting data: ${requestCount}/${minRequired} requests before drift is shown.`;
  }

  if (drift.confidence === "provisional") {
    return `Provisional signal: recent window is below ${fullConfidence} requests.`;
  }

  return "Manual refresh against the latest logged requests.";
}

function setMonitoringTone(status) {
  monitoringStatus.dataset.status = status || "";
  monitoringPsi.dataset.status = status || "";
}

function setMonitoringButtonsDisabled(disabled) {
  refreshMonitoringButton.disabled = disabled;
  clearMonitoringLogButton.disabled = disabled;
}

function setModelUpdateButtonsDisabled(disabled) {
  trainLrCandidateButton.disabled = disabled;
  promoteLrCandidateButton.disabled = disabled;
  trainCandidateButton.disabled = disabled;
  promoteCandidateButton.disabled = disabled;
  refreshModelUpdateButton.disabled = disabled;
}

function renderMonitoringSummary(report) {
  const windowInfo = report?.monitoring_window || {};
  const reference = report.reference_profile;
  const recent = report.recent_live_requests;
  const drift = report.drift_analysis;
  monitoringStatus.textContent = formatMonitoringState(drift, windowInfo);
  monitoringPsi.textContent = drift?.psi === null || drift?.psi === undefined ? "-" : formatNumber(drift.psi, 4);
  monitoringPrimaryFeature.textContent = "Language mix drift";
  monitoringRecentCount.textContent = Number(windowInfo.prediction_count || 0).toLocaleString();
  monitoringWindowSize.textContent = windowInfo.capacity ? `${windowInfo.capacity} latest requests` : "-";
  monitoringReferenceInfo.textContent = reference?.profile_name && reference?.sample_count
    ? `${reference.profile_name} (${Number(reference.sample_count).toLocaleString()})`
    : "-";
  monitoringGeneratedAt.textContent = report?.generated_at
    ? `Refreshed ${formatDateTime(report.generated_at)}. ${buildMonitoringFootnote(report)}`
    : buildMonitoringFootnote(report);
  setMonitoringTone(drift?.status === "collecting_data" ? "collecting_data" : drift?.status || "");

  if (!recent) {
    monitoringDetailsGrid.innerHTML = `
      <div class="system-item">
        <span>Recent traffic</span>
        <strong>${report?.message || "No recent requests logged yet."}</strong>
      </div>
      <div class="system-item">
        <span>Reference language mix</span>
        <strong>${formatLanguageMix(reference?.language_mix)}</strong>
      </div>
    `;
    return;
  }

  const referenceMix = reference.language_mix || {};
  const recentMix = recent.language_mix || {};
  const mixOrder = [
    ["thai_only", "Thai only"],
    ["english_only", "English only"],
    ["mixed_script", "Mixed script"],
    ["other", "Other"],
  ];

  const mixCards = mixOrder
    .map(
      ([key, label]) => `
        <div class="system-item">
          <span>${label}</span>
          <strong>${formatPercent(referenceMix[key])} -> ${formatPercent(recentMix[key])}</strong>
        </div>
      `
    )
    .join("");

  monitoringDetailsGrid.innerHTML = `
    <div class="system-item">
      <span>Toxic ratio</span>
      <strong>${formatPercent(reference.toxic_ratio)} -> ${formatPercent(recent.toxic_ratio)}</strong>
    </div>
    <div class="system-item">
      <span>Average toxicity score</span>
      <strong>${formatNumber(reference.average_toxicity_score, 4)} -> ${formatNumber(recent.average_toxicity_score, 4)}</strong>
    </div>
    <div class="system-item">
      <span>Average text length</span>
      <strong>${formatNumber(reference.average_text_length)} -> ${formatNumber(recent.average_text_length)} chars</strong>
    </div>
    ${mixCards}
  `;
}

function formatEventLabel(label) {
  return String(label || "-").replaceAll("_", " ");
}

function renderMonitoringEvents(payload) {
  const events = Array.isArray(payload?.events) ? payload.events : [];
  monitoringLogPath.textContent = payload?.log_path || "models/monitoring_recent_requests.jsonl";
  monitoringLogCount.textContent = `${Number(payload?.total_logged_requests || 0).toLocaleString()} total`;

  if (events.length === 0) {
    monitoringEventsTable.innerHTML = `
      <tr>
        <td colspan="5">No recent monitoring events yet.</td>
      </tr>
    `;
    return;
  }

  monitoringEventsTable.innerHTML = events
    .map(
      (event) => `
        <tr>
          <td>${formatDateTime(event.timestamp)}</td>
          <td>${formatEventLabel(event.language_bucket)}</td>
          <td>${formatEventLabel(event.predicted_label)}</td>
          <td>${formatNumber(event.toxicity_score, 4)}</td>
          <td>${Number(event.text_length || 0).toLocaleString()}</td>
        </tr>
      `
    )
    .join("");
}

async function loadAdminData() {
  try {
    const response = await fetch("/api/admin/overview");
    if (!response.ok) {
      if (response.status === 401) {
        window.location.assign("/login?next=/admin");
        return;
      }
      throw new Error("Unable to load admin data");
    }

    const payload = await response.json();
    const health = payload.health;
    const info = payload.model_info;

    adminHealthBadge.textContent = "API online";
    adminHealthBadge.className = "pill pill-highlight";
    adminModelName.textContent = info.model_name;
    adminDeploymentMode.textContent = info.deployment_mode;
    adminCacheStatus.textContent = info.cache_status;
    adminStatus.textContent = health.status.toUpperCase();
    adminThreshold.textContent = Number(info.default_threshold).toFixed(2);
    adminLoaded.textContent = health.model_loaded ? "YES" : "NO";
    adminDatasetRows.textContent = Number(info.dataset_rows).toLocaleString();
    renderMetrics(info.metrics);
    renderConfusionMatrix(info.metrics.confusion_matrix);
  } catch (error) {
    adminHealthBadge.textContent = "API offline";
    adminHealthBadge.className = "pill";
    adminStatus.textContent = "ERROR";
    adminModelName.textContent = "Unavailable";
    adminDeploymentMode.textContent = "Unavailable";
    adminCacheStatus.textContent = "Unavailable";
    adminThreshold.textContent = "-";
    adminLoaded.textContent = "-";
    adminDatasetRows.textContent = "-";
    renderConfusionMatrix(null);
    adminMetricsList.innerHTML = `
      <div class="system-item">
        <span>Error</span>
        <strong>${String(error)}</strong>
      </div>
    `;
  }
}

async function loadMonitoringData() {
  try {
    const [summaryResponse, eventsResponse] = await Promise.all([
      fetch("/api/monitoring"),
      fetch("/api/monitoring/events?limit=10"),
    ]);

    if (!summaryResponse.ok || !eventsResponse.ok) {
      if (summaryResponse.status === 401 || eventsResponse.status === 401) {
        window.location.assign("/login?next=/admin");
        return;
      }
      throw new Error("Unable to load monitoring data");
    }

    const [report, events] = await Promise.all([
      summaryResponse.json(),
      eventsResponse.json(),
    ]);
    renderMonitoringSummary(report);
    renderMonitoringEvents(events);
  } catch (error) {
    renderMonitoringSummary({
      message: String(error),
      reference_profile: null,
      recent_live_requests: null,
      monitoring_window: { prediction_count: 0, capacity: null, min_required: 20, full_confidence_required: 50 },
      drift_analysis: { status: "collecting_data", confidence: "collecting_data", psi: null },
    });
    renderMonitoringEvents({ events: [], total_logged_requests: 0 });
  }
}

async function refreshMonitoring() {
  try {
    setMonitoringButtonsDisabled(true);
    monitoringGeneratedAt.textContent = "Refreshing monitoring...";
    await loadMonitoringData();
  } finally {
    setMonitoringButtonsDisabled(false);
  }
}

async function clearMonitoringLog() {
  const confirmed = window.confirm(
    "Clear recent monitoring logs? This resets the admin drift demo window, but it does not delete datasets or model artifacts."
  );

  if (!confirmed) {
    return;
  }

  try {
    setMonitoringButtonsDisabled(true);
    monitoringGeneratedAt.textContent = "Clearing recent log...";
    const response = await fetch("/api/monitoring/reset", { method: "POST" });

    if (!response.ok) {
      if (response.status === 401) {
        window.location.assign("/login?next=/admin");
        return;
      }
      throw new Error("Unable to clear recent log");
    }

    await loadMonitoringData();
  } catch (error) {
    monitoringGeneratedAt.textContent = String(error);
  } finally {
    setMonitoringButtonsDisabled(false);
  }
}

refreshMonitoringButton.addEventListener("click", refreshMonitoring);
clearMonitoringLogButton.addEventListener("click", clearMonitoringLog);

function renderModelUpdateStatus(job) {
  const status = job?.status || "idle";
  modelUpdateStatus.textContent = status.toUpperCase();
  modelUpdateStatus.dataset.status = status;
  modelUpdateKind.textContent = job?.kind ? String(job.kind).replaceAll("-", " ") : "-";
  modelUpdateStarted.textContent = formatDateTime(job?.started_at);
  modelUpdateFinished.textContent = formatDateTime(job?.finished_at);
  modelUpdateReturnCode.textContent = job?.returncode === null || job?.returncode === undefined
    ? "-"
    : String(job.returncode);
  modelUpdateLogPath.textContent = job?.log_path || "-";

  if (status === "running") {
    modelUpdateMessage.textContent = "Script is running in the background. Refresh status to check progress.";
    setModelUpdateButtonsDisabled(true);
    refreshModelUpdateButton.disabled = false;
  } else {
    setModelUpdateButtonsDisabled(false);
    if (status === "completed") {
      modelUpdateMessage.textContent = "Script completed successfully. Restart the app after promotion to load a promoted model.";
      modelUpdateMessage.style.color = "var(--success, #53d6a0)";
    } else if (status === "failed") {
      if (job?.promotion_rejected) {
        // Safety guard blocked — show friendly message with metric details
        const details = job.promotion_details || {};
        const checks = details.checks || {};
        let checkLines = "";
        for (const [metric, result] of Object.entries(checks)) {
          const label = metric.replace(/_/g, " ");
          const curr = result.current !== undefined ? (result.current * 100).toFixed(2) + "%" : "?";
          const cand = result.candidate !== undefined ? (result.candidate * 100).toFixed(2) + "%" : "?";
          const icon = result.passes ? "✅" : "❌";
          checkLines += `  ${icon} ${label}: current ${curr} → candidate ${cand}\n`;
        }
        modelUpdateMessage.style.whiteSpace = "pre-wrap";
        modelUpdateMessage.style.color = "var(--warning, #ffaa00)";
        modelUpdateMessage.textContent =
          `⚠️ Promotion blocked — candidate did not improve on all safety metrics.\n` +
          (details.reason ? details.reason + "\n\n" : "") +
          checkLines +
          `\nYou can still force-promote via the API if you judge this acceptable.`;
      } else {
        modelUpdateMessage.style.color = "var(--error, #ff7b86)";
        modelUpdateMessage.textContent = "Script failed. Check the log file shown above.";
      }
    } else {
      modelUpdateMessage.style.color = "";
      modelUpdateMessage.textContent = "Retraining is manual because new labels require moderator review.";
    }
  }

  const candBox = document.getElementById("candidateMetricsContainer");
  const candLrF1 = document.getElementById("candLrF1");
  const candLrRecall = document.getElementById("candLrRecall");
  const candBertF1 = document.getElementById("candBertF1");
  const candBertRecall = document.getElementById("candBertRecall");
  const candidates = job?.candidates || {};

  if (candBox) {
    if (Object.keys(candidates).length > 0) {
      candBox.style.display = "flex";
      
      if (candidates.lr && candidates.lr.metrics) {
        candLrF1.textContent = formatPercent(candidates.lr.metrics.f2_score);
        candLrRecall.textContent = formatPercent(candidates.lr.metrics.recall);
        candLrF1.style.color = "var(--text-primary)";
        candLrRecall.style.color = "var(--text-primary)";
      } else {
        candLrF1.textContent = "Not trained";
        candLrRecall.textContent = "Not trained";
        candLrF1.style.color = "var(--text-muted)";
        candLrRecall.style.color = "var(--text-muted)";
      }
      
      if (candidates.bert && candidates.bert.metrics) {
        candBertF1.textContent = formatPercent(candidates.bert.metrics.f2_score);
        candBertRecall.textContent = formatPercent(candidates.bert.metrics.recall);
        candBertF1.style.color = "var(--text-primary)";
        candBertRecall.style.color = "var(--text-primary)";
      } else {
        candBertF1.textContent = "Not trained";
        candBertRecall.textContent = "Not trained";
        candBertF1.style.color = "var(--text-muted)";
        candBertRecall.style.color = "var(--text-muted)";
      }
    } else {
      candBox.style.display = "none";
    }
  }
}

async function loadModelUpdateStatus() {
  try {
    const [jobResponse, reviewedResponse] = await Promise.all([
      fetch("/api/admin/model-update/status"),
      fetch("/api/reviewed-examples/summary"),
    ]);
    if (!jobResponse.ok || !reviewedResponse.ok) {
      if (jobResponse.status === 401 || reviewedResponse.status === 401) {
        window.location.assign("/login?next=/admin");
        return;
      }
      throw new Error("Unable to load model update status");
    }
    renderModelUpdateStatus(await jobResponse.json());
    const reviewed = await reviewedResponse.json();
    reviewedExamplesCount.textContent = Number(reviewed.reviewed_count || 0).toLocaleString();
    reviewedExamplesPath.textContent = reviewed.path || "-";
  } catch (error) {
    renderModelUpdateStatus({ status: "failed" });
    modelUpdateMessage.textContent = String(error);
  }
}

async function startModelUpdateJob(endpoint, confirmationText) {
  const confirmed = window.confirm(confirmationText);
  if (!confirmed) {
    return;
  }

  try {
    setModelUpdateButtonsDisabled(true);
    modelUpdateMessage.textContent = "Starting script...";
    const response = await fetch(endpoint, { method: "POST" });

    if (!response.ok) {
      if (response.status === 401) {
        window.location.assign("/login?next=/admin");
        return;
      }
      if (response.status === 409) {
        throw new Error("Another model update job is already running.");
      }
      throw new Error("Unable to start model update script");
    }

    renderModelUpdateStatus(await response.json());
  } catch (error) {
    modelUpdateMessage.textContent = String(error);
    await loadModelUpdateStatus();
  }
}

trainLrCandidateButton.addEventListener("click", () => {
  startModelUpdateJob(
    "/api/admin/model-update/train-lr-candidate",
    "Start fast LR candidate retraining now? This is the recommended live demo path and will write models/candidates/lr_candidate/."
  );
});

promoteLrCandidateButton.addEventListener("click", () => {
  startModelUpdateJob(
    "/api/admin/model-update/promote-lr-candidate",
    "Promote the LR candidate if recall/F2 checks pass? This backs up and replaces the deployed LR cache artifact."
  );
});

trainCandidateButton.addEventListener("click", () => {
  startModelUpdateJob(
    "/api/admin/model-update/train-candidate",
    "Start WangchanBERTa candidate retraining now? This can take a long time and is better for offline/GPU runs."
  );
});

promoteCandidateButton.addEventListener("click", () => {
  startModelUpdateJob(
    "/api/admin/model-update/promote-candidate",
    "Promote the candidate model if recall/F2 checks pass? This backs up and replaces the deployed artifact."
  );
});

refreshModelUpdateButton.addEventListener("click", loadModelUpdateStatus);

async function loadSession() {
  try {
    const response = await fetch("/api/auth/me");
    if (!response.ok) {
      throw new Error(`Session check failed (${response.status})`);
    }
    const payload = await response.json();
    if (!payload.authenticated) {
      window.location.assign("/login?next=/admin");
      return false;
    }
    adminUserBadge.textContent = `User: ${payload.username || "moderator"}`;
    return true;
  } catch (error) {
    window.location.assign("/login?next=/admin");
    return false;
  }
}

async function logout() {
  logoutButton.disabled = true;
  logoutButton.textContent = "Logging out...";
  try {
    await fetch("/api/auth/logout", { method: "POST" });
  } finally {
    window.location.assign("/login");
  }
}

logoutButton.addEventListener("click", logout);

loadSession().then((ok) => {
  if (ok) {
    loadAdminData();
    loadMonitoringData();
    loadModelUpdateStatus();
  }
});
