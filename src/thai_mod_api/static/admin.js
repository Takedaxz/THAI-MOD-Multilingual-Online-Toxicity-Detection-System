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
const monitoringMetricsGrid = document.getElementById("monitoringMetricsGrid");
const monitoringLanguageMixGrid = document.getElementById("monitoringLanguageMixGrid");
const monitoringGeneratedAt = document.getElementById("monitoringGeneratedAt");
const refreshMonitoringButton = document.getElementById("refreshMonitoringButton");
const clearMonitoringLogButton = document.getElementById("clearMonitoringLogButton");
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
    monitoringMetricsGrid.innerHTML = `
      <div class="system-item">
        <span>Recent traffic</span>
        <strong>${report?.message || "No recent requests logged yet."}</strong>
      </div>
    `;
    monitoringLanguageMixGrid.innerHTML = `
      <div class="system-item">
        <span>Reference language mix</span>
        <strong>${formatLanguageMix(reference?.language_mix)}</strong>
      </div>
    `;
    return;
  }

  monitoringMetricsGrid.innerHTML = `
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
  `;

  const referenceMix = reference.language_mix || {};
  const recentMix = recent.language_mix || {};
  const mixOrder = [
    ["thai_only", "Thai only"],
    ["english_only", "English only"],
    ["mixed_script", "Mixed script"],
    ["other", "Other"],
  ];

  monitoringLanguageMixGrid.innerHTML = mixOrder
    .map(
      ([key, label]) => `
        <div class="system-item">
          <span>${label}</span>
          <strong>${formatPercent(referenceMix[key])} -> ${formatPercent(recentMix[key])}</strong>
        </div>
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
    const response = await fetch("/api/monitoring");

    if (!response.ok) {
      if (response.status === 401) {
        window.location.assign("/login?next=/admin");
        return;
      }
      throw new Error("Unable to load monitoring summary");
    }

    const report = await response.json();
    renderMonitoringSummary(report);
  } catch (error) {
    renderMonitoringSummary({
      message: String(error),
      reference_profile: null,
      recent_live_requests: null,
      monitoring_window: { prediction_count: 0, capacity: null, min_required: 20, full_confidence_required: 50 },
      drift_analysis: { status: "collecting_data", confidence: "collecting_data", psi: null },
    });
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
  }
});
