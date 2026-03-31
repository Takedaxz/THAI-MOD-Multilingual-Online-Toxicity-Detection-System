const adminHealthBadge = document.getElementById("adminHealthBadge");
const adminModelName = document.getElementById("adminModelName");
const adminDeploymentMode = document.getElementById("adminDeploymentMode");
const adminCacheStatus = document.getElementById("adminCacheStatus");
const adminStatus = document.getElementById("adminStatus");
const adminThreshold = document.getElementById("adminThreshold");
const adminLoaded = document.getElementById("adminLoaded");
const adminDatasetRows = document.getElementById("adminDatasetRows");
const adminMetricsList = document.getElementById("adminMetricsList");
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

async function loadAdminData() {
  try {
    const [healthResponse, modelInfoResponse] = await Promise.all([
      fetch("/api/health"),
      fetch("/api/model-info"),
    ]);

    if (!healthResponse.ok || !modelInfoResponse.ok) {
      throw new Error("Unable to load admin data");
    }

    const health = await healthResponse.json();
    const info = await modelInfoResponse.json();

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

loadAdminData();
