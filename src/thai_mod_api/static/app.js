const commentInput = document.getElementById("commentInput");
const thresholdInput = document.getElementById("thresholdInput");
const thresholdValue = document.getElementById("thresholdValue");
const heroThreshold = document.getElementById("heroThreshold");
const analyzeButton = document.getElementById("analyzeButton");
const historyList = document.getElementById("historyList");
const triageBanner = document.getElementById("triageBanner");
const resultAccent = document.getElementById("resultAccent");
const actionTag = document.getElementById("actionTag");
const processingPanel = document.getElementById("processingPanel");
const processingLabel = document.getElementById("processingLabel");

const predictedLabel = document.getElementById("predictedLabel");
const recommendation = document.getElementById("recommendation");
const toxicScore = document.getElementById("toxicScore");
const confidence = document.getElementById("confidence");
const resultThreshold = document.getElementById("resultThreshold");
const sourceModel = document.getElementById("sourceModel");
const reviewToxicButton = document.getElementById("reviewToxicButton");
const reviewNonToxicButton = document.getElementById("reviewNonToxicButton");
const reviewStatus = document.getElementById("reviewStatus");
const riskPercent = document.getElementById("riskPercent");
const confidencePercent = document.getElementById("confidencePercent");
const riskBar = document.getElementById("riskBar");
const confidenceBar = document.getElementById("confidenceBar");

const modelVersion = document.getElementById("modelVersion");
const healthStatus = document.getElementById("healthStatus");

const recentHistory = [];
let currentPrediction = null;

async function ensureAnalyzerAccess() {
  try {
    const response = await fetch("/api/auth/me");
    if (!response.ok) {
      return;
    }
    const payload = await response.json();
    if (payload.protect_analyzer && !payload.authenticated) {
      window.location.assign("/login?next=/");
    }
  } catch (_error) {
    // Ignore auth check failures; existing analyzer behavior should continue.
  }
}

function setThresholdLabel() {
  const value = Number(thresholdInput.value).toFixed(2);
  thresholdValue.textContent = value;
  heroThreshold.textContent = value;
}

function getRiskClass(label) {
  return label === "toxic" ? "risk" : "safe";
}

function getActionLabel(result) {
  return result.predicted_label === "toxic" ? "Hide / Review" : "Allow";
}

function toPercent(value) {
  return `${Math.round(Number(value) * 100)}%`;
}

function trimText(text, maxLength = 120) {
  if (!text) {
    return "-";
  }

  return text.length > maxLength ? `${text.slice(0, maxLength - 1)}...` : text;
}

function setPrediction(result) {
  const riskClass = getRiskClass(result.predicted_label);
  currentPrediction = result;

  predictedLabel.textContent = result.predicted_label;
  recommendation.textContent = result.recommendation;
  toxicScore.textContent = toPercent(result.toxic_score);
  confidence.textContent = toPercent(result.confidence);
  resultThreshold.textContent = Number(result.threshold).toFixed(2);
  sourceModel.textContent = result.source_model || "-";
  riskPercent.textContent = toPercent(result.toxic_score);
  confidencePercent.textContent = toPercent(result.confidence);
  riskBar.style.width = toPercent(result.toxic_score);
  confidenceBar.style.width = toPercent(result.confidence);

  triageBanner.className = `result-banner ${riskClass}`;
  resultAccent.className = `result-accent ${riskClass}`;
  actionTag.className = `result-action ${riskClass}`;
  actionTag.textContent = getActionLabel(result);
  reviewStatus.textContent = `Ready to save human label for request ${result.request_id.slice(0, 8)}.`;
}

function renderHistory() {
  historyList.innerHTML = "";

  if (recentHistory.length === 0) {
    historyList.innerHTML = `
      <tr class="history-empty">
        <td colspan="5">No checks yet. Analyze a comment to start the session trail.</td>
      </tr>
    `;
    return;
  }

  recentHistory.forEach((item) => {
    const row = document.createElement("tr");
    const riskClass = getRiskClass(item.predicted_label);

    row.innerHTML = `
      <td>${new Date(item.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</td>
      <td><span class="decision-badge ${riskClass}">${item.recommendation}</span></td>
      <td><span class="label-badge ${riskClass}">${item.predicted_label}</span></td>
      <td>${toPercent(item.toxic_score)}</td>
      <td class="history-comment">${trimText(item.text)}</td>
    `;

    historyList.appendChild(row);
  });
}

async function loadSystemInfo() {
  try {
    const [healthResponse, modelInfoResponse] = await Promise.all([
      fetch("/api/health"),
      fetch("/api/model-info"),
    ]);

    if (!healthResponse.ok || !modelInfoResponse.ok) {
      throw new Error("Unable to load system metadata");
    }

    const health = await healthResponse.json();
    const modelInfo = await modelInfoResponse.json();

    healthStatus.textContent = health.status === "ok" ? "Online" : "Degraded";
    modelVersion.textContent = modelInfo.model_name;
    sourceModel.textContent = modelInfo.model_name;
  } catch (error) {
    healthStatus.textContent = "Offline";
    modelVersion.textContent = "Unavailable";
  }
}

async function analyzeComment() {
  const text = commentInput.value.trim();
  if (!text) {
    commentInput.focus();
    return;
  }

  analyzeButton.disabled = true;
  analyzeButton.textContent = "Analyzing...";
  processingPanel.classList.remove("hidden");
  processingLabel.textContent = "Deconstructing semantic tokens...";

  recommendation.textContent = "RUNNING_ANALYSIS";
  currentPrediction = null;
  reviewStatus.textContent = "Analyze a comment before saving a reviewed label.";
  predictedLabel.textContent = "pending";
  toxicScore.textContent = "-";
  confidence.textContent = "-";
  riskPercent.textContent = "0%";
  confidencePercent.textContent = "0%";
  riskBar.style.width = "0%";
  confidenceBar.style.width = "0%";
  triageBanner.className = "result-banner neutral";
  resultAccent.className = "result-accent";
  actionTag.className = "result-action neutral";
  actionTag.textContent = "Processing";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text,
        threshold: Number(thresholdInput.value),
      }),
    });

    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const result = await response.json();
    setPrediction(result);

    recentHistory.unshift({
      ...result,
      timestamp: new Date().toISOString(),
    });
    recentHistory.splice(6);
    renderHistory();
  } catch (error) {
    predictedLabel.textContent = "error";
    recommendation.textContent = "API_UNAVAILABLE";
    triageBanner.className = "result-banner risk";
    resultAccent.className = "result-accent risk";
    actionTag.className = "result-action risk";
    actionTag.textContent = "Check API";
    toxicScore.textContent = "-";
    confidence.textContent = "-";
    riskPercent.textContent = "-";
    confidencePercent.textContent = "-";
    riskBar.style.width = "0%";
    confidenceBar.style.width = "0%";
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.textContent = "Analyze Comment";
    processingPanel.classList.add("hidden");
  }
}

function setReviewButtonsDisabled(disabled) {
  reviewToxicButton.disabled = disabled;
  reviewNonToxicButton.disabled = disabled;
}

async function submitReviewedLabel(reviewedLabel) {
  if (!currentPrediction) {
    reviewStatus.textContent = "Analyze a comment before saving a reviewed label.";
    return;
  }

  try {
    setReviewButtonsDisabled(true);
    reviewStatus.textContent = "Saving reviewed example...";
    const response = await fetch("/api/reviewed-examples", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        request_id: currentPrediction.request_id,
        text: currentPrediction.text,
        reviewed_label: reviewedLabel,
        predicted_label: currentPrediction.predicted_label,
        toxicity_score: currentPrediction.toxic_score,
        source_model: currentPrediction.source_model,
      }),
    });

    if (!response.ok) {
      if (response.status === 401) {
        reviewStatus.textContent = "Login required before saving reviewed examples.";
        return;
      }
      throw new Error(`Save failed with status ${response.status}`);
    }

    const payload = await response.json();
    reviewStatus.textContent = `Saved. Reviewed examples: ${Number(payload.reviewed_count).toLocaleString()}.`;
  } catch (error) {
    reviewStatus.textContent = String(error);
  } finally {
    setReviewButtonsDisabled(false);
  }
}

thresholdInput.addEventListener("input", setThresholdLabel);
analyzeButton.addEventListener("click", analyzeComment);
reviewToxicButton.addEventListener("click", () => submitReviewedLabel("neg"));
reviewNonToxicButton.addEventListener("click", () => submitReviewedLabel("neu"));

document.querySelectorAll(".sample-button").forEach((button) => {
  button.addEventListener("click", () => {
    commentInput.value = button.dataset.text;
    analyzeComment();
  });
});

setThresholdLabel();
renderHistory();
ensureAnalyzerAccess();
loadSystemInfo();
