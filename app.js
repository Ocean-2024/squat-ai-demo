const visionBaseUrl = "./node_modules/@mediapipe/tasks-vision";
let visionModule;
let mediaPipeBase = visionBaseUrl;
let modelPath = "./models/pose_landmarker_lite.task";

try {
  visionModule = await import(`${visionBaseUrl}/vision_bundle.mjs`);
} catch (error) {
  console.warn("Local MediaPipe bundle unavailable, falling back to CDN.", error);
  mediaPipeBase = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21";
  modelPath =
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
  visionModule = await import(`${mediaPipeBase}/vision_bundle.mjs`);
}

const { PoseLandmarker, FilesetResolver, DrawingUtils } = visionModule;

const video = document.querySelector("#video");
const overlay = document.querySelector("#overlay");
const overlayCtx = overlay.getContext("2d");
const stage = document.querySelector(".stage");
const emptyState = document.querySelector("#emptyState");
const trackingStatus = document.querySelector("#trackingStatus");
const cameraBtn = document.querySelector("#cameraBtn");
const uploadBtn = document.querySelector("#uploadBtn");
const videoInput = document.querySelector("#videoInput");
const demoBtn = document.querySelector("#demoBtn");
const resetBtn = document.querySelector("#resetBtn");
const repCount = document.querySelector("#repCount");
const phaseEl = document.querySelector("#phase");
const scoreRing = document.querySelector("#scoreRing");
const scoreText = document.querySelector("#scoreText");
const metricMode = document.querySelector("#metricMode");
const kneeMetric = document.querySelector("#kneeMetric");
const hipMetric = document.querySelector("#hipMetric");
const torsoMetric = document.querySelector("#torsoMetric");
const extraMetricLabel = document.querySelector("#extraMetricLabel");
const extraMetric = document.querySelector("#extraMetric");
const feedbackMessage = document.querySelector("#feedbackMessage");
const feedbackHistory = document.querySelector("#feedbackHistory");
const feedbackDot = document.querySelector(".feedback-dot");
const viewButtons = document.querySelectorAll("[data-view]");

const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

const viewLabels = {
  auto: "自动",
  side: "侧面",
  front: "正面",
};

function getActiveView(metrics) {
  if (state.view !== "auto") {
    return state.view;
  }

  if (!metrics || metrics.viewHint === "ambiguous") {
    return "side";
  }

  return metrics.viewHint;
}

function updateViewButtons() {
  viewButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.view === state.view);
  });
}

function updateViewLabels(activeView) {
  if (state.view === "auto") {
    metricMode.textContent = activeView
      ? activeView === "front"
        ? "自动 · 正面"
        : "自动 · 侧面"
      : "自动";
  } else {
    metricMode.textContent = viewLabels[state.view] ?? "侧面";
  }

  extraMetricLabel.textContent = activeView === "front" ? "膝内扣" : "膝前移";
}

function getTrackingStatusText(metrics) {
  if (!metrics) {
    return { text: "未检测到身体", warning: true };
  }

  if (metrics.visibility < 0.5) {
    return { text: "识别不稳定：光线不足或背景过杂", warning: true };
  }

  if (metrics.bodyHeight < 0.42) {
    return { text: "身体区域偏小：请靠近镜头", warning: true };
  }

  if (metrics.bodyHeight > 0.96) {
    return { text: "身体区域过大：请稍后退", warning: true };
  }

  if (Math.abs(metrics.bodyCenterX - 0.5) > 0.24) {
    return { text: "身体未居中：请调整站位", warning: true };
  }

  if (metrics.viewHint === "front" && state.view === "side") {
    return { text: "当前更像正面视角：请切换到正面模式查看膝内扣", warning: true };
  }

  if (metrics.viewHint === "front" && state.view !== "side") {
    return { text: "正面视角：重点看膝内扣，深度建议用侧面", warning: false };
  }

  if (metrics.viewHint === "ambiguous" && state.view === "auto") {
    return { text: "视角不明确：建议侧对镜头", warning: true };
  }

  return { text: "识别中", warning: false };
}

const state = {
  mode: "idle",
  view: "auto",
  raf: 0,
  poseLandmarker: null,
  drawingUtils: null,
  poseReady: false,
  cameraStream: null,
  videoUrl: null,
  landmarks: null,
  kneeAngle: 180,
  hipAngle: 180,
  torsoAngle: 0,
  shinAngle: 0,
  kneeForwardRatio: 1,
  kneeValgus: 0,
  depth: 0,
  phase: "idle",
  reachedBottom: false,
  trackingRep: null,
  repCount: 0,
  reps: [],
  smoothed: {
    kneeAngle: 180,
    hipAngle: 180,
    torsoAngle: 0,
    kneeValgus: 0,
  },
};

async function loadModelWithProgress(url, onProgress) {
  const response = await fetch(url, { cache: "force-cache" });

  if (!response.ok) {
    throw new Error(`Model request failed: ${response.status}`);
  }

  const total = Number(response.headers.get("Content-Length")) || 0;
  const reader = response.body?.getReader();

  if (!reader) {
    return new Uint8Array(await response.arrayBuffer());
  }

  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();

    if (done) {
      break;
    }

    chunks.push(value);
    received += value.length;

    if (total > 0) {
      onProgress?.(Math.round((received / total) * 100));
    }
  }

  const buffer = await new Blob(chunks).arrayBuffer();
  return new Uint8Array(buffer);
}

async function initPose() {
  if (state.poseReady) {
    return;
  }

  setStatus("正在加载模型");

  try {
    const modelBuffer = await loadModelWithProgress(modelPath, (percent) => {
      setStatus(`正在加载模型 ${percent}%`);
    });

    try {
      await createPose("GPU", modelBuffer);
    } catch (error) {
      console.warn("GPU delegate unavailable, falling back to CPU.", error);
      await createPose("CPU", modelBuffer);
    }
  } catch (error) {
    console.error(error);
    setStatus("模型加载失败，请检查网络后重试", true);
    throw error;
  }
}

async function createPose(delegate, modelBuffer) {
  const vision = await FilesetResolver.forVisionTasks(
    `${mediaPipeBase}/wasm`,
  );

  state.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetBuffer: modelBuffer,
      delegate,
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.55,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  state.drawingUtils = new DrawingUtils(overlayCtx);
  state.poseReady = true;
}

function setStatus(text, warning = false) {
  trackingStatus.textContent = text;
  trackingStatus.classList.toggle("is-warning", warning);
}

function hideEmptyState() {
  emptyState.classList.add("is-hidden");
}

function showEmptyState() {
  emptyState.classList.remove("is-hidden");
}

async function stopCurrent() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach((track) => track.stop());
    state.cameraStream = null;
  }

  if (state.videoUrl) {
    URL.revokeObjectURL(state.videoUrl);
    state.videoUrl = null;
  }

  video.pause();
  video.removeAttribute("src");
  video.load();
  video.classList.remove("is-live");
  state.mode = "idle";
  state.landmarks = null;
}

async function startCamera() {
  try {
    await initPose();
  } catch (error) {
    return;
  }

  await stopCurrent();

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });

    state.cameraStream = stream;
    video.srcObject = stream;
    await video.play();
    video.classList.add("is-live");
    state.mode = "camera";
    hideEmptyState();
    setStatus("识别中");
  } catch (error) {
    console.error(error);
    state.mode = "idle";
    showEmptyState();
    setStatus("摄像头不可用，可尝试上传视频或演示模式", true);
  }
}

async function startVideo(file) {
  if (!file) {
    return;
  }

  try {
    await initPose();
  } catch (error) {
    return;
  }

  await stopCurrent();

  const url = URL.createObjectURL(file);
  state.videoUrl = url;
  video.src = url;
  video.muted = true;
  video.loop = true;
  video.playsInline = true;

  try {
    await video.play();
  } catch (error) {
    console.error(error);
    URL.revokeObjectURL(state.videoUrl);
    state.videoUrl = null;
    showEmptyState();
    setStatus("视频无法播放", true);
    return;
  }

  video.classList.add("is-live");
  state.mode = "video";
  hideEmptyState();
  setStatus("识别中");
}

async function startDemo() {
  try {
    await initPose();
  } catch (error) {
    return;
  }

  await stopCurrent();

  state.mode = "demo";
  hideEmptyState();
  setStatus("演示模式");
  resetSessionData();
}

function resetSession() {
  stopCurrent();
  showEmptyState();
  resetSessionData();
  setStatus("准备开始");
  renderMetricsPlaceholder();
  setFeedback("侧对镜头，保持全身入镜后开始深蹲。", "ok");
}

function resetSessionData() {
  state.phase = "idle";
  state.reachedBottom = false;
  state.trackingRep = null;
  state.repCount = 0;
  state.reps = [];
  state.landmarks = null;
  state.view = "auto";
  updateViewButtons();
  state.smoothed = {
    kneeAngle: 180,
    hipAngle: 180,
    torsoAngle: 0,
    kneeValgus: 0,
  };
  updateSummary();
  updatePhase();
  updateScore(null);
  renderFeedbackHistory();
  renderMetricsPlaceholder();
}

function startLoop() {
  const frame = (now) => {
    state.raf = requestAnimationFrame(frame);

    if (state.mode === "camera" || state.mode === "video") {
      if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        return;
      }

      resizeCanvas();
      const results = state.poseLandmarker.detectForVideo(video, performance.now());
      handlePoseResults(results);
    }

    if (state.mode === "demo") {
      resizeCanvas();
      const landmarks = buildDemoLandmarks(now);
      const metrics = processLandmarks(landmarks);
      drawDemoFrame(landmarks, metrics, now);
    }
  };

  state.raf = requestAnimationFrame(frame);
}

function handlePoseResults(results) {
  const landmarks = results?.landmarks?.[0] ?? null;

  if (!landmarks) {
    state.landmarks = null;
    setStatus("未检测到身体", true);
    return;
  }

  state.landmarks = landmarks;
  setStatus("识别中");
  const metrics = processLandmarks(landmarks);
  drawCameraFrame(landmarks, metrics);
}

function processLandmarks(landmarks) {
  const metrics = computeMetrics(landmarks);

  if (!metrics) {
    state.landmarks = null;
    setStatus("未检测到身体", true);
    renderMetricsPlaceholder();
    return null;
  }

  state.kneeAngle = metrics.kneeAngle;
  state.hipAngle = metrics.hipAngle;
  state.torsoAngle = metrics.torsoAngle;
  state.shinAngle = metrics.shinAngle;
  state.kneeForwardRatio = metrics.kneeForwardRatio;
  state.smoothed.kneeValgus =
    state.smoothed.kneeValgus * 0.68 + metrics.kneeValgus * 0.32;
  state.kneeValgus = state.smoothed.kneeValgus;
  metrics.kneeValgus = state.kneeValgus;
  state.depth = metrics.depth;

  const previousSmoothedKneeAngle = state.smoothed.kneeAngle;
  state.smoothed.kneeAngle =
    state.smoothed.kneeAngle * 0.72 + metrics.kneeAngle * 0.28;
  state.smoothed.hipAngle =
    state.smoothed.hipAngle * 0.72 + metrics.hipAngle * 0.28;
  state.smoothed.torsoAngle =
    state.smoothed.torsoAngle * 0.72 + metrics.torsoAngle * 0.28;

  updateRepState(
    state.smoothed.kneeAngle,
    previousSmoothedKneeAngle,
    metrics,
  );
  const status = getTrackingStatusText(metrics);
  setStatus(status.text, status.warning);
  updateMetrics(metrics);
  updatePhase();
  updateFeedback(metrics);

  return metrics;
}

function computeMetrics(landmarks) {
  const p = (index) => landmarks[index] ?? { x: 0, y: 0, z: 0, visibility: 0 };
  const mid = (a, b) => ({
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: (a.z + b.z) / 2,
    visibility: Math.min(a.visibility, b.visibility),
  });

  const leftHip = p(23);
  const rightHip = p(24);
  const leftKnee = p(25);
  const rightKnee = p(26);
  const leftAnkle = p(27);
  const rightAnkle = p(28);
  const leftShoulder = p(11);
  const rightShoulder = p(12);
  const leftFoot = p(31);
  const rightFoot = p(32);

  const hip = mid(leftHip, rightHip);
  const knee = mid(leftKnee, rightKnee);
  const ankle = mid(leftAnkle, rightAnkle);
  const shoulder = mid(leftShoulder, rightShoulder);
  const toe = mid(leftFoot, rightFoot);

  const visibility = Math.min(
    leftHip.visibility,
    rightHip.visibility,
    leftKnee.visibility,
    rightKnee.visibility,
    leftAnkle.visibility,
    rightAnkle.visibility,
    leftShoulder.visibility,
    rightShoulder.visibility,
  );

  if (visibility < 0.42) {
    return null;
  }

  const kneeAngle = angleBetween(hip, knee, ankle);
  const hipAngle = angleBetween(shoulder, hip, knee);
  const torsoAngle = signedVerticalAngle(shoulder, hip);
  const shinAngle = signedVerticalAngle(knee, ankle);

  const forward = toe.x - ankle.x;
  const forwardSign = Math.abs(forward) < 0.004 ? 1 : Math.sign(forward);
  const toeTravel = Math.max(Math.abs(toe.x - ankle.x), 0.035);
  const kneeForward = (knee.x - ankle.x) * forwardSign;
  const kneeForwardRatio = clamp(kneeForward / toeTravel, -0.5, 2.5);

  const expectedLeftKneeX = (leftHip.x + leftAnkle.x) / 2;
  const expectedRightKneeX = (rightHip.x + rightAnkle.x) / 2;
  const leftKneeShift = leftKnee.x - expectedLeftKneeX;
  const rightKneeShift = rightKnee.x - expectedRightKneeX;
  const kneeScale = Math.max(
    Math.abs(leftHip.x - rightHip.x),
    Math.abs(leftAnkle.x - rightAnkle.x),
    0.09,
  );
  const kneeValgus =
    (Math.abs(leftKneeShift) + Math.abs(rightKneeShift)) / 2 / kneeScale;

  const depth = clamp((135 - kneeAngle) / 45, 0, 1);

  const landmarkPoints = [
    leftShoulder,
    rightShoulder,
    leftHip,
    rightHip,
    leftKnee,
    rightKnee,
    leftAnkle,
    rightAnkle,
  ];
  const minX = Math.min(...landmarkPoints.map((point) => point.x));
  const maxX = Math.max(...landmarkPoints.map((point) => point.x));
  const minY = Math.min(...landmarkPoints.map((point) => point.y));
  const maxY = Math.max(...landmarkPoints.map((point) => point.y));
  const bodyHeight = Math.max(maxY - minY, 0.02);
  const bodyCenterX = (minX + maxX) / 2;
  const shoulderWidth = Math.abs(leftShoulder.x - rightShoulder.x);
  const hipWidth = Math.abs(leftHip.x - rightHip.x);
  const ankleWidth = Math.abs(leftAnkle.x - rightAnkle.x);
  const landmarkWidth = Math.max(shoulderWidth, hipWidth, ankleWidth);
  const viewHint =
    landmarkWidth > 0.34 ? "front" : landmarkWidth < 0.22 ? "side" : "ambiguous";

  return {
    hip,
    knee,
    ankle,
    shoulder,
    toe,
    kneeAngle,
    hipAngle,
    torsoAngle,
    shinAngle,
    kneeForwardRatio,
    kneeValgus,
    depth,
    bodyHeight,
    bodyCenterX,
    viewHint,
  };
}

function angleBetween(a, b, c) {
  const v1 = { x: a.x - b.x, y: a.y - b.y };
  const v2 = { x: c.x - b.x, y: c.y - b.y };
  const dot = v1.x * v2.x + v1.y * v2.y;
  const mag = Math.hypot(v1.x, v1.y) * Math.hypot(v2.x, v2.y);

  if (mag < 0.00001) {
    return 180;
  }

  const radians = Math.acos(clamp(dot / mag, -1, 1));
  return (radians * 180) / Math.PI;
}

function signedVerticalAngle(top, bottom) {
  const dx = top.x - bottom.x;
  const dy = bottom.y - top.y;
  return (Math.atan2(dx, dy) * 180) / Math.PI;
}

function updateRepState(smoothedKneeAngle, previousSmoothedKneeAngle, metrics) {
  const velocity = smoothedKneeAngle - previousSmoothedKneeAngle;

  if (smoothedKneeAngle > 152) {
    if (state.reachedBottom && state.trackingRep) {
      completeRep();
    }
    state.reachedBottom = false;
    state.trackingRep = null;
    state.phase = "standing";
    return;
  }

  if (smoothedKneeAngle < 104) {
    state.reachedBottom = true;
    state.phase = "bottom";
    if (!state.trackingRep) {
      state.trackingRep = {
        minKneeAngle: metrics.kneeAngle,
        maxTorso: Math.abs(metrics.torsoAngle),
        maxValgus: metrics.kneeValgus,
        maxKneeForward: metrics.kneeForwardRatio,
      };
    }
    updateRepTracking(metrics);
    return;
  }

  if (velocity < -0.9) {
    state.phase = "lowering";
    if (!state.trackingRep) {
      state.trackingRep = {
        minKneeAngle: metrics.kneeAngle,
        maxTorso: Math.abs(metrics.torsoAngle),
        maxValgus: metrics.kneeValgus,
        maxKneeForward: metrics.kneeForwardRatio,
      };
    }
  } else if (velocity > 0.9) {
    state.phase = "rising";
  } else if (state.phase === "standing" || state.phase === "idle") {
    state.phase = "lowering";
  }

  if (!state.trackingRep) {
    state.trackingRep = {
      minKneeAngle: metrics.kneeAngle,
      maxTorso: Math.abs(metrics.torsoAngle),
      maxValgus: metrics.kneeValgus,
      maxKneeForward: metrics.kneeForwardRatio,
    };
  }

  updateRepTracking(metrics);
}

function updateRepTracking(metrics) {
  if (!state.trackingRep) {
    return;
  }

  state.trackingRep.minKneeAngle = Math.min(
    state.trackingRep.minKneeAngle,
    metrics.kneeAngle,
  );
  state.trackingRep.maxTorso = Math.max(
    state.trackingRep.maxTorso,
    Math.abs(metrics.torsoAngle),
  );
  state.trackingRep.maxValgus = Math.max(
    state.trackingRep.maxValgus,
    metrics.kneeValgus,
  );
  state.trackingRep.maxKneeForward = Math.max(
    state.trackingRep.maxKneeForward,
    metrics.kneeForwardRatio,
  );
}

function completeRep() {
  const rep = state.trackingRep;

  if (!rep) {
    return;
  }

  const depthQuality = clamp(1 - Math.abs(90 - rep.minKneeAngle) / 38, 0, 1);
  const valgusPenalty = clamp(rep.maxValgus / 0.34, 0, 1);
  const torsoPenalty = clamp((rep.maxTorso - 18) / 32, 0, 1);
  const kneePenalty = clamp((rep.maxKneeForward - 1.15) / 0.6, 0, 1);

  const raw =
    0.45 * depthQuality +
    0.28 * (1 - valgusPenalty) +
    0.15 * (1 - torsoPenalty) +
    0.12 * (1 - kneePenalty);

  const quality = Math.round(clamp(raw * 100, 42, 100));

  state.repCount += 1;
  state.reps.unshift({
    quality,
    minKneeAngle: Math.round(rep.minKneeAngle),
    time: Date.now(),
  });
  state.trackingRep = null;
  state.reachedBottom = false;
  updateSummary();
  updateScore(quality);
  renderFeedbackHistory();
}

function updateSummary() {
  repCount.textContent = String(state.repCount);
}

function updatePhase() {
  const labels = {
    idle: "待机",
    standing: "站姿",
    lowering: "下蹲",
    bottom: "底部",
    rising: "起身",
  };
  phaseEl.textContent = labels[state.phase] ?? "待机";
}

function updateScore(value) {
  if (value === null || value === undefined) {
    scoreRing.textContent = "--";
    scoreText.textContent = "等待动作";
    scoreRing.style.background =
      "radial-gradient(circle at center, var(--surface) 0 58%, transparent 60%), conic-gradient(var(--muted-2) 0 0%, transparent 0 0%)";
    return;
  }

  const color = value >= 85 ? "#34d399" : value >= 70 ? "#fbbf24" : "#f87171";
  const label = value >= 85 ? "动作稳定" : value >= 70 ? "基本达标" : "需要调整";

  scoreRing.textContent = String(value);
  scoreText.textContent = label;
  scoreRing.style.background = `radial-gradient(circle at center, var(--surface) 0 58%, transparent 60%), conic-gradient(${color} 0 ${value}%, rgba(255,255,255,.08) ${value}% 100%)`;
}

function updateMetrics(metrics) {
  kneeMetric.textContent = `${Math.round(metrics.kneeAngle)}°`;
  hipMetric.textContent = `${Math.round(metrics.hipAngle)}°`;
  torsoMetric.textContent = `${metrics.torsoAngle > 0 ? "+" : ""}${Math.round(
    metrics.torsoAngle,
  )}°`;
  const activeView = getActiveView(metrics);
  updateViewLabels(activeView);

  if (activeView === "side") {
    extraMetricLabel.textContent = "膝前移";
    extraMetric.textContent = `${metrics.kneeForwardRatio.toFixed(2)}×`;
  } else {
    extraMetricLabel.textContent = "膝内扣";
    extraMetric.textContent = `${Math.round(metrics.kneeValgus * 100)}%`;
  }
}

function renderMetricsPlaceholder() {
  kneeMetric.textContent = "--";
  hipMetric.textContent = "--";
  torsoMetric.textContent = "--";
  extraMetric.textContent = "--";
  updateViewLabels(null);
}

function updateFeedback(metrics) {
  const activeView = getActiveView(metrics);
  const feedback = getPrimaryFeedback(metrics, activeView);
  setFeedback(feedback.text, feedback.severity);
}

function getPrimaryFeedback(metrics, activeView) {
  if (activeView === "front" && metrics.kneeValgus > 0.24) {
    return {
      text: "膝盖明显内扣，注意让膝盖朝脚尖方向打开。",
      severity: "danger",
    };
  }

  if (activeView === "front" && metrics.kneeValgus > 0.13) {
    return {
      text: "膝盖有轻微内扣，收紧臀部，保持膝盖稳定。",
      severity: "warn",
    };
  }

  if (state.phase === "bottom" && metrics.kneeAngle > 108) {
    return {
      text: "再蹲低一点，目标是大腿接近与地面平行。",
      severity: "warn",
    };
  }

  if (state.phase === "bottom" && metrics.kneeAngle < 62) {
    return {
      text: "蹲得过深，注意保持下背中立，不要骨盆翻转。",
      severity: "danger",
    };
  }

  if (metrics.kneeForwardRatio > 1.12) {
    return {
      text: "膝盖过度前移，把重心放回脚掌中部。",
      severity: "warn",
    };
  }

  if (Math.abs(metrics.torsoAngle) > 32) {
    return {
      text: "躯干过度前倾，收紧核心并保持胸部打开。",
      severity: "warn",
    };
  }

  if (metrics.depth > 0.58) {
    return {
      text: "深度不错，保持当前节奏。",
      severity: "ok",
    };
  }

  if (state.phase === "standing") {
    return {
      text: "回到站姿，膝盖保持微屈，准备下一次。",
      severity: "ok",
    };
  }

  return {
    text: "继续下蹲，保持核心稳定。",
    severity: "ok",
  };
}

function setFeedback(text, severity) {
  feedbackMessage.textContent = text;
  feedbackMessage.className = "feedback-message";

  if (severity === "warn") {
    feedbackMessage.classList.add("is-warning");
    feedbackDot.style.background = "var(--amber)";
  } else if (severity === "danger") {
    feedbackMessage.classList.add("is-danger");
    feedbackDot.style.background = "var(--red)";
  } else {
    feedbackDot.style.background = "var(--green)";
  }
}

function renderFeedbackHistory() {
  if (!state.reps.length) {
    feedbackHistory.innerHTML = "";
    return;
  }

  feedbackHistory.innerHTML = state.reps
    .slice(0, 3)
    .map((rep, index) => {
      const number = state.repCount - index;
      return `
        <div class="history-item">
          <span>第 ${number} 次 · 最低膝角 ${rep.minKneeAngle}°</span>
          <strong>${rep.quality}</strong>
        </div>
      `;
    })
    .join("");
}

function resizeCanvas() {
  const rect = stage.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));

  if (overlay.width !== width || overlay.height !== height) {
    overlay.width = width;
    overlay.height = height;
  }
}

function toPixel(point) {
  return {
    x: point.x * overlay.width,
    y: point.y * overlay.height,
  };
}

function drawCameraFrame(landmarks, metrics) {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  state.drawingUtils.drawConnectors(
    landmarks,
    PoseLandmarker.POSE_CONNECTIONS,
    {
      color: "rgba(228, 240, 235, 0.82)",
      lineWidth: 2,
    },
  );

  state.drawingUtils.drawLandmarks(landmarks, {
    color: "#34d399",
    fillColor: "#34d399",
    radius: 3,
  });

  if (!metrics) {
    return;
  }

  highlightProblems(metrics, getActiveView(metrics));
  drawKneeAngleLabel(metrics);
}

function highlightProblems(metrics, activeView) {
  const colors = getProblemColor(metrics, activeView);

  if (!colors) {
    return;
  }

  const hip = toPixel(metrics.hip);
  const knee = toPixel(metrics.knee);
  const ankle = toPixel(metrics.ankle);
  const shoulder = toPixel(metrics.shoulder);

  overlayCtx.save();
  overlayCtx.strokeStyle = colors.line;
  overlayCtx.fillStyle = colors.dot;
  overlayCtx.lineWidth = 4;
  overlayCtx.lineCap = "round";
  overlayCtx.lineJoin = "round";

  overlayCtx.beginPath();
  overlayCtx.moveTo(shoulder.x, shoulder.y);
  overlayCtx.lineTo(hip.x, hip.y);
  overlayCtx.lineTo(knee.x, knee.y);
  overlayCtx.lineTo(ankle.x, ankle.y);
  overlayCtx.stroke();

  [shoulder, hip, knee, ankle].forEach((point) => {
    overlayCtx.beginPath();
    overlayCtx.arc(point.x, point.y, 6, 0, Math.PI * 2);
    overlayCtx.fill();
  });

  overlayCtx.restore();
}

function getProblemColor(metrics, activeView) {
  if (activeView === "front" && metrics.kneeValgus > 0.13) {
    return metrics.kneeValgus > 0.24
      ? { line: "rgba(248, 113, 113, .9)", dot: "#f87171" }
      : { line: "rgba(251, 191, 36, .9)", dot: "#fbbf24" };
  }

  if (metrics.kneeForwardRatio > 1.12 || Math.abs(metrics.torsoAngle) > 32) {
    return { line: "rgba(251, 191, 36, .88)", dot: "#fbbf24" };
  }

  if (state.phase === "bottom" && (metrics.kneeAngle > 108 || metrics.kneeAngle < 62)) {
    return { line: "rgba(251, 191, 36, .88)", dot: "#fbbf24" };
  }

  return null;
}

function drawKneeAngleLabel(metrics) {
  const knee = toPixel(metrics.knee);
  overlayCtx.save();
  overlayCtx.font = "600 14px Inter, sans-serif";
  overlayCtx.fillStyle = "#0b1017";
  overlayCtx.strokeStyle = "rgba(255,255,255,.76)";
  overlayCtx.lineWidth = 4;
  overlayCtx.textAlign = "center";
  overlayCtx.textBaseline = "bottom";

  const label = `${Math.round(metrics.kneeAngle)}°`;
  const x = clamp(knee.x, 28, overlay.width - 28);
  const y = clamp(knee.y - 10, 24, overlay.height - 14);

  overlayCtx.strokeText(label, x, y);
  overlayCtx.fillText(label, x, y);
  overlayCtx.restore();
}

function buildDemoLandmarks(now) {
  const time = now / 1000;
  const cycle = (Math.sin(time * 1.45) + 1) / 2;
  const p = Math.pow(cycle, 1.08);

  const hip = { x: 0.48 + p * 0.018, y: 0.47 + p * 0.12, z: 0 };
  const shoulder = {
    x: hip.x + p * 0.026,
    y: hip.y - 0.2,
    z: 0,
  };
  const knee = {
    x: hip.x + p * 0.032,
    y: hip.y + 0.24 - p * 0.015,
    z: 0,
  };
  const ankle = { x: 0.515, y: 0.93, z: 0 };
  const toe = { x: 0.565, y: 0.93, z: 0 };
  const head = { x: shoulder.x, y: shoulder.y - 0.085, z: 0 };
  const elbow = { x: shoulder.x + 0.04, y: shoulder.y + 0.075, z: 0 };
  const wrist = { x: shoulder.x + 0.075, y: shoulder.y + 0.16, z: 0 };

  const make = (point) => ({
    x: point.x,
    y: point.y,
    z: point.z,
    visibility: 1,
  });

  const landmarks = Array.from({ length: 33 }, () => ({
    x: 0,
    y: 0,
    z: 0,
    visibility: 0,
  }));

  landmarks[0] = make(head);
  landmarks[11] = make(shoulder);
  landmarks[12] = make(shoulder);
  landmarks[13] = make(elbow);
  landmarks[14] = make(elbow);
  landmarks[15] = make(wrist);
  landmarks[16] = make(wrist);
  landmarks[23] = make(hip);
  landmarks[24] = make(hip);
  landmarks[25] = make(knee);
  landmarks[26] = make(knee);
  landmarks[27] = make(ankle);
  landmarks[28] = make(ankle);
  landmarks[29] = make({ x: ankle.x - 0.025, y: ankle.y, z: 0 });
  landmarks[30] = make({ x: ankle.x - 0.025, y: ankle.y, z: 0 });
  landmarks[31] = make(toe);
  landmarks[32] = make(toe);

  return landmarks;
}

function drawDemoFrame(landmarks, metrics, now) {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  drawDemoGrid(now);

  if (!metrics) {
    return;
  }

  const shoulder = toPixel(metrics.shoulder);
  const hip = toPixel(metrics.hip);
  const knee = toPixel(metrics.knee);
  const ankle = toPixel(metrics.ankle);
  const toe = toPixel(metrics.toe);
  const head = toPixel(landmarks[0]);
  const activeView = getActiveView(metrics);
  const color = getProblemColor(metrics, activeView)?.line ?? "#34d399";

  overlayCtx.save();
  overlayCtx.lineCap = "round";
  overlayCtx.lineJoin = "round";
  overlayCtx.strokeStyle = color;
  overlayCtx.fillStyle = color;
  overlayCtx.lineWidth = 6;

  overlayCtx.beginPath();
  overlayCtx.moveTo(head.x, head.y);
  overlayCtx.lineTo(shoulder.x, shoulder.y);
  overlayCtx.lineTo(hip.x, hip.y);
  overlayCtx.lineTo(knee.x, knee.y);
  overlayCtx.lineTo(ankle.x, ankle.y);
  overlayCtx.lineTo(toe.x, toe.y);
  overlayCtx.stroke();

  [head, shoulder, hip, knee, ankle, toe].forEach((point) => {
    overlayCtx.beginPath();
    overlayCtx.arc(point.x, point.y, 7, 0, Math.PI * 2);
    overlayCtx.fill();
  });

  overlayCtx.beginPath();
  overlayCtx.arc(head.x, head.y, 13, 0, Math.PI * 2);
  overlayCtx.stroke();
  overlayCtx.restore();

  drawKneeAngleLabel(metrics);
  drawDemoLabel();
}

function drawDemoGrid(now) {
  overlayCtx.save();
  overlayCtx.strokeStyle = "rgba(255,255,255,.06)";
  overlayCtx.lineWidth = 1;

  const gap = 42;
  for (let x = 0; x < overlay.width; x += gap) {
    overlayCtx.beginPath();
    overlayCtx.moveTo(x, 0);
    overlayCtx.lineTo(x, overlay.height);
    overlayCtx.stroke();
  }

  for (let y = 0; y < overlay.height; y += gap) {
    overlayCtx.beginPath();
    overlayCtx.moveTo(0, y);
    overlayCtx.lineTo(overlay.width, y);
    overlayCtx.stroke();
  }

  const ankleY = overlay.height * 0.93;
  overlayCtx.strokeStyle = "rgba(56,189,248,.28)";
  overlayCtx.setLineDash([6, 8]);
  overlayCtx.beginPath();
  overlayCtx.moveTo(overlay.width * 0.22, ankleY);
  overlayCtx.lineTo(overlay.width * 0.78, ankleY);
  overlayCtx.stroke();
  overlayCtx.setLineDash([]);
  overlayCtx.restore();
}

function drawDemoLabel() {
  overlayCtx.save();
  overlayCtx.font = "600 13px Inter, sans-serif";
  overlayCtx.fillStyle = "rgba(226, 240, 235, .78)";
  overlayCtx.textAlign = "left";
  overlayCtx.textBaseline = "top";
  overlayCtx.fillText("DEMO MODE", 16, 16);
  overlayCtx.restore();
}

viewButtons.forEach((button) => {
  button.addEventListener("click", () => {
    state.view = button.dataset.view;
    updateViewButtons();
    updateViewLabels(null);
  });
});

cameraBtn.addEventListener("click", startCamera);
uploadBtn.addEventListener("click", () => videoInput.click());
demoBtn.addEventListener("click", startDemo);
resetBtn.addEventListener("click", resetSession);
videoInput.addEventListener("change", () => startVideo(videoInput.files[0]));
window.addEventListener("resize", resizeCanvas);


if (new URLSearchParams(window.location.search).has("demo")) {
  initPose()
    .then(() => startDemo())
    .catch((error) => {
      console.error(error);
      setStatus("演示模式启动失败，请刷新页面后重试", true);
    });
}

startLoop();
