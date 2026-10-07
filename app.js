import config from "./config.js";
import { pathCamera, projectPoint, buildDesignExport, buildDesignSvg } from "./presentation.js";
import { panCamera, zoomCamera, createHistory } from "./interaction.js";
import { parseDesignImport, MAX_IMPORT_BYTES } from "./design-import.js";

const JOINTS = ["A", "B", "C", "D", "P"];
const finitePoint = (point) => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite);

/** Public constructed geometry, not the output of inverse synthesis. */
export function constructedPreset(parameters, nFrames = 240) {
  const [r2, r3, r4, px, py] = parameters;
  if (!(Math.abs(1 - r2) > Math.abs(r3 - r4) && 1 + r2 < r3 + r4)) throw new Error("Preset must have strict full-cycle assembly margins.");
  const frames = Array.from({ length: nFrames }, (_, index) => {
    const theta = 2 * Math.PI * index / (nFrames - 1);
    const B = [r2 * Math.cos(theta), r2 * Math.sin(theta)];
    const dx = 1 - B[0], dy = -B[1], distance = Math.hypot(dx, dy);
    const ex = dx / distance, ey = dy / distance;
    const a = (r3 * r3 - r4 * r4 + distance * distance) / (2 * distance);
    const h = Math.sqrt(r3 * r3 - a * a);
    const C = [B[0] + a * ex - h * ey, B[1] + a * ey + h * ex];
    const ux = (C[0] - B[0]) / r3, uy = (C[1] - B[1]) / r3;
    const P = [B[0] + px * ux - py * uy, B[1] + px * uy + py * ux];
    return { A: [0, 0], B, C, D: [1, 0], P };
  });
  return {
    demo_kind: "constructed",
    solutions: [{
      params: { r1: 1, r2, r3, r4, px, py },
      transform: { k: 1, theta: 0, tx: 0, ty: 0 },
      frames, coupler_curve: frames.map((frame) => frame.P),
      error_percent: null,
    }],
  };
}

/** Exact similarity placement; input frame coordinates stay canonical. */
export function transformPoint(point, transform) {
  const { k, theta, tx, ty } = transform;
  const c = Math.cos(theta), s = Math.sin(theta);
  return [tx + k * (c * point[0] - s * point[1]), ty + k * (s * point[0] + c * point[1])];
}

export function transformFrame(frame, transform) {
  return Object.fromEntries(JOINTS.map((name) => [name, transformPoint(frame[name], transform)]));
}

export function validateResult(result) {
  if (!result || !Array.isArray(result.solutions) || !result.solutions.length) {
    throw new Error("No mechanism candidates were returned.");
  }
  const checkedSolutions = [...result.solutions];
  if (result.comparison?.initial_best) checkedSolutions.push(result.comparison.initial_best);
  for (const solution of checkedSolutions) {
    const p = solution.params, t = solution.transform;
    if (!p || !["r1", "r2", "r3", "r4", "px", "py"].every((key) => Number.isFinite(p[key])) ||
        ![p.r1, p.r2, p.r3, p.r4].every((length) => length > 0) ||
        !t || !["k", "theta", "tx", "ty"].every((key) => Number.isFinite(t[key])) || t.k <= 0 ||
        (result.demo_kind !== "constructed" && (!Number.isFinite(solution.error_percent) || solution.error_percent < 0)) ||
        !Array.isArray(solution.frames) || solution.frames.length < 2 ||
        !solution.frames.every((frame) => frame && JOINTS.every((name) => finitePoint(frame[name]))) ||
        !Array.isArray(solution.coupler_curve) || solution.coupler_curve.length < 2 ||
        !solution.coupler_curve.every(finitePoint)) {
      throw new Error("The result does not contain valid four-bar animation data.");
    }
    if (solution.mechanism_family && solution.mechanism_family !== "strict_full_crank_fourbar_branch_plus") {
      throw new Error("Only single-actuator four-bar results are supported.");
    }
    if (!(p.r1 > p.r2 && p.r3 + p.r4 > p.r1 + p.r2 && Math.abs(p.r3 - p.r4) < p.r1 - p.r2)) {
      throw new Error("The mechanism must preserve strict full-cycle assembly.");
    }
  }
  if (result.backend === "browser_onnx_exact_fk" || result.comparison) {
    const c = result.comparison, a = result.audit, r = result.reproducibility;
    const quality = result.profile === "browser_quality_v1";
    if (!c || c.kind !== "same_pool_best" || c.final_best_index !== 0 || !c.initial_best ||
        c.initial_best.source !== "diffusion" || typeof c.refinement_applied !== "boolean" ||
        !a || c.refinement_applied !== (a.refinement_steps > 0) || c.refinement_applied !== quality ||
        !["browser_quality_v1", "browser_fast_v1"].includes(result.profile) ||
        !Number.isFinite(a.raw_best_error_percent) ||
        Math.abs(c.initial_best.error_percent - a.raw_best_error_percent) > 1e-10 ||
        result.solutions[0].error_percent > c.initial_best.error_percent + 1e-10 ||
        (!quality && Math.abs(result.solutions[0].error_percent - c.initial_best.error_percent) > 1e-10)) {
      throw new Error("Invalid same-pool initial/final comparison.");
    }
    if (!r || !Number.isInteger(r.seed) || !/^[a-f0-9]{64}$/i.test(r.model_sha256 || "") ||
        r.requested_profile !== (quality ? "quality_v2" : "fast_v1") ||
        r.demo_version !== "post-submission-v2") {
      throw new Error("Missing browser-run reproducibility metadata.");
    }
  }
  return result;
}

export function solutionWorldData(solution) {
  return {
    frames: solution.frames.map((frame) => transformFrame(frame, solution.transform)),
    curve: solution.coupler_curve.map((point) => transformPoint(point, solution.transform)),
  };
}

export function computeBounds(points) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const point of points) {
    if (!finitePoint(point)) continue;
    minX = Math.min(minX, point[0]); maxX = Math.max(maxX, point[0]);
    minY = Math.min(minY, point[1]); maxY = Math.max(maxY, point[1]);
  }
  if (!Number.isFinite(minX)) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  return { minX, maxX, minY, maxY };
}

/** Pure camera helper exported for CPU-only tests. */
export function fitCamera(points, width, height) {
  const b = computeBounds(points);
  return {
    x: (b.minX + b.maxX) / 2,
    y: (b.minY + b.maxY) / 2,
    scale: Math.min(Math.max(1, width - 100) / Math.max(1e-6, b.maxX - b.minX), Math.max(1, height - 100) / Math.max(1e-6, b.maxY - b.minY)),
  };
}

export function frameIndex(phase, count) {
  return Math.max(0, Math.min(count - 1, Math.round(Math.max(0, Math.min(1, phase)) * (count - 1))));
}

function initializePage() {
  const $ = (id) => document.getElementById(id);
  const canvas = $("mechanism-canvas"), ctx = canvas.getContext("2d");
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const state = {
    mode: "recorded", examples: [], exampleId: null, result: null, active: 0,
    target: [], world: null, selectedWorld: null, initialWorld: null, view: "selected",
    phase: 0, playing: false, speed: 1, lastTime: null,
    width: 1, height: 1, camera: { x: 0, y: 0, scale: 100 },
    drawing: false, pointerId: null, engine: null, engineState: "not-loaded", busy: false,
    requestGeneration: 0, tool: "inspect", pan: null, undo: null,
    metadata: null, replay: null,
  };
  const history = createHistory(5);

  function status(message, error = false) {
    $("viewer-status").textContent = message;
    $("viewer-status").classList.toggle("error", error);
  }

  function empty(title, message) {
    const el = $("canvas-empty");
    el.hidden = false;
    el.querySelector("strong").textContent = title;
    el.querySelector("span:last-child").textContent = message;
  }

  function updateEngineControls() {
    const labels = { "not-loaded": "Model not loaded", loading: "Loading browser model", ready: "Browser model ready", running: "Running in this browser", unsupported: "Browser inference unsupported", failed: "Browser inference failed" };
    $("engine-state").textContent = labels[state.engineState];
    $("engine-indicator").className = `status-dot ${state.engineState}`;
    $("load-model").disabled = state.busy || state.engineState === "ready";
    $("load-model").textContent = state.engineState === "ready" ? "Model loaded" : state.engineState === "failed" ? "Retry loading model" : "Load browser model";
    $("generate").disabled = state.busy || state.engineState !== "ready" || state.target.length < 5;
    $("clear-sketch").disabled = state.busy;
    $("inference-profile").disabled = state.busy;
    $("cancel-inference").hidden = !state.busy;
    for (const id of ["run-example", "use-target", "import-design", "import-file"]) $(id).disabled = state.busy;
    $("replay-design").disabled = state.busy || !state.replay;
    $("undo-sketch").disabled = state.busy || !state.undo;
    $("clear-history").disabled = state.busy || !history.list().length;
    for (const button of $("history-list").querySelectorAll("button")) button.disabled = state.busy;
    $("tool-draw").disabled = state.busy || state.mode !== "browser";
    $("tool-inspect").disabled = state.busy;
    canvas.classList.toggle("drawing-enabled", state.mode === "browser" && state.tool === "draw" && !state.busy);
    canvas.classList.toggle("inspect-enabled", state.tool === "inspect" && !state.busy);
    for (const tool of ["draw", "inspect"]) {
      $(`tool-${tool}`).setAttribute("aria-pressed", String(state.tool === tool));
      $(`tool-${tool}`).classList.toggle("active", state.tool === tool);
    }
    $("interaction-hint").textContent = state.tool === "draw"
      ? "Draw replaces the target. Undo restores the previous sketch and result."
      : "Drag to pan · + / − to zoom · focus canvas and use arrow keys · Fit view to reset.";
  }

  function updatePlayback() {
    const hasFrames = Boolean(state.world?.frames.length);
    $("play-pause").disabled = !hasFrames;
    $("phase").disabled = !hasFrames;
    $("play-pause").textContent = state.playing ? "Pause" : "Play";
    $("play-pause").setAttribute("aria-label", state.playing ? "Pause animation" : "Play animation");
    $("phase").value = String(Math.round(state.phase * 1000));
    $("phase-value").textContent = `${Math.round(state.phase * 360)}°`;
  }

  function clearResult() {
    state.result = null; state.world = null; state.selectedWorld = null; state.initialWorld = null;
    state.view = "selected"; state.active = 0; state.phase = 0; state.playing = false;
    state.replay = null;
    $("candidate-list").replaceChildren();
    $("candidate-empty").hidden = false;
    $("measurement-panel").hidden = true;
    for (const id of ["comparison-panel", "comparison-controls", "initial-overlay-control", "initial-legend-item"]) $(id).hidden = true;
    for (const id of ["export-json", "export-svg", "export-png"]) $(id).disabled = true;
    $("active-view-label").textContent = "Target path";
    updatePlayback();
  }

  function fitView() {
    const points = [...state.target];
    // Keep one camera for the initial/selected toggle, so a scale change cannot
    // masquerade as an improvement in fit.
    for (const world of [state.selectedWorld, state.initialWorld]) {
      if (!world) continue;
      points.push(...world.curve);
      for (const frame of world.frames) points.push(...JOINTS.map((name) => frame[name]));
    }
    state.camera = points.length ? fitCamera(points, state.width, state.height) : { x: 0, y: 0, scale: Math.min(state.width, state.height) / 4 };
    draw();
  }

  function displayParameters(solution) {
    const constructed = state.result?.demo_kind === "constructed";
    const names = [["r1", "r₁ · ground"], ["r2", "r₂ · crank"], ["r3", "r₃ · coupler"], ["r4", "r₄ · rocker"], ["px", "P · local x"], ["py", "P · local y"]];
    $("parameter-list").replaceChildren();
    for (const [key, label] of names) {
      const dt = document.createElement("dt"), dd = document.createElement("dd");
      dt.textContent = label; dd.textContent = Number(solution.params[key]).toPrecision(5);
      $("parameter-list").append(dt, dd);
    }
    $("error-card").hidden = constructed;
    $("metric-note").hidden = constructed;
    $("error-value").textContent = constructed ? "—" : `${solution.error_percent.toFixed(3)}%`;
    $("inference-audit").hidden = constructed;
    $("inference-audit").replaceChildren();
    if (!constructed) {
      const audit = state.result.audit || {}, seconds = state.result.timing?.browser_compute_seconds;
      const imported = Boolean(state.result.imported_record);
      const rows = [
        ...(imported ? [["Record source", "Imported file; execution, budget and timing are not independently verified."]] : []),
        [imported ? "Recorded budget" : "Budget", `${audit.candidates ?? "—"} candidates · ${audit.ddim_steps ?? "—"} DDIM`],
        ["Strictly valid", `${audit.strict_valid_candidates ?? "—"} / ${audit.total_candidates ?? "—"}`],
        [imported ? "Recorded selection FK calls" : "Selection exact-FK calls", String(audit.selection_fk_calls ?? "—")],
        [imported ? "Recorded compute time" : "Browser compute time", Number.isFinite(seconds) ? `${seconds.toFixed(2)} s` : "—"],
        ["Browser seed", String(state.result.reproducibility?.seed ?? "—")],
        ["Model SHA-256", state.result.reproducibility?.model_sha256 ?? "—"],
      ];
      for (const [label, value] of rows) {
        const dt = document.createElement("dt"), dd = document.createElement("dd"); dt.textContent = label; dd.textContent = value; $("inference-audit").append(dt, dd);
      }
    }
    $("measurement-panel").hidden = false;
  }

  function chooseCandidate(index) {
    state.active = index;
    const solution = state.result.solutions[index];
    state.selectedWorld = solutionWorldData(solution);
    [...$("candidate-list").children].forEach((button, i) => {
      button.classList.toggle("active", i === index);
      button.setAttribute("aria-pressed", String(i === index));
    });
    setResultView("selected"); fitView(); updatePlayback();
  }

  function setResultView(view) {
    if (!state.result || (view === "initial" && !state.initialWorld)) return;
    state.view = view;
    state.world = view === "initial" ? state.initialWorld : state.selectedWorld;
    for (const kind of ["selected", "initial"]) {
      $(`view-${kind}`).setAttribute("aria-pressed", String(kind === view));
      $(`view-${kind}`).classList.toggle("active", kind === view);
    }
    $("active-view-label").textContent = view === "initial" ? "Initial best · same generated pool"
      : state.result.demo_kind === "constructed" ? "Known geometry · no inference"
      : `Selected candidate #${state.active + 1}${state.active === 0 ? " · retained best" : " · alternative"}`;
    displayParameters(view === "initial" ? state.result.comparison.initial_best : state.result.solutions[state.active]);
    draw();
  }

  function displayComparison(result) {
    const comparison = result.comparison;
    const available = Boolean(comparison);
    for (const id of ["comparison-panel", "comparison-controls", "initial-overlay-control", "initial-legend-item"]) $(id).hidden = !available;
    state.initialWorld = available ? solutionWorldData(comparison.initial_best) : null;
    if (!available) return;
    const before = comparison.initial_best.error_percent, after = result.solutions[0].error_percent;
    $("initial-error").textContent = `${before.toFixed(3)}%`;
    $("refined-error").textContent = `${after.toFixed(3)}%`;
    $("improvement-value").textContent = !comparison.refinement_applied ? "Not applied"
      : before === 0 ? "—" : `${(100 * (before - after) / before).toFixed(1)}%`;
    $("comparison-note").textContent = comparison.refinement_applied
      ? `Same ${result.audit.candidates}-candidate pool · ${result.audit.selection_fk_calls} selection FK calls · best-to-best Web error, not the selected alternative. Initial candidates remain eligible.`
      : "Fast mode has no refinement. Initial and retained best are the same; this is not a Fast-versus-Quality ablation.";
    if (result.imported_record) $("comparison-note").textContent = `Imported record · pool provenance and budget are file claims. ${$("comparison-note").textContent}`;
  }

  function paintPaths(context, points, camera, color, lineWidth = 2, dash = [], closed = true) {
    if (points.length < 2) return;
    context.beginPath(); context.strokeStyle = color; context.lineWidth = lineWidth; context.setLineDash(dash);
    points.forEach((point, i) => {
      const [x, y] = projectPoint(point, camera, camera.width, camera.height);
      if (i) context.lineTo(x, y); else context.moveTo(x, y);
    });
    if (closed) context.closePath();
    context.stroke(); context.setLineDash([]);
  }

  function paintThumbnail(canvas, solution) {
    canvas.width = 240; canvas.height = 144;
    const context = canvas.getContext("2d"), world = solutionWorldData(solution), frame = world.frames[0];
    const camera = { ...pathCamera([...state.target, ...world.curve, ...JOINTS.map(key => frame[key])], 240, 144, 14), width: 240, height: 144 };
    context.fillStyle = "#f8fbfb"; context.fillRect(0, 0, 240, 144);
    paintPaths(context, state.target, camera, "#d18a39", 1.5, [4, 3]);
    paintPaths(context, world.curve, camera, "#087f83", 2);
    paintPaths(context, [frame.A, frame.B, frame.C, frame.D], camera, "#647a92", 2, [], false);
    paintPaths(context, [frame.B, frame.P, frame.C], camera, "#9aaeb9", 1, [], false);
    for (const key of JOINTS) {
      const [x, y] = projectPoint(frame[key], camera, camera.width, camera.height);
      context.beginPath(); context.arc(x, y, 2.5, 0, 2 * Math.PI);
      context.fillStyle = key === "P" ? "#087f83" : "#647a92"; context.fill();
    }
  }

  function presentResult(result) {
    validateResult(result);
    state.result = result; state.phase = 0; state.tool = "inspect";
    const r = result.reproducibility;
    state.replay = r && state.target.length >= 5 ? { seed: r.seed, profile: r.requested_profile, model_sha256: r.model_sha256 } : null;
    $("candidate-empty").hidden = true;
    $("canvas-empty").hidden = true;
    $("candidate-list").replaceChildren();
    const constructed = result.demo_kind === "constructed";
    displayComparison(result);
    for (const id of ["export-json", "export-svg", "export-png"]) $(id).disabled = false;
    $("candidate-heading").textContent = constructed ? "Known geometry" : "Compare model candidates";
    result.solutions.forEach((solution, index) => {
      const button = document.createElement("button");
      button.type = "button"; button.className = "candidate-button";
      const title = document.createElement("strong"), error = document.createElement("span");
      const thumbnail = document.createElement("canvas");
      thumbnail.className = "candidate-thumb"; thumbnail.setAttribute("aria-hidden", "true");
      paintThumbnail(thumbnail, solution);
      title.textContent = constructed ? "Preset" : `#${index + 1}`; error.textContent = constructed ? "Known geometry" : `${solution.error_percent.toFixed(2)}%`;
      button.append(thumbnail, title, error);
      button.setAttribute("aria-label", constructed ? "Known constructed mechanism, not a prediction" : `Candidate ${index + 1}, web similarity error ${solution.error_percent.toFixed(3)} percent`);
      button.addEventListener("click", () => chooseCandidate(index));
      $("candidate-list").append(button);
    });
    state.playing = !reducedMotion; state.lastTime = null;
    chooseCandidate(0); updateEngineControls();
  }

  function selectExample(example) {
    state.requestGeneration += 1;
    state.exampleId = example.id;
    state.target = example.points.map((point) => [...point]);
    $("result-kind").textContent = "CONSTRUCTED PRESET · NO MODEL INFERENCE";
    $("result-title").textContent = example.title;
    [...$("example-list").children].forEach((button) => {
      button.classList.toggle("active", button.dataset.exampleId === example.id);
      button.setAttribute("aria-pressed", String(button.dataset.exampleId === example.id));
    });
    try {
      presentResult(example.result);
      status("Known mechanism, constructed by exact forward kinematics. This is not a model result and has no synthesis error score. Choose Use this target to run actual inference.");
    } catch {
      clearResult(); empty("Preset unavailable", "The geometry did not pass the animation-data checks.");
      status("Invalid constructed preset. No replacement result has been displayed.", true);
    }
  }

  function setMode(mode) {
    if (state.mode === mode) return;
    if (state.busy) cancelComputation();
    state.requestGeneration += 1;
    state.mode = mode; state.target = []; state.drawing = false; state.pan = null;
    state.tool = mode === "browser" ? "draw" : "inspect";
    state.undo = null;
    if (state.pointerId !== null && canvas.hasPointerCapture(state.pointerId)) canvas.releasePointerCapture(state.pointerId);
    state.pointerId = null;
    clearResult();
    for (const name of ["recorded", "browser"]) {
      $(`${name}-panel`).hidden = name !== mode;
      $(`mode-${name}`).classList.toggle("active", name === mode);
      $(`mode-${name}`).setAttribute("aria-pressed", String(name === mode));
    }
    $("drawing-hint").hidden = mode !== "browser";
    if (mode === "recorded") {
      const example = state.examples.find((item) => item.id === state.exampleId) || state.examples[0];
      if (example) selectExample(example);
      else { empty("No preset loaded", "Known mechanisms are currently unavailable."); status("No constructed preset is available.", true); }
    } else {
      $("result-kind").textContent = "BROWSER INFERENCE · EXPLICIT MODEL LOAD REQUIRED";
      $("result-title").textContent = "Your target path";
      $("canvas-empty").hidden = true;
      status("Draw a path, then load the browser model. Generate is available only when the model is ready.");
      fitView();
    }
    updateEngineControls(); draw();
  }

  const screenPoint = (point) => [(point[0] - state.camera.x) * state.camera.scale + state.width / 2, state.height / 2 - (point[1] - state.camera.y) * state.camera.scale];
  function path(points, color, width, dash = [], closed = false) {
    if (points.length < 2) return;
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
    points.forEach((point, index) => { const [x, y] = screenPoint(point); if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    if (closed) ctx.closePath();
    ctx.stroke(); ctx.setLineDash([]);
  }

  function joint(point, radius, color, label) {
    const [x, y] = screenPoint(point);
    ctx.beginPath(); ctx.arc(x, y, radius, 0, 2 * Math.PI); ctx.fillStyle = "#fff"; ctx.fill();
    ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.stroke();
    if (label) { ctx.font = "11px ui-sans-serif, system-ui, sans-serif"; ctx.fillStyle = color; ctx.fillText(label, x + 10, y - 9); }
  }

  function draw() {
    ctx.clearRect(0, 0, state.width, state.height);
    ctx.fillStyle = "#e1e8ed";
    for (let x = 22; x < state.width; x += 26) for (let y = 22; y < state.height; y += 26) { ctx.beginPath(); ctx.arc(x, y, .8, 0, Math.PI * 2); ctx.fill(); }
    if ($("show-target").checked) path(state.target, "#d18a39", 1.7, [6, 5], !state.drawing);
    drawPathCloseup();
    if (!state.world) return;
    if (state.initialWorld && state.view === "selected" && $("show-initial").checked) path(state.initialWorld.curve, "#ce765c", 1.7, [3, 4], true);
    path(state.world.curve, state.view === "initial" ? "#ce765c" : "#087f83", 2.6, [], true);
    const frame = state.world.frames[frameIndex(state.phase, state.world.frames.length)];
    if ($("show-mechanism").checked) {
      const shade = state.view === "initial" ? "#b68479" : "#526b94";
      const vertices = [frame.B, frame.C, frame.P].map(screenPoint);
      ctx.beginPath(); vertices.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
      ctx.closePath(); ctx.fillStyle = "#66859312"; ctx.fill();
      for (const anchor of [frame.A, frame.D]) {
        const [x, y] = screenPoint(anchor);
        ctx.fillStyle = "#e3e9ed"; ctx.fillRect(x - 11, y - 5, 22, 15);
        ctx.strokeStyle = "#afbdc6"; ctx.lineWidth = 1; ctx.strokeRect(x - 11, y - 5, 22, 15);
      }
      path([frame.A, frame.D], "#b6c2cc", 3, [3, 4]);
      path([frame.A, frame.B], shade, 5);
      path([frame.B, frame.C], shade, 5);
      path([frame.C, frame.D], shade, 5);
      path([frame.B, frame.P, frame.C], "#93a8bc", 1.2);
      joint(frame.A, 5.5, "#526b94", "A"); joint(frame.D, 5.5, "#526b94", "D");
      joint(frame.B, 4, "#526b94", "B"); joint(frame.C, 4, "#526b94", "C");
    }
    joint(frame.P, 4.5, "#087f83", "P");
  }

  function drawPathCloseup() {
    const preview = $("path-canvas"), context = preview.getContext("2d");
    const box = preview.getBoundingClientRect(), width = Math.max(1, box.width), height = Math.max(1, box.height);
    const ratio = Math.min(window.devicePixelRatio || 1, 3);
    if (preview.width !== Math.round(width * ratio) || preview.height !== Math.round(height * ratio)) {
      preview.width = Math.round(width * ratio); preview.height = Math.round(height * ratio);
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    const points = [...state.target, ...(state.selectedWorld?.curve || []), ...(state.initialWorld?.curve || [])];
    if (!points.length) {
      context.font = "12px ui-sans-serif, system-ui, sans-serif"; context.fillStyle = "#82949c";
      context.textAlign = "center"; context.fillText("Your path appears here", width / 2, height / 2); return;
    }
    const camera = { ...pathCamera(points, width, height, 22), width, height };
    if ($("show-target").checked) paintPaths(context, state.target, camera, "#d18a39", 2, [5, 4], !state.drawing);
    if (state.initialWorld && (state.view === "initial" || $("show-initial").checked)) {
      paintPaths(context, state.initialWorld.curve, camera, "#ce765c", 1.7, [3, 3]);
    }
    if (state.view !== "initial" && state.selectedWorld) paintPaths(context, state.selectedWorld.curve, camera, "#087f83", 2.2);
    if (state.world) {
      const point = state.world.frames[frameIndex(state.phase, state.world.frames.length)].P;
      const [x, y] = projectPoint(point, camera, width, height);
      context.beginPath(); context.arc(x, y, 3.5, 0, 2 * Math.PI); context.fillStyle = state.view === "initial" ? "#ce765c" : "#087f83"; context.fill();
    }
  }

  function resize() {
    const box = canvas.getBoundingClientRect(), ratio = Math.min(window.devicePixelRatio || 1, 3);
    state.width = Math.max(1, box.width); state.height = Math.max(1, box.height);
    canvas.width = Math.round(state.width * ratio); canvas.height = Math.round(state.height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    fitView();
  }

  function tick(time) {
    if (state.playing && state.world && state.lastTime !== null) {
      const elapsed = Math.min(.1, Math.max(0, (time - state.lastTime) / 1000));
      state.phase = (state.phase + elapsed * state.speed / config.playbackSeconds) % 1;
      updatePlayback(); draw();
    }
    state.lastTime = time;
    window.requestAnimationFrame(tick);
  }

  function inputPoint(event) {
    const box = canvas.getBoundingClientRect();
    return [state.camera.x + (event.clientX - box.left - state.width / 2) / state.camera.scale, state.camera.y - (event.clientY - box.top - state.height / 2) / state.camera.scale];
  }

  function snapshotDesign() {
    return { mode: state.mode, target: state.target.map(p => [...p]), result: state.result, active: state.active,
      phase: state.phase, replay: state.replay, camera: { ...state.camera },
      kind: $("result-kind").textContent, title: $("result-title").textContent,
      profile: $("inference-profile").value, tool: state.tool };
  }

  function restoreDesign(record, fromHistory = false) {
    if (state.busy) return;
    setMode(record.mode || "browser"); state.requestGeneration += 1; clearResult();
    state.target = record.target.map(p => [...p]);
    if (record.result) {
      presentResult(record.result); chooseCandidate(record.active);
      state.phase = record.phase; state.playing = false;
    }
    state.replay = record.replay;
    state.tool = record.result ? "inspect" : record.tool;
    $("inference-profile").value = record.profile;
    $("result-kind").textContent = fromHistory ? `SESSION HISTORY · ${record.kind}` : record.kind;
    $("result-title").textContent = record.title;
    $("drawing-hint").hidden = state.tool !== "draw" || state.target.length > 0;
    $("canvas-empty").hidden = true;
    fitView();
    if (!fromHistory) state.camera = { ...record.camera };
    updateEngineControls(); updatePlayback(); draw();
    status(fromHistory ? "Restored a result from this tab. No new inference was run. Replay explicitly runs the saved target, seed and budget."
      : "Previous sketch and result restored. No new inference was run.");
  }

  function rememberDesign() {
    const record = snapshotDesign();
    record.kind = record.result.imported_record ? "IMPORTED FILE · NOT NEW INFERENCE" : "COMPUTED IN THIS TAB";
    history.push(record); renderHistory();
  }

  function renderHistory() {
    const entries = history.list(); $("history-list").replaceChildren();
    $("history-empty").hidden = entries.length > 0;
    for (const { id, record } of entries) {
      const button = document.createElement("button"), title = document.createElement("strong"), detail = document.createElement("span");
      button.type = "button"; button.className = "history-button";
      const result = record.result;
      title.textContent = `#${id} · ${result.imported_record ? "Imported" : "Computed"} · ${result.profile === "browser_quality_v1" ? "Quality" : result.demo_kind === "constructed" ? "Preset" : "Fast"}`;
      detail.textContent = result.demo_kind === "constructed" ? "Known geometry · unscored" : `Best web error ${result.solutions[0].error_percent.toFixed(3)}% · seed ${result.reproducibility.seed}`;
      button.append(title, detail);
      button.addEventListener("click", () => {
        const saved = history.get(id); if (!saved || state.busy) return;
        const previous = snapshotDesign(); restoreDesign(saved, true); state.undo = previous; updateEngineControls();
      });
      $("history-list").append(button);
    }
    updateEngineControls();
  }

  function setTool(tool) {
    if (state.busy || state.drawing || (tool === "draw" && state.mode !== "browser")) return;
    state.tool = tool; state.pan = null;
    $("drawing-hint").hidden = tool !== "draw" || state.target.length > 0;
    updateEngineControls();
  }

  function zoom(factor, anchor = [state.width / 2, state.height / 2]) {
    if (state.drawing) return;
    state.camera = zoomCamera(state.camera, factor, anchor, state.width, state.height); draw();
  }
  $("tool-draw").addEventListener("click", () => setTool("draw"));
  $("tool-inspect").addEventListener("click", () => setTool("inspect"));
  $("zoom-in").addEventListener("click", () => zoom(1.25));
  $("zoom-out").addEventListener("click", () => zoom(1 / 1.25));
  $("undo-sketch").addEventListener("click", () => {
    if (!state.undo || state.busy) return;
    const saved = state.undo; restoreDesign(saved); state.undo = null; updateEngineControls();
  });
  $("clear-history").addEventListener("click", () => {
    if (state.busy) return;
    history.clear(); renderHistory(); status("Session history cleared. The open design and exported files are unchanged.");
  });

  canvas.addEventListener("pointerdown", (event) => {
    if (state.busy || !event.isPrimary || (event.pointerType === "mouse" && event.button !== 0)) return;
    event.preventDefault(); canvas.focus({ preventScroll: true });
    if (state.tool === "inspect") {
      state.pan = { x: event.clientX, y: event.clientY }; state.pointerId = event.pointerId;
      canvas.setPointerCapture(event.pointerId); return;
    }
    if (state.mode !== "browser") return;
    state.undo = snapshotDesign(); state.requestGeneration += 1; clearResult();
    // Start each replacement stroke in the same coordinate system as Clear.
    state.camera = { x: 0, y: 0, scale: Math.min(state.width, state.height) / 4 };
    state.drawing = true; state.pointerId = event.pointerId; state.target = [inputPoint(event)];
    canvas.setPointerCapture(event.pointerId);
    $("canvas-empty").hidden = true; $("drawing-hint").hidden = true;
    $("result-kind").textContent = "NEW SKETCH · NOT YET SYNTHESIZED";
    $("result-title").textContent = "Your target path";
    status("Drawing a target path. This sketch has not been synthesized."); updateEngineControls(); draw();
  });
  canvas.addEventListener("pointermove", (event) => {
    if (state.pan && event.pointerId === state.pointerId) {
      event.preventDefault();
      state.camera = panCamera(state.camera, event.clientX - state.pan.x, event.clientY - state.pan.y);
      state.pan = { x: event.clientX, y: event.clientY }; draw(); return;
    }
    if (!state.drawing || event.pointerId !== state.pointerId) return;
    event.preventDefault();
    const point = inputPoint(event), last = state.target[state.target.length - 1];
    if (Math.hypot(point[0] - last[0], point[1] - last[1]) * state.camera.scale >= 2 && state.target.length < 4096) state.target.push(point);
    draw();
  });
  function finishStroke(event) {
    if (state.pan && event.pointerId === state.pointerId) {
      state.pan = null; state.pointerId = null;
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
      return;
    }
    if (!state.drawing || event.pointerId !== state.pointerId) return;
    state.drawing = false;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    state.pointerId = null;
    if ((event.type === "pointercancel" || state.target.length < 5) && state.undo) {
      const saved = state.undo; restoreDesign(saved); state.undo = null; updateEngineControls();
      status("Incomplete stroke discarded. The previous sketch and result are unchanged."); return;
    }
    state.tool = "inspect";
    updateEngineControls(); draw();
    status(state.target.length < 5 ? "The stroke is too short. Please draw a longer closed path." : "Sketch captured. The closing segment is shown; choose Generate when the browser model is ready.");
  }
  canvas.addEventListener("pointerup", finishStroke);
  canvas.addEventListener("pointercancel", finishStroke);
  canvas.addEventListener("lostpointercapture", (event) => {
    if (state.pointerId === event.pointerId) finishStroke({ pointerId: event.pointerId, type: "pointercancel" });
  });
  canvas.addEventListener("wheel", (event) => {
    // Do not trap page scrolling while drawing; inspect zoom is canvas-local.
    if (state.tool !== "inspect" || state.busy) return;
    event.preventDefault(); const box = canvas.getBoundingClientRect();
    zoom(Math.exp(-Math.max(-100, Math.min(100, event.deltaY)) * .002), [event.clientX - box.left, event.clientY - box.top]);
  }, { passive: false });
  canvas.addEventListener("keydown", (event) => {
    if (state.busy || state.drawing) return;
    const shifts = { ArrowLeft: [-24, 0], ArrowRight: [24, 0], ArrowUp: [0, -24], ArrowDown: [0, 24] };
    if (event.key === "+" || event.key === "=") zoom(1.25);
    else if (event.key === "-") zoom(1 / 1.25);
    else if (event.key === "0") fitView();
    else if (shifts[event.key]) { state.camera = panCamera(state.camera, ...shifts[event.key]); draw(); }
    else return;
    event.preventDefault();
  });

  function progressMessage(progress) {
    if (typeof progress === "string") return progress;
    if (progress && typeof progress.message === "string") return progress.message;
    return "Working in this browser…";
  }

  function engineFailure(error) {
    const unsupported = error?.code === "UNSUPPORTED" || error?.name === "NotSupportedError";
    state.engineState = unsupported ? "unsupported" : "failed";
    $("engine-detail").textContent = unsupported
      ? "This browser or device cannot run the current model. You can explicitly switch to Known mechanisms for a constructed demonstration."
      : "The browser model could not complete this operation. Check support and available memory, then retry. No known mechanism has been substituted.";
    return unsupported;
  }

  function cancelComputation() {
    state.requestGeneration += 1;
    state.engine?.cancelSynthesis?.();
    state.busy = false; state.engineState = "not-loaded"; state.metadata = null;
    $("engine-detail").textContent = "Cancelled. The computation worker was stopped; load the model again to continue.";
    if (!state.result) $("result-kind").textContent = "BROWSER INFERENCE · CANCELLED";
    updateEngineControls(); status("Computation cancelled. No replacement result was generated.");
  }

  $("cancel-inference").addEventListener("click", cancelComputation);

  async function loadEngine(generation) {
    if (state.engineState === "ready" && state.metadata) return;
    state.engineState = "loading"; updateEngineControls();
    $("engine-detail").textContent = "Loading the local inference module and checking browser support…";
    const engine = await import("./browser-engine.js");
    if (generation !== state.requestGeneration) return;
    if (typeof engine.initializeEngine !== "function" || typeof engine.synthesize !== "function") throw new Error("Invalid browser engine interface.");
    state.engine = engine;
    const metadata = await engine.initializeEngine((progress) => { if (generation === state.requestGeneration) $("engine-detail").textContent = progressMessage(progress); });
    if (generation !== state.requestGeneration) return;
    state.metadata = metadata; state.engineState = "ready";
    $("engine-detail").textContent = "Model loaded in this browser. Choose Generate to run inference.";
  }

  $("load-model").addEventListener("click", async () => {
    if (state.busy) return;
    const generation = ++state.requestGeneration;
    state.busy = true;
    try {
      await loadEngine(generation);
    } catch (error) {
      if (generation === state.requestGeneration) engineFailure(error);
    } finally {
      if (generation === state.requestGeneration) { state.busy = false; updateEngineControls(); }
    }
  });

  async function runInference({ autoLoad = false, replay = null, previousDesign = null } = {}) {
    if (state.busy || (!autoLoad && state.engineState !== "ready") || state.target.length < 5 || state.mode !== "browser") return;
    const generation = ++state.requestGeneration;
    const points = state.target.map((point) => [...point]);
    const options = { ...config.inferenceOptions, profile: replay?.profile ?? $("inference-profile").value };
    if (replay) options.seed = replay.seed;
    const previous = previousDesign || snapshotDesign();
    state.undo = previous;
    state.busy = true; updateEngineControls();
    try {
      if (autoLoad) await loadEngine(generation);
      if (generation !== state.requestGeneration) return;
      if (replay && state.metadata?.export_sha256 !== replay.model_sha256) {
        status("Replay stopped: the saved model hash does not match the verified local model. The saved design remains open; no inference was run.", true);
        return;
      }
      state.undo = previous;
      clearResult(); state.tool = "inspect"; state.engineState = "running"; updateEngineControls();
      $("result-kind").textContent = "BROWSER INFERENCE · COMPUTING";
      status(replay ? "Recomputing the saved target, seed and budget with the verified matching model. Browser versions may differ slightly."
        : "Running the model in this browser on this device’s CPU. Known mechanisms will not be used as a fallback.");
      const result = await state.engine.synthesize(points, options, (progress) => {
        if (generation !== state.requestGeneration) return;
        $("engine-detail").textContent = progressMessage(progress);
        if (generation === state.requestGeneration && state.mode === "browser") status(progressMessage(progress));
      });
      if (generation !== state.requestGeneration || state.mode !== "browser") return;
      validateResult(result);
      state.engineState = "ready";
      $("engine-detail").textContent = "Browser inference completed. The returned mechanism candidates are ready to inspect.";
      if (generation !== state.requestGeneration || state.mode !== "browser") return;
      presentResult(result);
      $("result-kind").textContent = "COMPUTED IN THIS BROWSER · NOT A RECORDING";
      $("result-title").textContent = "Your synthesized four-bar candidates";
      status("Computed in this browser for the current sketch. Solid lines show the returned exact-kinematic trajectories; error is the Web metric, not the paper benchmark.");
      rememberDesign();
    } catch (error) {
      if (generation === state.requestGeneration && state.mode === "browser") {
        engineFailure(error);
        if (!state.result) $("result-kind").textContent = "BROWSER INFERENCE · NO RESULT";
        status(state.result ? "Browser inference did not complete. The saved design remains open; no new result was generated."
          : "Browser inference did not complete. No new result was generated. You can undo to restore the previous design.", true);
      }
    } finally {
      if (generation === state.requestGeneration) { state.busy = false; updateEngineControls(); draw(); }
    }
  }
  $("generate").addEventListener("click", () => runInference());
  $("run-example").addEventListener("click", () => {
    if (state.busy) return;
    const previousDesign = snapshotDesign();
    const target = state.target.map(point => [...point]);
    setMode("browser"); state.target = target; state.tool = "inspect";
    $("inference-profile").value = "quality_v2";
    $("drawing-hint").hidden = true;
    $("result-kind").textContent = "KNOWN TARGET · LOADING FOR REAL INFERENCE";
    fitView(); runInference({ autoLoad: true, previousDesign });
  });
  $("replay-design").addEventListener("click", () => {
    if (!state.replay || state.busy) return;
    const replay = { ...state.replay }; $("inference-profile").value = replay.profile;
    runInference({ autoLoad: true, replay });
  });

  $("import-design").addEventListener("click", () => { if (!state.busy) $("import-file").click(); });
  $("import-file").addEventListener("change", async () => {
    const file = $("import-file").files[0]; $("import-file").value = "";
    if (!file || state.busy) return;
    const generation = ++state.requestGeneration;
    try {
      if (file.size > MAX_IMPORT_BYTES) throw new Error(`File exceeds the ${Math.round(MAX_IMPORT_BYTES / 1024 / 1024)} MB import limit.`);
      const imported = parseDesignImport(await file.text());
      if (generation !== state.requestGeneration || state.busy) return;
      validateResult(imported.result);
      const previous = snapshotDesign();
      setMode("browser"); state.requestGeneration += 1;
      state.target = imported.target; presentResult(imported.result);
      chooseCandidate(imported.selectedIndex); state.phase = imported.phase; state.playing = false;
      state.undo = previous; state.replay = imported.replay;
      if (imported.replay) $("inference-profile").value = imported.replay.profile;
      $("drawing-hint").hidden = true;
      $("result-kind").textContent = "IMPORTED FILE · NOT NEW INFERENCE";
      $("result-title").textContent = "Your saved four-bar design";
      status("Imported locally. Geometry checked against exact FK; run metadata comes from the file, not proof of a new run. No file was uploaded.");
      rememberDesign(); updatePlayback(); updateEngineControls(); draw();
    } catch (error) {
      if (generation === state.requestGeneration && !state.busy) status(`Import rejected: ${error.message} The current design is unchanged.`, true);
    }
  });

  $("clear-sketch").addEventListener("click", () => {
    if (state.busy) return;
    state.undo = snapshotDesign(); state.tool = "draw";
    state.requestGeneration += 1; state.target = []; clearResult();
    $("drawing-hint").hidden = false;
    $("result-kind").textContent = "NEW SKETCH · NOT YET SYNTHESIZED";
    $("result-title").textContent = "Your target path";
    status("Canvas cleared. Draw one continuous closed path."); updateEngineControls(); fitView();
  });
  $("mode-recorded").addEventListener("click", () => setMode("recorded"));
  $("mode-browser").addEventListener("click", () => setMode("browser"));
  $("use-target").addEventListener("click", () => {
    const target = state.target.map((point) => [...point]);
    setMode("browser"); state.target = target;
    state.tool = "inspect";
    $("drawing-hint").hidden = true;
    $("result-kind").textContent = "KNOWN TARGET · AWAITING MODEL INFERENCE";
    status("The constructed path is now an inference target. No known mechanism is supplied to the model; Generate must produce new candidates.");
    updateEngineControls(); fitView();
  });
  $("fit-view").addEventListener("click", fitView);
  $("view-selected").addEventListener("click", () => setResultView("selected"));
  $("view-initial").addEventListener("click", () => setResultView("initial"));
  $("play-pause").addEventListener("click", () => { state.playing = !state.playing; state.lastTime = null; updatePlayback(); });
  $("phase").addEventListener("input", () => { state.phase = Number($("phase").value) / 1000; state.playing = false; updatePlayback(); draw(); });
  $("speed").addEventListener("change", () => { state.speed = Number($("speed").value); });
  for (const id of ["show-mechanism", "show-target", "show-initial"]) $(id).addEventListener("change", draw);

  function saveBlob(blob, filename) {
    const url = URL.createObjectURL(blob), link = document.createElement("a");
    link.href = url; link.download = filename; document.body.append(link); link.click(); link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function exportDesign(format) {
    if (!state.result) return;
    const result = state.result, selectedIndex = state.active, generation = state.requestGeneration;
    const options = { target: state.target.map(point => [...point]), selectedIndex, phase: state.phase, showInitial: $("show-initial").checked };
    const name = `fourbar-${result.demo_kind === "constructed" ? "preset" : `candidate-${selectedIndex + 1}`}`;
    try {
      if (format === "json") {
        saveBlob(new Blob([JSON.stringify(buildDesignExport(result, options), null, 2)], { type: "application/json" }), `${name}.json`);
      } else {
        const svg = buildDesignSvg(result, options), blob = new Blob([svg], { type: "image/svg+xml" });
        if (format === "svg") saveBlob(blob, `${name}.svg`);
        else {
          const url = URL.createObjectURL(blob);
          try {
            const img = new Image();
            await new Promise((resolve, reject) => { img.onload = resolve; img.onerror = () => reject(new Error("Could not render the export image.")); img.src = url; });
            const output = document.createElement("canvas"); output.width = img.naturalWidth * 2; output.height = img.naturalHeight * 2;
            output.getContext("2d").drawImage(img, 0, 0, output.width, output.height);
            const png = await new Promise(resolve => output.toBlob(resolve, "image/png"));
            if (!png) throw new Error("Image export is unavailable in this browser.");
            saveBlob(png, `${name}.png`);
          } finally { URL.revokeObjectURL(url); }
        }
      }
      if (generation === state.requestGeneration && result === state.result && !state.busy) {
        status(`Exported ${name}.${format}. Geometry-only design; exports describe the selected candidate, not the initial-view toggle.`);
      }
    } catch (error) { if (generation === state.requestGeneration && result === state.result && !state.busy) status(`Export failed: ${error.message}`, true); }
  }
  for (const format of ["json", "svg", "png"]) $(`export-${format}`).addEventListener("click", () => exportDesign(format));
  new ResizeObserver(drawPathCloseup).observe($("path-canvas"));
  new ResizeObserver(resize).observe(canvas);
  updateEngineControls(); resize(); window.requestAnimationFrame(tick);

  function loadExamples() {
    try {
      state.examples = [
        { id: "compact_loop", title: "Compact loop", description: "A compact, smooth closed trajectory.", parameters: [0.42, 1.35, 1.25, 0.24, -0.31] },
        { id: "foot_loop", title: "Foot loop", description: "An elongated loop with a flatter lower sweep.", parameters: [0.24, 1.15, 1.02, 1.45, -0.55] },
      ].map((example) => {
        const result = constructedPreset(example.parameters);
        return { ...example, result, points: result.solutions[0].coupler_curve };
      });
      $("example-list").replaceChildren();
      for (const example of state.examples) {
        const button = document.createElement("button"), title = document.createElement("strong"), description = document.createElement("span");
        button.type = "button"; button.className = "example-button"; button.dataset.exampleId = example.id;
        title.textContent = example.title; description.textContent = example.description || "Known four-bar geometry";
        button.append(title, description); button.addEventListener("click", () => selectExample(example));
        $("example-list").append(button);
      }
      if (state.mode === "recorded") selectExample(state.examples.find((example) => example.id === config.startupExample) || state.examples[0]);
    } catch {
      $("example-list").replaceChildren();
      const message = document.createElement("p"); message.className = "help"; message.textContent = "Constructed presets are unavailable."; $("example-list").append(message);
      if (state.mode === "recorded") { empty("Presets unavailable", "Browser inference can be checked separately."); status("Constructed presets could not be loaded. Select Draw a path to check browser inference support.", true); }
    }
  }
  loadExamples();
}

// Importing this module in Node only exposes pure validation/geometry helpers.
if (typeof document !== "undefined") initializePage();
