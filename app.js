import config from "./config.js";

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
  for (const solution of result.solutions) {
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
    scale: Math.max(0.001, Math.min(Math.max(1, width - 100) / Math.max(1e-6, b.maxX - b.minX), Math.max(1, height - 100) / Math.max(1e-6, b.maxY - b.minY))),
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
    target: [], world: null, phase: 0, playing: false, speed: 1, lastTime: null,
    width: 1, height: 1, camera: { x: 0, y: 0, scale: 100 },
    drawing: false, pointerId: null, engine: null, engineState: "not-loaded", busy: false,
    requestGeneration: 0,
  };

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
    canvas.classList.toggle("drawing-enabled", state.mode === "browser" && !state.busy);
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
    state.result = null; state.world = null; state.active = 0; state.phase = 0; state.playing = false;
    $("candidate-list").replaceChildren();
    $("candidate-empty").hidden = false;
    $("measurement-panel").hidden = true;
    updatePlayback();
  }

  function fitView() {
    const points = [...state.target];
    if (state.world) {
      points.push(...state.world.curve);
      for (const frame of state.world.frames) points.push(...JOINTS.map((name) => frame[name]));
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
      const rows = [
        ["Budget", `${audit.candidates ?? "—"} candidates · ${audit.ddim_steps ?? "—"} DDIM`],
        ["Strictly valid", `${audit.strict_valid_candidates ?? "—"} / ${audit.total_candidates ?? "—"}`],
        ["Selection exact-FK calls", String(audit.selection_fk_calls ?? "—")],
        ["Browser compute time", Number.isFinite(seconds) ? `${seconds.toFixed(2)} s` : "—"],
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
    state.world = solutionWorldData(solution);
    [...$("candidate-list").children].forEach((button, i) => {
      button.classList.toggle("active", i === index);
      button.setAttribute("aria-pressed", String(i === index));
    });
    displayParameters(solution); fitView(); updatePlayback();
  }

  function presentResult(result) {
    validateResult(result);
    state.result = result; state.phase = 0;
    $("candidate-empty").hidden = true;
    $("canvas-empty").hidden = true;
    $("candidate-list").replaceChildren();
    const constructed = result.demo_kind === "constructed";
    $("candidate-heading").textContent = constructed ? "Known geometry" : "Compare model candidates";
    result.solutions.forEach((solution, index) => {
      const button = document.createElement("button");
      button.type = "button"; button.className = "candidate-button";
      const title = document.createElement("strong"), error = document.createElement("span");
      title.textContent = constructed ? "Preset" : `#${index + 1}`; error.textContent = constructed ? "Known geometry" : `${solution.error_percent.toFixed(2)}%`;
      button.append(title, error);
      button.setAttribute("aria-label", constructed ? "Known constructed mechanism, not a prediction" : `Candidate ${index + 1}, web similarity error ${solution.error_percent.toFixed(3)} percent`);
      button.addEventListener("click", () => chooseCandidate(index));
      $("candidate-list").append(button);
    });
    state.playing = !reducedMotion; state.lastTime = null;
    chooseCandidate(0);
  }

  function selectExample(example) {
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
    state.requestGeneration += 1;
    state.mode = mode; state.target = []; state.drawing = false;
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
    if (!state.world) return;
    path(state.world.curve, "#087f83", 2.4, [], true);
    const frame = state.world.frames[frameIndex(state.phase, state.world.frames.length)];
    if ($("show-mechanism").checked) {
      path([frame.A, frame.D], "#b6c2cc", 3, [3, 4]);
      path([frame.A, frame.B], "#526b94", 4);
      path([frame.B, frame.C], "#526b94", 4);
      path([frame.C, frame.D], "#526b94", 4);
      path([frame.B, frame.P, frame.C], "#93a8bc", 1.2);
      joint(frame.A, 5.5, "#526b94", "A"); joint(frame.D, 5.5, "#526b94", "D");
      joint(frame.B, 4, "#526b94", "B"); joint(frame.C, 4, "#526b94", "C");
    }
    joint(frame.P, 4.5, "#087f83", "P");
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

  canvas.addEventListener("pointerdown", (event) => {
    if (state.mode !== "browser" || state.busy || !event.isPrimary || (event.pointerType === "mouse" && event.button !== 0)) return;
    event.preventDefault(); clearResult();
    state.drawing = true; state.pointerId = event.pointerId; state.target = [inputPoint(event)];
    canvas.setPointerCapture(event.pointerId);
    $("canvas-empty").hidden = true; $("drawing-hint").hidden = true;
    $("result-kind").textContent = "NEW SKETCH · NOT YET SYNTHESIZED";
    $("result-title").textContent = "Your target path";
    status("Drawing a target path. This sketch has not been synthesized."); updateEngineControls(); draw();
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!state.drawing || event.pointerId !== state.pointerId) return;
    event.preventDefault();
    const point = inputPoint(event), last = state.target[state.target.length - 1];
    if (Math.hypot(point[0] - last[0], point[1] - last[1]) * state.camera.scale >= 2 && state.target.length < 4096) state.target.push(point);
    draw();
  });
  function finishStroke(event) {
    if (!state.drawing || event.pointerId !== state.pointerId) return;
    state.drawing = false;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    state.pointerId = null;
    updateEngineControls(); draw();
    status(state.target.length < 5 ? "The stroke is too short. Please draw a longer closed path." : "Sketch captured. The closing segment is shown; choose Generate when the browser model is ready.");
  }
  canvas.addEventListener("pointerup", finishStroke);
  canvas.addEventListener("pointercancel", finishStroke);

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

  $("load-model").addEventListener("click", async () => {
    if (state.busy) return;
    state.busy = true; state.engineState = "loading"; updateEngineControls();
    $("engine-detail").textContent = "Loading the local inference module and checking browser support…";
    try {
      const engine = await import("./browser-engine.js");
      if (typeof engine.initializeEngine !== "function" || typeof engine.synthesize !== "function") throw new Error("Invalid browser engine interface.");
      await engine.initializeEngine((progress) => { $("engine-detail").textContent = progressMessage(progress); });
      state.engine = engine; state.engineState = "ready";
      $("engine-detail").textContent = "Model loaded in this browser. Draw a path and choose Generate to run inference.";
    } catch (error) {
      engineFailure(error);
    } finally {
      state.busy = false; updateEngineControls();
    }
  });

  $("generate").addEventListener("click", async () => {
    if (state.busy || state.engineState !== "ready" || state.target.length < 5 || state.mode !== "browser") return;
    const generation = ++state.requestGeneration;
    const points = state.target.map((point) => [...point]);
    clearResult(); state.busy = true; state.engineState = "running"; updateEngineControls();
    $("result-kind").textContent = "BROWSER INFERENCE · COMPUTING";
    status("Running the model in this browser on this device’s CPU. Known mechanisms will not be used as a fallback.");
    try {
      const result = await state.engine.synthesize(points, { ...config.inferenceOptions, profile: $("inference-profile").value }, (progress) => {
        $("engine-detail").textContent = progressMessage(progress);
        if (generation === state.requestGeneration && state.mode === "browser") status(progressMessage(progress));
      });
      validateResult(result);
      state.engineState = "ready";
      $("engine-detail").textContent = "Browser inference completed. The returned mechanism candidates are ready to inspect.";
      if (generation !== state.requestGeneration || state.mode !== "browser") return;
      presentResult(result);
      $("result-kind").textContent = "COMPUTED IN THIS BROWSER · NOT A RECORDING";
      $("result-title").textContent = "Your synthesized four-bar candidates";
      status("Computed in this browser for the current sketch. Solid lines show the returned exact-kinematic trajectories; error is the Web metric, not the paper benchmark.");
    } catch (error) {
      engineFailure(error);
      if (generation === state.requestGeneration && state.mode === "browser") {
        $("result-kind").textContent = "BROWSER INFERENCE · NO RESULT";
        status("Browser inference did not complete. No known mechanism has been displayed in its place.", true);
      }
    } finally {
      state.busy = false; updateEngineControls(); draw();
    }
  });

  $("clear-sketch").addEventListener("click", () => {
    if (state.busy) return;
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
    $("drawing-hint").hidden = true;
    $("result-kind").textContent = "KNOWN TARGET · AWAITING MODEL INFERENCE";
    status("The constructed path is now an inference target. No known mechanism is supplied to the model; Generate must produce new candidates.");
    updateEngineControls(); fitView();
  });
  $("fit-view").addEventListener("click", fitView);
  $("play-pause").addEventListener("click", () => { state.playing = !state.playing; state.lastTime = null; updatePlayback(); });
  $("phase").addEventListener("input", () => { state.phase = Number($("phase").value) / 1000; state.playing = false; updatePlayback(); draw(); });
  $("speed").addEventListener("change", () => { state.speed = Number($("speed").value); });
  for (const id of ["show-mechanism", "show-target"]) $(id).addEventListener("change", draw);
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
