/** Pure presentation/export helpers. No model inference, DOM access, or FK calls. */
const JOINTS = ["A", "B", "C", "D", "P"];
export const DESIGN_COLORS = Object.freeze({ target: "#d18a39", initial: "#ce765c", selected: "#087f83" });
const finitePoint = (point) => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite);

function pointsChecked(points, label = "Points") {
  if (!Array.isArray(points) || !points.every(finitePoint)) throw new TypeError(`${label} must contain finite 2-D points.`);
  return points;
}

/** Clone without JSON's silent conversion of NaN/Infinity to null. */
function jsonCopy(value, seen = new Set()) {
  if (value === null || typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new TypeError("Export data must contain only finite numbers.");
    return value;
  }
  if (!value || typeof value !== "object") throw new TypeError("Export data must be JSON-safe.");
  if (seen.has(value)) throw new TypeError("Export data must not contain circular references.");
  seen.add(value);
  let copy;
  if (Array.isArray(value)) copy = value.map((item) => jsonCopy(item, seen));
  else {
    if (Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null) {
      throw new TypeError("Export metadata must use plain JSON objects.");
    }
    copy = Object.fromEntries(Object.entries(value).map(([key, item]) => [key, jsonCopy(item, seen)]));
  }
  seen.delete(value);
  return copy;
}

/** Uniform-aspect camera; it never independently stretches either axis. */
export function pathCamera(points, width, height, padding = 18) {
  pointsChecked(points);
  if (![width, height, padding].every(Number.isFinite) || width <= 0 || height <= 0 || padding < 0) {
    throw new RangeError("Camera dimensions must be finite and positive, with non-negative padding.");
  }
  let minX = -1, maxX = 1, minY = -1, maxY = 1;
  if (points.length) {
    minX = maxX = points[0][0]; minY = maxY = points[0][1];
    for (const [x, y] of points) {
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    }
  }
  const spanX = maxX - minX, spanY = maxY - minY;
  // Split addition avoids overflow when two finite bounds have the same sign.
  const camera = {
    x: minX / 2 + maxX / 2, y: minY / 2 + maxY / 2,
    scale: Math.min(Math.max(1, width - 2 * padding) / Math.max(1e-6, spanX),
      Math.max(1, height - 2 * padding) / Math.max(1e-6, spanY)),
  };
  if (![spanX, spanY, camera.x, camera.y, camera.scale].every(Number.isFinite) || camera.scale <= 0) {
    throw new RangeError("Coordinates exceed the supported camera range.");
  }
  return camera;
}

/** Mechanism/world coordinates are y-up; Canvas/SVG output is y-down. */
export function projectPoint(point, camera, width, height) {
  if (!finitePoint(point) || !camera || ![camera.x, camera.y, camera.scale, width, height].every(Number.isFinite)
      || camera.scale <= 0 || width <= 0 || height <= 0) throw new TypeError("Invalid finite camera or point.");
  const projected = [width / 2 + (point[0] - camera.x) * camera.scale, height / 2 - (point[1] - camera.y) * camera.scale];
  if (!projected.every(Number.isFinite)) throw new RangeError("Projected coordinates are outside the supported range.");
  return projected;
}

/** Apply the stored proper similarity once, without re-aligning the drawing. */
export function solutionWorldData(solution) {
  const t = solution?.transform;
  if (!t || ![t.k, t.theta, t.tx, t.ty].every(Number.isFinite) || t.k <= 0) throw new TypeError("Invalid solution transform.");
  const c = Math.cos(t.theta), s = Math.sin(t.theta);
  const transform = (point) => {
    if (!finitePoint(point)) throw new TypeError("Solution geometry must contain finite 2-D points.");
    const world = [t.tx + t.k * (c * point[0] - s * point[1]), t.ty + t.k * (s * point[0] + c * point[1])];
    if (!world.every(Number.isFinite)) throw new RangeError("Transformed coordinates exceed the supported range.");
    return world;
  };
  if (!Array.isArray(solution.frames) || solution.frames.length < 2 || !Array.isArray(solution.coupler_curve) || solution.coupler_curve.length < 2) {
    throw new TypeError("Solution must provide animation frames and a coupler curve.");
  }
  return {
    frames: solution.frames.map((frame) => Object.fromEntries(JOINTS.map((name) => [name, transform(frame?.[name])]))),
    curve: solution.coupler_curve.map(transform),
  };
}

function exportState(result, { target, selectedIndex = 0, phase = 0 } = {}) {
  if (!result || !Array.isArray(result.solutions) || !result.solutions.length) throw new TypeError("There is no result to export.");
  if (!Number.isInteger(selectedIndex) || selectedIndex < 0 || selectedIndex >= result.solutions.length) throw new RangeError("Selected candidate is not available.");
  if (!Number.isFinite(phase)) throw new TypeError("Animation phase must be finite.");
  const copy = jsonCopy(result);
  for (const solution of copy.solutions) solutionWorldData(solution);
  if (copy.comparison?.initial_best) solutionWorldData(copy.comparison.initial_best);
  const input = target === undefined ? null : pointsChecked(target, "Original target").map((point) => [...point]);
  const processed = copy.clean_curve == null ? null : pointsChecked(copy.clean_curve, "Processed target");
  const constructed = copy.demo_kind === "constructed";
  if (!constructed && !copy.solutions.every((solution) => Number.isFinite(solution.error_percent) && solution.error_percent >= 0)) {
    throw new TypeError("Synthesis results require finite non-negative reported errors.");
  }
  return { copy, input, processed, selectedIndex, phase: Math.max(0, Math.min(1, phase)), constructed };
}

function metricScope(constructed, metric) {
  if (constructed) return "Constructed forward-kinematic geometry; no model inference and no synthesis score.";
  if (metric === "web_similarity_rms_v1") return "Browser web-similarity RMS error (%); not the paper benchmark metric or benchmark success rate.";
  return "Reported result metric; no equivalence to the paper benchmark is implied.";
}

/** Full editable design package. Unknown provenance stays null, never guessed. */
export function buildDesignExport(result, options = {}) {
  const { copy, input, processed, selectedIndex, phase, constructed } = exportState(result, options);
  // Presets are deliberately unscored even if an old imported record has a score.
  const solutions = constructed ? copy.solutions.map((solution) => ({ ...solution, error_percent: null })) : copy.solutions;
  return {
    schema: "fourbar_design_export_v1",
    provenance: {
      kind: copy.imported_record ? "imported_record" : constructed ? "constructed_geometry" : copy.backend === "browser_onnx_exact_fk" ? "browser_inference" : "reported_synthesis_result",
      is_constructed_preset: constructed,
      backend: copy.backend ?? null,
      profile: copy.profile ?? null,
      note: copy.imported_record ? "Imported record, not a newly executed inference run. Run metadata is a file claim."
        : constructed ? "Known geometry, not a model prediction." : "Recorded result data; exporting performs no new inference or optimization.",
    },
    imported_record: copy.imported_record ?? null,
    original_input_target: input,
    processed_target: processed,
    coordinate_convention: {
      targets: "World coordinates, y-up; drawing coordinates are not independently normalized for export.",
      solutions: "Canonical mechanism coordinates; world = translation + k * rotation(theta) * canonical.",
      units: "Relative geometry unless the user separately supplies physical units; not fabrication-ready CAD.",
    },
    selection: { selected_index: selectedIndex, phase, phase_role: "Normalized input-crank animation phase, not an optimized time law." },
    solutions,
    comparison: constructed ? null : copy.comparison ?? null,
    comparison_scope: "Same-pool initial best versus final best, when provided. The selected candidate need not be the final best or an improvement over the initial best.",
    reproducibility: constructed ? null : copy.reproducibility ?? null,
    metric: { name: constructed ? null : copy.metric ?? null, scope: metricScope(constructed, copy.metric), selected_error_percent: constructed ? null : solutions[selectedIndex].error_percent },
    audit: constructed ? null : copy.audit ?? null,
    timing: constructed ? null : copy.timing ?? null,
    path: copy.path ?? null,
  };
}

function xml(value) {
  return String(value)
    .replace(/[^\u0009\u000A\u000D\u0020-\uD7FF\uE000-\uFFFD\u{10000}-\u{10FFFF}]/gu, "\uFFFD")
    .replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" }[character]));
}
const number = (value) => {
  if (!Number.isFinite(value)) throw new TypeError("SVG coordinates must be finite.");
  return String(Number(value.toFixed(3)));
};
const percent = (value) => Number.isFinite(value) ? `${value.toFixed(4)}%` : "not recorded";

/** Self-contained vector snapshot: complete mechanism plus an independently fitted path close-up. */
export function buildDesignSvg(result, { target, selectedIndex = 0, phase = 0, showInitial = true } = {}) {
  const exported = buildDesignExport(result, { target, selectedIndex, phase });
  const selected = exported.solutions[selectedIndex], world = solutionWorldData(selected);
  const comparison = exported.comparison;
  const initialSolution = showInitial && comparison?.kind === "same_pool_best" ? comparison.initial_best : null;
  const initial = initialSolution ? solutionWorldData(initialSolution) : null;
  const targetPoints = exported.processed_target ?? exported.original_input_target ?? [];
  const phaseIndex = Math.round(exported.selection.phase * (world.frames.length - 1));
  const frame = world.frames[phaseIndex];
  const allPoints = [...targetPoints, ...world.curve, ...(initial?.curve ?? [])];
  const mechanismPoints = [...allPoints, ...world.frames.flatMap((pose) => JOINTS.map((name) => pose[name]))];
  const geometryCamera = pathCamera(mechanismPoints, 690, 450, 34);
  const closeupCamera = pathCamera(allPoints, 390, 450, 34);
  const text = (x, y, contents, attributes = "") => `<text x="${number(x)}" y="${number(y)}" ${attributes}>${xml(contents)}</text>`;
  const path = (points, camera, width, color, attributes = "") => {
    if (points.length < 2) return "";
    const d = points.map((point, i) => `${i ? "L" : "M"}${projectPoint(point, camera, width, 450).map(number).join(",")}`).join(" ");
    return `<path d="${d} Z" fill="none" stroke="${color}" stroke-width="2.6" stroke-linejoin="round" ${attributes}/>`;
  };
  const curves = (camera, width) => [
    path(targetPoints, camera, width, DESIGN_COLORS.target, 'stroke-dasharray="7 5" data-role="target"'),
    initial ? path(initial.curve, camera, width, DESIGN_COLORS.initial, 'stroke-dasharray="4 4" data-role="initial-best"') : "",
    path(world.curve, camera, width, DESIGN_COLORS.selected, 'data-role="selected-path"'),
  ].join("\n");
  const f = Object.fromEntries(JOINTS.map((name) => [name, projectPoint(frame[name], geometryCamera, 690, 450)]));
  const line = (a, b, color, width, extra = "") => `<line x1="${number(f[a][0])}" y1="${number(f[a][1])}" x2="${number(f[b][0])}" y2="${number(f[b][1])}" stroke="${color}" stroke-width="${width}" stroke-linecap="round" ${extra}/>`;
  const anchors = ["A", "D"].map((name) => `<rect x="${number(f[name][0] - 11)}" y="${number(f[name][1] - 8)}" width="22" height="20" rx="3" fill="#eee1c7" stroke="#b89659"/>`).join("");
  const joints = JOINTS.map((name) => `<circle cx="${number(f[name][0])}" cy="${number(f[name][1])}" r="${name === "P" ? 5 : 6}" fill="${name === "P" ? DESIGN_COLORS.selected : "#ffffff"}" stroke="#27434c" stroke-width="2"/>`).join("");
  const mechanism = `${anchors}<polygon points="${["B", "C", "P"].map((name) => f[name].map(number).join(",")).join(" ")}" fill="#dce7e6" fill-opacity="0.65"/>${line("A", "D", "#bccbcf", 2, 'stroke-dasharray="5 5"')}${line("A", "B", "#087f83", 8)}${line("B", "C", "#627f87", 7)}${line("C", "D", "#087f83", 8)}${line("B", "P", "#8da6ab", 4)}${line("C", "P", "#8da6ab", 4)}${joints}`;
  const constructed = exported.provenance.is_constructed_preset;
  const title = (exported.imported_record ? "Imported · " : "") + (constructed ? "Constructed four-bar geometry" : `Four-bar design · selected candidate ${selectedIndex + 1}`);
  const metricText = constructed ? "Known geometry · no inference · unscored"
    : `Selected ${exported.metric.name === "web_similarity_rms_v1" ? "web" : "reported"} error: ${percent(selected.error_percent)}`;
  const bestIndex = comparison?.final_best_index;
  const finalBest = Number.isInteger(bestIndex) ? exported.solutions[bestIndex] : null;
  const comparisonText = comparison?.kind === "same_pool_best" && finalBest
    ? comparison.refinement_applied === false
      ? `No refinement applied · same-pool best: ${percent(finalBest.error_percent)}; selected candidate may differ.`
      : `Same-pool best: initial ${percent(comparison.initial_best?.error_percent)} → final ${percent(finalBest.error_percent)}; selected candidate may differ.`
    : constructed ? "This preset is not a synthesized model result." : "No same-pool before/after comparison was recorded.";
  const legend = (x, color, name, dashed = false) => `<line x1="${x}" y1="658" x2="${x + 30}" y2="658" stroke="${color}" stroke-width="3" ${dashed ? 'stroke-dasharray="6 4"' : ""}/>${text(x + 40, 663, name, 'font-size="15"')}`;
  const metadata = { schema: exported.schema, provenance: exported.provenance, selection: exported.selection, metric: exported.metric,
    reproducibility: exported.reproducibility, params: selected.params, transform: selected.transform,
    comparison: comparison ? { kind: comparison.kind, refinement_applied: comparison.refinement_applied,
      initial_error_percent: comparison.initial_best?.error_percent ?? null, final_best_index: comparison.final_best_index } : null };
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="790" viewBox="0 0 1200 790" role="img" aria-labelledby="design-title design-description">
<title id="design-title">${xml(title)}</title>
<desc id="design-description">${xml(`${metricText}. ${exported.metric.scope} Geometry and path close-up use separate uniform-aspect cameras.`)}</desc>
<metadata>${xml(JSON.stringify(metadata))}</metadata>
<rect width="1200" height="790" fill="#f5f8f8"/>
<g font-family="Arial, Helvetica, sans-serif" fill="#203c46">
${text(38, 45, title, 'font-size="25" font-weight="700"')}
${text(38, 77, metricText, 'font-size="17"')}
<rect x="30" y="102" width="716" height="530" rx="16" fill="white" stroke="#d9e3e5"/>
<rect x="762" y="102" width="408" height="530" rx="16" fill="white" stroke="#d9e3e5"/>
${text(48, 136, "Mechanism · complete motion envelope", 'font-size="17" font-weight="700"')}
${text(780, 136, "Path-fit close-up", 'font-size="17" font-weight="700"')}
<g transform="translate(43 157)" data-panel="mechanism">${curves(geometryCamera, 690)}${mechanism}</g>
<g transform="translate(771 157)" data-panel="path-closeup">${curves(closeupCamera, 390)}</g>
${legend(45, DESIGN_COLORS.target, "Target", true)}
${initial ? legend(230, DESIGN_COLORS.initial, "Initial best (same pool)", true) : ""}
${legend(535, DESIGN_COLORS.selected, constructed ? "Constructed path" : `Selected candidate ${selectedIndex + 1}`)}
${text(40, 699, comparisonText, 'font-size="14"')}
${text(40, 726, exported.metric.scope, 'font-size="13" fill="#506a72"')}
${text(40, 751, `Crank phase: ${number(exported.selection.phase * 360)}° · phase is for display, not a motor-speed profile.`, 'font-size="13" fill="#506a72"')}
${text(40, 773, exported.imported_record ? "Imported record · not new inference. File run metadata is not independently verified. Geometry only."
  : "Geometric design only; dynamics, collision checks and fabrication tolerances are not certified.", 'font-size="12" fill="#506a72"')}
</g></svg>`;
}
