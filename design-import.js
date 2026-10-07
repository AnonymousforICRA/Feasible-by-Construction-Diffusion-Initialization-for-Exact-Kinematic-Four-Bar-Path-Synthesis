/** Local-only design-package validation. No DOM, network, model, or optimization. */
import { frames, trace, score } from "./kinematics.js";
import { prepareCurve } from "./curve-input.js";

export const MAX_IMPORT_BYTES = 8 * 1024 * 1024;
const MAX_POINTS = 4096;
const MAX_FRAMES = 2048;
const JOINTS = ["A", "B", "C", "D", "P"];
const HASH = /^[a-f0-9]{64}$/i;
const NOTE = "Imported local record. Geometry, alignment, and web errors are checked locally; model execution, candidate-pool provenance, query budget, and timing remain file claims. Import performs no model inference or optimization.";
const fail = (message) => { throw new TypeError(`Invalid design JSON: ${message}`); };

function object(value, name) {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail(`${name} must be an object.`);
  return value;
}
function bounded(value, limit, name) {
  if (!Number.isFinite(value) || Math.abs(value) > limit) fail(`${name} is outside the supported finite range.`);
  return value;
}
function integer(value, minimum, maximum, name) {
  if (!Number.isInteger(value) || value < minimum || value > maximum) fail(`${name} is outside the supported integer range.`);
  return value;
}
function near(a, b, name, tolerance = 1e-8) {
  if (!Number.isFinite(a) || !Number.isFinite(b) || Math.abs(a - b) > tolerance * Math.max(1, Math.abs(a), Math.abs(b))) {
    fail(`${name} does not match the recomputed geometry or web metric.`);
  }
}
function points(value, name, minimum = 0, maximum = MAX_POINTS) {
  if (!Array.isArray(value) || value.length < minimum || value.length > maximum) fail(`${name} has an unsupported point count.`);
  return value.map((point) => {
    if (!Array.isArray(point) || point.length !== 2) fail(`${name} must contain 2-D points.`);
    return point.map((v) => bounded(v, 1e8, name));
  });
}
function comparePoints(actual, expected, name) {
  if (actual.length !== expected.length) fail(`${name} has an inconsistent point count.`);
  actual.forEach((point, i) => point.forEach((v, j) => near(v, expected[i][j], name)));
}

// Do this before processing any values. The byte, nesting, and node caps also
// bound work for unused metadata; unknown fields are never copied into results.
function auditStructure(root) {
  const pending = [[root, 0]];
  let nodes = 0;
  while (pending.length) {
    const [value, depth] = pending.pop();
    if (++nodes > 300000 || depth > 20) fail("the package is too deeply nested or complex.");
    if (typeof value === "number" && !Number.isFinite(value)) fail("numbers must be finite.");
    if (typeof value === "string" && value.length > 4096) fail("metadata strings are too long.");
    if (value && typeof value === "object") {
      for (const [key, item] of Object.entries(value)) {
        if (["__proto__", "prototype", "constructor"].includes(key)) fail("unsafe object keys are not accepted.");
        pending.push([item, depth + 1]);
      }
    }
  }
}

function readSolution(value, target, constructed) {
  const s = object(value, "solution"), input = object(s.params, "parameters");
  const params = Object.fromEntries(["r1", "r2", "r3", "r4", "px", "py"].map((key) => [key, bounded(input[key], 1e4, key)]));
  const { r1, r2, r3, r4, px, py } = params;
  if (r1 !== 1 || !(r2 > 0 && r2 < 1 && r3 > 0 && r4 > 0 && r3 + r4 > 1 + r2 && Math.abs(r3 - r4) < 1 - r2)) {
    fail("parameters must describe a normalized, strict full-crank four-bar.");
  }
  if (s.mechanism_family != null && s.mechanism_family !== "strict_full_crank_fourbar_branch_plus") fail("unsupported mechanism family.");
  if (s.path_closed != null && s.path_closed !== true) fail("only closed four-bar paths are supported.");
  const placement = object(s.transform, "transform");
  const transform = Object.fromEntries(["k", "theta", "tx", "ty"].map((key) => [key, bounded(placement[key], 1e8, `transform.${key}`)]));
  if (!(transform.k > 0)) fail("transform scale must be positive.");
  if (placement.cyclic_shift_samples_180 != null) {
    transform.cyclic_shift_samples_180 = bounded(placement.cyclic_shift_samples_180, 180, "cyclic shift");
    if (transform.cyclic_shift_samples_180 < 0) fail("cyclic shift must be non-negative.");
  }
  if (!Array.isArray(s.frames) || s.frames.length < 2 || s.frames.length > MAX_FRAMES) fail("unsupported animation frame count.");
  const curve = points(s.coupler_curve, "coupler curve", 2, MAX_FRAMES);
  if (!constructed && (s.frames.length !== 240 || curve.length !== 256)) fail("browser records require the original 240 animation frames and 256-point FK trace.");
  const physical = [r2, r3, r4, px, py];
  const checkedFrames = frames(physical, s.frames.length), checkedCurve = trace(physical, curve.length);
  s.frames.forEach((pose, i) => {
    object(pose, "animation frame");
    JOINTS.forEach((key) => comparePoints(points([pose[key]], `frame.${key}`, 1, 1), [checkedFrames[i][key]], `frame.${key}`));
  });
  comparePoints(curve, checkedCurve, "coupler curve");
  // Prevent finite but impractically enormous transformed coordinates from
  // reaching canvas/SVG operations, including points outside the coupler trace.
  const c = Math.cos(transform.theta), sn = Math.sin(transform.theta);
  for (const point of [...checkedCurve, ...checkedFrames.flatMap((pose) => JOINTS.map((name) => pose[name]))]) {
    bounded(transform.tx + transform.k * (c * point[0] - sn * point[1]), 1e12, "world coordinate");
    bounded(transform.ty + transform.k * (sn * point[0] + c * point[1]), 1e12, "world coordinate");
  }
  let error = null;
  if (!constructed) {
    if (!["diffusion", "retained_refinement", "endpoint_refinement"].includes(s.source)) fail("unsupported browser candidate source.");
    error = bounded(s.error_percent, 1e8, "reported error");
    if (error < 0) fail("reported error must be non-negative.");
    const checked = score(target, checkedCurve);
    near(error, checked.error * 100, "reported error", 1e-7);
    for (const key of ["k", "theta", "tx", "ty"]) near(transform[key], checked.transform[key], `alignment.${key}`, 1e-7);
    if (transform.cyclic_shift_samples_180 != null) near(transform.cyclic_shift_samples_180, checked.transform.cyclic_shift_samples_180, "alignment phase", 1e-7);
    // Keep the recorded rounding after verification, so import does not alter
    // before/after identities or historical numbers by machine epsilon.
  }
  return { params, transform, frames: checkedFrames, coupler_curve: checkedCurve, error_percent: error,
    ...(constructed ? {} : { source: s.source }), mechanism_family: "strict_full_crank_fourbar_branch_plus", path_closed: true };
}

function readBrowserMetadata(record, solutions, target) {
  const profile = record.provenance.profile;
  if (!["browser_quality_v1", "browser_fast_v1"].includes(profile)) fail("unsupported browser profile.");
  const quality = profile === "browser_quality_v1";
  const raw = object(record.comparison, "comparison"), audit = object(record.audit, "audit");
  if (raw.kind !== "same_pool_best" || raw.final_best_index !== 0 || raw.refinement_applied !== quality) fail("invalid same-pool comparison.");
  const initialBest = readSolution(raw.initial_best, target, false);
  if (initialBest.source !== "diffusion") fail("initial best must be a diffusion candidate.");
  near(audit.raw_best_error_percent, initialBest.error_percent, "raw-best error", 1e-10);
  if (solutions[0].error_percent > initialBest.error_percent + 1e-10 || (!quality && Math.abs(solutions[0].error_percent - initialBest.error_percent) > 1e-10)) fail("inconsistent initial/final best errors.");
  solutions.forEach((s, i) => { if (i && s.error_percent + 1e-10 < solutions[i - 1].error_percent) fail("candidates are not score ordered."); });
  const expected = { selection_fk_calls: quality ? 472 : 32, candidates: quality ? 64 : 32, ddim_steps: quality ? 50 : 25,
    strict_valid_candidates: quality ? 64 : 32, total_candidates: quality ? 64 : 32, refinement_steps: quality ? 100 : 0 };
  for (const [key, value] of Object.entries(expected)) if (audit[key] !== value) fail(`audit.${key} is inconsistent with the profile.`);
  const fixed = { selection_policy: "score_sorted_unit_distance_filter", unit_distance_threshold: 0.10, diversity_guarantee: false,
    gradient: "forward_mode_automatic_differentiation", preprocessing: "closed_polyline_no_smoothing",
    random_stream: "browser_seeded_rng_not_pytorch", runtime: "onnxruntime_web_wasm_cpu" };
  for (const [key, value] of Object.entries(fixed)) if (audit[key] !== value) fail(`unsupported audit.${key}.`);
  const extra = integer(audit.comparison_display_traces, 0, 1, "comparison display traces");
  if (audit.solution_display_traces !== solutions.length || audit.display_traces !== solutions.length + extra) fail("inconsistent display trace counts.");
  const repro = object(record.reproducibility, "reproducibility");
  const seed = integer(repro.seed, 0, 0xffffffff, "seed");
  if (!HASH.test(repro.model_sha256 ?? "") || repro.requested_profile !== (quality ? "quality_v2" : "fast_v1") || repro.demo_version !== "post-submission-v2") fail("unsupported reproducibility record.");
  const timing = object(record.timing, "timing");
  const seconds = bounded(timing.browser_compute_seconds, 7 * 24 * 60 * 60, "recorded compute time");
  if (seconds < 0) fail("recorded compute time must be non-negative.");
  return {
    backend: "browser_onnx_exact_fk", profile, metric: "web_similarity_rms_v1",
    comparison: { kind: "same_pool_best", refinement_applied: quality, initial_best: initialBest, final_best_index: 0 },
    reproducibility: { seed, requested_profile: repro.requested_profile, model_sha256: repro.model_sha256.toLowerCase(), demo_version: repro.demo_version },
    audit: { ...expected, ...fixed, raw_best_error_percent: initialBest.error_percent,
      solution_display_traces: solutions.length, comparison_display_traces: extra, display_traces: solutions.length + extra },
    timing: { browser_compute_seconds: seconds },
  };
}

/** Parse and fully validate before the caller changes the current design.
 * Browser-run provenance is not authenticated by importing a JSON file.
 * Replay is only a proposal: the caller MUST verify the loaded model SHA first.
 */
export function parseDesignImport(text) {
  if (typeof text !== "string" || text.length > MAX_IMPORT_BYTES || new TextEncoder().encode(text).byteLength > MAX_IMPORT_BYTES) fail("file exceeds the 8 MiB limit.");
  let record;
  try { record = JSON.parse(text); } catch { fail("file is not valid JSON."); }
  auditStructure(record);
  object(record, "package");
  if (record.schema !== "fourbar_design_export_v1") fail("unsupported export schema.");
  const provenance = object(record.provenance, "provenance");
  const constructed = provenance.is_constructed_preset === true;
  if (!constructed && provenance.is_constructed_preset !== false) fail("missing geometry provenance.");
  if (constructed) {
    if (!["constructed_geometry", "imported_record"].includes(provenance.kind) || provenance.backend != null || provenance.profile != null) fail("unsupported constructed provenance.");
    if (record.comparison != null || record.reproducibility != null || record.audit != null || record.timing != null) fail("constructed geometry cannot claim inference metadata.");
  } else if (!["browser_inference", "imported_record"].includes(provenance.kind) || provenance.backend !== "browser_onnx_exact_fk" || record.metric?.name !== "web_similarity_rms_v1") {
    fail("only constructed geometry and supported browser inference exports can be imported.");
  }
  const original = record.original_input_target == null ? null : points(record.original_input_target, "original target");
  const processed = record.processed_target == null ? null : points(record.processed_target, "processed target");
  if (!constructed) {
    if (!processed || processed.length < 5) fail("browser records require the processed target.");
    // Reject degenerate inputs before recomputing similarity, and verify that
    // replay would use precisely this preprocessing rather than a hidden curve.
    prepareCurve(processed);
    if (original) comparePoints(prepareCurve(original), processed, "original/processed target");
  }
  if (!Array.isArray(record.solutions) || !record.solutions.length || record.solutions.length > 6) fail("expected one to six candidates.");
  const solutions = record.solutions.map((s) => readSolution(s, processed, constructed));
  const selection = object(record.selection, "selection");
  const selectedIndex = integer(selection.selected_index, 0, solutions.length - 1, "selected candidate");
  const phase = bounded(selection.phase, 1, "phase");
  if (phase < 0) fail("phase must be between zero and one.");
  if (!constructed) near(record.metric.selected_error_percent, solutions[selectedIndex].error_percent, "selected metric", 1e-10);
  if (record.path != null && record.path.closed !== true) fail("only closed paths are supported.");
  const metadata = constructed ? { demo_kind: "constructed" } : readBrowserMetadata(record, solutions, processed);
  const result = { ...metadata, solutions, ...(processed ? { clean_curve: processed } : {}), path: { closed: true },
    imported_record: { schema: "fourbar_design_export_v1", source_kind: constructed ? "constructed_geometry" : "browser_inference",
      geometry_checked: true, scores_checked: !constructed, run_provenance_verified: false, note: NOTE } };
  const replay = !constructed && original ? { seed: metadata.reproducibility.seed,
    profile: metadata.reproducibility.requested_profile, model_sha256: metadata.reproducibility.model_sha256 } : null;
  return { result, target: original ?? processed ?? [], selectedIndex, phase, replay };
}
