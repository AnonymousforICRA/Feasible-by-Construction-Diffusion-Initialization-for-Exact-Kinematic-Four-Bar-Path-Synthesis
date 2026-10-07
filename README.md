# Browser four-bar demonstration

## Post-submission preview (v3 interface, v2 inference contract)

This version adds presentation and export features to the frozen browser model;
it does not introduce new training or paper benchmark results. The original
site has been archived separately before editing. Publishing this preview is
a separate, explicit step; local edits do not update the public site.

- **Same-pool comparison:** initial best and retained final best from one run,
  with initial/selected mechanism toggles. Selecting an alternative does not
  change the best-to-best statistics. This is not a single-start optimization
  history, or a Fast-versus-Quality ablation. Fast explicitly has no refinement.
- **Path close-up and thumbnails:** true stored FK curves and placements,
  uniformly scaled on both axes. No corner smoothing, residual exaggeration,
  geometry interpolation, or changed selection objective.
- **JSON / SVG / PNG:** export the selected candidate (even when the initial
  view is open). JSON retains the original and processed targets, all returned
  candidates, same-pool initial best, seed, model SHA-256, profile, budgets and
  measured time. SVG/PNG show an independent path close-up and scope notes.
  These are relative geometric designs, not fabrication-ready CAD.
- **Mobile and cancellation:** viewer-first mobile layout, two-way control
  shortcuts, collapsed run details, and explicit worker cancellation. Cancelling
  stops the worker and requires loading it again before another run.

- **Draw / Inspect:** results default to Inspect. Drag pans without changing the
  target; wheel, zoom buttons and focused-canvas keyboard controls change only
  the camera. Draw explicitly replaces the target. Undo restores the previous
  sketch and result; an incomplete/cancelled stroke is discarded automatically.
- **One-click example:** Run this example explicitly loads the model and performs
  actual Quality inference on the selected target points, never its source
  mechanism parameters. Cancellation is available during loading and synthesis.
- **Import / replay:** open a local v1 design JSON (up to 8 MiB). Geometry and
  browser scores are checked against FK and the Web metric. Imported model-run
  provenance, timing and budget remain file claims, visibly labelled as such.
  Run again uses the saved original target, seed and profile only after the
  loaded model hash matches. Browser/runtime versions can introduce numerical
  differences; this is not a bit-identical cross-platform guarantee.
- **Recent designs:** the last five completed/imported designs are held in this
  tab's memory only. Restore does not rerun inference. Refresh clears history;
  export JSON to keep a design. No localStorage, account or file upload is used.

Only the two endpoints are shown; no intermediate optimization movie is claimed.

Standalone anonymous static interface for GitHub Pages. Local asset paths work
under a repository subpath. The interface contains no remote inference API,
CDN fonts, analytics, accounts, or tracking.

## Two explicit modes

- **Known mechanisms** constructs two public hand-selected legal geometries with
  exact forward kinematics. They are not model predictions or recorded model
  results. No model error score is displayed for these presets. **Use this target
  for inference** transfers only target points, never known mechanism parameters.
  **Run this example · Quality** combines that transfer, model loading and actual
  Quality synthesis in one explicit action.
- **Draw a path** loads the browser engine after an explicit **Load browser model**
  action. Generate remains disabled until the model is ready and at least five
  points exist. A straight segment closes the input polyline; no smoothing removes
  its corners. Fast requests K32 / DDIM25 without refinement; Quality requests
  K64 / DDIM50 and a 472-call exact-FK selection budget. Errors and timings come
  from the actual browser run. Failures never substitute a known mechanism.

The initial model and WebAssembly runtime download totals about 19 MB. Inference
runs on the device CPU in a worker; no GPU or inference server is required.
Successful model initialization is checked before inference is enabled.

The page displays the target as a dashed line and the realized coupler trajectory
as a solid line. Playback uses uniform crank phase; it is not a timing optimizer,
controller, collision test, or dynamics simulation. The Web similarity error is
not the paper benchmark metric. A single-actuator four-bar cannot reproduce
arbitrary paths exactly.

## Publish with GitHub Pages

This directory is the complete site root. Publish its contents in the anonymous
repository's `main` branch, without any original repository history. In repository
**Settings → Pages → Build and deployment → Source**, select **GitHub Actions**.
The included workflow deploys on a push to `main` and can also be run manually.
No secrets, GPU server, build toolchain, or external model download are required.

For local use, serve this directory with a static HTTP server and open localhost;
do not open `index.html` directly as a `file://` URL. Production hosting must use
HTTPS for model-integrity checking. Use a current browser with WebAssembly SIMD
and module-worker support. Older browsers may fail explicitly during loading.

The mathematical port, numerical validation and reference-application differences
are documented in [VALIDATION.md](./VALIDATION.md). Browser PRNG, arithmetic and
preprocessing are intentionally disclosed; this is not a bit-identical CUDA run.

## Integration contract

`browser-engine.js` is loaded only after an explicit load, one-click example or
saved-settings replay request. It must
export these asynchronous functions:

```js
initializeEngine(onProgress)
synthesize(points, options, onProgress)
```

Progress may be a string or an object with a `message` field. The UI recognizes
`error.code === "UNSUPPORTED"` and `NotSupportedError` as an unsupported runtime;
other thrown errors become a visible failure. The return value from `synthesize`
must have the `solutions` shape described below. Successful
initialization must mean the runtime and model are actually ready.

`config.js` contains local presentation defaults and the requested inference
options. See the validation document for the supported computation and differences
from the reference inference implementation.

The engine returns the following shape (arrays below are abbreviated):

```json
{
  "solutions": [{
        "params": {"r1": 1, "r2": 0.42, "r3": 1.35, "r4": 1.25, "px": 0.24, "py": -0.31},
        "transform": {"k": 1, "theta": 0, "tx": 0, "ty": 0},
        "frames": [],
        "coupler_curve": [],
        "error_percent": 0.1
  }],
  "clean_curve": [],
  "audit": {},
  "timing": {}
}
```

The abbreviated arrays above are illustrative, not valid animation data. Each frame
must contain canonical `A`, `B`, `C`, `D`, and `P` pairs; trajectories and frames
must contain at least two samples. Input `points` are in target/world coordinates.
Placement is `world = k * R(theta) * canonical + [tx, ty]`; theta is in radians.
The interface uses returned errors and does not fabricate candidate metrics.
The audit block displays `candidates`, `ddim_steps`, `strict_valid_candidates`,
`total_candidates`, and `selection_fk_calls`; measured browser CPU time comes
from `timing.browser_compute_seconds`.

The v2 browser response additionally requires `comparison` with
`kind: "same_pool_best"`, `refinement_applied`, a complete `initial_best`
solution, and `final_best_index: 0`. `reproducibility` records `seed`,
`requested_profile`, the verified `model_sha256`, and
`demo_version: "post-submission-v2"`. `presentation.js` implements pure
camera and export helpers. `interaction.js` implements camera navigation and
session-only history; `design-import.js` validates local imports. Any release
allowlist must include all three files. UI v3 deliberately retains the frozen
`post-submission-v2` inference identity; it changes no sampling or optimization.
Animation-only FK evaluations are reported separately as `display_traces`;
the initial comparison may add one such evaluation, not a selection query.

## Lightweight checks

```sh
node --check app.js
node --check config.js
```

From the repository root, run the Node regression suite:

```sh
node --test tests/browser_kinematics.test.mjs tests/browser_denoiser.test.mjs tests/browser_pipeline.test.mjs tests/browser_presentation.test.mjs tests/canvas_viewport.test.mjs tests/browser_import.test.mjs tests/browser_interaction.test.mjs
```

`app.js` also exports pure result-validation, similarity-transform, frame-index,
and camera-fit helpers for CPU-only tests. Importing it in Node does not load a
model, create a browser session, or modify any data.
