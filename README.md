# Browser four-bar demonstration

Standalone anonymous static interface for GitHub Pages. Local asset paths work
under a repository subpath. The interface contains no remote inference API,
CDN fonts, analytics, accounts, or tracking.

## Two explicit modes

- **Known mechanisms** constructs two public hand-selected legal geometries with
  exact forward kinematics. They are not model predictions or recorded model
  results. No model error score is displayed for these presets. **Use this target
  for inference** transfers only target points, never known mechanism parameters.
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

`browser-engine.js` is loaded only after an explicit model-load request. It must
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

## Lightweight checks

```sh
node --check app.js
node --check config.js
```

`app.js` also exports pure result-validation, similarity-transform, frame-index,
and camera-fit helpers for CPU-only tests. Importing it in Node does not load a
model, create a browser session, or modify any data.
