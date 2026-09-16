# Browser implementation scope

This is a real, client-side inference port, not a remote API or a lookup table.
The frozen EMA denoiser is exported without optimizer state, training records,
or dataset examples. All required runtime and model assets are self-hosted.

## Computation

- A 48-dimensional Fourier descriptor conditions the 5-dimensional diffusion
  model: 256-wide FiLM residual MLP, six blocks, 128-dimensional time embedding.
- DDIM uses CFG = 1 and eta = 0. A constructive parameterization maps the unit
  cube into the strictly assemblable full-crank four-bar family.
- Fast: K32 / DDIM25, 32 exact-FK candidate evaluations, no refinement.
- Quality: K64 / DDIM50, four independent Adam starts, 100 steps, lr = 0.03.
  Five-dimensional forward-mode automatic differentiation traverses the exact
  kinematics and piecewise-differentiable shape objective. This is not a
  finite-difference optimizer or a learned forward surrogate.
- Quality selection uses 64 + 400 + 8 = 472 exact-FK candidate evaluations.
  Original candidates, retained hard-best iterates, and endpoints are scored
  together. Animation frames are evaluated separately and are not hidden in
  that selection budget.

## Numerical checks

The exported model was checked against PyTorch with shared synthetic conditions
and fixed initial noise. Maximum absolute unit-coordinate discrepancies for
the actual WASM + JavaScript sampler were approximately 2.44e-5 (K32/DDIM25)
and 1.25e-4 (K64/DDIM50). These are port tolerances, not trajectory errors.

Six synthetic float64 comparisons checked constructive geometry, FK, the
hard-cyclic objective and all five gradients, the reporting metric, and Fourier
encoding against the Python implementation. Maximum discrepancies were below
1e-13. Independent finite-difference tests verify the analytic gradients.

A local Chrome end-to-end check additionally covered model loading, real
quality-profile synthesis, candidate switching, phase scrubbing, a freehand
triangle, a 390-pixel layout, and visible failure when model loading is blocked.
All observed requests were same-origin GETs; no sketch POST was sent. This is
a functional check on one desktop browser, not a cross-device speed benchmark.

## Differences from the reference application

- The browser PRNG is seeded but is not PyTorch's PRNG. Equal numeric seeds do
  not produce the same initial candidate noise as the Python application.
- Denoising uses float32 ONNX/WASM; local optimization uses JavaScript float64
  arithmetic. Full optimizer trajectories need not be bit-identical to CUDA.
- Strokes are treated as explicitly closed polylines, without automatic
  smoothing or corner classification. A straight segment closes the path.
- The on-screen error is the fractional-phase, proper-similarity-aligned
  `web_similarity_rms_v1`, normalized by the target gyration radius. It is not
  the canonical paper benchmark metric. No benchmark claim is inferred from
  an individual interactive sketch.
- Known-mechanism presets are forward-kinematic constructions. They are not
  inverse-design model predictions. Clicking **Use this target** followed by
  **Generate** runs an actual independent inverse-design computation.
- Playback uses uniform crank phase, not MINCO, minimum-snap, dynamics,
  collision avoidance, or a motor controller.

A single-input four-bar cannot exactly reproduce every arbitrary trajectory.
The returned result is a finite-budget approximation, not a global-optimum
certificate. Measured browser duration depends on the visitor's hardware and
must not be compared directly with a device-resident GPU benchmark.

## Privacy and availability

There is no application-level analytics, external inference request, or
server-side sketch storage. Hosting providers may keep ordinary access logs.
The initial download is approximately 19 MB before HTTP compression/caching.
Model loading is explicit. Unsupported or failed computation is reported;
there is no silent replacement with a prerecorded result.
