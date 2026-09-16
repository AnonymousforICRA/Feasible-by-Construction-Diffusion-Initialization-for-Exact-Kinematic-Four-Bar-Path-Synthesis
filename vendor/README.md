# Self-hosted runtime

ONNX Runtime Web 1.22.0, from the official `onnxruntime-web` npm package.
Distributed files: `ort.wasm.min.mjs`, `ort-wasm-simd-threaded.mjs`,
`ort-wasm-simd-threaded.wasm`. See `LICENSE-onnxruntime.txt` (MIT).

The application uses the WASM CPU execution provider with one thread inside
a dedicated worker. It does not need WebGPU, a remote inference service, or
cross-origin isolation headers. All runtime/model URLs are same-origin.
