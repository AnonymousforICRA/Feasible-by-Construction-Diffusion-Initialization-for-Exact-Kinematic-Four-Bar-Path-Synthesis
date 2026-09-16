/** Browser-only EMA denoising. All model/runtime assets are same-origin. */
const ASSET_BASE = new URL('./', import.meta.url);
const f32 = Math.fround;
let loading;
let activeSampler;

function progress(callback, stage, completed, total, message) {
  callback?.({ stage, completed, total, fraction: total ? completed / total : 0, message });
}

function finiteVector(value, length, name, positive = false) {
  if (!Array.isArray(value) || value.length !== length ||
      value.some((v) => !Number.isFinite(v) || (positive && v <= 0))) {
    throw new Error(`Invalid ${name} in model metadata.`);
  }
  return Float32Array.from(value);
}

export function validateMetadata(metadata) {
  if (metadata?.format_version !== 1 || metadata.architecture?.param_dim !== 5 ||
      metadata.architecture?.cond_dim !== 48 || metadata.sampling?.cfg_scale !== 1 ||
      metadata.sampling?.eta !== 0 || metadata.sampling?.timesteps !== 1000 ||
      !/^[a-f0-9]{64}$/.test(metadata.export_sha256 ?? '')) {
    throw new Error('Unsupported denoiser metadata.');
  }
  const n = metadata.normalization;
  if (!n || !metadata.clip) throw new Error('Missing normalization or clipping bounds.');
  const parsed = {
    unitMean: finiteVector(n.unit_mean, 5, 'unit_mean'),
    unitStd: finiteVector(n.unit_std, 5, 'unit_std', true),
    codeMean: finiteVector(n.code_mean, 48, 'code_mean'),
    codeStd: finiteVector(n.code_std, 48, 'code_std', true),
    low: finiteVector(metadata.clip.low, 5, 'clip.low'),
    high: finiteVector(metadata.clip.high, 5, 'clip.high'),
    alpha: finiteVector(metadata.alpha_cumprod, 1000, 'alpha_cumprod', true),
  };
  for (let i = 0; i < 5; i++) {
    if (parsed.low[i] >= parsed.high[i]) throw new Error('Invalid clipping interval.');
  }
  for (let i = 0; i < parsed.alpha.length; i++) {
    if (parsed.alpha[i] >= 1 || (i && parsed.alpha[i] >= parsed.alpha[i - 1])) {
      throw new Error('Invalid diffusion schedule.');
    }
  }
  return parsed;
}

/** Mirrors torch.linspace(-1, 999, steps + 1).long().flip(0). */
export function samplingTimes(steps) {
  if (steps !== 25 && steps !== 50) throw new Error('DDIM steps must be 25 or 50.');
  return Array.from({ length: steps + 1 }, (_, i) => Math.trunc(-1 + (steps - i) * 1000 / steps));
}

/** Seeded browser PRNG; seed IDs do not claim to reproduce PyTorch's RNG. */
export function initialGaussianNoise(count, seed = 0) {
  if (!Number.isInteger(count) || count <= 0 || !Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new Error('Noise size must be positive and seed must be a uint32 integer.');
  }
  let state = seed >>> 0;
  const uniform = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return (((value ^ (value >>> 14)) >>> 0) + 0.5) / 4294967296;
  };
  const noise = new Float32Array(count);
  for (let i = 0; i < count; i += 2) {
    const radius = Math.sqrt(-2 * Math.log(uniform()));
    const angle = 2 * Math.PI * uniform();
    noise[i] = radius * Math.cos(angle);
    if (i + 1 < count) noise[i + 1] = radius * Math.sin(angle);
  }
  return noise;
}

function flattenRows(rows, width, name) {
  if (!Array.isArray(rows) || rows.length < 1 || rows.length > 64) {
    throw new Error(`${name} requires between 1 and 64 rows.`);
  }
  const data = new Float32Array(rows.length * width);
  rows.forEach((row, r) => {
    if ((!Array.isArray(row) && !ArrayBuffer.isView(row)) || row.length !== width) {
      throw new Error(`${name} rows must contain ${width} values.`);
    }
    for (let c = 0; c < width; c++) {
      if (!Number.isFinite(row[c]) || !Number.isFinite(f32(row[c]))) {
        throw new Error(`${name} contains a non-finite value.`);
      }
      data[r * width + c] = row[c];
    }
  });
  return data;
}

/** One eta=0, cfg=1 update, including per-operation float32 rounding. */
export function ddimStep(x, epsilon, time, nextTime, schedule) {
  const sqrtAlpha = f32(Math.sqrt(schedule.alpha[time]));
  const sqrtOneMinusAlpha = f32(Math.sqrt(f32(1 - schedule.alpha[time])));
  const sqrtNextAlpha = nextTime < 0 ? 0 : f32(Math.sqrt(schedule.alpha[nextTime]));
  const direction = nextTime < 0 ? 0 : f32(Math.sqrt(f32(1 - schedule.alpha[nextTime])));
  const output = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    if (!Number.isFinite(epsilon[i])) throw new Error('Denoiser produced non-finite noise.');
    const dim = i % 5;
    const predicted = f32(f32(x[i] - f32(sqrtOneMinusAlpha * epsilon[i])) / Math.max(sqrtAlpha, 1e-12));
    const clean = Math.min(schedule.high[dim], Math.max(schedule.low[dim], predicted));
    output[i] = nextTime < 0 ? clean : f32(f32(sqrtNextAlpha * clean) + f32(direction * epsilon[i]));
    if (!Number.isFinite(output[i])) throw new Error('DDIM produced a non-finite candidate.');
  }
  return output;
}

/** Injectable runtime boundary also used by the numerical parity test. */
export function createSampler(runtime, session, metadata) {
  const norm = validateMetadata(metadata);
  return async (conditionBatch, { steps = 25, seed = 0, initialNoise } = {}, onProgress) => {
    const times = samplingTimes(steps);
    const raw = flattenRows(conditionBatch, 48, 'conditionBatch');
    const batch = conditionBatch.length;
    const condition = new Float32Array(raw.length);
    for (let i = 0; i < raw.length; i++) {
      condition[i] = f32(f32(raw[i] - norm.codeMean[i % 48]) / norm.codeStd[i % 48]);
      if (!Number.isFinite(condition[i])) throw new Error('Condition normalization overflow.');
    }
    let x;
    if (initialNoise === undefined) {
      x = initialGaussianNoise(batch * 5, seed);
    } else if (ArrayBuffer.isView(initialNoise) || (Array.isArray(initialNoise) && typeof initialNoise[0] === 'number')) {
      if (initialNoise.length !== batch * 5) throw new Error('initialNoise has the wrong shape.');
      x = Float32Array.from(initialNoise);
    } else {
      x = flattenRows(initialNoise, 5, 'initialNoise');
      if (x.length !== batch * 5) throw new Error('initialNoise has the wrong shape.');
    }
    if (x.some((value) => !Number.isFinite(value))) throw new Error('initialNoise must be finite.');
    const condTensor = new runtime.Tensor('float32', condition, [batch, 48]);
    progress(onProgress, 'sampling', 0, steps, `DDIM ${steps}: ${batch} candidates`);
    try {
      for (let i = 0; i < steps; i++) {
        const xTensor = new runtime.Tensor('float32', x, [batch, 5]);
        const tTensor = new runtime.Tensor('int64', BigInt64Array.from({ length: batch }, () => BigInt(times[i])), [batch]);
        let result;
        try {
          result = await session.run({ x: xTensor, t: tTensor, cond: condTensor });
          if (!result.epsilon || result.epsilon.data.length !== x.length) throw new Error('Invalid denoiser output shape.');
          x = ddimStep(x, result.epsilon.data, times[i], times[i + 1], norm);
        } finally {
          xTensor.dispose?.();
          tTensor.dispose?.();
          if (result) Object.values(result).forEach((tensor) => tensor.dispose?.());
        }
        progress(onProgress, 'sampling', i + 1, steps, `DDIM ${i + 1}/${steps}`);
        // Let progress paint and input events run even without a worker.
        if (i % 4 === 3) await new Promise((resolve) => setTimeout(resolve, 0));
      }
    } finally {
      condTensor.dispose?.();
    }
    return Array.from({ length: batch }, (_, row) => {
      const unit = new Float32Array(5);
      for (let d = 0; d < 5; d++) {
        unit[d] = Math.max(-1, Math.min(1, f32(f32(x[row * 5 + d] * norm.unitStd[d]) + norm.unitMean[d])));
      }
      return unit;
    });
  };
}

async function fetchModel(url, onProgress) {
  const response = await fetch(url, { credentials: 'same-origin' });
  if (!response.ok) throw new Error('Unable to download the inference model.');
  const total = Number(response.headers.get('content-length')) || 0;
  if (!response.body?.getReader) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    progress(onProgress, 'download', loaded, total, 'Downloading local inference weights');
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return bytes;
}

/** Loads an anonymous EMA-only ONNX model into a single-threaded WASM session. */
export async function loadDenoiser(onProgress) {
  if (!loading) {
    loading = (async () => {
      progress(onProgress, 'loading', 0, 1, 'Loading self-hosted inference runtime');
      const [runtime, response] = await Promise.all([
        import('./vendor/ort.wasm.min.mjs'),
        fetch(new URL('weights/model.json', ASSET_BASE), { credentials: 'same-origin' }),
      ]);
      if (!response.ok) throw new Error('Unable to load inference metadata.');
      const metadata = await response.json();
      validateMetadata(metadata);
      runtime.env.wasm.wasmPaths = new URL('vendor/', ASSET_BASE).href;
      runtime.env.wasm.numThreads = 1;
      runtime.env.wasm.proxy = false;
      const bytes = await fetchModel(new URL('weights/model.onnx', ASSET_BASE), onProgress);
      if (!globalThis.crypto?.subtle) throw new Error('Secure-context cryptography is required to verify inference weights.');
      const digest = await crypto.subtle.digest('SHA-256', bytes);
      const observed = [...new Uint8Array(digest)].map((v) => v.toString(16).padStart(2, '0')).join('');
      if (observed !== metadata.export_sha256) throw new Error('Inference weight integrity check failed.');
      progress(onProgress, 'initializing', 0, 1, 'Initializing browser WASM inference');
      const session = await runtime.InferenceSession.create(bytes, {
        executionProviders: ['wasm'], graphOptimizationLevel: 'all',
      });
      activeSampler = createSampler(runtime, session, metadata);
      progress(onProgress, 'ready', 1, 1, 'Browser inference is ready');
      return metadata;
    })().catch((error) => { loading = undefined; activeSampler = undefined; throw error; });
  }
  return loading;
}

/** One raw 48-D Fourier condition per candidate; returns rows in [-1, 1]^5. */
export async function sample(conditionBatch, options, onProgress) {
  await loadDenoiser(onProgress);
  return activeSampler(conditionBatch, options, onProgress);
}
