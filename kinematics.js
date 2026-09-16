/**
 * Browser-local constructive four-bar geometry and exact forward-mode AD.
 * No requests, model loading, finite-difference gradients, or external packages.
 * Arithmetic is IEEE-754 float64 (the research optimizer uses float32), so this
 * is a mathematical port, not a promise of bit-identical optimizer trajectories.
 */
export const FEASIBILITY_MARGIN = 1e-5;
export const BOUNDARY_EPSILON = 1e-5;
export const SCORE_METRIC = "web_similarity_rms_v1";
const DIM = 5;
const counts = { trace: 0, frames: 0, score: 0, refinement: 0 };
export function getEvaluationCounts() { return { ...counts }; }
export function resetEvaluationCounts() { for (const key in counts) counts[key] = 0; }

function vector5(x, name) {
  if (!x || x.length !== DIM || !Array.from(x).every(Number.isFinite)) {
    throw new TypeError(`${name} must contain five finite numbers`);
  }
}
function pointArray(points) {
  if (!Array.isArray(points) || !points.length || points.some(p =>
    !p || p.length !== 2 || !Number.isFinite(p[0]) || !Number.isFinite(p[1]))) {
    throw new TypeError("points must be a nonempty array of finite [x,y] pairs");
  }
}
function sampleCount(n, minimum = 2) {
  if (!Number.isInteger(n) || n < minimum) throw new RangeError(`n must be >= ${minimum}`);
}

// A dual is [value, d/dlogit0, ..., d/dlogit4]. Scalars stay scalars where
// possible. Branch decisions are intentionally stop-gradient assignments.
const value = a => typeof a === "number" ? a : a[0];
const derivative = (a, i) => typeof a === "number" ? 0 : a[i];
function binary(a, b, v, da, db) {
  if (typeof a === "number" && typeof b === "number") return v;
  const out = [v, 0, 0, 0, 0, 0];
  for (let i = 1; i <= DIM; i++) out[i] = da * derivative(a, i) + db * derivative(b, i);
  return out;
}
const add = (a,b) => binary(a,b,value(a)+value(b),1,1);
const sub = (a,b) => binary(a,b,value(a)-value(b),1,-1);
const mul = (a,b) => binary(a,b,value(a)*value(b),value(b),value(a));
const div = (a,b) => binary(a,b,value(a)/value(b),1/value(b),-value(a)/(value(b)*value(b)));
function unary(a, v, d) {
  if (typeof a === "number") return v;
  return [v, a[1]*d, a[2]*d, a[3]*d, a[4]*d, a[5]*d];
}
const sqrt = a => { const v = Math.sqrt(value(a)); return unary(a,v,v > 0 ? 0.5/v : 0); };
const abs = a => unary(a,Math.abs(value(a)),Math.sign(value(a)));
const square = a => mul(a,a);
function maximum(a,b) {
  return value(a) > value(b) ? a : value(a) < value(b) ? b : mul(add(a,b),0.5);
}
function clamp(a, low, high) {
  return value(a) < low ? low : value(a) > high ? high : a;
}
const norm2 = (x,y) => sqrt(add(square(x),square(y)));

function decode(unit) {
  const [u2, ud, us, ux, uy] = unit;
  const r2 = add(0.1,mul(0.8,mul(0.5,add(u2,1))));
  const delta = mul(sub(sub(1,r2),FEASIBILITY_MARGIN),ud);
  const low = maximum(add(add(1,r2),FEASIBILITY_MARGIN),add(0.6,abs(delta)));
  const high = sub(5,abs(delta));
  const sum = add(low,mul(mul(0.5,add(us,1)),sub(high,low)));
  return [r2,clamp(mul(0.5,add(sum,delta)),0.3,2.5),
    clamp(mul(0.5,sub(sum,delta)),0.3,2.5),add(0.5,mul(2,ux)),mul(2,uy)];
}
export function unitToPhysical(unit) {
  vector5(unit,"unit");
  if (Array.from(unit).some(x => x < -1 || x > 1)) throw new RangeError("unit must lie in [-1,1]^5");
  return decode(unit);
}

function geometry(params,n,returnFrames = false) {
  const [r2,r3,r4,px,py] = params;
  const out = [];
  for (let i = 0; i < n; i++) {
    const theta = n === 1 ? 0 : (2*Math.PI*i)/(n-1);
    const bx = mul(r2,Math.cos(theta)), by = mul(r2,Math.sin(theta));
    const dx = sub(1,bx), dy = mul(-1,by), dsq = add(square(dx),square(dy));
    const d = sqrt(dsq), ex = div(dx,d), ey = div(dy,d);
    const a = div(add(sub(square(r3),square(r4)),dsq),mul(2,d));
    const h = sqrt(maximum(sub(square(r3),square(a)),0));
    const cx = sub(add(bx,mul(a,ex)),mul(h,ey));
    const cy = add(add(by,mul(a,ey)),mul(h,ex));
    const bcx = sub(cx,bx), bcy = sub(cy,by), length = norm2(bcx,bcy);
    const ux = div(bcx,length), uy = div(bcy,length);
    const x = sub(add(bx,mul(px,ux)),mul(py,uy));
    const y = add(add(by,mul(px,uy)),mul(py,ux));
    out.push(returnFrames ? { A:[0,0], B:[bx,by], C:[cx,cy], D:[1,0], P:[x,y] } : [x,y]);
  }
  return out;
}
export function trace(params,n = 256) {
  vector5(params,"params"); sampleCount(n,1); counts.trace++;
  return geometry(params,n);
}
export function frames(params,n = 240) {
  vector5(params,"params"); sampleCount(n,1); counts.frames++;
  return geometry(params,n,true);
}

// NumPy-compatible resampler for reporting; unlike the differentiable Torch
// path below, zero-length segments are not lengthened by an epsilon.
export function resample(points,n) {
  pointArray(points); sampleCount(n);
  const m = points.length, boundaries = [0];
  for (let i = 0; i < m; i++) {
    const p = points[i], q = points[(i+1)%m];
    boundaries.push(boundaries[i]+Math.hypot(q[0]-p[0],q[1]-p[1]));
  }
  const total = boundaries[m];
  if (total < 1e-12) return Array.from({length:n},() => points[0].slice());
  let j = 0;
  return Array.from({length:n},(_,i) => {
    const t = total*i/n;
    while (j+1 < m && boundaries[j+1] <= t) j++;
    const p = points[j], q = points[(j+1)%m], ds = boundaries[j+1]-boundaries[j];
    const alpha = ds > 0 ? (t-boundaries[j])/ds : 0;
    return [p[0]+alpha*(q[0]-p[0]),p[1]+alpha*(q[1]-p[1])];
  });
}
function differentiableResample(points,n) {
  const m = points.length;
  if (m < 2) return Array.from({length:n},() => points[0].slice());
  const lengths = [], boundaries = [0];
  for (let i = 0; i < m; i++) {
    const p = points[i], q = points[(i+1)%m];
    const length = clamp(norm2(sub(q[0],p[0]),sub(q[1],p[1])),1e-12,Infinity);
    lengths.push(length); boundaries.push(add(boundaries[i],length));
  }
  const total = clamp(boundaries[m],1e-12,Infinity), out = [];
  let j = 0;
  for (let i = 0; i < n; i++) {
    const t = mul(total,i/n);
    while (j+1 < m && value(boundaries[j+1]) <= value(t)) j++;
    const alpha = clamp(div(sub(t,boundaries[j]),lengths[j]),0,1);
    const p = points[j], q = points[(j+1)%m];
    out.push([add(p[0],mul(alpha,sub(q[0],p[0]))),add(p[1],mul(alpha,sub(q[1],p[1])))]);
  }
  return out;
}
function centered(points) {
  let x = 0, y = 0;
  for (const p of points) { x = add(x,p[0]); y = add(y,p[1]); }
  x = div(x,points.length); y = div(y,points.length);
  return points.map(p => [sub(p[0],x),sub(p[1],y)]);
}
function normalizedScorePoints(points) {
  const zeroMean = centered(differentiableResample(points,256));
  let variance = 0;
  for (const p of zeroMean) variance = add(variance,add(square(p[0]),square(p[1])));
  const radius = sqrt(clamp(div(variance,256),1e-12,Infinity));
  return centered(differentiableResample(zeroMean.map(p => [div(p[0],radius),div(p[1],radius)]),180));
}
function cyclicShift(target,candidate) {
  const n = target.length;
  let best = -Infinity, bestShift = 0;
  for (let shift = 0; shift < n; shift++) {
    let re = 0, im = 0;
    for (let i = 0; i < n; i++) {
      const t = target[i], c = candidate[(i+shift)%n];
      const tx = value(t[0]), ty = value(t[1]), cx = value(c[0]), cy = value(c[1]);
      re += tx*cx+ty*cy; im += tx*cy-ty*cx;
    }
    const magnitude = re*re+im*im;
    if (magnitude > best) { best = magnitude; bestShift = shift; }
  }
  return bestShift;
}
function objective(logits,targetScore) {
  const scale = 1-BOUNDARY_EPSILON;
  const unit = logits.map((x,i) => {
    const t = Math.tanh(x), dual = [scale*t,0,0,0,0,0];
    dual[i+1] = scale*(1-t*t); return dual;
  });
  counts.trace++; counts.refinement++;
  const candidate = normalizedScorePoints(geometry(decode(unit),256));
  const shift = cyclicShift(targetScore,candidate), n = targetScore.length;
  let a = 0, b = 0;
  for (let i = 0; i < n; i++) {
    const c = candidate[(i+shift)%n], t = targetScore[i];
    a = add(a,add(mul(c[0],t[0]),mul(c[1],t[1])));
    b = add(b,sub(mul(c[0],t[1]),mul(c[1],t[0])));
  }
  const magnitude = clamp(norm2(a,b),1e-12,Infinity);
  const cosine = div(a,magnitude), sine = div(b,magnitude);
  let loss = 0;
  for (let i = 0; i < n; i++) {
    const c = candidate[(i+shift)%n], t = targetScore[i];
    const x = sub(mul(cosine,c[0]),mul(sine,c[1]));
    const y = add(mul(sine,c[0]),mul(cosine,c[1]));
    loss = add(loss,add(square(sub(t[0],x)),square(sub(t[1],y))));
  }
  loss = div(loss,n);
  const result = {loss:value(loss),gradient:loss.slice(1),shift,unit:unit.map(value)};
  if (![result.loss,...result.gradient].every(Number.isFinite)) throw new Error("non-finite exact-FK objective or gradient");
  return result;
}
export function lossAndGradient(logits,target) {
  vector5(logits,"logits"); pointArray(target);
  return objective(Array.from(logits),normalizedScorePoints(target));
}

/** Retain pre-update hard-best iterates and return true post-update endpoints.
 * Endpoint production scores and original candidates belong to the caller.
 * onProgress receives {step, steps, exactFkEvaluations, bestLosses}.
 */
export async function refine(startsUnit,target,{steps = 100,lr = 0.03} = {},onProgress) {
  if (!Array.isArray(startsUnit) || !startsUnit.length) throw new TypeError("startsUnit must contain one or more unit vectors");
  startsUnit.forEach(u => vector5(u,"start")); pointArray(target);
  if (!Number.isInteger(steps) || steps <= 0 || !Number.isFinite(lr) || lr <= 0) throw new RangeError("steps and lr must be positive");
  const scale = 1-BOUNDARY_EPSILON, n = startsUnit.length;
  const logits = startsUnit.map(u => Array.from(u,x => Math.atanh(Math.max(-scale+1e-7,Math.min(scale-1e-7,x))/scale)));
  const moments = logits.map(() => [Array(DIM).fill(0),Array(DIM).fill(0)]);
  const targetScore = normalizedScorePoints(target);
  const best = Array(n).fill(Infinity), retainedUnits = Array(n), retainedSteps = Array(n).fill(0);
  let firstLosses, firstShifts, lastLosses, lastShifts;
  for (let step = 0; step < steps; step++) {
    const losses = [], shifts = [];
    for (let j = 0; j < n; j++) {
      const result = objective(logits[j],targetScore);
      losses.push(result.loss); shifts.push(result.shift);
      if (result.loss < best[j]) { best[j] = result.loss; retainedUnits[j] = result.unit; retainedSteps[j] = step; }
      const [m,v] = moments[j], bc1 = 1-0.9**(step+1), bc2 = 1-0.999**(step+1);
      for (let d = 0; d < DIM; d++) {
        const g = result.gradient[d];
        m[d] = 0.9*m[d]+0.1*g; v[d] = 0.999*v[d]+0.001*g*g;
        logits[j][d] -= lr*(m[d]/bc1)/(Math.sqrt(v[d]/bc2)+1e-8);
      }
    }
    if (step === 0) { firstLosses = losses.slice(); firstShifts = shifts.slice(); }
    lastLosses = losses; lastShifts = shifts;
    if (onProgress) await onProgress({step:step+1,steps,exactFkEvaluations:(step+1)*n,bestLosses:best.slice()});
    // Yield without requiring requestAnimationFrame (also usable in a Worker).
    if (step%4 === 3) await new Promise(resolve => setTimeout(resolve,0));
  }
  return {
    retainedUnits,
    endpointUnits:logits.map(row => row.map(x => scale*Math.tanh(x))),
    audit:{
      objective:"squared_proper_similarity_normalized_shape_error_with_full_discrete_cyclic_assignment",
      optimizer:"Adam", arithmetic:"float64_forward_mode_autodiff", steps, learning_rate:lr,
      boundary_epsilon:BOUNDARY_EPSILON, cyclic_score_points:180, cyclic_max_shift:90,
      exact_fk_candidate_evaluations:steps*n,
      first_update_objective:firstLosses, last_update_pre_step_objective:lastLosses,
      retained_pre_update_objective:best, retained_pre_update_step:retainedSteps,
      final_endpoint_evaluated_inside_optimizer:false,
      first_update_cyclic_shift:firstShifts, last_update_pre_step_cyclic_shift:lastShifts,
      iterate_retention:"best already-evaluated pre-update iterate per start; caller scores this and the true endpoint",
    },
  };
}

const trigTables = new Map();
function transformTable(n) {
  if (!trigTables.has(n)) {
    trigTables.set(n,Array.from({length:n},(_,k) => [Math.cos(2*Math.PI*k/n),Math.sin(2*Math.PI*k/n)]));
  }
  return trigTables.get(n);
}
function dft(points) {
  const n = points.length, table = transformTable(n), result = [];
  for (let k = 0; k < n; k++) {
    let re = 0, im = 0;
    for (let j = 0; j < n; j++) {
      const [co,si] = table[(k*j)%n], [x,y] = points[j];
      re += x*co+y*si; im += y*co-x*si;
    }
    result.push([re,im]);
  }
  return result;
}

// Bounded Brent minimization, with the tolerances and stopping convention used
// by the reference score (xatol=1e-8, maxiter/maxfun=64).
function boundedMinimum(f,left,right) {
  const golden = 0.5*(3-Math.sqrt(5)), sqrtEps = Math.sqrt(2.2e-16), xatol = 1e-8;
  let a = left, b = right, x = a+golden*(b-a), w = x, v = x;
  let fx = f(x), fw = fx, fv = fx, e = 0, delta = 0, calls = 1;
  while (calls < 64) {
    const mid = 0.5*(a+b), tol = sqrtEps*Math.abs(x)+xatol/3, tol2 = 2*tol;
    if (Math.abs(x-mid) <= tol2-0.5*(b-a)) return {x,fun:fx,success:true};
    let useGolden = true;
    if (Math.abs(e) > tol) {
      const r = (x-w)*(fx-fv);
      let q = (x-v)*(fx-fw), p = (x-v)*q-(x-w)*r;
      q = 2*(q-r); if (q > 0) p = -p; q = Math.abs(q);
      const oldE = e; e = delta;
      if (Math.abs(p) < Math.abs(0.5*q*oldE) && p > q*(a-x) && p < q*(b-x)) {
        delta = p/q; const u = x+delta; useGolden = false;
        if (u-a < tol2 || b-u < tol2) delta = tol*(Math.sign(mid-x)||1);
      }
    }
    if (useGolden) { e = x >= mid ? a-x : b-x; delta = golden*e; }
    const u = x+(Math.sign(delta)||1)*Math.max(Math.abs(delta),tol), fu = f(u); calls++;
    if (fu <= fx) {
      if (u >= x) a = x; else b = x;
      v = w; fv = fw; w = x; fw = fx; x = u; fx = fu;
    } else {
      if (u < x) a = u; else b = u;
      if (fu <= fw || w === x) { v = w; fv = fw; w = u; fw = fu; }
      else if (fu <= fv || v === x || v === w) { v = u; fv = fu; }
    }
  }
  return {x,fun:fx,success:false};
}

/** Full reporting metric: proper similarity + fractional 180-sample phase.
 * The DFT and Parseval form below equal the reference FFT/ifft residual.
 * Traversal reversal/reflection are NOT permitted by this score.
 */
export function score(target,candidate) {
  pointArray(target); pointArray(candidate); counts.score++;
  const n = 180, tp = resample(target,n), cp = resample(candidate,n);
  const mean = points => points.reduce((a,p) => [a[0]+p[0]/points.length,a[1]+p[1]/points.length],[0,0]);
  const tm = mean(tp), cm = mean(cp);
  const t = tp.map(p => [p[0]-tm[0],p[1]-tm[1]]), c = cp.map(p => [p[0]-cm[0],p[1]-cm[1]]);
  const ts = dft(t), cs = dft(c);
  const candidateNorm = c.reduce((sum,p) => sum+p[0]*p[0]+p[1]*p[1],0);
  if (!(candidateNorm > 1e-18)) throw new RangeError("proper-similarity alignment failed: degenerate candidate");
  // Swap arguments: the reporting metric shifts TARGET, unlike hard loss.
  const integerShift = cyclicShift(c,t);
  function alignmentAt(shift) {
    const shifted = [], inverseNorm = 1/(n*candidateNorm);
    let mr = 0, mi = 0;
    for (let k = 0; k < n; k++) {
      const frequency = (k < n/2 ? k : k-n)/n, angle = 2*Math.PI*frequency*shift;
      const co = Math.cos(angle), si = Math.sin(angle), [tr,ti] = ts[k];
      const ar = tr*co-ti*si, ai = tr*si+ti*co, [cr,ci] = cs[k];
      shifted.push([ar,ai]); mr += cr*ar+ci*ai; mi += cr*ai-ci*ar;
    }
    mr *= inverseNorm; mi *= inverseNorm;
    let residual = 0;
    for (let k = 0; k < n; k++) {
      const [cr,ci] = cs[k], [tr,ti] = shifted[k];
      const dx = mr*cr-mi*ci-tr, dy = mr*ci+mi*cr-ti;
      residual += dx*dx+dy*dy;
    }
    return {mse:residual/(n*n),mr,mi,shift};
  }
  let best = alignmentAt(integerShift);
  const refined = boundedMinimum(s => alignmentAt(s).mse,integerShift-1,integerShift+1);
  if (refined.success && Number.isFinite(refined.fun)) {
    const result = alignmentAt(refined.x); if (result.mse < best.mse) best = result;
  }
  const radiusPoints = resample(target,256), rm = mean(radiusPoints);
  const radius = Math.sqrt(radiusPoints.reduce((sum,p) => sum+(p[0]-rm[0])**2+(p[1]-rm[1])**2,0)/256);
  if (!(radius > 1e-12) || !Number.isFinite(radius)) throw new RangeError("target curve is degenerate");
  const scale = Math.hypot(best.mr,best.mi), theta = Math.atan2(best.mi,best.mr);
  const transform = {k:scale,theta,
    tx:tm[0]-(best.mr*cm[0]-best.mi*cm[1]),
    ty:tm[1]-(best.mi*cm[0]+best.mr*cm[1]),
    cyclic_shift_samples_180:((best.shift%n)+n)%n};
  const error = Math.sqrt(Math.max(best.mse,0))/radius;
  if (!(scale > 0) || !Number.isFinite(error) || !Object.values(transform).every(Number.isFinite)) throw new RangeError("proper-similarity score or transform is invalid");
  return {error,transform};
}
