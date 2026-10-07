/** Pure viewport and session-history helpers. No DOM, storage, network, or inference. */
const MIN_ZOOM = .001;
const MAX_ZOOM = 1e7;

function checkedCamera(camera) {
  if (!camera || ![camera.x, camera.y, camera.scale].every(Number.isFinite) || camera.scale <= 0) {
    throw new TypeError("Camera position and positive scale must be finite.");
  }
  return {x: camera.x, y: camera.y, scale: camera.scale};
}

function finiteCamera(camera) {
  if (![camera.x, camera.y, camera.scale].every(Number.isFinite)) {
    throw new RangeError("Viewport coordinates exceed the supported range.");
  }
  return camera;
}

/** Screen deltas move the content, not the world-space camera; world y is up. */
export function panCamera(camera, dx, dy) {
  const next = checkedCamera(camera);
  if (![dx, dy].every(Number.isFinite)) throw new TypeError("Pan deltas must be finite.");
  next.x -= dx / next.scale;
  next.y += dy / next.scale;
  return finiteCamera(next);
}

/** Keep the same world point under a screen-space cursor while zooming. */
export function zoomCamera(camera, factor, anchor, width, height) {
  const next = checkedCamera(camera);
  if (![factor, width, height].every(Number.isFinite) || factor <= 0 || width <= 0 || height <= 0
      || !Array.isArray(anchor) || anchor.length !== 2 || !anchor.every(Number.isFinite)) {
    throw new TypeError("Zoom requires positive finite dimensions/factor and a finite 2-D anchor.");
  }
  // A large mechanism may legitimately fit below MIN_ZOOM. Never snap such a
  // camera to the ordinary range: permit movement inward, but not farther out.
  const low = Math.min(MIN_ZOOM, next.scale), high = Math.max(MAX_ZOOM, next.scale);
  const scale = Math.max(low, Math.min(high, next.scale * factor));
  if (scale === next.scale) return next;
  const offsetX = anchor[0] - width / 2, offsetY = anchor[1] - height / 2;
  next.x += offsetX / next.scale - offsetX / scale;
  next.y -= offsetY / next.scale - offsetY / scale;
  next.scale = scale;
  return finiteCamera(next);
}

function checkJson(value, seen = new Set()) {
  if (value === null || typeof value === "string" || typeof value === "boolean") return;
  if (typeof value === "number" && Number.isFinite(value)) return;
  if (!value || typeof value !== "object") throw new TypeError("History records must be JSON-safe.");
  if (seen.has(value)) throw new TypeError("History records must not be circular.");
  if (!Array.isArray(value) && Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null) {
    throw new TypeError("History records must contain only plain JSON objects.");
  }
  if (Reflect.ownKeys(value).some((key) => typeof key === "symbol")) {
    throw new TypeError("History records must not contain symbol keys.");
  }
  seen.add(value);
  if (Array.isArray(value)) {
    for (let index = 0; index < value.length; index++) {
      if (!Object.hasOwn(value, index)) throw new TypeError("History arrays must not contain holes.");
      checkJson(value[index], seen);
    }
  } else {
    for (const item of Object.values(value)) checkJson(item, seen);
  }
  seen.delete(value);
}

/** Bounded, tab-lifetime snapshots. No persistence or silent sketch uploading. */
export function createHistory(limit = 5) {
  if (!Number.isInteger(limit) || limit < 1 || limit > 100) throw new RangeError("History limit must be an integer between 1 and 100.");
  let nextId = 1;
  const records = [];
  return Object.freeze({
    push(record) {
      checkJson(record);
      const entry = {id: nextId++, record: structuredClone(record)};
      records.unshift(entry);
      if (records.length > limit) records.length = limit;
      return entry.id;
    },
    list() { return structuredClone(records); },
    get(id) {
      const entry = records.find((item) => item.id === id);
      return entry ? structuredClone(entry.record) : null;
    },
    clear() { records.length = 0; },
  });
}
