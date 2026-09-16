// All inference stays in a dedicated local worker; no inference API exists.
let worker=null, nextId=0;
const pending=new Map();
function ensureWorker() {
  if (worker) return worker;
  if (typeof Worker==='undefined' || typeof WebAssembly==='undefined') {
    const error=new Error('This browser needs WebAssembly and module workers.');
    error.code='UNSUPPORTED';throw error;
  }
  worker=new Worker(new URL('./engine-worker.js',import.meta.url),{type:'module'});
  worker.onmessage=({data})=>{
    const call=pending.get(data.id);if(!call)return;
    if(data.type==='progress') {call.onProgress?.(data.progress);return;}
    pending.delete(data.id);
    if(data.type==='error') {const e=new Error(data.message);e.code=data.code;call.reject(e);}
    else call.resolve(data.result);
  };
  worker.onerror=()=>{
    for(const call of pending.values()) call.reject(new Error('Browser worker failed. Reload the model and retry.'));
    pending.clear();worker?.terminate();worker=null;
  };
  return worker;
}
function request(type,payload,onProgress) {
  return new Promise((resolve,reject)=>{
    let id;
    try {
      const w=ensureWorker();id=++nextId;
      pending.set(id,{resolve,reject,onProgress});w.postMessage({id,type,...payload});
    } catch(e){if(id!==undefined)pending.delete(id);reject(e);}
  });
}
export const initializeEngine=(onProgress)=>request('initialize',{},onProgress);
export const synthesize=(points,options={},onProgress)=>request('synthesize',{points,options},onProgress);
export function cancelSynthesis() {
  worker?.terminate();worker=null;
  for(const call of pending.values())call.reject(new Error('Computation cancelled; reload the model to continue.'));
  pending.clear();
}
