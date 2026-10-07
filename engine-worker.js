import {loadDenoiser,sample} from './denoiser.js';
import {unitToPhysical,trace,frames,score,refine,resample,getEvaluationCounts,resetEvaluationCounts} from './kinematics.js';
import {prepareCurve,conditionPool} from './curve-input.js';

let busy=false;
function strict(p) {
  const [a,b,c]=p;
  return p.every(Number.isFinite)&&a>0&&a<1&&b+c>1+a&&Math.abs(b-c)<1-a;
}
async function run(points,options,progress) {
  const started=performance.now();
  resetEvaluationCounts();
  const target=prepareCurve(points);
  if(options.profile && !['quality_v2','fast_v1'].includes(options.profile))throw new Error('Unsupported browser profile.');
  const quality=options.profile!=='fast_v1',k=quality?64:32,steps=quality?50:25;
  // loadDenoiser resolves only after checking the downloaded ONNX SHA-256.
  const metadata=await loadDenoiser(progress);
  progress({message:'Encoding 32 orientation / phase / direction conditions'});
  const units=await sample(conditionPool(target,k),{steps,seed:options.seed??20260726},progress);
  let calls=0;
  function evaluate(unit,source) {
    const p=unitToPhysical(Array.from(unit));
    if(!strict(p))throw new Error('A candidate failed the strict full-cycle assembly audit.');
    const curve=trace(p,256);calls++;
    const matched=score(target,curve);
    if(!Number.isFinite(matched.error)||!(matched.transform.k>0))throw new Error('Curve alignment failed.');
    return {unit:Array.from(unit),p,curve,matched,source};
  }
  progress({message:`Scoring ${k} candidates through exact kinematics`});
  const initial=units.map(u=>evaluate(u,'diffusion'));
  initial.sort((a,b)=>a.matched.error-b.matched.error);
  let pool=[...initial],refinement=null;
  if(quality) {
    refinement=await refine(initial.slice(0,4).map(x=>x.unit),resample(target,256),{steps:100,lr:0.03},
      p=>progress({...p,message:`Exact-FK automatic-differentiation refinement ${p.step}/${p.steps}`}));
    if(refinement.audit.exact_fk_candidate_evaluations!==400)throw new Error('Refinement query budget audit failed.');
    calls+=refinement.audit.exact_fk_candidate_evaluations;
    pool.push(...refinement.retainedUnits.map(u=>evaluate(u,'retained_refinement')),
      ...refinement.endpointUnits.map(u=>evaluate(u,'endpoint_refinement')));
  }
  pool.sort((a,b)=>a.matched.error-b.matched.error);
  const selected=[];
  for(const candidate of pool) {
    if(selected.some(x=>Math.hypot(...x.unit.map((v,i)=>v-candidate.unit[i]))<=0.10))continue;
    selected.push(candidate);if(selected.length===6)break;
  }
  const toSolution=c=>({
    params:{r1:1,r2:c.p[0],r3:c.p[1],r4:c.p[2],px:c.p[3],py:c.p[4]},
    transform:c.matched.transform,error_percent:100*c.matched.error,
    frames:frames(c.p,240),coupler_curve:c.curve,source:c.source,
    mechanism_family:'strict_full_crank_fourbar_branch_plus',path_closed:true,
  });
  const solutions=selected.map(toSolution);
  // Keep the actual raw winner of this same candidate pool. Its scored trace
  // and alignment are reused; only animation frames may need to be generated.
  // This is a best-of-pool comparison, not a claimed trajectory of one start.
  const rawSelectedIndex=selected.indexOf(initial[0]);
  const initialBest=rawSelectedIndex>=0?solutions[rawSelectedIndex]:toSolution(initial[0]);
  const comparisonDisplayTraces=rawSelectedIndex>=0?0:1;
  if(calls!==(quality?472:32)||getEvaluationCounts().trace!==calls)throw new Error('Mechanics query budget audit failed.');
  if(getEvaluationCounts().frames!==solutions.length+comparisonDisplayTraces)throw new Error('Display animation budget audit failed.');
  return {
    solutions,clean_curve:target,path:{closed:true},backend:'browser_onnx_exact_fk',
    profile:quality?'browser_quality_v1':'browser_fast_v1',
    metric:'web_similarity_rms_v1',
    comparison:{kind:'same_pool_best',refinement_applied:quality,
      initial_best:initialBest,final_best_index:0},
    reproducibility:{seed:options.seed??20260726,requested_profile:options.profile??'quality_v2',
      model_sha256:metadata.export_sha256,demo_version:'post-submission-v2'},
    audit:{selection_fk_calls:calls,display_traces:solutions.length+comparisonDisplayTraces,
      solution_display_traces:solutions.length,comparison_display_traces:comparisonDisplayTraces,
      candidates:k,ddim_steps:steps,
      raw_best_error_percent:100*initial[0].matched.error,
      selection_policy:'score_sorted_unit_distance_filter',unit_distance_threshold:0.10,
      diversity_guarantee:false,
      strict_valid_candidates:k,total_candidates:k,
      refinement_steps:quality?100:0,
      gradient:'forward_mode_automatic_differentiation',
      preprocessing:'closed_polyline_no_smoothing',
      random_stream:'browser_seeded_rng_not_pytorch',
      runtime:'onnxruntime_web_wasm_cpu',
    },
    timing:{browser_compute_seconds:(performance.now()-started)/1000},
  };
}
self.onmessage=async({data})=>{
  const {id,type,points,options}=data;
  if(busy){self.postMessage({id,type:'error',message:'A computation is already running.'});return;}
  busy=true;
  const progress=p=>self.postMessage({id,type:'progress',progress:p});
  try {
    const result=type==='initialize'?await loadDenoiser(progress):await run(points,options||{},progress);
    self.postMessage({id,type:'result',result});
  } catch(error) {
    self.postMessage({id,type:'error',message:String(error.message||error),code:error.code});
  } finally {busy=false;}
};
