// Browser input protocol: explicit closed polyline, no hidden smoothing.
import {resample} from './kinematics.js';

export function prepareCurve(points) {
  if (!Array.isArray(points) || points.length < 5 || points.length > 8192) {
    throw new Error('Draw a non-degenerate curve with 5–8192 points.');
  }
  const clean = [];
  for (const p of points) {
    if (!Array.isArray(p) || p.length !== 2 || !p.every(Number.isFinite) || p.some(x => Math.abs(x) > 1e8)) {
      throw new Error('The stroke contains invalid coordinates.');
    }
    if (!clean.length || Math.hypot(p[0]-clean.at(-1)[0], p[1]-clean.at(-1)[1]) > 1e-10) clean.push([...p]);
  }
  if (clean.length < 5) throw new Error('Draw a longer curve before generating.');
  const sampled = resample(clean, 256);
  const mean = [0, 0];
  sampled.forEach(p => {mean[0] += p[0]/256; mean[1] += p[1]/256;});
  const variance = sampled.reduce((s,p) => s+(p[0]-mean[0])**2+(p[1]-mean[1])**2,0)/256;
  if (!(variance > 1e-16)) throw new Error('The stroke has no measurable extent.');
  return clean;
}

export function encodeFourier(points) {
  const p = resample(points, 256);
  const mean = [0,0];
  p.forEach(q => {mean[0]+=q[0]/p.length;mean[1]+=q[1]/p.length;});
  const centered = p.map(q=>[q[0]-mean[0],q[1]-mean[1]]);
  const radius = Math.sqrt(centered.reduce((s,q)=>s+q[0]**2+q[1]**2,0)/p.length);
  if (!(radius > 1e-12)) throw new Error('Cannot encode a degenerate curve.');
  const re=[],im=[];
  for (const k of [...Array.from({length:12},(_,i)=>i+1),...Array.from({length:12},(_,i)=>-i-1)]) {
    let a=0,b=0;
    for (let j=0;j<p.length;j++) {
      const angle=2*Math.PI*k*j/p.length,c=Math.cos(angle),s=Math.sin(angle);
      a+=centered[j][0]*c+centered[j][1]*s;
      b+=centered[j][1]*c-centered[j][0]*s;
    }
    re.push(a/p.length/radius); im.push(b/p.length/radius);
  }
  return [...re,...im];
}

export function transformCode(code,phase,reverse) {
  const out=Array(48).fill(0);
  for (let i=0;i<24;i++) {
    const source=reverse?(i+12)%24:i;
    const k=i<12?i+1:-(i-12+1),a=2*Math.PI*k*phase;
    out[i]=code[source]*Math.cos(a)-code[source+24]*Math.sin(a);
    out[i+24]=code[source]*Math.sin(a)+code[source+24]*Math.cos(a);
  }
  return out;
}

export function conditionPool(points,count=64) {
  if (count!==32 && count!==64) throw new Error('Candidate count must be 32 or 64.');
  let p=resample(points,256);
  const center=[0,0];p.forEach(q=>{center[0]+=q[0]/256;center[1]+=q[1]/256;});
  p=p.map(q=>[q[0]-center[0],q[1]-center[1]]);
  let xx=0,xy=0,yy=0;
  p.forEach(q=>{xx+=q[0]**2;xy+=q[0]*q[1];yy+=q[1]**2;});
  // Sign of the principal axis is immaterial to this full 4-orientation pool.
  const principal=0.5*Math.atan2(2*xy,xx-yy),pool=[];
  for (let orientation=0;orientation<4;orientation++) {
    const angle=-principal+orientation*Math.PI/2,c=Math.cos(angle),s=Math.sin(angle);
    const code=encodeFourier(p.map(q=>[c*q[0]-s*q[1],s*q[0]+c*q[1]]));
    for (const reverse of [false,true]) for (const phase of [0,0.25,0.5,0.75]) {
      const branch=transformCode(code,phase,reverse);
      for (let i=0;i<count/32;i++) pool.push([...branch]);
    }
  }
  return pool;
}
