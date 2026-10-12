from pathlib import Path
import json

OUT = Path(__file__).resolve().parents[1] / "visualizations"
data = json.loads((OUT / "joint-base-relationship-data.json").read_text())
payload = json.dumps(data, separators=(",", ":"))

html = r'''<div id="joint-base-relationship" role="figure" aria-label="Absolute predicted handle and panel joint angles at rollout start in simulation and real-world recordings">
  <style>
    #joint-base-relationship { color:var(--foreground); font:400 14px/1.35 system-ui,sans-serif; max-width:1024px; }
    #joint-base-relationship h2 { font-size:18px; font-weight:500; margin:0 0 4px; }
    #joint-base-relationship p { color:var(--muted-foreground); margin:0 0 14px; }
    #joint-base-relationship .grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:20px; }
    #joint-base-relationship .wide { grid-column:1/-1; }
    #joint-base-relationship svg { width:100%; height:auto; display:block; overflow:visible; }
    #joint-base-relationship .axis { stroke:var(--border); stroke-width:1; }
    #joint-base-relationship .gridline { stroke:var(--border); stroke-opacity:.35; stroke-width:1; }
    #joint-base-relationship text { fill:var(--foreground); font-size:11px; }
    #joint-base-relationship .muted { fill:var(--muted-foreground); }
    #joint-base-relationship .simline { fill:none; stroke:var(--viz-series-1); stroke-opacity:.24; stroke-width:1.2; }
    #joint-base-relationship .median { fill:none; stroke:var(--viz-series-1); stroke-width:2.5; }
    #joint-base-relationship .panel { fill:none; stroke:var(--viz-series-2); stroke-opacity:.24; stroke-width:1.2; }
    #joint-base-relationship .panelmedian { fill:none; stroke:var(--viz-series-2); stroke-width:2.5; }
    #joint-base-relationship .handlesim { fill:none; stroke:var(--viz-series-1); stroke-opacity:.20; stroke-width:1; }
    #joint-base-relationship .jointnorm { fill:none; stroke:var(--viz-series-2); stroke-opacity:.20; stroke-width:1; }
    #joint-base-relationship .jointmedian { fill:none; stroke:var(--viz-series-2); stroke-width:2.5; }
    #joint-base-relationship .paneltrace { fill:none; stroke:var(--viz-series-3); stroke-opacity:.20; stroke-width:1; }
    #joint-base-relationship .paneltrace-median { fill:none; stroke:var(--viz-series-3); stroke-width:2.5; }
    #joint-base-relationship .realnote { fill:var(--muted-foreground); }
    #joint-base-relationship .legend { display:flex; flex-wrap:wrap; gap:10px 18px; color:var(--muted-foreground); margin:3px 0 8px; }
    #joint-base-relationship .sw { display:inline-block; width:18px; border-top:2px solid currentColor; vertical-align:middle; margin-right:5px; }
    #joint-base-relationship .dash { border-top-style:dashed; }
    @media (max-width:700px) { #joint-base-relationship .grid { grid-template-columns:1fr; } #joint-base-relationship .wide { grid-column:auto; } }
  </style>
  <h2>Starting joint predictions: simulation vs real</h2>
  <p>Absolute predicted joint angles in radians. Simulation traces begin at rollout step 0; the thick line is the median across 12 episodes.</p>
  <div class="legend"><span><i class="sw" style="color:var(--viz-series-2)"></i>|handle joint (latch)|</span><span><i class="sw" style="color:var(--viz-series-3)"></i>|panel joint (hinge)|</span></div>
  <div class="grid"><div><svg id="sim" viewBox="0 0 480 320" role="img" aria-label="Absolute simulated handle and panel joint predictions at rollout start"></svg></div><div><svg id="real" viewBox="0 0 480 320" role="img" aria-label="Real-world prediction chart pending accessible recordings"></svg></div><div class="wide"><svg id="sim-full" viewBox="0 0 1000 260" role="img" aria-label="Absolute handle and panel joint prediction trajectories throughout simulation episodes"></svg></div></div>
  <script>
  (()=>{
    const D=__DATA__;
    const root=document.getElementById('joint-base-relationship');
    const ns='http://www.w3.org/2000/svg';
    const css=getComputedStyle(root);
    const C={fg:css.getPropertyValue('--foreground').trim(),muted:css.getPropertyValue('--muted-foreground').trim(),border:css.getPropertyValue('--border').trim(),s1:css.getPropertyValue('--viz-series-1').trim(),s2:css.getPropertyValue('--viz-series-2').trim(),s3:css.getPropertyValue('--viz-series-3').trim(),s4:css.getPropertyValue('--viz-series-4').trim(),s5:css.getPropertyValue('--viz-series-5').trim()};
    function el(tag,a={}){const x=document.createElementNS(ns,tag); for(const [k,v] of Object.entries(a))x.setAttribute(k,v); return x;}
    function txt(svg,x,y,s,cls=''){const t=el('text',{x,y,class:cls});t.textContent=s;svg.append(t);}
    function path(vals,x,y,w,h,min,max,cls){const n=vals.length; let d=''; for(let i=0;i<n;i++){const X=x+w*(n===1?0:i/(n-1)),Y=y+h-(vals[i]-min)/(max-min||1)*h; d+=(i?'L':'M')+X.toFixed(1)+','+Y.toFixed(1);} return el('path',{d,class:cls});}
    function frame(svg,x,y,w,h,min,max,ylabel,xlabel='normalized phase: handle onset → panel onset'){svg.append(el('rect',{x,y,width:w,height:h,fill:'none',class:'axis'})); [0,.5,1].forEach(q=>{const Y=y+h*(1-q);svg.append(el('line',{x1:x,x2:x+w,y1:Y,y2:Y,class:'gridline'}));txt(svg,x-8,Y+4,(min+(max-min)*q).toFixed(2),'muted');});txt(svg,x+w/2,y+h+25,xlabel);txt(svg,x-47,y+h/2,ylabel,'muted');}
    const N=120, sims=D.sim;
    const samples=(key,i)=>{const a=sims.map(e=>e[key]); return a.map(v=>v[Math.min(v.length-1,Math.floor(i*(v.length-1)/(N-1)))])};
    const med=a=>a.slice().sort((x,y)=>x-y)[Math.floor(a.length/2)];
    function series(svg, valuesByEpisode, colorClass, x,y,w,h,hi, medianClass){
      valuesByEpisode.forEach(v=>svg.append(path(v,x,y,w,h,0,hi,colorClass)));
      const n=Math.min(...valuesByEpisode.map(v=>v.length));
      const m=Array.from({length:n},(_,i)=>med(valuesByEpisode.map(v=>v[i])));
      svg.append(path(m,x,y,w,h,0,hi,medianClass));
    }
    function axes(svg,x,y,w,h,hi,title,xlabel){
      txt(svg,x,y-3,title,'muted');
      svg.append(el('rect',{x,y,width:w,height:h,fill:'none',class:'axis'}));
      [0,.5,1].forEach(q=>{const Y=y+h*(1-q);svg.append(el('line',{x1:x,x2:x+w,y1:Y,y2:Y,class:'gridline'}));txt(svg,x-5,Y+4,(hi*q).toFixed(2),'muted');});
      txt(svg,x+w/2,y+h+23,xlabel);
    }
    const drawStart=()=>{
      const svg=document.getElementById('sim'),x=45,y=34,w=390,h=210;
      const hvals=D.sim.map(e=>e.initial_joint_prediction_abs[1]),pvals=D.sim.map(e=>e.initial_joint_prediction_abs[0]);
      const hi=Math.max(...hvals,...pvals)*1.12;
      axes(svg,x,y,w,h,hi,'Simulation, step 0 (rad)','12 recorded episodes');
      hvals.forEach((v,i)=>svg.append(el('circle',{cx:x+24+i*(w-48)/(hvals.length-1),cy:y+h-v/hi*h,r:4,fill:'var(--viz-series-2)'})));
      pvals.forEach((v,i)=>svg.append(el('rect',{x:x+20+i*(w-48)/(pvals.length-1),y:y+h-v/hi*h-3,width:6,height:6,fill:'var(--viz-series-3)'})));
      const hmed=med(hvals),pmed=med(pvals);
      svg.append(el('line',{x1:x+8,x2:x+w-8,y1:y+h-hmed/hi*h,y2:y+h-hmed/hi*h,stroke:'var(--viz-series-2)','stroke-width':2.5}));
      svg.append(el('line',{x1:x+8,x2:x+w-8,y1:y+h-pmed/hi*h,y2:y+h-pmed/hi*h,stroke:'var(--viz-series-3)','stroke-width':2.5,'stroke-dasharray':'5 3'}));
    };
    const drawReal=()=>{const svg=document.getElementById('real');txt(svg,20,30,'Real rollout, step 0 (rad)','muted');txt(svg,20,76,'Real joint-prediction PTs are unavailable.','realnote');txt(svg,20,101,'Glorbo4 at 128.2.213.13: SSH timed out.','realnote');txt(svg,20,126,'No local real prediction PT was found.','realnote');txt(svg,20,174,'This panel is intentionally blank; no real values','realnote');txt(svg,20,193,'are inferred or substituted.','realnote');};
    const drawFull=()=>{const svg=document.getElementById('sim-full'),x=75,y=26,w=890,h=170;const hvals=D.sim.map(e=>e.full_handle_joint_abs),pvals=D.sim.map(e=>e.full_panel_joint_abs);const hi=Math.max(...hvals.flat(),...pvals.flat())*1.05;axes(svg,x,y,w,h,hi,'Simulation prediction trajectories from rollout start (rad)','rollout step');series(svg,hvals,'jointnorm',x,y,w,h,hi,'jointmedian');series(svg,pvals,'paneltrace',x,y,w,h,hi,'paneltrace-median');};
    drawStart();drawReal();drawFull();
  })();
  </script>
</div>'''.replace('__DATA__', payload)
(OUT / 'joint-base-relationship.html').write_text(html)
print(OUT / 'joint-base-relationship.html')
