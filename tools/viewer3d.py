"""Generate the interactive 3D section of the page.

Alon et al. build their result as explicit primitives -- spheres with position,
radius and colour -- rather than a voxel volume, because that is what keeps the
object legible.  The same choice is made here, and the data is inlined rather than
fetched: the published page runs from a file://-like sandbox where a relative
fetch is not dependable.

The renderer is ~120 lines of plain canvas 2D.  No three.js, no CDN, no WebGL
context to lose -- a few thousand circles painted back-to-front is fast enough and
has no dependencies to break.
"""
import json
import os

PAL = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
       "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

VIEW_JS = """
(function(){
  const PH = %(pal)s;

  function rotY(a){const c=Math.cos(a),s=Math.sin(a);
    return [c,0,s, 0,1,0, -s,0,c];}
  function rotX(a){const c=Math.cos(a),s=Math.sin(a);
    return [1,0,0, 0,c,-s, 0,s,c];}
  function mul(A,B){const C=new Array(9);
    for(let i=0;i<3;i++)for(let j=0;j<3;j++){let s=0;
      for(let k=0;k<3;k++)s+=A[i*3+k]*B[k*3+j];C[i*3+j]=s;}return C;}
  function apply(M,v){return [
    M[0]*v[0]+M[1]*v[1]+M[2]*v[2],
    M[3]*v[0]+M[4]*v[1]+M[5]*v[2],
    M[6]*v[0]+M[7]*v[1]+M[8]*v[2]];}

  function Viewer(root, D){
    const cv = document.createElement('canvas');
    cv.width = 900; cv.height = 620;
    cv.style.width='100%%'; cv.style.height='auto';
    cv.style.borderRadius='8px'; cv.style.cursor='grab';
    cv.style.background='#0e1116'; cv.style.display='block';
    root.appendChild(cv);
    const g = cv.getContext('2d');

    const bar = document.createElement('div');
    bar.className='vbar';
    root.appendChild(bar);

    const n = D.x.length;
    // centre in x,y; z is already centred on the section plane
    let cx=0, cy=0;
    for(let i=0;i<n;i++){cx+=D.x[i];cy+=D.y[i];}
    cx/=n; cy/=n;
    const pts = [];
    for(let i=0;i<n;i++)
      pts.push({x:D.x[i]-cx, y:D.y[i]-cy, z:D.z[i],
                r:D.r_true[i], p:D.phenotype[i]});

    let ax=-0.55, ay=0.6, zoom=1.0, drag=null;
    const modes=['All phenotypes'];
    for(let k=0;k<D.n_phenotypes;k++) modes.push('P'+k);
    let mode=0, showPanel=false;

    const btn=document.createElement('button');
    btn.className='vbtn'; btn.textContent='view: all phenotypes';
    btn.onclick=()=>{mode=(mode+1)%%modes.length;
      btn.textContent='view: '+modes[mode].toLowerCase();
      if(modes[mode]!=='All phenotypes') btn.textContent='view: '+modes[mode];
      draw();};
    bar.appendChild(btn);

    const pb=document.createElement('button');
    pb.className='vbtn'; pb.textContent='cross-section: off';
    pb.onclick=()=>{showPanel=!showPanel;
      pb.textContent='cross-section: '+(showPanel?'on':'off'); draw();};
    bar.appendChild(pb);

    const hint=document.createElement('span');
    hint.className='vhint';
    hint.textContent='drag to rotate  ·  scroll to zoom';
    bar.appendChild(hint);

    function draw(){
      const M = mul(rotX(ax), rotY(ay));
      const w=cv.width, h=cv.height;
      g.fillStyle='#0e1116'; g.fillRect(0,0,w,h);

      let maxr=0;
      for(const p of pts) maxr=Math.max(maxr, Math.abs(p.x), Math.abs(p.y));
      const F = (Math.min(w,h)/ (2.4*maxr)) * zoom;

      const vis=[];
      for(const p of pts){
        if(mode>0 && p.p!==(mode-1)) continue;
        if(showPanel && p.z < 0) continue;
        const q=apply(M,[p.x,p.y,p.z]);
        vis.push({sx:w/2+q[0]*F, sy:h/2+q[1]*F,
                  r:Math.max(1.1,p.r*F), d:q[2], p:p.p});
      }
      vis.sort((a,b)=>a.d-b.d);
      for(const v of vis){
        if(v.sx<-40||v.sx>w+40||v.sy<-40||v.sy>h+40) continue;
        const depth=Math.max(0,Math.min(1,(v.d/(maxr*1.4)+1)/2));
        const col=PH[v.p%%PH.length];
        g.globalAlpha=0.30+0.55*depth;
        g.fillStyle=col;
        g.beginPath(); g.arc(v.sx,v.sy,v.r,0,6.2832); g.fill();
        g.globalAlpha=0.16+0.30*depth;
        g.strokeStyle='rgba(255,255,255,0.5)'; g.lineWidth=0.6; g.stroke();
      }
      g.globalAlpha=1;
      g.fillStyle='rgba(255,255,255,0.62)';
      g.font='13px ui-monospace,Menlo,monospace';
      g.fillText(vis.length+' cells shown  ·  section plane at z = 0'
                 +(showPanel?'  (below-plane hidden)':''), 14, 22);
    }

    cv.addEventListener('mousedown', e=>{drag={x:e.clientX,y:e.clientY}; 
      cv.style.cursor='grabbing';});
    window.addEventListener('mouseup', ()=>{drag=null;
      cv.style.cursor='grab';});
    window.addEventListener('mousemove', e=>{
      if(!drag) return;
      ay += (e.clientX-drag.x)*0.008;
      ax += (e.clientY-drag.y)*0.008;
      ax = Math.max(-1.5, Math.min(1.5, ax));
      drag={x:e.clientX,y:e.clientY};
      draw();
    });
    cv.addEventListener('wheel', e=>{
      e.preventDefault();
      zoom *= (e.deltaY>0 ? 0.92 : 1.08);
      zoom = Math.max(0.3, Math.min(6, zoom));
      draw();
    }, {passive:false});
    // touch
    cv.addEventListener('touchstart', e=>{
      if(e.touches.length===1)
        drag={x:e.touches[0].clientX,y:e.touches[0].clientY};});
    cv.addEventListener('touchmove', e=>{
      if(!drag||e.touches.length!==1) return;
      e.preventDefault();
      ay += (e.touches[0].clientX-drag.x)*0.01;
      ax += (e.touches[0].clientY-drag.y)*0.01;
      drag={x:e.touches[0].clientX,y:e.touches[0].clientY};
      draw();}, {passive:false});
    cv.addEventListener('touchend', ()=>{drag=null;});

    draw();
  }

  window.__init3d = function(id, data){
    const root = document.getElementById(id);
    if(root) Viewer(root, data);
  };
})();
"""


def section(tag, d, blurb=""):
    """One 3D panel: stats, the viewer, and a legend."""
    data = {k: d[k] for k in ("x", "y", "z", "r_true", "r_apparent",
                              "phenotype", "n_phenotypes")}
    stats = [
        ("cells in the field", f'{d["n_cells"]:,}'),
        ("phenotypes present", f'{len(set(d["phenotype"]))}'),
        ("field", f'{d["extent_um"]*(2*d["block"]+1):.0f} µm'),
        ("section slab", f'±{d["slab_um"]:.0f} µm'),
        ("radius sd: 2D → 3D",
         f'{d["apparent_radius_sd"]:.2f} → <strong>{d["true_radius_sd"]:.2f}</strong> µm'),
        ("packing-limited radius", f'{d["packing_limit_um"]:.2f} µm'),
        ("packing realised", f'{d["packing_realised"]:.3f}'),
        ("2D conflicts resolved in 3D",
         f'<strong>{d["overlap_resolved_frac"]*100:.0f}%</strong>'),
    ]
    cells = "".join(
        f'<div class="stat"><div class="v">{v}</div><div class="l">{k}</div></div>'
        for k, v in stats)
    legend = "".join(
        f'<span class="lg"><i style="background:{PAL[i%len(PAL)]}"></i>P{i}</span>'
        for i in range(d["n_phenotypes"]))
    return f"""
<h3>{tag} — {d["tile"][0]}×{d["tile"][1]} tile block, {d["n_phenotypes"]} phenotypes</h3>
{blurb}
<div class="grid">{cells}</div>
<div class="viewer" id="view3d-{tag}"></div>
<div class="vlegend">{legend}</div>
<script>window.__d3_{tag} = {json.dumps(data)};</script>
"""


def init_script(tags):
    calls = "".join(
        f"window.__init3d('view3d-{t}', window.__d3_{t});" for t in tags)
    return f"<script>window.addEventListener('load',function(){{{calls}}});</script>"


def head_css():
    return """
 .viewer { background:#0e1116; border-radius:10px; overflow:hidden; margin:14px 0;
           border:1px solid #1d2127 }
 .vbar { display:flex; align-items:center; gap:10px; padding:9px 12px;
         background:#151a21; flex-wrap:wrap }
 .vbtn { background:#232b36; color:#dfe6ee; border:1px solid #333d4a;
         border-radius:6px; padding:5px 11px; font-size:12.5px; cursor:pointer;
         font-family:inherit }
 .vbtn:hover { background:#2c3644 }
 .vhint { color:#7d8896; font-size:12px; margin-left:auto }
 .vlegend { display:flex; flex-wrap:wrap; gap:14px; margin:6px 0 18px;
            font-size:13px; color:#4a5560 }
 .lg { display:flex; align-items:center; gap:6px }
 .lg i { width:12px; height:12px; border-radius:3px; display:inline-block }
"""
