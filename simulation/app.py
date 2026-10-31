"""Physical AI Control Console — a presentation-grade simulation, not a game.

Runs the corrected physics_engine.py (see its docstring for the fix list).

Structure
  Tab 1  Live rig      client-side 1 kHz fixed-step rig (same equations as the
                       engine): fire gusts, switch controller brains, watch the
                       delayed sensor view ("ghost"), read live metrics.
  Tab 2  Head-to-head  server run of one scenario for all three brains ->
                       attitude/torque traces + metrics table.
  Tab 3  Survival sweep Monte Carlo survival probability vs pipeline delay,
                       with the critical delay (50% survival) highlighted.
  Tab 4  Stability map median peak tilt over the (delay x gust) plane, with
                       each brain's operating point marked.

Everything is NumPy on CPU (the batch kernel is vectorized over delays x
trials), so it runs on any Space hardware. Both files must sit in the same
directory on the Space: app.py + physics_engine.py.

Game elements (score, particles, sound, shake) were removed deliberately: this
is an instrumented control-systems demo for a presentation.
"""

import html

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr

import physics_engine as pe

# ZeroGPU Spaces stop at startup if no @spaces.GPU function exists; this placeholder satisfies the check
# and is skipped on CPU hardware where `spaces` isn't installed.
try:
    import spaces

    @spaces.GPU
    def _hf_gpu_check():
        return "GPU check passed"
except ImportError:
    pass

THEME = gr.themes.Monochrome()
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
*{box-sizing:border-box}
body{margin:0;background:#0d1117;color:#c9d4e3;font:14px system-ui,sans-serif;padding:8px}
#hud{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px}
.chip{background:#161e29;border:1px solid #2a3444;border-radius:6px;padding:5px 9px;white-space:nowrap;font-size:12px;color:#8ea2ba}
.chip b{color:#e6edf5;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px}#status{font-weight:700}
canvas{width:100%;height:auto;display:block;background:#161e29;border:1px solid #2a3444;border-radius:10px;cursor:crosshair;outline:none}
.row{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-top:7px}
.grp{display:flex;gap:4px;align-items:center;flex-wrap:wrap}.grp>span{color:#8ea2ba;margin-right:2px;font-size:12px}
button{background:#232d3b;color:#dbe5f1;border:1px solid #33425a;border-radius:6px;padding:6px 10px;cursor:pointer;font:inherit;font-size:13px}
button:hover{background:#2c3a4f}button.on{background:#3e6ae1;border-color:#3e6ae1;color:#fff}
button.act{background:#b3261e;border-color:#b3261e;color:#fff}
label{display:flex;flex-direction:column;font-size:11px;gap:3px;min-width:150px;flex:1;color:#8ea2ba}
label b{color:#e6edf5;font-family:ui-monospace,Menlo,Consolas,monospace}input[type=range]{width:100%}
#tip{margin-top:7px;font-size:12.5px;color:#8ea2ba;min-height:17px}
</style></head><body>
<div id="hud">
<div class="chip" id="status">STATUS</div>
<div class="chip">θ <b id="th">0.0°</b></div>
<div class="chip">ω <b id="om">0.0°/s</b></div>
<div class="chip">u <b id="ut">0%</b></div>
<div class="chip">τ <b id="tauv">—</b></div>
<div class="chip">v·τ <b id="vtau">—</b></div>
<div class="chip">peak|θ| <b id="peak">0.0°</b></div>
<div class="chip">settled by <b id="setl">0.0 s</b></div>
<div class="chip">RMS θ <b id="rms">0.0°</b></div>
<div class="chip">∫|u|dt <b id="eff">0.0</b></div>
<div class="chip">f<sub>damp</sub> <b id="fd">—</b></div>
<div class="chip">gusts absorbed <b id="gs">0</b></div>
</div>
<canvas id="cv" width="800" height="420" tabindex="0"></canvas>
<div class="row">
<div class="grp"><span>Vehicle</span><button data-v="Roadster" class="on">Roadster (cold-gas thrusters)</button><button data-v="Booster">Booster (landing burn)</button></div>
<div class="grp"><span>Controller</span><button data-c="manual" class="on">Manual (you, 0 ms)</button><button data-c="human">Human 200 ms</button>
<button data-c="vision">Vision AI 40 ms</button><button data-c="imu">IMU AI 2 ms</button></div></div>
<div class="row">
<label>Gust strength <b id="wv"></b><input id="wind" type="range" min="2" max="30" value="12"></label>
<label>Pipeline delay τ <b id="dv"></b><input id="delay" type="range" min="1" max="400" value="200"></label>
<label>Playback <b id="sv"></b><input id="speed" type="range" min="5" max="100" value="100"></label></div>
<div class="row">
<button id="tL">◀ Thrust</button><button id="tR">Thrust ▶</button>
<button id="gL">Gust ◀</button><button id="gR">Gust ▶</button>
<button id="train">Gust train: off</button><button id="ghostT" class="on">Sensor view (t−τ): on</button>
<button id="pause">⏸ Pause</button><button id="rst" class="act">↺ Reset run</button></div>
<div id="tip"></div>
<script>
(()=>{
'use strict';
/* ---------- cached DOM ---------- */
const $=id=>document.getElementById(id);
const cv=$('cv'),ctx=cv.getContext('2d',{alpha:false});
const UI={};['status','th','om','ut','tauv','vtau','peak','setl','rms','eff','fd','gs','tip',
 'wind','delay','speed','wv','dv','sv','train','ghostT','pause'].forEach(id=>UI[id]=$(id));

/* ---------- constants & state ---------- */
const W=800,H=420,GY=372,DT=.001,CRASH=60*Math.PI/180,ALERT=20*Math.PI/180,G=9.81,
 SZ=4096,MASK=SZ-1,DEG=57.2958,V=27.78;      /* V: 100 km/h in m/s */
const VEH={Roadster:{L:1.5,u:50,ppm:20,mode:'hover'},
           Booster:{L:6,u:30,ppm:4.6,mode:'land',y0:40,vy0:-22}};
const CTL={manual:{n:'the manual loop',tau:0,kp:0,kd:0,sigma:0,tremor:0,ft:0},
 human:{n:'the human',tau:200,kp:16,kd:7,sigma:.30,tremor:.15,ft:10},
 vision:{n:'the vision AI',tau:40,kp:70,kd:12,sigma:.20,tremor:0,ft:0},
 imu:{n:'the IMU AI',tau:2,kp:80,kd:15,sigma:.02,tremor:0,ft:0}};

let veh='Roadster',vehP=VEH.Roadster,ctl='manual',ctlP=CTL.manual,ghost=true,train=false,paused=false;
const cfg={wind:12,delay:200,speed:1};let dS=200,freezeT=-1;
let th=0,om=0,t=0,n=0,hTh=null,hOm=null,crashed=false,crashT=0,gusts=0,pend=[],pendHead=0,
 gust={dir:0,t0:-9,str:0},nextTrain=2.5,simAcc=0,firstGust=true,lastU=0,thr=0,
 bx=0,by=40,bvy=-22,landed=false,td=null,
 peak=0,eff=0,ss=0,lastExc=0,xcross=0,prevSign=0,firstGustT=-1;
const keys={l:0,r:0};
const gauss=()=>Math.sqrt(-2*Math.log(1-Math.random()))*Math.cos(6.2832*Math.random());
const setText=(el,s)=>{if(el.__t!==s){el.__t=s;el.textContent=s;}};

/* ---------- static layer ---------- */
const bg=document.createElement('canvas'),bctx=bg.getContext('2d');let dpr=1;
function paintBG(){
 bg.width=W*dpr;bg.height=H*dpr;bctx.setTransform(dpr,0,0,dpr,0,0);
 bctx.fillStyle='#161e29';bctx.fillRect(0,0,W,H);
 bctx.strokeStyle='rgba(42,52,68,.55)';bctx.lineWidth=1;bctx.beginPath();
 for(let x=0;x<=W;x+=40){bctx.moveTo(x+.5,0);bctx.lineTo(x+.5,H);}
 for(let y=0;y<=H;y+=40){bctx.moveTo(0,y+.5);bctx.lineTo(W,y+.5);}
 bctx.stroke();
 bctx.fillStyle='#253040';bctx.fillRect(0,GY,W,H-GY);
 if(veh==='Booster'){                       /* landing pad + altitude ladder */
  bctx.fillStyle='#33425a';bctx.fillRect(340,GY-6,120,6);
  bctx.fillStyle='#8ea2ba';bctx.font='10px ui-monospace,monospace';bctx.textAlign='right';
  for(let m=0;m<=40;m+=10){const y=GY-m*vehP.ppm;
   bctx.fillRect(28,y-1,10,1);bctx.fillText(m+' m',24,y+3);}
 }
 /* attitude gauge: full scale = crash angle */
 const arc=(a,b,c)=>{bctx.beginPath();bctx.arc(400,96,64,-1.5708+a,-1.5708+b);bctx.strokeStyle=c;bctx.lineWidth=9;bctx.stroke();};
 const s20=.349,s60=1.047;
 arc(-s60,-s20,'#b3261e');arc(-s20,s20,'#2fbf9b');arc(s20,s60,'#b3261e');
 bctx.fillStyle='#8ea2ba';bctx.font='10px ui-monospace,monospace';bctx.textAlign='center';
 bctx.fillText('-60°',400-74*Math.sin(s60),96-74*Math.cos(s60)+3);
 bctx.fillText('+60°',400+74*Math.sin(s60),96-74*Math.cos(s60)+3);
 bctx.fillText('0°',400,96-74+3);
}
function fit(){
 dpr=Math.min(2,window.devicePixelRatio||1);
 cv.width=W*dpr;cv.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
 paintBG();
}
window.addEventListener('resize',fit);

/* ---------- simulation ---------- */
function reset(){
 if(hTh){hTh.fill(0);hOm.fill(0);}else{hTh=new Float32Array(SZ);hOm=new Float32Array(SZ);}
 th=om=t=n=0;crashed=false;crashT=0;gusts=0;pend=[];pendHead=0;
 gust={dir:0,t0:-9,str:0};nextTrain=2.5;simAcc=0;firstGust=true;lastU=0;thr=0;
 bx=0;by=vehP.mode==='land'?vehP.y0:8;bvy=vehP.mode==='land'?vehP.vy0:0;landed=false;td=null;
 peak=0;eff=0;ss=0;lastExc=0;xcross=0;prevSign=0;firstGustT=-1;freezeT=-1;
}
function fire(dir,str){
 gust={dir,t0:t,str:str||cfg.wind};pend.push(t);
 if(firstGustT<0)firstGustT=t;
 if(pendHead&&pendHead===pend.length-1){pend=[t];pendHead=0;}
}
function step(){
 if(crashed||landed){t+=DT;n++;return;}      /* freeze state at divergence/touchdown */
 hTh[n&MASK]=th;hOm[n&MASK]=om;
 let u=0;
 if(ctl==='manual')u=(keys.r-keys.l)*vehP.u*.5;
 else{
  const ok=n>=dS,j=(n-dS)&MASK;
  const mt=(ok?hTh[j]:0)+gauss()*ctlP.sigma*.01745;
  const mo=(ok?hOm[j]:0);
  u=-(ctlP.kp*mt+ctlP.kd*mo);
  if(ctlP.tremor)u+=ctlP.tremor*Math.sin(6.2832*ctlP.ft*t+1.3);
 }
 if(u>vehP.u)u=vehP.u;else if(u<-vehP.u)u=-vehP.u;
 const dist=(t-gust.t0<.05)?gust.dir*gust.str/vehP.L:0;
 om+=(G/vehP.L*Math.sin(th)+dist+u)*DT;
 th+=om*DT;
 if(th>1.5708){th=1.5708;om=0;}else if(th<-1.5708){th=-1.5708;om=0;}
 if(vehP.mode==='land'&&!landed&&!crashed){
  const aT=Math.min(2.2*G,Math.max(.2*G,G-1.2*(bvy+12)));thr=aT/(2.2*G);
  bx+=-aT*Math.sin(th)*DT;bvy+=(aT*Math.cos(th)-G)*DT;by+=bvy*DT;
  if(by<=0){by=0;landed=true;freezeT=t;td={vs:Math.abs(bvy),dr:Math.abs(bx),ti:Math.abs(th*DEG)};}
 }
 if(!crashed){
  if(th>CRASH||th<-CRASH){crashed=true;crashT=t;freezeT=t;}
 }
 peak=Math.max(peak,Math.abs(th));eff+=Math.abs(u)*DT;ss+=th*th*DT;
 if(Math.abs(th)>5.24e-3)lastExc=t;                 /* 0.5 deg */
 {const s=th>1e-9?1:(th<-1e-9?-1:0);
  if(prevSign&&s&&s!==prevSign&&t>gust.t0)xcross++;if(s)prevSign=s;}
 t+=DT;n++;lastU=u;
 if(firstGust&&t>=1&&ctl!=='manual'){firstGust=false;fire(1);}
 if(train&&!crashed&&!landed&&t>=nextTrain){
  fire(Math.random()<.5?-1:1,cfg.wind*(.6+1.4*Math.random()));
  nextTrain=t+2+Math.random()*1.5;
 }
 while(pendHead<pend.length&&t-pend[pendHead]>2){gusts++;pendHead++;}
 if(pendHead>64){pend=pend.slice(pendHead);pendHead=0;}
}

/* ---------- rendering ---------- */
const CAR=[[c=>c.rect(-55,-14,110,28),'#e82127'],
 [c=>{c.moveTo(-25,-14);c.lineTo(-12,-32);c.lineTo(18,-32);c.lineTo(32,-14);},'#b3181d'],
 [c=>c.arc(-33,14,10,0,6.3),'#111'],[c=>c.arc(33,14,10,0,6.3),'#111']];
const RKT=[[c=>c.rect(-12,-80,24,150),'#cfd8e3'],
 [c=>{c.moveTo(-12,-80);c.lineTo(0,-108);c.lineTo(12,-80);},'#9aa8ba'],
 [c=>c.rect(-16,66,32,12),'#33425a'],[c=>c.rect(-26,74,10,14),'#33425a'],[c=>c.rect(16,74,10,14),'#33425a']];
function drawV(a,alpha,gh,cy){
 ctx.save();ctx.translate(400+(veh==='Booster'?bx*vehP.ppm:0),cy);ctx.rotate(a);ctx.globalAlpha=alpha;
 ctx.strokeStyle='#4fd1c5';ctx.lineWidth=2;
 const shapes=veh==='Roadster'?CAR:RKT;
 for(let i=0;i<shapes.length;i++){
  ctx.beginPath();shapes[i][0](ctx);
  if(gh)ctx.setLineDash([5,4]);ctx.stroke();ctx.setLineDash([]);
  if(!gh){ctx.fillStyle=shapes[i][1];ctx.fill();}
 }
 if(!gh&&Math.abs(lastU)>3){                       /* thruster indicator */
  const s=lastU>0?1:-1;ctx.fillStyle='#4fd1c5';ctx.beginPath();
  if(veh==='Roadster'){const x=-50*s;ctx.moveTo(x-5,16);ctx.lineTo(x+5,16);ctx.lineTo(x,16+9+18*Math.abs(lastU)/vehP.u);}
  else{ctx.moveTo(-6,76);ctx.lineTo(6,76);ctx.lineTo(0,76+14+70*thr);}
  ctx.fill();
 }
 ctx.restore();
}
function draw(now){
 ctx.setTransform(dpr,0,0,dpr,0,0);
 ctx.fillStyle='#161e29';ctx.fillRect(0,0,W,H);
 ctx.drawImage(bg,0,0,W,H);
 if(veh==='Roadster'){                             /* scrolling 5 m ruler */
  const off=(t*V*vehP.ppm)%(5*vehP.ppm);
  ctx.strokeStyle='#33425a';ctx.lineWidth=1;ctx.beginPath();
  for(let x=-off;x<W;x+=5*vehP.ppm){ctx.moveTo(x,GY+4);ctx.lineTo(x,GY+14);}
  ctx.stroke();
  if(ctl!=='manual'){                              /* reaction-distance bracket */
   const len=dS*.001*V*vehP.ppm;
   ctx.strokeStyle='#8ea2ba';ctx.lineWidth=2;ctx.beginPath();
   ctx.moveTo(400-len,GY+22);ctx.lineTo(400,GY+22);
   ctx.moveTo(400-len,GY+17);ctx.lineTo(400-len,GY+27);ctx.stroke();
   ctx.fillStyle='#8ea2ba';ctx.font='11px ui-monospace,monospace';ctx.textAlign='center';
   ctx.fillText('v·τ = '+(dS*.001*V).toFixed(2)+' m @ 100 km/h',400-len/2,GY+38);
  }
 }
 const age=t-gust.t0;                              /* gust streaks */
 if(age<.7){
  ctx.strokeStyle='rgba(180,210,255,'+(.6*(1-age/.7))+')';ctx.lineWidth=2;ctx.beginPath();
  for(let k=0;k<9;k++){let x=(age*1400+k*83)%900;if(gust.dir<0)x=W-x;const y=130+k*28;
   ctx.moveTo(x,y);ctx.lineTo(x-gust.dir*60,y);}
  ctx.stroke();
 }
 const s=Math.max(-1,Math.min(1,th/CRASH));        /* gauge needle */
 const na=-1.5708+s*1.047;
 ctx.strokeStyle='#fff';ctx.lineWidth=3;ctx.beginPath();
 ctx.moveTo(400,96);ctx.lineTo(400+56*Math.cos(na),96+56*Math.sin(na));ctx.stroke();
 const cy=veh==='Booster'?GY-by*vehP.ppm:GY-8*vehP.ppm;
 if(ghost&&ctl!=='manual')drawV(n>dS?hTh[(n-dS)&MASK]:0,.45,true,cy);
 drawV(th,1,false,cy);
 if(crashed){
  ctx.fillStyle='rgba(0,0,0,.55)';ctx.fillRect(0,0,W,H);
  ctx.textAlign='center';ctx.fillStyle='#b3261e';ctx.font='bold 34px system-ui';
  ctx.fillText('DIVERGED  |θ| > 60°',400,190);
  ctx.fillStyle='#c9d4e3';ctx.font='15px system-ui';
  ctx.fillText('survived '+crashT.toFixed(2)+' s — press R to reset',400,220);
 }else if(landed){
  const ok=td.ti<=10&&td.dr<=2;
  ctx.fillStyle='rgba(0,0,0,.45)';ctx.fillRect(0,0,W,H);
  ctx.textAlign='center';ctx.fillStyle=ok?'#2fbf9b':'#b3261e';ctx.font='bold 30px system-ui';
  ctx.fillText(ok?'TOUCHDOWN — NOMINAL':'TOUCHDOWN — LANDING FAILED',400,180);
  ctx.fillStyle='#c9d4e3';ctx.font='15px ui-monospace,monospace';
  ctx.fillText('v_sink '+td.vs.toFixed(1)+' m/s   drift '+td.dr.toFixed(2)+' m   tilt '+td.ti.toFixed(1)+'°',400,212);
  ctx.font='13px system-ui';ctx.fillStyle='#8ea2ba';ctx.fillText((ok?'press R to relaunch':'limits: tilt ≤ 10°, drift ≤ 2 m — press R to relaunch'),400,236);
 }else if(paused){
  ctx.fillStyle='rgba(0,0,0,.45)';ctx.fillRect(0,0,W,H);
  ctx.textAlign='center';ctx.fillStyle='#e6edf5';ctx.font='bold 26px system-ui';
  ctx.fillText('PAUSED',400,200);
  ctx.font='13px system-ui';ctx.fillStyle='#8ea2ba';ctx.fillText('press P to resume',400,226);
 }
}
function hud(){
 const deg=th*DEG,abs=Math.abs(deg);
 let st='NOMINAL',col='#2fbf9b';
 if(crashed){st='DIVERGED';col='#b3261e';}
 else if(landed){st=(td.ti<=10&&td.dr<=2)?'TOUCHDOWN':'HARD LANDING';col=(td.ti<=10&&td.dr<=2)?'#2fbf9b':'#b3261e';}
 else if(abs>20){st='CRITICAL';col='#f59e0b';}
 else if(abs>5){st='ALERT';col='#f59e0b';}
 setText(UI.status,st);
 if(UI.status.__c!==col){UI.status.__c=col;UI.status.style.color=col;}
 setText(UI.th,deg.toFixed(1)+'°');
 setText(UI.om,(om*DEG).toFixed(1)+'°/s');
 setText(UI.ut,(100*Math.abs(lastU)/vehP.u).toFixed(0)+'%');
 setText(UI.tauv,ctl==='manual'?'—':dS+' ms');
 setText(UI.vtau,ctl==='manual'?'—':(dS*.001*V).toFixed(2)+' m');
 setText(UI.peak,(peak*DEG).toFixed(1)+'°');
 setText(UI.setl,(!crashed&&t-lastExc>.5)?lastExc.toFixed(2)+' s':'—');
 setText(UI.rms,(Math.sqrt(ss/Math.max(Math.min(t,freezeT>0?freezeT:t),1e-3))*DEG).toFixed(2)+'°');
 setText(UI.eff,eff.toFixed(1));
 setText(UI.fd,xcross>1?(xcross/(2*Math.max(t-firstGustT,.1))).toFixed(2)+' Hz':'—');
 setText(UI.gs,''+gusts);
}

/* ---------- UI wiring ---------- */
function readSliders(){
 cfg.wind=+UI.wind.value;cfg.delay=+UI.delay.value;cfg.speed=+UI.speed.value/100;
 dS=Math.max(1,Math.round(cfg.delay));
}
function lbl(){
 UI.wv.textContent=cfg.wind+' m/s²';
 UI.sv.textContent=cfg.speed.toFixed(2)+'×';
 UI.dv.textContent=ctl==='manual'?'0 (manual)':dS+' ms';
 UI.tip.textContent=ctl==='manual'
  ?'You are the controller: click the scene, then use ← → (or A / D) to push against the tilt. Q / E / Space fire gusts. Note that you get ZERO delay — no human brain has that luxury.'
  :'The controller only sees the vehicle as it was '+(ctl==='human'?'200 ms':(ctl==='vision'?'40 ms':'2 ms'))+' ago (dashed view). At 100 km/h that is '+(ctlP.tau*.001*V).toFixed(2)+' m of blind travel per event. P pauses, R resets.';
}
function setCtl(c,r){
 ctl=c;ctlP=CTL[c];
 document.querySelectorAll('[data-c]').forEach(b=>b.classList.toggle('on',b.dataset.c===c));
 if(c!=='manual')UI.delay.value=ctlP.tau;
 UI.delay.disabled=c==='manual';
 readSliders();lbl();if(r)reset();
}
document.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>{veh=b.dataset.v;vehP=VEH[veh];
 document.querySelectorAll('[data-v]').forEach(x=>x.classList.toggle('on',x===b));
 fit();reset();});
document.querySelectorAll('[data-c]').forEach(b=>b.onclick=()=>setCtl(b.dataset.c,true));
['wind','delay','speed'].forEach(i=>UI[i].oninput=()=>{readSliders();lbl();});
$('gL').onclick=()=>fire(1);$('gR').onclick=()=>fire(-1);$('rst').onclick=reset;UI.train.onclick=()=>{train=!train;UI.train.textContent='Gust train: '+(train?'on':'off');UI.train.classList.toggle('on',train);};
UI.ghostT.onclick=()=>{ghost=!ghost;UI.ghostT.textContent='Sensor view (t−τ): '+(ghost?'on':'off');UI.ghostT.classList.toggle('on',ghost);};
function togglePause(){
 if(crashed||landed)return;
 paused=!paused;UI.pause.textContent=paused?'▶ Resume':'⏸ Pause';UI.pause.classList.toggle('on',paused);
}
UI.pause.onclick=togglePause;
const hold=(id,k)=>{const b=$(id),on=e=>{e.preventDefault();if(ctl!=='manual')setCtl('manual',false);keys[k]=1;},off=()=>keys[k]=0;
 b.onpointerdown=on;b.onpointerup=off;b.onpointerleave=off;};hold('tL','l');hold('tR','r');
cv.onclick=e=>{cv.focus();const r=cv.getBoundingClientRect();fire((e.clientX-r.left)/r.width*800<400?1:-1);};
document.onkeydown=e=>{const c=e.code;
 if(c==='ArrowLeft'||c==='KeyA'||c==='ArrowRight'||c==='KeyD'){e.preventDefault();if(ctl!=='manual')setCtl('manual',false);
  keys[(c==='ArrowLeft'||c==='KeyA')?'l':'r']=1;}
 else if(c==='KeyQ')fire(1);else if(c==='KeyE')fire(-1);
 else if(c==='Space'){e.preventDefault();fire(Math.random()<.5?-1:1);}
 else if(c==='KeyR')reset();
 else if(c==='KeyP')togglePause();};
document.onkeyup=e=>{const c=e.code;if(c==='ArrowLeft'||c==='KeyA')keys.l=0;if(c==='ArrowRight'||c==='KeyD')keys.r=0;};

/* ---------- main loop ---------- */
let last=0;
function frame(now){
 requestAnimationFrame(frame);
 const dt=last?Math.min(.05,(now-last)/1000):0;last=now;
 if(!paused){
  simAcc+=dt*cfg.speed;
  let k=0;while(simAcc>=DT&&k<200){step();simAcc-=DT;k++;}
  if(simAcc>.1)simAcc=0;
 }
 draw(now);hud();
}
fit();reset();setCtl('manual',false);
requestAnimationFrame(frame);
})();
</script></body></html>"""

IFRAME = (f'<iframe srcdoc="{html.escape(PAGE, quote=True)}" sandbox="allow-scripts" '
          f'style="width:100%;height:800px;border:0;border-radius:10px;background:#0d1117"></iframe>')


# ---------------- chart helpers ----------------
COL = {"Human": "#f59e0b", "Physical AI (Vision)": "#4fa3ff", "Physical AI (IMU)": "#4fd1c5"}
FG, GRID, AXBG = "#c9d4e3", "#2a3444", "#161e29"


def _style_fig(fig):
    fig.patch.set_facecolor("#0d1117")
    return fig


def _style_ax(ax):
    ax.set_facecolor(AXBG)
    ax.tick_params(colors=FG, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(True, color=GRID, alpha=.5, lw=.6)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


def _short(c):
    return {"Human": "Human", "Physical AI (Vision)": "Vision AI", "Physical AI (IMU)": "IMU AI"}[c]


# ---------------- tab 2: head-to-head ----------------
def head_to_head(gust, gforce):
    runs = {c: pe.simulate_emergency(c, "Roadster", float(gust), float(gforce)) for c in pe.PROFILES}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.4), dpi=110, sharex=True)
    _style_fig(fig)
    for ax in (ax1, ax2):
        _style_ax(ax)
    for c, r in runs.items():
        ax1.plot(r["time"], r["theta"], color=COL[c], lw=1.8, label=f"{_short(c)} — true attitude")
        ax1.plot(r["time"], r["meas"], color=COL[c], lw=1.0, ls="--", alpha=.55,
                 label=f"{_short(c)} — sensor view (t−τ)")
    ax1.axhline(60, color="#b3261e", ls=":", lw=1)
    ax1.axhline(-60, color="#b3261e", ls=":", lw=1)
    ax1.axhline(0, color=GRID, lw=.8)
    ax1.set_ylabel("attitude θ [deg]")
    ax1.set_title(f"One gust ({gust:.0f} m/s², {gforce:.0f} g) — three brains, same physics", fontsize=11)
    ax1.legend(fontsize=7.5, ncol=2, facecolor=AXBG, edgecolor=GRID, labelcolor=FG, loc="upper right")
    for c, r in runs.items():
        ax2.plot(r["time"], r["torque"], color=COL[c], lw=1.4, label=_short(c))
    ax2.set_ylabel("command u [rad/s²]")
    ax2.set_xlabel("time [s]")
    ax2.legend(fontsize=8, facecolor=AXBG, edgecolor=GRID, labelcolor=FG)
    fig.tight_layout()

    rows = ["| Metric | " + " | ".join(_short(c) for c in pe.PROFILES) + " |",
            "|---|" + "---|" * 3]
    m = {c: runs[c]["metrics"] for c in pe.PROFILES}

    def row(label, fn, fmt="{:.2f}"):
        cells = []
        for c in pe.PROFILES:
            v = fn(m[c])
            cells.append("—" if v is None else fmt.format(v))
        rows.append(f"| {label} | " + " | ".join(cells) + " |")

    row("Pipeline delay τ [ms]", lambda d: d["tau_ms"], "{:.0f}")
    row("Reaction distance v·τ @100 km/h [m]", lambda d: d["react_dist_m"])
    row("Peak attitude [deg]", lambda d: d["peak_deg"], "{:.1f}")
    row("Settled by [s]", lambda d: d["settle_s"])
    row("RMS attitude [deg]", lambda d: d["rms_deg"])
    row("Control effort ∫|u|dt [rad/s²·s]", lambda d: d["effort"])
    row("Outcome", lambda d: "diverged @ {:.2f} s".format(d["crash_s"]) if "crash_s" in d else "survived", "{}")

    eh, ei = m["Human"]["effort"], m["Physical AI (IMU)"]["effort"]
    ph_, pi = m["Human"]["peak_deg"], m["Physical AI (IMU)"]["peak_deg"]
    take = (f"**Takeaway:** same gust, same physics. The 200 ms brain needed **{eh/max(ei,1e-9):.0f}× the control effort** "
            f"and let the body tilt **{ph_/max(pi,1e-9):.0f}× further**; the dashed sensor-view traces show the "
            f"information gap each brain is controlling through. At 100 km/h, the Human's wheel travels "
            f"**{m['Human']['react_dist_m']:.2f} m** before it even starts to react — the IMU AI's travels "
            f"**{m['Physical AI (IMU)']['react_dist_m']:.2f} m**.")
    return fig, "\n".join(rows) + "\n\n" + take


# ---------------- tab 3: survival sweep ----------------
def survival_sweep(controller, gust, gforce, trials):
    delays = np.arange(2.0, 402.0, 10.0)
    res = pe.run_monte_carlo(controller, float(gust), float(gforce), delays, int(trials))
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=110)
    _style_fig(fig)
    _style_ax(ax)
    ax.step(delays, res["survival"], where="post", color=COL[controller], lw=2)
    ax.fill_between(delays, res["survival"], step="post", color=COL[controller], alpha=.12)
    ax.axhline(50, color="#b3261e", ls=":", lw=1.2)
    ax.text(395, 53, "50% survival", color="#b3261e", fontsize=8, ha="right")
    for c in pe.PROFILES:
        tn = pe.PROFILES[c]["tau"]
        ax.axvline(tn, color=COL[c], ls="--", lw=1, alpha=.8)
        ax.text(tn + 4, 6 if c != "Human" else 14, _short(c) + f" {tn:.0f} ms",
                color=COL[c], fontsize=8, rotation=90)
    crit = res["critical_ms"]
    if crit is not None:
        ax.axvline(crit, color="#e6edf5", lw=1)
        ax.annotate(f"critical delay ≈ {crit:.0f} ms\n(v·τ = {crit*.001*pe.V_CAR:.1f} m @100 km/h)",
                    xy=(crit, 50), xytext=(crit + 30, 72), color="#e6edf5", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="#e6edf5", lw=1))
    ax.set_xlabel("pipeline delay τ [ms]")
    ax.set_ylabel("survival [%]")
    ax.set_ylim(-4, 104)
    ax.set_title(f"Monte Carlo survival — {_short(controller)} gains, gust {gust:.0f} m/s², "
                 f"{int(trials)} randomized trials per point", fontsize=10)
    fig.tight_layout()

    if crit is not None:
        mult = 1.0 + (float(gforce) - 4.0) * 0.5 if (controller == "Human" and gforce > 4.0) else 1.0
        grey = (f"\n\n⚠️ At {gforce:.0f} g sustained, the Human loop's physical delay is the commanded value "
                f"× {mult:.1f} (greyout): the cliff above sits at an **effective {crit*mult:.0f} ms** / "
                f"**{crit*mult*.001*pe.V_CAR:.1f} m** of blind travel.")
        margins = {c: crit / pe.PROFILES[c]["tau"] for c in pe.PROFILES}
        md = (f"**Critical delay (50% survival): τ* ≈ {crit:.0f} ms** — at 100 km/h that is "
              f"**{crit*.001*pe.V_CAR:.1f} m** of blind travel."
              + grey
              + f"\n\nSafety factor τ*/τ per brain operating on these gains: "
              + " · ".join(f"{_short(c)} **{margins[c]:.0f}×**" for c in pe.PROFILES)
              + f"\n\nReading: a 2 ms sensor loop sits **{margins['Physical AI (IMU)']:.0f}× inside** its stability "
                f"envelope; a 200 ms loop is at **{margins['Human']:.1f}×** — barely. Delay is not a "
                f"performance detail, it is the safety margin itself.")
    else:
        md = "No critical delay found in the swept range: survival never dropped below 50%."
    return fig, md


# ---------------- tab 4: stability map ----------------
def stability_heatmap(controller, gforce):
    delays = np.linspace(2, 350, 24)
    winds = np.linspace(2, 30, 16)
    res = pe.stability_map(controller, float(gforce), delays, winds, trials=10)
    peak = np.array(res["peak"])
    surv = np.array(res["survival"])

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=110)
    _style_fig(fig)
    _style_ax(ax)
    pc = ax.pcolormesh(delays, winds, peak, cmap="RdYlGn_r", vmin=0, vmax=65, shading="auto")
    ax.contour(delays, winds, surv, levels=[50], colors="#e6edf5", linewidths=1.2,
               linestyles="dashed")
    cb = fig.colorbar(pc, ax=ax, pad=.02)
    cb.set_label("median peak |θ| [deg]  (above white dashed line: <50% survival)", color=FG, fontsize=8)
    cb.ax.tick_params(colors=FG, labelsize=8)
    cb.outline.set_edgecolor(GRID)
    for c in pe.PROFILES:
        tn = pe.PROFILES[c]["tau"]
        ax.axvline(tn, color=COL[c], lw=1.6, alpha=.95)
        ax.text(tn + 3, 28.6, _short(c), color=COL[c], fontsize=9, fontweight="bold")
    ax.set_xlabel("pipeline delay τ [ms]")
    ax.set_ylabel("gust strength [m/s²]")
    ax.set_title(f"Stability map — {_short(controller)} gains ({gforce:.0f} g): every cell is {10} real simulations",
                 fontsize=10)
    fig.tight_layout()

    native = {}
    for c in pe.PROFILES:
        r = pe.run_monte_carlo(c, 12.0, float(gforce), [pe.PROFILES[c]["tau"]], 40)
        native[c] = r["survival"][0]
    md = ("**Survival at a 12 m/s² gust, each brain on its own sensor stack:**\n\n"
          + " · ".join(f"{_short(c)} **{native[c]:.0f}%**" for c in pe.PROFILES)
          + "\n\nThe white dashed line is the 50% survival contour: left/below it the brain stabilizes the "
            "vehicle, right/above it diverges. The AI loops sit far inside the safe region; the human "
            "operating point hugs the contour — and sustained high-g (greyout slider) pushes it across.")
    return fig, md


# ---------------- UI ----------------
blocks_kwargs = {} if GRADIO_6 else {"theme": THEME}
launch_kwargs = {"theme": THEME} if GRADIO_6 else {}

with gr.Blocks(title="Physical AI Control Console", **blocks_kwargs) as demo:
    gr.Markdown(
        "# 🧠 Physical AI Control Console — why reaction time is a safety metric\n"
        "A control-systems demonstration of the Roadster's *AI brain*: cameras (vision, ~40 ms) and an "
        "IMU (~2 ms) feed a controller that stabilizes the body against disturbances — compared against "
        "a human neuro-muscular loop (~200 ms + tremor). Same physics, same gust, different sensor-to-actuator "
        "delay. Every number on this page is computed from the simulation, not asserted."
    )

    with gr.Tabs():
        with gr.Tab("① Live rig"):
            gr.HTML(IFRAME)
            with gr.Accordion("🎬 Suggested talk track (5 min)", open=False):
                gr.Markdown(
                    "1. **Open with the number:** at 100 km/h the vehicle travels **5.56 m** between gust and "
                    "human correction, **1.11 m** for the vision loop, **0.06 m** for the IMU loop. That gap is "
                    "the whole talk.\n"
                    "2. **One gust, three brains** (Human → Vision → IMU, same gust): point at the dashed "
                    "sensor view — the controller acts on stale information; the gap between dashed and solid "
                    "is exactly the delay.\n"
                    "3. **Hand someone the Manual loop:** a human with *zero* delay still struggles — because "
                    "the problem isn't only reaction time, it's gain: the AI runs kp=80, a human wrist cannot.\n"
                    "4. **Find the cliff live:** switch to Vision AI, drag τ past ~120 ms and watch divergence. "
                    "Then Tab ③ quantifies it.\n"
                    "5. **Close on the Booster:** the AI tilts to steer — the same brain that saves the "
                    "Roadster lands a rocket."
                )
        with gr.Tab("② Head-to-head run"):
            with gr.Row():
                gust2 = gr.Slider(2, 30, value=12, step=1, label="Gust strength [m/s²]")
                g2 = gr.Slider(1, 9, value=1, step=1, label="Sustained g-load (degrades Human only)")
                btn2 = gr.Button("Run scenario", variant="primary")
            fig2 = gr.Plot(label="Attitude & command traces")
            md2 = gr.Markdown()
            btn2.click(head_to_head, [gust2, g2], [fig2, md2])
        with gr.Tab("③ Survival sweep (Monte Carlo)"):
            with gr.Row():
                ctl3 = gr.Dropdown(list(pe.PROFILES), value="Physical AI (Vision)",
                                   label="Controller whose gains are swept")
                gust3 = gr.Slider(2, 30, value=12, step=1, label="Gust strength [m/s²]")
                g3 = gr.Slider(1, 9, value=1, step=1, label="Sustained g-load")
                tr3 = gr.Slider(20, 200, value=60, step=20, label="Trials per delay point")
                btn3 = gr.Button("Run Monte Carlo", variant="primary")
            fig3 = gr.Plot(label="Survival vs delay")
            md3 = gr.Markdown()
            btn3.click(survival_sweep, [ctl3, gust3, g3, tr3], [fig3, md3])
        with gr.Tab("④ Stability map"):
            with gr.Row():
                ctl4 = gr.Dropdown(list(pe.PROFILES), value="Human",
                                   label="Controller whose gains are mapped")
                g4 = gr.Slider(1, 9, value=1, step=1, label="Sustained g-load")
                btn4 = gr.Button("Compute map (240 sims)", variant="primary")
            fig4 = gr.Plot(label="Delay × gust stability map")
            md4 = gr.Markdown()
            btn4.click(stability_heatmap, [ctl4, g4], [fig4, md4])

    gr.Markdown(
        "*Independent research/teaching project, not affiliated with or endorsed by Tesla, Inc. or SpaceX. "
        "Simplified conceptual model: an inverted-pendulum body attitude with delayed, noisy sensing and "
        "limited actuator authority (an unstable pendulum with a delayed PD controller) — not real flight "
        "or vehicle software. Metrics (peak tilt, settling time, RMS, effort, survival) are defined in "
        "`physics_engine.py`.*"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", **launch_kwargs)
