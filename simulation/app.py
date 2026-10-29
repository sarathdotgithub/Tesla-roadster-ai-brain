import html

import gradio as gr

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
#hud{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px}
.chip{background:#161e29;border:1px solid #2a3444;border-radius:8px;padding:6px 10px;white-space:nowrap}
.chip b{color:#fff}#status{font-weight:700}
canvas{width:100%;display:block;background:#161e29;border:1px solid #2a3444;border-radius:10px;cursor:crosshair;outline:none}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:8px}
.grp{display:flex;gap:4px;align-items:center;flex-wrap:wrap}.grp>span{color:#8ea2ba;margin-right:2px}
button{background:#232d3b;color:#dbe5f1;border:1px solid #33425a;border-radius:6px;padding:7px 11px;cursor:pointer;font:inherit}
button:hover{background:#2c3a4f}button.on{background:#3e6ae1;border-color:#3e6ae1;color:#fff}
button.act{background:#e82127;border-color:#e82127;color:#fff}
label{display:flex;flex-direction:column;font-size:12px;gap:3px;min-width:150px;flex:1;color:#8ea2ba}
label b{color:#fff}input[type=range]{width:100%}
#tip{margin-top:8px;font-size:13px;color:#8ea2ba;min-height:18px}
</style></head><body>
<div id="hud">
<div class="chip" id="status"></div><div class="chip">Tilt <b id="tilt">0.0°</b></div>
<div class="chip">Reaction <b id="dl"></b></div><div class="chip">Balanced <b id="tm">0.0 s</b></div>
<div class="chip">Gusts survived <b id="gs">0</b></div></div>
<canvas id="cv" width="800" height="420" tabindex="0"></canvas>
<div class="row">
<div class="grp"><span>Vehicle</span><button data-v="Car" class="on">🏎️ Car</button><button data-v="Rocket">🚀 Rocket</button></div>
<div class="grp"><span>Controller</span><button data-c="you">🎮 You</button><button data-c="human" class="on">🧍 Human</button>
<button data-c="vision">🤖 AI (Vision)</button><button data-c="imu">⚡ AI (IMU)</button></div></div>
<div class="row">
<label>Gust strength <b id="wv"></b><input id="wind" type="range" min="1" max="30" value="10"></label>
<label>Reaction delay <b id="dv"></b><input id="delay" type="range" min="1" max="400" value="200"></label>
<label>Speed <b id="sv"></b><input id="speed" type="range" min="5" max="100" value="100"></label></div>
<div class="row">
<button id="gL">💨 Gust from left</button><button id="gR">Gust from right 💨</button>
<button id="tL">◀ Thrust</button><button id="tR">Thrust ▶</button>
<button id="auto">🌪️ Auto gusts: off</button><button id="ghost" class="on">👻 Ghost: on</button>
<button id="rst" class="act">↺ Reset</button></div>
<div id="tip"></div>
<script>
(()=>{
const $=id=>document.getElementById(id),cv=$('cv'),ctx=cv.getContext('2d');
const DT=.001,CRASH=30*Math.PI/180,G=9.81,SZ=4096,MASK=SZ-1;
const VEH={Car:{L:1.5,u:50},Rocket:{L:6,u:30}};
const CTL={you:{n:'You',d:0},
 human:{n:'the human',d:200,kp:12,kd:4,trem:.15,noise:.3},
 vision:{n:'the vision AI',d:40,kp:60,kd:12,trem:0,noise:.2},
 imu:{n:'the IMU AI',d:2,kp:80,kd:15,trem:0,noise:.02}};
let veh='Car',ctl='human',ghost=true,auto=false;
let th,om,t,n,hTh,hOm,crashed,survT,gusts,pend,gust,phase,nextAuto,simAcc,firstGust,lastU,keys={l:0,r:0},last=0;
const gauss=()=>Math.sqrt(-2*Math.log(1-Math.random()))*Math.cos(6.2832*Math.random());
function reset(){th=om=t=n=0;hTh=new Float32Array(SZ);hOm=new Float32Array(SZ);crashed=false;survT=0;gusts=0;
 pend=[];gust={dir:0,t0:-9,str:0};phase=Math.random()*6.28;nextAuto=2.5;simAcc=0;firstGust=true;lastU=0;}
function fire(dir,str){gust={dir:dir,t0:t,str:str||+$('wind').value};pend.push(t);}
function step(){
 const P=CTL[ctl],V=VEH[veh],d=Math.round(+$('delay').value);
 hTh[n&MASK]=th;hOm[n&MASK]=om;
 let u=0;
 if(ctl==='you'){u=(keys.r-keys.l)*V.u*.5;}
 else{const ok=n>=d,j=(n-d)&MASK;
  const mt=(ok?hTh[j]:0)+gauss()*P.noise*.01745,mo=(ok?hOm[j]:0)+gauss()*P.noise*.0873;
  u=-(P.kp*mt+P.kd*mo);if(P.trem)u+=P.trem*Math.sin(62.83*t+phase);}
 u=Math.max(-V.u,Math.min(V.u,u));
 const dist=(t-gust.t0<.05)?gust.dir*gust.str:0;
 om+=(G/V.L*Math.sin(th)+dist+u)*DT;th+=om*DT;
 if(Math.abs(th)>1.5708){th=Math.sign(th)*1.5708;om=0;}
 if(!crashed&&Math.abs(th)>CRASH)crashed=true;
 if(!crashed)survT+=DT;
 t+=DT;n++;lastU=u;
 if(firstGust&&t>=1&&ctl!=='you'){firstGust=false;fire(1);}
 if(auto&&!crashed&&t>=nextAuto){fire(Math.random()<.5?-1:1,+$('wind').value*(.6+.8*Math.random()));nextAuto=t+2.5+Math.random()*2.5;}
 while(pend.length&&t-pend[0]>2){if(!crashed)gusts++;pend.shift();}
}
function drawV(a,alpha,gh){
 ctx.save();ctx.translate(400,250+Math.sin(performance.now()/500)*3);ctx.rotate(a);ctx.globalAlpha=alpha;
 ctx.strokeStyle='#4fd1c5';ctx.lineWidth=2;
 const S=(f,c)=>{ctx.beginPath();f();if(gh)ctx.stroke();else{ctx.fillStyle=c;ctx.fill();}};
 if(veh==='Car'){
  S(()=>ctx.rect(-55,-14,110,28),'#e82127');
  S(()=>{ctx.moveTo(-25,-14);ctx.lineTo(-12,-32);ctx.lineTo(18,-32);ctx.lineTo(32,-14);},'#b3181d');
  S(()=>ctx.arc(-33,14,10,0,6.3),'#111');S(()=>ctx.arc(33,14,10,0,6.3),'#111');
 }else{
  S(()=>ctx.rect(-12,-80,24,150),'#e6edf5');
  S(()=>{ctx.moveTo(-12,-80);ctx.lineTo(0,-110);ctx.lineTo(12,-80);},'#cfd8e3');
  S(()=>ctx.rect(-12,-30,24,16),'#222');
  S(()=>{ctx.moveTo(-12,45);ctx.lineTo(-30,78);ctx.lineTo(-12,70);},'#9aa8ba');
  S(()=>{ctx.moveTo(12,45);ctx.lineTo(30,78);ctx.lineTo(12,70);},'#9aa8ba');
 }
 if(!gh&&Math.abs(lastU)>3){
  const len=8+30*Math.abs(lastU)/VEH[veh].u,s=lastU>0?1:-1;ctx.fillStyle='#4fd1c5';ctx.beginPath();
  if(veh==='Car'){const x=-50*s;ctx.moveTo(x-6,16);ctx.lineTo(x+6,16);ctx.lineTo(x,16+len);}
  else{const x=-12*s,y=-60;ctx.moveTo(x,y-6);ctx.lineTo(x,y+6);ctx.lineTo(x-len*s,y);}
  ctx.fill();}
 ctx.restore();
}
function draw(){
 const W=800,H=420,gy=372;ctx.clearRect(0,0,W,H);
 ctx.fillStyle='#253040';ctx.fillRect(0,gy,W,H-gy);
 ctx.fillStyle='rgba(0,0,0,.3)';ctx.beginPath();ctx.ellipse(400,gy+12,60,7,0,0,6.3);ctx.fill();
 const age=t-gust.t0;
 if(age<.7){ctx.strokeStyle='rgba(180,210,255,'+(.6*(1-age/.7))+')';ctx.lineWidth=2;
  for(let k=0;k<9;k++){let x=(age*1400+k*83)%900;if(gust.dir<0)x=W-x;const y=130+k*28;
   ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x-gust.dir*60,y);ctx.stroke();}}
 const arc=(a,b,c)=>{ctx.beginPath();ctx.arc(400,100,70,-1.5708+a,-1.5708+b);ctx.strokeStyle=c;ctx.lineWidth=10;ctx.stroke();};
 arc(-1.047,-CRASH,'#e82127');arc(-CRASH,-.35,'#f59e0b');arc(-.35,.35,'#2fbf9b');arc(.35,CRASH,'#f59e0b');arc(CRASH,1.047,'#e82127');
 const na=-1.5708+Math.max(-1.047,Math.min(1.047,th));
 ctx.strokeStyle='#fff';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(400,100);ctx.lineTo(400+62*Math.cos(na),100+62*Math.sin(na));ctx.stroke();
 if(ghost&&ctl!=='you'){const d=Math.round(+$('delay').value);drawV(n>d?hTh[(n-d)&MASK]:0,.45,true);}
 drawV(th,1,false);
 if(crashed){ctx.fillStyle='rgba(0,0,0,.5)';ctx.fillRect(0,0,W,H);ctx.textAlign='center';ctx.fillStyle='#e82127';
  ctx.font='bold 44px system-ui';ctx.fillText('💥 CRASHED',400,200);ctx.fillStyle='#fff';ctx.font='18px system-ui';
  ctx.fillText('Survived '+survT.toFixed(1)+' s · press R or Reset to try again',400,235);}
}
function hud(){
 const deg=th*57.2958,s=$('status');
 s.textContent=crashed?'💥 CRASHED':Math.abs(deg)>20?'⚠️ DANGER':'✅ STABLE';
 s.style.color=crashed?'#e82127':Math.abs(deg)>20?'#f59e0b':'#4fd1c5';
 $('tilt').textContent=deg.toFixed(1)+'°';$('tm').textContent=survT.toFixed(1)+' s';$('gs').textContent=gusts;
}
function lbl(){
 const d=+$('delay').value;$('wv').textContent=$('wind').value;$('sv').textContent=($('speed').value/100).toFixed(2)+'×';
 $('dv').textContent=ctl==='you'?'your own reflexes':d+' ms';$('dl').textContent=ctl==='you'?'yours':d+' ms';
 $('tip').textContent=ctl==='you'
  ?'You are the controller: click the scene, then use ← → (or A / D) to thrust against the tilt. Q / E / Space send gusts.'
  :'At 100 km/h a vehicle travels '+(d/1000*27.78).toFixed(2)+' m before '+CTL[ctl].n+' even starts to react. The dashed ghost shows what the controller still "sees".';
}
function setCtl(c,r){ctl=c;document.querySelectorAll('[data-c]').forEach(b=>b.classList.toggle('on',b.dataset.c===c));
 if(c!=='you')$('delay').value=CTL[c].d;$('delay').disabled=c==='you';lbl();if(r)reset();}
document.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>{veh=b.dataset.v;
 document.querySelectorAll('[data-v]').forEach(x=>x.classList.toggle('on',x===b));reset();});
document.querySelectorAll('[data-c]').forEach(b=>b.onclick=()=>setCtl(b.dataset.c,true));
['wind','delay','speed'].forEach(i=>$(i).oninput=lbl);
$('gL').onclick=()=>fire(1);$('gR').onclick=()=>fire(-1);$('rst').onclick=reset;
$('auto').onclick=()=>{auto=!auto;$('auto').textContent='🌪️ Auto gusts: '+(auto?'on':'off');$('auto').classList.toggle('on',auto);};
$('ghost').onclick=()=>{ghost=!ghost;$('ghost').textContent='👻 Ghost: '+(ghost?'on':'off');$('ghost').classList.toggle('on',ghost);};
const hold=(id,k)=>{const b=$(id),on=e=>{e.preventDefault();if(ctl!=='you')setCtl('you',false);keys[k]=1;},off=()=>keys[k]=0;
 b.onpointerdown=on;b.onpointerup=off;b.onpointerleave=off;};hold('tL','l');hold('tR','r');
cv.onclick=e=>{cv.focus();const r=cv.getBoundingClientRect();fire((e.clientX-r.left)/r.width*800<400?1:-1);};
document.onkeydown=e=>{const c=e.code;
 if(c==='ArrowLeft'||c==='KeyA'||c==='ArrowRight'||c==='KeyD'){e.preventDefault();if(ctl!=='you')setCtl('you',false);
  keys[(c==='ArrowLeft'||c==='KeyA')?'l':'r']=1;}
 else if(c==='KeyQ')fire(1);else if(c==='KeyE')fire(-1);
 else if(c==='Space'){e.preventDefault();fire(Math.random()<.5?-1:1);}
 else if(c==='KeyR')reset();};
document.onkeyup=e=>{const c=e.code;if(c==='ArrowLeft'||c==='KeyA')keys.l=0;if(c==='ArrowRight'||c==='KeyD')keys.r=0;};
function frame(now){
 const dt=Math.min(.05,(now-last)/1000||0);last=now;simAcc+=dt*(+$('speed').value/100);
 let k=0;while(simAcc>=DT&&k<150){step();simAcc-=DT;k++;}if(simAcc>.1)simAcc=0;
 draw();hud();requestAnimationFrame(frame);}
reset();setCtl('human',false);requestAnimationFrame(frame);
})();
</script></body></html>"""

IFRAME = (f'<iframe srcdoc="{html.escape(PAGE, quote=True)}" sandbox="allow-scripts" '
          f'style="width:100%;height:780px;border:0;border-radius:10px;background:#0d1117"></iframe>')

blocks_kwargs = {} if GRADIO_6 else {"theme": THEME}
launch_kwargs = {"theme": THEME} if GRADIO_6 else {}

with gr.Blocks(title="Physical AI vs Human", **blocks_kwargs) as demo:
    gr.Markdown(
        "# 🏎️🚀 Physical AI vs Human: Can You Beat the Reaction-Time Gap?\n"
        "Tesla's mission is to build a world of amazing abundance, and Physical AI is how it gets there: "
        "safer vehicles and robots that react to the physical world faster than any human. "
        "Balance a mid-air **car** or **rocket** against wind gusts, then watch how a human, a vision-based AI "
        "and an IMU-based AI compare. Or take the controls yourself."
    )
    gr.HTML(IFRAME)
    with gr.Accordion("🧭 How to use this simulation", open=False):
        gr.Markdown(
            "- **Pick a vehicle and a controller.** 🎮 *You* means you balance it with the keyboard or the thrust buttons.\n"
            "- **Send gusts:** click the scene (the wind pushes away from where you click), use the gust buttons, "
            "or press **Q / E / Space**. **Auto gusts** keeps them coming.\n"
            "- **Watch the ghost.** The dashed outline is what the controller *sees* after its reaction delay. "
            "The bigger the gap between ghost and vehicle, the harder it is to catch the fall.\n"
            "- **Drag the delay slider** to find the point where each controller starts to fail.\n"
            "- **Slow the speed down** to see exactly how the thrusters respond.\n"
            "- **Keys:** ← → or A / D to thrust, R to reset. Click the scene first so it receives the keys."
        )
    gr.Markdown(
        "*Independent research project, not affiliated with or endorsed by Tesla, Inc. or SpaceX. "
        "A simplified conceptual model (an unstable pendulum with a delayed controller), not real flight software.*"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", **launch_kwargs)