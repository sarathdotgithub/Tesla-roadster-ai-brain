import html
import json
import math

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from physics_engine import CRASH_DEG, PROFILES, simulate_emergency

# ZeroGPU Spaces stop at startup if no @spaces.GPU function exists. This placeholder satisfies the
# check, and is skipped automatically on CPU hardware where `spaces` isn't installed.
try:
    import spaces

    @spaces.GPU
    def _hf_gpu_check():
        return "GPU check passed"
except ImportError:
    pass

CONTROLLERS = list(PROFILES)
COLORS = {"Human": "#f59e0b", "Physical AI (IMU)": "#8b5cf6", "Physical AI (Vision)": "#38bdf8"}
THEME = gr.themes.Monochrome()
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6


def _jsonable(o):
    """Convert numpy types to plain Python and replace NaN/inf (invalid in JSON) with 0."""
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if math.isfinite(v) else 0.0
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;background:#0d1117;font-family:sans-serif;color:#8ea2ba;text-align:center}
canvas{background:#161e29;border:1px solid #2a3444;border-radius:8px;width:100%;max-width:800px;margin-top:10px}
button{background:#3e6ae1;color:#fff;border:0;padding:8px 16px;border-radius:5px;cursor:pointer;font-size:14px}
#statusText{margin-left:15px;font-weight:bold}
</style></head><body>
<canvas id="simCanvas" width="800" height="300"></canvas>
<div style="margin:10px 0"><button id="playBtn">&#9654; Play Real-Time Simulation</button>
<span id="statusText"></span></div>
<script>
(function () {
  const data = __DATA__;
  const isCar = "__VEHICLE__" === "Car";
  const canvas = document.getElementById('simCanvas');
  const ctx = canvas.getContext('2d');
  const btn = document.getElementById('playBtn');
  const status = document.getElementById('statusText');
  const N = data.time.length;
  const frameMs = N > 1 ? (data.time[1] - data.time[0]) * 1000 : 10;   // real-time playback
  const scaleX = 2.0, scaleY = 1.5, centerX = 100, centerY = 220;
  let raf = null;

  function drawVehicle(x, y, theta, torque) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#253040";
    ctx.fillRect(0, centerY + 20, canvas.width, 50);
    ctx.save();
    let sx = Math.min(centerX + x * scaleX, 700);
    let sy = centerY - y * scaleY;
    ctx.translate(sx, sy);
    ctx.rotate(theta * Math.PI / 180);
    if (isCar) {
      ctx.fillStyle = "#e82127"; ctx.fillRect(-30, -15, 60, 20);
      ctx.fillStyle = "#111";
      ctx.beginPath(); ctx.arc(-20, 5, 8, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(20, 5, 8, 0, Math.PI * 2); ctx.fill();
    } else {
      ctx.fillStyle = "#e6edf5";
      ctx.beginPath(); ctx.moveTo(0, -30); ctx.lineTo(10, 20); ctx.lineTo(-10, 20); ctx.closePath(); ctx.fill();
    }
    if (Math.abs(torque) > 5) {
      ctx.fillStyle = "#4fd1c5";
      const left = torque > 0, fx = left ? -15 : 15, fy = isCar ? 0 : 20;
      ctx.beginPath(); ctx.moveTo(fx, fy);
      ctx.lineTo(fx + (left ? -10 : 10), fy + 15); ctx.lineTo(fx + (left ? 5 : -5), fy + 5); ctx.fill();
    }
    ctx.restore();
  }

  function play() {
    if (raf) cancelAnimationFrame(raf);
    const t0 = performance.now();
    function step(now) {
      const i = Math.min(N - 1, Math.floor((now - t0) / frameMs));
      drawVehicle(data.x[i], data.y[i], data.theta[i], data.torque[i]);
      if (i < N - 1) {
        status.textContent = "Delay: " + data.delay_ms.toFixed(0) + " ms | Tilt: " + data.theta[i].toFixed(1) + "\\u00b0";
        status.style.color = "#8ea2ba";
        raf = requestAnimationFrame(step);
      } else {
        status.textContent = data.crashed ? "\\ud83d\\udca5 CRASHED" : "\\u2705 STABILIZED";
        status.style.color = data.crashed ? "#e82127" : "#4fd1c5";
      }
    }
    raf = requestAnimationFrame(step);
  }

  btn.addEventListener('click', play);
  drawVehicle(data.x[0], data.y[0], 0, 0);
  play();   // autoplay once on load; the button replays
})();
</script></body></html>"""


def generate_canvas_html(data, vehicle_type):
    n = len(data["time"])
    payload = _jsonable({
        "time": data["time"], "theta": data["theta"], "torque": data["torque"],
        "x": data.get("x", [0.0] * n), "y": data.get("y", [0.0] * n),
        "delay_ms": data.get("delay_ms", 0.0),
        "crashed": data.get("crashed", bool(np.max(np.abs(data["theta"])) > CRASH_DEG)),
    })
    page = PAGE.replace("__DATA__", json.dumps(payload)).replace("__VEHICLE__", vehicle_type)
    # Scripts inserted via innerHTML never execute, so the page lives in an iframe's srcdoc instead.
    return (f'<iframe srcdoc="{html.escape(page, quote=True)}" sandbox="allow-scripts" '
            f'style="width:100%;height:410px;border:0;border-radius:10px;background:#0d1117"></iframe>')


def run_visual_sim(controller, vehicle_type, delay_ms, wind, g_force):
    try:
        data = simulate_emergency(controller, vehicle_type, wind, g_force, delay_ms)
        html_output = generate_canvas_html(data, vehicle_type)
    except Exception as e:
        return f"<p style='color:#e82127'>Simulation error: {html.escape(str(e))}</p>", go.Figure()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.1,
                        subplot_titles=("Vehicle Tilt (Degrees)", "Control Effort (Thrust)"))
    fig.add_trace(go.Scatter(x=data["time"], y=data["theta"], name="Tilt",
                             line=dict(color=COLORS.get(controller, "#f59e0b"), width=3)), row=1, col=1)
    fig.add_hrect(y0=CRASH_DEG, y1=90, fillcolor="red", opacity=0.12, line_width=0, row=1, col=1)
    fig.add_hrect(y0=-90, y1=-CRASH_DEG, fillcolor="red", opacity=0.12, line_width=0, row=1, col=1)
    fig.add_trace(go.Scatter(x=data["time"], y=data["torque"], name="Thrust",
                             line=dict(color="#4fd1c5", width=2)), row=2, col=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=400, showlegend=False)
    fig.update_yaxes(title_text="Tilt (°)", range=[-90, 90], row=1, col=1)
    fig.update_yaxes(title_text="Thrust", row=2, col=1)
    return html_output, fig


def default_delay(controller):
    return PROFILES[controller]["delay_ms"]


blocks_kwargs = {} if GRADIO_6 else {"theme": THEME}
launch_kwargs = {"theme": THEME} if GRADIO_6 else {}

with gr.Blocks(title="Physical AI Visual Simulator", **blocks_kwargs) as demo:
    gr.Markdown("# 🏎️🚀 Real-Time Physical AI Simulator")
    gr.Markdown("Watch how a Tesla Roadster or SpaceX Falcon 9 handles an emergency mid-air crosswind. "
                "The animation plays automatically; click **Play** to replay it.")
    with gr.Row():
        with gr.Column(scale=1):
            vehicle_type = gr.Radio(["Car", "Rocket"], value="Car", label="Select Vehicle")
            controller = gr.Dropdown(CONTROLLERS, value="Human", label="Controller")
            delay = gr.Slider(1, 400, value=200, step=1, label="Reaction Delay (ms)")
            wind = gr.Slider(1, 30, value=10, step=1, label="Wind Impulse")
            g_force = gr.Slider(1, 9, value=1, step=0.5, label="G-Force (humans only)")
            run_btn = gr.Button("Simulate", variant="primary")
        with gr.Column(scale=2):
            canvas_out = gr.HTML(label="Real-Time Render")
            plot_out = gr.Plot(label="Telemetry Data")

    controller.change(default_delay, controller, delay)
    inputs = [controller, vehicle_type, delay, wind, g_force]
    run_btn.click(run_visual_sim, inputs, [canvas_out, plot_out])
    demo.load(run_visual_sim, inputs, [canvas_out, plot_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", **launch_kwargs)