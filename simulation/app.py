try:
    import spaces
    def gpu_decorator(func): return spaces.GPU(func)
except ImportError:
    def gpu_decorator(func): return func

import gradio as gr
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from physics_engine import CRASH_DEG, PROFILES, simulate_emergency, run_monte_carlo, stability_map

CONTROLLERS = list(PROFILES)
THEME = gr.themes.Monochrome()
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6

# --- HTML5 CANVAS RENDERER ---
def generate_canvas_html(data, vehicle_type):
    data_json = json.dumps(data)
    
    # Using standard string and replace() to prevent f-string syntax conflicts with JavaScript
    html_template = """
    <div style="background: #0d1117; border-radius: 10px; padding: 20px; text-align: center; font-family: sans-serif;">
        <canvas id="simCanvas" width="800" height="300" style="background: #161e29; border: 1px solid #2a3444; border-radius: 8px; width: 100%; max-width: 800px;"></canvas>
        <div style="margin-top: 10px; color: #8ea2ba;">
            <button onclick="playSim()" style="background: #3e6ae1; color: white; border: none; padding: 8px 16px; border-radius: 5px; cursor: pointer;">▶ Play Real-Time Simulation</button>
            <span id="statusText" style="margin-left: 15px; font-weight: bold;"></span>
        </div>
    </div>
    
    <script>
        const data = __DATA__;
        const isCar = "__VEHICLE__" === "Car";
        
        const canvas = document.getElementById('simCanvas');
        const ctx = canvas.getContext('2d');
        
        let animationId;
        let currentFrame = 0;
        
        const scaleX = 2.0; 
        const scaleY = 1.5;
        const centerX = 100;
        const centerY = 220;

        function drawVehicle(x, y, theta, torque) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            ctx.fillStyle = "#253040";
            ctx.fillRect(0, centerY + 20, canvas.width, 50);
            
            ctx.save();
            let screenX = centerX + (x * scaleX);
            let screenY = centerY - (y * scaleY);
            
            if (screenX > 700) screenX = 700; 
            
            ctx.translate(screenX, screenY);
            ctx.rotate(theta * Math.PI / 180);
            
            if (isCar) {
                ctx.fillStyle = "#e82127";
                ctx.fillRect(-30, -15, 60, 20);
                ctx.fillStyle = "#111";
                ctx.beginPath(); ctx.arc(-20, 5, 8, 0, Math.PI*2); ctx.fill();
                ctx.beginPath(); ctx.arc(20, 5, 8, 0, Math.PI*2); ctx.fill();
            } else {
                ctx.fillStyle = "#e6edf5";
                ctx.beginPath();
                ctx.moveTo(0, -30);
                ctx.lineTo(10, 20);
                ctx.lineTo(-10, 20);
                ctx.closePath();
                ctx.fill();
            }
            
            if (Math.abs(torque) > 5) {
                ctx.fillStyle = "#4fd1c5";
                ctx.beginPath();
                let flameX = torque > 0 ? -15 : 15;
                let flameY = isCar ? 0 : 20;
                ctx.moveTo(flameX, flameY);
                ctx.lineTo(flameX + (torque>0?-10:10), flameY + 15);
                ctx.lineTo(flameX + (torque>0?5:-5), flameY + 5);
                ctx.fill();
            }
            ctx.restore();
        }

        function playSim() {
            cancelAnimationFrame(animationId);
            currentFrame = 0;
            const status = document.getElementById('statusText');
            
            function animate() {
                if (currentFrame >= data.time.length) {
                    status.innerHTML = data.crashed ? "💥 CRASHED" : "✅ STABILIZED";
                    status.style.color = data.crashed ? "#e82127" : "#4fd1c5";
                    return;
                }
                
                let x = data.x[currentFrame];
                let y = data.y[currentFrame];
                let theta = data.theta[currentFrame];
                let torque = data.torque[currentFrame];
                
                drawVehicle(x, y, theta, torque);
                status.innerHTML = `Delay: ${data.delay_ms.toFixed(0)}ms | Tilt: ${theta.toFixed(1)}°`;
                status.style.color = "#8ea2ba";
                
                currentFrame++;
                animationId = setTimeout(() => { requestAnimationFrame(animate); }, 16);
            }
            animate();
        }
        
        drawVehicle(data.x[0], data.y[0], 0, 0);
    </script>
    """
    
    html = html_template.replace("__DATA__", data_json).replace("__VEHICLE__", vehicle_type)
    return html

# --- GRADIO APP ---
def run_visual_sim(controller, vehicle_type, delay_ms, wind, g_force):
    data = simulate_emergency(controller, vehicle_type, wind, g_force, delay_ms)
    html_output = generate_canvas_html(data, vehicle_type)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.1,
                        subplot_titles=("Vehicle Tilt (Degrees)", "Control Effort (Thrust)"))
    
    color = "#f59e0b" if controller == "Human" else "#8b5cf6"
    if controller == "Physical AI (Vision)": color = "#38bdf8"
    
    fig.add_trace(go.Scatter(x=data["time"], y=data["theta"], name="Tilt", line=dict(color=color, width=3)), row=1, col=1)
    fig.add_hrect(y0=CRASH_DEG, y1=90, fillcolor="red", opacity=0.12, line_width=0, row=1, col=1)
    fig.add_hrect(y0=-90, y1=-CRASH_DEG, fillcolor="red", opacity=0.12, line_width=0, row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data["time"], y=data["torque"], name="Thrust", line=dict(color="#4fd1c5", width=2)), row=2, col=1)
    
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400, showlegend=False)
    fig.update_yaxes(title_text="Tilt (°)", range=[-90, 90], row=1, col=1)
    fig.update_yaxes(title_text="Thrust", row=2, col=1)
    
    return html_output, fig

with gr.Blocks(theme=THEME, title="Physical AI Visual Simulator") as demo:
    gr.Markdown("# 🏎️🚀 Real-Time Physical AI Simulator")
    gr.Markdown("Watch how a Tesla Roadster or SpaceX Falcon 9 handles an emergency mid-air crosswind. Click **Play** to see the AI react in real-time.")
    
    with gr.Row():
        with gr.Column(scale=1):
            vehicle_type = gr.Radio(["Car", "Rocket"], value="Car", label="Select Vehicle")
            controller = gr.Dropdown(CONTROLLERS, value="Human", label="Controller")
            delay = gr.Slider(1, 400, value=200, step=1, label="Reaction Delay (ms)")
            wind = gr.Slider(1, 30, value=10, step=1, label="Wind Impulse")
            g_force = gr.Slider(1, 9, value=1, step=0.5, label="G-Force")
            run_btn = gr.Button("Simulate", variant="primary")
            
        with gr.Column(scale=2):
            canvas_out = gr.HTML(label="Real-Time Render")
            plot_out = gr.Plot(label="Telemetry Data")

    run_btn.click(
        run_visual_sim, 
        inputs=[controller, vehicle_type, delay, wind, g_force], 
        outputs=[canvas_out, plot_out]
    )
    
    demo.load(run_visual_sim, inputs=[controller, vehicle_type, delay, wind, g_force], outputs=[canvas_out, plot_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")