import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from physics_engine import simulate_emergency, run_monte_carlo
import spaces

@spaces.GPU
def dummy_gpu_check(): return "ok"

def run_dashboard(controller, wind_impulse, g_force, delay_ms):
    data = simulate_emergency(controller, wind_impulse, g_force, delay_ms)
    
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Inverted Pendulum Dynamics", "Phase Portrait (State Space)"),
    )
    
    color = "#e82127" if controller == "Human" else "#8b5cf6"
    
    # 1. Time Series
    fig.add_trace(go.Scatter(x=data["time"], y=data["theta"], mode='lines', 
                             line=dict(color=color, width=3), name="Tilt"), row=1, col=1)
    fig.update_yaxes(title_text="Tilt Angle (Degrees)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    
    # 2. Phase Portrait (Theta vs Theta_dot)
    # We approximate theta_dot from the data for visualization
    theta_dot = np.gradient(data["theta"], data["time"])
    fig.add_trace(go.Scatter(x=data["theta"], y=theta_dot, mode='lines', 
                             line=dict(color=color, width=2), name="Trajectory"), row=1, col=2)
    fig.update_xaxes(title_text="Angle (Deg)", row=1, col=2)
    fig.update_yaxes(title_text="Angular Velocity (Deg/s)", row=1, col=2)
    
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", 
                      plot_bgcolor="rgba(0,0,0,0)", height=450, showlegend=False)

    if controller == "Human":
        verdict = f"### 🚨 **DELAYED CONTROL CRASH**\nReaction delay: **{data['delay_ms']:.0f}ms**. As the inverted pendulum tips, the human reacts too late. The delay causes positive feedback, leading to catastrophic failure (tilt > 90°)."
    else:
        verdict = f"### ✅ **STABILIZED**\nReaction delay: **{data['delay_ms']:.0f}ms**. The Physical AI reads the IMU and fires thrusters with sub-millisecond latency. The inverted pendulum is actively stabilized before it passes the point of no return."

    return fig, verdict

def run_sweep():
    sweep_data = run_monte_carlo()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep_data["delays"], y=sweep_data["survival"], 
                             mode='lines+markers', line=dict(color='#f5a524', width=3)))
    
    fig.update_layout(
        title="Critical Delay Threshold (Monte Carlo)",
        xaxis_title="Reaction Delay (ms)",
        yaxis_title="Survival Probability (%)",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=400
    )
    return fig

with gr.Blocks(theme=gr.themes.Monochrome(), title="Control Theory Simulation") as demo:
    gr.Markdown("# ⚙️ Inverted Pendulum: Human vs Physical AI")
    gr.Markdown("Modeling the Roadster as an unstable plant with delayed control. Observe the **Critical Delay** where human stabilization becomes mathematically impossible.")
    
    with gr.Tabs():
        with gr.TabItem("Single Simulation"):
            with gr.Row():
                with gr.Column(scale=1):
                    controller = gr.Radio(choices=["Human", "Physical AI"], value="Human", label="Controller")
                    wind = gr.Slider(5, 20, 10, label="Impulse Disturbance")
                    g_force = gr.Slider(1, 8, 1, step=0.5, label="G-Force (Degrades Human)")
                    delay = gr.Slider(50, 300, 200, step=10, label="Manual Reaction Delay (ms)")
                    btn = gr.Button("Run Simulation", variant="primary")
                with gr.Column(scale=2):
                    plot = gr.Plot()
                    text = gr.Markdown()
            btn.click(run_dashboard, inputs=[controller, wind, g_force, delay], outputs=[plot, text])
            
        with gr.TabItem("Parameter Sweep (Monte Carlo)"):
            gr.Markdown("Run a 20-trial Monte Carlo simulation at each delay step to find the survival curve. Notice the sharp drop-off.")
            sweep_btn = gr.Button("Run Sweep", variant="primary")
            sweep_plot = gr.Plot()
            sweep_btn.click(run_sweep, inputs=[], outputs=[sweep_plot])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")