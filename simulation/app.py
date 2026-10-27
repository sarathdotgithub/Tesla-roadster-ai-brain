import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from physics_engine import simulate_stabilization

def run_dashboard(vehicle, reaction_ms, wind):
    data = simulate_stabilization(vehicle, reaction_ms, wind)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Vehicle Pitch (Tilt)", "Reaction Time Gap"))
    
    fig.add_trace(go.Scatter(x=data["time_array"], y=data["tilt_array"], 
                             mode='lines', name='Pitch', line=dict(color=data["color"], width=3)), row=1, col=1)
    
    fig.add_vline(x=data["reaction_s"], line_dash="dash", line_color="yellow", row=1, col=1, 
                  annotation_text=f"Reacted at {reaction_ms}ms", annotation_position="top right")

    labels = ['Human', 'Physical AI']
    times = [200, 2]
    bar_colors = ['#8ea2ba', '#8b5cf6']
    
    fig.add_trace(go.Bar(x=labels, y=times, marker_color=bar_colors, text=times, textposition='auto'), row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    fig.update_yaxes(title_text="Degrees (°)", row=1, col=1)
    fig.update_yaxes(title_text="Milliseconds (ms)", type="log", row=1, col=2)

    if reaction_ms <= 10:
        verdict = "**Result:** The Physical AI stabilized the vehicle almost instantly. The AI processed the crosswind and adjusted the thrusters before the vehicle could tilt more than a few degrees."
    else:
        verdict = f"**Result:** The Human reaction time of {reaction_ms}ms was too slow. The vehicle tilted significantly due to the crosswind. (In a real scenario, a 2-ton vehicle would likely crash)."

    return fig, verdict

# Standard theme placement for maximum compatibility
with gr.Blocks(theme=gr.themes.Monochrome(), title="Physical AI vs Human") as demo:
    gr.Markdown("# 🚀 Physical AI vs Human: Attitude Control Simulation")
    gr.Markdown("Compare the reaction times of an elite human (e.g., Max Verstappen) versus a Physical AI controller in stabilizing a vehicle mid-air.")
    
    with gr.Row():
        with gr.Column(scale=1):
            vehicle_dropdown = gr.Dropdown(
                choices=["Tesla Roadster", "SpaceX Falcon 9"], 
                value="Tesla Roadster", 
                label="Select Vehicle"
            )
            reaction_slider = gr.Slider(
                minimum=1, maximum=250, value=200, step=1, 
                label="Reaction Time (ms)", 
                info="Human ~200ms | AI ~2ms"
            )
            crosswind_slider = gr.Slider(
                minimum=1, maximum=20, value=10, step=1, 
                label="Crosswind Disturbance Force",
                info="Higher means stronger wind pushing the vehicle over"
            )
            run_btn = gr.Button("Run Simulation", variant="primary")
            
        with gr.Column(scale=2):
            plot_output = gr.Plot(label="Telemetry Data")
            text_output = gr.Markdown()

    run_btn.click(
        fn=run_dashboard, 
        inputs=[vehicle_dropdown, reaction_slider, crosswind_slider], 
        outputs=[plot_output, text_output]
    )

# Standard launch, no extra experimental flags
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Binds to all network interfaces
    )