import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from physics_engine import simulate_stabilization

def run_dashboard(vehicle, reaction_ms, wind):
    # Call the backend engine
    data = simulate_stabilization(vehicle, reaction_ms, wind)
    
    # Build UI
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Vehicle Pitch (Tilt)", "Reaction Time Gap"))
    
    fig.add_trace(go.Scatter(x=data["time_array"], y=data["tilt_array"], 
                             mode='lines', name='Pitch', line=dict(color=data["color"], width=3)), row=1, col=1)
    
    fig.add_vline(x=data["reaction_s"], line_dash="dash", line_color="yellow", row=1, col=1)
    
    # ... (rest of the plotly code from before)
    
    return fig, "Simulation complete."