try:
    import spaces
    # This decorator tells Hugging Face ZeroGPU to use the GPU for these functions
    def gpu_decorator(func):
        return spaces.GPU(func)
except ImportError:  # Local CPU fallback
    def gpu_decorator(func):
        return func

import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from physics_engine import (CRASH_DEG, IMPULSE_START, PROFILES, run_monte_carlo,
                            simulate_emergency, stability_map)

CONTROLLERS = list(PROFILES)
COLORS = {"Human": "#f59e0b", "Physical AI (IMU)": "#8b5cf6", "Physical AI (Vision)": "#38bdf8"}
THEME = gr.themes.Monochrome()
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6

def _style(fig, height=430):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=height,
                      margin=dict(l=20, r=20, t=50, b=20),
                      legend=dict(orientation="h", y=-0.15))
    return fig

def _empty(msg):
    return _style(go.Figure()), f"**Error:** {msg}"

def default_delay(controller):
    return PROFILES[controller]["delay_ms"]

# ---------- Tab 1: single run ----------
def run_single(controller, delay_ms, wind, g_force, compare):
    try:
        main = simulate_emergency(controller, wind, g_force, delay_ms)
        runs = {controller: main}
        if compare:
            for c in CONTROLLERS:
                if c != controller:
                    runs[c] = simulate_emergency(c, wind, g_force, 0)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                            vertical_spacing=0.08,
                            subplot_titles=("Vehicle tilt (degrees)", "Control effort"))
        for name, r in runs.items():
            sel = name == controller
            fig.add_trace(go.Scatter(x=r["time"], y=r["theta"], name=name, mode="lines",
                                     line=dict(color=COLORS[name], width=4 if sel else 2,
                                               dash="solid" if sel else "dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=r["time"], y=r["torque"], name=name, mode="lines",
                                     showlegend=False,
                                     line=dict(color=COLORS[name], width=2 if sel else 1)), row=2, col=1)
        fig.add_hrect(y0=CRASH_DEG, y1=90, fillcolor="red", opacity=0.12, line_width=0, row=1, col=1)
        fig.add_hrect(y0=-90, y1=-CRASH_DEG, fillcolor="red", opacity=0.12, line_width=0, row=1, col=1)
        fig.add_vline(x=IMPULSE_START, line_dash="dash", line_color="yellow", row=1, col=1,
                      annotation_text="Wind gust", annotation_position="top right")
        fig.update_yaxes(range=[-90, 90], row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        _style(fig, 520)

        status = "💥 **CRASHED**" if main["crashed"] else "✅ **Stabilized**"
        md = (f"{status} &nbsp; | &nbsp; **Peak tilt:** {'>90' if main['peak_deg'] > 90 else format(main['peak_deg'], '.1f')}° &nbsp; | &nbsp; "
              f"**Effective delay:** {main['delay_ms']:.0f} ms &nbsp; | &nbsp; "
              f"**Mean effort:** {main['effort']:.1f}")
        return fig, md
    except Exception as e:
        return _empty(e)

# ---------- Tab 2: Monte Carlo (GPU Accelerated) ----------
@gpu_decorator
def run_mc(controller, wind, g_force, trials):
    try:
        top = 350 if controller == "Human" else 400
        res = run_monte_carlo(controller, wind, g_force, delays=np.linspace(10, top, 30),
                              trials=int(trials))
        fig = go.Figure(go.Scatter(x=res["delays"], y=res["survival"], mode="lines+markers",
                                   line=dict(color=COLORS[controller], width=3), name="Survival"))
        fig.add_hline(y=50, line_dash="dash", line_color="yellow", annotation_text="50% survival")
        fig.update_layout(title=f"Survival probability vs reaction delay: {controller}",
                          xaxis_title="Base reaction delay (ms)", yaxis_title="Survival (%)",
                          yaxis_range=[-2, 102])
        crit = res["critical_ms"]
        md = (f"**Critical delay (survival drops below 50%):** ~{crit:.0f} ms" if crit
              else "**No critical delay in this range.** Survival stays at 50% or higher.")
        md += f"\n\n*Each point is {int(trials)} runs. PyTorch GPU Tensors accelerated this calculation.*"
        return _style(fig), md
    except Exception as e:
        return _empty(e)

# ---------- Tab 3: stability map (GPU Accelerated) ----------
@gpu_decorator
def run_map(controller, g_force):
    try:
        res = stability_map(controller, g_force)
        fig = go.Figure(go.Heatmap(x=res["delays"], y=res["winds"], z=res["peak"],
                                   colorscale="Turbo", zmin=0, zmax=90,
                                   colorbar=dict(title="Peak tilt (°)")))
        fig.add_trace(go.Contour(x=res["delays"], y=res["winds"], z=res["peak"],
                                 contours=dict(start=CRASH_DEG, end=CRASH_DEG, size=1, coloring="none"),
                                 line=dict(color="white", width=3, dash="dash"),
                                 showscale=False, name="Crash boundary"))
        fig.update_layout(title=f"Stability map: {controller}",
                          xaxis_title="Reaction delay (ms)", yaxis_title="Wind impulse strength")
        return _style(fig, 480), ("Below and left of the dashed line the vehicle survives. "
                                  "Faster controllers push the boundary far to the right.")
    except Exception as e:
        return _empty(e)

blocks_kwargs = {} if GRADIO_6 else {"theme": THEME}
launch_kwargs = {"theme": THEME} if GRADIO_6 else {}

with gr.Blocks(title="Physical AI vs Human", **blocks_kwargs) as demo:
    gr.Markdown(
        "# 🚗⚡ Physical AI vs Human: The Reaction-Time Gap\n"
        "**Tesla's mission is to build a world of amazing abundance, and Physical AI is how it gets there.** "
        "This simulation measures that gap using a GPU-accelerated inverted pendulum physics engine."
    )
    
    with gr.Row():
        controller = gr.Dropdown(CONTROLLERS, value="Human", label="Controller")
        delay = gr.Slider(1, 400, value=200, step=1, label="Reaction delay (ms)")
        wind = gr.Slider(1, 30, value=10, step=1, label="Wind impulse strength")
        g_force = gr.Slider(1, 9, value=1, step=0.5, label="Sustained G-load (humans only)")

    with gr.Tabs():
        with gr.Tab("Single run"):
            compare = gr.Checkbox(value=True, label="Overlay the other controllers")
            run_btn = gr.Button("Run simulation", variant="primary")
            plot1, md1 = gr.Plot(), gr.Markdown()
        with gr.Tab("Monte Carlo (GPU)"):
            trials = gr.Slider(10, 100, value=40, step=10, label="Trials per delay")
            mc_btn = gr.Button("Run Monte Carlo", variant="primary")
            plot2, md2 = gr.Plot(), gr.Markdown()
        with gr.Tab("Stability map (GPU)"):
            map_btn = gr.Button("Build map", variant="primary")
            plot3, md3 = gr.Plot(), gr.Markdown()

    controller.change(default_delay, controller, delay)
    single_in = [controller, delay, wind, g_force, compare]
    run_btn.click(run_single, single_in, [plot1, md1])
    mc_btn.click(run_mc, [controller, wind, g_force, trials], [plot2, md2])
    map_btn.click(run_map, [controller, g_force], [plot3, md3])
    demo.load(run_single, single_in, [plot1, md1])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", **launch_kwargs)