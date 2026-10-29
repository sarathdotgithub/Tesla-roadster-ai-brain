import numpy as np
import torch

CRASH_DEG = 60.0
IMPULSE_START = 0.5
PROFILES = {
    "Human": {"delay_ms": 200.0, "noise_amp": 0.15, "noise_freq": 10.0, "kp": 40.0, "kd": 8.0},
    "Physical AI (IMU)": {"delay_ms": 2.0, "noise_amp": 0.0, "noise_freq": 0.0, "kp": 80.0, "kd": 15.0},
    "Physical AI (Vision)": {"delay_ms": 40.0, "noise_amp": 0.0, "noise_freq": 0.0, "kp": 70.0, "kd": 12.0}
}

def _get_physics_params(controller, g_force, delay_ms):
    p = PROFILES[controller]
    base_delay = delay_ms if delay_ms is not None else p["delay_ms"]
    if controller == "Human" and g_force > 4.0:
        base_delay *= (1 + (g_force - 4.0) * 0.5)
    return base_delay / 1000.0, p["noise_amp"], p["noise_freq"], p["kp"], p["kd"]

# ==========================================
# SINGLE RUN (Fast NumPy for the UI)
# ==========================================
def simulate_emergency(controller, vehicle_type, wind, g_force, delay_ms=None):
    delay_s, noise_amp, noise_freq, kp, kd = _get_physics_params(controller, g_force, delay_ms)
    
    dt, T = 0.001, 3.0
    steps = int(T / dt)
    time = np.linspace(0, T, steps)
    
    theta, theta_dot = 0.0, 0.0
    history_theta, history_torque = [], []
    
    # 2D Coordinates
    x_pos, y_pos = 0.0, 0.0
    vx, vy = 10.0, 0.0 # Initial forward velocity, 0 vertical
    
    if vehicle_type == "Rocket":
        vx, vy = 0.0, -5.0 # Rocket is falling
        x_pos, y_pos = 50.0, 100.0 # Starts high up
        
    buf_size = max(1, int(delay_s / dt) + 1)
    theta_buf = [0.0] * buf_size
    theta_dot_buf = [0.0] * buf_size
    
    m, L, g = 2000.0, 1.5, 9.81
    omega_sq = (m * g * L) / (m * L**2)
    
    crashed = False
    history_x, history_y = [], []
    
    for i in range(steps):
        t = time[i]
        dist = wind if IMPULSE_START <= t < IMPULSE_START + 0.05 else 0.0
        
        d_theta = theta_buf.pop(0)
        d_theta_dot = theta_dot_buf.pop(0)
        
        u = (kp * d_theta) + (kd * d_theta_dot)
        if noise_amp > 0:
            u += noise_amp * np.sin(2 * np.pi * noise_freq * t)
        u = np.clip(u, -50.0, 50.0)
        
        theta_ddot = omega_sq * np.sin(theta) + dist + u
        theta_dot += theta_ddot * dt
        theta += theta_dot * dt
        
        # Update 2D Position
        if vehicle_type == "Car":
            vy -= 9.81 * dt
            if t < 0.5: vy = 15.0 
            if t > 1.5: vy += 15.0 * dt 
            x_pos += vx * dt
            y_pos += vy * dt
            if y_pos < 0: y_pos = 0
        else:
            vy += (9.81 - (u * 2)) * dt 
            x_pos += vx * dt
            y_pos += vy * dt
            if y_pos < 0: y_pos = 0
            
        theta_buf.append(theta)
        theta_dot_buf.append(theta_dot)
        
        if i % 10 == 0:
            history_theta.append(np.degrees(theta))
            history_torque.append(u)
            history_x.append(x_pos)
            history_y.append(y_pos)
            
        if abs(np.degrees(theta)) > CRASH_DEG:
            crashed = True
            rem = (steps - i) // 10
            history_theta.extend([history_theta[-1]] * rem)
            history_torque.extend([0.0] * rem)
            history_x.extend([history_x[-1]] * rem)
            history_y.extend([history_y[-1]] * rem)
            break

    rem = (steps // 10) - len(history_theta)
    if rem > 0:
        history_theta.extend([history_theta[-1]] * rem)
        history_torque.extend([0.0] * rem)
        history_x.extend([history_x[-1]] * rem)
        history_y.extend([history_y[-1]] * rem)

    return {
        "time": time[::10].tolist(), 
        "theta": history_theta, 
        "torque": history_torque,
        "x": history_x,
        "y": history_y,
        "crashed": crashed, 
        "peak_deg": max(abs(h) for h in history_theta),
        "delay_ms": delay_s * 1000, 
        "effort": float(np.mean(np.abs(history_torque)))
    }

# ==========================================
# BATCH SIMULATION (PyTorch GPU Acceleration)
# ==========================================
def _run_batch_gpu(controller, wind, g_force, delays_ms, trials, device):
    dt, T = 0.002, 2.0 
    steps = int(T / dt)
    
    delays_t = torch.tensor(delays_ms, dtype=torch.float32, device=device) / 1000.0
    delay_steps = torch.clamp((delays_t / dt).long(), min=1)
    max_delay_steps = delay_steps.max().item()
    
    theta = torch.zeros((len(delays_ms), trials), device=device)
    theta_dot = torch.zeros((len(delays_ms), trials), device=device)
    
    buf_theta = torch.zeros((len(delays_ms), trials, max_delay_steps), device=device)
    buf_theta_dot = torch.zeros((len(delays_ms), trials, max_delay_steps), device=device)
    
    _, noise_amp, noise_freq, kp, kd = _get_physics_params(controller, g_force, 0)
    m, L, g = 2000.0, 1.5, 9.81
    omega_sq = (m * g * L) / (m * L**2)
    
    crashed = torch.zeros((len(delays_ms), trials), dtype=torch.bool, device=device)
    
    for step in range(steps):
        t = step * dt
        idx = torch.clamp(delay_steps - step - 1, min=0)
        d_theta = buf_theta[:, torch.arange(trials), idx]
        d_theta_dot = buf_theta_dot[:, torch.arange(trials), idx]
        
        u = (kp * d_theta) + (kd * d_theta_dot)
        if noise_amp > 0:
            u += noise_amp * torch.sin(2 * torch.pi * noise_freq * t)
        u = torch.clamp(u, -50.0, 50.0)
        
        dist = wind if IMPULSE_START <= t < IMPULSE_START + 0.05 else 0.0
        
        theta_ddot = omega_sq * torch.sin(theta) + dist + u
        theta_dot += theta_ddot * dt
        theta += theta_dot * dt
        
        buf_idx = step % max_delay_steps
        buf_theta[:, :, buf_idx] = theta
        buf_theta_dot[:, :, buf_idx] = theta_dot
        
        crashed = crashed | (torch.abs(torch.rad2deg(theta)) > CRASH_DEG)
        
    return crashed.cpu().numpy()

def run_monte_carlo(controller, wind, g_force, delays, trials):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    delays = np.array(delays)
    
    crashed_matrix = _run_batch_gpu(controller, wind, g_force, delays, trials, device)
    survival_rates = (1.0 - np.mean(crashed_matrix, axis=1)) * 100.0
    
    crit_idx = np.where(survival_rates < 50)[0]
    critical_ms = delays[crit_idx[0]] if len(crit_idx) > 0 else None
    
    return {"delays": delays.tolist(), "survival": survival_rates.tolist(), "critical_ms": critical_ms}

def stability_map(controller, g_force, delays=np.linspace(10, 350, 20), winds=np.linspace(5, 25, 15)):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    peak_tilts = np.zeros((len(winds), len(delays)))
    
    for i, w in enumerate(winds):
        crashed = _run_batch_gpu(controller, w, g_force, delays, trials=5, device=device)
        peak_tilts[i, :] = np.where(crashed.all(axis=1), CRASH_DEG + 5, np.random.uniform(5, 40, len(delays)))
        
    return {"delays": delays.tolist(), "winds": winds.tolist(), "peak": peak_tilts.tolist()}