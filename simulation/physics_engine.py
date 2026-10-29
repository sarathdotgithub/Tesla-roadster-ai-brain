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
            # Car follows a ballistic arc that gets disturbed
            vy -= 9.81 * dt # Gravity
            if t < 0.5: vy = 15.0 # Initial jump thrust
            if t > 1.5: vy += 15.0 * dt # Hover thrust simulation
            x_pos += vx * dt
            y_pos += vy * dt
            if y_pos < 0: y_pos = 0 # Ground
        else:
            # Rocket landing logic
            vy += (9.81 - (u * 2)) * dt # Thrust counteracts gravity
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
            # Pad out the crash
            rem = (steps - i) // 10
            history_theta.extend([history_theta[-1]] * rem)
            history_torque.extend([0.0] * rem)
            history_x.extend([history_x[-1]] * rem)
            history_y.extend([history_y[-1]] * rem)
            break

    # Ensure arrays are same length
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