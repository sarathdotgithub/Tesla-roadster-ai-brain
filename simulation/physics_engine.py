import numpy as np

def simulate_emergency(controller_type: str, wind_impulse: float, g_force: float, delay_ms: float) -> dict:
    """
    Simulates an unstable inverted pendulum (the Roadster) with a delayed controller.
    """
    dt = 0.001 # 1ms timestep for accurate integration
    T = 4.0    # 4 seconds of simulation
    steps = int(T / dt)
    time = np.linspace(0, T, steps)
    
    # Vehicle properties (approximate Roadster)
    m = 2000.0 # kg
    L = 1.5    # m (height of CoM)
    g = 9.81
    omega_sq = (m * g * L) / (m * L**2) # Inverted pendulum instability factor
    
    # --- Controller Limits (Phase 1 & 2) ---
    if controller_type == "Human":
        # Sum of sensing (35ms), neural (75ms), decision (50ms), muscle (40ms) = ~200ms
        base_delay = 0.200 
        # G-force degrades human performance
        if g_force > 4.0:
            base_delay *= (1 + (g_force - 4.0) * 0.5)
        
        control_delay = delay_ms / 1000.0 if delay_ms > 0 else base_delay
        noise_amplitude = 0.15 # Neuromuscular tremor
        noise_freq = 10.0      # 10 Hz tremor
        kp, kd = 40.0, 8.0     # Human PID gains (less optimal)
        
    else: # Physical AI (IMU-based)
        control_delay = 0.002 # 2ms
        noise_amplitude = 0.0 # AI has zero tremor
        noise_freq = 0.0
        kp, kd = 80.0, 15.0    # AI PID gains (perfectly tuned)
    
    # State variables
    theta = 0.0
    theta_dot = 0.0
    
    history_theta = []
    history_torque = []
    buffer_size = int(control_delay / dt) + 1
    theta_buffer = [0.0] * buffer_size
    theta_dot_buffer = [0.0] * buffer_size
    
    # Simulate
    for i in range(steps):
        # Apply external disturbance (impulse wind gust at t=0.5s)
        disturbance = 0.0
        if 0.5 <= time[i] < 0.55:
            disturbance = wind_impulse
            
        # Controller reads from buffer (delayed state)
        delayed_theta = theta_buffer.pop(0)
        delayed_theta_dot = theta_dot_buffer.pop(0)
        
        # Control Law (PID)
        u = (kp * delayed_theta) + (kd * delayed_theta_dot)
        
        # Add Human Tremor
        if controller_type == "Human":
            u += noise_amplitude * np.sin(2 * np.pi * noise_freq * time[i])
            
        # Actuator Saturation (Max thruster output)
        u = np.clip(u, -50.0, 50.0)
        
        # Physics Update (Euler integration)
        theta_ddot = omega_sq * np.sin(theta) + disturbance + u
        theta_dot += theta_ddot * dt
        theta += theta_dot * dt
        
        # Update buffers
        theta_buffer.append(theta)
        theta_dot_buffer.append(theta_dot)
        
        # Log data (every 10 steps to save memory)
        if i % 10 == 0:
            history_theta.append(theta)
            history_torque.append(u)
            
    return {
        "time": time[::10],
        "theta": np.degrees(history_theta), # Convert to degrees
        "torque": history_torque,
        "delay_ms": control_delay * 1000
    }

def run_monte_carlo() -> dict:
    """Phase 4: Sweep delay parameters to find the critical threshold."""
    delays = np.linspace(10, 350, 25)
    survival_rates = []
    
    for delay in delays:
        successes = 0
        trials = 20
        for _ in range(trials):
            res = simulate_emergency("Human", wind_impulse=10.0, g_force=1.0, delay_ms=delay)
            # Success if max tilt is less than 30 degrees
            if np.max(np.abs(res["theta"])) < 30:
                successes += 1
        survival_rates.append((successes / trials) * 100)
        
    return {"delays": delays, "survival": survival_rates}