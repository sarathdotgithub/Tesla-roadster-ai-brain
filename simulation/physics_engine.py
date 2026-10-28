import numpy as np

def simulate_emergency(system_type: str, wind_force: float, g_force: float) -> dict:
    """
    Simulates an emergency crosswind hit during a jump.
    Compares Human (Max Verstappen) vs Physical AI.
    """
    t = np.linspace(0, 2.5, 500) # 2.5 seconds of flight
    
    # --- Setup Parameters based on system ---
    if "Human" in system_type:
        base_reaction_ms = 200.0
        # G-Force effect: If G > 4, human reaction time worsens significantly (G-LOC)
        if g_force > 4.0:
            reaction_ms = base_reaction_ms * (1 + (g_force - 4.0) * 0.5)
        else:
            reaction_ms = base_reaction_ms
        control_hz = 10.0
        color = "#e82127" # Tesla Red
        overshoot_factor = 2.5 # Humans over-correct
        
    else: # Physical AI
        reaction_ms = 2.0
        control_hz = 1000.0
        color = "#8b5cf6" # AI Purple
        overshoot_factor = 0.0 # AI has perfect PID control
        
    reaction_s = reaction_ms / 1000.0
    tilt_data = []
    
    # Initial sudden disturbance (e.g., a massive gust of wind at t=0.1s)
    disturbance_time = 0.1
    disturbance_angle = wind_force * 2.0 
    
    for time in t:
        current_tilt = 0.0
        
        if time > disturbance_time:
            time_since_disturbance = time - disturbance_time
            
            if time_since_disturbance < reaction_s:
                # System is blind to the event, drifting
                current_tilt = disturbance_angle * time_since_disturbance * 5
            else:
                # System has reacted and is correcting
                time_since_correction = time_since_disturbance - reaction_s
                max_tilt = disturbance_angle * reaction_s * 5
                
                # Base correction curve
                current_tilt = max_tilt * np.exp(-4.0 * time_since_correction)
                
                # Human jitter/over-correction (pneumatic valves are opened/closed clumsily)
                if "Human" in system_type:
                    jitter = np.sin(time_since_correction * (control_hz * 2 * np.pi)) * (overshoot_factor * 0.5)
                    current_tilt += jitter
                    
        tilt_data.append(current_tilt)

    return {
        "time": t,
        "tilt": tilt_data,
        "color": color,
        "reaction_s": reaction_s,
        "control_hz": control_hz,
        "reaction_ms": reaction_ms
    }