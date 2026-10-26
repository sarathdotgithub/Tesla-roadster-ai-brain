import numpy as np

class VehiclePhysics:
    def __init__(self, mass_kg: float, base_tilt: float, color_hex: str):
        self.mass_kg = mass_kg
        self.base_tilt = base_tilt
        self.color_hex = color_hex

def simulate_stabilization(vehicle_name: str, reaction_ms: float, wind_force: float) -> dict:
    """
    Simulates the attitude control of a vehicle in mid-air.
    Returns a dictionary of telemetry data for plotting.
    """
    # 1. Define vehicle profiles
    if "Tesla" in vehicle_name:
        vehicle = VehiclePhysics(mass_kg=2000, base_tilt=5.0, color_hex="#e82127")
    else:
        vehicle = VehiclePhysics(mass_kg=25000, base_tilt=15.0, color_hex="#3e6ae1")
        
    reaction_s = reaction_ms / 1000.0
    t = np.linspace(0, 3, 300)
    tilt_data = []
    
    # 2. Run the math loop
    for time in t:
        if time < reaction_s:
            # Drifting due to wind before reaction
            current_tilt = vehicle.base_tilt + (wind_force * time)
        else:
            # Stabilization after reaction
            time_since = time - reaction_s
            max_tilt_at_reaction = vehicle.base_tilt + (wind_force * reaction_s)
            current_tilt = max_tilt_at_reaction * np.exp(-3.5 * time_since)
            
            # Add human over-correction jitter if reaction time is slow
            if reaction_ms > 50:
                current_tilt += np.sin(time * 15) * 2.5 * np.exp(-1.5 * time_since)
                
        tilt_data.append(current_tilt)

    return {
        "time_array": t,
        "tilt_array": tilt_data,
        "color": vehicle.color_hex,
        "reaction_s": reaction_s
    }