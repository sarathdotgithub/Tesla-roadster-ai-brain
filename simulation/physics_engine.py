"""Physics core: an unstable inverted pendulum (the Roadster mid-air) with a delayed controller.

Everything is vectorised across trials, so Monte Carlo and stability maps run in well under a second.
"""
import numpy as np

G = 9.81
L_COM = 1.5                    # height of centre of mass (m)
OMEGA_SQ = G / L_COM           # instability rate: (m*g*L)/(m*L^2) simplifies to g/L
DT = 0.001                     # 1 ms integration step
T_END = 4.0
U_MAX = 50.0                   # actuator saturation (rad/s^2)
IMPULSE_START, IMPULSE_LEN = 0.5, 0.05
CRASH_DEG = 30.0

# Default controller profiles (delay in ms, PD gains, noise sources)
PROFILES = {
    "Human": dict(delay_ms=200, kp=12.0, kd=4.0, tremor=0.15, tremor_hz=10.0, noise_deg=0.3, jitter=0.15),
    "Physical AI (IMU)": dict(delay_ms=2, kp=80.0, kd=15.0, tremor=0.0, tremor_hz=0.0, noise_deg=0.02, jitter=0.0),
    "Physical AI (Vision)": dict(delay_ms=40, kp=60.0, kd=12.0, tremor=0.0, tremor_hz=0.0, noise_deg=0.2, jitter=0.05),
}


def effective_delay_ms(controller_type, delay_ms, g_force):
    """Sustained G-load slows humans down; the AI is unaffected."""
    if controller_type == "Human" and g_force > 4.0:
        return delay_ms * (1 + (g_force - 4.0) * 0.5)
    return delay_ms


def _simulate(delay_s, wind, p, rng):
    """Simulate N parallel runs. delay_s and wind are arrays of shape (N,)."""
    delay_s = np.atleast_1d(np.asarray(delay_s, float))
    n = delay_s.size
    wind = np.broadcast_to(np.asarray(wind, float), (n,))
    steps = int(round(T_END / DT))
    d = np.rint(delay_s / DT).astype(int)
    cols = np.arange(n)
    t = np.arange(steps) * DT

    sig = np.radians(p["noise_deg"])
    noise_th = rng.normal(0, sig, (steps, n)) if sig > 0 else np.zeros((steps, n))
    noise_om = rng.normal(0, 5 * sig, (steps, n)) if sig > 0 else np.zeros((steps, n))
    phase = rng.uniform(0, 2 * np.pi, n)

    th = np.zeros((steps + 1, n))
    om = np.zeros((steps + 1, n))
    u_log = np.zeros((steps, n))

    for i in range(steps):
        j = np.maximum(i - d, 0)                       # controller only sees the past
        u = -(p["kp"] * (th[j, cols] + noise_th[i]) + p["kd"] * (om[j, cols] + noise_om[i]))
        if p["tremor"] > 0:
            u = u + p["tremor"] * np.sin(2 * np.pi * p["tremor_hz"] * t[i] + phase)
        u = np.clip(u, -U_MAX, U_MAX)
        dist = wind if IMPULSE_START <= t[i] < IMPULSE_START + IMPULSE_LEN else 0.0
        om[i + 1] = om[i] + (OMEGA_SQ * np.sin(th[i]) + dist + u) * DT
        th[i + 1] = th[i] + om[i + 1] * DT
        u_log[i] = u
    return t, np.degrees(th[1:]), u_log


def _recovery_time(t, theta_deg):
    """Seconds after the gust until tilt stays under 2 degrees (None if it never settles)."""
    bad = np.where(np.abs(theta_deg) > 2.0)[0]
    if bad.size == 0:
        return 0.0
    if bad[-1] >= len(t) - 1:
        return None
    return max(0.0, float(t[bad[-1]] - IMPULSE_START))


def simulate_emergency(controller_type: str, wind_impulse: float, g_force: float,
                       delay_ms: float, seed: int = 0) -> dict:
    """One run. delay_ms <= 0 means 'use the controller's default delay'."""
    p = PROFILES[controller_type]
    base = delay_ms if delay_ms and delay_ms > 0 else p["delay_ms"]
    eff = effective_delay_ms(controller_type, base, g_force)
    t, th, u = _simulate([eff / 1000], [wind_impulse], p, np.random.default_rng(seed))
    th, u = th[:, 0], u[:, 0]
    return {
        "time": t[::10], "theta": th[::10], "torque": u[::10],
        "delay_ms": eff,
        "peak_deg": float(np.max(np.abs(th))),
        "crashed": bool(np.max(np.abs(th)) > CRASH_DEG),
        "recovery_s": _recovery_time(t, th),
        "effort": float(np.mean(np.abs(u))),
    }


def run_monte_carlo(controller_type="Human", wind_impulse=10.0, g_force=1.0,
                    delays=None, trials=40, seed=0) -> dict:
    """Survival probability vs delay, with per-trial delay jitter, sensor noise and tremor phase."""
    p = PROFILES[controller_type]
    rng = np.random.default_rng(seed)
    delays = np.linspace(10, 350, 25) if delays is None else np.asarray(delays, float)
    base = np.repeat(delays, trials)
    base = np.array([effective_delay_ms(controller_type, x, g_force) for x in base])
    jittered = np.clip(base * (1 + p["jitter"] * rng.normal(size=base.size)), 0, None)
    _, th, _ = _simulate(jittered / 1000, wind_impulse, p, rng)
    ok = (np.max(np.abs(th), axis=0) < CRASH_DEG).reshape(len(delays), trials)
    surv = ok.mean(axis=1) * 100
    half = np.where(surv < 50)[0]
    critical = float(delays[half[0]]) if half.size else None
    return {"delays": delays, "survival": surv, "critical_ms": critical}


def stability_map(controller_type="Human", g_force=1.0, delays=None, winds=None, seed=0) -> dict:
    """Peak tilt over a delay x wind grid (one batched run)."""
    p = PROFILES[controller_type]
    delays = np.linspace(10, 350, 30) if delays is None else np.asarray(delays, float)
    winds = np.linspace(2, 30, 15) if winds is None else np.asarray(winds, float)
    dd, ww = np.meshgrid(delays, winds)               # (winds, delays)
    eff = np.vectorize(lambda x: effective_delay_ms(controller_type, x, g_force))(dd)
    _, th, _ = _simulate(eff.ravel() / 1000, ww.ravel(), p, np.random.default_rng(seed))
    peak = np.minimum(np.max(np.abs(th), axis=0), 90).reshape(dd.shape)
    return {"delays": delays, "winds": winds, "peak": peak}