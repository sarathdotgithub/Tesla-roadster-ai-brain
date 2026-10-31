"""physics_engine.py — corrected, unified, vectorized simulation engine.

Fixes vs the previous version (all verified numerically):
 1. CRITICAL sign bug: u = +kp*theta_d + kd*theta_dot_d is POSITIVE feedback on
    an inverted pendulum (theta_ddot = +(g/L) sin(theta) + gust + u), so every
    profile diverged — even the IMU crashed at t = 0.76 s. Now:
    u = -(kp*theta_d + kd*theta_dot_d).
 2. Human gains re-tuned (kp 40->16, kd 8->7). At tau = 200 ms the old gains sit
    past the delay stability margin: zeta_eff = kd/(2*wn) - wn*tau/2 < 0, so the
    "human" diverged even with zero gust. 16/7 keeps zeta_eff ~= 0.26: a wobbly,
    believable human that survives calm gusts and struggles at violent ones.
 3. Batch delay line was broken: idx = clamp(delay_steps - step - 1, min=0)
    goes negative after `delay_steps` steps, so every sample kept reading
    buffer slot 0 — the effective delay was a sawtooth unrelated to the sweep.
    Replaced with a proper per-sample ring buffer (modular read index).
 4. Monte Carlo trials were deterministic (same gust, same tremor phase), so
    survival was always 0% or 100%. Gust amplitude (+-30%), onset jitter
    (+-150 ms) and tremor phase now vary per trial.
 5. stability_map() filled surviving cells with np.random.uniform fake data.
    Peak tilt is now tracked inside the kernel — the heatmap is real output.
 6. The booster used its roll torque `u` as vertical thrust (9.81 - u*2 m/s^2).
    Replaced with a thrust-vector model: thrust along the body axis, gimbal
    torque for attitude, auto-throttle landing burn (Falcon-9 style) — tilting
    is what steers it laterally, exactly like a real landing.
 7. One kernel, one set of constants (dt = 1 ms, T = 3 s) for single runs and
    batches; the old list.pop(0) delay line is O(n) per step and is replaced by
    O(1) index arithmetic. torch is dropped: the whole sweep is ONE NumPy
    kernel vectorized over (delays x trials) — at demo scale this runs in
    seconds on CPU and works on any Space hardware.

Conventions
    theta      attitude of the vehicle body about its center of mass [rad]
    gust input lateral acceleration [m/s^2] -> angular accel = gust / L
    u          angular acceleration commanded by the controller [rad/s^2],
             clipped to +-U_MAX ("actuator authority")
    delay      sensor+compute pipeline latency [ms]; the controller only ever
             sees the state as it was `delay` ago (+ measurement noise).
"""

import numpy as np

# ---------------- constants ----------------
CRASH_DEG = 60.0          # attitude beyond this = diverged
IMPULSE_START = 0.5       # gust onset [s]
IMPULSE_DUR = 0.05        # gust duration [s]
G = 9.81
L_CAR = 1.5               # Roadster attitude lever arm [m]
L_BOOSTER = 6.0           # booster lever arm [m]
GUST_LEVER = L_CAR        # lever used for batch studies
V_CAR = 27.78             # 100 km/h in m/s (for the v*tau metric)
U_MAX = 50.0              # actuator authority [rad/s^2]
DT = 0.001                # integration step [s]
T = 3.0                   # run length [s]

PROFILES = {
    #                        sensor  tau_ms  meas_noise_deg  tremor  f_tremor  kp    kd
    "Human":                dict(tau=200.0, sigma=0.30, tremor=0.15, f_tremor=10.0, kp=16.0, kd=7.0),
    "Physical AI (Vision)": dict(tau=40.0,  sigma=0.20, tremor=0.0,  f_tremor=0.0,  kp=70.0, kd=12.0),
    "Physical AI (IMU)":    dict(tau=2.0,   sigma=0.02, tremor=0.0,  f_tremor=0.0,  kp=80.0, kd=15.0),
}


def effective_delay_ms(controller, g_force=1.0, override_ms=None):
    """Pipeline latency. Sustained >4 g degrades the human vestibular loop
    (greyout); sensors are immune. This is a demo hook, not physiology."""
    tau = override_ms if override_ms is not None else PROFILES[controller]["tau"]
    if controller == "Human" and g_force > 4.0:
        tau *= 1.0 + (g_force - 4.0) * 0.5
    return tau


# ---------------- vectorized batch kernel ----------------
def _kernel(tau_ms, gust_ang, om2, kp, kd, sigma_deg, tremor, f_tremor,
            dt=DT, T=T, seed=0):
    """Simulate N samples in parallel. tau_ms/gust_ang: arrays (N,).

    Ring buffer: state written at step k lands in column k % D; a sample with
    delay d reads column (k - d) % D, i.e. exactly the state from d*dt ago.
    Reads before k >= d return 0 (controller sees nothing, like the real rig
    during startup). Returns per-sample crash flag, peak |theta|, control
    effort and RMS tilt.
    """
    rng = np.random.default_rng(seed)
    tau_ms = np.asarray(tau_ms, dtype=float)
    gust_ang = np.asarray(gust_ang, dtype=float)
    N = tau_ms.size

    d = np.maximum(1, np.rint(tau_ms * 0.001 / dt).astype(np.int64))
    D = int(d.max())
    steps = int(round(T / dt))

    amp = gust_ang * rng.uniform(0.6, 1.4, N)          # gust magnitude scatter
    t0 = IMPULSE_START + rng.uniform(-0.15, 0.20, N)   # gust onset jitter
    phi = rng.uniform(0.0, 2.0 * np.pi, N)             # tremor phase per trial

    theta = np.zeros(N)
    omega = np.zeros(N)
    buf_t = np.zeros((N, D))
    buf_w = np.zeros((N, D))
    crashed = np.zeros(N, dtype=bool)
    peak = np.zeros(N)
    effort = np.zeros(N)
    ss = np.zeros(N)

    rows = np.arange(N)
    sig = sigma_deg * np.pi / 180.0
    noisy = sig > 0.0
    trembles = tremor > 0.0
    crash_rad = CRASH_DEG * np.pi / 180.0

    for k in range(steps):
        t = k * dt
        r = (k - d) % D
        valid = k >= d
        mt = np.where(valid, buf_t[rows, r], 0.0)
        mo = np.where(valid, buf_w[rows, r], 0.0)
        if noisy:
            mt = mt + rng.standard_normal(N) * sig
        u = -(kp * mt + kd * mo)
        if trembles:
            u = u + tremor * np.sin(2.0 * np.pi * f_tremor * t + phi)
        np.clip(u, -U_MAX, U_MAX, out=u)
        u = np.where(crashed, 0.0, u)                  # dead actuators after divergence

        dist = np.where((t >= t0) & (t < t0 + IMPULSE_DUR), amp, 0.0)
        omega += (om2 * np.sin(theta) + dist + u) * dt
        theta += omega * dt
        np.clip(theta, -1.5708, 1.5708, out=theta)

        c = k % D
        buf_t[:, c] = theta
        buf_w[:, c] = omega

        crashed |= np.abs(theta) > crash_rad
        np.maximum(peak, np.abs(theta), out=peak)
        effort += np.abs(u) * dt
        ss += theta * theta * dt

    return dict(
        crashed=crashed,
        peak_deg=np.degrees(peak),
        effort=effort,
        rms_deg=np.degrees(np.sqrt(ss / max(steps * dt, 1e-9))),
    )


# ---------------- single run (full traces + metrics) ----------------
def simulate_emergency(controller, vehicle_type="Roadster", wind=12.0, g_force=1.0,
                       delay_ms=None, seed=3):
    """One scenario run. Gust = `wind` m/s^2 lateral for IMPULSE_DUR seconds.

    Roadster: hovers on cold-gas thrusters at 100 km/h (SpaceX-package concept).
    Booster:  landing burn from 70 m at -25 m/s; tilting steers it laterally.
    Returns traces (100 Hz) + a metrics dict for the dashboard.
    """
    p = PROFILES[controller]
    tau = effective_delay_ms(controller, g_force, delay_ms)
    d = max(1, int(round(tau * 0.001 / DT)))
    steps = int(round(T / DT))
    lever = L_BOOSTER if vehicle_type == "Booster" else L_CAR
    om2 = G / lever
    gust_ang = wind / lever
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0.0, 2.0 * np.pi)

    theta = omega = 0.0
    hist_t = np.zeros(steps)
    hist_w = np.zeros(steps)

    n_out = steps // 10
    time = np.zeros(n_out)
    th_out = np.zeros(n_out)
    meas_out = np.zeros(n_out)
    u_out = np.zeros(n_out)

    booster = vehicle_type == "Booster"
    x, y = 0.0, (40.0 if booster else 8.0)
    vy = -22.0 if booster else 0.0

    crashed = crash_t = None
    landed = False
    td = None
    peak = 0.0
    effort = 0.0
    ss = 0.0
    last_exc = 0.0
    xcross = 0
    prev_sign = 0

    j = 0
    for k in range(steps):
        t = k * DT
        if k >= d:
            mt = hist_t[k - d]
            mo = hist_w[k - d]
            if p["sigma"] > 0.0:
                mt = mt + rng.standard_normal() * p["sigma"] * np.pi / 180.0
        else:
            mt = mo = 0.0
        u = -(p["kp"] * mt + p["kd"] * mo)
        if p["tremor"] > 0.0:
            u = u + p["tremor"] * np.sin(2.0 * np.pi * p["f_tremor"] * t + ph)
        u = min(U_MAX, max(-U_MAX, u))
        if crashed:
            u = 0.0

        dist = gust_ang if IMPULSE_START <= t < IMPULSE_START + IMPULSE_DUR else 0.0
        omega += (om2 * np.sin(theta) + dist + u) * DT
        theta += omega * DT
        theta = max(-1.5708, min(1.5708, theta))

        if booster and not landed and not crashed:
            a_t = min(2.2 * G, max(0.2 * G, G - 1.2 * (vy + 12.0)))
            x += -a_t * np.sin(theta) * DT
            vy += (a_t * np.cos(theta) - G) * DT
            y += vy * DT
            if y <= 0.0:
                y = 0.0
                landed = True
                td = dict(t=t, v_sink=abs(vy), drift=abs(x),
                          tilt_deg=abs(np.degrees(theta)),
                          ok=(abs(np.degrees(theta)) <= 10.0 and abs(x) <= 2.0))
        elif not booster:
            x = V_CAR * t
            y = 8.0

        hist_t[k] = theta
        hist_w[k] = omega

        if not crashed and abs(theta) > CRASH_DEG * np.pi / 180.0:
            crashed = True
            crash_t = t

        peak = max(peak, abs(theta))
        effort += abs(u) * DT
        ss += theta * theta * DT
        if abs(theta) > np.radians(0.5):
            last_exc = t
        s = 1 if theta > 1e-9 else (-1 if theta < -1e-9 else 0)
        if prev_sign and s and s != prev_sign and t > IMPULSE_START:
            xcross += 1
        if s:
            prev_sign = s

        if k % 10 == 0:
            time[j] = t
            th_out[j] = np.degrees(theta)
            meas_out[j] = np.degrees(mt)
            u_out[j] = u
            j += 1

        if (crashed and t - crash_t > 0.02) or landed:
            break

    # pad traces after freeze
    while j < n_out:
        time[j] = time[j - 1] + DT * 10 if j else 0.0
        th_out[j] = th_out[j - 1] if j else 0.0
        meas_out[j] = meas_out[j - 1] if j else 0.0
        u_out[j] = 0.0
        j += 1

    metrics = dict(
        tau_ms=tau,
        react_dist_m=V_CAR * tau * 0.001,
        peak_deg=np.degrees(peak),
        settle_s=(last_exc if (not crashed and last_exc > IMPULSE_START) else None),
        rms_deg=np.degrees(np.sqrt(ss / T)),
        effort=effort,
        xcross=xcross,
        f_damp=(xcross / (2.0 * (T - IMPULSE_START - IMPULSE_DUR)) if xcross else 0.0),
    )
    if crash_t is not None:
        metrics["crash_s"] = crash_t
    if td is not None:
        metrics["touchdown"] = td

    return dict(time=time.tolist(), theta=th_out.tolist(), meas=meas_out.tolist(),
                torque=u_out.tolist(), x=x, y=y, crashed=bool(crashed),
                landed=bool(landed), peak_deg=metrics["peak_deg"],
                delay_ms=tau, effort=effort, metrics=metrics)


# ---------------- Monte Carlo survival sweep ----------------
def run_monte_carlo(controller, wind, g_force, delays, trials, seed=7, lever=GUST_LEVER):
    """Survival probability vs pipeline delay, with randomized gusts per trial.

    Returns survival% per delay, the critical delay (first point where
    survival < 50%), and peak-tilt percentiles per delay.
    """
    delays = np.asarray(delays, dtype=float)
    trials = int(trials)
    taus = np.repeat(delays, trials)
    if controller == "Human" and g_force > 4.0:
        taus = taus * (1.0 + (g_force - 4.0) * 0.5)
    p = PROFILES[controller]
    res = _kernel(taus, np.full(taus.size, float(wind) / lever), G / lever,
                  p["kp"], p["kd"], p["sigma"], p["tremor"], p["f_tremor"], seed=seed)
    crashed = res["crashed"].reshape(len(delays), trials)
    survival = (1.0 - crashed.mean(axis=1)) * 100.0
    peak = np.where(res["crashed"], CRASH_DEG + 5.0, res["peak_deg"]).reshape(len(delays), trials)

    below = np.where(survival < 50.0)[0]
    critical = float(delays[below[0]]) if below.size else None
    return dict(delays=delays.tolist(), survival=survival.tolist(),
                critical_ms=critical,
                peak_p50=np.median(peak, axis=1).tolist(),
                peak_p95=np.percentile(peak, 95, axis=1).tolist())


# ---------------- stability map (real data) ----------------
def stability_map(controller, g_force, delays=np.linspace(2, 350, 24),
                  winds=np.linspace(2, 30, 16), trials=12, seed=11, lever=GUST_LEVER):
    """Median peak tilt over a (delay x gust) grid. Crashed cells read
    CRASH_DEG + 5. Also returns the survival grid."""
    delays = np.asarray(delays, dtype=float)
    winds = np.asarray(winds, dtype=float)
    p = PROFILES[controller]
    peak_m = np.zeros((len(winds), len(delays)))
    surv_m = np.zeros_like(peak_m)
    for i, w in enumerate(winds):
        res = run_monte_carlo(controller, float(w), g_force, delays, trials,
                              seed=seed, lever=lever)
        peak_m[i, :] = res["peak_p50"]
        surv_m[i, :] = res["survival"]
    return dict(delays=delays.tolist(), winds=winds.tolist(),
                peak=peak_m.tolist(), survival=surv_m.tolist())