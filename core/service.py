from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
from numba import jit, prange
import uvicorn

app = FastAPI()


class SimulationParams(BaseModel):
    n_neurons: int = Field(default=50, gt=0)
    t_end: float = Field(default=200.0, gt=0.0)
    dt: float = Field(default=0.05, gt=0.0)
    n_param: float = 3.0
    base_gamma: float = 1.01
    delta_gamma: float = 0.005
    d_couple: float = 0.0
    alpha: float = 0.0
    seed: int = 42


class ScanParams(BaseModel):
    n_neurons: int = Field(default=2, gt=0)
    t_end: float = Field(default=200.0, gt=0.0)
    dt: float = Field(default=0.1, gt=0.0)
    n_param: float = 3.0
    base_gamma: float = 1.01
    # Δ-spread of γ across the ensemble (d_alpha mode) — was hardcoded to 0.01
    delta_gamma: float = 0.01

    scan_type: str = "d_alpha"

    d_min: float = 0.0
    d_max: float = 0.5
    d_steps: int = Field(default=30, gt=0)

    alpha_min: float = 0.0
    alpha_max: float = 3.14
    alpha_steps: int = Field(default=30, gt=0)

    delta_min: float = 0.0
    delta_max: float = 0.02
    delta_steps: int = Field(default=30, gt=0)
    fixed_alpha: float = 2.094

    seed: int = 42


class CompareNParams(BaseModel):
    t_end: float = Field(default=80.0, gt=0.0)
    dt: float = Field(default=0.02, gt=0.0)
    base_gamma: float = 1.01
    n_values: List[float] = [1.0, 3.0, 5.0]


# ---------------------------------------------------------------------------
# Right-hand sides and RK4 integrators
# ---------------------------------------------------------------------------

@jit(nopython=True)
def _rhs_meanfield(phases, n_neurons, gammas, n_param, d_couple, alpha):
    sum_sin = 0.0
    sum_cos = 0.0
    for j in range(n_neurons):
        sum_sin += np.sin(phases[j])
        sum_cos += np.cos(phases[j])
    mean_sin = sum_sin / n_neurons
    mean_cos = sum_cos / n_neurons
    ms_rot = mean_sin * np.cos(alpha) - mean_cos * np.sin(alpha)
    mc_rot = mean_cos * np.cos(alpha) + mean_sin * np.sin(alpha)

    out = np.empty(n_neurons)
    for i in range(n_neurons):
        phi = phases[i]
        interaction = ms_rot * np.cos(phi) - mc_rot * np.sin(phi)
        out[i] = gammas[i] - np.sin(phi / n_param) + d_couple * interaction
    return out


@jit(nopython=True)
def _rk4_step_meanfield(phases, dt, n_neurons, gammas, n_param, d_couple, alpha):
    k1 = _rhs_meanfield(phases, n_neurons, gammas, n_param, d_couple, alpha)
    k2 = _rhs_meanfield(phases + 0.5 * dt * k1, n_neurons, gammas, n_param, d_couple, alpha)
    k3 = _rhs_meanfield(phases + 0.5 * dt * k2, n_neurons, gammas, n_param, d_couple, alpha)
    k4 = _rhs_meanfield(phases + dt * k3, n_neurons, gammas, n_param, d_couple, alpha)
    return phases + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@jit(nopython=True)
def _rhs_two(phases, gammas, n_param, d_couple, alpha):
    p1 = phases[0]
    p2 = phases[1]
    int_1 = np.sin(p2 - p1 - alpha)
    int_2 = np.sin(p1 - p2 - alpha)
    out = np.empty(2)
    out[0] = gammas[0] - np.sin(p1 / n_param) + d_couple * int_1
    out[1] = gammas[1] - np.sin(p2 / n_param) + d_couple * int_2
    return out


@jit(nopython=True)
def _rk4_step_two(phases, dt, gammas, n_param, d_couple, alpha):
    k1 = _rhs_two(phases, gammas, n_param, d_couple, alpha)
    k2 = _rhs_two(phases + 0.5 * dt * k1, gammas, n_param, d_couple, alpha)
    k3 = _rhs_two(phases + 0.5 * dt * k2, gammas, n_param, d_couple, alpha)
    k4 = _rhs_two(phases + dt * k3, gammas, n_param, d_couple, alpha)
    return phases + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------------------------------------------------------------------
# Compute kernels (RK4-based, replacing the old Euler versions)
# ---------------------------------------------------------------------------

@jit(nopython=True)
def compute_dynamics_rk4(n_neurons, steps, dt, gammas, n_param, d_couple, alpha, initial_phases):
    phases = np.zeros((steps, n_neurons))
    current = initial_phases.copy()
    for i in range(n_neurons):
        phases[0, i] = current[i]
    for t in range(steps - 1):
        current = _rk4_step_meanfield(current, dt, n_neurons, gammas, n_param, d_couple, alpha)
        for i in range(n_neurons):
            phases[t + 1, i] = current[i]
    return phases


@jit(nopython=True, parallel=True)
def compute_sync_regions(d_vals, delta_vals, alpha, steps, dt, n_param, base_gamma):
    """D × Δ scan: synchronization regions for a two-neuron pair.

    Replaces the legacy `compute_arnold_tongue` kernel; uses RK4 instead of
    explicit Euler and also returns the observed mean frequencies.
    """
    n_d = len(d_vals)
    n_delta = len(delta_vals)
    sync_map = np.zeros((n_delta, n_d))
    omega1_map = np.zeros((n_delta, n_d))
    omega2_map = np.zeros((n_delta, n_d))

    for i in prange(n_delta):
        delta = delta_vals[i]
        gammas = np.array([base_gamma, base_gamma + delta])

        for j in range(n_d):
            d = d_vals[j]
            phases = np.array([0.0, 0.5])

            sync_metric = 0.0
            count = 0
            start_measure = int(steps * 0.6)
            phi_start_0 = 0.0
            phi_start_1 = 0.0

            for t in range(steps):
                if t == start_measure:
                    phi_start_0 = phases[0]
                    phi_start_1 = phases[1]

                if t > start_measure:
                    ms = (np.sin(phases[0]) + np.sin(phases[1])) / 2.0
                    mc = (np.cos(phases[0]) + np.cos(phases[1])) / 2.0
                    R = np.sqrt(ms * ms + mc * mc)
                    sync_metric += R
                    count += 1

                phases = _rk4_step_two(phases, dt, gammas, n_param, d, alpha)

            t_window = (steps - start_measure) * dt
            if t_window > 0.0:
                omega1_map[i, j] = (phases[0] - phi_start_0) / t_window
                omega2_map[i, j] = (phases[1] - phi_start_1) / t_window
            if count > 0:
                sync_map[i, j] = sync_metric / count

    return sync_map, omega1_map, omega2_map


@jit(nopython=True, parallel=True)
def compute_grid_sync(d_vals, alpha_vals, n_neurons, steps, dt, gammas, n_param, initial_phases):
    """D × α scan: N-neuron mean-field sync map (replaces compute_grid_jit).

    Returns the order parameter R and the ensemble-averaged observed frequency.
    """
    n_d = len(d_vals)
    n_a = len(alpha_vals)
    sync_map = np.zeros((n_d, n_a))
    omega_mean_map = np.zeros((n_d, n_a))

    for i in prange(n_d):
        d = d_vals[i]
        for j in range(n_a):
            alpha = alpha_vals[j]
            current = initial_phases.copy()
            order_param_accum = 0.0
            count = 0
            start_measure = int(steps * 0.5)
            phi_start = np.zeros(n_neurons)

            for t in range(steps):
                if t == start_measure:
                    for k in range(n_neurons):
                        phi_start[k] = current[k]

                if t > start_measure:
                    sum_sin = 0.0
                    sum_cos = 0.0
                    for k in range(n_neurons):
                        sum_sin += np.sin(current[k])
                        sum_cos += np.cos(current[k])
                    mean_sin = sum_sin / n_neurons
                    mean_cos = sum_cos / n_neurons
                    R = np.sqrt(mean_sin * mean_sin + mean_cos * mean_cos)
                    order_param_accum += R
                    count += 1

                current = _rk4_step_meanfield(current, dt, n_neurons, gammas, n_param, d, alpha)

            t_window = (steps - start_measure) * dt
            if t_window > 0.0:
                omega_sum = 0.0
                for k in range(n_neurons):
                    omega_sum += (current[k] - phi_start[k]) / t_window
                omega_mean_map[i, j] = omega_sum / n_neurons
            if count > 0:
                sync_map[i, j] = order_param_accum / count

    return sync_map, omega_mean_map


@jit(nopython=True)
def compute_single_trace_rk4(t_end, dt, base_gamma, n_param, initial_phase):
    """Single isolated oscillator phi(t) for the n-comparison plot."""
    steps = int(t_end / dt)
    phases = np.zeros(steps)
    phases[0] = initial_phase
    p = initial_phase
    for t in range(steps - 1):
        k1 = base_gamma - np.sin(p / n_param)
        k2 = base_gamma - np.sin((p + 0.5 * dt * k1) / n_param)
        k3 = base_gamma - np.sin((p + 0.5 * dt * k2) / n_param)
        k4 = base_gamma - np.sin((p + dt * k3) / n_param)
        p = p + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        phases[t + 1] = p
    return phases


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.post("/simulate")
def run_simulation(params: SimulationParams):
    np.random.seed(params.seed)
    steps = int(params.t_end / params.dt)
    if steps < 2:
        raise HTTPException(status_code=422,
                            detail=f"t_end ({params.t_end}) too small for dt ({params.dt})")

    gammas = np.linspace(params.base_gamma,
                         params.base_gamma + params.delta_gamma,
                         params.n_neurons)
    initial_phases = np.random.random(params.n_neurons) * 2 * np.pi

    print(f"[DYNAMICS RK4] N={params.n_neurons} steps={steps} n={params.n_param} "
          f"Gamma_1={params.base_gamma:.5f} Delta={params.delta_gamma:.5f} "
          f"alpha={params.alpha:.4f} d={params.d_couple:.4f}")

    raw = compute_dynamics_rk4(
        params.n_neurons, steps, params.dt, gammas,
        params.n_param, params.d_couple, params.alpha, initial_phases
    )

    t_start_idx = steps // 2
    t_window = (steps - 1 - t_start_idx) * params.dt
    if t_window > 0:
        omegas = (raw[-1, :] - raw[t_start_idx, :]) / t_window
    else:
        omegas = np.zeros(params.n_neurons)

    target_points = 1500
    skip = max(1, steps // target_points)
    data = np.sin(raw[::skip].T)
    dx = skip * params.dt

    print(f"[DYNAMICS RK4] Omega: mean={float(np.mean(omegas)):.4f} "
          f"std={float(np.std(omegas)):.4f} "
          f"min={float(np.min(omegas)):.4f} max={float(np.max(omegas)):.4f}")

    return {
        "status": "success",
        "heatmap": data.tolist(),
        "x0": 0.0,
        "dx": dx,
        "t_end": steps * params.dt,
        "gamma_1": params.base_gamma,
        "delta_gamma": params.delta_gamma,
        "alpha": params.alpha,
        "d_couple": params.d_couple,
        "n_param": params.n_param,
        "omegas": omegas.tolist(),
        "omega_mean": float(np.mean(omegas)),
        "omega_std": float(np.std(omegas)),
        "integrator": "RK4",
    }


@app.post("/scan")
def run_scan(params: ScanParams):
    np.random.seed(params.seed)
    steps = int(params.t_end / params.dt)
    if steps < 2:
        raise HTTPException(status_code=422,
                            detail=f"t_end ({params.t_end}) too small for dt ({params.dt})")

    print(f"[SCAN RK4] type={params.scan_type} Gamma_1={params.base_gamma:.5f} "
          f"Delta=[{params.delta_min:.5f}, {params.delta_max:.5f}] steps={steps}")

    if params.scan_type == "d_delta":
        d_values = np.linspace(params.d_min, params.d_max, params.d_steps)
        delta_values = np.linspace(params.delta_min, params.delta_max, params.delta_steps)

        result_matrix, omega1, omega2 = compute_sync_regions(
            d_values, delta_values, params.fixed_alpha,
            steps, params.dt, params.n_param, params.base_gamma
        )

        print(f"[SCAN RK4] Omega1: mean={float(np.mean(omega1)):.4f}, "
              f"Omega2: mean={float(np.mean(omega2)):.4f}")

        return {
            "status": "success",
            "map_z": result_matrix.tolist(),
            "map_x": d_values.tolist(),
            "map_y": delta_values.tolist(),
            "omega1": omega1.tolist(),
            "omega2": omega2.tolist(),
            "gamma_1": params.base_gamma,
            "delta_min": params.delta_min,
            "delta_max": params.delta_max,
            "fixed_alpha": params.fixed_alpha,
            "integrator": "RK4",
        }

    d_values = np.linspace(params.d_min, params.d_max, params.d_steps)
    alpha_values = np.linspace(params.alpha_min, params.alpha_max, params.alpha_steps)

    gammas = np.linspace(params.base_gamma,
                         params.base_gamma + params.delta_gamma,
                         params.n_neurons)
    initial_phases = np.random.random(params.n_neurons) * 2 * np.pi

    result_matrix, omega_mean = compute_grid_sync(
        d_values, alpha_values,
        params.n_neurons, steps, params.dt,
        gammas, params.n_param, initial_phases
    )

    print(f"[SCAN RK4] Omega mean field: mean={float(np.mean(omega_mean)):.4f}")

    return {
        "status": "success",
        "map_z": result_matrix.tolist(),
        "map_x": alpha_values.tolist(),
        "map_y": d_values.tolist(),
        "omega_mean": omega_mean.tolist(),
        "gamma_1": params.base_gamma,
        "delta_gamma": params.delta_gamma,
        "integrator": "RK4",
    }


@app.post("/compare_n")
def run_compare_n(params: CompareNParams):
    steps = int(params.t_end / params.dt)
    if steps < 2:
        raise HTTPException(status_code=422,
                            detail=f"t_end ({params.t_end}) too small for dt ({params.dt})")
    target_points = 1500
    skip = max(1, steps // target_points)
    t_axis = (np.arange(0, steps, skip) * params.dt).tolist()

    print(f"[COMPARE n] Gamma_1={params.base_gamma:.5f} "
          f"n_values={params.n_values} t_end={params.t_end}")

    traces = []
    for n_val in params.n_values:
        phi = compute_single_trace_rk4(params.t_end, params.dt,
                                       params.base_gamma, float(n_val), 0.0)
        t_start_idx = steps // 2
        t_window = (steps - 1 - t_start_idx) * params.dt
        omega = float((phi[-1] - phi[t_start_idx]) / t_window) if t_window > 0 else 0.0
        traces.append({
            "n": float(n_val),
            "t": t_axis,
            "phi": phi[::skip].tolist(),
            "sin_phi": np.sin(phi[::skip]).tolist(),
            "omega": omega,
        })
        print(f"[COMPARE n] n={n_val}: Omega={omega:.4f}")

    return {
        "status": "success",
        "gamma_1": params.base_gamma,
        "traces": traces,
        "integrator": "RK4",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
