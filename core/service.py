from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os
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
    # spread of gamma across the ensemble (d_alpha mode); was hardcoded to 0.01
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
    """D x Delta scan: synchronization regions for a two-neuron pair.

    Replaces the legacy compute_arnold_tongue kernel; uses RK4 instead of
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
    """D x alpha scan: N-neuron mean-field sync map (replaces compute_grid_jit).

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


@jit(nopython=True, parallel=True)
def compute_sync_regions_meanfield(d_vals, delta_vals, fixed_alpha,
                                   n_neurons, steps, dt, base_gamma,
                                   n_param, initial_phases):
    """D x Delta скан для N-нейронного среднего поля при фиксированном alpha.

    Delta здесь = разброс gamma по ансамблю: gammas = linspace(g1, g1+Delta, N).
    Для N=2 это ядро отличается от compute_sync_regions: связь идёт с множителем
    d/N (а не d) и включает self-член j=i, дающий постоянный сдвиг частоты
    -(d/N)*sin(alpha). Численные значения d между двумя ядрами не сопоставимы
    напрямую.
    """
    n_d = len(d_vals)
    n_delta = len(delta_vals)
    sync_map = np.zeros((n_delta, n_d))
    omega_mean_map = np.zeros((n_delta, n_d))

    for i in prange(n_delta):
        delta = delta_vals[i]
        gammas = np.linspace(base_gamma, base_gamma + delta, n_neurons)

        for j in range(n_d):
            d = d_vals[j]
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

                current = _rk4_step_meanfield(current, dt, n_neurons,
                                              gammas, n_param, d, fixed_alpha)

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
# Post-scan analysis: пороги синхронизации (для агента-помощника)
# ---------------------------------------------------------------------------

def _compute_sync_thresholds(matrix, d_values, delta_values, R_thresh=0.95):
    """Для D x Delta карты [n_delta, n_d]: d_min на каждом ряду + d_min (минимум по карте), delta_max."""
    mat = np.asarray(matrix)
    n_delta, n_d = mat.shape
    d_min_per_delta = []
    for i in range(n_delta):
        row = mat[i]
        idx = np.where(row >= R_thresh)[0]
        d_min_per_delta.append(float(d_values[idx[0]]) if len(idx) else None)

    finite = [v for v in d_min_per_delta if v is not None]
    if finite:
        d_min = float(min(finite))
        i_opt = next(i for i, v in enumerate(d_min_per_delta)
                     if v is not None and v == d_min)
        delta_at_d_min = float(delta_values[i_opt])
        i_last = max(i for i, v in enumerate(d_min_per_delta) if v is not None)
        delta_max = float(delta_values[i_last])
    else:
        d_min = delta_at_d_min = delta_max = None

    return {
        "kind": "d_delta",
        "R_threshold": R_thresh,
        "d_min_per_delta": d_min_per_delta,
        "d_min": d_min,
        "delta_at_d_min": delta_at_d_min,
        "delta_max": delta_max,
    }


def _compute_sync_thresholds_d_alpha(matrix, d_values, alpha_values, R_thresh=0.95):
    """Для D x alpha карты [n_d, n_alpha]: d_min на каждом столбце + alpha_opt, alpha_crit."""
    mat = np.asarray(matrix)
    n_d, n_alpha = mat.shape
    d_min_per_alpha = []
    for j in range(n_alpha):
        col = mat[:, j]
        idx = np.where(col >= R_thresh)[0]
        d_min_per_alpha.append(float(d_values[idx[0]]) if len(idx) else None)

    finite = [v for v in d_min_per_alpha if v is not None]
    if finite:
        d_min_global = float(min(finite))
        j_opt = next(j for j, v in enumerate(d_min_per_alpha)
                     if v is not None and v == d_min_global)
        alpha_opt = float(alpha_values[j_opt])
        j_last = max(j for j, v in enumerate(d_min_per_alpha) if v is not None)
        alpha_crit = float(alpha_values[j_last])
    else:
        d_min_global = alpha_opt = alpha_crit = None

    return {
        "kind": "d_alpha",
        "R_threshold": R_thresh,
        "d_min_per_alpha": d_min_per_alpha,
        "d_min": d_min_global,
        "alpha_opt": alpha_opt,
        "alpha_crit": alpha_crit,
    }


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

    tail = raw[t_start_idx:]
    ms = np.sin(tail).mean(axis=1)
    mc = np.cos(tail).mean(axis=1)
    R_series = np.sqrt(ms * ms + mc * mc)
    R_steady = float(R_series.mean())
    R_final = float(R_series[-1])
    phi_diffs = raw[-1, :] - raw[-1, 0]
    phi_diffs_wrapped = (phi_diffs + np.pi) % (2 * np.pi) - np.pi
    phase_diff_std = float(np.std(phi_diffs_wrapped))

    target_points = 1500
    skip = max(1, steps // target_points)
    data = np.sin(raw[::skip].T)
    dx = skip * params.dt

    print(f"[DYNAMICS RK4] Omega: mean={float(np.mean(omegas)):.4f} "
          f"std={float(np.std(omegas)):.4f} "
          f"min={float(np.min(omegas)):.4f} max={float(np.max(omegas)):.4f} "
          f"R_steady={R_steady:.4f} phase_std={phase_diff_std:.3f}")

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
        "R_steady": R_steady,
        "R_final": R_final,
        "phase_diff_std": phase_diff_std,
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

        if params.n_neurons <= 2:
            # пара с фиксированным IC [0, 0.5]
            result_matrix, omega1, omega2 = compute_sync_regions(
                d_values, delta_values, params.fixed_alpha,
                steps, params.dt, params.n_param, params.base_gamma
            )
            print(f"[SCAN RK4 PAIR] N=2 Omega1: mean={float(np.mean(omega1)):.4f}, "
                  f"Omega2: mean={float(np.mean(omega2)):.4f}")
            extra = {
                "omega1": omega1.tolist(),
                "omega2": omega2.tolist(),
                "kernel": "pair",
                "n_used": 2,
            }
        else:
            # mean-field: Delta = разброс gamma по ансамблю, alpha фиксирован.
            # IC случайные по сиду, поэтому seed тут впервые имеет смысл.
            initial_phases = np.random.random(params.n_neurons) * 2 * np.pi
            result_matrix, omega_mean = compute_sync_regions_meanfield(
                d_values, delta_values, params.fixed_alpha,
                params.n_neurons, steps, params.dt, params.base_gamma,
                params.n_param, initial_phases
            )
            print(f"[SCAN RK4 MEAN-FIELD] N={params.n_neurons} "
                  f"Omega: mean={float(np.mean(omega_mean)):.4f}")
            extra = {
                "omega_mean": omega_mean.tolist(),
                "kernel": "meanfield",
                "n_used": params.n_neurons,
            }

        thresholds = _compute_sync_thresholds(
            result_matrix, d_values, delta_values, R_thresh=0.95
        )

        return {
            "status": "success",
            "map_z": result_matrix.tolist(),
            "map_x": d_values.tolist(),
            "map_y": delta_values.tolist(),
            "gamma_1": params.base_gamma,
            "delta_min": params.delta_min,
            "delta_max": params.delta_max,
            "fixed_alpha": params.fixed_alpha,
            "integrator": "RK4",
            "sync_thresholds": thresholds,
            **extra,
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

    thresholds = _compute_sync_thresholds_d_alpha(
        result_matrix, d_values, alpha_values, R_thresh=0.95
    )

    return {
        "status": "success",
        "map_z": result_matrix.tolist(),
        "map_x": alpha_values.tolist(),
        "map_y": d_values.tolist(),
        "omega_mean": omega_mean.tolist(),
        "gamma_1": params.base_gamma,
        "delta_gamma": params.delta_gamma,
        "integrator": "RK4",
        "sync_thresholds": thresholds,
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


# ---------------------------------------------------------------------------
# AI chat assistant (Gemini) with function calling
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_RU = """Ты — научный ассистент в интерфейсе NeuroSim Core, симуляторе ансамбля связанных φ-нейронов. Помогаешь студенту-дипломнику разбирать результаты численных экспериментов по фазовой синхронизации.

УРАВНЕНИЕ МОДЕЛИ:
    dφᵢ/dt = γᵢ − sin(φᵢ/n) + (d/N) · Σⱼ sin(φⱼ − φᵢ − α)

ОБОЗНАЧЕНИЯ:
• γᵢ — параметр возбудимости i-го нейрона. γ>1 — тонические спайки, γ≤1 — возбудимый режим (без предельного цикла).
• γ₁ (base_gamma) — частота первого нейрона; γᵢ = linspace(γ₁, γ₁+Δ, N).
• Δ (delta_gamma) — расстройка собственных частот по ансамблю.
• d (d_couple) — сила связи между нейронами.
• α — фазовый сдвиг связи. ВАЖНО: у нас нормировка `d·sin(φⱼ−φᵢ−α)` (НЕ Курамото-Сакагути `d·sin(α)·…`). Раскрытие: sin(Δφ−α) = sin(Δφ)·cos(α) − cos(Δφ)·sin(α).
  – α=0: чистое притяжение к синфазе (минимум потенциала при Δφ=0).
  – α=π/2: член становится −cos(Δφ); минимум всё равно при Δφ=0 → синфаза УСТОЙЧИВА (это НЕ граница диссипативности — в нашей нормировке α=π/2 продолжает стабилизировать sync).
  – α=2π/3 (≈α_crit): эффективное взаимодействие меняет знак на отталкивание; «купольная» область синхр. по d.
  – α=π: чистое отталкивание, устойчивая антифаза (Δφ=π).
  Поэтому НЕ переноси автоматически вывод Курамото-Сакагути «α=π/2 → синхр. разрушается»: в нашей нормализации это не так. Делай вывод из формулы.
• n (n_param) — параметр формы спайка. n=1 — классическая СПАЙКОВАЯ активность; n>1 — БЁРСТОВАЯ (пачечная) активность; цикл бёрстинга растягивается во времени в n раз относительно спайкового.
• N — размер ансамбля.

КЛЮЧЕВЫЕ МЕТРИКИ:
• R = |⟨exp(iφⱼ)⟩| ∈ [0,1] — параметр порядка Курамото. R≈1 — синфазная синхронизация; R≈0 — асинхронность.
• Ω — наблюдаемая средняя частота. Для изолированного нейрона Ω∞ = √(γ²−1) (инвариант по n, уравнение Адлера). Замены ψ=φ/n приводит уравнение к каноническому виду dψ/dt = γ/n − sin(ψ)/n.
• Классификация режимов по R: ≥0.95 — синфазная (полная) синхронизация; 0.7–0.95 — сильная частичная; 0.3–0.7 — противофазная или частичная; <0.3 — асинхронность или подавление колебаний.

ТИПЫ ГРАФИКОВ В ИНТЕРФЕЙСЕ:
• dynamics — пространственно-временная диаграмма sin(φᵢ(t)). Ось X = время, ось Y = номер нейрона.
• d_alpha — карта R(d, α) при фиксированном Δ. Карта динамических режимов.
• d_delta — карта R(d, Δ) при фиксированном α. При N=2 — пара осцилляторов с IC=[0, 0.5]; при N>2 — N-нейронное среднее поле.
• compare_n — временные ряды sin(φ(t)) изолированных нейронов при разных n.

ТЕРМИНОЛОГИЯ ОТЧЁТА (используй ИМЕННО ЭТИ ТЕРМИНЫ; они приняты руководителем):
• «область синхронизации», «зона синфазного режима» — НЕ «язык Арнольда» / «Arnold tongue».
• «осцилляторная смерть» (oscillator death) — режим, где R может быть высоким (фазы заморожены), а Ω→0 (нет вращения по предельному циклу). Колебания полностью гасятся диссипативной связью.
• «частотный захват» — все нейроны имеют одинаковую Ω (малое σ_Ω). «Фазовая синхронизация» — дополнительно R≈1. Различай: частотный захват ≠ фазовая синхронизация. При α≈2π/3 возможен частотный захват без фазового выравнивания (R≈0, но σ_Ω→0).
• «кластерная синхронизация» — ансамбль разбивается на группы с близкими внутри группы фазами; R ∈ [0.3, 0.7]; часто асимметричное распределение размеров групп; Ω≠0 (нейроны вращаются).
• «купольная форма» (арка) — для замкнутой области синхронизации на карте d×Δ при отталкивающей связи (α≈2π/3): синхр. ограничена снизу и сверху по d.
• «бёрстовая активность» (а не «пачечная активность через запятую»).

КАК ОПРЕДЕЛИТЬ РЕЖИМ ПО (R_steady, Ω, σ_Ω) ДЛЯ ДИНАМИКИ:
┌─────────────────────────────────┬──────────────┬──────────────┬───────────────────────┐
│ Режим                           │ R_steady     │ Ω vs Ω∞      │ σ_Ω (разброс Ω)       │
├─────────────────────────────────┼──────────────┼──────────────┼───────────────────────┤
│ Синфазная синхронизация         │ ≥0.95        │ Ω≈Ω∞ или ≈0  │ ≈0                    │
│ Кластерная (частичная) синхр.   │ 0.3–0.7      │ Ω<Ω∞, но >0  │ ≈0 (один общий ритм)  │
│ Частотный захват без фазового   │ ≈0 (<0.2)    │ Ω≈Ω∞         │ ≈0 (все на одной Ω)   │
│ Осцилляторная смерть            │ — не важно — │ Ω→0          │ ≈0                    │
│ Асинхронность                   │ ≈0           │ ω₀ для γ₀    │ ~Δ (расстройка ансам.)│
└─────────────────────────────────┴──────────────┴──────────────┴───────────────────────┘
ПРАВИЛО (приоритет, применяй СВЕРХУ ВНИЗ):
0. ПРИОРИТЕТ ЧАСТОТНОГО ЗАХВАТА. Если σ_Ω > Δ (разброс наблюдаемых частот ПРЕВЫШАЕТ исходную расстройку γ) — это АСИНХРОННОСТЬ, КАК БЫ ни выглядел R. Никакая «кластерная синхр.» по R∈[0.3,0.7] не валидна, если частоты не захвачены. Сначала проверь σ_Ω, потом R.
1. Если Ω→0 (Ω/Ω∞ < 0.05) — это ОСЦИЛЛЯТОРНАЯ СМЕРТЬ (даже если R высокий: фазы заморожены, но не вращаются). Это особенно типично при γ→1⁺ (хрупкий предельный цикл, легко гасится связью).
2. Если σ_Ω ≪ Δ (есть захват) и R≥0.95 — синфазная синхронизация. Ω близко к среднему γ.
3. Если σ_Ω ≪ Δ и R∈[0.3,0.7] — кластерная синхронизация (общий ритм, но фазы разбиты на группы).
4. Если σ_Ω ≪ Δ и R<0.2 — частотный захват БЕЗ фазового выравнивания (характерно для α≈2π/3).
5. Если σ_Ω ~ Δ — асинхронность (нет ни фазовой, ни частотной синхр.).
Никогда не классифицируй по R без проверки σ_Ω.

ПРАВИЛА ОТВЕТА:
1. Отвечай на русском, академическим стилем, как в дипломной работе. 1–4 абзаца или нумерованный список из 2–5 пунктов с короткими bold-заголовками.
2. НЕ выдумывай числа. Извлекай из переданной сводки (`result_summary`) или вызывай инструмент.
3. Когда возможно — сравнивай численные значения с теорией: Ω vs Ω∞=√(γ²−1), R vs пороги, d vs Δ-граница.
4. Если режим узнаётся (осцилляторная смерть, частотный захват без фазового, купольная форма, кластерная синхр.) — НАЗОВИ его, используя терминологию из списка выше.
5. Если в сводке данных мало для уверенного ответа — скажи об этом и предложи запустить tool `run_dynamics_tool`.
6. Когда видишь изображение графика — описывай и его (форма зон, цвета, асимметрии), а не только числа.
7. Структура хорошего ответа: что наблюдаем → как называется режим → как это согласуется с теорией / параметрами → физическая интерпретация.
"""


def run_dynamics_tool(
    d_couple: float,
    alpha: float,
    n_param: float = 3.0,
    base_gamma: float = 1.01,
    delta_gamma: float = 0.005,
    n_neurons: int = 50,
    t_end: float = 200.0,
) -> dict:
    """Короткая симуляция динамики ансамбля и сводная статистика.

    Использовать при гипотетических вопросах вида "что будет при d=0.1, alpha=pi/4"
    или "достигнет ли синхронизации при d=0.05 и alpha=0".

    Args:
        d_couple: сила связи (0..0.5)
        alpha: фазовый сдвиг связи в радианах (0..pi)
        n_param: параметр формы спайка n (1..5)
        base_gamma: gamma_1, базовая частота (обычно 1.01)
        delta_gamma: разброс gamma по ансамблю (0.001..0.05)
        n_neurons: число нейронов (2..100)
        t_end: длительность моделирования (100..400)

    Returns:
        R_steady, omega_mean, omega_std, regime (текстовый класс режима).
    """
    dt = 0.1
    n_neurons = max(2, min(int(n_neurons), 100))
    t_end = max(50.0, min(float(t_end), 400.0))
    steps = int(t_end / dt)
    np.random.seed(42)
    gammas = np.linspace(base_gamma, base_gamma + delta_gamma, n_neurons)
    initial_phases = np.random.random(n_neurons) * 2 * np.pi

    raw = compute_dynamics_rk4(
        n_neurons, steps, dt, gammas,
        float(n_param), float(d_couple), float(alpha), initial_phases
    )
    half = steps // 2
    ms = np.sin(raw[half:]).mean(axis=1)
    mc = np.cos(raw[half:]).mean(axis=1)
    R_series = np.sqrt(ms * ms + mc * mc)
    R_mean = float(R_series.mean())

    t_window = (steps - 1 - half) * dt
    if t_window > 0:
        omegas = (raw[-1] - raw[half]) / t_window
    else:
        omegas = np.zeros(n_neurons)

    if R_mean >= 0.95:
        regime = "полная синфазная синхронизация"
    elif R_mean >= 0.7:
        regime = "сильная частичная синхронизация"
    elif R_mean >= 0.3:
        regime = "частичная или противофазная синхронизация"
    else:
        regime = "асинхронность или подавление колебаний"

    print(f"[AI TOOL run_dynamics] d={d_couple:.4f} alpha={alpha:.4f} n={n_param} "
          f"N={n_neurons} R={R_mean:.3f} Omega={float(np.mean(omegas)):.3f} ({regime})")

    return {
        "R_steady": round(R_mean, 4),
        "omega_mean": round(float(np.mean(omegas)), 4),
        "omega_std": round(float(np.std(omegas)), 4),
        "regime": regime,
    }


def isolated_frequency_tool(gamma: float) -> dict:
    """Асимптотическая частота изолированного нейрона: Omega_inf = sqrt(gamma^2-1) при gamma>1.

    При gamma<=1 нейрон в возбудимом режиме, постоянных колебаний нет.
    Использовать для сравнения наблюдаемой в ансамбле частоты с одиночной.

    Args:
        gamma: параметр возбудимости.

    Returns:
        omega_infinity (None для возбудимого режима), is_spiking.
    """
    if gamma > 1:
        omega = float(np.sqrt(gamma * gamma - 1))
        print(f"[AI TOOL isolated_frequency] gamma={gamma} Omega_inf={omega:.5f}")
        return {"omega_infinity": round(omega, 5), "is_spiking": True}
    print(f"[AI TOOL isolated_frequency] gamma={gamma} <= 1, excitable regime")
    return {
        "omega_infinity": None,
        "is_spiking": False,
        "note": "gamma<=1: возбудимый режим, постоянных колебаний нет.",
    }


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    plot_kind: str = ""
    params: dict = {}
    result_summary: dict = {}
    image_b64: Optional[str] = None
    history: List[ChatMessage] = []
    model: Optional[str] = None


_genai_client = None
_genai_init_error: Optional[str] = None


def _get_genai_client():
    global _genai_client, _genai_init_error
    if _genai_client is not None:
        return _genai_client
    if _genai_init_error is not None:
        raise HTTPException(status_code=503, detail=_genai_init_error)

    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        load_dotenv(dotenv_path=env_path)
    except ImportError:
        pass

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        _genai_init_error = (
            "GEMINI_API_KEY не задан. Добавь ключ в core/.env "
            "(GEMINI_API_KEY=...) или экспортируй в окружении."
        )
        raise HTTPException(status_code=503, detail=_genai_init_error)

    try:
        from google import genai as _genai_mod
        _genai_client = _genai_mod.Client(api_key=api_key)
        print("[AI CHAT] Gemini client initialized")
        return _genai_client
    except Exception as e:
        _genai_init_error = f"Не удалось инициализировать google-genai: {e}"
        raise HTTPException(status_code=503, detail=_genai_init_error)


@app.post("/ai/chat")
def ai_chat(req: ChatRequest):
    client = _get_genai_client()
    from google.genai import types as genai_types

    relevant_param_keys = {
        "n_neurons", "t_end", "dt", "n_param", "base_gamma", "delta_gamma",
        "d_couple", "alpha", "seed",
        "scan_type", "d_min", "d_max", "alpha_min", "alpha_max",
        "delta_min", "delta_max", "fixed_alpha",
    }
    ctx_lines = [f"Тип графика: {req.plot_kind or 'не указан'}"]
    if req.params:
        kv = []
        for k, v in req.params.items():
            if k not in relevant_param_keys:
                continue
            if isinstance(v, float):
                kv.append(f"{k}={v:.5g}")
            else:
                kv.append(f"{k}={v}")
        if kv:
            ctx_lines.append("Параметры запуска: " + ", ".join(kv))
    if req.result_summary:
        ctx_lines.append(f"Сводка результата: {req.result_summary}")
    context_block = "\n".join(ctx_lines)

    model_name = req.model or os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
    is_gemma = model_name.lower().startswith("gemma")

    contents = []
    for msg in req.history[-8:]:
        role = "user" if msg.role == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg.content}]})

    user_text = f"[КОНТЕКСТ ГРАФИКА]\n{context_block}\n\n[ВОПРОС]\n{req.question}"
    user_parts: list = [{"text": user_text}]
    if req.image_b64:
        user_parts.append({
            "inline_data": {"mime_type": "image/png", "data": req.image_b64}
        })
    contents.append({"role": "user", "parts": user_parts})

    tools_enabled = os.environ.get("GEMINI_TOOLS", "1") not in ("0", "false", "no")
    tools_list = [run_dynamics_tool, isolated_frequency_tool] if tools_enabled else None
    # потолок на цепочку tool-call/tool-response, защита от зацикливания
    afc_cfg = genai_types.AutomaticFunctionCallingConfig(maximum_remote_calls=8)
    if is_gemma:
        # Gemma не принимает system_instruction в config, поэтому промпт
        # встраивается в первый user-turn. thinking_config она тоже не поддерживает.
        config = genai_types.GenerateContentConfig(
            tools=tools_list,
            temperature=0.4,
            automatic_function_calling=afc_cfg,
        )
        contents = ([{"role": "user", "parts": [{"text": SYSTEM_PROMPT_RU}]},
                     {"role": "model", "parts": [{"text": "Понял. Отвечаю на русском, академическим стилем, использую терминологию отчёта. Если нужно, буду вызывать инструменты."}]}]
                    + contents)
    else:
        # thinking_budget=-1: динамическое thinking (для 3.1-flash-lite дефолт 0).
        config = genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT_RU,
            tools=tools_list,
            temperature=0.4,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=-1),
            automatic_function_calling=afc_cfg,
        )

    print(f"[AI CHAT] model={model_name} plot={req.plot_kind} q='{req.question[:80]}' "
          f"history={len(req.history)} image={'yes' if req.image_b64 else 'no'}")

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
            retry_after = None
            import re
            m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", msg)
            if m:
                retry_after = float(m.group(1))
            print(f"[AI CHAT QUOTA] retry_after={retry_after}s")
            detail = "Лимит запросов к Gemini исчерпан"
            if retry_after:
                detail += f", повтори через {int(retry_after)+1} сек."
            else:
                detail += ", подожди около 30 сек."
            raise HTTPException(status_code=429, detail=detail)
        print(f"[AI CHAT ERROR] {e}")
        raise HTTPException(status_code=502, detail=f"Gemini error: {e}")

    answer = (response.text or "").strip() or "(модель вернула пустой ответ)"

    tools_used: list = []
    afc_history = getattr(response, "automatic_function_calling_history", None) or []
    for entry in afc_history:
        parts = getattr(entry, "parts", None) or []
        for p in parts:
            fc = getattr(p, "function_call", None)
            if fc and getattr(fc, "name", None):
                tools_used.append(fc.name)

    print(f"[AI CHAT] reply_len={len(answer)} tools_used={tools_used}")
    return {"answer": answer, "tools_used": tools_used}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
