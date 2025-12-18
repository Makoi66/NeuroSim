from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from numba import jit, prange
import uvicorn

app = FastAPI()


class SimulationParams(BaseModel):
    n_neurons: int = 50
    t_end: float = 200.0
    dt: float = 0.05
    n_param: float = 3.0
    base_gamma: float = 1.01
    d_couple: float = 0.0
    alpha: float = 0.0
    seed: int = 42


class ScanParams(BaseModel):
    n_neurons: int = 2
    t_end: float = 200.0
    dt: float = 0.1
    n_param: float = 3.0
    base_gamma: float = 1.01

    scan_type: str = "d_alpha"

    d_min: float = 0.0
    d_max: float = 0.5
    d_steps: int = 30

    alpha_min: float = 0.0
    alpha_max: float = 3.14
    alpha_steps: int = 30

    delta_min: float = 0.0
    delta_max: float = 0.02
    delta_steps: int = 30
    fixed_alpha: float = 2.094

    seed: int = 42


@jit(nopython=True, parallel=True)
def compute_dynamics_jit(n_neurons, steps, dt, gammas, n_param, d_couple, alpha, initial_phases):
    phases = np.zeros((steps, n_neurons))
    for i in range(n_neurons):
        phases[0, i] = initial_phases[i]

    for t in range(steps - 1):
        sum_sin = 0.0
        sum_cos = 0.0
        for j in range(n_neurons):
            s = phases[t, j]
            sum_sin += np.sin(s)
            sum_cos += np.cos(s)

        mean_sin = sum_sin / n_neurons
        mean_cos = sum_cos / n_neurons

        ms_rot = mean_sin * np.cos(alpha) - mean_cos * np.sin(alpha)
        mc_rot = mean_cos * np.cos(alpha) + mean_sin * np.sin(alpha)

        for i in prange(n_neurons):
            phi = phases[t, i]
            interaction = (ms_rot * np.cos(phi) - mc_rot * np.sin(phi))
            dphi = gammas[i] - np.sin(phi / n_param) + d_couple * interaction
            phases[t + 1, i] = phi + dphi * dt
    return phases


@jit(nopython=True, parallel=True)
def compute_arnold_tongue(d_vals, delta_vals, alpha, steps, dt, n_param, base_gamma):
    n_d = len(d_vals)
    n_delta = len(delta_vals)
    results = np.zeros((n_delta, n_d))

    for i in prange(n_delta):
        delta = delta_vals[i]
        gammas = np.array([base_gamma, base_gamma + delta])

        for j in range(n_d):
            d = d_vals[j]
            phases = np.array([0.0, 0.5])

            sync_metric = 0.0
            count = 0
            start_measure = int(steps * 0.6)

            for t in range(steps):
                p1 = phases[0]
                p2 = phases[1]

                int_1 = np.sin(p2 - p1 - alpha)
                int_2 = np.sin(p1 - p2 - alpha)

                dp1 = gammas[0] - np.sin(p1 / n_param) + d * int_1
                dp2 = gammas[1] - np.sin(p2 / n_param) + d * int_2

                phases[0] += dp1 * dt
                phases[1] += dp2 * dt

                if t > start_measure:
                    ms = (np.sin(phases[0]) + np.sin(phases[1])) / 2
                    mc = (np.cos(phases[0]) + np.cos(phases[1])) / 2
                    R = np.sqrt(ms ** 2 + mc ** 2)
                    sync_metric += R
                    count += 1

            if count > 0:
                results[i, j] = sync_metric / count

    return results


@jit(nopython=True, parallel=True)
def compute_grid_jit(d_vals, alpha_vals, n_neurons, steps, dt, gammas, n_param, initial_phases):
    n_d = len(d_vals)
    n_a = len(alpha_vals)
    results = np.zeros((n_d, n_a))

    for i in prange(n_d):
        d = d_vals[i]
        for j in range(n_a):
            alpha = alpha_vals[j]
            current_phases = initial_phases.copy()
            order_param_accum = 0.0
            count = 0
            start_measure = int(steps * 0.5)

            for t in range(steps):
                sum_sin = np.sum(np.sin(current_phases))
                sum_cos = np.sum(np.cos(current_phases))
                mean_sin = sum_sin / n_neurons
                mean_cos = sum_cos / n_neurons

                if t > start_measure:
                    R = np.sqrt(mean_sin ** 2 + mean_cos ** 2)
                    order_param_accum += R
                    count += 1

                ms_rot = mean_sin * np.cos(alpha) - mean_cos * np.sin(alpha)
                mc_rot = mean_cos * np.cos(alpha) + mean_sin * np.sin(alpha)
                interaction = ms_rot * np.cos(current_phases) - mc_rot * np.sin(current_phases)
                dphi = gammas - np.sin(current_phases / n_param) + d * interaction
                current_phases += dphi * dt

            if count > 0:
                results[i, j] = order_param_accum / count
    return results


@app.post("/simulate")
def run_simulation(params: SimulationParams):
    np.random.seed(params.seed)
    steps = int(params.t_end / params.dt)

    # Очень маленький разброс, чтобы увидеть кластерную синхронизацию
    # Если разброс большой, будет просто хаос
    gammas = np.linspace(params.base_gamma, params.base_gamma + 0.005, params.n_neurons)

    initial_phases = np.random.random(params.n_neurons) * 2 * np.pi

    print(f"Dynamics Mode: N={params.n_neurons}, steps={steps}, n={params.n_param}, alpha={params.alpha}")

    raw = compute_dynamics_jit(
        params.n_neurons, steps, params.dt, gammas,
        params.n_param, params.d_couple, params.alpha, initial_phases
    )

    target_points = 1500
    skip = max(1, steps // target_points)

    data = np.sin(raw[::skip].T)

    return {"status": "success", "heatmap": data.tolist()}


@app.post("/scan")
def run_scan(params: ScanParams):
    np.random.seed(params.seed)
    steps = int(params.t_end / params.dt)

    print(f"Scan Mode: {params.scan_type}")

    if params.scan_type == "d_delta":
        d_values = np.linspace(params.d_min, params.d_max, params.d_steps)
        delta_values = np.linspace(params.delta_min, params.delta_max, params.delta_steps)

        result_matrix = compute_arnold_tongue(
            d_values, delta_values, params.fixed_alpha,
            steps, params.dt, params.n_param, params.base_gamma
        )

        return {
            "status": "success",
            "map_z": result_matrix.tolist(),
            "map_x": d_values.tolist(),
            "map_y": delta_values.tolist()
        }

    else:
        d_values = np.linspace(params.d_min, params.d_max, params.d_steps)
        alpha_values = np.linspace(params.alpha_min, params.alpha_max, params.alpha_steps)

        gammas = np.linspace(params.base_gamma, params.base_gamma + 0.01, params.n_neurons)
        initial_phases = np.random.random(params.n_neurons) * 2 * np.pi

        result_matrix = compute_grid_jit(
            d_values, alpha_values,
            params.n_neurons, steps, params.dt,
            gammas, params.n_param, initial_phases
        )

        return {
            "status": "success",
            "map_z": result_matrix.tolist(),
            "map_x": alpha_values.tolist(),
            "map_y": d_values.tolist()
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)