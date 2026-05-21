import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';

// Discrete (step-based) colorscale for the synchronization-region map.
// Bands chosen to separate the 4 dynamical regimes the supervisor asked us
// to distinguish visually (Замечания №7, №8):
//   R < 0.30  : asynchronous / oscillation death
//   0.30–0.70 : antiphase / partial sync
//   0.70–0.95 : high partial sync
//   R ≥ 0.95  : in-phase (full) sync
const SYNC_COLORSCALE = [
  [0.00, '#0d1117'], [0.30, '#0d1117'],
  [0.30, '#bb1e62'], [0.70, '#bb1e62'],
  [0.70, '#f59e0b'], [0.95, '#f59e0b'],
  [0.95, '#22d3ee'], [1.00, '#22d3ee'],
];

// Discrete colorscale for the dynamics heatmap (sin(phi) ∈ [-1, 1]).
// 4 bands so trough / falling / rising / peak phases are visually separated.
const DYNAMICS_COLORSCALE = [
  [0.00, '#1e1b4b'], [0.25, '#1e1b4b'],
  [0.25, '#4338ca'], [0.50, '#4338ca'],
  [0.50, '#f59e0b'], [0.75, '#f59e0b'],
  [0.75, '#fef08a'], [1.00, '#fef08a'],
];

function App() {
  const [mode, setMode] = useState('dynamics');

  const [params, setParams] = useState({
    n_neurons: 50,
    t_end: 400,
    dt: 0.1,
    n_param: 3.0,
    base_gamma: 1.01,
    delta_gamma: 0.005,
    seed: 42,
    d_couple: 0.12,
    alpha: 2.09,

    scan_type: 'd_alpha',
    d_min: 0.0, d_max: 0.2,
    alpha_min: 0.0, alpha_max: 3.14,
    delta_min: 0.0, delta_max: 0.012,
    fixed_alpha: 2.0944,

    d_steps: 40, alpha_steps: 40, delta_steps: 40,

    cmp_t_end: 80.0,
    cmp_dt: 0.02,
    cmp_n_values: [1, 3, 5],
  });

  const [isRandomSeed, setIsRandomSeed] = useState(false);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isOnline, setIsOnline] = useState(false);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        await axios.get('http://localhost:8080/health', { timeout: 1500 });
        setIsOnline(true);
      } catch {
        setIsOnline(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleChange = (e) => {
    let val = parseFloat(e.target.value);
    if (isNaN(val) && e.target.name === 'seed') val = 0;
    setParams({ ...params, [e.target.name]: val });
  };

  const runSimulation = async () => {
    setData(null);
    setLoading(true);
    const currentSeed = isRandomSeed ? Math.floor(Math.random() * 1000000) : params.seed;

    let endpoint = '/api/run';
    let payload = { ...params, seed: currentSeed };

    if (mode === 'map') {
      endpoint = '/api/scan';
    } else if (mode === 'compare_n') {
      endpoint = '/api/compare_n';
      payload = {
        t_end: params.cmp_t_end,
        dt: params.cmp_dt,
        base_gamma: params.base_gamma,
        n_values: params.cmp_n_values,
      };
    }

    try {
      const response = await axios.post(`http://localhost:8080${endpoint}`, payload);
      if (response.data) setData(response.data);
    } catch (err) {
      console.error(err);
      setData(null);
      const detail = err?.response?.data?.detail || err?.response?.statusText || err?.message || 'unknown';
      alert(
        '⚠ ОШИБКА ВЫЧИСЛИТЕЛЬНОГО ЯДРА\n\n' +
        'Запрос к бэкенду не выполнен.\n' +
        'Возможные причины: ядро упало, нехватка памяти при выбранном разрешении, недоступен прокси.\n\n' +
        'Детали: ' + detail
      );
    } finally {
      setLoading(false);
    }
  };

  const getXTitle = () => {
    if (mode === 'dynamics') return 'Время t';
    if (mode === 'compare_n') return 'Время t';
    if (params.scan_type === 'd_alpha') return 'Фаза (α)';
    return 'Связь (d)';
  };

  const getYTitle = () => {
    if (mode === 'dynamics') return 'Индекс Нейрона';
    if (mode === 'compare_n') return 'sin(φ(t))';
    if (params.scan_type === 'd_alpha') return 'Связь (d)';
    return 'Расстройка (Δ)';
  };

  const buildPlotData = () => {
    if (!data) return [];
    if (mode === 'dynamics') {
      const nCols = data.heatmap && data.heatmap[0] ? data.heatmap[0].length : 1;
      // Prefer dx from the backend (matches the actual subsampling). Fallback
      // to a UI-side estimate only if the response predates that field.
      const dx = typeof data.dx === 'number' ? data.dx : (params.t_end / nCols);
      return [{
        z: data.heatmap, type: 'heatmap',
        x0: 0, dx,
        colorscale: DYNAMICS_COLORSCALE, zmin: -1, zmax: 1,
        showscale: true,
        colorbar: { title: 'sin(φ)', tickvals: [-1, -0.5, 0, 0.5, 1] },
      }];
    }
    if (mode === 'compare_n') {
      const palette = ['#22d3ee', '#f59e0b', '#bb1e62'];
      return data.traces.map((tr, i) => ({
        x: tr.t, y: tr.sin_phi,
        type: 'scatter', mode: 'lines',
        name: `n = ${tr.n} (Ω=${tr.omega.toFixed(3)})`,
        line: { width: 2, color: palette[i % palette.length] },
      }));
    }
    return [{
      z: data.map_z, x: data.map_x, y: data.map_y,
      type: 'heatmap',
      colorscale: SYNC_COLORSCALE, zmin: 0, zmax: 1,
      showscale: true,
      colorbar: {
        title: 'Sync R',
        tickvals: [0.15, 0.5, 0.825, 0.975],
        ticktext: ['async/death', 'antiphase', 'partial', 'in-phase'],
      },
    }];
  };

  return (
    <>
    <aside className="w-[360px] shrink-0 flex flex-col border-r border-slate-800 bg-panel-dark/60 backdrop-blur-xl relative z-30 shadow-glass">

    <div className="h-20 flex items-center gap-3 px-6 border-b border-slate-800/60 bg-bg-dark/40">
    <div className="relative w-10 h-10 flex items-center justify-center">
    <div className="absolute inset-0 bg-primary/20 rounded-lg blur-md animate-pulse"></div>
    <div className="relative w-full h-full rounded-lg bg-gradient-to-br from-primary/20 to-blue-600/20 border border-primary/50 flex items-center justify-center">
    <span className="material-symbols-outlined text-primary-glow text-2xl">hub</span>
    </div>
    </div>
    <div>
    <h1 className="text-xl font-bold text-white tracking-wide font-mono">NeuroSim Core</h1>
    <p className="text-[10px] text-primary/60 uppercase tracking-[0.2em]">Lab Edition {__APP_VERSION__}</p>
    </div>
    </div>

    <div className="flex-1 overflow-y-auto p-6 space-y-8">
    <div className="p-1 rounded-xl bg-slate-900/80 border border-slate-800 flex relative overflow-hidden gap-1">
    <button onClick={() => { setMode('dynamics'); setData(null); }} className={`flex-1 py-3 px-1 rounded-lg text-[10px] font-bold uppercase tracking-wider transition-all relative z-10 ${mode === 'dynamics' ? 'text-white bg-primary/20 border border-primary/40 shadow-neon-cyan' : 'text-slate-500 hover:text-white hover:bg-white/5'}`}>Динамика</button>
    <button onClick={() => { setMode('map'); setData(null); }} className={`flex-1 py-3 px-1 rounded-lg text-[10px] font-bold uppercase tracking-wider transition-all relative z-10 ${mode === 'map' ? 'text-white bg-secondary/20 border border-secondary/40 shadow-neon-purple' : 'text-slate-500 hover:text-white hover:bg-white/5'}`}>Области синх.</button>
    <button onClick={() => { setMode('compare_n'); setData(null); }} className={`flex-1 py-3 px-1 rounded-lg text-[10px] font-bold uppercase tracking-wider transition-all relative z-10 ${mode === 'compare_n' ? 'text-white bg-amber-400/20 border border-amber-400/40' : 'text-slate-500 hover:text-white hover:bg-white/5'}`}>Сравн. n</button>
    </div>

    <div className="p-3 rounded-xl bg-slate-900/40 border border-slate-800 flex items-center justify-between text-[10px] font-mono">
    <div className="flex flex-col">
    <span className="text-slate-500 uppercase tracking-wider">γ₁ (base_gamma)</span>
    <span className="text-primary text-sm">{params.base_gamma.toFixed(5)}</span>
    </div>
    <div className="flex flex-col text-right">
    <span className="text-slate-500 uppercase tracking-wider">Δ (расстройка)</span>
    <span className="text-secondary text-sm">{(mode === 'map' && params.scan_type === 'd_delta'
      ? `[${params.delta_min.toFixed(4)}, ${params.delta_max.toFixed(4)}]`
      : params.delta_gamma.toFixed(5))}</span>
    </div>
    </div>

    {mode === 'dynamics' && (
      <div className="space-y-5 animate-fade-in">
      <div className="flex items-center gap-2 text-primary border-b border-slate-800 pb-2">
      <span className="material-symbols-outlined text-lg">science</span>
      <h3 className="text-xs font-bold uppercase tracking-[0.1em] text-white">Параметры Модели</h3>
      </div>

      <div className="grid grid-cols-2 gap-3 mb-2">
      <div className="space-y-2">
      <label className="text-[10px] text-slate-400 font-bold uppercase">Форма (n)</label>
      <input className="w-full bg-slate-900/50 border border-slate-700 rounded p-2 text-xs text-white font-mono focus:border-primary outline-none" type="number" step="0.1" name="n_param" value={params.n_param} onChange={handleChange}/>
      </div>
      <div className="space-y-2">
      <label className="text-[10px] text-slate-400 font-bold uppercase">γ₁ (base_gamma)</label>
      <input className="w-full bg-slate-900/50 border border-slate-700 rounded p-2 text-xs text-white font-mono focus:border-primary outline-none" type="number" step="0.01" name="base_gamma" value={params.base_gamma} onChange={handleChange}/>
      </div>
      <div className="space-y-2 col-span-2">
      <label className="text-[10px] text-slate-400 font-bold uppercase">Δ (расстройка по ансамблю)</label>
      <input className="w-full bg-slate-900/50 border border-slate-700 rounded p-2 text-xs text-white font-mono focus:border-primary outline-none" type="number" step="0.001" name="delta_gamma" value={params.delta_gamma} onChange={handleChange}/>
      </div>
      </div>

      <div className="space-y-3">
      <div className="flex justify-between items-center text-xs"><label className="text-slate-400 font-medium">Связь (d)</label><span className="font-mono text-primary bg-primary/10 px-2 py-0.5 rounded border border-primary/20">{params.d_couple}</span></div>
      <input type="range" name="d_couple" min="0" max="0.5" step="0.005" value={params.d_couple} onChange={handleChange} className="w-full text-primary"/>
      </div>
      <div className="space-y-3">
      <div className="flex justify-between items-center text-xs"><label className="text-slate-400 font-medium">Фаза (α)</label><span className="font-mono text-secondary bg-secondary/10 px-2 py-0.5 rounded border border-secondary/20">{params.alpha}</span></div>
      <input type="range" name="alpha" min="0" max="3.14" step="0.01" value={params.alpha} onChange={handleChange} className="w-full text-secondary"/>
      </div>
      </div>
    )}

    {mode === 'compare_n' && (
      <div className="space-y-5 animate-fade-in">
      <div className="flex items-center gap-2 text-amber-400 border-b border-slate-800 pb-2">
      <span className="material-symbols-outlined text-lg">timeline</span>
      <h3 className="text-xs font-bold uppercase tracking-[0.1em] text-white">Временные реализации φ(t)</h3>
      </div>
      <p className="text-[10px] text-slate-400 leading-relaxed">
      Сравнение φ(t) изолированного нейрона при <span className="font-mono text-amber-400">n = 1, 3, 5</span> —
      одиночные спайки против пачечной (bursting) активности.
      </p>
      <div className="grid grid-cols-2 gap-3">
      <div className="space-y-1">
      <label className="text-[10px] text-slate-500 uppercase font-bold">T end</label>
      <input className="w-full bg-slate-900/50 border border-slate-700 rounded p-2 text-xs text-white font-mono focus:border-amber-400 outline-none" type="number" name="cmp_t_end" value={params.cmp_t_end} onChange={handleChange}/>
      </div>
      <div className="space-y-1">
      <label className="text-[10px] text-slate-500 uppercase font-bold">dt</label>
      <input className="w-full bg-slate-900/50 border border-slate-700 rounded p-2 text-xs text-white font-mono focus:border-amber-400 outline-none" type="number" step="0.01" name="cmp_dt" value={params.cmp_dt} onChange={handleChange}/>
      </div>
      </div>
      <div className="space-y-1">
      <label className="text-[10px] text-slate-500 uppercase font-bold">γ₁ (base_gamma)</label>
      <input className="w-full bg-slate-900/50 border border-slate-700 rounded p-2 text-xs text-white font-mono focus:border-amber-400 outline-none" type="number" step="0.01" name="base_gamma" value={params.base_gamma} onChange={handleChange}/>
      </div>
      <div className="text-[10px] text-slate-500 font-mono">
      n_values = [{params.cmp_n_values.join(', ')}]
      </div>
      </div>
    )}

    {mode === 'map' && (
      <div className="space-y-5 animate-fade-in">
      <div className="flex items-center gap-2 text-secondary border-b border-slate-800 pb-2">
      <span className="material-symbols-outlined text-lg">grid_view</span>
      <h3 className="text-xs font-bold uppercase tracking-[0.1em] text-white">Сканирование</h3>
      </div>

      <div className="flex gap-2 mb-4">
      <button onClick={() => setParams({...params, scan_type: 'd_alpha'})} className={`flex-1 py-2 text-[10px] font-bold border rounded ${params.scan_type === 'd_alpha' ? 'border-secondary text-white bg-secondary/10' : 'border-slate-700 text-slate-500'}`}>D vs Alpha</button>
      <button onClick={() => setParams({...params, scan_type: 'd_delta'})} className={`flex-1 py-2 text-[10px] font-bold border rounded ${params.scan_type === 'd_delta' ? 'border-secondary text-white bg-secondary/10' : 'border-slate-700 text-slate-500'}`}>D vs Delta (Δ)</button>
      </div>

      <div className="grid grid-cols-2 gap-3">
      <div className="space-y-1"><label className="text-[10px] text-slate-500 uppercase font-bold">D Min</label><input className="w-full bg-slate-900/50 border border-slate-700 rounded p-2 text-xs text-white font-mono focus:border-secondary outline-none" type="number" name="d_min" value={params.d_min} onChange={handleChange}/></div>
      <div className="space-y-1"><label className="text-[10px] text-slate-500 uppercase font-bold">D Max</label><input className="w-full bg-slate-900/50 border border-slate-700 rounded p-2 text-xs text-white font-mono focus:border-secondary outline-none" type="number" name="d_max" value={params.d_max} onChange={handleChange}/></div>

      {params.scan_type === 'd_alpha' ? (
        <>
        <div className="space-y-1"><label className="text-[10px] text-slate-500 uppercase font-bold">α Min</label><input className="w-full bg-slate-900/50 border border-slate-700 rounded p-2 text-xs text-white font-mono focus:border-secondary outline-none" type="number" name="alpha_min" value={params.alpha_min} onChange={handleChange}/></div>
        <div className="space-y-1"><label className="text-[10px] text-slate-500 uppercase font-bold">α Max</label><input className="w-full bg-slate-900/50 border border-slate-700 rounded p-2 text-xs text-white font-mono focus:border-secondary outline-none" type="number" name="alpha_max" value={params.alpha_max} onChange={handleChange}/></div>
        </>
      ) : (
        <>
        <div className="space-y-1"><label className="text-[10px] text-slate-500 uppercase font-bold">Δ Min</label><input className="w-full bg-slate-900/50 border border-slate-700 rounded p-2 text-xs text-white font-mono focus:border-secondary outline-none" type="number" name="delta_min" value={params.delta_min} onChange={handleChange}/></div>
        <div className="space-y-1"><label className="text-[10px] text-slate-500 uppercase font-bold">Δ Max</label><input className="w-full bg-slate-900/50 border border-slate-700 rounded p-2 text-xs text-white font-mono focus:border-secondary outline-none" type="number" name="delta_max" value={params.delta_max} onChange={handleChange}/></div>
        </>
      )}
      </div>

      {params.scan_type === 'd_delta' && (
        <div className="mt-2 space-y-1">
        <label className="text-[10px] text-slate-500 uppercase font-bold">Фикс. Alpha</label>
        <input className="w-full bg-slate-900/50 border border-slate-700 rounded p-2 text-xs text-white font-mono focus:border-secondary outline-none" type="number" name="fixed_alpha" value={params.fixed_alpha} onChange={handleChange}/>
        </div>
      )}
      <div className="mt-2 grid grid-cols-2 gap-3">
      <div className="space-y-1">
      <label className="text-[10px] text-slate-500 uppercase font-bold">Форма (n)</label>
      <input className="w-full bg-slate-900/50 border border-slate-700 rounded p-2 text-xs text-white font-mono focus:border-secondary outline-none" type="number" name="n_param" value={params.n_param} onChange={handleChange}/>
      </div>
      <div className="space-y-1">
      <label className="text-[10px] text-slate-500 uppercase font-bold">γ₁ (base_gamma)</label>
      <input className="w-full bg-slate-900/50 border border-slate-700 rounded p-2 text-xs text-white font-mono focus:border-secondary outline-none" type="number" step="0.01" name="base_gamma" value={params.base_gamma} onChange={handleChange}/>
      </div>
      </div>

      <div className="space-y-3 pt-2 border-t border-slate-800 mt-2">
      <div className="flex justify-between items-center"><label className="text-[10px] text-slate-500 uppercase font-bold">Разрешение</label><span className="text-[10px] font-mono text-secondary bg-secondary/10 px-2 rounded">{params.d_steps} x {params.scan_type === 'd_alpha' ? params.alpha_steps : params.delta_steps}</span></div>
      <div className="flex gap-2">
      <input type="number" className="w-16 bg-slate-900/50 border border-slate-700 rounded p-2 text-xs text-white font-mono focus:border-secondary outline-none text-center" value={params.d_steps} onChange={(e) => { const val = parseInt(e.target.value) || 10; setParams({...params, d_steps: val, alpha_steps: val, delta_steps: val}); }}/>
      <div className="flex flex-1 gap-1">
      <button onClick={() => setParams({...params, d_steps: 50, alpha_steps: 50, delta_steps: 50})} className="flex-1 px-1 py-1 text-[10px] bg-slate-800 hover:bg-slate-700 rounded text-slate-400 border border-slate-700">50</button>
      <button onClick={() => setParams({...params, d_steps: 100, alpha_steps: 100, delta_steps: 100})} className="flex-1 px-1 py-1 text-[10px] bg-slate-800 hover:bg-slate-700 rounded text-slate-400 border border-slate-700">100</button>
      <button onClick={() => setParams({...params, d_steps: 250, alpha_steps: 250, delta_steps: 250})} className="flex-1 px-1 py-1 text-[10px] bg-slate-800 hover:bg-slate-700 rounded text-slate-400 border border-slate-700">250</button>
      </div>
      </div>
      </div>
      </div>
    )}

    <div className="space-y-5">
    <div className="flex items-center gap-2 text-primary border-b border-slate-800 pb-2"><span className="material-symbols-outlined text-lg">memory</span><h3 className="text-xs font-bold uppercase tracking-[0.1em] text-white">Система</h3></div>
    <div className="grid grid-cols-2 gap-4">
    <div className="space-y-2 group"><label className="text-[10px] text-slate-500 uppercase tracking-wider font-bold group-focus-within:text-white">Нейроны (N)</label><input className="w-full bg-slate-900/50 border border-slate-700 rounded-lg p-3 text-xs text-white font-mono focus:border-primary outline-none" type="number" name="n_neurons" value={params.n_neurons} onChange={handleChange}/></div>
    <div className="space-y-2 group"><label className="text-[10px] text-slate-500 uppercase tracking-wider font-bold group-focus-within:text-white">Время (T)</label><input className="w-full bg-slate-900/50 border border-slate-700 rounded-lg p-3 text-xs text-white font-mono focus:border-primary outline-none" type="number" name="t_end" value={params.t_end} onChange={handleChange}/></div>
    </div>
    <div className="bg-slate-900/40 p-4 rounded-xl border border-slate-800/50 space-y-4">
    <div className="flex items-center justify-between"><div className="flex items-center gap-2"><span className="material-symbols-outlined text-slate-400 text-sm">casino</span><label className="text-xs text-slate-300 font-medium">Случайное число</label></div><label className="relative inline-flex items-center cursor-pointer"><input className="sr-only peer" type="checkbox" checked={isRandomSeed} onChange={e => setIsRandomSeed(e.target.checked)}/><div className="w-9 h-5 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-primary"></div></label></div>
    <input className={`w-full bg-slate-950/50 border border-slate-800 rounded p-2 text-xs text-slate-400 font-mono ${isRandomSeed ? 'opacity-50' : ''}`} disabled={isRandomSeed} type="number" name="seed" value={params.seed} onChange={handleChange}/>
    </div>
    </div>
    </div>

    <div className="p-6 border-t border-slate-800 bg-bg-dark/80 backdrop-blur-md">
    <button onClick={runSimulation} disabled={loading} className={`w-full group relative overflow-hidden font-bold py-4 px-4 rounded-xl transition-all transform hover:-translate-y-1 active:translate-y-0 ${mode === 'dynamics' ? 'bg-primary hover:bg-primary-glow text-bg-dark shadow-neon-cyan' : 'bg-secondary hover:bg-secondary-glow text-white shadow-neon-purple'} ${loading ? 'opacity-80' : ''}`}>
    <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full group-hover:animate-[shimmer_1.5s_infinite]"></div>
    <span className="relative z-10 flex items-center justify-center gap-3"><span className={`material-symbols-outlined text-2xl ${loading ? 'animate-spin' : ''}`}>{loading ? 'sync' : 'rocket_launch'}</span><span className="tracking-wider">{loading ? 'ВЫЧИСЛЕНИЕ...' : 'ЗАПУСТИТЬ'}</span></span>
    </button>
    </div>
    </aside>

    <main className="flex-1 flex flex-col h-full bg-bg-dark relative overflow-hidden">
    <div className="absolute inset-0 bg-grid-pattern opacity-20 pointer-events-none"></div>
    <div className={`absolute top-0 left-0 w-[500px] h-[500px] rounded-full blur-[100px] pointer-events-none transition-colors duration-1000 ${mode === 'dynamics' ? 'bg-primary/5' : 'bg-secondary/5'}`}></div>

    <header className="h-20 shrink-0 flex items-center justify-between px-8 border-b border-slate-800 bg-bg-dark/60 backdrop-blur-md relative z-20">
    <div className="flex flex-col"><h2 className="text-2xl font-bold text-white tracking-tight">{mode === 'dynamics' ? 'Динамика ансамбля' : mode === 'compare_n' ? 'Сравнение n = 1, 3, 5' : 'Области синхронизации'}</h2><span className="text-xs text-slate-500 font-mono mt-1">ID: #992-{mode.toUpperCase()}-NEURO · RK4</span></div>
    <div className="flex items-center gap-6">
    <div className={`flex items-center gap-2 px-4 py-2 rounded-full border backdrop-blur-sm transition-all duration-500 ${isOnline ? 'border-green-500/20 bg-green-900/10' : 'border-red-500/20 bg-red-900/10'}`}>
    <span className="relative flex h-2 w-2"><span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${isOnline ? 'bg-green-400' : 'bg-red-400'}`}></span><span className={`relative inline-flex rounded-full h-2 w-2 ${isOnline ? 'bg-green-500' : 'bg-red-500'}`}></span></span>
    <span className={`text-xs font-bold tracking-wide font-mono ${isOnline ? 'text-green-400' : 'text-red-400'}`}>{isOnline ? 'SYSTEM ONLINE' : 'DISCONNECTED'}</span>
    </div>
    </div>
    </header>

    <div className="flex-1 p-8 flex flex-col gap-6 min-h-0 relative z-10">
    <div className="flex-1 relative rounded-2xl border border-slate-700 bg-[#080c14] shadow-2xl overflow-hidden group">
    <div className={`absolute inset-0 rounded-2xl border transition-colors duration-500 z-20 pointer-events-none ${mode === 'dynamics' ? 'border-primary/20 group-hover:border-primary/50' : 'border-secondary/20 group-hover:border-secondary/50'}`}></div>
    <div className="absolute inset-0 flex flex-col items-center justify-center z-30">
    {!data ? (
      <div className="p-6 rounded-2xl bg-bg-dark/80 backdrop-blur-md border border-slate-700 flex flex-col items-center shadow-lg">
      <span className={`material-symbols-outlined text-4xl animate-pulse-slow mb-3 ${mode === 'dynamics' ? 'text-primary' : 'text-secondary'}`}>{mode === 'dynamics' ? 'ssid_chart' : 'grid_view'}</span>
      <span className="text-xl font-light text-white tracking-[0.2em] uppercase">ОБЛАСТЬ ГРАФИКА</span>
      <span className="text-[10px] text-slate-400 mt-2 font-mono">Нажмите "Запустить" для генерации</span>
      </div>
    ) : (
      <Plot
        data={buildPlotData()}
        layout={{
          paper_bgcolor: 'rgba(0,0,0,0)',
         plot_bgcolor: 'rgba(0,0,0,0)',
         font: { color: '#94a3b8', family: 'Inter' },
         xaxis: { title: getXTitle(), showgrid: false },
         yaxis: { title: getYTitle(), showgrid: mode === 'compare_n', gridcolor: '#1e293b' },
         margin: { t: 40, r: 20, l: 60, b: 60 },
         autosize: true,
         hovermode: 'closest',
         showlegend: mode === 'compare_n',
         legend: { font: { color: '#cbd5e1' }, bgcolor: 'rgba(15,23,42,0.6)' },
        }}
        useResizeHandler={true}
        style={{ width: "100%", height: "100%" }}
        config={{ displayModeBar: true, displaylogo: false }}
        />
    )}
    </div>
    </div>

    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 h-auto shrink-0">
    <div className="bg-panel-dark/40 border border-slate-700/60 rounded-xl p-4 relative overflow-hidden group hover:border-primary/50"><div className="relative z-10"><p className="text-[10px] text-slate-400 font-bold uppercase tracking-wider mb-1">γ₁ (base_gamma)</p><h3 className="text-xl font-bold text-primary font-mono">{params.base_gamma.toFixed(5)}</h3></div></div>
    <div className="bg-panel-dark/40 border border-slate-700/60 rounded-xl p-4 relative overflow-hidden group hover:border-secondary/50"><div className="relative z-10"><p className="text-[10px] text-slate-400 font-bold uppercase tracking-wider mb-1">Δ (расстройка)</p><h3 className="text-xl font-bold text-secondary font-mono">{mode === 'map' && params.scan_type === 'd_delta' ? `${params.delta_min.toFixed(4)}…${params.delta_max.toFixed(4)}` : params.delta_gamma.toFixed(5)}</h3></div></div>
    <div className="bg-panel-dark/40 border border-slate-700/60 rounded-xl p-4 relative overflow-hidden group hover:border-amber-400/50"><div className="relative z-10"><p className="text-[10px] text-slate-400 font-bold uppercase tracking-wider mb-1">⟨Ω⟩ (наблюд. частота)</p><h3 className="text-xl font-bold text-amber-400 font-mono">{data && data.omega_mean !== undefined ? (typeof data.omega_mean === 'number' ? data.omega_mean.toFixed(4) : '—') : (data && data.traces ? data.traces.map(t => t.omega.toFixed(2)).join(' / ') : '—')}</h3></div></div>
    <div className="bg-panel-dark/40 border border-slate-700/60 rounded-xl p-4 relative overflow-hidden group hover:border-white/30"><div className="relative z-10"><p className="text-[10px] text-slate-400 font-bold uppercase tracking-wider mb-1">Статус · интегратор</p><h3 className="text-xl font-bold text-white font-mono">{loading ? 'BUSY' : 'IDLE'} · RK4</h3></div></div>
    </div>
    </div>
    </main>
    </>
  );
}

export default App;
