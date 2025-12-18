import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import axios from 'axios';

function App() {
  const [mode, setMode] = useState('dynamics');

  const [params, setParams] = useState({
    n_neurons: 50,
    t_end: 400,
    dt: 0.1,
    n_param: 3.0,
    base_gamma: 1.01,
    seed: 42,
    d_couple: 0.12,
    alpha: 2.09,

    scan_type: 'd_alpha',
    d_min: 0.0, d_max: 0.2,
    alpha_min: 0.0, alpha_max: 3.14,
    delta_min: 0.0, delta_max: 0.012,
    fixed_alpha: 2.0944,

    d_steps: 40, alpha_steps: 40, delta_steps: 40
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
      } catch (e) {
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
    setLoading(true);
    const currentSeed = isRandomSeed ? Math.floor(Math.random() * 1000000) : params.seed;

    const payload = { ...params, seed: currentSeed };
    if (params.scan_type === 'd_delta') {
      payload.alpha_steps = params.delta_steps;
    }

    try {
      const endpoint = mode === 'dynamics' ? '/api/run' : '/api/scan';
      const response = await axios.post(`http://localhost:8080${endpoint}`, payload);
      if (response.data) setData(response.data);
    } catch (err) {
      console.error(err);
      alert("Core Error. Check backend.");
    } finally {
      setLoading(false);
    }
  };

  const getXTitle = () => {
    if (mode === 'dynamics') return 'Время (шаги)';
    if (params.scan_type === 'd_alpha') return 'Фаза (α)';
    return 'Связь (d)';
  };

  const getYTitle = () => {
    if (mode === 'dynamics') return 'Индекс Нейрона';
    if (params.scan_type === 'd_alpha') return 'Связь (d)';
    return 'Расстройка (Δ)';
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
    <p className="text-[10px] text-primary/60 uppercase tracking-[0.2em]">Lab Edition v0.9</p>
    </div>
    </div>

    <div className="flex-1 overflow-y-auto p-6 space-y-8">
    <div className="p-1 rounded-xl bg-slate-900/80 border border-slate-800 flex relative overflow-hidden">
    <button onClick={() => { setMode('dynamics'); setData(null); }} className={`flex-1 py-3 px-2 rounded-lg text-[10px] font-bold uppercase tracking-wider transition-all relative z-10 ${mode === 'dynamics' ? 'text-white bg-primary/20 border border-primary/40 shadow-neon-cyan' : 'text-slate-500 hover:text-white hover:bg-white/5'}`}>Динамика</button>
    <button onClick={() => { setMode('map'); setData(null); }} className={`flex-1 py-3 px-2 rounded-lg text-[10px] font-bold uppercase tracking-wider transition-all relative z-10 ${mode === 'map' ? 'text-white bg-secondary/20 border border-secondary/40 shadow-neon-purple' : 'text-slate-500 hover:text-white hover:bg-white/5'}`}>Карта режимов</button>
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
      <label className="text-[10px] text-slate-400 font-bold uppercase">Gamma (γ)</label>
      <input className="w-full bg-slate-900/50 border border-slate-700 rounded p-2 text-xs text-white font-mono focus:border-primary outline-none" type="number" step="0.01" name="base_gamma" value={params.base_gamma} onChange={handleChange}/>
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
      <div className="mt-2 space-y-1">
      <label className="text-[10px] text-slate-500 uppercase font-bold">Форма (n)</label>
      <input className="w-full bg-slate-900/50 border border-slate-700 rounded p-2 text-xs text-white font-mono focus:border-secondary outline-none" type="number" name="n_param" value={params.n_param} onChange={handleChange}/>
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
    <div className="flex flex-col"><h2 className="text-2xl font-bold text-white tracking-tight">{mode === 'dynamics' ? 'Динамика ансамбля' : 'Карта синхронизации'}</h2><span className="text-xs text-slate-500 font-mono mt-1">ID: #992-{mode.toUpperCase()}-NEURO</span></div>
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
      data={mode === 'dynamics'
        ? [{ z: data.heatmap, type: 'heatmap', colorscale: 'Viridis', showscale: true }]
        : [{ z: data.map_z, x: data.map_x, y: data.map_y, type: 'heatmap', colorscale: 'Portland', showscale: true, colorbar: { title: 'Sync (R)' } }]}
        layout={{
          paper_bgcolor: 'rgba(0,0,0,0)',
         plot_bgcolor: 'rgba(0,0,0,0)',
         font: { color: '#94a3b8', family: 'Inter' },
         xaxis: { title: getXTitle(), showgrid: false },
         yaxis: { title: getYTitle(), showgrid: false },
         margin: { t: 40, r: 20, l: 60, b: 60 },
         autosize: true,
         hovermode: 'closest'
        }}
        useResizeHandler={true}
        style={{ width: "100%", height: "100%" }}
        config={{ displayModeBar: true, displaylogo: false }}
        />
    )}
    </div>
    </div>

    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 h-auto shrink-0">
    <div className="bg-panel-dark/40 border border-slate-700/60 rounded-xl p-5 relative overflow-hidden group hover:border-primary/50"><div className="absolute -right-6 -top-6 w-24 h-24 bg-primary/10 rounded-full blur-2xl group-hover:bg-primary/20"></div><div className="relative z-10 flex justify-between items-start"><div><p className="text-[11px] text-slate-400 font-bold uppercase tracking-wider mb-2">Статус ядра</p><h3 className="text-2xl font-bold text-white font-mono">{loading ? 'BUSY' : 'IDLE'}</h3></div><div className="p-3 bg-slate-800/50 rounded-lg text-primary"><span className="material-symbols-outlined">cpu</span></div></div></div>
    <div className="bg-panel-dark/40 border border-slate-700/60 rounded-xl p-5 relative overflow-hidden group hover:border-secondary/50"><div className="absolute -right-6 -top-6 w-24 h-24 bg-secondary/10 rounded-full blur-2xl group-hover:bg-secondary/20"></div><div className="relative z-10 flex justify-between items-start"><div><p className="text-[11px] text-slate-400 font-bold uppercase tracking-wider mb-2">Режим</p><h3 className="text-2xl font-bold text-white font-mono uppercase">{mode} ({params.scan_type})</h3></div><div className="p-3 bg-slate-800/50 rounded-lg text-secondary"><span className="material-symbols-outlined">tune</span></div></div></div>
    <div className="bg-panel-dark/40 border border-slate-700/60 rounded-xl p-5 relative overflow-hidden group hover:border-white/30"><div className="relative z-10 flex justify-between items-start"><div><p className="text-[11px] text-slate-400 font-bold uppercase tracking-wider mb-2">Объекты (N)</p><h3 className="text-2xl font-bold text-white font-mono">{mode === 'map' ? (params.scan_type === 'd_delta' ? 2 : '~200') : params.n_neurons}</h3></div><div className="p-3 bg-slate-800/50 rounded-lg text-slate-300"><span className="material-symbols-outlined">memory</span></div></div></div>
    </div>
    </div>
    </main>
    </>
  );
}

export default App;
