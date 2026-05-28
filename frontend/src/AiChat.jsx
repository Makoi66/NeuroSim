import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const MD_COMPONENTS = {
  p: (props) => <p className="my-1.5 leading-relaxed" {...props} />,
  ul: (props) => <ul className="list-disc pl-5 my-1.5 space-y-0.5" {...props} />,
  ol: (props) => <ol className="list-decimal pl-5 my-1.5 space-y-0.5" {...props} />,
  li: (props) => <li className="leading-relaxed" {...props} />,
  h1: (props) => <h3 className="text-sm font-bold text-white mt-2 mb-1" {...props} />,
  h2: (props) => <h3 className="text-sm font-bold text-white mt-2 mb-1" {...props} />,
  h3: (props) => <h4 className="text-xs font-bold text-white mt-2 mb-1 uppercase tracking-wider" {...props} />,
  strong: (props) => <strong className="text-white font-semibold" {...props} />,
  em: (props) => <em className="text-slate-300 italic" {...props} />,
  a: (props) => (
    <a
      className="text-primary-glow underline hover:text-primary"
      target="_blank"
      rel="noopener noreferrer"
      {...props}
    />
  ),
  blockquote: (props) => (
    <blockquote className="border-l-2 border-primary/40 pl-3 my-1.5 text-slate-300 italic" {...props} />
  ),
  hr: () => <hr className="my-2 border-slate-700" />,
};

const QUICK_ACTIONS = {
  dynamics: [
    'Что показывает этот график?',
    'Достигнута ли синхронизация и какого типа?',
    'Сравни наблюдаемую Ω с теоретической √(γ²−1)',
  ],
  compare_n: [
    'В чём разница между n = 1, 3 и 5?',
    'Что такое пачечная (bursting) активность?',
  ],
  d_alpha: [
    'Где оптимум синхронизации на этой карте?',
    'Опиши структуру языков синхронизации',
    'Как фазовый сдвиг α влияет на режим?',
  ],
  d_delta: [
    'Где порог захвата по силе связи d?',
    'Сравни ширину области захвата с оценкой Δ ≤ d/2',
    'Опиши форму области синхронизации',
  ],
};

function buildResultSummary(mode, scanType, data) {
  if (!data) return {};

  if (mode === 'dynamics') {
    return {
      kind: 'dynamics',
      R_steady: data.R_steady,
      R_final: data.R_final,
      phase_diff_std: data.phase_diff_std,
      omega_mean: data.omega_mean,
      omega_std: data.omega_std,
      t_end: data.t_end,
      n_neurons: data.heatmap?.length ?? null,
    };
  }

  if (mode === 'compare_n') {
    return {
      kind: 'compare_n',
      traces: data.traces?.map((tr) => ({ n: tr.n, omega: tr.omega })),
    };
  }

  if (mode === 'map' && Array.isArray(data.map_z)) {
    const flat = data.map_z.flat();
    if (flat.length === 0) return { kind: scanType };

    let rMin = Infinity, rMax = -Infinity, rSum = 0;
    let imax = 0, jmax = 0;
    for (let i = 0; i < data.map_z.length; i++) {
      const row = data.map_z[i];
      for (let j = 0; j < row.length; j++) {
        const v = row[j];
        if (v < rMin) rMin = v;
        if (v > rMax) { rMax = v; imax = i; jmax = j; }
        rSum += v;
      }
    }
    const rMean = rSum / flat.length;

    let nFull = 0, nHigh = 0, nPartial = 0, nAsync = 0;
    for (const v of flat) {
      if (v >= 0.95) nFull++;
      else if (v >= 0.7) nHigh++;
      else if (v >= 0.3) nPartial++;
      else nAsync++;
    }
    const total = flat.length;

    const summary = {
      kind: scanType,
      R_max: +rMax.toFixed(4),
      R_min: +rMin.toFixed(4),
      R_mean: +rMean.toFixed(4),
      R_max_at: { x: data.map_x?.[jmax], y: data.map_y?.[imax] },
      fractions: {
        sync_full: +(nFull / total).toFixed(3),
        sync_high: +(nHigh / total).toFixed(3),
        sync_partial: +(nPartial / total).toFixed(3),
        async_or_death: +(nAsync / total).toFixed(3),
      },
      kernel: data.kernel,
      n_used: data.n_used,
    };
    if (data.sync_thresholds) {
      const t = data.sync_thresholds;
      summary.sync_thresholds = {
        R_threshold: t.R_threshold,
        ...(scanType === 'd_alpha' && {
          d_min: t.d_min,
          alpha_opt: t.alpha_opt,
          alpha_crit: t.alpha_crit,
        }),
        ...(scanType === 'd_delta' && {
          d_opt: t.d_opt,
          delta_at_d_opt: t.delta_at_d_opt,
          delta_max: t.delta_max,
        }),
      };
    }
    return summary;
  }

  return {};
}

function getPlotKind(mode, scanType) {
  if (mode === 'dynamics') return 'dynamics';
  if (mode === 'compare_n') return 'compare_n';
  if (mode === 'map') return scanType === 'd_alpha' ? 'd_alpha' : 'd_delta';
  return '';
}

const MODEL_OPTIONS = [
  { id: 'gemini-3.1-flash-lite', label: '3.1 Flash Lite', hint: 'Дефолт, быстро (3-10 с), vision + tools' },
  { id: 'gemma-4-31b-it',        label: 'Gemma 4 31B',    hint: 'Точнее, но медленно (30-60 с), vision + tools' },
];

export default function AiChat({ mode, params, data, plotDivRef }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [includeImage, setIncludeImage] = useState(true);
  const [modelId, setModelId] = useState('gemini-3.1-flash-lite');
  const [modelMenuOpen, setModelMenuOpen] = useState(false);
  const scrollRef = useRef(null);
  const currentModel = MODEL_OPTIONS.find((m) => m.id === modelId) || MODEL_OPTIONS[0];

  const plotKind = getPlotKind(mode, params.scan_type);
  const quickActions = QUICK_ACTIONS[plotKind] || [];

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading]);

  const captureChartPng = async () => {
    const gd = plotDivRef?.current;
    if (!gd || !window.Plotly || !data) return null;
    try {
      const dataUrl = await window.Plotly.toImage(gd, {
        format: 'png',
        width: 800,
        height: 600,
      });
      const comma = dataUrl.indexOf(',');
      return comma >= 0 ? dataUrl.slice(comma + 1) : null;
    } catch (e) {
      console.warn('Plotly.toImage failed:', e);
      return null;
    }
  };

  const send = async (text) => {
    const q = text.trim();
    if (!q || loading) return;
    const userMsg = { role: 'user', content: q };
    const next = [...messages, userMsg];
    setMessages(next);
    setInput('');
    setLoading(true);

    const image_b64 = includeImage ? await captureChartPng() : null;

    const payload = {
      question: q,
      plot_kind: plotKind,
      params: { ...params },
      result_summary: buildResultSummary(mode, params.scan_type, data),
      image_b64,
      history: messages.slice(-8),
      model: modelId,
    };

    try {
      const resp = await axios.post('http://localhost:8080/api/ai/chat', payload, {
        timeout: 90000,
      });
      const answer = resp?.data?.answer || '(пустой ответ)';
      const tools = resp?.data?.tools_used || [];
      setMessages([...next, { role: 'model', content: answer, tools, withImage: !!image_b64 }]);
    } catch (err) {
      const detail =
        err?.response?.data?.detail || err?.response?.statusText || err?.message || 'неизвестная ошибка';
      setMessages([...next, { role: 'model', content: 'Ошибка AI: ' + detail, error: true }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send(input);
    }
  };

  return (
    <>
    {!open && (
      <button
        onClick={() => setOpen(true)}
        className="absolute bottom-12 right-12 z-50 h-14 w-14 rounded-full flex items-center justify-center
          border transition-all shadow-lg
          bg-primary/20 border-primary/60 text-primary-glow hover:bg-primary/30 shadow-neon-cyan"
        title="Открыть AI-помощника"
      >
        <span className="material-symbols-outlined text-2xl">smart_toy</span>
      </button>
    )}

    <aside
      className={`absolute top-0 right-0 h-full w-[420px] z-40 flex flex-col
        bg-panel-dark/95 border-l border-slate-800 backdrop-blur-xl
        transition-transform duration-300 ${open ? 'translate-x-0' : 'translate-x-full'}`}
    >
      <div className="shrink-0 px-6 pt-4 pb-3 border-b border-slate-800/60 bg-bg-dark/40 space-y-3">
        <div className="flex items-center gap-3">
          <div className="relative w-10 h-10 flex items-center justify-center">
            <div className="absolute inset-0 bg-primary/20 rounded-lg blur-md animate-pulse"></div>
            <div className="relative w-full h-full rounded-lg bg-gradient-to-br from-primary/20 to-blue-600/20 border border-primary/50 flex items-center justify-center">
              <span className="material-symbols-outlined text-primary-glow text-xl">smart_toy</span>
            </div>
          </div>
          <div className="flex-1 min-w-0">
            <h2 className="text-base font-bold text-white tracking-wide font-mono">AI-помощник</h2>
            <p className="text-[10px] text-primary/60 uppercase tracking-[0.2em] truncate">
              {data ? plotKind || 'no plot' : 'график не построен'}
            </p>
          </div>
          <button
            onClick={() => setOpen(false)}
            className="shrink-0 w-9 h-9 rounded-lg flex items-center justify-center
              border border-slate-700 text-slate-400 hover:text-white hover:border-slate-500
              hover:bg-slate-800/60 transition-colors"
            title="Закрыть"
          >
            <span className="material-symbols-outlined text-lg">close</span>
          </button>
        </div>

        <div className="relative">
          <button
            onClick={() => setModelMenuOpen((v) => !v)}
            disabled={loading}
            className="w-full flex items-center justify-between px-3 py-2 rounded-lg
              bg-slate-900/60 border border-slate-700 hover:border-primary/50
              text-left transition-colors disabled:opacity-50"
          >
            <div className="flex items-center gap-2 min-w-0">
              <span className="material-symbols-outlined text-primary/70 text-sm">tune</span>
              <div className="min-w-0">
                <div className="text-xs font-mono text-white truncate">{currentModel.label}</div>
                <div className="text-[10px] text-slate-500 truncate">{currentModel.hint}</div>
              </div>
            </div>
            <span className={`material-symbols-outlined text-slate-400 text-base transition-transform
              ${modelMenuOpen ? 'rotate-180' : ''}`}>expand_more</span>
          </button>

          {modelMenuOpen && (
            <div className="absolute top-full left-0 right-0 mt-1 z-50 rounded-lg
              bg-slate-900/95 border border-slate-700 backdrop-blur-xl shadow-xl overflow-hidden">
              {MODEL_OPTIONS.map((opt) => (
                <button
                  key={opt.id}
                  onClick={() => { setModelId(opt.id); setModelMenuOpen(false); }}
                  className={`w-full px-3 py-2 text-left transition-colors flex items-center gap-2
                    ${opt.id === modelId
                      ? 'bg-primary/10 text-white'
                      : 'text-slate-300 hover:bg-slate-800/60 hover:text-white'}`}
                >
                  <span className={`material-symbols-outlined text-sm
                    ${opt.id === modelId ? 'text-primary' : 'text-slate-600'}`}>
                    {opt.id === modelId ? 'radio_button_checked' : 'radio_button_unchecked'}
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="text-xs font-mono">{opt.label}</div>
                    <div className="text-[10px] text-slate-500 truncate">{opt.hint}</div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 && (
          <div className="space-y-3 text-xs">
            <div className="p-3 rounded-lg bg-slate-900/40 border border-slate-800 text-slate-300 leading-relaxed">
              Задай вопрос про текущий график: интерпретация, оптимумы,
              "а что если". У помощника есть инструменты для запуска коротких
              симуляций, он может проверить твою гипотезу численно.
            </div>
            {!data && (
              <div className="p-3 rounded-lg bg-amber-400/5 border border-amber-400/30 text-amber-300/90">
                График ещё не построен. Запусти любой режим, и у AI
                появится конкретный контекст для ответов.
              </div>
            )}
            {data && quickActions.length > 0 && (
              <div className="space-y-2">
                <p className="text-[10px] text-slate-500 uppercase tracking-wider font-bold">
                  Быстрые вопросы
                </p>
                {quickActions.map((q) => (
                  <button
                    key={q}
                    onClick={() => send(q)}
                    disabled={loading}
                    className="w-full text-left text-xs p-2 rounded-md bg-slate-900/60 border border-slate-800 text-slate-300 hover:border-primary/50 hover:text-white transition-colors disabled:opacity-50"
                  >
                    {q}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
              className={`max-w-[92%] px-3 py-2 rounded-lg text-xs leading-relaxed
                ${m.role === 'user'
                  ? 'bg-primary/15 border border-primary/30 text-white whitespace-pre-wrap'
                  : m.error
                    ? 'bg-red-900/20 border border-red-500/40 text-red-200 whitespace-pre-wrap'
                    : 'bg-slate-900/60 border border-slate-700 text-slate-200'}`}
            >
              {m.role === 'user' || m.error ? (
                m.content
              ) : (
                <div className="ai-md">
                  <ReactMarkdown
                    remarkPlugins={[remarkMath]}
                    rehypePlugins={[rehypeKatex]}
                    components={MD_COMPONENTS}
                  >
                    {m.content}
                  </ReactMarkdown>
                </div>
              )}
              {(m.role === 'model' && !m.error) && ((m.tools && m.tools.length > 0) || m.withImage) && (
                <div className="mt-2 pt-2 border-t border-slate-700/50 text-[10px] text-primary/70 font-mono flex flex-wrap gap-x-3 gap-y-1">
                  {m.tools && m.tools.length > 0 && (
                    <span>использовал: {m.tools.join(', ')}</span>
                  )}
                  {m.withImage && (
                    <span className="flex items-center gap-1 text-secondary/70">
                      <span className="material-symbols-outlined text-[12px]">image</span>
                      + снимок графика
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="px-3 py-2 rounded-lg bg-slate-900/60 border border-slate-700 text-slate-400 text-xs flex items-center gap-2">
              <span className="material-symbols-outlined text-sm animate-spin">sync</span>
              думаю...
            </div>
          </div>
        )}
      </div>

      <div className="p-3 border-t border-slate-800 bg-bg-dark/60 space-y-2">
        <div
          className={`flex items-center justify-between px-3 py-2 rounded-lg border
            ${data
              ? 'bg-slate-900/40 border-slate-800'
              : 'bg-slate-900/20 border-slate-800/60 opacity-60'}`}
          title="Прикреплять PNG-снимок графика к запросу, Gemini увидит его глазами"
        >
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-slate-400 text-sm">image</span>
            <span className="text-[10px] text-slate-300 font-bold uppercase tracking-wider">
              Снимок графика
            </span>
          </div>
          <label className={`relative inline-flex items-center ${data && !loading ? 'cursor-pointer' : 'cursor-not-allowed'}`}>
            <input
              type="checkbox"
              className="sr-only peer"
              checked={includeImage && !!data}
              disabled={!data || loading}
              onChange={(e) => setIncludeImage(e.target.checked)}
            />
            <div className="w-9 h-5 bg-slate-700 rounded-full peer
              peer-checked:bg-primary
              peer-focus:outline-none
              peer-disabled:bg-slate-800
              after:content-[''] after:absolute after:top-[2px] after:left-[2px]
              after:bg-white after:border after:rounded-full after:h-4 after:w-4
              after:transition-all
              peer-checked:after:translate-x-full peer-checked:after:border-white">
            </div>
          </label>
        </div>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={data ? 'Спросить про график...' : 'Сначала построй график'}
          rows={2}
          disabled={loading}
          className="w-full bg-slate-900/60 border border-slate-700 rounded-lg p-2 text-xs text-white font-mono resize-none focus:border-primary outline-none disabled:opacity-50"
        />
        <button
          onClick={() => send(input)}
          disabled={loading || !input.trim()}
          className="w-full py-2 px-3 rounded-lg bg-primary/20 border border-primary/40 text-primary-glow hover:bg-primary/30 transition-colors text-xs font-bold uppercase tracking-wider disabled:opacity-40 disabled:hover:bg-primary/20 flex items-center justify-center gap-2"
        >
          <span className="material-symbols-outlined text-base">send</span>
          Отправить
        </button>
      </div>
    </aside>
    </>
  );
}
