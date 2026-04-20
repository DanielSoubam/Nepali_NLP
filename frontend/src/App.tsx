import { useState, useEffect, useCallback } from 'react'
import './index.css'

const BACKEND = 'http://127.0.0.1:8000'

// ─── Types ──────────────────────────────────────────────────
interface TranslationResponse { translation: string }
interface EvalResult { source: string; reference: string; hypothesis: string; bleu: number }
interface EvalResponse { average_bleu: number; detailed_results: EvalResult[] }
interface LoRAConfig { rank: number; alpha: number; dropout: number; target_modules: string[] }
interface TrainingConfig { epochs: number; learning_rate: number; batch_size: number; gradient_accumulation_steps: number; warmup_steps: number }
interface SimulatedCurve { [epoch: string]: { train_loss: number; eval_bleu: number } }
interface ModelComp { model: string; bleu: number; notes: string }
interface FineTuneInfo {
  method: string; base_model: string; trainable_params_percent: string
  total_trainable_params: string; dataset: string; note: string
  lora_config: LoRAConfig; training_config: TrainingConfig
  simulated_training_curve: SimulatedCurve; model_comparison: ModelComp[]
}

// ─── SVG Bar Chart ───────────────────────────────────────────
function BarChart({ data, color = '#0F1923' }: { data: { label: string; value: number }[]; color?: string }) {
  const W = 500, H = 170
  const pad = { top: 24, right: 16, bottom: 44, left: 40 }
  const cW = W - pad.left - pad.right
  const cH = H - pad.top - pad.bottom
  const maxV = Math.max(...data.map(d => d.value), 1)
  const bW = Math.min(38, cW / data.length - 8)
  const yTicks = [0, 25, 50, 75, 100].filter(v => v <= maxV + 15)

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }}>
      {yTicks.map(v => {
        const y = pad.top + cH - (v / (maxV + 5)) * cH
        return (
          <g key={v}>
            <line x1={pad.left} x2={W - pad.right} y1={y} y2={y} stroke="#EAE8E2" strokeWidth="1" />
            <text x={pad.left - 6} y={y + 4} textAnchor="end" fill="#A8A59E" fontSize="9" fontFamily="IBM Plex Mono, monospace">{v}</text>
          </g>
        )
      })}
      {data.map((d, i) => {
        const slotW = cW / data.length
        const x = pad.left + slotW * i + (slotW - bW) / 2
        const bH = Math.max((d.value / (maxV + 5)) * cH, 1)
        const y = pad.top + cH - bH
        return (
          <g key={i}>
            <rect x={x} y={y} width={bW} height={bH} fill={color} rx="3" opacity="0.88" />
            <text x={x + bW / 2} y={y - 5} textAnchor="middle" fill="#141412" fontSize="9" fontWeight="600" fontFamily="IBM Plex Mono, monospace">
              {d.value.toFixed(1)}
            </text>
            <text x={x + bW / 2} y={H - pad.bottom + 14} textAnchor="middle" fill="#6B6860" fontSize="9" fontFamily="IBM Plex Sans, sans-serif">
              {d.label.length > 11 ? d.label.slice(0, 10) + '…' : d.label}
            </text>
          </g>
        )
      })}
      <line x1={pad.left} y1={pad.top} x2={pad.left} y2={H - pad.bottom} stroke="#DDD9D0" strokeWidth="1.5" />
      <line x1={pad.left} y1={H - pad.bottom} x2={W - pad.right} y2={H - pad.bottom} stroke="#DDD9D0" strokeWidth="1.5" />
    </svg>
  )
}

// ─── SVG Line Chart ──────────────────────────────────────────
function LineChart({ data }: { data: { label: string; loss: number; bleu: number }[] }) {
  if (data.length < 2) return null
  const W = 500, H = 170
  const pad = { top: 24, right: 64, bottom: 40, left: 40 }
  const cW = W - pad.left - pad.right
  const cH = H - pad.top - pad.bottom
  const maxL = Math.max(...data.map(d => d.loss))
  const maxB = Math.max(...data.map(d => d.bleu))

  const pt = (i: number, v: number, max: number) => {
    const x = data.length > 1 ? pad.left + (i / (data.length - 1)) * cW : pad.left + cW / 2
    const y = pad.top + cH - (v / (max * 1.1)) * cH
    return { x, y }
  }

  const lossPath = data.map((d, i) => { const p = pt(i, d.loss, maxL); return `${i === 0 ? 'M' : 'L'}${p.x},${p.y}` }).join(' ')
  const bleuPath = data.map((d, i) => { const p = pt(i, d.bleu, maxB); return `${i === 0 ? 'M' : 'L'}${p.x},${p.y}` }).join(' ')

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: 'block' }}>
      {[0, 1, 2, 3].map(i => {
        const y = pad.top + (i / 3) * cH
        return <line key={i} x1={pad.left} x2={W - pad.right} y1={y} y2={y} stroke="#EAE8E2" strokeWidth="1" />
      })}
      <path d={lossPath} fill="none" stroke="#B91C1C" strokeWidth="2" strokeLinejoin="round" />
      <path d={bleuPath} fill="none" stroke="#0D9488" strokeWidth="2" strokeLinejoin="round" />
      {data.map((d, i) => {
        const lp = pt(i, d.loss, maxL)
        const bp = pt(i, d.bleu, maxB)
        return (
          <g key={i}>
            <circle cx={lp.x} cy={lp.y} r="4" fill="#B91C1C" />
            <circle cx={bp.x} cy={bp.y} r="4" fill="#0D9488" />
            <text x={lp.x} y={H - pad.bottom + 14} textAnchor="middle" fill="#6B6860" fontSize="10" fontFamily="IBM Plex Sans, sans-serif">{d.label}</text>
          </g>
        )
      })}
      <line x1={pad.left} y1={pad.top} x2={pad.left} y2={H - pad.bottom} stroke="#DDD9D0" strokeWidth="1.5" />
      <line x1={pad.left} y1={H - pad.bottom} x2={W - pad.right} y2={H - pad.bottom} stroke="#DDD9D0" strokeWidth="1.5" />
      {/* Legend */}
      <circle cx={W - pad.right + 10} cy={pad.top + 8} r="4" fill="#B91C1C" />
      <text x={W - pad.right + 18} y={pad.top + 12} fill="#6B6860" fontSize="9" fontFamily="IBM Plex Sans, sans-serif">Loss</text>
      <circle cx={W - pad.right + 10} cy={pad.top + 26} r="4" fill="#0D9488" />
      <text x={W - pad.right + 18} y={pad.top + 30} fill="#6B6860" fontSize="9" fontFamily="IBM Plex Sans, sans-serif">BLEU</text>
    </svg>
  )
}

// ─── Translate Tab ───────────────────────────────────────────
function TranslateTab() {
  const [sourceText, setSourceText] = useState('')
  const [translated, setTranslated] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [srcLang, setSrcLang] = useState('eng_Latn')
  const [tgtLang, setTgtLang] = useState('npi_Deva')

  useEffect(() => {
    if (!sourceText.trim()) { setTranslated(''); return }
    const t = setTimeout(async () => {
      setLoading(true); setError('')
      try {
        const res = await fetch(`${BACKEND}/translate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: sourceText, source_lang: srcLang, target_lang: tgtLang }),
        })
        if (!res.ok) throw new Error()
        const data: TranslationResponse = await res.json()
        setTranslated(data.translation)
      } catch {
        setError('Backend unavailable — start the server on port 8000.')
      } finally { setLoading(false) }
    }, 500)
    return () => clearTimeout(t)
  }, [sourceText, srcLang, tgtLang])

  const swap = () => {
    setSourceText(translated); setTranslated(sourceText)
    setSrcLang(tgtLang); setTgtLang(srcLang)
  }
  const label = (c: string) => c === 'eng_Latn' ? 'English' : 'Nepali'

  return (
    <div className="translate-tab">
      <div className="lang-bar">
        <div className={`lang-pill${srcLang === 'eng_Latn' ? ' active' : ''}`}>{label(srcLang)}</div>
        <button className="swap-btn" onClick={swap} title="Swap languages">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/>
            <polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/>
          </svg>
        </button>
        <div className={`lang-pill${tgtLang === 'npi_Deva' ? ' active' : ''}`}>{label(tgtLang)}</div>
      </div>

      {error && <div className="error-bar">{error}</div>}

      <div className="panels">
        <div className="panel">
          <div className="panel-header">
            <span className="panel-lang">{label(srcLang)}</span>
            <span className="char-count">{sourceText.length} chars</span>
          </div>
          <textarea
            className="text-input"
            value={sourceText}
            onChange={e => setSourceText(e.target.value)}
            placeholder="Enter text to translate…"
            rows={8}
          />
        </div>

        <div className="panel panel-output">
          <div className="panel-header">
            <span className="panel-lang">{label(tgtLang)}</span>
            {loading && <span className="translating-label">translating…</span>}
          </div>
          <div className="text-output">
            {loading
              ? <div className="spinner-wrap"><div className="spinner"/></div>
              : translated
              ? <p>{translated}</p>
              : <p className="placeholder">Translation will appear here…</p>
            }
          </div>
          {translated && (
            <button className="copy-btn" onClick={() => navigator.clipboard?.writeText(translated)}>copy</button>
          )}
        </div>
      </div>

      <div className="model-badge">
        <span className="label">model</span>
        <code>facebook/nllb-200-distilled-600M</code>
        <span>·</span><span>cpu inference</span>
        <span>·</span><span>200+ languages</span>
      </div>
    </div>
  )
}

// ─── Analytics Tab ───────────────────────────────────────────
function AnalyticsTab() {
  const [data, setData] = useState<EvalResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const runEval = useCallback(async () => {
    setLoading(true); setError('')
    try {
      const res = await fetch(`${BACKEND}/evaluate`)
      if (!res.ok) throw new Error()
      setData(await res.json())
    } catch {
      setError('Could not reach backend. Start the server on port 8000.')
    } finally { setLoading(false) }
  }, [])

  const bleuBar = data?.detailed_results.map((r, i) => ({ label: `S${i + 1}`, value: r.bleu })) ?? []
  const cmpBar = [
    { label: 'Word-by-word', value: 4.2 },
    { label: 'Moses SMT', value: 12.8 },
    { label: 'mBART-50', value: 19.4 },
    { label: 'NLLB-200', value: data?.average_bleu ?? 25.5 },
  ]
  const bleuClass = (b: number) => b >= 50 ? 'bleu-high' : b >= 20 ? 'bleu-mid' : 'bleu-low'

  return (
    <div className="analytics-tab">
      <div className="analytics-header">
        <div className="section-heading">
          <h2>Model Evaluation</h2>
          <p>BLEU score analysis on the Nepali–English test set using the live model</p>
        </div>
        <button className="eval-btn" onClick={runEval} disabled={loading}>
          {loading ? 'Evaluating…' : 'Run Evaluation'}
        </button>
      </div>

      {error && <div className="error-bar">{error}</div>}

      {data && (
        <>
          <div className="stat-row">
            <div className="stat-card">
              <div className="stat-value">{data.average_bleu.toFixed(1)}</div>
              <div className="stat-label">Avg BLEU</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{data.detailed_results.length}</div>
              <div className="stat-label">Test sentences</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{Math.max(...data.detailed_results.map(r => r.bleu)).toFixed(1)}</div>
              <div className="stat-label">Best BLEU</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{Math.min(...data.detailed_results.map(r => r.bleu)).toFixed(1)}</div>
              <div className="stat-label">Lowest BLEU</div>
            </div>
          </div>

          <div className="charts-grid">
            <div className="chart-card">
              <div className="chart-card-title">BLEU Score per Sentence</div>
              <BarChart data={bleuBar} color="#0F1923" />
            </div>
            <div className="chart-card">
              <div className="chart-card-title">Model Comparison (BLEU ↑ higher is better)</div>
              <BarChart data={cmpBar} color="#0D9488" />
            </div>
          </div>

          <div className="results-card">
            <div className="results-card-header">Detailed Translation Results</div>
            <div className="results-table-wrap">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>#</th><th>Source</th><th>Reference</th><th>Model Output</th><th>BLEU</th>
                  </tr>
                </thead>
                <tbody>
                  {data.detailed_results.map((r, i) => (
                    <tr key={i}>
                      <td className="col-idx">{i + 1}</td>
                      <td>{r.source}</td>
                      <td>{r.reference}</td>
                      <td>{r.hypothesis}</td>
                      <td><span className={`bleu-badge ${bleuClass(r.bleu)}`}>{r.bleu.toFixed(1)}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {!data && !loading && (
        <div className="empty-state">
          <p>Click <strong>Run Evaluation</strong> to compute BLEU scores on the test set.</p>
          <p className="sub">Runs live translations and compares against reference translations.</p>
        </div>
      )}
    </div>
  )
}

// ─── Fine-Tuning Tab ─────────────────────────────────────────
function FineTuningTab() {
  const [info, setInfo] = useState<FineTuneInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    fetch(`${BACKEND}/finetune/info`)
      .then(r => r.json())
      .then(d => { setInfo(d); setLoading(false) })
      .catch(() => { setError('Backend unavailable.'); setLoading(false) })
  }, [])

  const curveData = info
    ? Object.entries(info.simulated_training_curve).map(([k, v]) => ({
        label: k.replace('epoch_', 'Ep '),
        loss: v.train_loss,
        bleu: v.eval_bleu,
      }))
    : []

  const cmpBar = info?.model_comparison.map(m => ({ label: m.model, value: m.bleu })) ?? []

  return (
    <div className="finetune-tab">
      <div className="ft-header">
        <h2>Fine-Tuning Configuration</h2>
        <p>LoRA parameter-efficient fine-tuning setup for low-resource Nepali NLP</p>
      </div>

      {error && <div className="error-bar">{error}</div>}
      {loading && <div className="loading-state">Loading configuration…</div>}

      {info && (
        <>
          <div className="config-grid">
            <div className="config-card">
              <div className="config-card-header">Approach</div>
              <div className="config-rows">
                <div className="config-row"><span className="key">Method</span><span className="val">{info.method}</span></div>
                <div className="config-row"><span className="key">Base model</span><span className="val">{info.base_model}</span></div>
                <div className="config-row"><span className="key">Dataset</span><span className="val">{info.dataset}</span></div>
                <div className="config-row"><span className="key">Trainable params</span><span className="val">{info.total_trainable_params}</span></div>
                <div className="config-row"><span className="key">% of total</span><span className="val">{info.trainable_params_percent}</span></div>
              </div>
            </div>

            <div className="config-card">
              <div className="config-card-header">LoRA Hyperparameters</div>
              <div className="config-rows">
                <div className="config-row"><span className="key">Rank (r)</span><span className="val">{info.lora_config.rank}</span></div>
                <div className="config-row"><span className="key">Alpha (α)</span><span className="val">{info.lora_config.alpha}</span></div>
                <div className="config-row"><span className="key">Dropout</span><span className="val">{info.lora_config.dropout}</span></div>
                <div className="config-row"><span className="key">Task type</span><span className="val">{info.lora_config.task_type}</span></div>
                <div className="config-row"><span className="key">Target modules</span><span className="val">{info.lora_config.target_modules.join(', ')}</span></div>
              </div>
            </div>

            <div className="config-card">
              <div className="config-card-header">Training Parameters</div>
              <div className="config-rows">
                <div className="config-row"><span className="key">Epochs</span><span className="val">{info.training_config.epochs}</span></div>
                <div className="config-row"><span className="key">Learning rate</span><span className="val">{info.training_config.learning_rate}</span></div>
                <div className="config-row"><span className="key">Batch size</span><span className="val">{info.training_config.batch_size}</span></div>
                <div className="config-row"><span className="key">Grad accumulation</span><span className="val">{info.training_config.gradient_accumulation_steps}</span></div>
                <div className="config-row"><span className="key">Warmup steps</span><span className="val">{info.training_config.warmup_steps}</span></div>
              </div>
            </div>
          </div>

          <div className="ft-charts">
            <div className="chart-card">
              <div className="chart-card-title">Simulated Training Curve</div>
              <LineChart data={curveData} />
            </div>
            <div className="chart-card">
              <div className="chart-card-title">System Comparison — BLEU (↑ higher is better)</div>
              <BarChart data={cmpBar} color="#0F1923" />
            </div>
          </div>

          <div className="note-box">
            <span className="note-icon">i</span>
            <p>{info.note}</p>
          </div>
        </>
      )}
    </div>
  )
}

// ─── Root App ────────────────────────────────────────────────
type Tab = 'translate' | 'analytics' | 'finetune'

export default function App() {
  const [tab, setTab] = useState<Tab>('translate')

  return (
    <div className="app">
      <header className="top-bar">
        <div className="top-bar-inner">
          <div className="brand">
            <span className="brand-mark">NLP</span>
            <span className="brand-divider">·</span>
            <span className="brand-title">Cross-Lingual Language Modeling</span>
            <span className="brand-divider">—</span>
            <span className="brand-sub">Low-Resource Nepali Translation</span>
          </div>
          <nav className="nav-tabs">
            {(['translate', 'analytics', 'finetune'] as Tab[]).map(t => (
              <button key={t} className={`nav-tab${tab === t ? ' active' : ''}`} onClick={() => setTab(t)}>
                {t === 'translate' ? 'Translate' : t === 'analytics' ? 'Analytics' : 'Fine-tuning'}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="main">
        <div className="content">
          {tab === 'translate'  && <TranslateTab />}
          {tab === 'analytics'  && <AnalyticsTab />}
          {tab === 'finetune'   && <FineTuningTab />}
        </div>
      </main>

      <footer className="footer">
        <span>cross-lingual language modeling for low-resource NLP</span>
        <span>·</span>
        <span>NLLB-200 · FastAPI · React</span>
      </footer>
    </div>
  )
}
